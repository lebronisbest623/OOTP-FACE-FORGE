"""Train a production U-Net for detail map prediction and export to ONNX.

Usage:
  python scripts/train_detail_cnn.py gen   --out models/detail_cnn_data --n 5000
  python scripts/train_detail_cnn.py train --data models/detail_cnn_data --model models/detail_cnn.pt --epochs 100
  python scripts/train_detail_cnn.py export --model models/detail_cnn.pt --onnx models/detail_cnn.onnx
  python scripts/train_detail_cnn.py eval  --model models/detail_cnn.pt --fg-dir ... --out-dir models/detail_cnn_eval
"""
from __future__ import annotations

import argparse
import io
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ootp_faceforge.basis import get_basis  # noqa: E402
from ootp_faceforge.fgformat import FgFile  # noqa: E402
from ootp_faceforge.render import _build_ootp_assets, _render_assets  # noqa: E402

DEFAULT_FG_DIR = Path(
    r"C:\Users\user\Documents\Out of the Park Developments"
    r"\OOTP Baseball 27\fg_files"
)
DETAIL_SIZE = 256
N_SYM, N_ASYM, N_TEX = 50, 30, 50


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _warp_texture_to_detail_uv(img: np.ndarray, basis, size: int):
    from ootp_faceforge.texture import detail_px
    dpx, dvalid = detail_px(basis, size)
    tris = basis.tris
    tri_uv = basis.vert_uv[tris]
    src_tri = tri_uv * np.array([img.shape[1], img.shape[0]], np.float32)
    dst_tri = dpx[tris]
    out = np.full((size, size, 3), 64.0, np.float32)
    valid = np.zeros((size, size), bool)
    for ti in range(len(tris)):
        src = src_tri[ti].astype(np.float32)
        dst = dst_tri[ti].astype(np.float32)
        x0 = max(int(np.floor(dst[:, 0].min())), 0)
        x1 = min(int(np.ceil(dst[:, 0].max())) + 1, size)
        y0 = max(int(np.floor(dst[:, 1].min())), 0)
        y1 = min(int(np.ceil(dst[:, 1].max())) + 1, size)
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        dst_local = dst - [x0, y0]
        M = cv2.getAffineTransform(src.astype(np.float32), dst_local.astype(np.float32))
        patch = cv2.warpAffine(img, M, (x1 - x0, y1 - y0),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillConvexPoly(mask, np.round(dst_local).astype(np.int32), 1)
        mb = mask.astype(bool)
        if mb.any():
            out[y0:y1, x0:x1][mb] = patch[mb]
            valid[y0:y1, x0:x1][mb] = True
    return np.clip(out, 0, 255).astype(np.uint8), valid


def augment_render(img: np.ndarray, rng: np.random.Generator,
                   severity: float = 1.0) -> np.ndarray:
    out = img.astype(np.float32)
    h, w = img.shape[:2]
    gains = rng.normal(1.0, 0.04 * severity, 3).astype(np.float32)
    out *= gains[None, None, :]
    contrast = 1.0 + rng.normal(0.0, 0.10 * severity)
    bright = rng.normal(0.0, 10.0 * severity)
    out = (out - 127.5) * contrast + 127.5 + bright
    if rng.random() < 0.6:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        xx = xx / max(w - 1, 1) - 0.5; yy = yy / max(h - 1, 1) - 0.5
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        grad = xx * np.cos(theta) + yy * np.sin(theta)
        amp = float(rng.normal(0.0, 0.18 * severity))
        out *= np.clip(1.0 + amp * grad, 0.65, 1.40)[..., None]
    if rng.random() < 0.3:
        out = cv2.GaussianBlur(out, (0, 0), float(rng.uniform(0.3, 0.9) * severity))
    if rng.random() < 0.3:
        out += rng.normal(0.0, 3.0 * severity, out.shape).astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if rng.random() < 0.6:
        buf = io.BytesIO()
        Image.fromarray(out).save(buf, "JPEG", quality=int(rng.integers(50, 92)))
        out = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
    return out


def _decode_detail(fg: FgFile, size: int) -> np.ndarray | None:
    if not fg.detail_jpeg:
        return None
    detail = np.asarray(
        Image.open(io.BytesIO(fg.detail_jpeg)).convert("RGB"), np.float32)
    if detail.shape[:2] != (size, size):
        detail = cv2.resize(detail, (size, size), interpolation=cv2.INTER_LINEAR)
    return detail


# ---------------------------------------------------------------------------
# Data generation (saves as sharded .npz for memory efficiency)
# ---------------------------------------------------------------------------

def stage_gen(args: argparse.Namespace) -> None:
    basis = get_basis()
    fg_dir = Path(args.fg_dir)
    paths = sorted(fg_dir.glob("*.fg"))
    if args.n:
        paths = paths[: args.n]
    if not paths:
        raise SystemExit(f"no .fg files: {fg_dir}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    shard_size = args.shard_size
    X_shard, Y_shard = [], []
    shard_idx = 0
    total = 0
    skipped = 0

    for i, path in enumerate(paths):
        try:
            fg = FgFile.read(str(path))
            if (len(fg.sym_shape) != N_SYM or len(fg.asym_shape) != N_ASYM
                    or len(fg.sym_tex) != N_TEX):
                skipped += 1; continue
            target = _decode_detail(fg, DETAIL_SIZE)
            if target is None:
                skipped += 1; continue
        except Exception:
            skipped += 1; continue

        try:
            assets = _build_ootp_assets(fg, include_eyes=True)
            rendered = _render_assets(assets, fg, DETAIL_SIZE, shade=True, aa=1)
        except Exception:
            skipped += 1; continue

        n_aug = max(int(args.augs), 1)
        for aug_id in range(n_aug):
            img = rendered if aug_id == 0 else augment_render(
                rendered, rng, severity=args.aug_strength)
            warped, valid = _warp_texture_to_detail_uv(img, basis, DETAIL_SIZE)
            if not valid.any():
                continue
            X_shard.append(warped.astype(np.float32) / 255.0)
            Y_shard.append(target.astype(np.float32) / 255.0)
            total += 1

            if len(X_shard) >= shard_size:
                Xs = np.stack(X_shard).transpose(0, 3, 1, 2)
                Ys = np.stack(Y_shard).transpose(0, 3, 1, 2)
                np.savez(out_dir / f"shard_{shard_idx:04d}.npz", X=Xs, Y=Ys)
                print(f"shard {shard_idx}: {len(X_shard)} samples (total={total})")
                X_shard, Y_shard = [], []
                shard_idx += 1

        if (i + 1) % 100 == 0:
            print(f"{i+1}/{len(paths)} total={total} skipped={skipped}")

    # Final shard
    if X_shard:
        Xs = np.stack(X_shard).transpose(0, 3, 1, 2)
        Ys = np.stack(Y_shard).transpose(0, 3, 1, 2)
        np.savez(out_dir / f"shard_{shard_idx:04d}.npz", X=Xs, Y=Ys)
        print(f"shard {shard_idx}: {len(X_shard)} samples (total={total})")

    # Save manifest
    with open(out_dir / "manifest.txt", "w") as f:
        f.write(f"total_samples={total}\nshards={shard_idx + 1}\n"
                f"size={DETAIL_SIZE}\nfg_dir={fg_dir}\n")
    print(f"done: {total} samples in {shard_idx + 1} shards -> {out_dir}")


# ---------------------------------------------------------------------------
# Training (with data loader for large datasets)
# ---------------------------------------------------------------------------

class ShardDataset:
    """Iterable over sharded .npz files."""
    def __init__(self, data_dir: str):
        self.dir = Path(data_dir)
        self.shards = sorted(self.dir.glob("shard_*.npz"))
        if not self.shards:
            raise FileNotFoundError(f"no shards in {data_dir}")
        # Read first shard to determine total
        d0 = np.load(self.shards[0], mmap_mode='r')
        self.n_per_shard = len(d0["X"])
        self.total = sum(len(np.load(s, mmap_mode='r')["X"]) for s in self.shards)

    def __len__(self):
        return self.total


def stage_train(args: argparse.Namespace) -> None:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    dataset = ShardDataset(args.data)
    print(f"loaded {len(dataset.shards)} shards, {dataset.total} total samples")
    train_size = args.size

    # Shuffle shards and use 85/15 split
    n_shards = len(dataset.shards)
    idx = torch.randperm(n_shards).tolist()
    n_train_shards = max(1, int(n_shards * 0.85))
    train_shards = [dataset.shards[i] for i in idx[:n_train_shards]]
    val_shards = [dataset.shards[i] for i in idx[n_train_shards:]]
    print(f"train shards={len(train_shards)} val shards={len(val_shards)}")

    class DetailUNet(nn.Module):
        def __init__(self, in_ch=3, out_ch=3, base=48):
            super().__init__()
            # Encoder
            self.enc1 = nn.Sequential(
                nn.Conv2d(in_ch, base, 3, padding=1), nn.BatchNorm2d(base), nn.ReLU(True),
                nn.Conv2d(base, base, 3, padding=1), nn.BatchNorm2d(base), nn.ReLU(True))
            self.pool1 = nn.MaxPool2d(2)
            self.enc2 = nn.Sequential(
                nn.Conv2d(base, base*2, 3, padding=1), nn.BatchNorm2d(base*2), nn.ReLU(True),
                nn.Conv2d(base*2, base*2, 3, padding=1), nn.BatchNorm2d(base*2), nn.ReLU(True))
            self.pool2 = nn.MaxPool2d(2)
            self.enc3 = nn.Sequential(
                nn.Conv2d(base*2, base*4, 3, padding=1), nn.BatchNorm2d(base*4), nn.ReLU(True),
                nn.Conv2d(base*4, base*4, 3, padding=1), nn.BatchNorm2d(base*4), nn.ReLU(True))
            self.pool3 = nn.MaxPool2d(2)
            # Bottleneck
            self.bottleneck = nn.Sequential(
                nn.Conv2d(base*4, base*8, 3, padding=1), nn.BatchNorm2d(base*8), nn.ReLU(True),
                nn.Conv2d(base*8, base*4, 3, padding=1), nn.BatchNorm2d(base*4), nn.ReLU(True))
            # Decoder
            self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.dec3 = nn.Sequential(
                nn.Conv2d(base*8, base*4, 3, padding=1), nn.BatchNorm2d(base*4), nn.ReLU(True),
                nn.Conv2d(base*4, base*2, 3, padding=1), nn.BatchNorm2d(base*2), nn.ReLU(True))
            self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.dec2 = nn.Sequential(
                nn.Conv2d(base*4, base*2, 3, padding=1), nn.BatchNorm2d(base*2), nn.ReLU(True),
                nn.Conv2d(base*2, base, 3, padding=1), nn.BatchNorm2d(base), nn.ReLU(True))
            self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.dec1 = nn.Sequential(
                nn.Conv2d(base*2, base, 3, padding=1), nn.BatchNorm2d(base), nn.ReLU(True),
                nn.Conv2d(base, out_ch, 3, padding=1))

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool1(e1))
            e3 = self.enc3(self.pool2(e2))
            b = self.bottleneck(self.pool3(e3))
            d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return torch.sigmoid(d1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device={device}")
    model = DetailUNet(base=48).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params={n_params:,}")

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    best_val = float('inf')
    bs = args.batch_size

    for epoch in range(args.epochs):
        # Train over all training shards
        model.train()
        train_loss = 0.0
        train_n = 0
        for shard_path in train_shards:
            data = np.load(shard_path, mmap_mode='r')
            X = torch.from_numpy(np.array(data["X"], dtype=np.float32, copy=False))
            Y = torch.from_numpy(np.array(data["Y"], dtype=np.float32, copy=False))
            if train_size != 256:
                X = F.interpolate(X, size=train_size, mode='bilinear', align_corners=False)
                Y = F.interpolate(Y, size=train_size, mode='bilinear', align_corners=False)
            w = 1.0 + 50.0 * (Y - 0.25).abs().mean(dim=1, keepdim=True)
            n = len(X)
            perm = torch.randperm(n)
            for start in range(0, n, bs):
                bi = perm[start:start + bs]
                xb, yb, wb = X[bi].to(device), Y[bi].to(device), w[bi].to(device)
                pred = model(xb)
                loss = (wb * (pred - yb) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                train_loss += loss.item() * len(bi)
                train_n += len(bi)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_n = 0
        with torch.no_grad():
            for shard_path in val_shards:
                data = np.load(shard_path, mmap_mode='r')
                X = torch.from_numpy(np.array(data["X"], dtype=np.float32, copy=False))
                Y = torch.from_numpy(np.array(data["Y"], dtype=np.float32, copy=False))
                if train_size != 256:
                    X = F.interpolate(X, size=train_size, mode='bilinear', align_corners=False)
                    Y = F.interpolate(Y, size=train_size, mode='bilinear', align_corners=False)
                w = 1.0 + 50.0 * (Y - 0.25).abs().mean(dim=1, keepdim=True)
                n = len(X)
                for start in range(0, n, bs):
                    bi = slice(start, start + bs)
                    xb, yb, wb = X[bi].to(device), Y[bi].to(device), w[bi].to(device)
                    pred = model(xb)
                    val_loss += (wb * (pred - yb) ** 2).mean().item() * len(xb)
                    val_n += len(xb)

        tloss = train_loss / max(train_n, 1)
        vloss = val_loss / max(val_n, 1)
        if (epoch + 1) % 5 == 0:
            print(f"epoch {epoch+1:3d}  train={tloss:.6f}  val={vloss:.6f}  "
                  f"lr={scheduler.get_last_lr()[0]:.6f}")
        if vloss < best_val:
            best_val = vloss
            torch.save(model.state_dict(), args.model)

    print(f"best val_loss={best_val:.6f}")
    model.load_state_dict(torch.load(args.model))
    torch.save(model.state_dict(), args.model)
    print(f"saved {args.model}")


# ---------------------------------------------------------------------------
# ONNX Export
# ---------------------------------------------------------------------------

def stage_export(args: argparse.Namespace) -> None:
    import torch
    import torch.nn as nn

    class DetailUNet(nn.Module):
        def __init__(self, in_ch=3, out_ch=3, base=48):
            super().__init__()
            self.enc1 = nn.Sequential(
                nn.Conv2d(in_ch, base, 3, padding=1), nn.BatchNorm2d(base), nn.ReLU(True),
                nn.Conv2d(base, base, 3, padding=1), nn.BatchNorm2d(base), nn.ReLU(True))
            self.pool1 = nn.MaxPool2d(2)
            self.enc2 = nn.Sequential(
                nn.Conv2d(base, base*2, 3, padding=1), nn.BatchNorm2d(base*2), nn.ReLU(True),
                nn.Conv2d(base*2, base*2, 3, padding=1), nn.BatchNorm2d(base*2), nn.ReLU(True))
            self.pool2 = nn.MaxPool2d(2)
            self.enc3 = nn.Sequential(
                nn.Conv2d(base*2, base*4, 3, padding=1), nn.BatchNorm2d(base*4), nn.ReLU(True),
                nn.Conv2d(base*4, base*4, 3, padding=1), nn.BatchNorm2d(base*4), nn.ReLU(True))
            self.pool3 = nn.MaxPool2d(2)
            self.bottleneck = nn.Sequential(
                nn.Conv2d(base*4, base*8, 3, padding=1), nn.BatchNorm2d(base*8), nn.ReLU(True),
                nn.Conv2d(base*8, base*4, 3, padding=1), nn.BatchNorm2d(base*4), nn.ReLU(True))
            self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.dec3 = nn.Sequential(
                nn.Conv2d(base*8, base*4, 3, padding=1), nn.BatchNorm2d(base*4), nn.ReLU(True),
                nn.Conv2d(base*4, base*2, 3, padding=1), nn.BatchNorm2d(base*2), nn.ReLU(True))
            self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.dec2 = nn.Sequential(
                nn.Conv2d(base*4, base*2, 3, padding=1), nn.BatchNorm2d(base*2), nn.ReLU(True),
                nn.Conv2d(base*2, base, 3, padding=1), nn.BatchNorm2d(base), nn.ReLU(True))
            self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.dec1 = nn.Sequential(
                nn.Conv2d(base*2, base, 3, padding=1), nn.BatchNorm2d(base), nn.ReLU(True),
                nn.Conv2d(base, out_ch, 3, padding=1))

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool1(e1))
            e3 = self.enc3(self.pool2(e2))
            b = self.bottleneck(self.pool3(e3))
            d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return torch.sigmoid(d1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DetailUNet(base=48).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    dummy = torch.randn(1, 3, DETAIL_SIZE, DETAIL_SIZE, device=device)
    onnx_path = args.onnx or args.model.replace('.pt', '.onnx')

    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        opset_version=17,
    )
    print(f"exported {onnx_path}")

    # Verify
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    out = sess.run(None, {'input': dummy.cpu().numpy()})[0]
    print(f"verify: in={dummy.shape} out={out.shape}")
    print("OK")


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def stage_eval(args: argparse.Namespace) -> None:
    import onnxruntime as ort
    from PIL import Image, ImageDraw

    basis = get_basis()
    sess = ort.InferenceSession(args.onnx or args.model.replace('.pt', '.onnx'),
                                providers=['CPUExecutionProvider'])

    fg_dir = Path(args.fg_dir)
    # Pick test files from the end (not in training data - approximate split)
    paths = sorted(fg_dir.glob("*.fg"))[-args.n_test:]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in paths:
        stem = path.stem
        try:
            fg = FgFile.read(str(path))
            target = _decode_detail(fg, DETAIL_SIZE)
            if target is None: continue
        except Exception: continue

        assets = _build_ootp_assets(fg, include_eyes=True)
        rendered = _render_assets(assets, fg, DETAIL_SIZE, shade=True, aa=1)
        warped, _ = _warp_texture_to_detail_uv(rendered, basis, DETAIL_SIZE)

        x = warped.astype(np.float32) / 255.0
        x = x.transpose(2, 0, 1)[None]  # (1,3,H,W)
        pred = sess.run(None, {'input': x})[0][0].transpose(1, 2, 0)
        pred = np.clip(pred * 255, 0, 255).astype(np.uint8)

        t = target.astype(np.float32)
        m = pred.astype(np.float32)
        delta = np.abs(t - 64).max(-1)
        interesting = delta > 8
        if interesting.sum():
            mi = np.abs(warped.astype(np.float32)[interesting] - t[interesting]).mean()
            mm = np.abs(m[interesting] - t[interesting]).mean()
            print(f"{stem}: input(int)={mi:.1f}  model(int)={mm:.1f}")

        # Render comparisons
        fg0 = FgFile.read(str(path))
        fg0.detail_jpeg = None
        nod = Image.fromarray(_render_assets(
            _build_ootp_assets(fg0, include_eyes=True), fg0, 384, True, 2))
        fg1 = FgFile.read(str(path))
        buf = io.BytesIO(); Image.fromarray(pred).save(buf, 'JPEG', quality=90)
        fg1.detail_jpeg = buf.getvalue()
        mod = Image.fromarray(_render_assets(
            _build_ootp_assets(fg1, include_eyes=True), fg1, 384, True, 2))
        fg2 = FgFile.read(str(path))
        tgt = Image.fromarray(_render_assets(
            _build_ootp_assets(fg2, include_eyes=True), fg2, 384, True, 2))
        w, h = nod.size; gap = 8
        out = Image.new('RGB', (w*3+gap*2, h+36), (30,30,30))
        out.paste(nod, (0,36)); out.paste(mod, (w+gap,36)); out.paste(tgt, (w*2+gap*2,36))
        draw = ImageDraw.Draw(out)
        for i,lab in enumerate(['NO DETAIL', 'CNN MODEL', 'TARGET']):
            draw.text((i*(w+gap)+8,8), lab, fill=(255,255,255))
        out.save(out_dir / f"{stem}_compare.png")

    print(f"\ndone -> {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    gen = sub.add_parser("gen")
    gen.add_argument("--fg-dir", default=str(DEFAULT_FG_DIR))
    gen.add_argument("--out", default="models/detail_cnn_data")
    gen.add_argument("--n", type=int, default=5000)
    gen.add_argument("--augs", type=int, default=2)
    gen.add_argument("--aug-strength", type=float, default=1.0)
    gen.add_argument("--shard-size", type=int, default=1000)
    gen.add_argument("--seed", type=int, default=0)

    train = sub.add_parser("train")
    train.add_argument("--data", default="models/detail_cnn_data")
    train.add_argument("--model", default="models/detail_cnn.pt")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--lr", type=float, default=0.001)
    train.add_argument("--size", type=int, default=128)

    exp = sub.add_parser("export")
    exp.add_argument("--model", default="models/detail_cnn.pt")
    exp.add_argument("--onnx", default=None)

    ev = sub.add_parser("eval")
    ev.add_argument("--model", default="models/detail_cnn.pt")
    ev.add_argument("--onnx", default=None)
    ev.add_argument("--fg-dir", default=str(DEFAULT_FG_DIR))
    ev.add_argument("--out-dir", default="models/detail_cnn_eval")
    ev.add_argument("--n-test", type=int, default=5)

    args = p.parse_args()
    d = vars(args)
    if d.get("cmd") == "gen":
        stage_gen(args)
    elif d.get("cmd") == "train":
        stage_train(args)
    elif d.get("cmd") == "export":
        stage_export(args)
    elif d.get("cmd") == "eval":
        stage_eval(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
