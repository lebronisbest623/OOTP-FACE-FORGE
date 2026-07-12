"""CNN smoke test: U-Net for detail map prediction.

Usage:
  python scripts/smoke_cnn.py gen   --out models/smoke_cnn_data.npz --n 200
  python scripts/smoke_cnn.py train --data models/smoke_cnn_data.npz --model models/smoke_cnn.pt --epochs 50
  python scripts/smoke_cnn.py eval  --data models/smoke_cnn_data.npz --model models/smoke_cnn.pt --out-dir models/smoke_cnn_eval
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

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
# Data generation (same logic, but saves full images)
# ---------------------------------------------------------------------------

def _warp_texture_to_detail_uv(img: np.ndarray, basis, size: int):
    """Warp a rendered image into detail-map UV space."""
    from ootp_faceforge.texture import detail_px

    dpx, dvalid = detail_px(basis, size)
    vert_uv = basis.vert_uv
    tris = basis.tris
    tri_uv = vert_uv[tris]
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
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
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
        xx = xx / max(w - 1, 1) - 0.5
        yy = yy / max(h - 1, 1) - 0.5
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        grad = xx * np.cos(theta) + yy * np.sin(theta)
        amp = float(rng.normal(0.0, 0.18 * severity))
        gain = np.clip(1.0 + amp * grad, 0.65, 1.40)
        out *= gain[..., None]
    if rng.random() < 0.3:
        sigma = float(rng.uniform(0.3, 0.9) * severity)
        out = cv2.GaussianBlur(out, (0, 0), sigma)
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


def stage_gen(args: argparse.Namespace) -> None:
    basis = get_basis()
    fg_dir = Path(args.fg_dir)
    paths = sorted(fg_dir.glob("*.fg"))[: args.n]
    if not paths:
        raise SystemExit(f"no .fg files: {fg_dir}")

    rng = np.random.default_rng(args.seed)
    X_list, Y_list = [], []
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

        for aug_id in range(max(int(args.augs), 1)):
            img = rendered if aug_id == 0 else augment_render(
                rendered, rng, severity=args.aug_strength)
            warped, valid = _warp_texture_to_detail_uv(img, basis, DETAIL_SIZE)
            if not valid.any():
                continue
            X_list.append(warped.astype(np.float32) / 255.0)
            Y_list.append(target.astype(np.float32) / 255.0)

        if (i + 1) % 25 == 0:
            print(f"{i+1}/{len(paths)} samples={len(X_list)} skipped={skipped}")

    if not X_list:
        raise SystemExit("no usable samples")
    X = np.stack(X_list).transpose(0, 3, 1, 2)  # (N,3,H,W)
    Y = np.stack(Y_list).transpose(0, 3, 1, 2)
    print(f"X={X.shape} Y={Y.shape} range=[{Y.min():.3f},{Y.max():.3f}]")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, X=X, Y=Y)
    print(f"saved {out}")


# ---------------------------------------------------------------------------
# U-Net model (PyTorch)
# ---------------------------------------------------------------------------

def stage_train(args: argparse.Namespace) -> None:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    data = np.load(args.data)
    X = torch.from_numpy(data["X"].astype(np.float32))
    Y = torch.from_numpy(data["Y"].astype(np.float32))
    print(f"loaded X={X.shape} Y={Y.shape}")

    n = len(X)
    n_train = int(n * 0.85)
    n_val = n - n_train
    idx = torch.randperm(n)
    X_train, Y_train = X[idx[:n_train]], Y[idx[:n_train]]
    X_val, Y_val = X[idx[n_train:]], Y[idx[n_train:]]
    print(f"train={n_train} val={n_val}")

    class TinyUNet(nn.Module):
        def __init__(self):
            super().__init__()
            # Encoder
            self.enc1 = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
                nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True))
            self.pool1 = nn.MaxPool2d(2)
            self.enc2 = nn.Sequential(
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
                nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True))
            self.pool2 = nn.MaxPool2d(2)
            # Bottleneck
            self.bottleneck = nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
                nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True))
            # Decoder
            self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.dec2 = nn.Sequential(
                nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
                nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True))
            self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.dec1 = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
                nn.Conv2d(32, 3, 3, padding=1))

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool1(e1))
            b = self.bottleneck(self.pool2(e2))
            d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return torch.sigmoid(d1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device={device}")
    model = TinyUNet().to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    # Weight mask: pixels far from 0.25 (~64/255) get higher weight
    weight = 1.0 + 50.0 * (Y_train - 0.25).abs().mean(dim=1, keepdim=True)
    weight_val = 1.0 + 50.0 * (Y_val - 0.25).abs().mean(dim=1, keepdim=True)

    best_val = float('inf')
    bs = args.batch_size
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n_train)
        train_loss = 0.0
        for start in range(0, n_train, bs):
            batch_idx = perm[start:start + bs]
            xb = X_train[batch_idx].to(device)
            yb = Y_train[batch_idx].to(device)
            wb = weight[batch_idx].to(device)
            pred = model(xb)
            loss = (wb * (pred - yb) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(batch_idx)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            xv = X_val.to(device)
            yv = Y_val.to(device)
            wv = weight_val.to(device)
            pred_val = model(xv)
            vloss = (wv * (pred_val - yv) ** 2).mean().item()

        if (epoch + 1) % 5 == 0:
            print(f"epoch {epoch+1:3d}  train={train_loss/n_train:.6f}  val={vloss:.6f}")
        if vloss < best_val:
            best_val = vloss
            torch.save(model.state_dict(), args.model)

    print(f"best val_loss={best_val:.6f}")
    model.load_state_dict(torch.load(args.model))
    torch.save(model.state_dict(), args.model)
    print(f"saved {args.model}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def stage_eval(args: argparse.Namespace) -> None:
    import torch
    import torch.nn as nn

    class TinyUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
                nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True))
            self.pool1 = nn.MaxPool2d(2)
            self.enc2 = nn.Sequential(
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
                nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True))
            self.pool2 = nn.MaxPool2d(2)
            self.bottleneck = nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
                nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True))
            self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.dec2 = nn.Sequential(
                nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
                nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True))
            self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.dec1 = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
                nn.Conv2d(32, 3, 3, padding=1))

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool1(e1))
            b = self.bottleneck(self.pool2(e2))
            d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return torch.sigmoid(d1)

    basis = get_basis()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TinyUNet().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    fg_dir = Path(args.fg_dir)
    paths = sorted(fg_dir.glob("*.fg"))[-args.n_test:]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in paths:
        stem = path.stem
        try:
            fg = FgFile.read(str(path))
            target = _decode_detail(fg, DETAIL_SIZE)
            if target is None:
                continue
        except Exception:
            continue

        assets = _build_ootp_assets(fg, include_eyes=True)
        rendered = _render_assets(assets, fg, DETAIL_SIZE, shade=True, aa=1)
        warped, valid = _warp_texture_to_detail_uv(rendered, basis, DETAIL_SIZE)

        x = torch.from_numpy(warped.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x)[0].permute(1, 2, 0).cpu().numpy()
        pred = np.clip(pred * 255, 0, 255).astype(np.uint8)

        Image.fromarray(warped).save(out_dir / f"{stem}_input.png")
        Image.fromarray(target.astype(np.uint8)).save(out_dir / f"{stem}_target.png")
        Image.fromarray(pred).save(out_dir / f"{stem}_model.png")

        t = target.astype(np.float32)
        m = pred.astype(np.float32)
        delta = np.abs(t - 64).max(-1)
        interesting = delta > 8
        if interesting.sum():
            mi = np.abs(warped.astype(np.float32)[interesting] - t[interesting]).mean()
            mm = np.abs(m[interesting] - t[interesting]).mean()
            print(f"{stem}: input(int)={mi:.1f}  model(int)={mm:.1f}")

    # Render comparison
    for stem in [paths[0].stem, paths[1].stem]:
        fg0 = FgFile.read(str(fg_dir / f"{stem}.fg"))
        fg0.detail_jpeg = None
        nod = Image.fromarray(_render_assets(
            _build_ootp_assets(fg0, include_eyes=True), fg0, 384, True, 2))
        fg1 = FgFile.read(str(fg_dir / f"{stem}.fg"))
        model_d = Image.open(out_dir / f"{stem}_model.png")
        buf = io.BytesIO(); model_d.save(buf, 'JPEG', quality=90)
        fg1.detail_jpeg = buf.getvalue()
        mod = Image.fromarray(_render_assets(
            _build_ootp_assets(fg1, include_eyes=True), fg1, 384, True, 2))
        fg2 = FgFile.read(str(fg_dir / f"{stem}.fg"))
        tgt = Image.fromarray(_render_assets(
            _build_ootp_assets(fg2, include_eyes=True), fg2, 384, True, 2))
        w, h = nod.size; gap = 8
        out = Image.new('RGB', (w*3+gap*2, h+36), (30,30,30))
        out.paste(nod, (0,36)); out.paste(mod, (w+gap,36)); out.paste(tgt, (w*2+gap*2,36))
        draw = ImageDraw.Draw(out)
        for i,lab in enumerate(['NO DETAIL', 'CNN MODEL', 'TARGET official']):
            draw.text((i*(w+gap)+8,8), lab, fill=(255,255,255))
        out.save(out_dir / f"{stem}_compare.png")
        print(f"saved {stem}_compare.png")

    print(f"\ndone -> {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    gen = sub.add_parser("gen")
    gen.add_argument("--fg-dir", default=str(DEFAULT_FG_DIR))
    gen.add_argument("--out", default="models/smoke_cnn_data.npz")
    gen.add_argument("--n", type=int, default=200)
    gen.add_argument("--augs", type=int, default=3)
    gen.add_argument("--aug-strength", type=float, default=1.0)
    gen.add_argument("--seed", type=int, default=0)

    train = sub.add_parser("train")
    train.add_argument("--data", default="models/smoke_cnn_data.npz")
    train.add_argument("--model", default="models/smoke_cnn.pt")
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--lr", type=float, default=0.001)

    ev = sub.add_parser("eval")
    ev.add_argument("--data", default="models/smoke_cnn_data.npz")
    ev.add_argument("--model", default="models/smoke_cnn.pt")
    ev.add_argument("--fg-dir", default=str(DEFAULT_FG_DIR))
    ev.add_argument("--out-dir", default="models/smoke_cnn_eval")
    ev.add_argument("--n-test", type=int, default=5)

    args = p.parse_args()
    if args.cmd == "gen":
        stage_gen(args)
    elif args.cmd == "train":
        stage_train(args)
    elif args.cmd == "eval":
        stage_eval(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
