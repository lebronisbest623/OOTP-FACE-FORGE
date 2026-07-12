"""Smoke test: can a learned model extract clean detail maps better than the
hand-crafted neutralization pipeline?

Stage 1 (gen):  Render official .fg files with augmentation to create
                (warped_texture, target_detail) training pairs.
Stage 2 (train): Fit a tiny patch→pixel model (linear → tiny CNN).
Stage 3 (eval):  Compare model output vs current pipeline on hold-out .fg files.

Usage:
  python scripts/smoke_detail_model.py gen   --out models/smoke_detail_data.npz --n 100
  python scripts/smoke_detail_model.py train --data models/smoke_detail_data.npz --model models/smoke_detail_model.npz
  python scripts/smoke_detail_model.py eval  --data models/smoke_detail_data.npz --model models/smoke_detail_model.npz --out-dir models/smoke_eval
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ootp_faceforge.basis import get_basis  # noqa: E402
from ootp_faceforge.fgformat import FgFile  # noqa: E402

DEFAULT_FG_DIR = Path(
    r"C:\Users\user\Documents\Out of the Park Developments"
    r"\OOTP Baseball 27\fg_files"
)
DETAIL_SIZE = 256
N_SYM, N_ASYM, N_TEX = 50, 30, 50


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def _render_fg(fg: FgFile, basis, size: int, apply_detail: bool) -> np.ndarray:
    """Render a .fg to an RGB image using the local OOTP renderer."""
    from ootp_faceforge.render import _build_ootp_assets, _render_assets

    # Build assets without detail so we can control it
    assets = _build_ootp_assets(fg, include_eyes=True)
    if apply_detail:
        # _build_ootp_assets bakes detail into face_hi, so use as-is
        pass
    return _render_assets(assets, fg, size, shade=True, aa=1)


def _warp_texture_to_detail_uv(img: np.ndarray, basis, size: int) -> np.ndarray:
    """Warp a rendered image into detail-map UV space using the FIM mapping."""
    from ootp_faceforge.texture import detail_px

    dpx, dvalid = detail_px(basis, size)
    vert_uv = basis.vert_uv
    tris = basis.tris
    tri_uv = vert_uv[tris]  # (T,3,2) UV coords
    src_tri = tri_uv * np.array([img.shape[1], img.shape[0]], np.float32)
    dst_tri = dpx[tris]

    h, w = img.shape[:2]
    out = np.full((size, size, 3), 64.0, np.float32)
    valid = np.zeros((size, size), bool)

    for ti in range(len(tris)):
        src = src_tri[ti].astype(np.float32)
        dst = dst_tri[ti].astype(np.float32)

        # Bounding box in output
        x0 = max(int(np.floor(dst[:, 0].min())), 0)
        x1 = min(int(np.ceil(dst[:, 0].max())) + 1, size)
        y0 = max(int(np.floor(dst[:, 1].min())), 0)
        y1 = min(int(np.ceil(dst[:, 1].max())) + 1, size)
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue

        dst_local = dst - [x0, y0]
        M = cv2.getAffineTransform(src.astype(np.float32), dst_local.astype(np.float32))
        patch = cv2.warpAffine(
            img, M, (x1 - x0, y1 - y0),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillConvexPoly(mask, np.round(dst_local).astype(np.int32), 1)
        mb = mask.astype(bool)
        if mb.any():
            out[y0:y1, x0:x1][mb] = patch[mb]
            valid[y0:y1, x0:x1][mb] = True

    return np.clip(out, 0, 255).astype(np.uint8), valid


def augment_render(img: np.ndarray, rng: np.random.Generator,
                   severity: float = 1.0) -> np.ndarray:
    """Apply photo-like augmentation to a clean render so the model learns to
    handle real-world variation (lighting, noise, compression)."""
    out = img.astype(np.float32)
    h, w = img.shape[:2]

    # Colour balance shift
    gains = rng.normal(1.0, 0.04 * severity, 3).astype(np.float32)
    out *= gains[None, None, :]

    # Exposure / contrast
    contrast = 1.0 + rng.normal(0.0, 0.10 * severity)
    bright = rng.normal(0.0, 10.0 * severity)
    out = (out - 127.5) * contrast + 127.5 + bright

    # Lighting gradient across the face
    if rng.random() < 0.6:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        xx = xx / max(w - 1, 1) - 0.5
        yy = yy / max(h - 1, 1) - 0.5
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        grad = xx * np.cos(theta) + yy * np.sin(theta)
        amp = float(rng.normal(0.0, 0.18 * severity))
        gain = np.clip(1.0 + amp * grad, 0.65, 1.40)
        out *= gain[..., None]

    # Mild blur
    if rng.random() < 0.3:
        sigma = float(rng.uniform(0.3, 0.9) * severity)
        out = cv2.GaussianBlur(out, (0, 0), sigma)

    # Noise
    if rng.random() < 0.3:
        noise = rng.normal(0.0, 3.0 * severity, out.shape).astype(np.float32)
        out += noise

    out = np.clip(out, 0, 255).astype(np.uint8)

    # JPEG compression
    if rng.random() < 0.6:
        buf = io.BytesIO()
        Image.fromarray(out).save(buf, "JPEG",
                                  quality=int(rng.integers(50, 92)))
        out = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))

    return out


def _decode_detail(fg: FgFile, size: int) -> np.ndarray | None:
    """Decode a .fg's detail JPEG to a (size,size,3) float32 array, or None."""
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
    paths = sorted(fg_dir.glob("*.fg"))
    if not paths:
        raise SystemExit(f"no .fg files found: {fg_dir}")
    if args.n:
        paths = paths[: args.n]

    rng = np.random.default_rng(args.seed)
    X_patches, Y_pixels = [], []
    count, skipped = 0, 0

    for i, path in enumerate(paths):
        try:
            fg = FgFile.read(str(path))
            if (len(fg.sym_shape) != N_SYM or len(fg.asym_shape) != N_ASYM
                    or len(fg.sym_tex) != N_TEX):
                skipped += 1
                continue
            target = _decode_detail(fg, DETAIL_SIZE)
            if target is None:
                skipped += 1
                continue
        except Exception:
            skipped += 1
            continue

        # Render WITH detail → this is our "photo-like" source
        try:
            rendered = _render_fg(fg, basis, DETAIL_SIZE, apply_detail=True)
        except Exception:
            skipped += 1
            continue

        # Generate augmented versions
        for aug_id in range(max(int(args.augs), 1)):
            img = rendered if aug_id == 0 else augment_render(
                rendered, rng, severity=args.aug_strength)

            # Warp the "photo" into detail UV space → model input
            warped, valid = _warp_texture_to_detail_uv(img, basis, DETAIL_SIZE)
            if not valid.any():
                continue

            # Extract patches and target pixels (subsampled for speed)
            step = args.patch_stride
            half = args.patch_radius
            padded = np.pad(warped, ((half, half), (half, half), (0, 0)),
                            mode='edge')

            for y in range(half, DETAIL_SIZE - half, step):
                for x in range(half, DETAIL_SIZE - half, step):
                    if not valid[y, x]:
                        continue
                    patch = padded[y - half:y + half + 1,
                                    x - half:x + half + 1, :]
                    feat = patch.astype(np.float32).reshape(-1).tolist()
                    feat.append(x / DETAIL_SIZE)  # spatial context
                    feat.append(y / DETAIL_SIZE)
                    X_patches.append(np.array(feat, np.float32))
                    Y_pixels.append(target[y, x].astype(np.float32))

            count += 1
            if (i + 1) % 20 == 0:
                print(f"{i + 1}/{len(paths)} samples={len(X_patches)} "
                      f"skipped={skipped}", flush=True)

    if not X_patches:
        raise SystemExit("no usable training samples generated")

    X = np.stack(X_patches)
    Y = np.stack(Y_pixels)
    print(f"generated {len(X)} samples, X={X.shape}, Y={Y.shape}")
    print(f"Y range: [{Y.min():.1f}, {Y.max():.1f}]")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, X=X, Y=Y, patch_radius=args.patch_radius,
             patch_stride=args.patch_stride)
    print(f"saved {out}")


# ---------------------------------------------------------------------------
# Training (tiny patch→pixel model)
# ---------------------------------------------------------------------------

class PatchRegressor:
    """Tiny 2-hidden-layer MLP: patch → center pixel RGB detail value."""

    def __init__(self, input_dim: int, hidden: int = 32):
        rng = np.random.default_rng(42)
        self.w1 = rng.normal(0, np.sqrt(2.0 / input_dim),
                             (input_dim, hidden)).astype(np.float32)
        self.b1 = np.zeros(hidden, np.float32)
        self.w2 = rng.normal(0, np.sqrt(2.0 / hidden),
                             (hidden, hidden)).astype(np.float32)
        self.b2 = np.zeros(hidden, np.float32)
        self.w3 = rng.normal(0, np.sqrt(2.0 / hidden),
                             (hidden, 3)).astype(np.float32)
        self.b3 = np.full(3, 64.0, np.float32)  # detail neutral = 64

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.z1 = X @ self.w1 + self.b1
        self.a1 = np.maximum(self.z1, 0)  # relu
        self.z2 = self.a1 @ self.w2 + self.b2
        self.a2 = np.maximum(self.z2, 0)
        return self.a2 @ self.w3 + self.b3

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)


def stage_train(args: argparse.Namespace) -> None:
    data = np.load(args.data)
    X = data["X"].astype(np.float32)
    Y = data["Y"].astype(np.float32)
    patch_radius = int(data["patch_radius"])

    # Normalize inputs to ~[0,1]
    X = X / 255.0
    # Keep Y in [0,255] for MSE with weight toward neutral (64)

    # Train/val split (90/10)
    n = len(X)
    n_train = int(n * 0.9)
    idx = np.random.default_rng(42).permutation(n)
    X_train, Y_train = X[idx[:n_train]], Y[idx[:n_train]]
    X_val, Y_val = X[idx[n_train:]], Y[idx[n_train:]]

    input_dim = X.shape[1]
    model = PatchRegressor(input_dim, hidden=args.hidden)

    lr = args.lr
    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(args.epochs):
        # Mini-batch SGD
        perm = np.random.permutation(n_train)
        total_loss = 0.0
        for start in range(0, n_train, args.batch_size):
            batch_idx = perm[start:start + args.batch_size]
            Xb, Yb = X_train[batch_idx], Y_train[batch_idx]

            # Forward
            pred = model.forward(Xb)
            err = pred - Yb
            loss = float((err ** 2).mean())

            # Backward (manual for tiny 2-layer net)
            m = len(Xb)
            # layer 3
            dw3 = model.a2.T @ err / m
            db3 = err.mean(0)
            da2 = err @ model.w3.T
            # layer 2
            dz2 = da2 * (model.z2 > 0)
            dw2 = model.a1.T @ dz2 / m
            db2 = dz2.mean(0)
            da1 = dz2 @ model.w2.T
            # layer 1
            dz1 = da1 * (model.z1 > 0)
            dw1 = Xb.T @ dz1 / m
            db1 = dz1.mean(0)

            # Update
            for param, grad in [
                (model.w3, dw3), (model.b3, db3),
                (model.w2, dw2), (model.b2, db2),
                (model.w1, dw1), (model.b1, db1),
            ]:
                param -= lr * grad.astype(param.dtype)

            total_loss += loss * m

        # Validation
        val_pred = model.predict(X_val)
        val_loss = float(((val_pred - Y_val) ** 2).mean())

        if (epoch + 1) % max(1, args.epochs // 10) == 0:
            print(f"epoch {epoch + 1:4d}  train_loss={total_loss / n_train:.4f}  "
                  f"val_loss={val_loss:.4f}  "
                  f"pred_range=[{val_pred.min():.1f},{val_pred.max():.1f}]")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {
                "w1": model.w1, "b1": model.b1,
                "w2": model.w2, "b2": model.b2,
                "w3": model.w3, "b3": model.b3,
            }

    if best_weights is not None:
        for k, v in best_weights.items():
            setattr(model, k, v)
        print(f"best val_loss={best_val_loss:.4f}")

    out = Path(args.model)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **best_weights,
             input_dim=input_dim, hidden=args.hidden,
             patch_radius=patch_radius,
             val_loss=float(best_val_loss))
    print(f"saved {out}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _predict_detail_map(model: PatchRegressor, warped: np.ndarray,
                         valid: np.ndarray, patch_radius: int,
                         detail_size: int) -> np.ndarray:
    """Run the patch model over every valid pixel to produce a full detail map."""
    half = patch_radius
    padded = np.pad(warped.astype(np.float32),
                    ((half, half), (half, half), (0, 0)), mode='edge')
    D = np.full((detail_size, detail_size, 3), 64.0, np.float32)

    for y in range(detail_size):
        for x in range(detail_size):
            if not valid[y, x]:
                continue
            patch = padded[y:y + 2 * half + 1, x:x + 2 * half + 1, :]
            feat = patch.astype(np.float32).reshape(-1).tolist()
            feat.append(x / detail_size)
            feat.append(y / detail_size)
            feat_arr = np.array(feat, np.float32).reshape(1, -1) / 255.0
            D[y, x] = model.predict(feat_arr)[0]

    return np.clip(D, 0, 255)


def stage_eval(args: argparse.Namespace) -> None:
    from ootp_faceforge.texture import build_detail as pipeline_build_detail
    from ootp_faceforge import landmarks as lm

    basis = get_basis()
    model_data = np.load(args.model, allow_pickle=True)
    model = PatchRegressor(int(model_data["input_dim"]), int(model_data["hidden"]))
    for k in ["w1", "b1", "w2", "b2", "w3", "b3"]:
        setattr(model, k, model_data[k])
    patch_radius = int(model_data["patch_radius"])

    # Load a few hold-out .fg files
    fg_dir = Path(args.fg_dir)
    paths = sorted(fg_dir.glob("*.fg"))[-args.n_test:]
    if not paths:
        raise SystemExit("no test .fg files")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, path in enumerate(paths[:args.n_test]):
        try:
            fg = FgFile.read(str(path))
            target = _decode_detail(fg, DETAIL_SIZE)
            if target is None:
                continue
        except Exception:
            continue

        # Render the face (as our simulated "photo")
        rendered = _render_fg(fg, basis, DETAIL_SIZE, apply_detail=True)

        # Warp to detail UV space
        warped, valid = _warp_texture_to_detail_uv(rendered, basis, DETAIL_SIZE)

        # Model prediction
        model_detail = _predict_detail_map(model, warped, valid, patch_radius,
                                           DETAIL_SIZE)

        # Pipeline comparison: run the hand-crafted build_detail on the warped texture
        # We use a minimal set of params matching the build pipeline defaults
        try:
            lms = lm.detect(rendered)
        except Exception:
            lms = None

        # For pipeline comparison, we need proj2d etc. — simplify by comparing
        # the raw warped input to the model output directly vs target.

        # Save comparison images
        stem = path.stem
        Image.fromarray(np.clip(warped, 0, 255).astype(np.uint8)).save(
            out_dir / f"{stem}_input.png")
        Image.fromarray(target.astype(np.uint8)).save(
            out_dir / f"{stem}_target.png")
        Image.fromarray(model_detail.astype(np.uint8)).save(
            out_dir / f"{stem}_model.png")

        # Compute per-pixel MAE vs target (valid pixels only)
        if valid.any():
            mae_model = float(np.abs(model_detail[valid] - target[valid]).mean())
            mae_input = float(np.abs(warped[valid] - target[valid]).mean())
            print(f"{stem}: input_mae={mae_input:.2f} model_mae={mae_model:.2f}")

    print(f"\ncomparison images saved to {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Smoke-test a learned detail model.")
    sub = p.add_subparsers(dest="cmd")

    gen = sub.add_parser("gen")
    gen.add_argument("--fg-dir", default=str(DEFAULT_FG_DIR))
    gen.add_argument("--out", default="models/smoke_detail_data.npz")
    gen.add_argument("--n", type=int, default=100)
    gen.add_argument("--augs", type=int, default=3)
    gen.add_argument("--aug-strength", type=float, default=1.0)
    gen.add_argument("--patch-radius", type=int, default=3)
    gen.add_argument("--patch-stride", type=int, default=4)
    gen.add_argument("--seed", type=int, default=0)

    train = sub.add_parser("train")
    train.add_argument("--data", default="models/smoke_detail_data.npz")
    train.add_argument("--model", default="models/smoke_detail_model.npz")
    train.add_argument("--hidden", type=int, default=32)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--lr", type=float, default=0.001)

    eval_p = sub.add_parser("eval")
    eval_p.add_argument("--data", default="models/smoke_detail_data.npz")
    eval_p.add_argument("--model", default="models/smoke_detail_model.npz")
    eval_p.add_argument("--fg-dir", default=str(DEFAULT_FG_DIR))
    eval_p.add_argument("--out-dir", default="models/smoke_eval")
    eval_p.add_argument("--n-test", type=int, default=5)

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
