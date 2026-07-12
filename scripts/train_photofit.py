"""Train and evaluate the fast direct photofit regressor.

The old emb2shape model learns from ArcFace embedding only. This script trains a
slightly richer direct model from:

  ArcFace embedding + normalized MediaPipe landmark geometry -> FaceGen coeffs

The generated model is intentionally simple (.npz ridge regression) so builds can
use it with only numpy. Slow render/identity refinement remains useful as QA, not
as the primary runtime path.

Usage:
  python scripts/train_photofit.py gen   --out models/photofit_data.npz --augs 4
  python scripts/train_photofit.py train --data models/photofit_data.npz --model models/photofit.npz
  python scripts/train_photofit.py eval  --data models/photofit_data.npz --model models/photofit.npz --out-dir models
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ootp_faceforge import emb2shape, identity, photofit  # noqa: E402
from ootp_faceforge.fgformat import FgFile  # noqa: E402


DEFAULT_FG_DIR = Path(
    r"C:\Program Files (x86)\Steam\steamapps\common"
    r"\Out of the Park Baseball 27\data\fg_files"
)
DEFAULT_SIZE = 192
N_SYM, N_ASYM, N_TEX = 50, 30, 50


def _parse_lambdas(text: str) -> list[float]:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("at least one lambda is required")
    return vals


def _target_from_fg(path: Path) -> tuple[FgFile, np.ndarray] | None:
    fg = FgFile.read(str(path))
    if (len(fg.sym_shape) != N_SYM or len(fg.asym_shape) != N_ASYM
            or len(fg.sym_tex) != N_TEX):
        return None
    y = np.concatenate([fg.sym_shape, fg.asym_shape, fg.sym_tex])
    return fg, y.astype(np.float32)


def _render_fg(fg: FgFile, size: int) -> np.ndarray:
    from ootp_faceforge.render import render

    img, _asset_name = render(fg, size=size, shade=True, aa=1)
    return img


def _render_y(y: np.ndarray, size: int) -> np.ndarray:
    fg = FgFile(
        sym_shape=y[:N_SYM],
        asym_shape=y[N_SYM:N_SYM + N_ASYM],
        sym_tex=y[N_SYM + N_ASYM:N_SYM + N_ASYM + N_TEX],
        asym_tex=np.zeros(0),
        detail_jpeg=None,
    )
    return _render_fg(fg, size)


def _jpeg_roundtrip(img: np.ndarray, quality: int) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(
        buf, "JPEG", quality=int(np.clip(quality, 25, 95)))
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))


def augment_image(img: np.ndarray, rng: np.random.Generator,
                  severity: float = 1.0) -> np.ndarray:
    """Make a clean OOTP render look more like a web/player headshot."""
    severity = float(max(severity, 0.0))
    if severity <= 0:
        return img.copy()

    h, w = img.shape[:2]
    out = img.astype(np.float32)

    scale = 1.0 + rng.normal(0.0, 0.025 * severity)
    tx = rng.normal(0.0, 0.025 * severity * w)
    ty = rng.normal(0.0, 0.025 * severity * h)
    M = np.array([[scale, 0.0, tx], [0.0, scale, ty]], np.float32)
    out = cv2.warpAffine(
        out, M, (w, h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(190, 190, 190))

    if rng.random() < 0.45:
        factor = float(rng.uniform(0.55, 0.9))
        small = cv2.resize(out, (max(32, int(w * factor)), max(32, int(h * factor))),
                           interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    contrast = 1.0 + rng.normal(0.0, 0.12 * severity)
    bright = rng.normal(0.0, 12.0 * severity)
    out = (out - 127.5) * contrast + 127.5 + bright

    gains = rng.normal(1.0, 0.035 * severity, 3).astype(np.float32)
    out *= gains[None, None, :]

    if rng.random() < 0.8:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        xx = xx / max(w - 1, 1) - 0.5
        yy = yy / max(h - 1, 1) - 0.5
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        grad = xx * np.cos(theta) + yy * np.sin(theta)
        amp = float(rng.normal(0.0, 0.22 * severity))
        gain = np.clip(1.0 + amp * grad, 0.62, 1.45)
        out *= gain[..., None]

    if rng.random() < 0.35:
        sigma = float(rng.uniform(0.35, 1.15) * severity)
        out = cv2.GaussianBlur(out, (0, 0), sigma)

    if rng.random() < 0.35:
        noise = rng.normal(0.0, 3.5 * severity, out.shape).astype(np.float32)
        out += noise

    out = np.clip(out, 0, 255).astype(np.uint8)
    if rng.random() < 0.75:
        out = _jpeg_roundtrip(out, int(rng.integers(45, 92)))
    return out


def stage_gen(args: argparse.Namespace) -> None:
    from ootp_faceforge.landmarks import detect

    if not identity.available(args.id_model):
        raise SystemExit("ArcFace model not found; run scripts/download_restore_model.py")

    fg_dir = Path(args.fg_dir)
    paths = sorted(fg_dir.glob("*.fg"))
    if not paths:
        raise SystemExit(f"no .fg files found: {fg_dir}")
    if args.limit:
        paths = paths[: args.limit]

    rng = np.random.default_rng(args.seed)
    X, Y, names, aug_ids = [], [], [], []
    skipped = 0
    for i, path in enumerate(paths):
        try:
            pair = _target_from_fg(path)
            if pair is None:
                skipped += 1
                continue
            fg, y = pair
            clean = _render_fg(fg, args.render_size)
        except Exception:
            skipped += 1
            continue

        for aug_id in range(max(int(args.augs), 1)):
            img = clean if aug_id == 0 else augment_image(
                clean, rng, severity=args.aug_strength)
            try:
                lms = detect(img)
                feat = photofit.feature_from_image(img, lms, args.id_model)
            except Exception:
                skipped += 1
                continue
            if feat is None:
                skipped += 1
                continue
            X.append(feat.feature)
            Y.append(y)
            names.append(path.stem)
            aug_ids.append(aug_id)
        if (i + 1) % 250 == 0:
            print(f"{i + 1}/{len(paths)} samples={len(X)} skipped={skipped}",
                  flush=True)

    if not X:
        raise SystemExit("no usable photofit samples generated")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        X=np.stack(X).astype(np.float32),
        Y=np.stack(Y).astype(np.float32),
        names=np.asarray(names),
        aug_ids=np.asarray(aug_ids, np.int16),
        feature_lms=np.asarray(photofit.FEATURE_LMS, np.int32),
        feature_names=np.asarray(photofit.feature_names()),
        fg_dir=str(fg_dir),
        render_size=int(args.render_size),
        augs=int(args.augs),
        aug_strength=float(args.aug_strength),
    )
    print(f"wrote {out}: samples={len(X)} identities={len(set(names))} "
          f"skipped={skipped}")


def _split(X: np.ndarray, Y: np.ndarray, holdout: int, seed: int):
    if len(X) < 3:
        raise ValueError("need at least 3 samples")
    holdout = min(int(holdout), max(1, len(X) // 5), len(X) - 1)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    te, tr = idx[:holdout], idx[holdout:]
    return X[tr], Y[tr], X[te], Y[te], te


def _standardize(Xtr: np.ndarray, Xte: np.ndarray):
    mean = Xtr.mean(0).astype(np.float32)
    scale = Xtr.std(0).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    return (Xtr - mean) / scale, (Xte - mean) / scale, mean, scale


def _r2(err: np.ndarray, truth: np.ndarray) -> np.ndarray:
    denom = np.maximum(((truth - truth.mean(0)) ** 2).sum(0), 1e-9)
    return 1.0 - (err ** 2).sum(0) / denom


def stage_train(args: argparse.Namespace) -> None:
    d = np.load(args.data, allow_pickle=True)
    Xtr, Ytr, Xte, Yte, _ = _split(d["X"], d["Y"], args.holdout, args.seed)
    Xtr_s, Xte_s, x_mean, x_scale = _standardize(Xtr, Xte)
    A = np.hstack([Xtr_s, np.ones((len(Xtr_s), 1), np.float32)])
    At = np.hstack([Xte_s, np.ones((len(Xte_s), 1), np.float32)])
    reg = np.eye(A.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0

    best = None
    for lam in _parse_lambdas(args.lambdas):
        W = np.linalg.solve(A.T @ A + lam * reg, A.T @ Ytr)
        P = At @ W
        r2 = _r2(P - Yte, Yte)
        metrics = {
            "all": float(np.mean(r2)),
            "shape": float(np.mean(r2[:N_SYM + N_ASYM])),
            "sym": float(np.mean(r2[:N_SYM])),
            "asym": float(np.mean(r2[N_SYM:N_SYM + N_ASYM])),
            "tex": float(np.mean(r2[N_SYM + N_ASYM:])),
        }
        print(f"lam={lam}: mean R2 all={metrics['all']:.3f} "
              f"shape={metrics['shape']:.3f} sym={metrics['sym']:.3f} "
              f"asym={metrics['asym']:.3f} tex={metrics['tex']:.3f}")
        score = metrics["shape"] + 0.25 * metrics["tex"]
        if best is None or score > best[0]:
            best = (score, lam, W.astype(np.float32), metrics)

    _score, lam, W, metrics = best
    out = Path(args.model)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        schema="photofit_v1",
        W=W,
        x_mean=x_mean,
        x_scale=x_scale,
        lam=float(lam),
        n_sym=N_SYM,
        n_asym=N_ASYM,
        n_tex=N_TEX,
        feature_dim=photofit.feature_dim(),
        feature_lms=np.asarray(photofit.FEATURE_LMS, np.int32),
        feature_names=np.asarray(photofit.feature_names()),
        render_size=int(d["render_size"]) if "render_size" in d else DEFAULT_SIZE,
        holdout=int(args.holdout),
        seed=int(args.seed),
        r2_all=metrics["all"],
        r2_shape=metrics["shape"],
        r2_sym=metrics["sym"],
        r2_asym=metrics["asym"],
        r2_tex=metrics["tex"],
    )
    print(f"wrote {out} (lam={lam}, feature_dim={photofit.feature_dim()})")


def _summarize(vals: list[float]) -> dict[str, float | int]:
    arr = np.asarray(vals, np.float64)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return {"mean": float("nan"), "p10": float("nan"),
                "p90": float("nan"), "n": 0}
    return {
        "mean": float(np.mean(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "n": int(len(arr)),
    }


def _embed_render(img: np.ndarray, id_model: str | None):
    from ootp_faceforge.landmarks import detect

    lms = detect(img)
    return identity.embed(img, lms, id_model)


def stage_eval(args: argparse.Namespace) -> None:
    if not identity.available(args.id_model):
        raise SystemExit("ArcFace model not found; run scripts/download_restore_model.py")

    d = np.load(args.data, allow_pickle=True)
    model = photofit.PhotofitModel(args.model)
    Xtr, Ytr, Xte, Yte, te_idx = _split(d["X"], d["Y"], args.holdout, args.seed)
    del Xtr, Ytr
    names = d["names"][te_idx] if "names" in d else np.asarray([str(i) for i in te_idx])
    out = Path(args.out_dir or Path(args.model).parent)
    out.mkdir(parents=True, exist_ok=True)
    render_size = int(args.render_size or (
        int(d["render_size"]) if "render_size" in d else DEFAULT_SIZE))

    rows: dict[str, list[float]] = {
        "emb_photofit": [],
        "emb_mean": [],
        "emb_emb2shape": [],
    }
    coeff_preds = []
    emb_model = None
    if args.emb2shape_model != "off" and emb2shape.available(args.emb2shape_model):
        emb_model = emb2shape.load(args.emb2shape_model)

    n = min(int(args.n), len(Xte))
    for k in range(n):
        x = Xte[k]
        y_true = Yte[k]
        pred = model.predict_feature(x)
        y_pred = pred.raw
        coeff_preds.append(y_pred)

        try:
            img_true = _render_y(y_true, render_size)
            img_pred = _render_y(y_pred, render_size)
            img_mean = _render_y(np.zeros_like(y_true), render_size)
            e_true = _embed_render(img_true, args.id_model)
            e_pred = _embed_render(img_pred, args.id_model)
            e_mean = _embed_render(img_mean, args.id_model)
            if e_true is not None and e_pred is not None:
                rows["emb_photofit"].append(float(e_true @ e_pred))
            if e_true is not None and e_mean is not None:
                rows["emb_mean"].append(float(e_true @ e_mean))

            panels = [img_true, img_pred]
            if emb_model is not None:
                e = x[:512]
                ep = emb_model.predict(e)
                y_emb = np.concatenate([ep.sym_shape, ep.asym_shape, ep.sym_tex])
                img_emb = _render_y(y_emb, render_size)
                e_emb = _embed_render(img_emb, args.id_model)
                if e_true is not None and e_emb is not None:
                    rows["emb_emb2shape"].append(float(e_true @ e_emb))
                panels.append(img_emb)
            panels.append(img_mean)
            if k < args.save_images:
                Image.fromarray(np.concatenate(panels, 1)).save(
                    out / f"photofit_eval_{names[k]}.png")
        except Exception as exc:
            print(f"eval skip {names[k]}: {type(exc).__name__}: {exc}",
                  file=sys.stderr)

    if coeff_preds:
        P = np.stack(coeff_preds)
        T = Yte[:len(coeff_preds)]
        r2 = _r2(P - T, T)
        coeff_metrics = {
            "all": float(np.mean(r2)),
            "shape": float(np.mean(r2[:N_SYM + N_ASYM])),
            "sym": float(np.mean(r2[:N_SYM])),
            "asym": float(np.mean(r2[N_SYM:N_SYM + N_ASYM])),
            "tex": float(np.mean(r2[N_SYM + N_ASYM:])),
        }
    else:
        coeff_metrics = {}

    metrics = {key: _summarize(vals) for key, vals in rows.items()}
    for key, vals in metrics.items():
        print(f"{key}: mean={vals['mean']:.3f} p10={vals['p10']:.3f} "
              f"p90={vals['p90']:.3f} (n={vals['n']})")
    if coeff_metrics:
        print("coeff R2:", " ".join(
            f"{k}={v:.3f}" for k, v in coeff_metrics.items()))

    report = {
        "params": {k: v for k, v in vars(args).items() if k != "func"},
        "model": str(args.model),
        "data": str(args.data),
        "metrics": metrics,
        "coeff_r2": coeff_metrics,
    }
    report_path = out / "photofit_eval_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {report_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)

    g = sub.add_parser("gen")
    g.add_argument("--out", required=True)
    g.add_argument("--fg-dir", default=str(DEFAULT_FG_DIR))
    g.add_argument("--id-model")
    g.add_argument("--render-size", type=int, default=DEFAULT_SIZE)
    g.add_argument("--limit", type=int, default=0)
    g.add_argument("--augs", type=int, default=4,
                   help="Samples per .fg, including clean sample 0.")
    g.add_argument("--aug-strength", type=float, default=1.0)
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(func=stage_gen)

    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--model", required=True)
    t.add_argument("--holdout", type=int, default=300)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--lambdas", default="0.03,0.1,0.3,1,3,10,30,100")
    t.set_defaults(func=stage_train)

    e = sub.add_parser("eval")
    e.add_argument("--data", required=True)
    e.add_argument("--model", required=True)
    e.add_argument("--out-dir")
    e.add_argument("--n", type=int, default=120)
    e.add_argument("--save-images", type=int, default=12)
    e.add_argument("--holdout", type=int, default=300)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--id-model")
    e.add_argument("--render-size", type=int)
    e.add_argument("--emb2shape-model", default=None,
                   help="Optional baseline .npz, or 'off'.")
    e.set_defaults(func=stage_eval)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
