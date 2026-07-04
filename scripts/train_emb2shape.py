"""Learn a direct ArcFace-embedding -> FaceGen-coefficient regressor.

The official OOTP 27 .fg library provides thousands of real identities in
coefficient space, and the vectorized renderer makes rendering them cheap.
That yields unlimited (embedding, coefficients) supervision for inverting
the render+recognition pipeline: at build time the player's photo embedding
can then be decoded straight into FaceGen coefficients, capturing identity
cues that 2D landmarks cannot see.

Stages:
  gen    render every official .fg, detect + embed, save dataset
  train  ridge-regress coefficients from embeddings (with held-out split)
  eval   held-out: re-render predictions, compare embedding similarity and
         landmark geometry against the true renders

Usage:
  python scripts/train_emb2shape.py gen   --out models/emb2shape_data.npz
  python scripts/train_emb2shape.py train --data models/emb2shape_data.npz --model models/emb2shape.npz
  python scripts/train_emb2shape.py eval  --data models/emb2shape_data.npz --model models/emb2shape.npz --out-dir models
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ootp_faceforge import identity  # noqa: E402
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


def stage_gen(args) -> None:
    from ootp_faceforge.landmarks import detect
    from ootp_faceforge.render import _build_ootp_assets, render_fast

    if not identity.available(args.id_model):
        raise SystemExit("ArcFace model not found; run scripts/download_restore_model.py")
    fg_dir = Path(args.fg_dir)
    paths = sorted(fg_dir.glob("*.fg"))
    if not paths:
        raise SystemExit(f"no .fg files found: {fg_dir}")
    if args.limit:
        paths = paths[: args.limit]
    X, Y, names = [], [], []
    for i, p in enumerate(paths):
        try:
            fg = FgFile.read(str(p))
            if (len(fg.sym_shape) != N_SYM or len(fg.asym_shape) != N_ASYM
                    or len(fg.sym_tex) != N_TEX):
                continue
            assets = _build_ootp_assets(
                fg,
                include_eyes=True,
                include_cap=not args.no_cap,
                include_body=not args.no_body,
                include_mouth=not args.no_mouth,
            )
            img = render_fast(assets, fg, args.render_size, shade=True)
            lms = detect(img)
            e = identity.embed(img, lms, args.id_model)
            if e is None:
                continue
        except Exception:
            continue
        X.append(e.astype(np.float32))
        Y.append(np.concatenate([fg.sym_shape, fg.asym_shape,
                                 fg.sym_tex]).astype(np.float32))
        names.append(p.stem)
        if (i + 1) % 250 == 0:
            print(f"{i + 1}/{len(paths)} ok={len(X)}", flush=True)
    if not X:
        raise SystemExit("no usable embeddings generated")
    np.savez(args.out, X=np.stack(X), Y=np.stack(Y),
             names=np.array(names), fg_dir=str(fg_dir),
             render_size=int(args.render_size))
    print(f"wrote {args.out}: {len(X)} samples")


def _split(X, Y, holdout=300, seed=0):
    if len(X) <= holdout:
        raise ValueError(f"holdout {holdout} requires more than {len(X)} samples")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    te, tr = idx[:holdout], idx[holdout:]
    return X[tr], Y[tr], X[te], Y[te], te


def stage_train(args) -> None:
    d = np.load(args.data, allow_pickle=True)
    Xtr, Ytr, Xte, Yte, _ = _split(d["X"], d["Y"], args.holdout, args.seed)
    # embeddings are unit-norm; add bias feature
    A = np.hstack([Xtr, np.ones((len(Xtr), 1), np.float32)])
    At = np.hstack([Xte, np.ones((len(Xte), 1), np.float32)])
    reg = np.eye(A.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0
    lambdas = _parse_lambdas(args.lambdas)
    best = None
    for lam in lambdas:
        W = np.linalg.solve(A.T @ A + lam * reg, A.T @ Ytr)
        P = At @ W
        err = P - Yte
        denom = np.maximum(((Yte - Yte.mean(0)) ** 2).sum(0), 1e-9)
        r2 = 1 - (err ** 2).sum(0) / denom
        sym = float(np.mean(r2[:N_SYM]))
        asym = float(np.mean(r2[N_SYM:N_SYM + N_ASYM]))
        tex = float(np.mean(r2[N_SYM + N_ASYM:]))
        shape = float(np.mean(r2[:N_SYM + N_ASYM]))
        all_score = float(np.mean(r2))
        score = all_score
        print(f"lam={lam}: mean R2 all={all_score:.3f} "
              f"shape={shape:.3f} sym={sym:.3f} asym={asym:.3f} tex={tex:.3f}")
        if best is None or score > best[0]:
            best = (score, lam, W, dict(all=all_score, shape=shape,
                                        sym=sym, asym=asym, tex=tex))
    _, lam, W, metrics = best
    np.savez(args.model, W=W.astype(np.float32), lam=lam,
             n_sym=N_SYM, n_asym=N_ASYM, n_tex=N_TEX,
             size=int(d["render_size"]) if "render_size" in d else DEFAULT_SIZE,
             holdout=args.holdout, seed=args.seed,
             r2_all=metrics["all"], r2_shape=metrics["shape"],
             r2_sym=metrics["sym"], r2_asym=metrics["asym"],
             r2_tex=metrics["tex"])
    print(f"wrote {args.model} (lam={lam}, holdout={args.holdout})")


def stage_eval(args) -> None:
    from ootp_faceforge.basis import get_basis
    from ootp_faceforge.calibrate import calibrated_pairs
    from ootp_faceforge.fit import _pair_geometry, fit_shape_multi_dense
    from ootp_faceforge.landmarks import detect
    from ootp_faceforge.render import _build_ootp_assets, render_fast

    if not identity.available(args.id_model):
        raise SystemExit("ArcFace model not found; run scripts/download_restore_model.py")
    d = np.load(args.data, allow_pickle=True)
    m = np.load(args.model)
    W = m["W"]
    render_size = int(args.render_size or (int(m["size"]) if "size" in m else DEFAULT_SIZE))
    _, _, Xte, Yte, te_idx = _split(d["X"], d["Y"], args.holdout, args.seed)
    names = d["names"][te_idx]

    basis = get_basis()
    pairs = calibrated_pairs(basis)
    _, _, w_lm, lm_ids = _pair_geometry(basis, pairs, 0.45)
    sel = w_lm > 0.2

    def procrustes_rms(u, v, ww):
        wm = (ww / ww.sum())[:, None]
        mu, mv = (wm * u).sum(0), (wm * v).sum(0)
        uc, vc = u - mu, v - mv
        C = (wm * vc).T @ uc
        U, S, Vt = np.linalg.svd(C)
        s = S.sum() / (wm * uc ** 2).sum()
        return float(np.sqrt((wm * (vc - s * uc @ (U @ Vt).T) ** 2).sum()))

    def render_of(y):
        fg = FgFile(sym_shape=y[:N_SYM], asym_shape=y[N_SYM:N_SYM + N_ASYM],
                    sym_tex=y[N_SYM + N_ASYM:], asym_tex=np.zeros(0),
                    detail_jpeg=None)
        assets = _build_ootp_assets(
            fg,
            include_eyes=True,
            include_cap=not args.no_cap,
            include_body=not args.no_body,
            include_mouth=not args.no_mouth,
        )
        return render_fast(assets, fg, render_size, shade=True)

    n = min(args.n, len(Xte))
    out = Path(args.out_dir or Path(args.model).parent)
    out.mkdir(parents=True, exist_ok=True)
    rows = {
        "emb_pred": [],
        "emb_lmfit": [],
        "emb_mean": [],
        "rms_pred": [],
        "rms_lmfit": [],
        "rms_mean": [],
    }
    for k in range(n):
        e, y_true = Xte[k], Yte[k]
        y_pred = np.hstack([e, 1.0]).astype(np.float32) @ W
        img_t = render_of(y_true)
        try:
            lms_t = detect(img_t)
        except Exception:
            continue
        img_p = render_of(y_pred)
        try:
            lms_p = detect(img_p)
        except Exception:
            continue
        e_t = identity.embed(img_t, lms_t, args.id_model)
        e_p = identity.embed(img_p, lms_p, args.id_model)
        if e_t is None or e_p is None:
            continue
        y_mean = np.zeros_like(y_true)
        img_m = render_of(y_mean)
        lms_m = detect(img_m)
        e_m = identity.embed(img_m, lms_m, args.id_model)
        rows["emb_pred"].append(float(e_t @ e_p))
        if e_m is not None:
            rows["emb_mean"].append(float(e_t @ e_m))
        rows["rms_pred"].append(procrustes_rms(lms_p[lm_ids][sel],
                                               lms_t[lm_ids][sel],
                                               w_lm[sel]))
        rows["rms_mean"].append(procrustes_rms(lms_m[lm_ids][sel],
                                               lms_t[lm_ids][sel],
                                               w_lm[sel]))
        # landmark-fit baseline on the same render
        c_fit, _, _ = fit_shape_multi_dense(
            basis, [lms_t],
            lam_sym=args.shape_lam,
            lam_asym=args.asym_lam,
            dense_weight=args.dense_weight,
        )
        y_fit = y_true.copy()
        y_fit[: N_SYM + N_ASYM] = np.clip(c_fit * args.shape_gain, -4, 4)
        img_f = render_of(y_fit)
        try:
            lms_f = detect(img_f)
            e_f = identity.embed(img_f, lms_f, args.id_model)
            if e_f is not None:
                rows["emb_lmfit"].append(float(e_t @ e_f))
            rows["rms_lmfit"].append(procrustes_rms(lms_f[lm_ids][sel],
                                                    lms_t[lm_ids][sel],
                                                    w_lm[sel]))
        except Exception:
            rows["rms_lmfit"].append(float("nan"))
        if k < args.save_images:
            from PIL import Image
            Image.fromarray(np.concatenate([img_t, img_p, img_f], 1)).save(
                out / f"emb2shape_eval_{names[k]}.png")

    def summarize(vals):
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

    metrics = {key: summarize(vals) for key, vals in rows.items()}
    for key, vals in rows.items():
        s = metrics[key]
        print(f"{key}: mean={s['mean']:.3f} p10={s['p10']:.3f} "
              f"p90={s['p90']:.3f} (n={s['n']})")

    emb_delta = metrics["emb_pred"]["mean"] - metrics["emb_lmfit"]["mean"]
    rms_delta = metrics["rms_lmfit"]["mean"] - metrics["rms_pred"]["mean"]
    if emb_delta > 0 and rms_delta > 0:
        winner = "emb2shape"
    elif emb_delta < 0 and rms_delta < 0:
        winner = "landmark_fit"
    else:
        winner = "mixed"
    params = {k: v for k, v in vars(args).items() if k != "func"}
    report = {
        "params": params,
        "model": str(args.model),
        "data": str(args.data),
        "metrics": metrics,
        "decision": {
            "winner": winner,
            "emb_delta_pred_minus_lmfit": float(emb_delta),
            "rms_delta_lmfit_minus_pred": float(rms_delta),
            "positive_deltas_mean_emb2shape_is_better": True,
        },
    }
    (out / "emb2shape_eval_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print("winner:", winner)
    print("wrote", out / "emb2shape_eval_report.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--out", required=True)
    g.add_argument("--fg-dir", default=str(DEFAULT_FG_DIR))
    g.add_argument("--id-model")
    g.add_argument("--render-size", type=int, default=DEFAULT_SIZE)
    g.add_argument("--limit", type=int, default=0)
    g.add_argument("--no-cap", action="store_true")
    g.add_argument("--no-body", action="store_true")
    g.add_argument("--no-mouth", action="store_true")
    g.set_defaults(func=stage_gen)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--model", required=True)
    t.add_argument("--holdout", type=int, default=300)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--lambdas", default="0.03,0.1,0.3,1,3,10,30")
    t.set_defaults(func=stage_train)
    e = sub.add_parser("eval")
    e.add_argument("--data", required=True)
    e.add_argument("--model", required=True)
    e.add_argument("--out-dir")
    e.add_argument("--n", type=int, default=60)
    e.add_argument("--save-images", type=int, default=6)
    e.add_argument("--holdout", type=int, default=300)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--id-model")
    e.add_argument("--render-size", type=int)
    e.add_argument("--shape-lam", type=float, default=0.005)
    e.add_argument("--asym-lam", type=float, default=0.3)
    e.add_argument("--dense-weight", type=float, default=0.45)
    e.add_argument("--shape-gain", type=float, default=1.3)
    e.add_argument("--no-cap", action="store_true")
    e.add_argument("--no-body", action="store_true")
    e.add_argument("--no-mouth", action="store_true")
    e.set_defaults(func=stage_eval)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
