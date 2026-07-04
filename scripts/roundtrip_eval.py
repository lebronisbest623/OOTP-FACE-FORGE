"""Round-trip evaluation of the shape-fitting pipeline.

Renders faces with KNOWN OOTP FaceGen coefficients, runs landmark detection +
shape fitting on those renders, and measures how
much of the true identity the fit recovers.

Cases:
  mean  - fit the rendered mean face (all coefficients zero). Any nonzero
          result is pure systematic bias shared by every build.
  synth - N random synthetic identities. Reports per-face recovery
          correlation, identity collapse (pairwise similarity of fits vs
          pairwise similarity of truths), and asym leakage.

Usage:
  python scripts/roundtrip_eval.py --out-dir <dir> [--n-faces 6] [--size 640]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ootp_faceforge.basis import get_basis  # noqa: E402
from ootp_faceforge.fgformat import FgFile  # noqa: E402
from ootp_faceforge.fit import fit_shape_multi_dense  # noqa: E402
from ootp_faceforge.landmarks import detect  # noqa: E402
from ootp_faceforge.render import render  # noqa: E402


def render_coeffs(basis, c: np.ndarray, size: int) -> np.ndarray:
    fg = FgFile(
        sym_shape=c[: basis.n_sym],
        asym_shape=c[basis.n_sym :],
        sym_tex=np.zeros(basis.egt.sym.shape[0]),
        asym_tex=np.zeros(0),
        detail_jpeg=None,
    )
    img, _ = render(fg, size=size, shade=True, aa=2)
    return img


def fit_render(basis, img: np.ndarray, lam_sym: float, lam_asym: float,
               dense_weight: float) -> np.ndarray:
    lms = detect(img)
    c, _poses, _info = fit_shape_multi_dense(
        basis, [lms], lam_sym=lam_sym, lam_asym=lam_asym,
        dense_weight=dense_weight)
    return c


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def cos(a: np.ndarray, b: np.ndarray) -> float:
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 1e-9 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-faces", type=int, default=6)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--sigma", type=float, default=0.9,
                    help="Std of true sym coefficients.")
    ap.add_argument("--lam-sym", type=float, default=0.005)
    ap.add_argument("--lam-asym", type=float, default=0.3)
    ap.add_argument("--dense-weight", type=float, default=0.45)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    basis = get_basis()
    nm = basis.n_sym + basis.n_asym
    report: dict = {"params": vars(args)}

    # ---- case: mean face ------------------------------------------------
    img = render_coeffs(basis, np.zeros(nm), args.size)
    Image.fromarray(img).save(out / "mean_true.png")
    c_bias = fit_render(basis, img, args.lam_sym, args.lam_asym,
                        args.dense_weight)
    Image.fromarray(render_coeffs(basis, c_bias, args.size)).save(
        out / "mean_refit.png")
    top = np.argsort(-np.abs(c_bias))[:8]
    report["mean_face"] = {
        "bias_norm": float(np.linalg.norm(c_bias)),
        "bias_norm_sym": float(np.linalg.norm(c_bias[: basis.n_sym])),
        "bias_norm_asym": float(np.linalg.norm(c_bias[basis.n_sym :])),
        "top_modes": [(int(i), round(float(c_bias[i]), 3)) for i in top],
    }
    print("mean-face bias |c| =", round(report["mean_face"]["bias_norm"], 3),
          "(sym", round(report["mean_face"]["bias_norm_sym"], 3),
          "/ asym", round(report["mean_face"]["bias_norm_asym"], 3), ")")

    # ---- case: synthetic identities -------------------------------------
    rng = np.random.default_rng(7)
    trues, fits = [], []
    per_face = []
    for k in range(args.n_faces):
        c_true = np.zeros(nm)
        c_true[: basis.n_sym] = np.clip(
            rng.normal(0, args.sigma, basis.n_sym), -2.5, 2.5)
        img = render_coeffs(basis, c_true, args.size)
        Image.fromarray(img).save(out / f"synth{k}_true.png")
        try:
            c_fit = fit_render(basis, img, args.lam_sym, args.lam_asym,
                               args.dense_weight)
        except Exception as exc:  # detection failure etc.
            print(f"synth{k}: FIT FAILED: {exc}")
            per_face.append({"k": k, "error": str(exc)})
            continue
        Image.fromarray(render_coeffs(basis, c_fit, args.size)).save(
            out / f"synth{k}_refit.png")
        c_adj = c_fit - c_bias
        row = {
            "k": k,
            "true_norm": float(np.linalg.norm(c_true)),
            "fit_norm": float(np.linalg.norm(c_fit)),
            "corr_sym": corr(c_fit[: basis.n_sym], c_true[: basis.n_sym]),
            "corr_sym_debiased": corr(c_adj[: basis.n_sym],
                                      c_true[: basis.n_sym]),
            "asym_leak_norm": float(np.linalg.norm(c_fit[basis.n_sym :])),
        }
        per_face.append(row)
        trues.append(c_true)
        fits.append(c_fit)
        print(f"synth{k}: corr={row['corr_sym']:.3f} "
              f"debiased={row['corr_sym_debiased']:.3f} "
              f"|fit|={row['fit_norm']:.2f} |true|={row['true_norm']:.2f} "
              f"asym_leak={row['asym_leak_norm']:.2f}")

    if len(fits) >= 2:
        def pairwise(vs):
            sims = [cos(vs[i], vs[j])
                    for i in range(len(vs)) for j in range(i + 1, len(vs))]
            return float(np.mean(sims))

        collapse = {
            "pairwise_cos_true": pairwise([t[: basis.n_sym] for t in trues]),
            "pairwise_cos_fit": pairwise([f[: basis.n_sym] for f in fits]),
            "pairwise_cos_fit_debiased": pairwise(
                [(f - c_bias)[: basis.n_sym] for f in fits]),
        }
        report["collapse"] = collapse
        print("pairwise cos: true", round(collapse["pairwise_cos_true"], 3),
              "| fit", round(collapse["pairwise_cos_fit"], 3),
              "| fit-debiased",
              round(collapse["pairwise_cos_fit_debiased"], 3))

    report["per_face"] = per_face
    np.savez(out / "coeffs.npz",
             c_bias=c_bias,
             trues=np.stack(trues) if trues else np.zeros((0, nm)),
             fits=np.stack(fits) if fits else np.zeros((0, nm)))
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print("wrote", out / "report.json")


if __name__ == "__main__":
    main()
