"""Calibrated MediaPipe -> FaceGen dense correspondence.

The hand-made name-level correspondence (landmarks.CORR) carries a systematic
placement error: where MediaPipe puts a landmark and where the equally-named
FaceGen surface point sits differ by a consistent offset for every human face.
Under weak regularization that shared offset is absorbed into the shape
coefficients, so every build converges to the same distorted head.

The fix: render the FaceGen MEAN face with a known orthographic projection,
run MediaPipe on that render, and record which mesh point (triangle +
barycentric coords) each landmark lands on. Fitting a photo against this
table makes the mean face fit to exactly zero coefficients by construction;
MediaPipe's systematic bias cancels to first order and only the subject's
deviation from the mean face drives the fit.

The table depends only on the basis mesh and MediaPipe model, so it is built
once and cached next to the package.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .basis import Basis, get_basis
from .fgformat import FgFile
from .paths import workspace_root

WORKSPACE_ROOT = workspace_root()
CACHE_ROOT = WORKSPACE_ROOT / "cache"
DENSE_CORR = CACHE_ROOT / "dense_corr_ootp.npz"
CORR_VERSION = 3  # bump to invalidate cached tables built by older code


def dense_corr_path(basis: Basis) -> Path:
    return CACHE_ROOT / f"dense_corr_{basis.source}_v{CORR_VERSION}.npz"


def mean_face_screen(basis: Basis, size: int, aa: int = 2) -> np.ndarray:
    """Per-vertex 2D screen coords of the mean face exactly as render() draws
    it at this size (before the anti-alias downscale)."""
    from .render import _project_with_params, _screen_params

    canvas = size * aa
    lo, scale = _screen_params(basis.verts, canvas, basis.orientation)
    return (
        _project_with_params(basis.verts, canvas, basis.orientation, lo, scale)
        / aa
    )


def match_landmarks(proj2d: np.ndarray, tris: np.ndarray, fronts: np.ndarray,
                    lms_px: np.ndarray, tol: float = -0.02):
    """Map each landmark pixel to (tri, bary) on the projected mesh.

    Landmarks outside every front-facing triangle (e.g. jaw contour points
    just past the mesh silhouette) snap to the nearest front vertex, which is
    the correct correspondence for silhouette points. Returns
    (lm_idx, tri_idx, bary) triples."""
    tp = proj2d[tris]                                     # (T,3,2)
    cand_idx = np.nonzero(fronts)[0]
    lo = tp[cand_idx].min(1)
    hi = tp[cand_idx].max(1)
    front_verts = np.unique(tris[cand_idx])
    vert_tri = {}                                         # vertex -> a front tri
    for ti in cand_idx:
        for k, v in enumerate(tris[ti]):
            vert_tri.setdefault(int(v), (int(ti), k))
    out = []
    for li in range(len(lms_px)):
        p = lms_px[li]
        near = cand_idx[(lo[:, 0] <= p[0]) & (hi[:, 0] >= p[0]) &
                        (lo[:, 1] <= p[1]) & (hi[:, 1] >= p[1])]
        hit = None
        for ti in near:
            a, b_, c_ = tp[ti]
            M = np.array([[b_[0] - a[0], c_[0] - a[0]],
                          [b_[1] - a[1], c_[1] - a[1]]])
            det = np.linalg.det(M)
            if abs(det) < 1e-9:
                continue
            w = np.linalg.solve(M, p - a)
            bary = np.array([1 - w.sum(), w[0], w[1]])
            if (bary >= tol).all():
                hit = (li, int(ti), np.clip(bary, 0, 1))
                break
        if hit is None:
            d = np.linalg.norm(proj2d[front_verts] - p, axis=1)
            v = int(front_verts[np.argmin(d)])
            ti, k = vert_tri[v]
            bary = np.zeros(3)
            bary[k] = 1.0
            hit = (li, ti, bary)
        out.append(hit)
    return out


def build_dense_corr(basis: Basis | None = None, size: int = 1024,
                     out_path: Path | None = None) -> dict:
    """Render the mean face, detect landmarks, build and save the table."""
    from .landmarks import detect
    from .render import render
    from .texture import front_tris

    basis = basis or get_basis()
    out_path = out_path or dense_corr_path(basis)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fg = FgFile(
        sym_shape=np.zeros(basis.n_sym),
        asym_shape=np.zeros(basis.n_asym),
        sym_tex=np.zeros(basis.egt.sym.shape[0]),
        asym_tex=np.zeros(0),
        detail_jpeg=None,
    )
    img, _ = render(fg, size=size, shade=True, aa=2)
    lms = detect(img)
    scr = mean_face_screen(basis, size, aa=2)
    anchor = basis.front_anchor_tri
    fronts = front_tris(basis, scr, anchor)
    pairs = match_landmarks(scr, basis.tris, fronts, lms)

    lm = np.array([p[0] for p in pairs], np.int32)
    tri = np.array([p[1] for p in pairs], np.int32)
    bary = np.stack([p[2] for p in pairs]).astype(np.float32)
    # residual of the calibration itself: projected table point vs detected
    # landmark on the calibration render (should be ~0 by construction)
    table_px = np.einsum("lk,lkd->ld", bary, scr[basis.tris[tri]])
    resid = np.linalg.norm(table_px - lms[lm], axis=1)
    np.savez(out_path, lm=lm, tri=tri, bary=bary,
             version=np.int32(CORR_VERSION), size=np.int32(size),
             source=np.asarray(basis.source),
             calib_resid=resid.astype(np.float32))
    return {"n": len(lm), "max_resid_px": float(resid.max()),
            "mean_resid_px": float(resid.mean()), "path": str(out_path)}


def calibrated_pairs(basis: Basis | None = None):
    """Load (or build on first use) the calibrated correspondence table.
    Returns (lm_idx, tri_idx, bary) triples."""
    basis = basis or get_basis()
    path = dense_corr_path(basis)
    if path.exists():
        d = np.load(path)
        if (
            int(d.get("version", np.int32(0))) == CORR_VERSION
            and str(d.get("source", "")) == basis.source
        ):
            return list(zip(d["lm"], d["tri"], d["bary"]))
    info = build_dense_corr(basis)
    d = np.load(info["path"])
    return list(zip(d["lm"], d["tri"], d["bary"]))


if __name__ == "__main__":
    print(build_dense_corr())
