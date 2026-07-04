"""Fit FaceGen shape coefficients + weak-perspective camera to 2D landmarks."""
from __future__ import annotations

import numpy as np

from .basis import Basis
from .landmarks import CORR, WEIGHT


def _pose(X: np.ndarray, u: np.ndarray, w: np.ndarray):
    """Weighted scaled-orthographic pose: u ~ s*R2 @ X + t.
    Returns s (float), R2 (2,3 orthonormal rows), t (2,)."""
    wm = w / w.sum()
    Xm = (wm[:, None] * X).sum(0)
    um = (wm[:, None] * u).sum(0)
    Xc, uc = X - Xm, u - um
    # affine M (2,3): weighted least squares
    A = Xc * np.sqrt(w)[:, None]
    Bv = uc * np.sqrt(w)[:, None]
    M, *_ = np.linalg.lstsq(A, Bv, rcond=None)
    M = M.T                                   # (2,3)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    R2 = U @ Vt                               # orthonormal rows
    s = float(S.mean())
    t = um - s * R2 @ Xm
    return s, R2, t


def fit_shape(basis: Basis, lms_px: np.ndarray,
              lam_sym: float = 3.0, lam_asym: float = 40.0,
              n_iter: int = 8):
    """Anchor-only fit against the hand-named CORR points. Kept as a
    standalone helper; the production path uses fit_shape_multi_dense."""
    sp = basis.surface_points()
    names = [n for n in CORR if n in sp]
    base = np.stack([sp[n][0] for n in names])            # (L,3)
    Bm = np.stack([sp[n][1] for n in names])              # (L,3,80)
    u = np.stack([lms_px[CORR[n]] for n in names])        # (L,2)
    w = np.array([WEIGHT[n] for n in names])
    L = len(names)

    nm = basis.n_sym + basis.n_asym
    lam = np.concatenate([np.full(basis.n_sym, lam_sym),
                          np.full(basis.n_asym, lam_asym)])
    c = np.zeros(nm)
    for _ in range(n_iter):
        X = base + Bm @ c                                 # (L,3)
        s, R2, t = _pose(X, u, w)
        # residual eqs in mesh units: (u - t)/s - R2 base = R2 B c
        A = np.einsum("ij,ljm->lim", R2, Bm).reshape(2 * L, nm)
        rhs = ((u - t) / s @ np.eye(2)).reshape(L, 2) - base @ R2.T
        rhs = rhs.reshape(2 * L)
        ww = np.repeat(w, 2)
        AtA = A.T @ (ww[:, None] * A) + np.diag(lam)
        Atb = A.T @ (ww * rhs)
        c = np.linalg.solve(AtA, Atb)

    X = base + Bm @ c
    s, R2, t = _pose(X, u, w)
    r3 = np.cross(R2[0], R2[1])
    R = np.vstack([R2, r3])
    resid = np.sqrt((((s * X @ R2.T + t) - u) ** 2).sum(1))
    return c, s, R, t, dict(zip(names, resid))


def project(basis: Basis, coeffs: np.ndarray, s: float, R: np.ndarray,
            t: np.ndarray):
    """All verts -> (V,2) pixel coords and (V,) camera-space depth."""
    V = basis.shaped_verts(coeffs)
    cam = V @ R.T                                          # (V,3)
    return s * cam[:, :2] + t, cam


# jawline: visible contour that defines face slimness -> high weight
JAW_LMS = {397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172,
           58, 132, 288, 361, 323, 93, 234}
# upper face oval: usually under a cap, mediapipe guesses it -> near-zero
FOREHEAD_LMS = {10, 338, 297, 332, 284, 251, 389, 356, 454, 21, 54, 103, 67,
                109, 162, 127}


def _pair_geometry(basis: Basis, pairs, dense_weight: float):
    """Turn calibrated (lm_idx, tri, bary) correspondence triples into fit
    observations.

    Returns (base, Bm, w, lm_ids):
      base    (L,3)      mean-face 3D point at each landmark's bary location
      Bm      (L,3,nm)   per-mode Jacobian of that point
      w       (L,)       per-landmark fit weight
      lm_ids  (L,)       source MediaPipe landmark index of each observation

    The calibrated table (calibrate.calibrated_pairs) already cancels
    MediaPipe's systematic offset against the FaceGen mean face, so no
    separate hand-named anchor set is needed."""
    base_list, B_list, w_list, id_list = [], [], [], []
    for li, ti, bary in pairs:
        li, ti = int(li), int(ti)
        vidx = basis.tris[ti]
        w3 = np.asarray(bary, np.float32)
        base_list.append(w3 @ basis.verts[vidx])
        B_list.append(np.einsum("k,mkd->dm", w3, basis.modes[:, vidx, :]))
        if li in FOREHEAD_LMS:
            w_list.append(0.05)
        elif li in JAW_LMS:
            w_list.append(0.9)
        else:
            w_list.append(dense_weight)
        id_list.append(li)
    return (np.stack(base_list), np.stack(B_list),
            np.asarray(w_list, np.float64), np.asarray(id_list, np.int32))


def fit_shape_multi_dense(basis: Basis, lms_list: list[np.ndarray],
                          lam_sym: float = 0.005, lam_asym: float = 0.3,
                          dense_weight: float = 0.45,
                          photo_weights: list[float] | None = None,
                          n_iter: int = 8):
    """Fit one shared FaceGen shape to landmarks from several photos.

    Every photo shares the FaceGen shape coefficients but gets its own
    weak-perspective pose, so a clean front shot, a cap shot, and a mild side
    shot each contribute their own geometry evidence. Correspondence comes
    entirely from the calibrated dense table, which removes the systematic
    mean-face bias that made every build converge to the same head.
    """
    from .calibrate import calibrated_pairs

    if not lms_list:
        raise ValueError("fit_shape_multi_dense requires at least one photo")
    if photo_weights is None:
        photo_weights = [1.0] * len(lms_list)
    if len(photo_weights) != len(lms_list):
        raise ValueError("photo_weights length must match lms_list")

    base, Bm, w, lm_ids = _pair_geometry(basis, calibrated_pairs(basis),
                                         dense_weight)
    us = [np.asarray(lms)[lm_ids] for lms in lms_list]
    w_photo = np.asarray(photo_weights, np.float64)
    w_photo = w_photo / max(float(w_photo.mean()), 1e-6)

    nm = basis.n_sym + basis.n_asym
    # Scale the ridge by the total observation weight so a fixed lambda has a
    # stable effect regardless of table size. This keeps the near-frontal
    # asymmetric modes (weakly constrained by landmarks) from leaking spurious
    # asymmetry, at the cost of shrinking magnitude -- the identity refine and
    # the emb2shape prior restore identity amplitude downstream.
    lam = np.concatenate([np.full(basis.n_sym, lam_sym),
                          np.full(basis.n_asym, lam_asym)]) * float(w.sum())

    c = np.zeros(nm)
    for _ in range(n_iter):
        X = base + Bm @ c
        AtA = np.diag(lam).astype(np.float64)
        Atb = np.zeros(nm, np.float64)
        for pw, u in zip(w_photo, us):
            s, R2, t = _pose(X, u, w)
            A = np.einsum("ij,ljm->lim", R2, Bm).reshape(2 * len(w), nm)
            rhs = (((u - t) / s) - base @ R2.T).reshape(2 * len(w))
            wwp = np.repeat(w * float(pw), 2)
            AtA += A.T @ (wwp[:, None] * A)
            Atb += A.T @ (wwp * rhs)
        c = np.linalg.solve(AtA, Atb)

    X = base + Bm @ c
    per_photo, poses_out = [], []
    total_resid = total_w = 0.0
    for pw, u in zip(w_photo, us):
        s, R2, t = _pose(X, u, w)
        R = np.vstack([R2, np.cross(R2[0], R2[1])])
        resid_px = np.sqrt((((s * X @ R2.T + t) - u) ** 2).sum(1))
        mean_resid = float((resid_px * w).sum() / w.sum())
        per_photo.append(mean_resid)
        total_resid += mean_resid * float(pw) * float(w.sum())
        total_w += float(pw) * float(w.sum())
        poses_out.append((s, R, t))

    return c, poses_out, {
        "n_photos": len(lms_list),
        "n_dense": int(len(w)),
        "mean_resid": float(total_resid / max(total_w, 1e-6)),
        "per_photo_resid": per_photo,
        "photo_weights": [float(x) for x in w_photo],
    }
