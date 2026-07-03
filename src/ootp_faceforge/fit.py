"""Fit FaceGen shape coefficients + weak-perspective camera to 2D landmarks."""
from __future__ import annotations

from pathlib import Path

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
    """Returns coeffs (80,), s, R (3,3), t (2,)."""
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


def _densify(basis: Basis, lms_px: np.ndarray, coeffs, s, R, t,
             fronts: np.ndarray):
    """Map each mediapipe landmark (0..467) to (tri, bary) on the fitted,
    projected mesh. Returns list of (lm_idx, tri_idx, bary)."""
    proj2d, _ = project(basis, coeffs, s, R, t)
    tp = proj2d[basis.tris]                                # (T,3,2)
    out = []
    cand_idx = np.nonzero(fronts)[0]
    lo = tp[cand_idx].min(1)                               # (F,2)
    hi = tp[cand_idx].max(1)
    for li in range(468):
        p = lms_px[li]
        near = cand_idx[(lo[:, 0] <= p[0]) & (hi[:, 0] >= p[0]) &
                        (lo[:, 1] <= p[1]) & (hi[:, 1] >= p[1])]
        for ti in near:
            a, b_, c_ = tp[ti]
            M = np.array([[b_[0] - a[0], c_[0] - a[0]],
                          [b_[1] - a[1], c_[1] - a[1]]])
            det = np.linalg.det(M)
            if abs(det) < 1e-9:
                continue
            w = np.linalg.solve(M, p - a)
            bary = np.array([1 - w.sum(), w[0], w[1]])
            if (bary >= -0.02).all():
                out.append((li, int(ti), np.clip(bary, 0, 1)))
                break
    return out


DENSE_CORR = Path(__file__).with_name("dense_corr.npz")


# jawline: visible contour that defines face slimness -> high weight
JAW_LMS = {397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172,
           58, 132, 288, 361, 323, 93, 234}
# upper face oval: usually under a cap, mediapipe guesses it -> near-zero
FOREHEAD_LMS = {10, 338, 297, 332, 284, 251, 389, 356, 454, 21, 54, 103, 67,
                109, 162, 127}


def fit_shape_dense(basis: Basis, lms_px: np.ndarray,
                    lam_sym: float = 1.0, lam_asym: float = 15.0,
                    dense_weight: float = 0.3):
    """Fit with 31 named anchors + the static 468-landmark correspondence
    table (built once from a verified fit; mediapipe topology is identical
    across faces, so the table is face-independent)."""
    import os

    c, s, R, t, resid = fit_shape(basis, lms_px, lam_sym, lam_asym)
    if os.path.exists(DENSE_CORR):
        d = np.load(DENSE_CORR)
        pairs = list(zip(d["lm"], d["tri"], d["bary"]))
    else:
        from .texture import front_tris
        proj2d, _ = project(basis, c, s, R, t)
        anchor = basis.tri.surface_points["NOSE_TIP"][0]
        fronts = front_tris(basis, proj2d, anchor)
        pairs = _densify(basis, lms_px, c, s, R, t, fronts)

    # anchor correspondences (weight from WEIGHT), dense ones at dense_weight
    sp = basis.surface_points()
    names = [n for n in CORR if n in sp]
    base_list = [sp[n][0] for n in names]
    B_list = [sp[n][1] for n in names]
    u_list = [lms_px[CORR[n]] for n in names]
    w_list = [WEIGHT[n] for n in names]
    for li, ti, bary in pairs:
        vidx = basis.tris[ti]
        w3 = bary.astype(np.float32)
        base_list.append(w3 @ basis.verts[vidx])
        B_list.append(np.einsum("k,mkd->dm", w3, basis.modes[:, vidx, :]))
        u_list.append(lms_px[li])
        if int(li) in FOREHEAD_LMS:
            w_list.append(0.05)
        elif int(li) in JAW_LMS:
            w_list.append(0.9)
        else:
            w_list.append(dense_weight)

    base = np.stack(base_list)
    Bm = np.stack(B_list)
    u = np.stack(u_list)
    w = np.array(w_list)
    L = len(w)
    nm = basis.n_sym + basis.n_asym
    lam = np.concatenate([np.full(basis.n_sym, lam_sym),
                          np.full(basis.n_asym, lam_asym)])
    # scale ridge with total weight so it matches the anchor-only stage
    lam = lam * w.sum() / sum(WEIGHT[n] for n in names)

    for _ in range(6):
        X = base + Bm @ c
        s, R2, t = _pose(X, u, w)
        A = np.einsum("ij,ljm->lim", R2, Bm).reshape(2 * L, nm)
        rhs = ((u - t) / s) - base @ R2.T
        rhs = rhs.reshape(2 * L)
        ww = np.repeat(w, 2)
        AtA = A.T @ (ww[:, None] * A) + np.diag(lam)
        c = np.linalg.solve(AtA, A.T @ (ww * rhs))

    X = base + Bm @ c
    s, R2, t = _pose(X, u, w)
    R = np.vstack([R2, np.cross(R2[0], R2[1])])
    resid_px = np.sqrt((((s * X @ R2.T + t) - u) ** 2).sum(1))
    return c, s, R, t, {"n_dense": len(pairs),
                        "mean_resid": float((resid_px * w).sum() / w.sum())}


def _load_dense_pairs(basis: Basis, lms_px: np.ndarray, coeffs, s, R, t):
    import os

    if os.path.exists(DENSE_CORR):
        d = np.load(DENSE_CORR)
        return list(zip(d["lm"], d["tri"], d["bary"]))

    from .texture import front_tris
    proj2d, _ = project(basis, coeffs, s, R, t)
    anchor = basis.tri.surface_points["NOSE_TIP"][0]
    fronts = front_tris(basis, proj2d, anchor)
    return _densify(basis, lms_px, coeffs, s, R, t, fronts)


def _dense_observation_arrays(basis: Basis, lms_px: np.ndarray,
                              pairs, dense_weight: float):
    sp = basis.surface_points()
    names = [n for n in CORR if n in sp]
    base_list = [sp[n][0] for n in names]
    B_list = [sp[n][1] for n in names]
    u_list = [lms_px[CORR[n]] for n in names]
    w_list = [WEIGHT[n] for n in names]
    for li, ti, bary in pairs:
        vidx = basis.tris[int(ti)]
        w3 = np.asarray(bary, np.float32)
        base_list.append(w3 @ basis.verts[vidx])
        B_list.append(np.einsum("k,mkd->dm", w3, basis.modes[:, vidx, :]))
        u_list.append(lms_px[int(li)])
        if int(li) in FOREHEAD_LMS:
            w_list.append(0.05)
        elif int(li) in JAW_LMS:
            w_list.append(0.9)
        else:
            w_list.append(dense_weight)
    return np.stack(base_list), np.stack(B_list), np.stack(u_list), np.array(w_list)


def fit_shape_multi_dense(basis: Basis, lms_list: list[np.ndarray],
                          lam_sym: float = 0.16, lam_asym: float = 0.08,
                          dense_weight: float = 0.45,
                          photo_weights: list[float] | None = None,
                          n_iter: int = 8):
    """Fit one shared shape to landmarks from several photos.

    Each photo gets its own weak-perspective pose while all photos share the
    same FaceGen shape coefficients. This lets a clean front shot, a cap shot,
    and a mild side shot contribute different geometry evidence.
    """
    if not lms_list:
        raise ValueError("fit_shape_multi_dense requires at least one photo")
    if photo_weights is None:
        photo_weights = [1.0] * len(lms_list)
    if len(photo_weights) != len(lms_list):
        raise ValueError("photo_weights length must match lms_list")

    singles = [
        fit_shape_dense(basis, lms, lam_sym=lam_sym, lam_asym=lam_asym,
                        dense_weight=dense_weight)
        for lms in lms_list
    ]
    w_photo = np.asarray(photo_weights, np.float64)
    w_photo = w_photo / max(float(w_photo.mean()), 1e-6)
    c = sum(float(w) * single[0] for w, single in zip(w_photo, singles)) / w_photo.sum()

    observations = []
    for lms, single in zip(lms_list, singles):
        c0, s0, R0, t0, _ = single
        pairs = _load_dense_pairs(basis, lms, c0, s0, R0, t0)
        observations.append(_dense_observation_arrays(basis, lms, pairs, dense_weight))

    sp = basis.surface_points()
    names = [n for n in CORR if n in sp]
    anchor_weight = sum(WEIGHT[n] for n in names)
    total_weight = sum(float(pw) * obs[3].sum()
                       for pw, obs in zip(w_photo, observations))
    lam = np.concatenate([np.full(basis.n_sym, lam_sym),
                          np.full(basis.n_asym, lam_asym)])
    lam = lam * total_weight / max(anchor_weight * float(w_photo.sum()), 1e-6)
    nm = basis.n_sym + basis.n_asym

    poses = []
    for _ in range(n_iter):
        AtA = np.diag(lam).astype(np.float64)
        Atb = np.zeros(nm, np.float64)
        poses = []
        for pw, (base, Bm, u, w) in zip(w_photo, observations):
            X = base + Bm @ c
            s, R2, t = _pose(X, u, w)
            A = np.einsum("ij,ljm->lim", R2, Bm).reshape(2 * len(w), nm)
            rhs = ((u - t) / s) - base @ R2.T
            rhs = rhs.reshape(2 * len(w))
            ww = np.repeat(w * float(pw), 2)
            AtA += A.T @ (ww[:, None] * A)
            Atb += A.T @ (ww * rhs)
            poses.append((s, R2, t))
        c = np.linalg.solve(AtA, Atb)

    per_photo = []
    total_resid = 0.0
    total_resid_w = 0.0
    poses_out = []
    for pw, (base, Bm, u, w) in zip(w_photo, observations):
        X = base + Bm @ c
        s, R2, t = _pose(X, u, w)
        R = np.vstack([R2, np.cross(R2[0], R2[1])])
        resid_px = np.sqrt((((s * X @ R2.T + t) - u) ** 2).sum(1))
        mean_resid = float((resid_px * w).sum() / w.sum())
        per_photo.append(mean_resid)
        total_resid += mean_resid * float(pw) * w.sum()
        total_resid_w += float(pw) * w.sum()
        poses_out.append((s, R, t))

    return c, poses_out, {
        "n_photos": len(lms_list),
        "n_dense": len(observations[0][3]) - len(names),
        "mean_resid": float(total_resid / max(total_resid_w, 1e-6)),
        "per_photo_resid": per_photo,
        "photo_weights": [float(x) for x in w_photo],
    }
