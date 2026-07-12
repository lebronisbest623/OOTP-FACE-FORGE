"""Modeller-style direct FaceGen coefficient fitting.

This is a deterministic coefficient solve, not a learned photo->coeff regressor.
It keeps the existing landmark camera/shape fit but adds an optional face-space
prior term from a retrieved/proposed FG coefficient vector:

  landmark reprojection error + zero prior + candidate face-space prior

The candidate prior is solved inside the same least-squares system as the
landmarks, so it cannot simply overwrite the geometry after the fact.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fit import _pair_geometry, _pose


@dataclass(frozen=True)
class DirectFitResult:
    coeffs: np.ndarray
    poses: list[tuple[float, np.ndarray, np.ndarray]]
    mean_resid: float
    per_photo_resid: list[float]
    delta_norm: float
    prior_norm: float
    prior_strength: float


def _cap_shape(c: np.ndarray, n_sym: int, shape_cap: float,
               shape_norm_cap: float) -> np.ndarray:
    out = np.asarray(c, np.float64).copy()
    out[:n_sym] = np.clip(out[:n_sym], -float(shape_cap), float(shape_cap))
    sym_norm = float(np.linalg.norm(out[:n_sym]))
    if sym_norm > float(shape_norm_cap):
        out[:n_sym] *= float(shape_norm_cap) / max(sym_norm, 1e-9)
    return out


def _trust_anchor(c: np.ndarray, anchor: np.ndarray, n_sym: int,
                  r_max: float) -> np.ndarray:
    if r_max <= 0:
        return c
    out = c.copy()
    d = out[:n_sym] - anchor[:n_sym]
    r = float(np.sqrt(np.mean(d ** 2)))
    if r > r_max:
        out[:n_sym] = anchor[:n_sym] + d * (float(r_max) / max(r, 1e-9))
    asym_start = n_sym
    if len(out) > asym_start:
        d_asym = out[asym_start:] - anchor[asym_start:]
        r_asym = float(np.sqrt(np.mean(d_asym ** 2)))
        asym_r_max = float(r_max) * 0.35
        if r_asym > asym_r_max:
            out[asym_start:] = (
                anchor[asym_start:]
                + d_asym * (asym_r_max / max(r_asym, 1e-9))
            )
    return out


def fit_shape_direct_prior(
    basis,
    lms_list: list[np.ndarray],
    anchor_coeffs: np.ndarray,
    prior_coeffs: np.ndarray,
    prior_strength: float,
    lam_sym: float = 0.005,
    lam_asym: float = 0.3,
    dense_weight: float = 0.45,
    photo_weights: list[float] | None = None,
    n_iter: int = 4,
    prior_lam_sym: float = 0.04,
    prior_lam_asym: float = 0.015,
    r_max: float = 0.75,
    shape_cap: float = 3.5,
    shape_norm_cap: float = 9.5,
) -> DirectFitResult:
    """Refit shape coefficients with a candidate prior inside the solve."""
    from .calibrate import calibrated_pairs

    if not lms_list:
        raise ValueError("fit_shape_direct_prior requires at least one photo")
    nm = basis.n_sym + basis.n_asym
    anchor = np.asarray(anchor_coeffs, np.float64).reshape(-1)[:nm]
    prior = np.asarray(prior_coeffs, np.float64).reshape(-1)[:nm]
    if len(anchor) != nm or len(prior) != nm:
        raise ValueError("anchor/prior coefficient length mismatch")

    strength = float(np.clip(prior_strength, 0.0, 1.0))
    if strength <= 1e-6:
        strength = 0.0

    if photo_weights is None:
        photo_weights = [1.0] * len(lms_list)
    if len(photo_weights) != len(lms_list):
        raise ValueError("photo_weights length must match lms_list")

    base, Bm, w, lm_ids = _pair_geometry(
        basis, calibrated_pairs(basis), dense_weight
    )
    us = [np.asarray(lms, np.float64)[lm_ids] for lms in lms_list]
    w_photo = np.asarray(photo_weights, np.float64)
    w_photo = w_photo / max(float(w_photo.mean()), 1e-6)

    obs_weight = float(w.sum())
    zero_lam = np.concatenate([
        np.full(basis.n_sym, lam_sym),
        np.full(basis.n_asym, lam_asym),
    ]).astype(np.float64) * obs_weight
    prior_lam = np.concatenate([
        np.full(basis.n_sym, prior_lam_sym),
        np.full(basis.n_asym, prior_lam_asym),
    ]).astype(np.float64) * obs_weight * strength

    c = anchor.copy()
    for _ in range(max(1, int(n_iter))):
        X = base + Bm @ c
        AtA = np.diag(zero_lam + prior_lam).astype(np.float64)
        Atb = prior_lam * prior
        for pw, u in zip(w_photo, us):
            s, R2, t = _pose(X, u, w)
            A = np.einsum("ij,ljm->lim", R2, Bm).reshape(2 * len(w), nm)
            rhs = (((u - t) / s) - base @ R2.T).reshape(2 * len(w))
            wwp = np.repeat(w * float(pw), 2)
            AtA += A.T @ (wwp[:, None] * A)
            Atb += A.T @ (wwp * rhs)
        c = np.linalg.solve(AtA, Atb)
        c = _trust_anchor(c, anchor, basis.n_sym, float(r_max))
        c = _cap_shape(c, basis.n_sym, float(shape_cap), float(shape_norm_cap))

    X = base + Bm @ c
    per_photo: list[float] = []
    poses: list[tuple[float, np.ndarray, np.ndarray]] = []
    total_resid = 0.0
    total_w = 0.0
    for pw, u in zip(w_photo, us):
        s, R2, t = _pose(X, u, w)
        R = np.vstack([R2, np.cross(R2[0], R2[1])])
        resid_px = np.sqrt((((s * X @ R2.T + t) - u) ** 2).sum(1))
        mean_resid = float((resid_px * w).sum() / w.sum())
        per_photo.append(mean_resid)
        total_resid += mean_resid * float(pw) * obs_weight
        total_w += float(pw) * obs_weight
        poses.append((s, R, t))

    return DirectFitResult(
        coeffs=c.astype(np.float32),
        poses=poses,
        mean_resid=float(total_resid / max(total_w, 1e-6)),
        per_photo_resid=per_photo,
        delta_norm=float(np.linalg.norm((c - anchor)[:basis.n_sym])),
        prior_norm=float(np.linalg.norm((prior - anchor)[:basis.n_sym])),
        prior_strength=strength,
    )
