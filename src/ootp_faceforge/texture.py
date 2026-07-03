"""Texture pipeline: warp photo into FaceGen texture spaces, fit texture
coefficients, and build the detail modulation map."""
from __future__ import annotations

import cv2
import numpy as np

from .basis import Basis


def warp_tris(src: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray,
              dst_size: int, tri_sel: np.ndarray,
              order: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise-affine warp of selected triangles.
    src_pts/dst_pts: (T,3,2). Returns (dst_size,dst_size,3) float and coverage
    mask. order: optional paint order (indices into T); later wins overlaps."""
    out = np.zeros((dst_size, dst_size, 3), np.float32)
    cov = np.zeros((dst_size, dst_size), np.uint8)
    idxs = np.nonzero(tri_sel)[0] if order is None else order[tri_sel[order]]
    for ti in idxs:
        d = dst_pts[ti]
        x0, y0 = np.floor(d.min(0)).astype(int)
        x1, y1 = np.ceil(d.max(0)).astype(int) + 1
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, dst_size), min(y1, dst_size)
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        M = cv2.getAffineTransform(src_pts[ti].astype(np.float32),
                                   (d - [x0, y0]).astype(np.float32))
        patch = cv2.warpAffine(src, M, (x1 - x0, y1 - y0),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillConvexPoly(mask, np.round(d - [x0, y0]).astype(np.int32), 1)
        m = mask.astype(bool)
        out[y0:y1, x0:x1][m] = patch[m] if patch.ndim == 3 else patch[m, None]
        cov[y0:y1, x0:x1][m] = 1
    return out, cov.astype(bool)


def front_tris(basis: Basis, proj2d: np.ndarray, anchor_tri: int,
               cam: np.ndarray | None = None, min_cos: float = 0.0) -> np.ndarray:
    """Boolean (T,) of camera-facing triangles, sign chosen so the anchor
    triangle (e.g. under NOSE_TIP) counts as front-facing. With cam (V,3)
    camera-space verts, triangles at grazing angles (facing cosine below
    min_cos) are excluded — those smear background into the texture."""
    p = proj2d[basis.tris]                                # (T,3,2)
    area = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    sign = np.sign(area[anchor_tri]) or 1.0
    if cam is None:
        return area * sign > 1e-6
    v = cam[basis.tris]                                   # (T,3,3)
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    cos = n[:, 2] / np.maximum(np.linalg.norm(n, axis=1), 1e-9)
    return cos * sign > min_cos


def tri_coords(basis: Basis, per_vert_xy: np.ndarray) -> np.ndarray:
    """(T,3,2) coordinates per triangle corner from per-vertex 2D coords."""
    return per_vert_xy[basis.tris]


def uv_px(basis: Basis, size: int) -> np.ndarray:
    """Per-vertex UV -> pixel coords in a size x size texture image (row 0 = top)."""
    uv = basis.vert_uv
    return np.stack([uv[:, 0] * size, (1 - uv[:, 1]) * size], 1)


def detail_px(basis: Basis, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-vertex detail-texture pixel coords via FIM; also validity mask."""
    duv = basis.fim_lookup(basis.vert_uv)                 # (V,2), -1 = unmapped
    valid = (duv[:, 0] >= 0) & (duv[:, 1] >= 0)
    px = np.stack([duv[:, 0] * size, (1 - duv[:, 1]) * size], 1)
    return px, valid


def fit_tex_coeffs(basis: Basis, photo_uv: np.ndarray, cov: np.ndarray,
                   lam: float = 30.0, max_px: int = 15000,
                   erode: int = 3) -> np.ndarray:
    """Ridge-fit 50 sym texture coeffs to the photo warped into UV256 space."""
    # face-front region of UV space = where FIM has a valid mapping
    fim_valid = basis.fim[..., 0] >= 0
    mask = cov & fim_valid
    if erode > 0:
        k = np.ones((erode, erode), np.uint8)
        mask = cv2.erode(mask.astype(np.uint8), k).astype(bool)
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return np.zeros(basis.egt.sym.shape[0])
    if len(ys) > max_px:
        sel = np.random.default_rng(0).choice(len(ys), max_px, replace=False)
        ys, xs = ys[sel], xs[sel]
    A = basis.egt.sym[:, ys, xs, :].reshape(basis.egt.sym.shape[0], -1).T
    b = (photo_uv[ys, xs] - basis.mean_tex[ys, xs]).reshape(-1)
    n = A.shape[0]
    AtA = A.T @ A / n + lam * np.eye(A.shape[1])
    coeffs = np.linalg.solve(AtA, A.T @ b / n)
    return np.clip(coeffs, -4, 4)


def build_detail(basis: Basis, photo: np.ndarray, proj2d: np.ndarray,
                 tex_coeffs: np.ndarray, anchor_tri: int,
                 photo_valid: np.ndarray | None = None,
                 cam: np.ndarray | None = None, min_cos: float = 0.22,
                 size: int = 1024, feather: float = 24.0,
                 detail_strength: float = 1.0,
                 chroma_strength: float = 0.08,
                 edge_strength: float = 0.85,
                 flat_neutralize: float = 0.45,
                 neutralize_eyes: bool = True,
                 eye_detail_strength: float = 0.0):
    """Returns detail modulation map (size,size,3) uint8 (64 = identity).
    photo_valid: optional (H,W) bool mask of usable photo pixels (e.g. face
    hull without cap/background)."""
    fronts = front_tris(basis, proj2d, anchor_tri, cam, min_cos)
    dpx, dvalid = detail_px(basis, size)
    tri_ok = fronts & dvalid[basis.tris].all(1)

    src_photo = tri_coords(basis, proj2d)
    dst = dpx[basis.tris]

    # kill smear triangles: tiny in the photo but huge in detail space
    a_src = np.abs(np.cross(src_photo[:, 1] - src_photo[:, 0],
                            src_photo[:, 2] - src_photo[:, 0]))
    a_dst = np.abs(np.cross(dst[:, 1] - dst[:, 0], dst[:, 2] - dst[:, 0]))
    med_mag = np.median(a_dst[tri_ok] / np.maximum(a_src[tri_ok], 1e-6))
    tri_ok &= a_dst / np.maximum(a_src, 1e-6) < 4.0 * med_mag

    # paint least-frontal first so frontal triangles win overlaps
    if cam is not None:
        v3 = cam[basis.tris]
        n3 = np.cross(v3[:, 1] - v3[:, 0], v3[:, 2] - v3[:, 0])
        cosf = np.abs(n3[:, 2]) / np.maximum(np.linalg.norm(n3, axis=1), 1e-9)
        order = np.argsort(cosf)
    else:
        order = None

    det_photo, cov1 = warp_tris(photo.astype(np.float32), src_photo, dst, size,
                                tri_ok, order)
    if photo_valid is not None:
        vm = np.repeat(photo_valid[..., None].astype(np.float32), 3, 2)
        det_vm, _ = warp_tris(vm, src_photo, dst, size, tri_ok, order)
        cov1 &= det_vm[..., 0] > 0.9

    S256 = basis.stat_texture(tex_coeffs)
    src_uv = uv_px(basis, 256)[basis.tris]
    det_S, cov2 = warp_tris(S256, src_uv, dst, size, tri_ok, order)

    valid = cov1 & cov2
    D = np.full((size, size, 3), 64.0, np.float32)
    Dv = 64.0 * det_photo[valid] / np.maximum(det_S[valid], 8.0)
    # neutralize: keep only detail, move average tone/color out of D so it
    # composes sanely onto whatever base texture the renderer picks
    Dv *= 64.0 / np.maximum(Dv.mean(0), 1.0)
    # clamp chroma: ratio color halos (brow/eye misalignment, rim light tint)
    g = Dv.mean(1, keepdims=True)
    Dv = g + np.clip(Dv - g, -0.25 * g, 0.25 * g)
    # soft-compress the high side like FaceGen does (official p99 ~ 93)
    hi = Dv > 90.0
    Dv[hi] = 90.0 + (Dv[hi] - 90.0) * 0.3
    D[valid] = Dv

    D = _limit_chroma(D, chroma_strength)

    # edge-preserving smoothing: kills skin mottling, keeps feature lines
    D = cv2.bilateralFilter(D.astype(np.float32), 7, 11, 4)
    # mild unsharp to restore pore/stubble micro-contrast
    blur = cv2.GaussianBlur(D, (0, 0), 2.5)
    D = np.clip(D + edge_strength * (D - blur), 0, 255)

    D = np.clip(64.0 + detail_strength * (D - 64.0), 0, 255)
    D = _neutralize_flat_detail(D, valid, flat_neutralize)

    if neutralize_eyes:
        # Neutralizing is useful for in-game eyeballs, but reference OOTP .fg
        # files often bake the photo eyes into the detail map.
        neutral = _neutralize_eyes(basis, D, size)
        D = neutral + eye_detail_strength * (D - neutral)

    # feather toward identity (64) at coverage boundary
    dist = cv2.distanceTransform(valid.astype(np.uint8), cv2.DIST_L2, 3)
    alpha = np.clip(dist / feather, 0, 1)[..., None]
    D = 64.0 + alpha * (D - 64.0)
    return np.clip(D, 0, 255).astype(np.uint8), valid


def _limit_chroma(D: np.ndarray, strength: float) -> np.ndarray:
    """Keep mostly luma detail; color detail bloats JPEGs and looks blotchy in OOTP."""
    strength = float(np.clip(strength, 0.0, 1.0))
    luma = D @ np.array([0.299, 0.587, 0.114], np.float32)
    return luma[..., None] + strength * (D - luma[..., None])


def _neutralize_flat_detail(D: np.ndarray, valid: np.ndarray,
                            amount: float) -> np.ndarray:
    """Blend low-information skin flats toward FaceGen's neutral detail value."""
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0:
        return D
    luma = D @ np.array([0.299, 0.587, 0.114], np.float32)
    blur = cv2.GaussianBlur(luma, (0, 0), 1.8)
    micro = np.abs(luma - blur)
    delta = np.abs(luma - 64.0)
    edge_keep = np.clip((micro - 1.4) / 5.5, 0.0, 1.0)
    feature_keep = np.clip((delta - 4.0) / 14.0, 0.0, 1.0)
    keep = np.maximum(edge_keep, feature_keep)
    blend = amount * (1.0 - keep) * valid.astype(np.float32)
    return 64.0 + (D - 64.0) * (1.0 - blend[..., None])


def _surface_point_detail_px(basis: Basis, name: str, size: int) -> np.ndarray:
    fidx, bary = basis.tri.surface_points[name]
    vidx = basis.tris[fidx]
    uv = np.asarray(bary, np.float32) @ basis.vert_uv[vidx]
    duv = basis.fim_lookup(uv[None])[0]
    return np.array([duv[0] * size, (1 - duv[1]) * size], np.float32)


def _neutralize_eyes(basis: Basis, D: np.ndarray, size: int,
                     core: float = 0.75) -> np.ndarray:
    """Blend D toward identity inside each eye opening (feathered ellipse)."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    for side in ("LEFT", "RIGHT"):
        c = _surface_point_detail_px(basis, f"EYE_{side}_CENTRE", size)
        i = _surface_point_detail_px(basis, f"EYE_{side}_INNER", size)
        o = _surface_point_detail_px(basis, f"EYE_{side}_OUTER", size)
        if (c < 0).any() or (i < 0).any() or (o < 0).any():
            continue
        a = 0.62 * np.linalg.norm(i - o)          # semi-major
        b = 0.48 * a                               # semi-minor
        r = np.sqrt(((xx - c[0]) / a) ** 2 + ((yy - c[1]) / b) ** 2)
        # 0 inside the core (fully neutral), ramps to 1 at the ellipse edge
        blend = np.clip((r - core) / (1.0 - core), 0.0, 1.0)[..., None]
        D = 64.0 + (D - 64.0) * blend
    return D
