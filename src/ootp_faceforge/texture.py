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


def fuse_maps(maps: list[np.ndarray], covs: list[np.ndarray],
              weights: list[float], neutral: np.ndarray | float,
              weight_power: float = 2.0,
              high_power: float | None = None,
              split_sigma: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Quality-weighted fusion of same-space photo maps.

    Invalid pixels fall back to neutral. The weight power lets the clearest
    photo dominate a region without discarding useful coverage from others.

    high_power: when set, split each map into low/high frequency bands at
    split_sigma and fuse the high band with this (much larger) power. Photos
    warped through one shared shape are misaligned by a few pixels, so a plain
    weighted mean averages stubble, brow edges, and pores into mush; near
    winner-take-all high-band weights keep the best photo's detail crisp while
    the low band still blends tone evidence from every photo.
    """
    if not maps:
        raise ValueError("fuse_maps requires at least one map")
    if len(maps) != len(covs) or len(maps) != len(weights):
        raise ValueError("maps, covs, and weights length must match")

    first = np.asarray(maps[0], np.float32)
    if np.isscalar(neutral):
        out = np.full(first.shape, float(neutral), np.float32)
    else:
        out = np.asarray(neutral, np.float32).copy()
        if out.shape != first.shape:
            out = np.broadcast_to(out, first.shape).astype(np.float32).copy()

    def _wfuse(bands: list[np.ndarray], power: float):
        accum = np.zeros_like(first, np.float32)
        wsum = np.zeros(first.shape[:2], np.float32)
        p = max(float(power), 0.01)
        for src, cov, weight in zip(bands, covs, weights):
            cov = np.asarray(cov, bool)
            if src.shape != first.shape or cov.shape != first.shape[:2]:
                raise ValueError("all maps and coverage masks must share shape")
            w = max(float(weight), 0.0) ** p
            if w <= 0:
                continue
            pix_w = cov.astype(np.float32) * w
            accum += src * pix_w[..., None]
            wsum += pix_w
        return accum, wsum

    srcs = [np.asarray(m, np.float32) for m in maps]
    if high_power is None:
        accum, wsum = _wfuse(srcs, weight_power)
        valid = wsum > 1e-6
        out[valid] = accum[valid] / wsum[valid, None]
        return out, valid

    lows = [cv2.GaussianBlur(s, (0, 0), max(split_sigma, 0.5)) for s in srcs]
    lo_acc, lo_w = _wfuse(lows, weight_power)
    hi_acc, hi_w = _wfuse([s - lo for s, lo in zip(srcs, lows)], high_power)
    valid = lo_w > 1e-6
    out[valid] = (lo_acc[valid] / lo_w[valid, None]
                  + hi_acc[valid] / np.maximum(hi_w[valid, None], 1e-6))
    return out, valid


def vertex_normals(basis: Basis, cam: np.ndarray, anchor_tri: int) -> np.ndarray:
    """Area-weighted per-vertex unit normals in camera space.

    The weak-perspective pose leaves the toward-camera z sign ambiguous, so the
    sign is fixed the same way front_tris does: the anchor triangle (under the
    nose tip) must face the camera (+z)."""
    v = cam[basis.tris]                                    # (T,3,3)
    fn = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])    # (T,3), |fn| ~ area
    sign = np.sign(fn[anchor_tri, 2]) or 1.0
    n = np.zeros_like(cam, np.float64)
    for k in range(3):
        np.add.at(n, basis.tris[:, k], fn)
    n *= sign
    return (n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
            ).astype(np.float32)


def _sh_basis(n: np.ndarray) -> np.ndarray:
    """Order-2 spherical-harmonic irradiance basis (N,9) from unit normals."""
    x, y, z = n[:, 0], n[:, 1], n[:, 2]
    return np.stack([np.ones_like(x), x, y, z, x * y, x * z, y * z,
                     x * x - y * y, 3.0 * z * z - 1.0], 1).astype(np.float32)


def estimate_shading(img: np.ndarray, sample_mask: np.ndarray, basis: Basis,
                     proj2d: np.ndarray, cam: np.ndarray, anchor_tri: int,
                     norm_mask: np.ndarray | None = None,
                     min_samples: int = 400, lam: float = 3e-3,
                     min_r2: float = 0.12) -> tuple[np.ndarray, float] | None:
    """Estimate the photo's smooth baked-in lighting from the fitted geometry.

    Lambertian irradiance is an order-2 SH function of the surface normal, so
    regressing photo intensity (sampled at visible mesh vertices) on the SH
    basis separates lighting from albedo: the smooth normal-dependent part is
    light, everything else is skin. Dividing the photo by this field removes
    attached shading without flattening pores, stubble, or feature lines the
    way blur-based neutralizers do.

    sample_mask: (H,W) bool of pixels the fit may sample (skin only; exclude
    brows/eyes/lips, they are dark albedo, not shadow). norm_mask: region whose
    median shading is normalized to 1 (defaults to sample_mask).
    Returns ((H,W,3) float multiplicative shading, r2) or None when the fit is
    unreliable (few samples, flat lighting, bad geometry alignment).
    """
    h, w = img.shape[:2]
    normals = vertex_normals(basis, cam, anchor_tri)
    px = np.round(proj2d).astype(int)
    inb = ((px[:, 0] >= 1) & (px[:, 0] < w - 1)
           & (px[:, 1] >= 1) & (px[:, 1] < h - 1))
    sel = np.nonzero(inb & (normals[:, 2] > 0.25))[0]
    sel = sel[sample_mask[px[sel, 1], px[sel, 0]]]
    if len(sel) < min_samples:
        return None
    sm = cv2.blur(img.astype(np.float32), (3, 3))
    vals = sm[px[sel, 1], px[sel, 0]]                      # (N,3)
    A = _sh_basis(normals[sel])
    luma = vals @ np.array([0.299, 0.587, 0.114], np.float32)

    def _solve(Ai: np.ndarray, bi: np.ndarray) -> np.ndarray:
        AtA = Ai.T @ Ai / len(Ai) + lam * np.eye(Ai.shape[1], dtype=np.float32)
        return np.linalg.solve(AtA, Ai.T @ bi / len(Ai))

    # Dark-albedo leftovers (stubble, moles) and cast shadows (nose, cap brim)
    # violate the attached-shading model; MAD-trimmed refits keep them out.
    keep = np.ones(len(sel), bool)
    res = np.zeros(len(sel), np.float32)
    for _ in range(2):
        l9 = _solve(A[keep], luma[keep])
        res = luma - A @ l9
        mad = float(np.median(np.abs(res[keep] - np.median(res[keep]))))
        keep = np.abs(res) < max(3.5 * 1.4826 * mad, 4.0)
        if keep.sum() < min_samples:
            return None
    r2 = 1.0 - float(res[keep].var() / max(luma[keep].var(), 1e-6))
    if r2 < min_r2:
        return None

    # Per-channel light coefficients also capture one-sided color casts.
    L = np.stack([_solve(A[keep], vals[keep, ch]) for ch in range(3)], 1)

    # Splat per-vertex predicted shading into a coarse photo-space grid and
    # blur-normalize; SH shading is smooth, so this reconstruction is exact
    # enough and extrapolates over masked-out regions like capped foreheads.
    splat = np.nonzero(inb & (normals[:, 2] > 0.05))[0]
    sv = np.clip(_sh_basis(normals[splat]) @ L, 1.0, None)  # (S,3)
    ds = 8
    gh, gw = h // ds + 2, w // ds + 2
    acc = np.zeros((gh, gw, 3), np.float32)
    wgt = np.zeros((gh, gw), np.float32)
    gy, gx = px[splat, 1] // ds, px[splat, 0] // ds
    np.add.at(acc, (gy, gx), sv)
    np.add.at(wgt, (gy, gx), 1.0)
    sigma = max(float(np.ptp(px[splat, 0])) / ds / 10.0, 1.5)
    acc = cv2.GaussianBlur(acc, (0, 0), sigma)
    wgt = cv2.GaussianBlur(wgt, (0, 0), sigma)
    field = np.ones((gh, gw, 3), np.float32)
    ok = wgt > 1e-4
    field[ok] = acc[ok] / wgt[ok, None]
    shading = cv2.resize(field, (gw * ds, gh * ds),
                         interpolation=cv2.INTER_LINEAR)[:h, :w]

    mask = norm_mask if norm_mask is not None else sample_mask
    med = np.median(shading[mask].reshape(-1, 3), 0)
    shading /= np.maximum(med, 1e-3)
    return np.clip(shading, 0.45, 2.2), r2


def build_detail(basis: Basis, photo: np.ndarray, proj2d: np.ndarray,
                 tex_coeffs: np.ndarray, anchor_tri: int,
                 photo_valid: np.ndarray | None = None,
                 cam: np.ndarray | None = None, min_cos: float = 0.22,
                 size: int = 1024, feather: float = 24.0,
                 detail_strength: float = 1.0,
                 chroma_strength: float = 0.08,
                 edge_strength: float = 0.85,
                 flat_neutralize: float = 0.45,
                 shadow_neutralize: float = 0.8,
                 highlight_neutralize: float = 0.0,
                 dark_keep: float = 0.0,
                 neutralize_eyes: bool = True,
                 eye_detail_strength: float = 0.0,
                 source_lms: np.ndarray | None = None,
                 likeness_detail: float = 0.0,
                 likeness_detail_gain: float = 1.2,
                 protect: np.ndarray | None = None,
                 protect_gain: float = 1.0,
                 suppress: np.ndarray | None = None,
                 suppress_strength: float = 0.95,
                 frame: np.ndarray | None = None,
                 frame_color: np.ndarray | None = None,
                 frame_strength: float = 0.55):
    """Returns detail modulation map (size,size,3) uint8 (64 = identity).
    photo_valid: optional (H,W) bool mask of usable photo pixels (e.g. face
    hull without cap/background).
    protect: optional (H,W) bool photo-space mask (e.g. eyeglasses) whose detail
    is kept out of the neutralize/eye/shadow passes so it survives into OOTP;
    protect_gain deepens the kept detail's contrast about the 64 midpoint.
    suppress: optional (H,W) bool photo-space mask whose detail should be
    removed from the final map (e.g. eyeglasses in source photos when the
    generated player should not wear glasses).
    frame: optional (H,W) float 0..1 photo-space eyeglass-frame signal
    composited onto the fully-cleaned detail (an alternative to `protect` that
    carries only the frame, no lens glare); frame_strength sets how strong."""
    fronts = front_tris(basis, proj2d, anchor_tri, cam, min_cos)
    dpx, dvalid = detail_px(basis, size)
    tri_ok = fronts & dvalid[basis.tris].all(1)

    src_photo = tri_coords(basis, proj2d)
    dst = dpx[basis.tris]

    # kill smear triangles: tiny in the photo but huge in detail space
    a_src = np.abs(np.cross(src_photo[:, 1] - src_photo[:, 0],
                            src_photo[:, 2] - src_photo[:, 0]))
    a_dst = np.abs(np.cross(dst[:, 1] - dst[:, 0], dst[:, 2] - dst[:, 0]))
    if not tri_ok.any():
        return np.full((size, size, 3), 64, np.uint8), np.zeros((size, size), bool)
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

    def _to_detail(sig: np.ndarray, blur_sigma: float) -> np.ndarray:
        rep = np.repeat(sig[..., None].astype(np.float32), 3, 2)
        det, _ = warp_tris(rep, src_photo, dst, size, tri_ok, order)
        return cv2.GaussianBlur(np.clip(det[..., 0], 0, 1), (0, 0), blur_sigma)

    protect_det = None
    if protect is not None and protect.any():
        protect_det = _to_detail(protect.astype(np.float32), 1.5)
    suppress_det = None
    if suppress is not None and suppress.any():
        suppress_det = _to_detail(suppress.astype(np.float32), 1.0)
        k = max(3, int(round(size * 0.008)) | 1)
        suppress_det = cv2.dilate(
            suppress_det.astype(np.float32),
            np.ones((k, k), np.uint8),
        )
        suppress_det = cv2.GaussianBlur(
            np.clip(suppress_det, 0, 1),
            (0, 0),
            max(1.0, size * 0.003),
        )
    likeness_det = None
    if source_lms is not None and likeness_detail > 0:
        src_mask = _likeness_source_mask(photo.shape[:2], source_lms)
        if src_mask.any():
            likeness_det = _to_detail(src_mask, 1.2)
    if likeness_det is not None and suppress_det is not None:
        likeness_det *= 1.0 - np.clip(suppress_det * 1.15, 0.0, 1.0)
    frame_det = None
    if frame is not None and frame.any():
        frame_det = _to_detail(frame, 1.0)

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
    # Snapshot before the neutralizers so a protected region (eyeglasses) can
    # keep its raw frame/lens detail; those passes are what erase frames.
    D_pre = D.copy()
    # dark_keep > 0 (delit photos): sub-64 detail is albedo (brows, stubble,
    # lip lines), not baked shadow, so let more of it survive into OOTP.
    D = _compress_dark_detail(D, valid, float(np.clip(dark_keep, 0.0, 1.0)))
    D = _lift_detail_shadows(D, valid, shadow_neutralize)
    D = _lower_detail_highlights(D, valid, highlight_neutralize)
    D = _neutralize_flat_detail(D, valid, flat_neutralize)

    if neutralize_eyes:
        # Neutralizing is useful for in-game eyeballs, but reference OOTP .fg
        # files often bake the photo eyes into the detail map.
        neutral = _neutralize_eyes(basis, D, size)
        D = neutral + eye_detail_strength * (D - neutral)

    if likeness_det is not None:
        a = np.clip(
            likeness_det * float(np.clip(likeness_detail, 0.0, 1.0))
            * valid.astype(np.float32),
            0.0,
            1.0,
        )[..., None]
        gain = float(np.clip(likeness_detail_gain, 0.0, 3.0))
        kept = np.clip(64.0 + gain * (D_pre - 64.0), 0, 255)
        D = a * kept + (1.0 - a) * D

    if protect_det is not None:
        a = (protect_det * valid.astype(np.float32))[..., None]
        kept = np.clip(64.0 + protect_gain * (D_pre - 64.0), 0, 255)
        D = a * kept + (1.0 - a) * D

    if likeness_det is not None or protect_det is not None:
        D = _lower_detail_highlights(D, valid, highlight_neutralize)

    if suppress_det is not None:
        a = np.clip(
            suppress_det * float(np.clip(suppress_strength, 0.0, 1.0))
            * valid.astype(np.float32),
            0.0,
            1.0,
        )[..., None]
        D = a * 64.0 + (1.0 - a) * D

    if frame_det is not None:
        # Composite the frame onto the cleaned detail. With frame_color present
        # we preserve colored frames (red/blue sports glasses); otherwise we
        # multiplicatively darken like the older grayscale path.
        f = (frame_det * valid.astype(np.float32))[..., None]
        if frame_color is not None:
            color = np.asarray(frame_color, np.float32).reshape(1, 1, 3)
            D = f * color + (1.0 - f) * D
        else:
            D = D * (1.0 - frame_strength * f)

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


def _likeness_source_mask(shape: tuple[int, int], lms: np.ndarray) -> np.ndarray:
    """Photo-space mask for feature lines that carry player likeness.

    The ordinary clean-up passes deliberately suppress baked-in lighting and
    mottled skin. For likeness builds we keep the photo detail around brows,
    eyelids, nose, mouth, and chin because those are the cues the limited OOTP
    shape basis cannot reliably reproduce.
    """
    h, w = int(shape[0]), int(shape[1])
    pts = np.asarray(lms, np.float32)
    if pts.ndim != 2 or pts.shape[0] < 474:
        return np.zeros((h, w), np.float32)
    eye_d = float(np.linalg.norm(pts[473] - pts[468]))
    if eye_d <= 1e-6:
        eye_d = max(float(np.ptp(pts[:, 0])), 1.0) * 0.25
    line = max(2, int(round(0.045 * eye_d)))
    broad = max(line + 1, int(round(0.075 * eye_d)))
    mask = np.zeros((h, w), np.float32)

    def poly(indices: list[int], value: float, thickness: int,
             closed: bool = False) -> None:
        if max(indices) >= pts.shape[0]:
            return
        p = np.round(pts[indices]).astype(np.int32)
        cv2.polylines(mask, [p], closed, float(value), thickness,
                      lineType=cv2.LINE_AA)

    def blob(indices: list[int], value: float, radius_scale: float) -> None:
        if max(indices) >= pts.shape[0]:
            return
        c = pts[indices].mean(0)
        radius = max(2, int(round(radius_scale * eye_d)))
        cv2.circle(mask, tuple(np.round(c).astype(np.int32)), radius,
                   float(value), -1, lineType=cv2.LINE_AA)

    # Eyebrows and eyelids do the most identity work in OOTP's bald front view.
    poly([70, 63, 105, 66, 107], 1.00, broad)
    poly([336, 296, 334, 293, 300], 1.00, broad)
    poly([46, 53, 52, 65, 55], 0.75, line)
    poly([276, 283, 282, 295, 285], 0.75, line)
    poly([33, 7, 163, 144, 145, 153, 154, 155, 133], 0.85, line, True)
    poly([263, 249, 390, 373, 374, 380, 381, 382, 362], 0.85, line, True)
    blob([33, 133], 0.42, 0.16)
    blob([263, 362], 0.42, 0.16)

    # Nose bridge, tip, and nostrils anchor the center-face impression.
    poly([168, 6, 197, 195, 5, 4, 1, 2], 0.70, line)
    poly([98, 97, 2, 326, 327], 0.65, line)
    blob([1, 2, 4, 5], 0.38, 0.13)

    # Mouth and chin often distinguish otherwise similar FaceGen heads.
    poly([61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
          375, 321, 405, 314, 17, 84, 181, 91, 146], 0.72, line, True)
    poly([78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
          415, 310, 311, 312, 13, 82, 81, 80, 191], 0.55, line, True)
    poly([172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397],
         0.35, line)

    mask = cv2.GaussianBlur(np.clip(mask, 0, 1), (0, 0), max(1.0, 0.018 * eye_d))
    return np.clip(mask, 0, 1).astype(np.float32)


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


def _compress_dark_detail(D: np.ndarray, valid: np.ndarray,
                          keep: np.ndarray | float = 0.0) -> np.ndarray:
    """Soft-limit extreme dark detail values that usually come from shadows.

    keep (scalar or (H,W), 0..1) relaxes the compressor: 0 = strict shadow
    clean-up, 1 = mostly preserve dark albedo (brow/beard zones of delit
    photos)."""
    k = np.clip(np.asarray(keep, np.float32), 0.0, 1.0)
    floor = 52.0 - 20.0 * k
    slope = 0.18 + 0.5 * k
    luma = D @ np.array([0.299, 0.587, 0.114], np.float32)
    dark = (luma < floor) & valid
    if not dark.any():
        return D
    target = floor + (luma - floor) * slope
    lift = np.maximum(target - luma, 0.0)
    out = D.copy()
    out[dark] = np.clip(out[dark] + lift[dark][:, None], 0, 255)
    return out


def _lift_detail_shadows(D: np.ndarray, valid: np.ndarray,
                         amount: float) -> np.ndarray:
    """Suppress baked-in low-frequency shadows while preserving feature edges."""
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0 or not valid.any():
        return D
    luma = D @ np.array([0.299, 0.587, 0.114], np.float32)
    low = cv2.GaussianBlur(luma - 64.0, (0, 0), 4.2)
    shadow = np.minimum(low, 0.0)
    micro = np.abs(luma - cv2.GaussianBlur(luma, (0, 0), 1.2))
    edge_keep = np.clip((micro - 3.0) / 9.0, 0.0, 1.0)
    lift = amount * (1.0 - edge_keep) * valid.astype(np.float32)
    return np.clip(D - shadow[..., None] * lift[..., None], 0, 255)


def _lower_detail_highlights(D: np.ndarray, valid: np.ndarray,
                             amount: float) -> np.ndarray:
    """Suppress broad baked-in highlights while preserving feature edges."""
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0 or not valid.any():
        return D
    luma = D @ np.array([0.299, 0.587, 0.114], np.float32)
    high = cv2.GaussianBlur(luma - 64.0, (0, 0), 4.2)
    highlight = np.maximum(high, 0.0)
    micro = np.abs(luma - cv2.GaussianBlur(luma, (0, 0), 1.2))
    edge_keep = np.clip((micro - 3.0) / 9.0, 0.0, 1.0)
    lower = amount * (1.0 - edge_keep) * valid.astype(np.float32)
    return np.clip(D - highlight[..., None] * lower[..., None], 0, 255)


def _surface_point_detail_px(basis: Basis, name: str, size: int) -> np.ndarray:
    fidx, bary = basis.tri.surface_points[name]
    vidx = basis.tris[fidx]
    uv = np.asarray(bary, np.float32) @ basis.vert_uv[vidx]
    duv = basis.fim_lookup(uv[None])[0]
    return np.array([duv[0] * size, (1 - duv[1]) * size], np.float32)


def _neutralize_eyes(basis: Basis, D: np.ndarray, size: int,
                     core: float = 0.75) -> np.ndarray:
    """Blend D toward identity inside each eye opening (feathered ellipse)."""
    if not basis.tri.surface_points:
        return D
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
