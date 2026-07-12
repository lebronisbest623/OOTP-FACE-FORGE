"""Multi-photo photo folder -> OOTP-ready .fg.

Shape is fit jointly from every usable photo. Texture/detail default to
multi-photo fusion so the generated file is tuned for OOTP in-game rendering,
where small portrait views and game lighting can suppress subtle face detail.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import emb2shape, glasses, glasses_mesh, identity, modeller_fit, photofit, render_retrieval, restore, retrieval
from .basis import get_basis
from .fgformat import FgFile
from .fit import fit_shape_multi_dense, project
from .landmarks import (
    FACE_OVAL,
    detect,
    face_mask,
    feature_exclude_mask,
    illum_correct,
    skin_luminance,
)
from .texture import build_detail, detail_px, estimate_shading, fit_tex_coeffs, front_tris, fuse_maps, tri_coords, uv_px, warp_tris


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
RESTORE_EYE_D = 60.0  # faces smaller than this benefit from GFPGAN restoration
TEXTURE_FUSE_WEIGHT_POWER = 2.0
DETAIL_FUSE_WEIGHT_POWER = 1.6
# near winner-take-all for the detail high band: cross-photo misalignment
# averages crisp albedo detail (stubble, brows) into mush otherwise
DETAIL_FUSE_HIGH_POWER = 6.0


def _restore_if_small(img: np.ndarray, lms: np.ndarray, mode: str,
                      model_path: str | None):
    """Optionally GFPGAN-restore a small face and re-detect landmarks.

    Small scraped headshots carry enough signal for the shape fit but almost
    none for the detail texture; restoration hallucinates identity-preserving
    detail. Returns (img, lms, restored)."""
    if mode == "off":
        return img, lms, False
    eye_d = float(np.linalg.norm(lms[468] - lms[473]))
    if mode == "auto" and eye_d >= RESTORE_EYE_D:
        return img, lms, False
    if not restore.available(model_path):
        return img, lms, False
    out = restore.restore_image(img, lms, eye_d, model_path=model_path)
    if out is None:
        return img, lms, False
    try:
        lms2 = detect(out)
    except Exception:
        return img, lms, False
    return out, lms2, True


def identity_prior(images: list[np.ndarray], landmarks: list[np.ndarray],
                   id_model: str | None, use_render_retrieval: bool,
                   render_index: str | None, render_top_k: int,
                   render_geom_weight: float, render_temperature: float,
                   render_rerank_weight: float, render_min_matches: int,
                   use_retrieval: bool,
                   retrieval_index: str | None, retrieval_top_k: int,
                   retrieval_geom_weight: float, retrieval_temperature: float,
                   use_photofit: bool,
                   photofit_model: str | None, use_emb2shape: bool,
                   emb_model: str | None, weights: list[float] | None = None):
    """Mean ArcFace embedding plus the best available direct coefficient prior.

    CUFP photo retrieval generates candidates. When a render-domain index is
    available, those candidates are reranked inside that candidate set only.
    Photofit/emb2shape are direct-regression fallbacks when no index is local.
    Returns (photo_emb, start_c, prior_source, prior_strength).
    """
    if not identity.available(id_model):
        return None, None, "none", 0.0
    photo_emb = None
    start_c = None
    prior_source = "none"
    prior_strength = 0.0
    render_rerank_index = (
        render_index
        if use_render_retrieval and render_retrieval.available(render_index)
        else None
    )
    if start_c is None and use_retrieval and retrieval.available(retrieval_index):
        pred = retrieval.predict_photos(
            images, landmarks, retrieval_index, id_model, weights=weights,
            top_k=retrieval_top_k, geom_weight=retrieval_geom_weight,
            temperature=retrieval_temperature,
            render_index_path=render_rerank_index,
            render_top_n=render_top_k,
            render_weight=render_rerank_weight,
            render_geom_weight=render_geom_weight,
            render_min_matches=render_min_matches)
        if pred is not None:
            start_c = pred.shape
            prior_source = "cufp_render_rerank" if pred.reranked else "cufp_retrieval"
            prior_strength = pred.confidence
            print("retrieval:",
                  f"conf={pred.confidence:.2f}",
                  f"render_rerank={int(pred.reranked)}",
                  f"render_matches={pred.render_matches}",
                  " ".join(
                      f"{hit.rank}:{hit.name}:score={hit.score:.3f}"
                      f":emb={hit.emb_score:.3f}"
                      f":geom={hit.geom_score:.3f}"
                      f"{f':rscore={hit.render_score:.3f}' if hit.render_score is not None else ''}"
                      f":w={hit.weight:.2f}"
                      for hit in pred.hits[:5]
                  ))
    if start_c is None and use_photofit and photofit.available(photofit_model):
        pred = photofit.predict_photos(
            images, landmarks, photofit_model, id_model, weights=weights)
        if pred is not None:
            photo_emb = pred.embedding
            start_c = pred.shape
            prior_source = "photofit"
            prior_strength = 1.0
    if photo_emb is None:
        photo_emb = identity.photos_embedding(
            images, landmarks, id_model, weights=weights)
    if photo_emb is None:
        return None, None, "none", 0.0
    if use_emb2shape and emb2shape.available(emb_model):
        pred = emb2shape.load(str(emb_model) if emb_model else None).predict(photo_emb)
        if start_c is None:
            start_c = np.concatenate([pred.sym_shape, pred.asym_shape])
            prior_source = "emb2shape"
            prior_strength = 1.0
    return photo_emb, start_c, prior_source, prior_strength


# mid-face skin points: cheeks + nose sides, avoiding eyes/brows/lips/hairline
CHEEK_LMS = [50, 205, 425, 280, 352, 123, 118, 347, 330, 101]


def _mean_skin(img: np.ndarray, lms: np.ndarray, ids: list[int],
               r: int = 6) -> np.ndarray | None:
    h, w = img.shape[:2]
    vals = []
    for i in ids:
        x, y = int(lms[i, 0]), int(lms[i, 1])
        patch = img[max(0, y - r):min(h, y + r),
                    max(0, x - r):min(w, x + r)].reshape(-1, 3)
        if len(patch):
            vals.append(patch.mean(0))
    return np.mean(vals, 0) if vals else None


def skin_tone_match(basis, c_shape: np.ndarray, tex_c: np.ndarray,
                    D: np.ndarray, photo_img: np.ndarray, photo_lms: np.ndarray,
                    lo: float = 0.9, hi: float = 1.35):
    """White-balance the face to the photo's skin tone.

    The detail map multiplies the texture at render time (tex*detail/64), so a
    per-channel scale of the detail map is exactly a global gain on the final
    face color. We render once, measure the rendered vs photo cheek tone, and
    bake the correction into the detail map. Returns (D, k_rgb|None)."""
    from .fgformat import FgFile
    from .landmarks import detect
    from .render import render

    target = _mean_skin(photo_img, photo_lms, CHEEK_LMS)
    if target is None:
        return D, None
    buf = io.BytesIO()
    Image.fromarray(D).save(buf, "JPEG", quality=90)
    fg = FgFile(sym_shape=c_shape[:basis.n_sym],
                asym_shape=c_shape[basis.n_sym:],
                sym_tex=tex_c, asym_tex=np.zeros(0),
                detail_jpeg=buf.getvalue())
    img, _ = render(fg, size=384, aa=1)
    try:
        cur = _mean_skin(img, detect(img), CHEEK_LMS)
    except Exception:
        return D, None
    if cur is None:
        return D, None
    k = np.clip(target / np.maximum(cur, 1.0), lo, hi)
    return np.clip(D.astype(np.float32) * k, 0, 255).astype(np.uint8), k


def _surface_point_detail_px(basis, name: str, size: int) -> np.ndarray | None:
    if not basis.tri.surface_points or name not in basis.tri.surface_points:
        return None
    fidx, bary = basis.tri.surface_points[name]
    vidx = basis.tris[fidx]
    uv = np.asarray(bary, np.float32) @ basis.vert_uv[vidx]
    duv = basis.fim_lookup(uv[None])[0]
    if (duv < 0).any():
        return None
    return np.array([duv[0] * size, (1 - duv[1]) * size], np.float32)


def eye_ring_detail(basis, proj2d: np.ndarray, lms: np.ndarray, size: int,
                    lens_scale: float = 1.03, aspect: float = 0.62,
                    n: int = 48) -> list[np.ndarray]:
    """Eyeglass-frame rings in detail-texture pixel coords, one per eye.

    Map photo-space eye rings through the containing projected mesh triangle.
    The older nearest-vertex mapping could jump across UV seams and draw a
    glasses arm through the cheek/ear.
    """
    dpx, dvalid = detail_px(basis, size)
    src_tri = proj2d[basis.tris].astype(np.float32)
    dst_tri = dpx[basis.tris].astype(np.float32)
    tri_ok = dvalid[basis.tris].all(1)
    src_tri = src_tri[tri_ok]
    dst_tri = dst_tri[tri_ok]
    if len(src_tri) == 0:
        return []

    def map_point(p: np.ndarray) -> np.ndarray | None:
        a = src_tri[:, 0]
        b = src_tri[:, 1]
        c = src_tri[:, 2]
        den = ((b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0])
               + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1]))
        good = np.abs(den) > 1e-6
        u = np.full(len(src_tri), np.nan, np.float32)
        v = np.full(len(src_tri), np.nan, np.float32)
        u[good] = (
            (b[good, 1] - c[good, 1]) * (p[0] - c[good, 0])
            + (c[good, 0] - b[good, 0]) * (p[1] - c[good, 1])
        ) / den[good]
        v[good] = (
            (c[good, 1] - a[good, 1]) * (p[0] - c[good, 0])
            + (a[good, 0] - c[good, 0]) * (p[1] - c[good, 1])
        ) / den[good]
        w = 1.0 - u - v
        inside = good & (u >= -0.025) & (v >= -0.025) & (w >= -0.025)
        if not inside.any():
            return None
        cent = src_tri[inside].mean(1)
        pick_rel = int(np.argmin(((cent - p) ** 2).sum(1)))
        idx = np.nonzero(inside)[0][pick_rel]
        bary = np.array([u[idx], v[idx], w[idx]], np.float32)
        return bary @ dst_tri[idx]

    rings: list[np.ndarray] = []
    sides = {"R": (468, 133, 33), "L": (473, 362, 263)}
    for centre, inner, outer in sides.values():
        c, i, o = lms[centre], lms[inner], lms[outer]
        vec = o - i
        width = float(np.linalg.norm(vec))
        if width < 3:
            continue
        a = 0.72 * width * lens_scale
        b = a * aspect
        ang = np.arctan2(vec[1], vec[0])
        rot = np.array([[np.cos(ang), -np.sin(ang)],
                        [np.sin(ang), np.cos(ang)]])
        pts = []
        for th in np.linspace(0, 2 * np.pi, n, endpoint=False):
            p = c + rot @ np.array([a * np.cos(th), b * np.sin(th)], np.float32)
            q = map_point(p)
            if q is not None and (q >= 0).all() and (q < size).all():
                pts.append(q)
        if len(pts) < max(10, n // 3):
            continue
        ring = np.asarray(pts, np.float32)
        if np.ptp(ring[:, 0]) > 0.28 * size or np.ptp(ring[:, 1]) > 0.28 * size:
            continue
        rings.append(ring)
    return rings


def source_frame_detail_paths(basis, proj2d: np.ndarray, img: np.ndarray,
                              lms: np.ndarray, mask: np.ndarray,
                              size: int) -> tuple[list[np.ndarray], np.ndarray]:
    """Map source-photo glasses frame contours into detail-texture space."""
    signal = glasses.frame_overlay(img, lms, mask)
    frame = (signal > 0.18).astype(np.uint8)
    cnts, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return [], glasses.frame_detail_color(img, lms, mask, 0.55)

    dpx, dvalid = detail_px(basis, size)
    src_tri = proj2d[basis.tris].astype(np.float32)
    dst_tri = dpx[basis.tris].astype(np.float32)
    tri_ok = dvalid[basis.tris].all(1)
    src_tri = src_tri[tri_ok]
    dst_tri = dst_tri[tri_ok]
    if len(src_tri) == 0:
        return [], glasses.frame_detail_color(img, lms, mask, 0.55)

    def map_point(p: np.ndarray) -> np.ndarray | None:
        a = src_tri[:, 0]
        b = src_tri[:, 1]
        c = src_tri[:, 2]
        den = ((b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0])
               + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1]))
        good = np.abs(den) > 1e-6
        u = np.full(len(src_tri), np.nan, np.float32)
        v = np.full(len(src_tri), np.nan, np.float32)
        u[good] = (
            (b[good, 1] - c[good, 1]) * (p[0] - c[good, 0])
            + (c[good, 0] - b[good, 0]) * (p[1] - c[good, 1])
        ) / den[good]
        v[good] = (
            (c[good, 1] - a[good, 1]) * (p[0] - c[good, 0])
            + (a[good, 0] - c[good, 0]) * (p[1] - c[good, 1])
        ) / den[good]
        w = 1.0 - u - v
        inside = good & (u >= -0.025) & (v >= -0.025) & (w >= -0.025)
        if not inside.any():
            return None
        cent = src_tri[inside].mean(1)
        pick_rel = int(np.argmin(((cent - p) ** 2).sum(1)))
        idx = np.nonzero(inside)[0][pick_rel]
        bary = np.array([u[idx], v[idx], w[idx]], np.float32)
        q = bary @ dst_tri[idx]
        if (q < 0).any() or (q >= size).any():
            return None
        return q

    paths: list[np.ndarray] = []
    for cnt in sorted(cnts, key=cv2.contourArea, reverse=True)[:10]:
        pts = cnt.reshape(-1, 2).astype(np.float32)
        if len(pts) < 10:
            continue
        step = max(1, len(pts) // 96)
        mapped = [map_point(p) for p in pts[::step]]
        mapped = [p for p in mapped if p is not None]
        if len(mapped) < 8:
            continue
        path = np.asarray(mapped, np.float32)
        if np.ptp(path[:, 0]) > 0.62 * size or np.ptp(path[:, 1]) > 0.32 * size:
            continue
        paths.append(path)
    color = glasses.frame_detail_color(img, lms, mask, 0.72)
    return paths, color


def image_paths(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(p for p in root.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def photo_metrics(img: np.ndarray, lms: np.ndarray) -> dict:
    oval = lms[FACE_OVAL]
    face_w = float(np.ptp(oval[:, 0]))
    face_h = float(np.ptp(oval[:, 1]))
    eye_d = float(np.linalg.norm(lms[468] - lms[473]))
    yaw_proxy = float((lms[1, 0] - 0.5 * (lms[234, 0] + lms[454, 0]))
                      / max(face_w, 1.0))
    size_score = np.clip(eye_d / 105.0, 0.55, 1.25)
    front_score = np.clip(1.0 - abs(yaw_proxy) / 0.48, 0.25, 1.0)
    mouth_open = float(abs(lms[13, 1] - lms[14, 1]) / max(eye_d, 1.0))
    mouth_penalty = float(np.clip((mouth_open - 0.045) / 0.18, 0.0, 1.0))

    yy = np.indices(img.shape[:2])[0]
    mask = face_mask(img, lms)
    brow_y = float(np.median(lms[[70, 63, 105, 66, 107, 336, 296, 334, 293, 300], 1]))
    face_mid_y = float(np.percentile(oval[:, 1], 60))
    top = mask & (yy >= float(oval[:, 1].min())) & (yy < brow_y)
    mid = mask & (yy >= brow_y) & (yy < face_mid_y)
    lum = img @ np.array([0.299, 0.587, 0.114])
    top_lum = float(lum[top].mean()) if top.any() else 0.0
    mid_lum = float(lum[mid].mean()) if mid.any() else 1.0
    top_mid_lum = top_lum / max(mid_lum, 1.0)
    rgb = img.astype(np.float32)
    chroma = (rgb.max(axis=2) - rgb.min(axis=2)) / np.maximum(rgb.max(axis=2), 1.0)
    top_chroma = float(chroma[top].mean()) if top.any() else 0.0
    shadow_penalty = max(
        float(np.clip((0.72 - top_mid_lum) / 0.42, 0.0, 1.0)),
        float(np.clip((top_chroma - 0.45) / 0.35, 0.0, 1.0)),
    )
    base_score = float(size_score * front_score)
    texture_score = float(base_score
                          * (1.0 - 0.75 * shadow_penalty)
                          * (1.0 - 0.85 * mouth_penalty))
    shape_score = float(base_score * (1.0 - 0.75 * mouth_penalty))
    return {
        "w": img.shape[1],
        "h": img.shape[0],
        "face_w": face_w,
        "face_h": face_h,
        "eye_d": eye_d,
        "yaw_proxy": yaw_proxy,
        "weight": base_score,
        "shape_score": shape_score,
        "top_mid_lum": float(top_mid_lum),
        "top_chroma": float(top_chroma),
        "shadow_penalty": float(shadow_penalty),
        "mouth_open": mouth_open,
        "mouth_penalty": mouth_penalty,
        "texture_score": texture_score,
    }


def choose_texture_index(paths: list[Path], metrics: list[dict],
                         requested: str | None) -> int:
    if requested:
        req = requested.lower()
        for i, p in enumerate(paths):
            if req in str(p).lower():
                return i
        raise ValueError(f"texture photo not found: {requested}")
    near_front = [
        (i, m["texture_score"]) for i, m in enumerate(metrics)
        if abs(m["yaw_proxy"]) < 0.12 and m["eye_d"] >= 70
    ]
    if near_front:
        return max(near_front, key=lambda x: x[1])[0]
    return int(np.argmax([m["texture_score"] for m in metrics]))


def texture_weights(metrics: list[dict], anchor_idx: int,
                    requested: str | None) -> list[float]:
    weights = [max(float(m["texture_score"]), 0.01) for m in metrics]
    if requested:
        weights[anchor_idx] *= 2.0
    return weights


def _merge_masks(*masks: np.ndarray | None) -> np.ndarray | None:
    valid = [mask for mask in masks if mask is not None and mask.any()]
    if not valid:
        return None
    out = np.zeros(valid[0].shape, bool)
    for mask in valid:
        out |= mask
    return out


def build_texture_sample(basis, img: np.ndarray, lms: np.ndarray, pose,
                         shape_c: np.ndarray, target_lum: float, anchor: int,
                         args: argparse.Namespace,
                         glasses_mask: np.ndarray | None = None) -> dict:
    source_img = img
    photo_valid = face_mask(img, lms)
    if glasses_mask is not None and args.glasses_method in ("suppress", "parametric", "mesh"):
        img = glasses.suppress_photo(img, lms, glasses_mask)
    # Full (wide-clip) gain toward the player's cross-photo tone anchor: a
    # dimly lit photo of a light-skinned player must not bake its lighting
    # into skin tone. The anchor itself was chosen from the best-exposed
    # photo, so genuinely dark skin still anchors dark.
    gain = float(np.clip(target_lum / max(skin_luminance(img, lms), 1.0),
                         0.5, 3.0))
    img = np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    s, R, t = pose
    proj2d, cam = project(basis, shape_c, s, R, t)
    delight_mode = getattr(args, "delight", None) or "sh"
    shading = None
    delight_r2 = 0.0
    if delight_mode == "sh":
        exclude = feature_exclude_mask(img.shape[:2], lms)
        fit = estimate_shading(img, photo_valid & ~exclude, basis, proj2d,
                               cam, anchor, norm_mask=photo_valid)
        if fit is not None:
            shading, delight_r2 = fit
    if shading is not None:
        # SH removes smooth attached shading; the mirror pass afterwards kills
        # what SH cannot model — hard one-sided cast shadows (nose wedge from
        # stage lighting) — and ends with specular compression. On an already
        # delit image its gains stay near 1, so it is a residual clean-up.
        img = np.clip(img.astype(np.float32) / shading, 0, 255)
        img = illum_correct(img.astype(np.uint8), photo_valid, lms).astype(np.uint8)
        delight_used = "sh"
    elif delight_mode == "off":
        delight_used = "off"
    else:
        img = illum_correct(img, photo_valid, lms).astype(np.uint8)
        delight_used = "mirror" if delight_mode == "mirror" else "mirror_fallback"
    fronts = front_tris(basis, proj2d, anchor, cam, min_cos=0.3)
    photo_uv, cov = warp_tris(
        img.astype(np.float32),
        tri_coords(basis, proj2d),
        uv_px(basis, 256)[basis.tris],
        256,
        fronts,
    )
    # Skin texture must not sample accessories: they skew skin coefficients and
    # skin-tone match. Suppress mode inpaints first, then still excludes the
    # detected accessory pixels from the low-rank texture fit.
    tex_exclude = (None if args.glasses_method in ("suppress", "parametric", "mesh")
                   else _merge_masks(glasses_mask))
    tex_valid = photo_valid & ~tex_exclude if tex_exclude is not None else photo_valid
    vm = np.repeat(tex_valid[..., None].astype(np.float32), 3, 2)
    vm_uv, _ = warp_tris(vm, tri_coords(basis, proj2d),
                         uv_px(basis, 256)[basis.tris], 256, fronts)
    cov &= vm_uv[..., 0] > 0.9
    frame = None
    frame_color = None
    if glasses_mask is not None and args.glasses_method == "frame":
        frame = glasses.frame_overlay(img, lms, glasses_mask)
        frame_color = glasses.frame_detail_color(
            img, lms, glasses_mask, args.glasses_frame_strength
        )
    return {
        "img": img,
        "source_img": source_img,
        "lms": lms,
        "photo_valid": photo_valid,
        "glasses": glasses_mask,
        "frame": frame,
        "frame_color": frame_color,
        "proj2d": proj2d,
        "cam": cam,
        "photo_uv": photo_uv,
        "uv_cov": cov,
        "gain": gain,
        "delight": delight_used,
        "delight_r2": delight_r2,
        "shading": shading,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build an .fg from multiple photos.")
    p.add_argument("photos", help="Photo folder or single image.")
    p.add_argument("out")
    p.add_argument("--texture-photo",
                   help="Substring/path of the photo to force or boost for texture/detail.")
    p.add_argument("--texture-mode", choices=("fuse", "best"), default="fuse",
                   help="Fuse all usable photos, or use the single best texture photo.")
    p.add_argument("--max-yaw", type=float, default=0.2,
                   help="Skip photos whose absolute yaw proxy is above this value.")
    p.add_argument("--shape-lam", type=float, default=0.005)
    p.add_argument("--asym-lam", type=float, default=0.3)
    p.add_argument("--dense-weight", type=float, default=0.45)
    p.add_argument("--tex-lam", type=float, default=5.0)
    p.add_argument("--tex-erode", type=int, default=1)
    p.add_argument("--exposure-lo", type=float, default=0.9)
    p.add_argument("--exposure-hi", type=float, default=1.45,
                   help="Clamp scalar exposure gain; lower preserves naturally dark skin tones.")
    p.add_argument("--delight", choices=("sh", "mirror", "off"), default="sh",
                   help="Photo lighting removal before texture/detail baking. "
                        "'sh': geometry-based spherical-harmonics delighting, "
                        "falling back to 'mirror' when the fit is unreliable. "
                        "'mirror': older midline illumination symmetrization.")
    p.add_argument("--detail-size", type=int, default=512)
    p.add_argument("--detail-strength", type=float, default=1.15)
    p.add_argument("--detail-chroma-strength", type=float, default=None,
                   help="Keep only this much color detail; lower compresses "
                        "cleaner. Default 0.2 when every photo was SH-delit, "
                        "else 0.08.")
    p.add_argument("--detail-edge-strength", type=float, default=1.1,
                   help="Unsharp amount for luma detail before JPEG encoding.")
    p.add_argument("--detail-flat-neutralize", type=float, default=0.25,
                   help="Blend low-information skin areas toward neutral detail.")
    p.add_argument("--detail-shadow-neutralize", type=float, default=None,
                   help="Suppress broad baked-in lighting shadows in the detail "
                        "map. Default 0.55 when every photo was SH-delit "
                        "(cast shadows still need it), else 0.8.")
    p.add_argument("--detail-dark-keep", type=float, default=None,
                   help="How much sub-neutral (dark) detail survives the dark "
                        "compressor (0..1). Default 0.6 when every photo was "
                        "SH-delit (darkness is albedo then), else 0.")
    p.add_argument("--detail-highlight-neutralize", type=float, default=0.0,
                   help="Suppress broad baked-in highlights in the detail map.")
    p.add_argument("--detail-jpeg-quality", type=int, default=90)
    p.add_argument("--eye-detail-strength", type=float, default=1.0)
    p.add_argument("--likeness-detail", type=float, default=0.65,
                   help="Preserve photo detail around brows, eyes, nose, mouth, "
                        "and chin after clean-up. 0 disables.")
    p.add_argument("--likeness-detail-gain", type=float, default=1.2,
                   help="Contrast gain for preserved likeness feature detail.")
    p.add_argument("--detail-min-cos", type=float, default=0.18)
    p.add_argument("--shape-gain", type=float, default=1.0,
                   help="Scale applied to the ridge-fit sym shape before capping.")
    p.add_argument("--shape-cap", type=float, default=3.5,
                   help="Per-mode absolute clip on sym shape coefficients.")
    p.add_argument("--shape-norm-cap", type=float, default=9.5,
                   help="Cap on the sym shape norm (official .fg median ~7.8).")
    p.add_argument("--restore", choices=("auto", "off", "force"), default="off",
                   help="GFPGAN restoration of small/low-res faces.")
    p.add_argument("--restore-model",
                   help="Path to the GFPGAN ONNX model (defaults to models/).")
    p.add_argument("--id-refine", type=int, default=0,
                   help="ArcFace identity refine iterations. 0 disables; "
                        "positive values enable the experimental slow path.")
    p.add_argument("--id-model",
                   help="Path to the ArcFace ONNX model (defaults to models/).")
    p.add_argument("--render-retrieval", choices=("auto", "off"), default="off",
                   help="Use the pre-rendered FaceGen nearest-neighbour prior.")
    p.add_argument("--render-index",
                   help="Path to fg_render_identity_index.npz (defaults to models/).")
    p.add_argument("--render-top-k", type=int, default=16)
    p.add_argument("--render-geom-weight", type=float, default=0.12)
    p.add_argument("--render-temperature", type=float, default=0.06)
    p.add_argument("--render-rerank-weight", type=float, default=0.05)
    p.add_argument("--render-min-matches", type=int, default=3)
    p.add_argument("--retrieval", choices=("auto", "off"), default="auto",
                   help="Use a CUFP nearest-neighbour FaceGen prior when the "
                        "index is present in models/.")
    p.add_argument("--retrieval-index",
                   help="Path to cufp_identity_index.npz (defaults to models/).")
    p.add_argument("--retrieval-top-k", type=int, default=12)
    p.add_argument("--retrieval-geom-weight", type=float, default=0.18)
    p.add_argument("--retrieval-temperature", type=float, default=0.055)
    p.add_argument("--photofit", choices=("auto", "off"), default="auto",
                   help="Use the learned direct photo-feature -> FaceGen prior "
                        "(fallback when no retrieval index matches).")
    p.add_argument("--photofit-model",
                   help="Path to the photofit .npz (defaults to models/).")
    p.add_argument("--emb2shape", choices=("auto", "off"), default="off",
                   help="Use the learned embedding->shape prior as refine start. "
                        "Trained in the render domain; underperforms the landmark "
                        "fit on real photos, so off by default.")
    p.add_argument("--emb2shape-model",
                   help="Path to the emb2shape .npz (defaults to models/).")
    p.add_argument("--modeller-fit", choices=("auto", "off"), default="auto",
                   help="Refit shape coefficients with the identity prior inside "
                        "the landmark solve.")
    p.add_argument("--modeller-iters", type=int, default=4)
    p.add_argument("--modeller-prior-lam", type=float, default=0.04)
    p.add_argument("--modeller-r-max", type=float, default=0.75)
    p.add_argument("--refine-size", type=int, default=128,
                   help="Render size used to score ArcFace similarity in refine.")
    p.add_argument("--refine-r-max", type=float, default=2.2,
                   help="Trust-region rms radius of refine around the landmark fit.")
    p.add_argument("--skin-tone-match", choices=("on", "off"), default="on",
                   help="White-balance the rendered face to the photo skin tone.")
    p.add_argument("--glasses", choices=("auto", "off", "on"), default="auto",
                   help="Detect eyeglasses with the optional BiSeNet parser. "
                        "'auto' activates only when the parser model is present.")
    p.add_argument("--glasses-model",
                   help="Path to the BiSeNet face-parser ONNX (defaults to models/).")
    p.add_argument("--glasses-method",
                   choices=("auto", "mesh", "parametric", "frame", "draw", "protect", "suppress"),
                   default="auto",
                   help="'auto': mesh when FaceGen accessory assets are "
                        "available locally, else parametric. "
                        "'mesh': bake a local FaceGen 3D glasses accessory into "
                        "the detail map. 'parametric': reconstruct a clean vector frame from "
                        "detected glasses placement/color. 'frame': extract "
                        "the source frame contour/color and bake it into the "
                        "detail map. 'draw': fit + draw a generic clean frame "
                        "at the detected lenses. "
                        "'protect': keep the whole warped region. "
                        "'suppress': remove detected glasses from the baked "
                        "face detail.")
    p.add_argument("--glasses-strength", type=float, default=1.5,
                   help="protect method: contrast gain on the kept glasses "
                        "detail about the neutral midpoint (1.0 = as-is).")
    p.add_argument("--glasses-suppress-strength", type=float, default=0.72,
                   help="suppress method: blend detected glasses detail toward "
                        "neutral 64 (0..1; higher removes more).")
    p.add_argument("--glasses-frame-strength", type=float, default=0.45,
                   help="draw/frame methods: how dark the frame rim is "
                        "(0..1; higher = darker/bolder).")
    p.add_argument("--glasses-mesh-assets",
                   help="FaceGen Accessories folder, a .gltf/.fbx/.obj/.dae "
                        "file, a folder containing .gltf/.fbx/.obj/.dae, or "
                        "a .zip containing .gltf/.fbx/.obj/.dae, or a named "
                        "local template alias. FaceGen "
                        "Glasses.tri/.egm/.tga are used for stock glasses; "
                        "custom assets are used as build-time meshes. "
                        "FBX/OBJ/DAE are converted through Blender and cached.")
    p.add_argument("--glasses-mesh-opacity", type=float, default=0.84,
                   help="mesh method: opacity of the baked accessory frame.")
    p.add_argument("--glasses-mesh-scale-x", type=float,
                   help="mesh method: horizontal accessory scale override.")
    p.add_argument("--glasses-mesh-scale-y", type=float,
                   help="mesh method: vertical accessory scale override.")
    p.add_argument("--glasses-mesh-offset-y", type=float, default=0.0,
                   help="mesh method: vertical model-space fine-tune after eye alignment.")
    p.add_argument("--glasses-style",
                   choices=("auto", "sports_goggle", "rectangular", "round", "oval"),
                   default="auto",
                   help="parametric method: template style. auto infers from "
                        "source mask/color.")
    p.add_argument("--glasses-color",
                   choices=("auto", "red", "black", "brown", "blue", "silver"),
                   default="auto",
                   help="parametric method: template frame color.")
    p.add_argument("--glasses-rim-width", type=float,
                   help="parametric method: rim thickness multiplier.")
    p.add_argument("--glasses-lens-width", type=float,
                   help="parametric method: lens width multiplier.")
    p.add_argument("--glasses-lens-height", type=float,
                   help="parametric method: lens height multiplier.")
    p.add_argument("--glasses-bridge",
                   choices=("auto", "thin", "thick"), default="auto",
                   help="parametric method: bridge thickness style.")
    p.add_argument("--debug-dir",
                   help="Optional directory for intermediate texture/debug images.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.glasses_method == "auto":
        args.glasses_method = (
            "mesh"
            if glasses_mesh.available(args.glasses_mesh_assets)
            else "parametric"
        )
        print(f"glasses method: auto -> {args.glasses_method}")
    paths = image_paths(Path(args.photos))
    if not paths:
        raise SystemExit(f"no images found: {args.photos}")
    debug_dir = Path(args.debug_dir) if args.debug_dir else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    basis = get_basis()
    images: list[np.ndarray] = []
    landmarks: list[np.ndarray] = []
    metrics: list[dict] = []
    usable_paths: list[Path] = []
    glasses_masks: list[np.ndarray | None] = []
    glasses_ready = args.glasses != "off" and glasses.available(args.glasses_model)
    if args.glasses != "off" and not glasses_ready:
        print("glasses: parser model unavailable, skipping "
              "(run scripts/download_restore_model.py)")

    for p in paths:
        img = np.asarray(Image.open(p).convert("RGB"))
        try:
            lms = detect(img)
        except Exception as exc:
            print("skip", p.name, "detect_failed", exc)
            continue
        img, lms, restored = _restore_if_small(img, lms, args.restore,
                                                args.restore_model)
        if restored:
            print("restore", p.name, "gfpgan")
        m = photo_metrics(img, lms)
        if args.max_yaw is not None and abs(m["yaw_proxy"]) > args.max_yaw:
            print("skip", p.name, f"yaw={m['yaw_proxy']:.3f}",
                  f"max_yaw={args.max_yaw:.3f}")
            continue
        gmask = None
        if glasses_ready:
            gres = glasses.segment(img, lms, args.glasses, args.glasses_model)
            if gres.any:
                gmask = gres.mask
                print("glasses", p.name, f"conf={gres.confidence:.2f}",
                      f"area_px={int(gmask.sum())}")
        images.append(img)
        landmarks.append(lms)
        metrics.append(m)
        usable_paths.append(p)
        glasses_masks.append(gmask)
        print(
            "photo", len(usable_paths) - 1, p.name,
            f"size={m['w']}x{m['h']}",
            f"eye_d={m['eye_d']:.1f}",
            f"yaw={m['yaw_proxy']:.3f}",
            f"weight={m['weight']:.3f}",
            f"shape_score={m['shape_score']:.3f}",
            f"tex_score={m['texture_score']:.3f}",
            f"shadow={m['shadow_penalty']:.2f}",
            f"mouth={m['mouth_open']:.3f}",
        )

    if not usable_paths:
        raise SystemExit("no usable faces detected")

    tex_idx = choose_texture_index(usable_paths, metrics, args.texture_photo)
    print("texture photo:", tex_idx, usable_paths[tex_idx].name)

    weights = [m["shape_score"] for m in metrics]
    c, poses, info = fit_shape_multi_dense(
        basis,
        landmarks,
        lam_sym=args.shape_lam,
        lam_asym=args.asym_lam,
        dense_weight=args.dense_weight,
        photo_weights=weights,
    )
    print(
        f"multi shape: |c|={np.linalg.norm(c):.2f}",
        f"range=[{c.min():.2f},{c.max():.2f}]",
        f"mean_resid={info['mean_resid']:.2f}px",
        "per_photo=" + ",".join(f"{r:.2f}" for r in info["per_photo_resid"]),
    )

    # Keep the fitted shape inside the FaceGen coefficient distribution of real
    # OOTP faces (official sym-norm median ~7.8). On noisy multi-photo input the
    # ridge fit can blow a single mode out to a caricature; per-mode clipping
    # plus a norm cap tames that without touching well-behaved fits.
    c0 = c.copy()
    sym = c0[:basis.n_sym] * args.shape_gain
    sym = np.clip(sym, -args.shape_cap, args.shape_cap)
    sym_norm = float(np.linalg.norm(sym))
    if sym_norm > args.shape_norm_cap:
        sym *= args.shape_norm_cap / sym_norm
    c0[:basis.n_sym] = sym
    print(f"shape norm: fit={np.linalg.norm(c[:basis.n_sym]):.2f} "
          f"-> capped={np.linalg.norm(c0[:basis.n_sym]):.2f}")

    id_refine = max(int(args.id_refine), 0)
    want_id = (
        id_refine > 0 or args.render_retrieval != "off"
        or args.retrieval != "off"
        or args.photofit != "off" or args.emb2shape != "off"
        or args.modeller_fit != "off"
    )
    photo_emb, start_c, prior_source, prior_strength = (None, None, "none", 0.0)
    if want_id:
        try:
            id_weights = [float(m["texture_score"]) for m in metrics]
            photo_emb, start_c, prior_source, prior_strength = identity_prior(
                images, landmarks, args.id_model,
                args.render_retrieval != "off", args.render_index,
                args.render_top_k, args.render_geom_weight,
                args.render_temperature, args.render_rerank_weight,
                args.render_min_matches,
                args.retrieval != "off", args.retrieval_index,
                args.retrieval_top_k, args.retrieval_geom_weight,
                args.retrieval_temperature,
                args.photofit != "off", args.photofit_model,
                args.emb2shape != "off", args.emb2shape_model,
                weights=id_weights)
            if start_c is not None:
                # Near-frontal photos do not constrain asymmetry well, so anchor
                # asym modes to the landmark fit for all direct priors.
                start_c[basis.n_sym:] = c0[basis.n_sym:]
            print("identity:",
                  "emb=yes" if photo_emb is not None else "emb=no",
                  f"prior={prior_source}",
                  f"strength={prior_strength:.2f}",
                  f"id_refine={id_refine}")
        except Exception as exc:
            photo_emb, start_c, prior_source, prior_strength = None, None, "none", 0.0
            id_refine = 0
            print("identity: disabled", type(exc).__name__, exc)

    modeller_applied = False
    if args.modeller_fit != "off" and start_c is not None and prior_strength > 0.0:
        try:
            mf = modeller_fit.fit_shape_direct_prior(
                basis,
                landmarks,
                anchor_coeffs=c0,
                prior_coeffs=start_c,
                prior_strength=prior_strength,
                lam_sym=args.shape_lam,
                lam_asym=args.asym_lam,
                dense_weight=args.dense_weight,
                photo_weights=weights,
                n_iter=args.modeller_iters,
                prior_lam_sym=args.modeller_prior_lam,
                prior_lam_asym=args.modeller_prior_lam * 0.35,
                r_max=args.modeller_r_max,
                shape_cap=args.shape_cap,
                shape_norm_cap=args.shape_norm_cap,
            )
            c0 = mf.coeffs
            c = c0.copy()
            poses = mf.poses
            modeller_applied = True
            print("modeller fit:",
                  f"prior={prior_source}",
                  f"strength={mf.prior_strength:.2f}",
                  f"|dc|={mf.delta_norm:.2f}",
                  f"|prior-c0|={mf.prior_norm:.2f}",
                  f"mean_resid={mf.mean_resid:.2f}px",
                  "per_photo=" + ",".join(
                      f"{r:.2f}" for r in mf.per_photo_resid
                  ))
        except Exception as exc:
            print("modeller fit: skipped", type(exc).__name__, exc)

    fimv = basis.fim[..., 0] >= 0
    ref_lum = float((basis.mean_tex[fimv] @ np.array([0.299, 0.587, 0.114])).mean())
    anchor = basis.front_anchor_tri

    # Cross-photo skin-tone consensus. The best-exposed photo (needed gain
    # closest to 1 against the basis mean) defines the player's true tone;
    # only that anchor gain is clamped by exposure_lo/hi, so dim photos of a
    # light-skinned player get fully lifted while genuinely dark skin, whose
    # well-lit photos are still dark, keeps its tone.
    lums = [skin_luminance(img, lms) for img, lms in zip(images, landmarks)]
    ideals = [1.12 * ref_lum / max(l, 1.0) for l in lums]
    tone_idx = min(range(len(lums)), key=lambda i: abs(float(np.log(ideals[i]))))
    # Adaptive clamp: a dark face in a dark frame is dim lighting, so allow a
    # stronger lift; a dark face in a well-exposed frame is genuinely dark
    # skin and keeps the conservative clamp.
    lw = np.array([0.299, 0.587, 0.114], np.float32)
    # p95 as the exposure probe: a properly exposed frame almost always holds
    # something near-white (jersey, background); a frame whose brightest
    # content is dark was shot underexposed and may be lifted safely.
    scene_p95 = float(np.percentile(images[tone_idx].astype(np.float32) @ lw, 95))
    # strict threshold: only clearly underexposed frames qualify, so a dark
    # face on a merely dark background (true dark skin) keeps the tight clamp
    dim = float(np.clip((150.0 - scene_p95) / 60.0, 0.0, 1.0))
    hi_eff = args.exposure_hi * (1.0 + 0.8 * dim)
    target_lum = lums[tone_idx] * float(
        np.clip(ideals[tone_idx], args.exposure_lo, hi_eff))
    print(
        "tone anchor:", tone_idx, usable_paths[tone_idx].name,
        f"lum={lums[tone_idx]:.1f}",
        f"scene_p95={scene_p95:.1f}",
        f"hi_eff={hi_eff:.2f}",
        f"target_lum={target_lum:.1f}",
        "photo_lums=" + ",".join(f"{l:.0f}" for l in lums),
    )

    print("texture mode:", args.texture_mode)
    tex_weights = texture_weights(metrics, tex_idx, args.texture_photo)
    tone_match_sample = None
    if args.texture_mode == "best":
        sample = build_texture_sample(
            basis, images[tex_idx], landmarks[tex_idx], poses[tex_idx],
            c, target_lum, anchor, args,
            glasses_masks[tex_idx],
        )
        tone_match_sample = sample
        print(f"exposure gain: {float(sample['gain']):.3f}")
        if debug_dir:
            Image.fromarray(sample["img"]).save(debug_dir / "illum.png")
            if sample.get("shading") is not None:
                sh8 = np.clip(sample["shading"] * 127.5, 0, 255).astype(np.uint8)
                Image.fromarray(sh8).save(debug_dir / "shading.png")
        photo_uv = sample["photo_uv"]
        cov = sample["uv_cov"]
        detail_samples = [(sample, tex_weights[tex_idx])]
    else:
        texture_samples = []
        for i, (img_i, lms_i, pose_i, path_i, weight_i) in enumerate(zip(
                images, landmarks, poses, usable_paths, tex_weights)):
            sample = build_texture_sample(
                basis, img_i, lms_i, pose_i, c, target_lum, anchor, args,
                glasses_masks[i],
            )
            sample["index"] = i
            sample["name"] = path_i.name
            sample["weight"] = weight_i
            uv_cov = float(sample["uv_cov"].mean())
            print(
                "texture sample:", i, path_i.name,
                f"gain={float(sample['gain']):.3f}",
                f"uv_cov={uv_cov:.3f}",
                f"weight={float(weight_i):.3f}",
                f"delight={sample['delight']}",
                f"delight_r2={float(sample['delight_r2']):.2f}",
            )
            if debug_dir and i == tex_idx:
                Image.fromarray(sample["img"]).save(debug_dir / "illum.png")
                if sample.get("shading") is not None:
                    sh8 = np.clip(sample["shading"] * 127.5, 0, 255).astype(np.uint8)
                    Image.fromarray(sh8).save(debug_dir / "shading.png")
            if sample["uv_cov"].any():
                texture_samples.append(sample)

        if not texture_samples:
            raise SystemExit("no usable texture coverage")

        tone_match_sample = next(
            (sample for sample in texture_samples if sample.get("index") == tex_idx),
            None,
        )
        if tone_match_sample is None:
            tone_match_sample = max(
                texture_samples,
                key=lambda sample: float(sample.get("weight", 0.0)),
            )

        photo_uv, cov = fuse_maps(
            [sample["photo_uv"] for sample in texture_samples],
            [sample["uv_cov"] for sample in texture_samples],
            [float(sample["weight"]) for sample in texture_samples],
            basis.mean_tex,
            weight_power=TEXTURE_FUSE_WEIGHT_POWER,
        )
        print(
            "texture fusion:",
            f"photos={len(texture_samples)}/{len(usable_paths)}",
            f"coverage={float(cov.mean()):.3f}",
            f"anchor={tex_idx} {usable_paths[tex_idx].name}",
        )
        detail_samples = [
            (sample, float(sample["weight"]))
            for sample in texture_samples
        ]

    # With every photo delit, most remaining detail-map darkness is albedo
    # (stubble, feature lines) worth keeping, so relax the blur-based
    # neutralizers; without delighting keep the old conservative clean-up.
    n_delit = sum(s.get("delight") == "sh" for s, _ in detail_samples)
    delighted = n_delit == len(detail_samples)
    if args.detail_chroma_strength is None:
        args.detail_chroma_strength = 0.20 if delighted else 0.08
    if args.detail_shadow_neutralize is None:
        args.detail_shadow_neutralize = 0.55 if delighted else 0.8
    if args.detail_dark_keep is None:
        args.detail_dark_keep = 0.6 if delighted else 0.0
    print(
        "delight:",
        f"mode={args.delight}",
        f"applied={n_delit}/{len(detail_samples)}",
        f"chroma={args.detail_chroma_strength:.2f}",
        f"shadow_neutralize={args.detail_shadow_neutralize:.2f}",
        f"dark_keep={args.detail_dark_keep:.2f}",
    )

    tex_c = fit_tex_coeffs(basis, photo_uv, cov,
                           lam=args.tex_lam, erode=args.tex_erode)
    print(
        f"tex fit: |c|={np.linalg.norm(tex_c):.2f}",
        f"range=[{tex_c.min():.2f},{tex_c.max():.2f}]",
    )

    detail_maps, detail_covs, detail_weights = [], [], []
    for sample, weight in detail_samples:
        has_protected_glasses = (
            args.glasses_method == "protect" and sample.get("glasses") is not None
        )
        protect = _merge_masks(
            sample.get("glasses") if args.glasses_method == "protect" else None,
        )
        suppress = _merge_masks(
            sample.get("glasses") if args.glasses_method == "suppress" else None,
        )
        protect_gain = args.glasses_strength if has_protected_glasses else 1.0
        dmap, dcov = build_detail(
            basis,
            sample["img"],
            sample["proj2d"],
            tex_c,
            anchor,
            photo_valid=sample["photo_valid"],
            cam=sample["cam"],
            min_cos=args.detail_min_cos,
            size=args.detail_size,
            detail_strength=args.detail_strength,
            chroma_strength=args.detail_chroma_strength,
            edge_strength=args.detail_edge_strength,
            flat_neutralize=args.detail_flat_neutralize,
            shadow_neutralize=args.detail_shadow_neutralize,
            highlight_neutralize=args.detail_highlight_neutralize,
            dark_keep=args.detail_dark_keep,
            neutralize_eyes=True,
            eye_detail_strength=args.eye_detail_strength,
            source_lms=sample["lms"],
            likeness_detail=args.likeness_detail,
            likeness_detail_gain=args.likeness_detail_gain,
            protect=protect,
            protect_gain=protect_gain,
            suppress=suppress,
            suppress_strength=args.glasses_suppress_strength,
            frame=(sample.get("frame")
                   if args.glasses_method == "frame" else None),
            frame_color=(sample.get("frame_color")
                         if args.glasses_method == "frame" else None),
            frame_strength=args.glasses_frame_strength,
        )
        if dcov.any():
            detail_maps.append(dmap)
            detail_covs.append(dcov)
            detail_weights.append(weight)

    if detail_maps:
        D, dvalid = fuse_maps(
            detail_maps,
            detail_covs,
            detail_weights,
            64.0,
            weight_power=DETAIL_FUSE_WEIGHT_POWER,
            high_power=DETAIL_FUSE_HIGH_POWER,
            split_sigma=3.0 * args.detail_size / 256.0,
        )
        D = np.clip(D, 0, 255).astype(np.uint8)
    else:
        D = np.full((args.detail_size, args.detail_size, 3), 64, np.uint8)
        dvalid = np.zeros((args.detail_size, args.detail_size), bool)

    if args.texture_mode == "fuse":
        print(
            "detail fusion:",
            f"photos={len(detail_maps)}/{len(detail_samples)}",
            f"coverage={float(dvalid.mean()):.3f}",
        )
    if args.likeness_detail > 0:
        print(
            "likeness detail:",
            f"strength={args.likeness_detail:.2f}",
            f"gain={args.likeness_detail_gain:.2f}",
        )
    if args.glasses_method == "suppress" and any(m is not None for m in glasses_masks):
        print(
            "glasses suppress:",
            f"strength={args.glasses_suppress_strength:.2f}",
        )
    print("detail coverage:", round(float(dvalid.mean()), 3))

    if args.skin_tone_match == "on" and dvalid.any():
        tone_img = images[tex_idx]
        tone_lms = landmarks[tex_idx]
        if tone_match_sample is not None:
            tone_img = tone_match_sample["img"]
            tone_lms = tone_match_sample["lms"]
        D, k_tone = skin_tone_match(basis, c0, tex_c, D,
                                    tone_img, tone_lms)
        if k_tone is not None:
            print(f"skin tone match: k_rgb={np.round(k_tone, 3).tolist()}")

    mesh_asset_info = glasses_mesh.glasses_template_info(args.glasses_mesh_assets)
    mesh_style = args.glasses_style
    mesh_color = args.glasses_color
    if mesh_asset_info is not None:
        if mesh_style == "auto" and mesh_asset_info.get("style"):
            mesh_style = str(mesh_asset_info["style"])
        if mesh_color == "auto" and mesh_asset_info.get("color"):
            mesh_color = str(mesh_asset_info["color"])

    mesh_forced = (
        args.glasses == "on"
        or bool(args.glasses_mesh_assets)
        or mesh_style != "auto"
        or mesh_color != "auto"
    )
    mesh_detected = (
        any(m is not None for m in glasses_masks)
        and tone_match_sample is not None
        and tone_match_sample.get("glasses") is not None
    )
    if (args.glasses_method == "mesh"
            and tone_match_sample is not None
            and (mesh_detected or mesh_forced)):
        # Only now do we know glasses will actually be baked; upscale the
        # detail map so the accessory rims stay crisp (small maps break them
        # into blotches) without inflating every bare-face build.
        if D.shape[0] < 1024:
            print(f"glasses mesh: detail {D.shape[0]} -> 1024")
            D = cv2.resize(D, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        if args.detail_jpeg_quality < 94:
            print(f"glasses mesh: detail_jpeg_quality "
                  f"{args.detail_jpeg_quality} -> 94")
            args.detail_jpeg_quality = 94
        template = glasses.infer_template(
            tone_match_sample.get("source_img", tone_match_sample["img"]),
            tone_match_sample["lms"],
            tone_match_sample["glasses"],
            style=mesh_style,
            color=mesh_color,
            rim_width=args.glasses_rim_width,
            lens_width=args.glasses_lens_width,
            lens_height=args.glasses_lens_height,
            bridge=args.glasses_bridge,
        )
        mesh_asset = (args.glasses_mesh_assets
                      or glasses_mesh.default_template_for_style(template.style))
        if mesh_asset and not args.glasses_mesh_assets:
            print(f"glasses mesh: style {template.style} -> template {mesh_asset}")
        mesh_res = glasses_mesh.bake_facegen_glasses(
            D,
            basis,
            c0,
            tex_c,
            color_name=template.color_name,
            style_name=template.style,
            asset_dir=mesh_asset,
            opacity=args.glasses_mesh_opacity,
            scale_x=args.glasses_mesh_scale_x,
            scale_y=args.glasses_mesh_scale_y,
            offset_y=args.glasses_mesh_offset_y,
            rim_width=float(args.glasses_rim_width or 1.0),
        )
        if mesh_res.applied:
            print("glasses mesh:",
                  f"source={mesh_res.source}",
                  f"color={mesh_res.color_name}",
                  f"screen_px={mesh_res.screen_pixels}",
                  f"detail_px={mesh_res.detail_pixels}",
                  f"opacity={args.glasses_mesh_opacity:.2f}")
        else:
            print("glasses mesh: skipped", mesh_res.reason)

    # Draw a crisp frame straight onto the finished detail map (after all the
    # blurring/neutralize passes) so eyeglasses read sharply in-game.
    if (args.glasses_method == "parametric"
            and any(m is not None for m in glasses_masks)
            and tone_match_sample is not None
            and tone_match_sample.get("glasses") is not None):
        rings = eye_ring_detail(basis, tone_match_sample["proj2d"],
                                tone_match_sample["lms"], D.shape[0],
                                lens_scale=1.0, aspect=0.62)
        template = glasses.infer_template(
            tone_match_sample.get("source_img", tone_match_sample["img"]),
            tone_match_sample["lms"],
            tone_match_sample["glasses"],
            style=args.glasses_style,
            color=args.glasses_color,
            rim_width=args.glasses_rim_width,
            lens_width=args.glasses_lens_width,
            lens_height=args.glasses_lens_height,
            bridge=args.glasses_bridge,
        )
        if glasses.draw_parametric_frame(
                D, rings, color=template.detail_color,
                strength=args.glasses_frame_strength,
                template=template):
            print(f"glasses: drew {len(rings)} parametric frame(s) "
                  f"style={template.style} color={template.color_name} "
                  f"rim={template.rim_width:.2f} "
                  f"lens={template.lens_width:.2f}x{template.lens_height:.2f} "
                  f"bridge={template.bridge} "
                  f"(strength={args.glasses_frame_strength})")

    if (args.glasses_method == "frame"
            and any(m is not None for m in glasses_masks)
            and tone_match_sample is not None
            and tone_match_sample.get("glasses") is not None):
        paths, color = source_frame_detail_paths(
            basis,
            tone_match_sample["proj2d"],
            tone_match_sample["img"],
            tone_match_sample["lms"],
            tone_match_sample["glasses"],
            D.shape[0],
        )
        if glasses.draw_detail_frame(
                D, paths, strength=args.glasses_frame_strength,
                color=color, fit_ellipse=False):
            print(f"glasses: drew {len(paths)} source frame contour(s) "
                  f"(strength={args.glasses_frame_strength})")
        elif args.glasses_method == "frame":
            rings = eye_ring_detail(basis, tone_match_sample["proj2d"],
                                    tone_match_sample["lms"], D.shape[0])
            if glasses.draw_detail_frame(
                    D, rings, strength=args.glasses_frame_strength):
                print(f"glasses: drew {len(rings)} fallback frame(s) "
                      f"(strength={args.glasses_frame_strength})")

    if (args.glasses_method == "draw"
            and any(m is not None for m in glasses_masks)
            and tone_match_sample is not None):
        rings = eye_ring_detail(basis, tone_match_sample["proj2d"],
                                tone_match_sample["lms"], D.shape[0])
        if glasses.draw_detail_frame(
                D, rings, strength=args.glasses_frame_strength):
            print(f"glasses: drew {len(rings)} frame(s) "
                  f"(strength={args.glasses_frame_strength})")

    if debug_dir:
        Image.fromarray(photo_uv.astype(np.uint8)).save(debug_dir / "photo_uv256.png")
        Image.fromarray(basis.stat_texture(tex_c).astype(np.uint8)).save(debug_dir / "stat256.png")
        Image.fromarray(D).save(debug_dir / "detail.png")

    buf = io.BytesIO()
    jpeg_quality = int(np.clip(args.detail_jpeg_quality, 1, 95))
    # 4:4:4 chroma: default 4:2:0 subsampling halves color resolution and
    # washes out thin colored detail such as eyeglass rims
    Image.fromarray(D).save(buf, "JPEG", quality=jpeg_quality, optimize=True,
                            subsampling=0)
    print(f"detail jpeg: quality={jpeg_quality} bytes={len(buf.getvalue())}")

    # Identity refine: hill-climb the sym shape so the OOTP-style render of this
    # exact texture/detail best matches the photo's ArcFace embedding. The
    # landmark fit stays the trust-region anchor; emb2shape seeds the search.
    c_final = c0
    if photo_emb is not None and id_refine > 0:
        c_final, sim0, sim_best = identity.refine_shape(
            basis, photo_emb, c0, tex_c, buf.getvalue(),
            n_iter=id_refine, r_max=args.refine_r_max,
            render_size=args.refine_size, start_c=start_c,
            model_path=args.id_model,
        )
        print(f"id refine: sim {sim0:.3f} -> {sim_best:.3f} "
              f"(|dc|={np.linalg.norm((c_final - c0)[:basis.n_sym]):.2f})")
    elif start_c is not None and not modeller_applied:
        strength = float(np.clip(prior_strength, 0.0, 1.0))
        if strength < 0.999:
            c_final = c0.copy()
            c_final[:len(start_c)] = (
                (1.0 - strength) * c0[:len(start_c)]
                + strength * start_c
            )
            print(f"id refine: skipped, blended {prior_source} prior "
                  f"strength={strength:.2f}")
        else:
            c_final = start_c
            print(f"id refine: skipped, using {prior_source} prior")
    elif modeller_applied:
        c_final = c0
        print("id refine: skipped, using modeller direct fit")

    fg = FgFile(
        sym_shape=c_final[:basis.n_sym],
        asym_shape=c_final[basis.n_sym:],
        sym_tex=tex_c,
        asym_tex=np.zeros(0),
        detail_jpeg=buf.getvalue(),
    )
    fg.write(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
