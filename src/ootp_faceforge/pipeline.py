"""Multi-photo photo folder -> OOTP-ready .fg.

Shape is fit jointly from every usable photo. Texture/detail default to
multi-photo fusion so the generated file is tuned for OOTP in-game rendering,
where small portrait views and game lighting can suppress subtle face detail.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
from PIL import Image

from . import emb2shape, identity, restore
from .basis import get_basis
from .fgformat import FgFile
from .fit import fit_shape_multi_dense, project
from .landmarks import (
    FACE_OVAL,
    detect,
    exposure_gain,
    face_mask,
    illum_correct,
)
from .texture import build_detail, fit_tex_coeffs, front_tris, fuse_maps, tri_coords, uv_px, warp_tris


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
RESTORE_EYE_D = 60.0  # faces smaller than this benefit from GFPGAN restoration
TEXTURE_FUSE_WEIGHT_POWER = 2.0
DETAIL_FUSE_WEIGHT_POWER = 1.6


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
                   id_model: str | None, use_emb2shape: bool,
                   emb_model: str | None, weights: list[float] | None = None):
    """Mean ArcFace embedding of the photos plus, if a regressor is present,
    an embedding-decoded FaceGen shape prior. Returns (photo_emb, start_c)."""
    if not identity.available(id_model):
        return None, None
    photo_emb = identity.photos_embedding(images, landmarks, id_model, weights=weights)
    if photo_emb is None:
        return None, None
    start_c = None
    if use_emb2shape and emb2shape.available(emb_model):
        pred = emb2shape.load(str(emb_model) if emb_model else None).predict(photo_emb)
        start_c = np.concatenate([pred.sym_shape, pred.asym_shape])
    return photo_emb, start_c


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
                    lo: float = 0.9, hi: float = 1.25):
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


def build_texture_sample(basis, img: np.ndarray, lms: np.ndarray, pose,
                         shape_c: np.ndarray, ref_lum: float, anchor: int,
                         args: argparse.Namespace) -> dict:
    photo_valid = face_mask(img, lms)
    gain = exposure_gain(img, lms, ref_lum,
                         lo=args.exposure_lo, hi=args.exposure_hi)
    img = np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    img = illum_correct(img, photo_valid, lms).astype(np.uint8)
    s, R, t = pose
    proj2d, cam = project(basis, shape_c, s, R, t)
    fronts = front_tris(basis, proj2d, anchor, cam, min_cos=0.3)
    photo_uv, cov = warp_tris(
        img.astype(np.float32),
        tri_coords(basis, proj2d),
        uv_px(basis, 256)[basis.tris],
        256,
        fronts,
    )
    vm = np.repeat(photo_valid[..., None].astype(np.float32), 3, 2)
    vm_uv, _ = warp_tris(vm, tri_coords(basis, proj2d),
                         uv_px(basis, 256)[basis.tris], 256, fronts)
    cov &= vm_uv[..., 0] > 0.9
    return {
        "img": img,
        "lms": lms,
        "photo_valid": photo_valid,
        "proj2d": proj2d,
        "cam": cam,
        "photo_uv": photo_uv,
        "uv_cov": cov,
        "gain": gain,
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
    p.add_argument("--detail-size", type=int, default=256)
    p.add_argument("--detail-strength", type=float, default=0.8)
    p.add_argument("--detail-chroma-strength", type=float, default=0.08,
                   help="Keep only this much color detail; lower compresses cleaner.")
    p.add_argument("--detail-edge-strength", type=float, default=0.9,
                   help="Unsharp amount for luma detail before JPEG encoding.")
    p.add_argument("--detail-flat-neutralize", type=float, default=0.4,
                   help="Blend low-information skin areas toward neutral detail.")
    p.add_argument("--detail-shadow-neutralize", type=float, default=0.8,
                   help="Suppress broad baked-in lighting shadows in the detail map.")
    p.add_argument("--detail-jpeg-quality", type=int, default=85)
    p.add_argument("--eye-detail-strength", type=float, default=0.25)
    p.add_argument("--detail-min-cos", type=float, default=0.18)
    p.add_argument("--shape-gain", type=float, default=1.06,
                   help="Scale applied to the ridge-fit sym shape before capping.")
    p.add_argument("--shape-cap", type=float, default=3.0,
                   help="Per-mode absolute clip on sym shape coefficients.")
    p.add_argument("--shape-norm-cap", type=float, default=9.0,
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
    p.add_argument("--emb2shape", choices=("auto", "off"), default="off",
                   help="Use the learned embedding->shape prior as refine start. "
                        "Trained in the render domain; underperforms the landmark "
                        "fit on real photos, so off by default.")
    p.add_argument("--emb2shape-model",
                   help="Path to the emb2shape .npz (defaults to models/).")
    p.add_argument("--refine-size", type=int, default=128,
                   help="Render size used to score ArcFace similarity in refine.")
    p.add_argument("--refine-r-max", type=float, default=2.2,
                   help="Trust-region rms radius of refine around the landmark fit.")
    p.add_argument("--skin-tone-match", choices=("on", "off"), default="on",
                   help="White-balance the rendered face to the photo skin tone.")
    p.add_argument("--debug-dir",
                   help="Optional directory for intermediate texture/debug images.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
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
        images.append(img)
        landmarks.append(lms)
        metrics.append(m)
        usable_paths.append(p)
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
    want_id = id_refine > 0 or args.emb2shape != "off"
    photo_emb, start_c = (None, None)
    if want_id:
        try:
            id_weights = [float(m["texture_score"]) for m in metrics]
            photo_emb, start_c = identity_prior(
                images, landmarks, args.id_model,
                args.emb2shape != "off", args.emb2shape_model,
                weights=id_weights)
            if start_c is not None:
                # keep only the emb2shape sym prediction; its asym modes are not
                # recoverable from a near-frontal embedding, so anchor asym to the
                # landmark fit.
                start_c[basis.n_sym:] = c0[basis.n_sym:]
            print("identity:",
                  "emb=yes" if photo_emb is not None else "emb=no",
                  "emb2shape=yes" if start_c is not None else "emb2shape=no",
                  f"id_refine={id_refine}")
        except Exception as exc:
            photo_emb, start_c = None, None
            id_refine = 0
            print("identity: disabled", type(exc).__name__, exc)

    fimv = basis.fim[..., 0] >= 0
    ref_lum = float((basis.mean_tex[fimv] @ np.array([0.299, 0.587, 0.114])).mean())
    anchor = basis.front_anchor_tri

    print("texture mode:", args.texture_mode)
    tex_weights = texture_weights(metrics, tex_idx, args.texture_photo)
    tone_match_sample = None
    if args.texture_mode == "best":
        sample = build_texture_sample(
            basis, images[tex_idx], landmarks[tex_idx], poses[tex_idx],
            c, ref_lum, anchor, args,
        )
        tone_match_sample = sample
        print(f"exposure gain: {float(sample['gain']):.3f}")
        if debug_dir:
            Image.fromarray(sample["img"]).save(debug_dir / "illum.png")
        photo_uv = sample["photo_uv"]
        cov = sample["uv_cov"]
        detail_samples = [(sample, tex_weights[tex_idx])]
    else:
        texture_samples = []
        for i, (img_i, lms_i, pose_i, path_i, weight_i) in enumerate(zip(
                images, landmarks, poses, usable_paths, tex_weights)):
            sample = build_texture_sample(
                basis, img_i, lms_i, pose_i, c, ref_lum, anchor, args,
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
            )
            if debug_dir and i == tex_idx:
                Image.fromarray(sample["img"]).save(debug_dir / "illum.png")
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

    tex_c = fit_tex_coeffs(basis, photo_uv, cov,
                           lam=args.tex_lam, erode=args.tex_erode)
    print(
        f"tex fit: |c|={np.linalg.norm(tex_c):.2f}",
        f"range=[{tex_c.min():.2f},{tex_c.max():.2f}]",
    )

    detail_maps, detail_covs, detail_weights = [], [], []
    for sample, weight in detail_samples:
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
            neutralize_eyes=True,
            eye_detail_strength=args.eye_detail_strength,
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

    if debug_dir:
        Image.fromarray(photo_uv.astype(np.uint8)).save(debug_dir / "photo_uv256.png")
        Image.fromarray(basis.stat_texture(tex_c).astype(np.uint8)).save(debug_dir / "stat256.png")
        Image.fromarray(D).save(debug_dir / "detail.png")

    buf = io.BytesIO()
    jpeg_quality = int(np.clip(args.detail_jpeg_quality, 1, 95))
    Image.fromarray(D).save(buf, "JPEG", quality=jpeg_quality, optimize=True)
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
    elif start_c is not None:
        c_final = start_c
        print("id refine: skipped, using emb2shape prior")

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
