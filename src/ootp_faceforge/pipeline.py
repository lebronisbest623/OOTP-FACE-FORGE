"""Multi-photo photo folder -> .fg.

Shape is fit jointly from every usable photo. Texture/detail are taken from one
selected texture photo, because mixing illumination and makeup/eye-black across
photos is usually worse than using the best near-front identity image.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
from PIL import Image

from .basis import get_basis
from .fgformat import FgFile
from .fit import fit_shape_multi_dense, project
from .landmarks import FACE_OVAL, detect, exposure_gain, face_mask, illum_correct
from .texture import build_detail, fit_tex_coeffs, front_tris, tri_coords, uv_px, warp_tris


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build an .fg from multiple photos.")
    p.add_argument("photos", help="Photo folder or single image.")
    p.add_argument("out")
    p.add_argument("--texture-photo",
                   help="Substring/path of the photo to use for texture/detail.")
    p.add_argument("--max-yaw", type=float, default=0.2,
                   help="Skip photos whose absolute yaw proxy is above this value.")
    p.add_argument("--shape-lam", type=float, default=0.16)
    p.add_argument("--asym-lam", type=float, default=0.08)
    p.add_argument("--dense-weight", type=float, default=0.45)
    p.add_argument("--tex-lam", type=float, default=5.0)
    p.add_argument("--tex-erode", type=int, default=1)
    p.add_argument("--exposure-lo", type=float, default=0.9)
    p.add_argument("--exposure-hi", type=float, default=1.15,
                   help="Clamp scalar exposure gain; lower preserves naturally dark skin tones.")
    p.add_argument("--detail-size", type=int, default=256)
    p.add_argument("--detail-strength", type=float, default=0.7)
    p.add_argument("--detail-chroma-strength", type=float, default=0.08,
                   help="Keep only this much color detail; lower compresses cleaner.")
    p.add_argument("--detail-edge-strength", type=float, default=0.85,
                   help="Unsharp amount for luma detail before JPEG encoding.")
    p.add_argument("--detail-flat-neutralize", type=float, default=0.45,
                   help="Blend low-information skin areas toward neutral detail.")
    p.add_argument("--detail-jpeg-quality", type=int, default=85)
    p.add_argument("--eye-detail-strength", type=float, default=0.25)
    p.add_argument("--detail-min-cos", type=float, default=0.18)
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

    img = images[tex_idx]
    lms = landmarks[tex_idx]
    s, R, t = poses[tex_idx]
    proj2d, cam = project(basis, c, s, R, t)

    photo_valid = face_mask(img, lms)
    fimv = basis.fim[..., 0] >= 0
    ref_lum = float((basis.mean_tex[fimv] @ np.array([0.299, 0.587, 0.114])).mean())
    gain = exposure_gain(img, lms, ref_lum,
                         lo=args.exposure_lo, hi=args.exposure_hi)
    print(f"exposure gain: {gain:.3f}")
    img = np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    img = illum_correct(img, photo_valid, lms).astype(np.uint8)
    if debug_dir:
        Image.fromarray(img).save(debug_dir / "illum.png")

    anchor = basis.tri.surface_points["NOSE_TIP"][0]
    fronts = front_tris(basis, proj2d, anchor, cam, min_cos=0.3)
    photo_uv, cov = warp_tris(img.astype(np.float32), tri_coords(basis, proj2d),
                              uv_px(basis, 256)[basis.tris], 256, fronts)
    vm = np.repeat(photo_valid[..., None].astype(np.float32), 3, 2)
    vm_uv, _ = warp_tris(vm, tri_coords(basis, proj2d),
                         uv_px(basis, 256)[basis.tris], 256, fronts)
    cov &= vm_uv[..., 0] > 0.9
    tex_c = fit_tex_coeffs(basis, photo_uv, cov,
                           lam=args.tex_lam, erode=args.tex_erode)
    print(
        f"tex fit: |c|={np.linalg.norm(tex_c):.2f}",
        f"range=[{tex_c.min():.2f},{tex_c.max():.2f}]",
    )

    D, dvalid = build_detail(
        basis,
        img,
        proj2d,
        tex_c,
        anchor,
        photo_valid=photo_valid,
        cam=cam,
        min_cos=args.detail_min_cos,
        size=args.detail_size,
        detail_strength=args.detail_strength,
        chroma_strength=args.detail_chroma_strength,
        edge_strength=args.detail_edge_strength,
        flat_neutralize=args.detail_flat_neutralize,
        neutralize_eyes=True,
        eye_detail_strength=args.eye_detail_strength,
    )
    print("detail coverage:", round(float(dvalid.mean()), 3))
    if debug_dir:
        Image.fromarray(photo_uv.astype(np.uint8)).save(debug_dir / "photo_uv256.png")
        Image.fromarray(basis.stat_texture(tex_c).astype(np.uint8)).save(debug_dir / "stat256.png")
        Image.fromarray(D).save(debug_dir / "detail.png")

    buf = io.BytesIO()
    jpeg_quality = int(np.clip(args.detail_jpeg_quality, 1, 95))
    Image.fromarray(D).save(buf, "JPEG", quality=jpeg_quality, optimize=True)
    print(f"detail jpeg: quality={jpeg_quality} bytes={len(buf.getvalue())}")
    fg = FgFile(
        sym_shape=c[:basis.n_sym],
        asym_shape=c[basis.n_sym:],
        sym_tex=tex_c,
        asym_tex=np.zeros(0),
        detail_jpeg=buf.getvalue(),
    )
    fg.write(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
