"""Eyeglasses segmentation via an optional BiSeNet face parser (ONNX).

FaceGen `.fg` files carry no accessory geometry, so glasses can only live in
the detail JPEG. The detail pipeline is tuned to *erase* anything that is not
skin micro-texture (eye neutralize, chroma limit, flat/shadow neutralize), so
frames vanish. This module segments the frame + lens region so the pipeline can
(a) keep it out of the skin texture fit and (b) protect it from the detail
neutralizers.

Classical heuristics cannot separate a frame from the intrinsically dark eye
region (brows, lashes, sockets, tear-trough and nose-bridge shadow) or from a
cap-brim shadow, so this uses the CelebAMask-HQ BiSeNet parser (class 6 =
eyeglasses). The model is optional and not bundled; when it is absent the
feature is a no-op and the face is left untouched. See
scripts/download_restore_model.py. A different segmenter can drop in behind the
same ``segment`` signature.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .paths import default_model_path

DEFAULT_MODEL = default_model_path("bisenet_resnet_34.onnx")
GLASSES_CLASS = 6                       # CelebAMask-HQ BiSeNet 'eye_g'
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
_SESSIONS: dict[str, object] = {}


@dataclass
class GlassesResult:
    mask: np.ndarray          # (H,W) bool: frame/lens pixels in the photo
    confidence: float         # 0..1, how sure we are glasses are present
    present: bool             # confidence passed the gate

    @property
    def any(self) -> bool:
        return self.present and bool(self.mask.any())


@dataclass
class GlassesTemplate:
    style: str = "sports_goggle"
    color_name: str = "red"
    detail_color: np.ndarray | None = None
    rim_width: float = 1.0
    lens_width: float = 1.0
    lens_height: float = 1.0
    bridge: str = "thick"
    lens_tint: float = 0.10
    temple: float = 0.45
    cleanup_strength: float = 0.54


_COLOR_RGB = {
    # Detail JPEG values are multiplicative modulation around 64, not opaque
    # RGB paint: on-screen ~= skin * value/64. A vivid frame color therefore
    # needs the OFF channels crushed well below 64 (e.g. red over skin
    # (210,165,145): 64*(200,40,45)/skin ~= (61,16,20)); keeping them near 64
    # renders as washed-out brick/denim instead of the frame's real color.
    "red": np.array([60.0, 18.0, 20.0], np.float32),
    "black": np.array([39.0, 38.0, 37.0], np.float32),
    "brown": np.array([34.0, 26.0, 20.0], np.float32),
    "blue": np.array([16.0, 27.0, 84.0], np.float32),
    "silver": np.array([70.0, 69.0, 67.0], np.float32),
}


def available(model_path: Path | str | None = None) -> bool:
    return Path(model_path or DEFAULT_MODEL).is_file()


def _session(model_path: Path):
    key = str(model_path)
    if key not in _SESSIONS:
        import onnxruntime as ort

        _SESSIONS[key] = ort.InferenceSession(
            key, providers=["CPUExecutionProvider"])
    return _SESSIONS[key]


def _crop_transform(lms: np.ndarray, margin: float = 0.6) -> np.ndarray:
    """Affine mapping full-image pixel coords -> a 512x512 face crop.

    The crop is a square around the landmark bounding box; BiSeNet was trained
    on loosely-cropped faces, so no similarity alignment is needed."""
    lo = lms.min(0)
    hi = lms.max(0)
    center = 0.5 * (lo + hi)
    side = float((hi - lo).max()) * (1.0 + margin)
    side = max(side, 1.0)
    src = np.array([
        [center[0] - side / 2, center[1] - side / 2],
        [center[0] + side / 2, center[1] - side / 2],
        [center[0] - side / 2, center[1] + side / 2],
    ], np.float32)
    dst = np.array([[0, 0], [512, 0], [0, 512]], np.float32)
    return cv2.getAffineTransform(src, dst)


def _parse_glasses(img: np.ndarray, lms: np.ndarray,
                   model_path: Path) -> tuple[np.ndarray, float]:
    """Return (full-image glasses mask bool, area fraction within the crop)."""
    h, w = img.shape[:2]
    M = _crop_transform(lms)
    crop = cv2.warpAffine(img, M, (512, 512), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)
    x = crop.astype(np.float32) / 255.0
    x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
    x = x.transpose(2, 0, 1)[None]
    sess = _session(model_path)
    logits = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]  # (19,512,512)
    seg = logits.argmax(0)
    glasses512 = (seg == GLASSES_CLASS).astype(np.float32)
    area = float(glasses512.mean())
    inv = cv2.invertAffineTransform(M)
    back = cv2.warpAffine(glasses512, inv, (w, h), flags=cv2.INTER_LINEAR)
    return back > 0.5, area


def segment(img: np.ndarray, lms: np.ndarray, mode: str = "auto",
            model_path: Path | str | None = None,
            min_area: float = 0.004) -> GlassesResult:
    """Segment eyeglasses.

    mode: 'auto'/'on' run the parser when the model is present; 'off' skips.
    Returns an empty, not-present result (a safe no-op) when the model is
    missing or too little of the crop is glasses. ``min_area`` is the glasses
    area fraction of the 512 crop below which we treat the face as bare
    (real frames cover ~5-8%; bare faces score ~0)."""
    h, w = img.shape[:2]
    empty = np.zeros((h, w), bool)
    if mode == "off":
        return GlassesResult(empty, 0.0, False)
    path = Path(model_path or DEFAULT_MODEL)
    if not path.is_file():
        return GlassesResult(empty, 0.0, False)

    try:
        mask, area = _parse_glasses(img, lms, path)
    except Exception:
        return GlassesResult(empty, 0.0, False)

    present = area >= min_area
    confidence = float(np.clip(area / 0.02, 0.0, 1.0))
    if not present:
        return GlassesResult(empty, confidence, False)

    # tidy: drop specks, close the gap the lens leaves inside the frame loop
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN,
                         np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    m = constrain_mask(m.astype(bool), lms) | _color_artifact_mask(img, lms)
    if not m.any():
        return GlassesResult(empty, confidence, False)
    return GlassesResult(m.astype(bool), confidence, True)


def _eye_openings(lms: np.ndarray, shape: tuple[int, int],
                  scale: float = 1.0) -> np.ndarray:
    """Filled ellipses over each eyeball opening (not the whole lens)."""
    h, w = shape[:2]
    m = np.zeros((h, w), np.float32)
    for outer, inner, centre in ((33, 133, 468), (263, 362, 473)):
        o, i, c = lms[outer], lms[inner], lms[centre]
        d = i - o
        ax = 0.62 * float(np.linalg.norm(d)) * scale
        if ax < 1:
            continue
        ang = float(np.degrees(np.arctan2(d[1], d[0])))
        cv2.ellipse(m, (int(c[0]), int(c[1])),
                    (int(ax), int(0.5 * ax)), ang, 0, 360, 1.0, -1)
    return m


def _glasses_roi(lms: np.ndarray, shape: tuple[int, int],
                 lens_scale: float = 1.42,
                 bridge_scale: float = 0.075) -> np.ndarray:
    """Conservative eye-region ROI for parser glasses masks.

    Face parsers occasionally confuse cap brims or saturated uniform accents
    with eyeglasses. The parser mask must therefore be clipped to plausible
    lens + bridge geometry before it can affect texture synthesis.
    """
    h, w = shape[:2]
    roi = np.zeros((h, w), np.uint8)
    for outer, inner, centre in ((33, 133, 468), (263, 362, 473)):
        o, i, c = lms[outer], lms[inner], lms[centre]
        d = i - o
        width = float(np.linalg.norm(d))
        if width < 3:
            continue
        ax = int(round(0.78 * width * lens_scale))
        ay = int(round(0.48 * width * lens_scale))
        ang = float(np.degrees(np.arctan2(d[1], d[0])))
        cv2.ellipse(roi, (int(round(c[0])), int(round(c[1]))),
                    (max(2, ax), max(2, ay)), ang, 0, 360, 1, -1)

    eye_d = float(np.linalg.norm(lms[468] - lms[473]))
    thickness = max(2, int(round(eye_d * bridge_scale)))
    cv2.line(roi, tuple(np.round(lms[133]).astype(int)),
             tuple(np.round(lms[362]).astype(int)), 1, thickness,
             lineType=cv2.LINE_AA)
    return roi.astype(bool)


def _color_artifact_mask(img: np.ndarray, lms: np.ndarray) -> np.ndarray:
    """High-chroma accessory pixels around the eyes missed by the parser."""
    roi = _glasses_roi(lms, img.shape, lens_scale=1.7, bridge_scale=0.11)
    imgf = img.astype(np.float32)
    mx = imgf.max(axis=2)
    mn = imgf.min(axis=2)
    chroma = (mx - mn) / np.maximum(mx, 1.0)
    r, g, b = imgf[..., 0], imgf[..., 1], imgf[..., 2]
    lum = imgf @ np.array([0.299, 0.587, 0.114], np.float32)
    yy, xx = np.indices(img.shape[:2])
    lo = np.maximum(np.floor(lms.min(0)).astype(int), 0)
    hi = np.minimum(np.ceil(lms.max(0)).astype(int), [img.shape[1] - 1, img.shape[0] - 1])
    lower_face = (
        (xx >= lo[0]) & (xx <= hi[0]) &
        (yy >= int(round(lms[2, 1]))) & (yy <= hi[1]) &
        (~roi)
    )
    skin_ref = float(np.median(lum[lower_face])) if lower_face.any() else float(np.median(lum))
    # Skin can be reddish; uniform/cap/glasses reds are much more saturated and
    # have far less green/blue than real eyelid skin.
    vivid_red = (r > 95) & (r > 1.22 * g) & (r > 1.22 * b) & (chroma > 0.28)
    vivid_blue = (b > 75) & (b > 1.18 * r) & (b > 1.18 * g) & (chroma > 0.24)
    vivid_green = (g > 75) & (g > 1.18 * r) & (g > 1.18 * b) & (chroma > 0.24)
    bright_glare = (lum > max(115.0, skin_ref + 24.0)) & (chroma < 0.55)
    m = roi & (vivid_red | vivid_blue | vivid_green | bright_glare)
    if not m.any():
        return m
    eye_d = float(np.linalg.norm(lms[468] - lms[473]))
    k = max(3, int(round(eye_d * 0.022)) | 1)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE,
                         np.ones((k, k), np.uint8))
    return m.astype(bool)


def constrain_mask(mask: np.ndarray, lms: np.ndarray) -> np.ndarray:
    """Clip an eyeglasses mask to plausible lens/bridge regions."""
    if mask is None or not mask.any():
        return mask
    return mask.astype(bool) & _glasses_roi(lms, mask.shape)


def frame_overlay(img: np.ndarray, lms: np.ndarray, mask: np.ndarray,
                  eye_suppress: float = 0.85) -> np.ndarray:
    """Isolate just the eyeglass *frame* as a 0..1 signal.

    Unlike keeping the raw warped lens region, this keeps high-chroma frame
    pixels and crisp dark edges inside a plausible eye band. This is meant to
    preserve the source glasses shape instead of drawing generic ellipses.
    """
    mask = constrain_mask(mask, lms)
    if mask is None or not mask.any():
        return np.zeros(img.shape[:2], np.float32)

    imgf = img.astype(np.float32)
    lum = imgf @ np.array([0.299, 0.587, 0.114], np.float32)
    eye_d = float(np.linalg.norm(lms[468] - lms[473]))
    yy = np.indices(img.shape[:2])[0]
    eye_y = float(0.5 * (lms[468, 1] + lms[473, 1]))
    band = (
        (yy >= eye_y - 0.52 * eye_d) &
        (yy <= eye_y + 0.42 * eye_d)
    )

    mx = imgf.max(axis=2)
    mn = imgf.min(axis=2)
    chroma = (mx - mn) / np.maximum(mx, 1.0)
    r, g, b = imgf[..., 0], imgf[..., 1], imgf[..., 2]
    vivid = (
        ((r > 90) & (r > 1.18 * g) & (r > 1.18 * b) & (chroma > 0.24)) |
        ((g > 70) & (g > 1.18 * r) & (g > 1.18 * b) & (chroma > 0.24)) |
        ((b > 70) & (b > 1.18 * r) & (b > 1.18 * g) & (chroma > 0.24))
    )

    sigma = max(eye_d, 8.0)

    outside = (~mask).astype(np.float32)
    blur = lambda a: cv2.GaussianBlur(a, (0, 0), sigma)
    skin_ref = blur(lum * outside) / np.maximum(blur(outside), 1e-3)
    darkness = np.clip((skin_ref - lum) / np.maximum(skin_ref, 1.0), 0.0, 1.0)

    vivid_mask = (mask & band & vivid).astype(np.uint8)
    vivid_edge = cv2.Canny(vivid_mask * 255, 1, 2) > 0
    if vivid_edge.any():
        vivid_edge = cv2.dilate(
            vivid_edge.astype(np.uint8),
            np.ones((3, 3), np.uint8),
        ).astype(bool)

    micro = np.abs(lum - cv2.GaussianBlur(lum, (0, 0), 0.9))
    dark_edge = (darkness > 0.22) & (micro > 7.0)
    eye_open = _eye_openings(lms, img.shape, scale=0.72) > 0.55
    frame_bool = mask & band & (vivid_edge | dark_edge) & (~eye_open | vivid_edge)

    if int(frame_bool.sum()) < max(12, int(0.00008 * img.shape[0] * img.shape[1])):
        m = mask & band
        er = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        frame_bool = m & ~er

    frame = frame_bool.astype(np.float32)
    k = max(1, int(round(eye_d * 0.006)) | 1)
    if k > 1:
        frame = cv2.dilate(frame, np.ones((k, k), np.uint8)).astype(np.float32)
    return cv2.GaussianBlur(np.clip(frame, 0, 1), (0, 0), 0.65)


def frame_detail_color(img: np.ndarray, lms: np.ndarray, mask: np.ndarray,
                       strength: float = 0.55) -> np.ndarray:
    """RGB detail-map value for the detected frame color."""
    signal = frame_overlay(img, lms, mask) > 0.2
    if not signal.any():
        val = 64.0 * (1.0 - float(np.clip(strength, 0.0, 1.0)))
        return np.array([val, val, val], np.float32)

    pix = img.astype(np.float32)[signal]
    mx = pix.max(axis=1)
    mn = pix.min(axis=1)
    chroma = (mx - mn) / np.maximum(mx, 1.0)
    colored = pix[chroma > 0.18]
    src = np.median(colored if len(colored) >= 8 else pix, axis=0)
    src_chroma = (float(src.max() - src.min()) / max(float(src.max()), 1.0))
    strength = float(np.clip(strength, 0.0, 1.0))
    dark = 64.0 * (1.0 - strength)
    if src_chroma < 0.16:
        return np.array([dark, dark, dark], np.float32)

    norm = src / max(float(src.max()), 1.0)
    peak = 64.0 * (0.95 + 0.55 * strength)
    floor = 64.0 * (0.18 + 0.22 * (1.0 - strength))
    color = floor + (peak - floor) * np.power(norm, 1.35)
    return np.clip(color, 8.0, 96.0).astype(np.float32)


def infer_template(
    img: np.ndarray,
    lms: np.ndarray,
    mask: np.ndarray | None,
    style: str = "auto",
    color: str = "auto",
    rim_width: float | None = None,
    lens_width: float | None = None,
    lens_height: float | None = None,
    bridge: str = "auto",
) -> GlassesTemplate:
    """Infer a clean eyeglass template from noisy source-photo evidence.

    The source mask is only evidence for placement/style/color. The generated
    frame is template-rendered, not copied from the mask.
    """
    eye_d = float(np.linalg.norm(lms[468] - lms[473]))
    m = constrain_mask(mask, lms) if mask is not None and mask.any() else None
    if m is None or not m.any():
        m = _glasses_roi(lms, img.shape, lens_scale=1.45)

    frame_sig = frame_overlay(img, lms, m)
    frame_mask = frame_sig > 0.18

    imgf = img.astype(np.float32)
    pix = imgf[frame_mask] if frame_mask.any() else imgf[m]
    if len(pix) == 0:
        pix = imgf.reshape(-1, 3)
    mx = pix.max(axis=1)
    mn = pix.min(axis=1)
    chroma = (mx - mn) / np.maximum(mx, 1.0)
    colored = pix[chroma > 0.20]
    sample = colored if len(colored) >= 8 else pix
    rgb = np.median(sample, axis=0).astype(np.float32)

    def color_name_from_rgb(rgb_val: np.ndarray) -> str:
        r, g, b = [float(x) for x in rgb_val]
        chroma_val = (max(r, g, b) - min(r, g, b)) / max(max(r, g, b), 1.0)
        if chroma_val < 0.16:
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            if lum > 145:
                return "silver"
            if r > g * 1.12 and r > b * 1.12:
                return "brown"
            return "black"
        # Skin/cap shadows are warm and easily look "reddish" numerically.
        # Treat a frame as red only when green/blue are strongly suppressed,
        # as with true red sports goggles.
        if r >= g * 1.65 and r >= b * 1.55 and (r - b) > 42:
            return "red"
        if b >= r * 1.45 and b >= g * 1.35 and (b - r) > 34:
            return "blue"
        if r > b and g > b:
            return "brown"
        return "black"

    color_name = color_name_from_rgb(rgb) if color == "auto" else color
    if color_name not in _COLOR_RGB:
        color_name = "black"
    detail_color = _COLOR_RGB[color_name].copy()

    ys, xs = np.nonzero(m)
    mask_w = float(xs.max() - xs.min()) if len(xs) else eye_d
    mask_h = float(ys.max() - ys.min()) if len(ys) else eye_d * 0.45
    wide_ratio = mask_w / max(eye_d, 1.0)
    tall_ratio = mask_h / max(eye_d, 1.0)
    per_eye_aspects: list[float] = []
    if len(xs):
        cR, cL = lms[468], lms[473]
        dR = (xs - cR[0]) ** 2 + (ys - cR[1]) ** 2
        dL = (xs - cL[0]) ** 2 + (ys - cL[1]) ** 2
        for sel in (dR <= dL, dL < dR):
            if int(sel.sum()) < 12:
                continue
            wx = float(xs[sel].max() - xs[sel].min() + 1)
            hy = float(ys[sel].max() - ys[sel].min() + 1)
            per_eye_aspects.append(wx / max(hy, 1.0))
    lens_aspect = float(np.median(per_eye_aspects)) if per_eye_aspects else 1.45
    frame_pixels = imgf[frame_mask] if frame_mask.any() else np.empty((0, 3), np.float32)
    colored_frame = False
    if len(frame_pixels) >= 8:
        fmx = frame_pixels.max(axis=1)
        fmn = frame_pixels.min(axis=1)
        fchroma = (fmx - fmn) / np.maximum(fmx, 1.0)
        colored_frame = bool(float(np.percentile(fchroma, 75)) > 0.20)

    if style == "auto":
        if color_name in {"red", "blue"} and colored_frame:
            style_name = "sports_goggle"
        elif lens_aspect < 1.12 or tall_ratio > 0.72:
            style_name = "round"
        elif lens_aspect < 1.45:
            style_name = "oval"
        else:
            style_name = "rectangular"
    else:
        style_name = style
    if style_name not in {"sports_goggle", "rectangular", "round", "oval"}:
        style_name = "sports_goggle"

    defaults = {
        "sports_goggle": dict(rim_width=0.72, lens_width=1.22,
                              lens_height=0.64, bridge="thick",
                              lens_tint=0.06, temple=0.55,
                              cleanup_strength=0.78),
        "rectangular": dict(rim_width=0.82, lens_width=1.08,
                            lens_height=0.76, bridge="thin",
                            lens_tint=0.04, temple=0.40,
                            cleanup_strength=0.52),
        "round": dict(rim_width=0.80, lens_width=0.96,
                      lens_height=1.00, bridge="thin",
                      lens_tint=0.03, temple=0.35,
                      cleanup_strength=0.48),
        "oval": dict(rim_width=0.78, lens_width=1.04,
                     lens_height=0.86, bridge="thin",
                     lens_tint=0.03, temple=0.35,
                     cleanup_strength=0.48),
    }[style_name]
    return GlassesTemplate(
        style=style_name,
        color_name=color_name,
        detail_color=detail_color,
        rim_width=float(rim_width if rim_width is not None else defaults["rim_width"]),
        lens_width=float(lens_width if lens_width is not None else defaults["lens_width"]),
        lens_height=float(lens_height if lens_height is not None else defaults["lens_height"]),
        bridge=str(defaults["bridge"] if bridge == "auto" else bridge),
        lens_tint=float(defaults["lens_tint"]),
        temple=float(defaults["temple"]),
        cleanup_strength=float(defaults["cleanup_strength"]),
    )


def suppress_photo(img: np.ndarray, lms: np.ndarray, mask: np.ndarray,
                   dilate_scale: float = 0.018,
                   radius_scale: float = 0.045) -> np.ndarray:
    """Inpaint detected eyeglasses before texture/detail fitting.

    Final detail neutralization alone removes dark rims but can leave lens glare
    as pale baked patches. Inpainting first gives the texture fit a plausible
    skin/eyelid color underneath the accessory; the later detail suppress pass
    only has to clean small residual edges.
    """
    if mask is None or not mask.any():
        return img
    mask = constrain_mask(mask, lms) | _color_artifact_mask(img, lms)
    if not mask.any():
        return img
    eye_d = float(np.linalg.norm(lms[468] - lms[473]))
    k = max(3, int(round(eye_d * dilate_scale)) | 1)
    m = cv2.dilate(mask.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)

    imgf = img.astype(np.float32)
    yy, xx = np.indices(img.shape[:2])
    lo = np.maximum(np.floor(lms.min(0)).astype(int), 0)
    hi = np.minimum(np.ceil(lms.max(0)).astype(int),
                    [img.shape[1] - 1, img.shape[0] - 1])
    lower_face = (
        (xx >= lo[0]) & (xx <= hi[0]) &
        (yy >= int(round(lms[2, 1]))) & (yy <= hi[1]) &
        (~m)
    )
    if lower_face.any():
        fill = np.median(imgf[lower_face], axis=0)
    else:
        fill = np.median(imgf.reshape(-1, 3), axis=0)
    fill = np.clip(fill * 0.84, 0, 255)

    # Telea pulls the red cap brim into broad sunglass masks. Use a conservative
    # skin-color fill instead; OOTP supplies real eyeballs at render time.
    out = imgf.copy()
    out[m] = fill
    out = cv2.GaussianBlur(out, (0, 0), 0.45)
    soft = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.2)[..., None]
    mixed = soft * out + (1.0 - soft) * imgf
    return np.clip(mixed, 0, 255).astype(np.uint8)


def synthetic_frame(img: np.ndarray, lms: np.ndarray, mask: np.ndarray,
                    thickness: float = 0.06) -> np.ndarray:
    """Draw a clean, bold eyeglass frame at the detected lens positions.

    At headshot resolution the real frame is a few faint pixels and warps into a
    smudge. Instead, fit an ellipse to each lens region and *draw* a crisp dark
    rim plus a nose bridge, so the result reads unambiguously as glasses. Returns
    a (H,W) float 0..1 darkening signal, same contract as ``frame_overlay``."""
    h, w = img.shape[:2]
    canvas = np.zeros((h, w), np.float32)
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return canvas
    eye_d = float(np.linalg.norm(lms[468] - lms[473]))
    t = max(2, int(round(eye_d * thickness)))

    cR, cL = lms[468], lms[473]                     # subject right / left eye
    to_right = ((xs - cR[0]) ** 2 + (ys - cR[1]) ** 2
                < (xs - cL[0]) ** 2 + (ys - cL[1]) ** 2)
    for sel in (to_right, ~to_right):
        if sel.sum() < 20:
            continue
        comp = np.zeros((h, w), np.uint8)
        comp[ys[sel], xs[sel]] = 1
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        if len(cnt) < 5:
            continue
        cv2.ellipse(canvas, cv2.fitEllipse(cnt), 1.0, t, lineType=cv2.LINE_AA)

    # nose bridge: a short bar between the inner eye corners
    p0 = lms[133].astype(int)
    p1 = lms[362].astype(int)
    cv2.line(canvas, tuple(p0), tuple(p1), 1.0, max(1, t),
             lineType=cv2.LINE_AA)
    return np.clip(cv2.GaussianBlur(canvas, (0, 0), 0.4), 0.0, 1.0)


def draw_detail_frame(D: np.ndarray, rings: list, strength: float = 0.5,
                      thickness: float = 0.011,
                      color: np.ndarray | tuple[float, float, float] | None = None,
                      fit_ellipse: bool = True) -> bool:
    """Draw crisp eyeglass frames straight onto the finished detail map.

    Drawing here -- after the neutralize/bilateral passes, in detail space --
    keeps the rim sharp instead of smearing a warped photo line into a blob.
    ``rings`` is a list of (K,2) ordered polygons in detail-texture pixel coords
    (one per eye), traced on orbital skin by the caller. Mutates D in place;
    returns True if anything was drawn."""
    if not rings:
        return False
    if color is None:
        val = float(np.clip(64.0 * (1.0 - strength), 0, 255))
        draw_color = (val, val, val)
    else:
        c = np.asarray(color, np.float32).reshape(3)
        draw_color = tuple(float(x) for x in np.clip(c, 0, 255))
    t = max(2, int(round(thickness * D.shape[0])))
    for ring in rings:
        if fit_ellipse and len(ring) >= 5:
            # a fitted ellipse reads as a clean round frame; the raw mapped
            # polygon is jagged from vertex quantization.
            cv2.ellipse(D, cv2.fitEllipse(ring.astype(np.float32)),
                        draw_color, t, lineType=cv2.LINE_AA)
        else:
            cv2.polylines(D, [np.round(ring).astype(np.int32)], True,
                          draw_color, t, lineType=cv2.LINE_AA)
    # nose bridge: connect the two closest points between the two eye rings
    if len(rings) == 2:
        r0, r1 = rings
        d = ((r0[:, None, :] - r1[None, :, :]) ** 2).sum(-1)
        a, b = np.unravel_index(int(d.argmin()), d.shape)
        p0, p1 = r0[a], r1[b]
        cv2.line(D, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])),
                 draw_color, max(2, t - 1), lineType=cv2.LINE_AA)
    return True


def draw_parametric_frame(
    D: np.ndarray,
    rings: list[np.ndarray],
    color: np.ndarray | tuple[float, float, float] | None = None,
    strength: float = 0.72,
    template: GlassesTemplate | None = None,
) -> bool:
    """Draw a clean reconstructed 2.5D eyeglass frame from lens rings.

    Source contours often include lens tint, glare, cap shadow, and parser
    noise. This path uses the detected glasses only for placement/color, then
    draws a smooth sports-goggle-like frame. Because `.fg` has no accessory
    geometry, the "3D" effect is baked as layered detail: frame shadow, rim
    color, inner highlight, faint lens tint, nose pads, and short temple hints.
    """
    if len(rings) < 2:
        return draw_detail_frame(
            D, rings, strength=strength, thickness=0.010,
            color=color, fit_ellipse=True,
        )

    h, w = D.shape[:2]
    ss = 3

    def _clip_path(path: np.ndarray) -> np.ndarray:
        pts = np.asarray(path, np.float32).copy()
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        return pts

    def _blend_mask(mask: np.ndarray, draw_color, alpha: float) -> None:
        if not mask.any():
            return
        a = (mask.astype(np.float32) / 255.0
             * float(np.clip(alpha, 0, 1)))[..., None]
        col = np.asarray(draw_color, np.float32).reshape(1, 1, 3)
        D[:] = np.clip(a * col + (1.0 - a) * D.astype(np.float32),
                       0, 255).astype(D.dtype)

    def blend_polyline(paths: list[np.ndarray], draw_color, thickness: int,
                       alpha: float, closed: bool = True,
                       offset: tuple[float, float] = (0.0, 0.0)) -> None:
        mask_hr = np.zeros((h * ss, w * ss), np.uint8)
        off = np.array(offset, np.float32)
        for path in paths:
            p = np.round(_clip_path(path + off) * ss).astype(np.int32)
            cv2.polylines(mask_hr, [p], closed, 255,
                          max(1, thickness * ss),
                          lineType=cv2.LINE_AA)
        mask = cv2.resize(mask_hr, (w, h), interpolation=cv2.INTER_AREA)
        _blend_mask(mask, draw_color, alpha)

    def blend_fill(paths: list[np.ndarray], draw_color, alpha: float) -> None:
        mask_hr = np.zeros((h * ss, w * ss), np.uint8)
        for path in paths:
            p = np.round(_clip_path(path) * ss).astype(np.int32)
            cv2.fillPoly(mask_hr, [p], 255, lineType=cv2.LINE_AA)
        mask = cv2.resize(mask_hr, (w, h), interpolation=cv2.INTER_AREA)
        _blend_mask(mask, draw_color, alpha)

    def blend_ellipse(center: np.ndarray, axes: tuple[int, int],
                      angle_deg: float, draw_color, alpha: float) -> None:
        mask_hr = np.zeros((h * ss, w * ss), np.uint8)
        c = tuple(np.round(center * ss).astype(int))
        ax = (max(1, int(round(axes[0] * ss))),
              max(1, int(round(axes[1] * ss))))
        cv2.ellipse(mask_hr, c, ax, angle_deg, 0, 360, 255,
                    -1, lineType=cv2.LINE_AA)
        mask = cv2.resize(mask_hr, (w, h), interpolation=cv2.INTER_AREA)
        _blend_mask(mask, draw_color, alpha)

    if template is None:
        template = GlassesTemplate(
            detail_color=(np.asarray(color, np.float32).reshape(3)
                          if color is not None else None)
        )
    if template.detail_color is not None:
        color = template.detail_color

    # Real sports goggles are bold; a thin faint band vanishes at the game's
    # ~21px-eye portrait scale, so this style floors its own strength/width.
    rim_width = float(template.rim_width)
    if template.style == "sports_goggle":
        strength = max(float(strength), 0.85)
        rim_width = max(rim_width, 1.5)

    if color is None:
        val = float(np.clip(64.0 * (1.0 - strength), 0, 255))
        rim = np.array([val, val, val], np.float32)
    else:
        c = np.asarray(color, np.float32).reshape(3)
        contrast = 0.68 + 0.52 * float(np.clip(strength, 0.0, 1.0))
        rim = 64.0 + contrast * (np.clip(c, 0, 255) - 64.0)
        # ceiling 96 (= frame_detail_color's own cap): a colored frame needs
        # bright-channel headroom in the multiplicative detail space or reds
        # collapse to brick/gray in the final render
        rim = np.clip(rim, 18, 96).astype(np.float32)

    strength = float(np.clip(strength, 0.0, 1.0))
    shadow = np.clip(rim - np.array([8.0, 7.0, 6.0], np.float32), 26, 60)
    highlight = np.clip(64.0 + 0.28 * (rim - 64.0), 52, 70)
    lens_tint = np.clip(64.0 + template.lens_tint * (rim - 64.0), 44, 86)

    t = max(1, int(round(0.0072 * max(h, w) * rim_width)))

    lens_paths: list[np.ndarray] = []
    cleanup_paths: list[np.ndarray] = []
    lens_meta: list[tuple[np.ndarray, np.ndarray, np.ndarray, float, float]] = []
    for ring in rings[:2]:
        if len(ring) < 5:
            continue
        ring = np.asarray(ring, np.float32)
        center = ring.mean(axis=0)
        rel = ring - center
        cov = np.cov(rel.T)
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        major = vecs[:, order[0]].astype(np.float32)
        minor = vecs[:, order[1]].astype(np.float32)
        # Keep the lens major axis roughly horizontal in detail space. This
        # avoids cv2.fitEllipse axis swaps that make tall vertical "glasses".
        if abs(float(major[1])) > abs(float(major[0])):
            major, minor = minor, major
        major /= max(float(np.linalg.norm(major)), 1e-6)
        minor /= max(float(np.linalg.norm(minor)), 1e-6)
        if minor[1] < 0:
            minor = -minor
        x_proj = rel @ major
        y_proj = rel @ minor
        a = float(np.percentile(np.abs(x_proj), 96)) * 1.08 * template.lens_width
        b = float(np.percentile(np.abs(y_proj), 96)) * 1.12 * template.lens_height
        if a <= 1 or b <= 1:
            continue
        theta = np.linspace(0, 2 * np.pi, 80, endpoint=False)
        if template.style == "round":
            px, py = 0.86, 0.86
        elif template.style == "rectangular":
            px, py = 0.42, 0.50
        elif template.style == "oval":
            px, py = 0.76, 0.84
        else:
            px, py = 0.48, 0.68
        def superellipse(aa: float, bb: float,
                         cc: np.ndarray = center) -> np.ndarray:
            x = aa * np.sign(np.cos(theta)) * (np.abs(np.cos(theta)) ** px)
            y = bb * np.sign(np.sin(theta)) * (np.abs(np.sin(theta)) ** py)
            pts = (np.outer(x, major) + np.outer(y, minor)).astype(np.float32)
            pts += cc
            pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
            return pts

        # Superellipse lens: cleaner than raw contour, style-aware, and stable
        # under noisy parser masks.
        pts = superellipse(a, b)
        cleanup_scale = 1.42 if template.style == "sports_goggle" else 1.24
        cleanup_center = center - minor * (0.12 * b)
        cleanup_paths.append(superellipse(a * 1.04, b * cleanup_scale,
                                          cleanup_center))
        if template.style == "sports_goggle":
            upper_cleanup = center - minor * (0.48 * b)
            cleanup_paths.append(superellipse(a * 1.02, b * 2.12,
                                              upper_cleanup))
        lens_paths.append(pts)
        lens_meta.append((center, major, minor, a, b))

    if not lens_paths:
        return False

    bridge = None
    sports_bridge: list[np.ndarray] = []
    bridge_t = t
    if len(lens_paths) == 2:
        r0, r1 = sorted(lens_paths, key=lambda pts: float(pts[:, 0].mean()))
        c0 = r0.mean(axis=0)
        c1 = r1.mean(axis=0)
        p0 = r0[np.argmin((r0[:, 0] - r0[:, 0].max()) ** 2
                          + 0.45 * (r0[:, 1] - c0[1]) ** 2)]
        p1 = r1[np.argmin((r1[:, 0] - r1[:, 0].min()) ** 2
                          + 0.45 * (r1[:, 1] - c1[1]) ** 2)]
        bridge = np.stack([p0, p1]).astype(np.float32)
        bridge_t = t + 1 if template.bridge == "thick" else max(1, t - 1)
        if template.style == "sports_goggle":
            left_meta, right_meta = sorted(lens_meta,
                                           key=lambda item: float(item[0][0]))
            avg_a = 0.5 * (left_meta[3] + right_meta[3])
            avg_b = 0.5 * (left_meta[4] + right_meta[4])
            mid = 0.5 * (p0 + p1)
            top = mid + np.array([0.0, -0.18 * avg_b], np.float32)
            bottom = mid + np.array([0.0, 0.62 * avg_b], np.float32)
            sports_bridge.append(np.stack([
                top + np.array([-0.13 * avg_a, 0.0], np.float32),
                bottom,
                top + np.array([0.13 * avg_a, 0.0], np.float32),
            ]).astype(np.float32))

            # Source sports-goggle photos often carry cap shadow and lens glare
            # as one broad band above both lenses. A single soft shield removes
            # that residue before the reconstructed frame is drawn back.
            all_pts = np.vstack(lens_paths)
            x0, y0 = all_pts.min(axis=0)
            x1, y1 = all_pts.max(axis=0)
            center = np.array([
                0.5 * (x0 + x1),
                0.5 * (y0 + y1) - 0.34 * avg_b,
            ], np.float32)
            a = 0.54 * float(x1 - x0)
            b = 1.95 * avg_b
            theta = np.linspace(0, 2 * np.pi, 96, endpoint=False)
            x = a * np.sign(np.cos(theta)) * (np.abs(np.cos(theta)) ** 0.52)
            y = b * np.sign(np.sin(theta)) * (np.abs(np.sin(theta)) ** 0.72)
            shield = np.stack([center[0] + x, center[1] + y], axis=1)
            shield[:, 0] = np.clip(shield[:, 0], 0, w - 1)
            shield[:, 1] = np.clip(shield[:, 1], 0, h - 1)
            cleanup_paths.append(shield.astype(np.float32))

    # First erase the source-photo lens/glare/shadow detail inside the accessory
    # footprint. Then draw clean accessory layers back on top.
    cleanup_alpha = float(np.clip(template.cleanup_strength, 0.0, 0.85))
    blend_fill(cleanup_paths, (64.0, 64.0, 64.0), alpha=cleanup_alpha)
    if bridge is not None:
        cleanup_bridge_paths = [bridge] + sports_bridge
        blend_polyline(cleanup_bridge_paths, (64.0, 64.0, 64.0),
                       thickness=bridge_t + 5, alpha=cleanup_alpha,
                       closed=False)

    # Subtle transparent lens area first, then shadow/rim/highlight strokes.
    blend_fill(lens_paths, tuple(float(x) for x in lens_tint), alpha=0.07)
    blend_polyline(
        lens_paths,
        tuple(float(x) for x in shadow),
        thickness=t + 2,
        alpha=0.44,
        offset=(1.2, 1.1),
    )
    blend_polyline(
        lens_paths,
        tuple(float(x) for x in rim),
        thickness=t,
        alpha=0.92,
    )
    blend_polyline(
        lens_paths,
        tuple(float(x) for x in highlight),
        thickness=max(1, t - 1),
        alpha=0.12,
        offset=(-0.7, -0.8),
    )

    if len(lens_paths) == 2 and bridge is not None:
        blend_polyline(
            [bridge],
            tuple(float(x) for x in shadow),
            thickness=bridge_t + 1,
            alpha=0.42,
            closed=False,
            offset=(1.0, 1.0),
        )
        blend_polyline(
            [bridge],
            tuple(float(x) for x in rim),
            thickness=max(1, bridge_t),
            alpha=0.92,
            closed=False,
        )
        blend_polyline(
            [bridge],
            tuple(float(x) for x in highlight),
            thickness=max(1, t - 2),
            alpha=0.35,
            closed=False,
            offset=(-0.5, -0.6),
        )
        if sports_bridge:
            blend_polyline(
                sports_bridge,
                tuple(float(x) for x in shadow),
                thickness=bridge_t + 1,
                alpha=0.34,
                closed=False,
                offset=(1.0, 1.0),
            )
            blend_polyline(
                sports_bridge,
                tuple(float(x) for x in rim),
                thickness=max(1, bridge_t),
                alpha=0.76,
                closed=False,
            )
            blend_polyline(
                sports_bridge,
                tuple(float(x) for x in highlight),
                thickness=max(1, t - 2),
                alpha=0.22,
                closed=False,
                offset=(-0.5, -0.5),
            )

        # Short temple hints on the outer edges keep the frame from looking
        # pasted-on, while staying inside the face texture.
        left_outer = r0[np.argmin(r0[:, 0])]
        right_outer = r1[np.argmax(r1[:, 0])]
        temple_len = template.temple * 0.40 * max(np.ptp(r0[:, 0]), np.ptp(r1[:, 0]))
        temple_drop = 0.10 * max(np.ptp(r0[:, 1]), np.ptp(r1[:, 1]))
        temples = [
            np.stack([left_outer, left_outer + np.array([-temple_len, temple_drop], np.float32)]),
            np.stack([right_outer, right_outer + np.array([temple_len, temple_drop], np.float32)]),
        ]
        blend_polyline(temples, tuple(float(x) for x in shadow),
                       thickness=max(1, t - 1), alpha=0.24, closed=False)
        blend_polyline(temples, tuple(float(x) for x in rim),
                       thickness=max(1, t - 2), alpha=0.34, closed=False)

    # Nose pads: tiny translucent raised pads near the bridge.
    if len(lens_meta) == 2:
        left, right = sorted(lens_meta, key=lambda item: float(item[0][0]))
        for center, major, minor, a, b in (left, right):
            inner_dir = 1.0 if center[0] < (left[0][0] + right[0][0]) * 0.5 else -1.0
            pad_center = center + major * (inner_dir * a * 0.55) + minor * (b * 0.28)
            axes = (max(1, int(round(a * 0.10))), max(1, int(round(b * 0.16))))
            angle = float(np.degrees(np.arctan2(minor[1], minor[0])))
            blend_ellipse(pad_center, axes, angle, (82.0, 77.0, 72.0), alpha=0.22)

    return bool(lens_paths)
