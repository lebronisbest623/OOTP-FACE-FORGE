"""Photo landmark detection (mediapipe legacy FaceMesh) and the FaceGen
surface-point <-> mediapipe index correspondence."""
from __future__ import annotations

from pathlib import Path

import numpy as np


# FaceGen surface point name -> mediapipe FaceMesh index (refine_landmarks=True).
# LEFT/RIGHT are the subject's anatomical sides.
CORR = {
    "BROW_CENTRE": 9,
    "BROW_LEFT": 334,
    "BROW_RIGHT": 105,
    "EYE_LEFT_INNER": 362,
    "EYE_LEFT_OUTER": 263,
    "EYE_RIGHT_INNER": 133,
    "EYE_RIGHT_OUTER": 33,
    "EYE_LEFT_CENTRE": 473,
    "EYE_RIGHT_CENTRE": 468,
    "EYE_LID_UPPER_LEFT": 386,
    "EYE_LID_LOWER_LEFT": 374,
    "EYE_LID_UPPER_RIGHT": 159,
    "EYE_LID_LOWER_RIGHT": 145,
    "NARE_LEFT": 358,
    "NARE_RIGHT": 129,
    "NOSE_BASE": 2,
    "NOSE_TIP": 1,
    "NOSE_BRIDGE": 195,
    "SELLION": 168,
    "MOUTH_LEFT": 291,
    "MOUTH_RIGHT": 61,
    "LIP_TIP_UPPER": 0,
    "LIP_TIP_LOWER": 17,
    "LIP_CHIN_CONCAVITY": 18,
    "CHIN": 199,
    "CHIN_LOWER": 152,
    "JAW_OUTER_LEFT": 288,
    "JAW_OUTER_RIGHT": 58,
    "CHEEKBONE_LEFT": 345,
    "CHEEKBONE_RIGHT": 116,
}

# fitting weights (downweight uncertain correspondences)
WEIGHT = {name: 1.0 for name in CORR}
for n in ("CHIN", "BROW_LEFT", "BROW_RIGHT"):
    WEIGHT[n] = 0.5
for n in ("JAW_OUTER_LEFT", "JAW_OUTER_RIGHT", "CHEEKBONE_LEFT", "CHEEKBONE_RIGHT"):
    WEIGHT[n] = 0.25


MODEL = str(Path(__file__).with_name("face_landmarker.task"))

# mediapipe FACE_OVAL landmark loop (standard ordering)
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397,
             365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
             132, 93, 234, 127, 162, 21, 54, 103, 67, 109]


def face_mask(img_rgb, lms) -> "np.ndarray":
    """(H,W) bool mask: inside the face-oval polygon, with cap/hair pixels
    above the brow line rejected by skin-color distance."""
    import cv2

    h, w = img_rgb.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    poly = lms[FACE_OVAL].astype(np.int32)
    cv2.fillPoly(mask, [poly], 1)
    mask = mask.astype(bool)

    # skin model from the central face box (between eyes and mouth)
    cx0, cx1 = int(lms[133][0]), int(lms[362][0])          # inner eye corners
    cy0 = int(max(lms[133][1], lms[362][1]))
    cy1 = int(lms[0][1])                                   # upper lip
    if cx1 <= cx0 + 4 or cy1 <= cy0 + 4:
        return mask
    patch = img_rgb[cy0:cy1, cx0:cx1].reshape(-1, 3).astype(np.float32)
    mu = np.median(patch, 0)
    sd = np.maximum(patch.std(0), 12.0)

    brow_y = min(lms[105][1], lms[334][1])                 # brow tops
    sd = np.clip(sd, 10.0, 20.0)
    dist = (np.abs(img_rgb.astype(np.float32) - mu) / sd).max(2)
    ys = np.arange(h)[:, None]
    xs = np.arange(w)[None, :]
    # cut everything above the brows: cap-brim shadow there reads as valid
    # skin color but is darkened; the statistical texture fills the forehead
    face_h = float(np.ptp(poly[:, 1]))
    mask &= ys >= brow_y - 0.01 * face_h
    # lateral strips (temples/cap sides), down to upper-lip level
    ox0, ox1 = poly[:, 0].min(), poly[:, 0].max()
    strip_w = 0.18 * (ox1 - ox0)
    lateral = ((xs < ox0 + strip_w) | (xs > ox1 - strip_w)) & (ys < lms[0][1])
    mask &= ~(lateral & (dist > 2.5))
    return mask


def exposure_gain(img_rgb: np.ndarray, lms: np.ndarray,
                  ref_lum: float, lo: float = 0.9, hi: float = 1.45) -> float:
    """Scalar exposure gain so cheek-skin luminance approaches the basis mean.
    Clipped so genuinely dark skin keeps most of its tone in the coefficients."""
    h, w = img_rgb.shape[:2]
    eye_d = float(np.linalg.norm(lms[468] - lms[473]))
    r = max(int(eye_d * 0.18), 4)
    samples = []
    for li in (50, 280, 195):          # both cheeks + nose bridge
        x, y = lms[li].astype(int)
        x0, x1 = max(x - r, 0), min(x + r, w)
        y0, y1 = max(y - r, 0), min(y + r, h)
        if x1 > x0 and y1 > y0:
            samples.append(img_rgb[y0:y1, x0:x1].reshape(-1, 3))
    skin = np.concatenate(samples).astype(np.float32)
    lum = float(np.median(skin @ np.array([0.299, 0.587, 0.114])))
    return float(np.clip(1.12 * ref_lum / max(lum, 1.0), lo, hi))


def illum_correct(img_rgb: np.ndarray, mask: np.ndarray,
                  lms: np.ndarray) -> np.ndarray:
    """Symmetrize low-frequency lighting across the face midline (kills the
    one-sided key light typical of pro headshots). Returns float32 RGB."""
    import cv2

    img = img_rgb.astype(np.float32)
    h, w = img.shape[:2]
    face_w = float(np.ptp(lms[FACE_OVAL][:, 0]))
    sigma = max(face_w / 6.0, 8.0)

    m = mask.astype(np.float32)
    blur = lambda a: cv2.GaussianBlur(a, (0, 0), sigma)
    denom = np.maximum(blur(m), 1e-3)
    LF = np.dstack([blur(img[..., ch] * m) / denom for ch in range(3)])

    # mirror the illumination field across the vertical face midline
    mid_x = float(lms[[168, 6, 195, 4]][:, 0].mean())
    xs = np.arange(w)
    mx = np.clip((2 * mid_x - xs).round().astype(int), 0, w - 1)
    LFm = LF[:, mx]
    mm = m[:, mx]
    both = (m > 0.5) & (mm > 0.5)
    gain = np.ones_like(LF)
    tgt = 0.5 * (LF + LFm)
    gain[both] = tgt[both] / np.maximum(LF[both], 1.0)
    gain = np.clip(gain, 0.65, 1.55)
    out = img.copy()
    out[mask] = np.clip(img[mask] * gain[mask], 0, 255)

    # soft-compress specular highlights inside the mask
    Y = out @ np.array([0.299, 0.587, 0.114], np.float32)
    knee = float(np.percentile(Y[mask], 90))
    hi = mask & (Y > knee)
    if hi.any():
        scale = (knee + (Y[hi] - knee) * 0.35) / np.maximum(Y[hi], 1.0)
        out[hi] *= scale[:, None]
    return np.clip(out, 0, 255)


def detect(img_rgb: np.ndarray) -> np.ndarray:
    """Return (478, 2) pixel coords of mediapipe landmarks, or raise."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    h, w = img_rgb.shape[:2]
    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL),
        running_mode=vision.RunningMode.IMAGE, num_faces=1,
        min_face_detection_confidence=0.3)
    with vision.FaceLandmarker.create_from_options(opts) as lmk:
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                          data=np.ascontiguousarray(img_rgb))
        res = lmk.detect(mp_img)
    if not res.face_landmarks:
        raise RuntimeError("no face detected")
    lm = res.face_landmarks[0]
    return np.array([[p.x * w, p.y * h] for p in lm], np.float32)
