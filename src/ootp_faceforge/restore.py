"""Optional GFPGAN (ONNX) face restoration for low-resolution photos.

Small scraped headshots carry enough signal for the shape fit but almost
none for the detail texture map. GFPGAN hallucinates plausible facial
detail at 512x512 from a tiny crop; the result is identity-preserving but
generated, so it is only applied when the source is genuinely too small
(or forced with --restore force).

Uses the standard FFHQ 5-point alignment: similarity-warp the face to the
512 template, run the network, inverse-warp and feather-blend the restored
crop back onto a Lanczos-upscaled copy of the whole photo.

The model file is not bundled; see scripts/download_restore_model.py.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

DEFAULT_MODEL = (Path(__file__).resolve().parents[2] / "models"
                 / "GFPGANv1.4.onnx")

# facexlib FaceRestoreHelper 512x512 template:
# [left eye, right eye, nose tip, left mouth, right mouth] in image coords
FFHQ_512 = np.array([
    [192.98138, 239.94708],
    [318.90277, 240.1936],
    [256.63416, 314.01935],
    [201.26117, 371.41043],
    [313.08905, 371.15118],
], np.float32)

# matching mediapipe indices (image-left eye = subject's right eye)
MP_5PT = [468, 473, 1, 61, 291]

_SESSIONS: dict[str, object] = {}


def available(model_path: Path | str | None = None) -> bool:
    return Path(model_path or DEFAULT_MODEL).is_file()


def _session(model_path: Path):
    key = str(model_path)
    if key not in _SESSIONS:
        import onnxruntime as ort

        _SESSIONS[key] = ort.InferenceSession(
            key, providers=["CPUExecutionProvider"])
    return _SESSIONS[key]


def _run_gfpgan(sess, crop512: np.ndarray) -> np.ndarray:
    """crop512: (512,512,3) uint8 RGB -> restored uint8 RGB."""
    x = crop512.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    x = x.transpose(2, 0, 1)[None]
    inp = sess.get_inputs()[0].name
    y = sess.run(None, {inp: x})[0][0]
    y = np.clip((y.transpose(1, 2, 0) + 1.0) * 0.5, 0.0, 1.0)
    return (y * 255.0 + 0.5).astype(np.uint8)


def restore_image(img_rgb: np.ndarray, lms: np.ndarray, eye_d: float,
                  model_path: Path | str | None = None,
                  target_eye_d: float = 126.0) -> np.ndarray | None:
    """Return an upscaled copy of the photo with the face region restored,
    or None if the model is missing or alignment fails.

    The output scale is chosen so the face fills the 512 crop the way GFPGAN
    was trained (template inter-eye distance ~126 px)."""
    model_path = Path(model_path or DEFAULT_MODEL)
    if not model_path.is_file():
        return None
    pts = lms[MP_5PT].astype(np.float32)

    up = float(np.clip(target_eye_d / max(eye_d, 1.0), 1.0, 8.0))
    big = cv2.resize(img_rgb, None, fx=up, fy=up,
                     interpolation=cv2.INTER_LANCZOS4)
    M, _ = cv2.estimateAffinePartial2D(pts * up, FFHQ_512,
                                       method=cv2.LMEDS)
    if M is None:
        return None
    crop = cv2.warpAffine(big, M, (512, 512), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)
    restored = _run_gfpgan(_session(model_path), crop)

    # feathered paste-back
    mask = np.zeros((512, 512), np.float32)
    cv2.rectangle(mask, (16, 16), (495, 495), 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 12)
    inv = cv2.invertAffineTransform(M)
    h, w = big.shape[:2]
    back = cv2.warpAffine(restored, inv, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)
    mback = cv2.warpAffine(mask, inv, (w, h), flags=cv2.INTER_LINEAR)[..., None]
    out = big.astype(np.float32) * (1 - mback) + back.astype(np.float32) * mback
    return np.clip(out, 0, 255).astype(np.uint8)
