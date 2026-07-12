"""Direct photo-feature -> FaceGen coefficient regression.

This is the fast path we want for Modeller-like builds: detect once, embed once,
then predict FaceGen coefficients without render/score/search loops. The model
is trained by scripts/train_photofit.py and stored as a small .npz ridge model.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import identity
from .landmarks import FACE_OVAL


DEFAULT_MODEL = (Path(__file__).resolve().parents[2] / "models"
                 / "photofit.npz")

# Dense enough to encode face proportions, but much smaller than all 478 points.
FEATURE_LMS = tuple(sorted(set(
    FACE_OVAL
    + [
        0, 1, 2, 4, 5, 6, 9, 17, 18, 33, 37, 39, 40, 52, 55, 61, 63, 66, 70,
        105, 107, 122, 129, 133, 145, 146, 152, 153, 154, 155, 159, 160, 161,
        168, 173, 178, 181, 185, 191, 195, 199, 205, 263, 267, 269, 270, 282,
        285, 291, 293, 296, 300, 334, 336, 351, 358, 362, 374, 380, 381, 382,
        386, 387, 388, 398, 402, 405, 409, 415, 425, 468, 473,
    ]
)))

METRIC_NAMES = (
    "face_w_eye_units",
    "face_h_eye_units",
    "nose_x_eye_units",
    "nose_y_eye_units",
    "mouth_open_eye_units",
    "eye_slope",
    "jaw_w_eye_units",
)


def _model_path(model_path: Path | str | None = None) -> Path:
    return Path(model_path or DEFAULT_MODEL)


def available(model_path: Path | str | None = None) -> bool:
    return _model_path(model_path).is_file()


@dataclass(frozen=True)
class FeatureResult:
    feature: np.ndarray
    embedding: np.ndarray
    n_photos: int


@dataclass(frozen=True)
class Prediction:
    sym_shape: np.ndarray
    asym_shape: np.ndarray
    sym_tex: np.ndarray
    raw: np.ndarray
    embedding: np.ndarray
    n_photos: int
    source: str = "photofit"

    @property
    def shape(self) -> np.ndarray:
        return np.concatenate([self.sym_shape, self.asym_shape])


def feature_names() -> list[str]:
    names = [f"arcface_{i:03d}" for i in range(512)]
    names.extend(f"lm_{li}_{axis}" for li in FEATURE_LMS for axis in ("x", "y"))
    names.extend(METRIC_NAMES)
    return names


def feature_dim() -> int:
    return 512 + 2 * len(FEATURE_LMS) + len(METRIC_NAMES)


def _landmark_feature(lms: np.ndarray) -> np.ndarray:
    pts = np.asarray(lms, np.float32)
    if pts.shape[0] <= max(FEATURE_LMS):
        raise ValueError("landmark array does not include refined iris points")

    eye_r = pts[468]
    eye_l = pts[473]
    center = 0.5 * (eye_r + eye_l)
    eye_vec = eye_l - eye_r
    eye_d = float(np.linalg.norm(eye_vec))
    if eye_d <= 1e-6:
        eye_d = float(np.ptp(pts[FACE_OVAL, 0]))
    eye_d = max(eye_d, 1.0)

    norm_pts = (pts[list(FEATURE_LMS)] - center) / eye_d
    oval = pts[FACE_OVAL]
    jaw_w = float(np.linalg.norm(pts[172] - pts[397])) / eye_d
    metrics = np.array([
        float(np.ptp(oval[:, 0])) / eye_d,
        float(np.ptp(oval[:, 1])) / eye_d,
        float((pts[1, 0] - center[0]) / eye_d),
        float((pts[1, 1] - center[1]) / eye_d),
        float(np.linalg.norm(pts[17] - pts[0]) / eye_d),
        float(eye_vec[1] / max(abs(eye_vec[0]), 1.0)),
        jaw_w,
    ], np.float32)
    return np.concatenate([norm_pts.reshape(-1), metrics]).astype(np.float32)


def feature_from_image(img_rgb: np.ndarray, lms: np.ndarray,
                       id_model: Path | str | None = None
                       ) -> FeatureResult | None:
    emb = identity.embed(img_rgb, lms, id_model)
    if emb is None:
        return None
    feat = np.concatenate([emb.astype(np.float32), _landmark_feature(lms)])
    if feat.shape[0] != feature_dim():
        raise ValueError(f"feature dim {feat.shape[0]} != {feature_dim()}")
    return FeatureResult(feat.astype(np.float32), emb.astype(np.float32), 1)


def photos_feature(images: list[np.ndarray], lms_list: list[np.ndarray],
                   id_model: Path | str | None = None,
                   weights: list[float] | None = None
                   ) -> FeatureResult | None:
    if weights is not None and len(weights) != len(images):
        raise ValueError("weights length must match images")
    feats = []
    ws = []
    for i, (img, lms) in enumerate(zip(images, lms_list)):
        res = feature_from_image(img, lms, id_model)
        if res is None:
            continue
        feats.append(res.feature)
        ws.append(max(float(weights[i]) if weights is not None else 1.0, 0.0))
    if not feats:
        return None
    w = np.asarray(ws, np.float32)
    if float(w.sum()) <= 1e-6:
        w = np.ones(len(feats), np.float32)
    feat = np.average(np.stack(feats), axis=0, weights=w).astype(np.float32)
    emb = feat[:512].copy()
    n = float(np.linalg.norm(emb))
    if n > 1e-9:
        emb /= n
        feat[:512] = emb
    return FeatureResult(feat, emb, len(feats))


class PhotofitModel:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = _model_path(path)
        data = np.load(self.path, allow_pickle=True)
        self.W = data["W"].astype(np.float32)
        self.n_sym = int(data["n_sym"])
        self.n_asym = int(data["n_asym"])
        self.n_tex = int(data["n_tex"])
        self.x_mean = data["x_mean"].astype(np.float32) if "x_mean" in data else None
        self.x_scale = data["x_scale"].astype(np.float32) if "x_scale" in data else None
        self.lam = float(data["lam"]) if "lam" in data else float("nan")
        expected_out = self.n_sym + self.n_asym + self.n_tex
        expected_in = feature_dim() + 1
        if self.W.shape != (expected_in, expected_out):
            raise ValueError(
                f"invalid photofit W shape: {self.W.shape}, "
                f"expected {(expected_in, expected_out)}"
            )

    def _prepare(self, feature: np.ndarray) -> np.ndarray:
        x = np.asarray(feature, np.float32).reshape(-1)
        if x.shape[0] != feature_dim():
            raise ValueError(f"feature dim {x.shape[0]} != {feature_dim()}")
        if self.x_mean is not None and self.x_scale is not None:
            x = (x - self.x_mean) / np.maximum(self.x_scale, 1e-6)
        return x

    def predict_feature(self, feature: np.ndarray,
                        embedding: np.ndarray | None = None,
                        n_photos: int = 1) -> Prediction:
        x = self._prepare(feature)
        y = np.hstack([x, np.float32(1.0)]) @ self.W
        y = y.astype(np.float32)
        a = self.n_sym
        b = a + self.n_asym
        c = b + self.n_tex
        emb = np.asarray(
            embedding if embedding is not None else feature[:512],
            np.float32,
        )
        return Prediction(
            sym_shape=y[:a],
            asym_shape=y[a:b],
            sym_tex=y[b:c],
            raw=y,
            embedding=emb,
            n_photos=int(n_photos),
        )

    def predict_photos(self, images: list[np.ndarray], lms_list: list[np.ndarray],
                       id_model: Path | str | None = None,
                       weights: list[float] | None = None) -> Prediction | None:
        feat = photos_feature(images, lms_list, id_model, weights)
        if feat is None:
            return None
        return self.predict_feature(feat.feature, feat.embedding, feat.n_photos)


@lru_cache(maxsize=4)
def load(model_path: str | None = None) -> PhotofitModel:
    return PhotofitModel(model_path)


def predict_photos(images: list[np.ndarray], lms_list: list[np.ndarray],
                   model_path: Path | str | None = None,
                   id_model: Path | str | None = None,
                   weights: list[float] | None = None) -> Prediction | None:
    if not available(model_path):
        return None
    return load(str(model_path) if model_path else None).predict_photos(
        images, lms_list, id_model, weights)
