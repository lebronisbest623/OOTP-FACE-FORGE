"""FaceGen render-domain retrieval prior.

Runtime target: detect/embed the user photo once, then do a vector lookup
against pre-rendered OOTP/FaceGen faces. This is closer to the final output
domain than matching against source photos, and keeps the expensive render and
landmark work in an offline index build.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import photofit
from .retrieval import confidence_from_score


DEFAULT_INDEX = (Path(__file__).resolve().parents[2] / "models"
                 / "fg_render_identity_index.npz")


def _index_path(index_path: Path | str | None = None) -> Path:
    return Path(index_path or DEFAULT_INDEX)


def available(index_path: Path | str | None = None) -> bool:
    return _index_path(index_path).is_file()


@dataclass(frozen=True)
class RenderHit:
    rank: int
    name: str
    source: str
    score: float
    emb_score: float
    geom_score: float
    weight: float


@dataclass(frozen=True)
class Prediction:
    sym_shape: np.ndarray
    asym_shape: np.ndarray
    sym_tex: np.ndarray
    raw: np.ndarray
    hits: list[RenderHit]
    n_photos: int
    confidence: float
    source: str = "fg_render_retrieval"

    @property
    def shape(self) -> np.ndarray:
        return np.concatenate([self.sym_shape, self.asym_shape])


class RenderIndex:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = _index_path(path)
        data = np.load(self.path, allow_pickle=True)
        self.names = data["names"].astype(str)
        self.sources = (
            data["sources"].astype(str)
            if "sources" in data else np.full(len(self.names), "fg", dtype=object)
        )
        self.coeffs = data["coeffs"].astype(np.float32)
        self.embeddings = data["embeddings"].astype(np.float32)
        self.geom = data["geom"].astype(np.float32)
        self.geom_mean = data["geom_mean"].astype(np.float32)
        self.geom_scale = data["geom_scale"].astype(np.float32)
        self.n_sym = int(data["n_sym"]) if "n_sym" in data else 50
        self.n_asym = int(data["n_asym"]) if "n_asym" in data else 30
        self.n_tex = int(data["n_tex"]) if "n_tex" in data else 50
        if self.embeddings.ndim != 2 or self.embeddings.shape[1] != 512:
            raise ValueError(f"invalid embeddings shape: {self.embeddings.shape}")
        if self.coeffs.shape[0] != self.embeddings.shape[0]:
            raise ValueError("coeff and embedding row counts differ")
        if self.coeffs.shape[1] != self.n_sym + self.n_asym + self.n_tex:
            raise ValueError(f"invalid coeff shape: {self.coeffs.shape}")
        self.name_to_index: dict[str, int] = {}
        for i, name in enumerate(self.names):
            self.name_to_index.setdefault(str(name), i)

    def _query_parts(self, feature: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        feature = np.asarray(feature, np.float32).reshape(-1)
        if feature.shape[0] != photofit.feature_dim():
            raise ValueError(
                f"feature dim {feature.shape[0]} != {photofit.feature_dim()}"
            )
        emb = feature[:512].copy()
        n = float(np.linalg.norm(emb))
        if n > 1e-9:
            emb /= n
        geom = (feature[512:] - self.geom_mean) / np.maximum(self.geom_scale, 1e-6)
        gn = float(np.linalg.norm(geom))
        if gn > 1e-9:
            geom /= gn
        return emb.astype(np.float32), geom.astype(np.float32)

    def predict_feature(self, feature: np.ndarray, top_k: int = 16,
                        geom_weight: float = 0.12, temperature: float = 0.06,
                        n_photos: int = 1) -> Prediction:
        emb, geom = self._query_parts(feature)
        emb_scores = self.embeddings @ emb
        geom_scores = self.geom @ geom
        scores = emb_scores + float(geom_weight) * geom_scores

        k = max(1, min(int(top_k), len(scores)))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        raw_scores = scores[idx]
        temp = max(float(temperature), 1e-4)
        logits = (raw_scores - float(raw_scores.max())) / temp
        weights = np.exp(np.clip(logits, -50.0, 0.0))
        weights = weights / max(float(weights.sum()), 1e-9)
        coeff = (self.coeffs[idx] * weights[:, None]).sum(0).astype(np.float32)

        hits = [
            RenderHit(
                rank=i + 1,
                name=str(self.names[j]),
                source=str(self.sources[j]),
                score=float(scores[j]),
                emb_score=float(emb_scores[j]),
                geom_score=float(geom_scores[j]),
                weight=float(weights[i]),
            )
            for i, j in enumerate(idx)
        ]
        a = self.n_sym
        b = a + self.n_asym
        c = b + self.n_tex
        return Prediction(
            sym_shape=coeff[:a],
            asym_shape=coeff[a:b],
            sym_tex=coeff[b:c],
            raw=coeff,
            hits=hits,
            n_photos=int(n_photos),
            confidence=confidence_from_score(float(raw_scores[0]), 0.22, 0.72),
        )

    def score_names(self, feature: np.ndarray, names: list[str],
                    geom_weight: float = 0.12) -> dict[str, tuple[float, float, float]]:
        """Score only the supplied candidate names against this render index."""
        emb, geom = self._query_parts(feature)
        out: dict[str, tuple[float, float, float]] = {}
        for name in names:
            idx = self.name_to_index.get(str(name))
            if idx is None:
                continue
            emb_score = float(self.embeddings[idx] @ emb)
            geom_score = float(self.geom[idx] @ geom)
            score = emb_score + float(geom_weight) * geom_score
            out[str(name)] = (score, emb_score, geom_score)
        return out

    def predict_photos(self, images: list[np.ndarray], lms_list: list[np.ndarray],
                       id_model: Path | str | None = None,
                       weights: list[float] | None = None,
                       top_k: int = 16,
                       geom_weight: float = 0.12,
                       temperature: float = 0.06) -> Prediction | None:
        feat = photofit.photos_feature(images, lms_list, id_model, weights)
        if feat is None:
            return None
        return self.predict_feature(
            feat.feature,
            top_k=top_k,
            geom_weight=geom_weight,
            temperature=temperature,
            n_photos=feat.n_photos,
        )


@lru_cache(maxsize=4)
def load(index_path: str | None = None) -> RenderIndex:
    return RenderIndex(index_path)


def predict_photos(images: list[np.ndarray], lms_list: list[np.ndarray],
                   index_path: Path | str | None = None,
                   id_model: Path | str | None = None,
                   weights: list[float] | None = None,
                   top_k: int = 16,
                   geom_weight: float = 0.12,
                   temperature: float = 0.06) -> Prediction | None:
    if not available(index_path):
        return None
    return load(str(index_path) if index_path else None).predict_photos(
        images,
        lms_list,
        id_model=id_model,
        weights=weights,
        top_k=top_k,
        geom_weight=geom_weight,
        temperature=temperature,
    )
