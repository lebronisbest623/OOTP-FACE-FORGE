"""CUFP/FaceGen retrieval prior.

The failed direction was treating photo->coeff as a global regression problem.
This module instead uses a CUFP photo/FG bank as a nearest-neighbour prior:

  query photos -> ArcFace + landmark geometry -> top-K similar CUFP faces
  -> coefficient blend -> optional local refine downstream

That keeps outputs inside the distribution of real FaceGen files without forcing
the model to average every noisy photo/FG pair into one bland regressor.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import photofit


DEFAULT_INDEX = (Path(__file__).resolve().parents[2] / "models"
                 / "cufp_identity_index.npz")
ADVISORY_SCORE = 0.35
FULL_SCORE = 0.85


def _index_path(index_path: Path | str | None = None) -> Path:
    return Path(index_path or DEFAULT_INDEX)


def available(index_path: Path | str | None = None) -> bool:
    return _index_path(index_path).is_file()


def confidence_from_score(score: float, advisory_score: float = ADVISORY_SCORE,
                          full_score: float = FULL_SCORE) -> float:
    """Map nearest-neighbour similarity to a conservative blend strength."""
    lo = float(advisory_score)
    hi = max(float(full_score), lo + 1e-6)
    return float(np.clip((float(score) - lo) / (hi - lo), 0.0, 1.0))


@dataclass(frozen=True)
class RetrievalHit:
    rank: int
    name: str
    score: float
    emb_score: float
    geom_score: float
    weight: float
    render_score: float | None = None
    render_emb_score: float | None = None
    render_geom_score: float | None = None


@dataclass(frozen=True)
class Prediction:
    sym_shape: np.ndarray
    asym_shape: np.ndarray
    sym_tex: np.ndarray
    raw: np.ndarray
    hits: list[RetrievalHit]
    n_photos: int
    confidence: float
    reranked: bool = False
    render_matches: int = 0
    source: str = "cufp_retrieval"

    @property
    def shape(self) -> np.ndarray:
        return np.concatenate([self.sym_shape, self.asym_shape])


class CufpIndex:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = _index_path(path)
        data = np.load(self.path, allow_pickle=True)
        self.names = data["names"].astype(str)
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
            geom = geom / gn
        return emb.astype(np.float32), geom.astype(np.float32)

    def predict_feature(self, feature: np.ndarray, top_k: int = 12,
                        geom_weight: float = 0.18, temperature: float = 0.055,
                        exclude_names: set[str] | None = None,
                        n_photos: int = 1,
                        render_index_path: Path | str | None = None,
                        render_top_n: int = 64,
                        render_weight: float = 0.05,
                        render_geom_weight: float = 0.12,
                        render_min_matches: int = 3) -> Prediction:
        emb, geom = self._query_parts(feature)
        emb_scores = self.embeddings @ emb
        geom_scores = self.geom @ geom
        scores = emb_scores + float(geom_weight) * geom_scores
        if exclude_names:
            mask = np.isin(self.names, list(exclude_names))
            scores = scores.copy()
            scores[mask] = -np.inf

        candidate_k = max(int(top_k), int(render_top_n) if render_index_path else 0)
        candidate_k = max(1, min(candidate_k, len(scores)))
        candidate_idx = np.argpartition(-scores, candidate_k - 1)[:candidate_k]
        candidate_idx = candidate_idx[np.argsort(-scores[candidate_idx])]
        final_scores = scores[candidate_idx].astype(np.float32).copy()
        render_scores_by_name: dict[str, tuple[float, float, float]] = {}
        reranked = False
        render_matches = 0
        if render_index_path:
            try:
                from . import render_retrieval

                render_scores_by_name = render_retrieval.load(
                    str(render_index_path)
                ).score_names(
                    feature,
                    [str(self.names[j]) for j in candidate_idx],
                    geom_weight=render_geom_weight,
                )
                render_matches = len(render_scores_by_name)
                if render_matches >= int(render_min_matches):
                    render_vec = np.full(len(candidate_idx), -1.0, np.float32)
                    for i, j in enumerate(candidate_idx):
                        vals = render_scores_by_name.get(str(self.names[j]))
                        if vals is not None:
                            render_vec[i] = vals[0]
                    final_scores = final_scores + float(render_weight) * render_vec
                    reranked = True
            except Exception:
                render_scores_by_name = {}
                render_matches = 0
                reranked = False

        k = max(1, min(int(top_k), len(candidate_idx)))
        order = np.argsort(-final_scores)[:k]
        idx = candidate_idx[order]
        raw_scores = final_scores[order]
        temp = max(float(temperature), 1e-4)
        logits = (raw_scores - float(raw_scores.max())) / temp
        weights = np.exp(np.clip(logits, -50.0, 0.0))
        weights = weights / max(float(weights.sum()), 1e-9)
        coeff = (self.coeffs[idx] * weights[:, None]).sum(0).astype(np.float32)

        hits = [
            RetrievalHit(
                rank=i + 1,
                name=str(self.names[j]),
                score=float(scores[j]),
                emb_score=float(emb_scores[j]),
                geom_score=float(geom_scores[j]),
                weight=float(weights[i]),
                render_score=(
                    render_scores_by_name[str(self.names[j])][0]
                    if str(self.names[j]) in render_scores_by_name else None
                ),
                render_emb_score=(
                    render_scores_by_name[str(self.names[j])][1]
                    if str(self.names[j]) in render_scores_by_name else None
                ),
                render_geom_score=(
                    render_scores_by_name[str(self.names[j])][2]
                    if str(self.names[j]) in render_scores_by_name else None
                ),
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
            confidence=confidence_from_score(float(scores[candidate_idx[0]])),
            reranked=reranked,
            render_matches=render_matches,
        )

    def predict_photos(self, images: list[np.ndarray], lms_list: list[np.ndarray],
                       id_model: Path | str | None = None,
                       weights: list[float] | None = None,
                       top_k: int = 12,
                       geom_weight: float = 0.18,
                       temperature: float = 0.055,
                       render_index_path: Path | str | None = None,
                       render_top_n: int = 64,
                       render_weight: float = 0.05,
                       render_geom_weight: float = 0.12,
                       render_min_matches: int = 3) -> Prediction | None:
        feat = photofit.photos_feature(images, lms_list, id_model, weights)
        if feat is None:
            return None
        return self.predict_feature(
            feat.feature,
            top_k=top_k,
            geom_weight=geom_weight,
            temperature=temperature,
            n_photos=feat.n_photos,
            render_index_path=render_index_path,
            render_top_n=render_top_n,
            render_weight=render_weight,
            render_geom_weight=render_geom_weight,
            render_min_matches=render_min_matches,
        )


@lru_cache(maxsize=4)
def load(index_path: str | None = None) -> CufpIndex:
    return CufpIndex(index_path)


def predict_photos(images: list[np.ndarray], lms_list: list[np.ndarray],
                   index_path: Path | str | None = None,
                   id_model: Path | str | None = None,
                   weights: list[float] | None = None,
                   top_k: int = 12,
                   geom_weight: float = 0.18,
                   temperature: float = 0.055,
                   render_index_path: Path | str | None = None,
                   render_top_n: int = 64,
                   render_weight: float = 0.05,
                   render_geom_weight: float = 0.12,
                   render_min_matches: int = 3) -> Prediction | None:
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
        render_index_path=render_index_path,
        render_top_n=render_top_n,
        render_weight=render_weight,
        render_geom_weight=render_geom_weight,
        render_min_matches=render_min_matches,
    )
