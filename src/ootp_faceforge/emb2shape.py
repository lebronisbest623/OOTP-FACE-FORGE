"""ArcFace embedding -> FaceGen coefficient regression model."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np


DEFAULT_MODEL = (Path(__file__).resolve().parents[2] / "models"
                 / "emb2shape.npz")


def _model_path(model_path: Path | str | None = None) -> Path:
    return Path(model_path or DEFAULT_MODEL)


def available(model_path: Path | str | None = None) -> bool:
    return _model_path(model_path).is_file()


@dataclass(frozen=True)
class Prediction:
    sym_shape: np.ndarray
    asym_shape: np.ndarray
    sym_tex: np.ndarray
    raw: np.ndarray

    @property
    def shape(self) -> np.ndarray:
        return np.concatenate([self.sym_shape, self.asym_shape])


class Emb2ShapeModel:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = _model_path(path)
        data = np.load(self.path)
        self.W = data["W"].astype(np.float32)
        self.n_sym = int(data["n_sym"])
        self.n_asym = int(data["n_asym"])
        self.n_tex = int(data["n_tex"])
        self.lam = float(data["lam"]) if "lam" in data else float("nan")
        if self.W.ndim != 2 or self.W.shape[1] != self.n_sym + self.n_asym + self.n_tex:
            raise ValueError(f"invalid emb2shape model shape: {self.W.shape}")

    @property
    def embedding_dim(self) -> int:
        return self.W.shape[0] - 1

    def predict(self, embedding: np.ndarray) -> Prediction:
        e = np.asarray(embedding, np.float32).reshape(-1)
        if e.shape[0] != self.embedding_dim:
            raise ValueError(
                f"embedding dim {e.shape[0]} != model dim {self.embedding_dim}"
            )
        n = float(np.linalg.norm(e))
        if n > 1e-9:
            e = e / n
        y = np.hstack([e, np.float32(1.0)]) @ self.W
        y = y.astype(np.float32)
        a = self.n_sym
        b = a + self.n_asym
        c = b + self.n_tex
        return Prediction(
            sym_shape=y[:a],
            asym_shape=y[a:b],
            sym_tex=y[b:c],
            raw=y,
        )


@lru_cache(maxsize=4)
def load(model_path: str | None = None) -> Emb2ShapeModel:
    return Emb2ShapeModel(model_path)
