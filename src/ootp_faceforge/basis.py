"""Load the FaceGen SI statistical appearance model from the Modeller demo data."""
from __future__ import annotations
import struct
from functools import cached_property

import numpy as np
from PIL import Image

from . import fgformat

PF = r"C:\Program Files\FaceGen\Modeller Demo 3\data\photofit"


class Basis:
    def __init__(self, pf: str = PF):
        self.tri = fgformat.TriMesh.read(pf + r"\si.tri")
        self.egm = fgformat.Egm.read(pf + r"\si.egm")
        self.egt = fgformat.Egt.read(pf + r"\si.egt")
        self.mean_tex = np.asarray(Image.open(pf + r"\si.bmp").convert("RGB"),
                                   np.float32)          # (256,256,3) RGB
        self.mask = np.asarray(Image.open(pf + r"\siMask.bmp").convert("L"),
                               np.float32)               # (256,256)
        self.fade = np.asarray(Image.open(pf + r"\siFade.bmp").convert("L"),
                               np.float32)
        data = open(pf + r"\si.fim", "rb").read()
        W, H = struct.unpack_from("<2L", data, 8)
        self.fim = np.frombuffer(data, "<f4", W * H * 2, 64).reshape(H, W, 2).copy()

        t = self.tri
        self.verts = t.verts[: t.V]                      # (V,3)
        self.tris = t.tris                               # (T,3)
        # geometry modes: (80, V, 3) sym first then asym
        self.modes = np.concatenate([self.egm.sym, self.egm.asym], 0)
        self.n_sym, self.n_asym = self.egm.sym.shape[0], self.egm.asym.shape[0]

    @cached_property
    def vert_uv(self) -> np.ndarray:
        """Per-vertex UV (V,2) averaged from per-facet UVs."""
        t = self.tri
        acc = np.zeros((t.V, 2), np.float64)
        cnt = np.zeros(t.V, np.int64)
        vi = t.tris.ravel()
        ui = t.tri_uv_idx.ravel()
        np.add.at(acc, vi, t.uvs[ui])
        np.add.at(cnt, vi, 1)
        cnt[cnt == 0] = 1
        return (acc / cnt[:, None]).astype(np.float32)

    def fim_lookup(self, uv: np.ndarray) -> np.ndarray:
        """Map UV coords (N,2 in [0,1]) -> detail-texture UV (N,2), nearest sample.
        Returns -1 pairs where unmapped."""
        H, W, _ = self.fim.shape
        # texture v axis: image row 0 = v=1 (top). si UVs use standard GL convention.
        px = np.clip((uv[:, 0] * W).astype(int), 0, W - 1)
        py = np.clip(((1 - uv[:, 1]) * H).astype(int), 0, H - 1)
        return self.fim[py, px]

    def surface_points(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """name -> (base_pos (3,), B (3,80)) linear model pos = base + B @ coeffs."""
        out = {}
        for name, (fidx, bary) in self.tri.surface_points.items():
            vidx = self.tris[fidx]                       # (3,)
            w = np.asarray(bary, np.float32)             # (3,)
            base = w @ self.verts[vidx]                  # (3,)
            B = np.einsum("k,mkd->dm", w, self.modes[:, vidx, :])  # (3,80)
            out[name] = (base, B)
        return out

    def shaped_verts(self, coeffs: np.ndarray) -> np.ndarray:
        """All vertices for coefficient vector (80,)."""
        return self.verts + np.einsum("m,mvd->vd", coeffs.astype(np.float32), self.modes)

    def stat_texture(self, tex_coeffs: np.ndarray) -> np.ndarray:
        """(256,256,3) float RGB statistical texture from 50 sym coeffs."""
        s = self.mean_tex + np.einsum("m,mrcd->rcd",
                                      tex_coeffs.astype(np.float32), self.egt.sym)
        return np.clip(s, 0, 255)
