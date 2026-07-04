"""Load OOTP's FaceGen appearance basis used for fitting."""
from __future__ import annotations

import struct
from functools import cached_property, lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import fgformat
from .paths import get_ootp_3d_path, normalize_ootp_3d_path


def _read_fim(path: Path) -> np.ndarray:
    data = path.read_bytes()
    w, h = struct.unpack_from("<2L", data, 8)
    return np.frombuffer(data, "<f4", w * h * 2, 64).reshape(h, w, 2).copy()


def _load_rgb(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, np.float32)
    if size is not None and arr.shape[:2] != (size[1], size[0]):
        arr = cv2.resize(arr, size, interpolation=cv2.INTER_LINEAR)
    return arr.astype(np.float32)


def _triangulate_for_fit(
    mesh: fgformat.TriMesh,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return triangle vertex indices plus matching per-corner UV indices.

    OOTP's face_hi mesh is quad-only while the fitting/texture code works on
    triangles. Split quads consistently with the renderer.
    """
    tris: list[np.ndarray] = []
    uv_idxs: list[np.ndarray] = []
    have_uv_idx = False

    if len(mesh.tris):
        tris.extend(v.astype(np.int32) for v in mesh.tris)
        if mesh.tri_uv_idx is not None:
            have_uv_idx = True
            uv_idxs.extend(v.astype(np.int32) for v in mesh.tri_uv_idx)

    if len(mesh.quads):
        for i, v_idx in enumerate(mesh.quads):
            tris.append(v_idx[[0, 1, 2]].astype(np.int32))
            tris.append(v_idx[[0, 2, 3]].astype(np.int32))
            if mesh.quad_uv_idx is not None:
                have_uv_idx = True
                uv_idx = mesh.quad_uv_idx[i]
                uv_idxs.append(uv_idx[[0, 1, 2]].astype(np.int32))
                uv_idxs.append(uv_idx[[0, 2, 3]].astype(np.int32))

    if not tris:
        raise ValueError("basis mesh has no triangle or quad facets")
    if have_uv_idx and len(uv_idxs) != len(tris):
        raise ValueError("basis mesh has incomplete per-facet UV indices")
    return (
        np.stack(tris).astype(np.int32),
        np.stack(uv_idxs).astype(np.int32) if have_uv_idx else None,
    )


class Basis:
    def __init__(self, root: str | Path | None = None):
        self.source = "ootp"
        self.orientation = "ootp"
        self.root = normalize_ootp_3d_path(root) if root is not None else get_ootp_3d_path()
        stem = "face_hi"
        self.tri = fgformat.TriMesh.read(str(self.root / f"{stem}.tri"))
        self.egm = fgformat.Egm.read(str(self.root / f"{stem}.egm"))
        self.egt = fgformat.Egt.read(str(self.root / f"{stem}.egt"))
        self.tris, self.tri_uv_idx = _triangulate_for_fit(self.tri)
        self.verts = self.tri.verts.astype(np.float32)
        self.modes = np.concatenate([self.egm.sym, self.egm.asym], 0)
        self.n_sym, self.n_asym = self.egm.sym.shape[0], self.egm.asym.shape[0]
        self.mean_tex = _load_rgb(
            self.root / f"{stem}.png",
            (self.egt.cols, self.egt.rows),
        )
        self.fim = _read_fim(self.root / f"{stem}.fim")

    @cached_property
    def vert_uv(self) -> np.ndarray:
        """Per-vertex UV (V,2) averaged from per-facet UVs."""
        if self.tri.uvs is None:
            raise ValueError("basis mesh has no UV coordinates")
        acc = np.zeros((len(self.verts), 2), np.float64)
        cnt = np.zeros(len(self.verts), np.int64)
        if self.tri_uv_idx is not None:
            vi = self.tris.ravel()
            ui = self.tri_uv_idx.ravel()
            np.add.at(acc, vi, self.tri.uvs[ui])
            np.add.at(cnt, vi, 1)
        else:
            count = min(len(self.tri.uvs), len(self.verts))
            acc[:count] = self.tri.uvs[:count]
            cnt[:count] = 1
        cnt[cnt == 0] = 1
        return (acc / cnt[:, None]).astype(np.float32)

    @cached_property
    def front_anchor_tri(self) -> int:
        """Pick a central, near-camera triangle for front-facing orientation."""
        cent = self.verts[self.tris].mean(1)
        xy = _screen_plane(cent, self.orientation)
        all_xy = _screen_plane(self.verts, self.orientation)
        xy0 = np.median(all_xy, 0)
        span = np.maximum(np.ptp(all_xy, axis=0), 1e-6)
        dist = np.linalg.norm((xy - xy0) / span, axis=1)
        z = cent[:, 2]
        z_lo, z_hi = np.percentile(z, [5, 95])
        zscore = (z - z_lo) / max(float(z_hi - z_lo), 1e-6)
        return int(np.argmax(zscore - 2.0 * dist))

    def fim_lookup(self, uv: np.ndarray) -> np.ndarray:
        """Map UV coords (N,2 in [0,1]) -> detail-texture UV (N,2)."""
        h, w, _ = self.fim.shape
        px = np.clip((uv[:, 0] * w).astype(int), 0, w - 1)
        py = np.clip(((1 - uv[:, 1]) * h).astype(int), 0, h - 1)
        return self.fim[py, px]

    @cached_property
    def _surface_points_cache(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """name -> (base_pos (3,), B (3,80)) if the mesh provides labels."""
        out = {}
        for name, (fidx, bary) in self.tri.surface_points.items():
            vidx = self.tris[fidx]
            w = np.asarray(bary, np.float32)
            base = w @ self.verts[vidx]
            b_modes = np.einsum("k,mkd->dm", w, self.modes[:, vidx, :])
            out[name] = (base, b_modes)
        return out

    def surface_points(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return self._surface_points_cache

    def shaped_verts(self, coeffs: np.ndarray) -> np.ndarray:
        """All vertices for coefficient vector (80,)."""
        return self.verts + np.einsum(
            "m,mvd->vd", coeffs.astype(np.float32), self.modes
        )

    def stat_texture(self, tex_coeffs: np.ndarray) -> np.ndarray:
        """(rows,cols,3) float RGB statistical texture from sym coeffs."""
        n = min(len(tex_coeffs), self.egt.sym.shape[0])
        s = self.mean_tex + np.einsum(
            "m,mrcd->rcd",
            tex_coeffs[:n].astype(np.float32),
            self.egt.sym[:n],
        )
        return np.clip(s, 0, 255)


def _screen_plane(verts: np.ndarray, orientation: str) -> np.ndarray:
    return np.stack([verts[:, 0], -verts[:, 1]], 1)


@lru_cache(maxsize=4)
def _get_basis_cached(root: str) -> Basis:
    """Load and cache OOTP's face_hi fitting basis."""
    return Basis(root)


def get_basis(root: str | Path | None = None) -> Basis:
    """Load and cache OOTP's face_hi fitting basis."""
    path = normalize_ootp_3d_path(root) if root is not None else get_ootp_3d_path()
    return _get_basis_cached(str(path))


get_basis.cache_clear = _get_basis_cached.cache_clear  # type: ignore[attr-defined]
