"""FaceGen binary file formats: .fg read/write, .tri/.egm/.egt read.

Spec: https://facegen.com/dl/sdk/doc/manual/fileformats.html
All little-endian.
"""
from __future__ import annotations
import struct
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------- .fg

@dataclass
class FgFile:
    geo_basis_version: int = 0
    tex_basis_version: int = 0
    sym_shape: np.ndarray = field(default_factory=lambda: np.zeros(50))
    asym_shape: np.ndarray = field(default_factory=lambda: np.zeros(30))
    sym_tex: np.ndarray = field(default_factory=lambda: np.zeros(50))
    asym_tex: np.ndarray = field(default_factory=lambda: np.zeros(0))
    detail_jpeg: bytes | None = None  # raw JPEG bytes or None

    @classmethod
    def read(cls, path: str) -> "FgFile":
        with open(path, "rb") as f:
            data = f.read()
        magic = data[:8]
        if magic != b"FRFG0001":
            raise ValueError(f"bad magic {magic!r}")
        gv, tv, ss, sa, ts, ta, res, dt = struct.unpack_from("<8L", data, 8)
        off = 8 + 32
        n = ss + sa + ts + ta
        coeffs = np.frombuffer(data, dtype="<i2", count=n, offset=off) / 1000.0
        off += 2 * n
        jpeg = None
        if dt:
            (size,) = struct.unpack_from("<L", data, off)
            off += 4
            jpeg = data[off : off + size]
        i = 0
        parts = []
        for cnt in (ss, sa, ts, ta):
            parts.append(coeffs[i : i + cnt].copy())
            i += cnt
        return cls(gv, tv, *parts, detail_jpeg=jpeg)

    def write(self, path: str) -> None:
        ss, sa = len(self.sym_shape), len(self.asym_shape)
        ts, ta = len(self.sym_tex), len(self.asym_tex)
        dt = 1 if self.detail_jpeg else 0
        out = bytearray()
        out += b"FRFG0001"
        out += struct.pack("<8L", self.geo_basis_version, self.tex_basis_version,
                           ss, sa, ts, ta, 0, dt)
        coeffs = np.concatenate([self.sym_shape, self.asym_shape,
                                 self.sym_tex, self.asym_tex])
        q = np.clip(np.round(coeffs * 1000), -32768, 32767).astype("<i2")
        out += q.tobytes()
        if dt:
            out += struct.pack("<L", len(self.detail_jpeg))
            out += self.detail_jpeg
        with open(path, "wb") as f:
            f.write(bytes(out))


# ---------------------------------------------------------------- .tri

def _read_str(data: bytes, off: int) -> tuple[str, int]:
    (n,) = struct.unpack_from("<L", data, off)
    s = data[off + 4 : off + 4 + n].rstrip(b"\0").decode("latin-1")
    return s, off + 4 + n


@dataclass
class TriMesh:
    verts: np.ndarray          # (V+K, 3) float32
    tris: np.ndarray           # (T, 3) int32
    quads: np.ndarray          # (Q, 4) int32
    uvs: np.ndarray | None     # per-vertex (V,2) or per-facet (X,2)
    uv_per_facet: bool = False
    tri_uv_idx: np.ndarray | None = None
    quad_uv_idx: np.ndarray | None = None
    labeled_verts: dict = field(default_factory=dict)   # name -> vertex index
    surface_points: dict = field(default_factory=dict)  # name -> (tri/quad idx, bary)
    V: int = 0

    @classmethod
    def read(cls, path: str) -> "TriMesh":
        with open(path, "rb") as f:
            data = f.read()
        if data[:8] != b"FRTRI003":
            raise ValueError(f"bad magic {data[:8]!r}")
        V, T, Q, LV, LS, X, ext, Md, Ms, K = struct.unpack_from("<10i", data, 8)
        off = 8 + 40 + 16
        has_uv = bool(ext & 1)
        labels16 = bool(ext & 2)

        verts = np.frombuffer(data, "<f4", (V + K) * 3, off).reshape(-1, 3).copy()
        off += (V + K) * 12
        tris = np.frombuffer(data, "<i4", T * 3, off).reshape(-1, 3).copy()
        off += T * 12
        quads = np.frombuffer(data, "<i4", Q * 4, off).reshape(-1, 4).copy()
        off += Q * 16

        labeled = {}
        for _ in range(LV):
            (idx,) = struct.unpack_from("<i", data, off)
            off += 4
            name, off = _read_str(data, off)
            labeled[name] = idx
        surface = {}
        for _ in range(LS):
            (idx,) = struct.unpack_from("<i", data, off)
            off += 4
            bary = struct.unpack_from("<3f", data, off)
            off += 12
            name, off = _read_str(data, off)
            surface[name] = (idx, bary)

        uvs = None
        uv_per_facet = X > 0
        tri_uv_idx = quad_uv_idx = None
        if has_uv:
            if uv_per_facet:
                uvs = np.frombuffer(data, "<f4", X * 2, off).reshape(-1, 2).copy()
                off += X * 8
                tri_uv_idx = np.frombuffer(data, "<i4", T * 3, off).reshape(-1, 3).copy()
                off += T * 12
                quad_uv_idx = np.frombuffer(data, "<i4", Q * 4, off).reshape(-1, 4).copy()
                off += Q * 16
            else:
                uvs = np.frombuffer(data, "<f4", V * 2, off).reshape(-1, 2).copy()
                off += V * 8

        return cls(verts, tris, quads, uvs, uv_per_facet,
                   tri_uv_idx, quad_uv_idx, labeled, surface, V)


# ---------------------------------------------------------------- .egm

@dataclass
class Egm:
    basis_version: int
    sym: np.ndarray   # (S, V, 3) float32 deltas
    asym: np.ndarray  # (A, V, 3)

    @classmethod
    def read(cls, path: str) -> "Egm":
        with open(path, "rb") as f:
            data = f.read()
        if data[:8] != b"FREGM002":
            raise ValueError(f"bad magic {data[:8]!r}")
        V, S, A, ver = struct.unpack_from("<4L", data, 8)
        off = 8 + 16 + 40

        def read_modes(n):
            nonlocal off
            modes = np.empty((n, V, 3), np.float32)
            for i in range(n):
                (scale,) = struct.unpack_from("<f", data, off)
                off += 4
                d = np.frombuffer(data, "<i2", V * 3, off).reshape(-1, 3)
                off += V * 6
                modes[i] = d.astype(np.float32) * scale
            return modes

        sym = read_modes(S)
        asym = read_modes(A)
        return cls(ver, sym, asym)


# ---------------------------------------------------------------- .egt

@dataclass
class Egt:
    basis_version: int
    rows: int
    cols: int
    sym: np.ndarray   # (S, R, C, 3) float32 deltas (RGB)
    asym: np.ndarray

    @classmethod
    def read(cls, path: str) -> "Egt":
        with open(path, "rb") as f:
            data = f.read()
        if data[:8] != b"FREGT003":
            raise ValueError(f"bad magic {data[:8]!r}")
        R, C, S, A, ver = struct.unpack_from("<5L", data, 8)
        off = 8 + 20 + 36
        npx = R * C

        def read_modes(n):
            nonlocal off
            modes = np.empty((n, R, C, 3), np.float32)
            for i in range(n):
                (scale,) = struct.unpack_from("<f", data, off)
                off += 4
                for ch in range(3):
                    img = np.frombuffer(data, "i1", npx, off).reshape(R, C)
                    off += npx
                    modes[i, :, :, ch] = img.astype(np.float32) * scale
            return modes

        sym = read_modes(S)
        asym = read_modes(A)
        return cls(ver, R, C, sym, asym)
