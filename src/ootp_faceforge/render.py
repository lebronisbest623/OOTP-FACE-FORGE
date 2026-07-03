"""Software renderer for .fg files.

Default mode uses OOTP's own face_hi mesh, texture basis, and FIM detail
mapping. The older FaceGen Modeller SI renderer remains available with
--basis si for debugging the fitting model itself.

Usage:
  python -m ootp_faceforge.render file.fg out.png [--basis ootp|si] [--size 512] [--aa 2]
"""
from __future__ import annotations

import argparse
import io
import struct
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .basis import Basis
from .fgformat import Egm, Egt, FgFile, TriMesh
from .texture import uv_px

TSIZE = 1024
OOTP_3D = Path(
    r"C:\Program Files (x86)\Steam\steamapps\common"
    r"\Out of the Park Baseball 27\data\facegen\3d"
)


@dataclass
class RenderAsset:
    name: str
    verts: np.ndarray
    modes: np.ndarray
    tris: np.ndarray
    src_px: np.ndarray
    tex: np.ndarray
    orientation: str


def _read_fim(path: Path) -> np.ndarray:
    data = path.read_bytes()
    w, h = struct.unpack_from("<2L", data, 8)
    return np.frombuffer(data, "<f4", w * h * 2, 64).reshape(h, w, 2).copy()


def _detail_modulation(fim: np.ndarray, fg: FgFile,
                       width: int, height: int) -> np.ndarray:
    mod = np.ones((height, width, 3), np.float32)
    if not fg.detail_jpeg:
        return mod
    detail = np.asarray(Image.open(io.BytesIO(fg.detail_jpeg)).convert("RGB"),
                        np.float32)
    dh, dw = detail.shape[:2]
    fim_up = cv2.resize(fim, (width, height), interpolation=cv2.INTER_NEAREST)
    ok = (fim_up[..., 0] >= 0) & (fim_up[..., 1] >= 0)
    dx = np.clip((fim_up[..., 0] * dw).astype(int), 0, dw - 1)
    dy = np.clip(((1 - fim_up[..., 1]) * dh).astype(int), 0, dh - 1)
    mod[ok] = detail[dy[ok], dx[ok]] / 64.0
    return mod


def _uv_to_px(uv: np.ndarray, width: int, height: int) -> np.ndarray:
    return np.stack([uv[..., 0] * width, (1.0 - uv[..., 1]) * height], -1)


def _triangulate_facets(mesh: TriMesh, tex_width: int,
                        tex_height: int) -> tuple[np.ndarray, np.ndarray]:
    """Return triangle vertex indices and matching per-corner texture pixels."""
    tris: list[np.ndarray] = []
    uvs: list[np.ndarray] = []

    def add_tri(v_idx: np.ndarray, uv: np.ndarray) -> None:
        tris.append(v_idx.astype(np.int32))
        uvs.append(_uv_to_px(uv.astype(np.float32), tex_width, tex_height))

    if len(mesh.tris):
        if mesh.tri_uv_idx is not None:
            for v_idx, uv_idx in zip(mesh.tris, mesh.tri_uv_idx):
                add_tri(v_idx, mesh.uvs[uv_idx])
        elif mesh.uvs is not None:
            for v_idx in mesh.tris:
                add_tri(v_idx, mesh.uvs[v_idx])

    if len(mesh.quads):
        if mesh.quad_uv_idx is not None:
            for v_idx, uv_idx in zip(mesh.quads, mesh.quad_uv_idx):
                uv = mesh.uvs[uv_idx]
                add_tri(v_idx[[0, 1, 2]], uv[[0, 1, 2]])
                add_tri(v_idx[[0, 2, 3]], uv[[0, 2, 3]])
        elif mesh.uvs is not None:
            for v_idx in mesh.quads:
                uv = mesh.uvs[v_idx]
                add_tri(v_idx[[0, 1, 2]], uv[[0, 1, 2]])
                add_tri(v_idx[[0, 2, 3]], uv[[0, 2, 3]])

    if not tris:
        raise ValueError("mesh has no renderable textured facets")
    return np.stack(tris), np.stack(uvs).astype(np.float32)


def _build_ootp_asset(fg: FgFile, stem: str = "face_hi",
                      apply_detail: bool = True,
                      texture_name: str | None = None) -> RenderAsset:
    mesh = TriMesh.read(str(OOTP_3D / f"{stem}.tri"))
    egm = Egm.read(str(OOTP_3D / f"{stem}.egm"))
    egt = Egt.read(str(OOTP_3D / f"{stem}.egt"))
    texture_name = texture_name or f"{stem}.png"
    base = np.asarray(Image.open(OOTP_3D / texture_name).convert("RGB"),
                      np.float32)
    h, w = base.shape[:2]

    coeff_tex = np.einsum("m,mrcd->rcd", fg.sym_tex.astype(np.float32), egt.sym)
    coeff_tex = cv2.resize(coeff_tex, (w, h), interpolation=cv2.INTER_LINEAR)
    tex = base + coeff_tex
    fim_path = OOTP_3D / f"{stem}.fim"
    if apply_detail and fim_path.exists():
        tex = tex * _detail_modulation(_read_fim(fim_path), fg, w, h)
    tex = np.clip(tex, 0, 255).astype(np.float32)

    tris, src_px = _triangulate_facets(mesh, w, h)
    modes = np.concatenate([egm.sym, egm.asym], 0)
    return RenderAsset(f"ootp-{stem}", mesh.verts, modes, tris, src_px, tex,
                       "ootp")


def _build_ootp_assets(fg: FgFile, include_eyes: bool) -> list[RenderAsset]:
    assets = [_build_ootp_asset(fg, "face_hi", apply_detail=True)]
    if include_eyes:
        assets.extend([
            _build_ootp_asset(fg, "eyer_hi", apply_detail=False,
                              texture_name="eyer_hi_brown.png"),
            _build_ootp_asset(fg, "eyel_hi", apply_detail=False,
                              texture_name="eyel_hi_brown.png"),
        ])
    return assets


def _build_si_asset(fg: FgFile) -> RenderAsset:
    basis = Basis()
    tex = _compose_si_texture(basis, fg)
    return RenderAsset(
        "facegen-si",
        basis.verts,
        basis.modes,
        basis.tris,
        uv_px(basis, TSIZE)[basis.tris].astype(np.float32),
        tex.astype(np.float32),
        "si",
    )


def _compose_si_texture(basis: Basis, fg: FgFile) -> np.ndarray:
    stat = basis.stat_texture(fg.sym_tex)
    stat = cv2.resize(stat, (TSIZE, TSIZE), interpolation=cv2.INTER_LINEAR)
    mod = _detail_modulation(basis.fim, fg, TSIZE, TSIZE)
    return np.clip(stat * mod, 0, 255)


def _screen_plane(verts: np.ndarray, orientation: str) -> np.ndarray:
    if orientation == "ootp":
        # OOTP face_hi uses a different in-plane convention than the SI demo
        # mesh: mesh +x is vertical in the front view and +y points left.
        return np.stack([verts[:, 0], -verts[:, 1]], 1)
    # FaceGen SI demo convention used by the fitting code.
    return np.stack([verts[:, 1], verts[:, 0]], 1)


def _screen_params(verts: np.ndarray, size: int,
                   orientation: str) -> tuple[np.ndarray, float]:
    xy = _screen_plane(verts, orientation)
    lo, hi = xy.min(0), xy.max(0)
    scale = (size * 0.88) / max(float((hi - lo).max()), 1e-6)
    return lo, scale


def _project_with_params(verts: np.ndarray, size: int, orientation: str,
                         lo: np.ndarray, scale: float) -> np.ndarray:
    return ((_screen_plane(verts, orientation) - lo) * scale + size * 0.06
            ).astype(np.float32)


def _shape_asset(asset: RenderAsset, fg: FgFile) -> np.ndarray:
    coeffs = np.concatenate([fg.sym_shape, fg.asym_shape]).astype(np.float32)
    n_modes = min(len(coeffs), asset.modes.shape[0])
    return asset.verts + np.einsum(
        "m,mvd->vd", coeffs[:n_modes], asset.modes[:n_modes]
    )


def _draw_asset(asset: RenderAsset, verts: np.ndarray, scr: np.ndarray,
                out: np.ndarray, covered: np.ndarray, shade: bool) -> None:
    canvas = out.shape[0]
    tri_v = verts[asset.tris]
    tri_s = scr[asset.tris]
    e1 = tri_v[:, 1] - tri_v[:, 0]
    e2 = tri_v[:, 2] - tri_v[:, 0]
    normals = np.cross(e1, e2)
    nlen = np.maximum(np.linalg.norm(normals, axis=1), 1e-9)
    nz = normals[:, 2] / nlen
    sign = 1.0 if (nz > 0).sum() >= (nz < 0).sum() else -1.0
    facing = nz * sign

    z = tri_v[:, :, 2].mean(1)
    order = np.argsort(z)
    keep = facing > 0.015
    order = order[keep[order]]

    for ti in order:
        dst = tri_s[ti]
        x0, y0 = np.maximum(np.floor(dst.min(0)).astype(int), 0)
        x1 = min(int(np.ceil(dst[:, 0].max())) + 1, canvas)
        y1 = min(int(np.ceil(dst[:, 1].max())) + 1, canvas)
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue

        M = cv2.getAffineTransform(asset.src_px[ti].astype(np.float32),
                                   (dst - [x0, y0]).astype(np.float32))
        patch = cv2.warpAffine(asset.tex, M, (x1 - x0, y1 - y0),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillConvexPoly(mask, np.round(dst - [x0, y0]).astype(np.int32), 1)
        mb = mask.astype(bool)

        lit = 1.0
        if shade:
            lit = 0.64 + 0.36 * max(float(facing[ti]), 0.0)
        out[y0:y1, x0:x1][mb] = patch[mb] * lit
        covered[y0:y1, x0:x1] |= mb


def _render_assets(assets: list[RenderAsset], fg: FgFile, size: int,
                   shade: bool, aa: int) -> np.ndarray:
    aa = max(1, int(aa))
    canvas = int(size) * aa
    out = np.full((canvas, canvas, 3), 190.0, np.float32)
    covered = np.zeros((canvas, canvas), bool)
    shaped = [_shape_asset(asset, fg) for asset in assets]
    lo, scale = _screen_params(shaped[0], canvas, assets[0].orientation)
    for asset, verts in zip(assets, shaped):
        scr = _project_with_params(verts, canvas, asset.orientation, lo, scale)
        _draw_asset(asset, verts, scr, out, covered, shade)

    out[~covered] = 190.0
    out = np.clip(out, 0, 255).astype(np.uint8)
    if aa > 1:
        out = cv2.resize(out, (size, size), interpolation=cv2.INTER_AREA)
    return out


def render(fg: FgFile, basis_name: str = "ootp", size: int = 512,
           shade: bool = True, aa: int = 2,
           include_eyes: bool = True) -> tuple[np.ndarray, str]:
    assets = (_build_ootp_assets(fg, include_eyes)
              if basis_name == "ootp"
              else [_build_si_asset(fg)])
    return _render_assets(assets, fg, size=size, shade=shade, aa=aa), "+".join(
        asset.name for asset in assets
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render an OOTP/FaceGen .fg preview.")
    p.add_argument("fg")
    p.add_argument("out")
    p.add_argument("--basis", choices=("ootp", "si"), default="ootp")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--aa", type=int, default=2)
    p.add_argument("--no-eyes", action="store_true",
                   help="Only affects --basis ootp.")
    p.add_argument("--shade", action="store_true",
                   help="Compatibility flag; shading is on unless --flat is used.")
    p.add_argument("--flat", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    fg = FgFile.read(args.fg)
    img, asset_name = render(
        fg,
        basis_name=args.basis,
        size=args.size,
        shade=not args.flat,
        aa=args.aa,
        include_eyes=not args.no_eyes,
    )
    Image.fromarray(img).save(args.out)
    print("rendered", args.out,
          "| basis:", asset_name,
          "| tex coeff norm:", round(float(np.linalg.norm(fg.sym_tex)), 2),
          "| shape norm:", round(float(np.linalg.norm(fg.sym_shape)), 2))


if __name__ == "__main__":
    main()
