"""Bake FaceGen-style 3D eyeglasses into an OOTP detail map.

FaceGen Modeller does not store eyeglasses as a hidden section inside ``.fg``.
It renders a separate accessory mesh (``Glasses.tri/.egm/.tga``) that shares the
face shape coefficients. OOTP ``.fg`` files only carry face coefficients plus a
detail JPEG, so the standalone export path has to render the accessory and bake
that render back into the face detail texture.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
import struct
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from urllib.parse import unquote

import cv2
import numpy as np
from PIL import Image

from . import render
from .basis import Basis
from .fgformat import Egm, FgFile, TriMesh
from .paths import get_ootp_3d_path, workspace_root
from .texture import detail_px


TRUE_FRAME_RGB = {
    "black": np.array([24.0, 23.0, 22.0], np.float32),
    "brown": np.array([70.0, 46.0, 32.0], np.float32),
    "red": np.array([128.0, 26.0, 24.0], np.float32),
    "blue": np.array([34.0, 44.0, 104.0], np.float32),
    "silver": np.array([126.0, 124.0, 118.0], np.float32),
    "gold": np.array([170.0, 138.0, 34.0], np.float32),
}


@dataclass
class BakeResult:
    applied: bool
    source: str = ""
    color_name: str = ""
    screen_pixels: int = 0
    detail_pixels: int = 0
    reason: str = ""


@dataclass
class _Accessory:
    source: Path
    verts: np.ndarray
    modes: np.ndarray
    tris: np.ndarray
    src_px: np.ndarray
    rgba: np.ndarray


@dataclass
class _SolidAccessory:
    source: str
    verts: np.ndarray
    tris: np.ndarray


_TEMPLATE_REGISTRY = Path(__file__).resolve().parent / "assets" / "glasses_templates.json"


def _expand_template_path(raw: str) -> Path:
    return Path(os.path.expandvars(raw)).expanduser()


@lru_cache(maxsize=1)
def glasses_template_registry() -> dict[str, dict]:
    entries: dict[str, dict] = {}
    paths = [
        _TEMPLATE_REGISTRY,
        workspace_root() / "glasses_templates" / "manifest.json",
    ]
    for path in paths:
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for entry in data.get("templates", []):
            if not isinstance(entry, dict):
                continue
            keys = [str(entry.get("id", "")).strip()]
            keys.extend(str(a).strip() for a in entry.get("aliases", []))
            for key in keys:
                if key:
                    entries[key.lower()] = dict(entry)
    return entries


def glasses_template_info(name_or_path: str | Path | None) -> dict | None:
    if name_or_path is None:
        return None
    raw = str(name_or_path).strip()
    if not raw:
        return None
    expanded = _expand_template_path(raw)
    if expanded.exists():
        return None
    return glasses_template_registry().get(raw.lower())


_STATUS_RANK = {"preferred": 0, "usable": 1, "weak": 2}
_STYLE_FALLBACKS = {"oval": ("oval", "round", "rectangular"),
                    "round": ("round", "oval", "rectangular")}


def default_template_for_style(style: str) -> str | None:
    """Best registered template id for a detected frame style.

    Only templates whose asset path actually resolves count, so a machine
    without the user's converted CC meshes silently falls back to the
    procedural glasses mesh."""
    style = (style or "").strip().lower()
    wanted = _STYLE_FALLBACKS.get(style, (style, "rectangular"))
    best: tuple[int, int, str] | None = None
    seen: set[str] = set()
    for entry in glasses_template_registry().values():
        tid = str(entry.get("id", "")).strip()
        if not tid or tid in seen:
            continue
        seen.add(tid)
        estyle = str(entry.get("style", "")).strip().lower()
        if estyle not in wanted:
            continue
        path = entry.get("path")
        if not isinstance(path, str) or not _has_custom_mesh_asset(path):
            continue
        rank = (_STATUS_RANK.get(str(entry.get("status", "usable")), 1),
                wanted.index(estyle))
        if best is None or rank < best[:2]:
            best = (*rank, tid)
    return best[2] if best else None


def resolve_glasses_mesh_asset(name_or_path: str | Path | None) -> Path | None:
    if name_or_path is None:
        return None
    raw = str(name_or_path).strip()
    if not raw:
        return None
    expanded = _expand_template_path(raw)
    if expanded.exists():
        return expanded
    info = glasses_template_info(raw)
    if info is None:
        return expanded
    path = info.get("path") or info.get("source")
    if not isinstance(path, str) or not path.strip():
        return expanded
    return _expand_template_path(path)


def _candidate_dirs() -> list[Path]:
    out: list[Path] = []
    env = os.environ.get("FACEGEN_ACCESSORIES_DIR")
    if env:
        out.append(Path(env))
    for root in (
        os.environ.get("ProgramFiles"),
        os.environ.get("ProgramFiles(x86)"),
        r"C:\Program Files",
        r"C:\Program Files (x86)",
    ):
        if not root:
            continue
        base = Path(root) / "FaceGen"
        for app in ("Modeller Demo 3", "Modeller 3", "Artist Demo 3", "Artist 3"):
            out.append(base / app / "data" / "csam" / "Animate" / "Accessories")
    # Preserve order but drop duplicates.
    seen = set()
    uniq = []
    for path in out:
        key = str(path).lower()
        if key not in seen:
            seen.add(key)
            uniq.append(path)
    return uniq


def find_accessory_dir(path: str | Path | None = None) -> Path | None:
    candidates = [Path(path)] if path else _candidate_dirs()
    for cand in candidates:
        if all((cand / name).is_file()
               for name in ("Glasses.tri", "Glasses.egm", "Glasses.tga")):
            return cand
    return None


def available(path: str | Path | None = None) -> bool:
    """The mesh route is always available: with no explicit asset path the
    license-clean procedural glasses mesh is used, so only an explicit but
    unresolvable asset path reports unavailable."""
    if path is None:
        return True
    return (find_accessory_dir(path) is not None
            or _has_custom_mesh_asset(path))


@lru_cache(maxsize=4)
def _load_accessory(path_str: str) -> _Accessory:
    root = Path(path_str)
    mesh = TriMesh.read(str(root / "Glasses.tri"))
    egm = Egm.read(str(root / "Glasses.egm"))
    rgba = np.asarray(Image.open(root / "Glasses.tga").convert("RGBA"),
                      np.float32)
    h, w = rgba.shape[:2]
    tris, src_px = render._triangulate_facets(mesh, w, h)
    return _Accessory(
        source=root,
        verts=mesh.verts.astype(np.float32),
        modes=np.concatenate([egm.sym, egm.asym], 0).astype(np.float32),
        tris=tris.astype(np.int32),
        src_px=src_px.astype(np.float32),
        rgba=rgba,
    )


def _shape(verts: np.ndarray, modes: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    coeffs = coeffs.astype(np.float32)
    n = min(len(coeffs), modes.shape[0])
    return verts + np.einsum("m,mvd->vd", coeffs[:n], modes[:n])


def _detail_modulation_from_array(fim: np.ndarray | None, D: np.ndarray,
                                  width: int, height: int) -> np.ndarray:
    mod = np.ones((height, width, 3), np.float32)
    if fim is None:
        return mod
    detail = D.astype(np.float32)
    dh, dw = detail.shape[:2]
    fim_up = cv2.resize(fim, (width, height), interpolation=cv2.INTER_NEAREST)
    ok = (fim_up[..., 0] >= 0) & (fim_up[..., 1] >= 0)
    dx = np.clip((fim_up[..., 0] * dw).astype(int), 0, dw - 1)
    dy = np.clip(((1.0 - fim_up[..., 1]) * dh).astype(int), 0, dh - 1)
    mod[ok] = detail[dy[ok], dx[ok]] / 64.0
    return mod


@dataclass
class _FaceMaps:
    rgb: np.ndarray
    detail_xy: np.ndarray
    valid: np.ndarray
    zbuf: np.ndarray
    lo: np.ndarray
    scale: float


def _render_face_maps(basis: Basis, shape_coeffs: np.ndarray,
                      tex_coeffs: np.ndarray, D: np.ndarray,
                      canvas: int) -> _FaceMaps:
    root = get_ootp_3d_path()
    source = render._load_ootp_source_asset(str(root), "face_hi", "face_hi.png")
    h, w = source.base_tex.shape[:2]
    coeff_tex = np.einsum(
        "m,mrcd->rcd",
        tex_coeffs[:source.tex_modes.shape[0]].astype(np.float32),
        source.tex_modes[:len(tex_coeffs)],
    )
    coeff_tex = cv2.resize(coeff_tex, (w, h), interpolation=cv2.INTER_LINEAR)
    tex = np.clip(
        (source.base_tex + coeff_tex)
        * _detail_modulation_from_array(source.fim, D, w, h),
        0,
        255,
    ).astype(np.float32)

    verts = _shape(source.verts, source.modes, shape_coeffs)
    lo, scale = render._screen_params(verts, canvas, source.orientation)
    scr = render._project_with_params(
        verts, canvas, source.orientation, lo, scale
    )
    dpx, dvalid = detail_px(basis, D.shape[0])

    rgb = np.full((canvas, canvas, 3), 190.0, np.float32)
    detail_xy = np.full((canvas, canvas, 2), -1.0, np.float32)
    valid = np.zeros((canvas, canvas), bool)
    zbuf = np.full((canvas, canvas), -np.inf, np.float32)

    tri_v = verts[source.tris]
    tri_s = scr[source.tris]
    tri_d = dpx[source.tris]
    tri_dvalid = dvalid[source.tris].all(1)
    e1 = tri_v[:, 1] - tri_v[:, 0]
    e2 = tri_v[:, 2] - tri_v[:, 0]
    normals = np.cross(e1, e2)
    nlen = np.maximum(np.linalg.norm(normals, axis=1), 1e-9)
    nz = normals[:, 2] / nlen
    sign = 1.0 if (nz > 0).sum() >= (nz < 0).sum() else -1.0
    facing = nz * sign
    order = np.argsort(tri_v[:, :, 2].mean(1))
    keep = (facing > 0.015) & tri_dvalid
    order = order[keep[order]]

    for ti in order:
        dst = tri_s[ti]
        x0, y0 = np.maximum(np.floor(dst.min(0)).astype(int), 0)
        x1 = min(int(np.ceil(dst[:, 0].max())) + 1, canvas)
        y1 = min(int(np.ceil(dst[:, 1].max())) + 1, canvas)
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        dst_local = (dst - [x0, y0]).astype(np.float32)
        M_tex = cv2.getAffineTransform(source.src_px[ti].astype(np.float32),
                                       dst_local)
        patch = cv2.warpAffine(tex, M_tex, (x1 - x0, y1 - y0),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillConvexPoly(mask, np.round(dst_local).astype(np.int32), 1)
        mb = mask.astype(bool)
        if not mb.any():
            continue
        zz = render._tri_depth(dst_local, tri_v[ti][:, 2], x1 - x0, y1 - y0)
        zb = zbuf[y0:y1, x0:x1]
        nearer = mb & (zz > zb)
        if not nearer.any():
            continue

        M_detail = cv2.getAffineTransform(dst_local, tri_d[ti].astype(np.float32))
        yy, xx = np.mgrid[0:y1 - y0, 0:x1 - x0].astype(np.float32)
        dxy = np.empty((y1 - y0, x1 - x0, 2), np.float32)
        dxy[..., 0] = M_detail[0, 0] * xx + M_detail[0, 1] * yy + M_detail[0, 2]
        dxy[..., 1] = M_detail[1, 0] * xx + M_detail[1, 1] * yy + M_detail[1, 2]

        lit = 0.64 + 0.36 * max(float(facing[ti]), 0.0)
        rgb[y0:y1, x0:x1][nearer] = patch[nearer] * lit
        detail_xy[y0:y1, x0:x1][nearer] = dxy[nearer]
        valid[y0:y1, x0:x1][nearer] = True
        zb[nearer] = zz[nearer]

    return _FaceMaps(rgb, detail_xy, valid, zbuf, lo, scale)


def _frame_rgb(color_name: str, fallback: str = "brown") -> np.ndarray:
    return TRUE_FRAME_RGB.get(color_name, TRUE_FRAME_RGB[fallback]).astype(np.float32)


def _eye_screen_points(
    basis: Basis,
    shape_coeffs: np.ndarray,
    lo: np.ndarray,
    scale: float,
    canvas: int,
) -> np.ndarray | None:
    try:
        from .calibrate import calibrated_pairs
    except Exception:
        return None

    verts = basis.shaped_verts(shape_coeffs)
    pts = []
    for li, ti, bary in calibrated_pairs(basis):
        if int(li) not in (468, 473):
            continue
        vidx = basis.tris[int(ti)]
        p3 = np.asarray(bary, np.float32) @ verts[vidx]
        scr = (render._screen_plane(p3[None], "ootp")[0] - lo) * scale
        scr = scr + canvas * 0.06
        pts.append(scr.astype(np.float32))
    if len(pts) < 2:
        return None
    return np.stack(pts).astype(np.float32)


def _eye_target_screen(
    basis: Basis,
    shape_coeffs: np.ndarray,
    lo: np.ndarray,
    scale: float,
    canvas: int,
) -> np.ndarray | None:
    """Front-render screen center between MediaPipe iris landmarks 468/473."""
    pts = _eye_screen_points(basis, shape_coeffs, lo, scale, canvas)
    if pts is None:
        return None
    return np.mean(pts, axis=0).astype(np.float32)


def _landmark_points_3d(
    basis: Basis,
    shape_coeffs: np.ndarray,
    landmark_ids: tuple[int, ...],
) -> dict[int, np.ndarray]:
    try:
        from .calibrate import calibrated_pairs
    except Exception:
        return {}

    wanted = {int(x) for x in landmark_ids}
    verts = basis.shaped_verts(shape_coeffs)
    out: dict[int, np.ndarray] = {}
    for li, ti, bary in calibrated_pairs(basis):
        li = int(li)
        if li not in wanted:
            continue
        vidx = basis.tris[int(ti)]
        out[li] = (np.asarray(bary, np.float32) @ verts[vidx]).astype(
            np.float32
        )
    return out


_GLTF_COMPONENT_DTYPE = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
_GLTF_TYPE_SIZE = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT4": 16,
}


def _find_gltf_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    if p.is_file() and p.suffix.lower() == ".gltf":
        return p
    if p.is_dir():
        matches = sorted(p.rglob("*.gltf"))
        return matches[0] if matches else None
    return None


_CONVERTIBLE_MODEL_SUFFIXES = {".fbx", ".obj", ".dae"}


def _find_convertible_model_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    if p.is_file() and p.suffix.lower() in _CONVERTIBLE_MODEL_SUFFIXES:
        return p
    if p.is_dir():
        matches = sorted(
            m for m in p.rglob("*")
            if m.is_file() and m.suffix.lower() in _CONVERTIBLE_MODEL_SUFFIXES
        )
        return matches[0] if matches else None
    return None


def _blender_exe() -> Path | None:
    env = os.environ.get("OOTP_FACEFORGE_BLENDER") or os.environ.get("BLENDER_EXE")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env))
    which = shutil.which("blender")
    if which:
        candidates.append(Path(which))
    for root in (
        os.environ.get("ProgramFiles"),
        os.environ.get("ProgramFiles(x86)"),
        r"C:\Program Files",
        r"C:\Program Files (x86)",
    ):
        if not root:
            continue
        base = Path(root) / "Blender Foundation"
        if base.is_dir():
            candidates.extend(sorted(base.rglob("blender.exe"), reverse=True))
    seen: set[str] = set()
    for cand in candidates:
        key = os.path.normcase(str(cand))
        if key in seen:
            continue
        seen.add(key)
        if cand.is_file():
            return cand
    return None


def _asset_cache_key(path: Path) -> str:
    try:
        st = path.stat()
        raw = f"mesh-convert-v2|{path.resolve()}|{st.st_mtime_ns}|{st.st_size}"
    except OSError:
        raw = f"mesh-convert-v2|{path}"
    return hashlib.sha1(raw.encode("utf-8", "ignore")).hexdigest()[:16]


def _safe_extract_zip(zf: zipfile.ZipFile, out_dir: Path) -> None:
    root = out_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for info in zf.infolist():
        rel = PurePosixPath(info.filename)
        if rel.is_absolute() or any(part in ("", "..") for part in rel.parts):
            continue
        target = (root / Path(*rel.parts)).resolve()
        if target != root and root not in target.parents:
            continue
        if info.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(zf.read(info))


def _extract_nested_zips(root: Path, max_rounds: int = 2) -> None:
    base = root.resolve()
    for _ in range(max(0, int(max_rounds))):
        changed = False
        for zpath in sorted(base.rglob("*.zip")):
            out_dir = zpath.with_suffix("")
            marker = out_dir / ".faceforge_extracted"
            if marker.exists():
                continue
            try:
                with zipfile.ZipFile(zpath) as zf:
                    _safe_extract_zip(zf, out_dir)
                marker.write_text("ok\n", encoding="utf-8")
                changed = True
            except (OSError, zipfile.BadZipFile):
                continue
        if not changed:
            break


def _convert_model_to_gltf(model_path: Path, out_dir: Path) -> Path | None:
    blender = _blender_exe()
    if blender is None:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    gltf = out_dir / (model_path.stem + ".gltf")
    if gltf.is_file():
        return gltf
    suffix = model_path.suffix.lower()
    if suffix == ".fbx":
        import_line = f"bpy.ops.import_scene.fbx(filepath={str(model_path)!r})"
    elif suffix == ".obj":
        import_line = (
            "bpy.ops.wm.obj_import(filepath="
            f"{str(model_path)!r}"
            ") if hasattr(bpy.ops.wm, 'obj_import') "
            "else bpy.ops.import_scene.obj(filepath="
            f"{str(model_path)!r}"
            ")"
        )
    elif suffix == ".dae":
        import_line = f"bpy.ops.wm.collada_import(filepath={str(model_path)!r})"
    else:
        return None
    expr = f"""
import addon_utils
import bpy
for mod in ('io_scene_fbx', 'io_scene_obj', 'io_scene_gltf2'):
    try:
        addon_utils.enable(mod, default_set=False, persistent=False)
    except Exception:
        pass
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
{import_line}
mesh_objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
tri_count = sum(sum(max(1, len(poly.vertices) - 2) for poly in o.data.polygons)
                for o in mesh_objs)
max_tris = 70000
if tri_count > max_tris:
    ratio = max(0.02, max_tris / max(tri_count, 1))
    for obj in mesh_objs:
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        mod = obj.modifiers.new('FaceForge_Decimate', 'DECIMATE')
        mod.ratio = ratio
        try:
            bpy.ops.object.modifier_apply(modifier=mod.name)
        except Exception:
            pass
bpy.ops.export_scene.gltf(filepath={str(gltf)!r}, export_format='GLTF_SEPARATE')
"""
    try:
        run = subprocess.run(
            [str(blender), "--background", "--factory-startup",
             "--python-expr", expr],
            cwd=str(out_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if run.returncode != 0 and not gltf.is_file():
        return None
    return gltf if gltf.is_file() else None


def _convert_mesh_asset_to_gltf_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    cache = workspace_root() / "cache" / "glasses_fbx" / _asset_cache_key(p)
    out_dir = cache / "converted"
    if p.is_file() and p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as zf:
            names = sorted(
                n for n in zf.namelist()
                if Path(n).suffix.lower() in _CONVERTIBLE_MODEL_SUFFIXES
            )
            nested = sorted(n for n in zf.namelist() if n.lower().endswith(".zip"))
            if not names and not nested:
                return None
            extract_dir = cache / "extracted"
            model = (
                extract_dir / Path(*PurePosixPath(names[0]).parts)
                if names else None
            )
            if model is None or not model.is_file():
                _safe_extract_zip(zf, extract_dir)
            _extract_nested_zips(extract_dir)
            model = _find_convertible_model_path(extract_dir)
            if model is None:
                return None
            return _convert_model_to_gltf(model, out_dir)
    model = _find_convertible_model_path(p)
    if model is None:
        return None
    return _convert_model_to_gltf(model, out_dir)


def _auto_align_external_yaw(raw: np.ndarray) -> np.ndarray:
    """Rotate models whose width axis is diagonal in the X/Z plane."""
    if raw.ndim != 2 or raw.shape[1] != 3 or len(raw) < 16:
        return raw
    xz = raw[:, [0, 2]].astype(np.float32)
    center = np.median(xz, axis=0)
    xzc = xz - center
    try:
        cov = np.cov(xzc.T)
        vals, vecs = np.linalg.eigh(cov)
    except Exception:
        return raw
    lo = max(float(vals.min()), 1e-9)
    hi = float(vals.max())
    if hi / lo < 1.35:
        return raw
    width = vecs[:, int(np.argmax(vals))].astype(np.float32)
    if width[0] < 0:
        width *= -1.0
    angle = float(np.degrees(np.arctan2(width[1], width[0])))
    if abs(angle) < 20.0 or abs(angle) > 70.0:
        return raw
    depth = np.array([-width[1], width[0]], np.float32)
    out = raw.copy().astype(np.float32)
    out[:, 0] = xzc @ width
    out[:, 2] = xzc @ depth
    return out


def _read_gltf_source(
    path: str | Path | None,
) -> tuple[dict, Path | None, zipfile.ZipFile | None, str] | None:
    if path is None:
        return None
    p = Path(path)
    if p.is_file() and p.suffix.lower() == ".zip":
        zf = zipfile.ZipFile(p)
        names = sorted(n for n in zf.namelist() if n.lower().endswith(".gltf"))
        if not names:
            zf.close()
            converted = _convert_mesh_asset_to_gltf_path(p)
            if converted is None:
                return None
            text = converted.read_text(encoding="utf-8")
            return (
                json.loads(text),
                converted,
                None,
                f"{p}!converted_fbx:{converted.name}",
            )
        text = zf.read(names[0]).decode("utf-8")
        return json.loads(text), Path(names[0]), zf, f"{p}!{names[0]}"
    gltf = _find_gltf_path(p)
    if gltf is None:
        converted = _convert_mesh_asset_to_gltf_path(p)
        if converted is None:
            return None
        gltf = converted
    return json.loads(gltf.read_text(encoding="utf-8")), gltf, None, str(gltf)


def _gltf_base64_payload(uri: str) -> bytes | None:
    if not uri.startswith("data:"):
        return None
    _, _, payload = uri.partition(",")
    return base64.b64decode(payload)


def _load_gltf_buffers(
    gltf: dict,
    gltf_path: Path | None,
    zf: zipfile.ZipFile | None,
) -> list[bytes]:
    buffers = []
    base_dir = (gltf_path.parent if gltf_path is not None else Path("."))
    for buf in gltf.get("buffers", []):
        uri = str(buf.get("uri", ""))
        raw = _gltf_base64_payload(uri)
        if raw is None:
            rel = unquote(uri)
            if zf is not None:
                name = str((base_dir / rel).as_posix())
                raw = zf.read(name)
            elif gltf_path is not None:
                raw = (base_dir / rel).read_bytes()
            else:
                raise ValueError("external glTF buffer needs a path")
        buffers.append(raw)
    return buffers


def _read_gltf_accessor(gltf: dict, buffers: list[bytes], idx: int) -> np.ndarray:
    acc = gltf["accessors"][idx]
    bv = gltf["bufferViews"][acc["bufferView"]]
    comp_dtype = np.dtype(_GLTF_COMPONENT_DTYPE[int(acc["componentType"])])
    ncomp = _GLTF_TYPE_SIZE[str(acc["type"])]
    count = int(acc["count"])
    stride = int(bv.get("byteStride", comp_dtype.itemsize * ncomp))
    offset = int(bv.get("byteOffset", 0)) + int(acc.get("byteOffset", 0))
    raw = buffers[int(bv["buffer"])]
    if stride == comp_dtype.itemsize * ncomp:
        arr = np.frombuffer(raw, dtype=comp_dtype, count=count * ncomp,
                            offset=offset).reshape(count, ncomp)
    else:
        arr = np.empty((count, ncomp), comp_dtype)
        for i in range(count):
            start = offset + i * stride
            arr[i] = np.frombuffer(raw, dtype=comp_dtype, count=ncomp,
                                   offset=start)
    if str(acc["type"]) == "SCALAR":
        return arr[:, 0].copy()
    return arr.copy()


def _node_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.asarray(node["matrix"], np.float32).reshape(4, 4).T
    m = np.eye(4, dtype=np.float32)
    if "scale" in node:
        sx, sy, sz = [float(x) for x in node["scale"]]
        m = m @ np.diag([sx, sy, sz, 1.0]).astype(np.float32)
    if "rotation" in node:
        x, y, z, w = [float(v) for v in node["rotation"]]
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        r = np.array([
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy), 0],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx), 0],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy), 0],
            [0, 0, 0, 1],
        ], np.float32)
        m = m @ r
    if "translation" in node:
        t = np.eye(4, dtype=np.float32)
        t[:3, 3] = np.asarray(node["translation"], np.float32)
        m = t @ m
    return m


def _iter_gltf_node_meshes(gltf: dict):
    nodes = gltf.get("nodes", [])
    if not nodes:
        return
    scene_idx = int(gltf.get("scene", 0))
    scenes = gltf.get("scenes") or [{"nodes": list(range(len(nodes)))}]
    roots = scenes[scene_idx].get("nodes", list(range(len(nodes))))

    def walk(node_idx: int, parent: np.ndarray):
        node = nodes[int(node_idx)]
        mat = parent @ _node_matrix(node)
        if "mesh" in node:
            yield int(node["mesh"]), mat, str(node.get("name", ""))
        for child in node.get("children", []):
            yield from walk(int(child), mat)

    for root in roots:
        yield from walk(int(root), np.eye(4, dtype=np.float32))


@lru_cache(maxsize=8)
def _load_external_gltf_mesh(path_str: str) -> _SolidAccessory | None:
    src = _read_gltf_source(path_str)
    if src is None:
        return None
    gltf, gltf_path, zf, source_name = src
    try:
        buffers = _load_gltf_buffers(gltf, gltf_path, zf)
        verts_all: list[np.ndarray] = []
        tris_all: list[np.ndarray] = []
        materials = gltf.get("materials", [])
        for mesh_idx, mat4, node_name in _iter_gltf_node_meshes(gltf):
            mesh = gltf["meshes"][mesh_idx]
            for prim in mesh.get("primitives", []):
                if int(prim.get("mode", 4)) != 4:
                    continue
                mat_idx = prim.get("material")
                if mat_idx is not None and int(mat_idx) < len(materials):
                    mat = materials[int(mat_idx)]
                    pbr = mat.get("pbrMetallicRoughness", {})
                    alpha = float((pbr.get("baseColorFactor") or [1, 1, 1, 1])[3])
                    mat_name = str(mat.get("name", "")).lower()
                    if alpha < 0.05 or "lens" in mat_name or mat_name == "glass":
                        continue
                attrs = prim.get("attributes", {})
                if "POSITION" not in attrs:
                    continue
                verts = _read_gltf_accessor(gltf, buffers, int(attrs["POSITION"]))
                if verts.ndim != 2 or verts.shape[1] != 3:
                    continue
                vh = np.concatenate(
                    [verts.astype(np.float32), np.ones((len(verts), 1), np.float32)],
                    axis=1,
                )
                verts = (vh @ mat4.T)[:, :3].astype(np.float32)
                if "indices" in prim:
                    idx = _read_gltf_accessor(gltf, buffers, int(prim["indices"]))
                    idx = idx.astype(np.int32).reshape(-1, 3)
                else:
                    idx = np.arange(len(verts), dtype=np.int32).reshape(-1, 3)
                base = sum(len(v) for v in verts_all)
                verts_all.append(verts)
                tris_all.append(idx + base)
        if not verts_all or not tris_all:
            return None
        return _SolidAccessory(
            source=f"external_gltf:{source_name}",
            verts=np.concatenate(verts_all, axis=0).astype(np.float32),
            tris=np.concatenate(tris_all, axis=0).astype(np.int32),
        )
    finally:
        if zf is not None:
            zf.close()


def _fit_external_goggle_mesh(
    solid: _SolidAccessory,
    basis: Basis,
    shape_coeffs: np.ndarray,
    scale_x: float | None,
    scale_y: float | None,
    offset_y: float,
) -> _SolidAccessory | None:
    lm = _landmark_points_3d(
        basis, shape_coeffs, (33, 133, 263, 362, 168, 6, 1)
    )
    if any(k not in lm for k in (33, 263)):
        return None
    left_outer = lm[33]
    right_outer = lm[263]
    if left_outer[0] > right_outer[0]:
        left_outer, right_outer = right_outer, left_outer
    eye_span = max(float(right_outer[0] - left_outer[0]), 1.0)
    target_center_x = float(0.5 * (left_outer[0] + right_outer[0]))
    target_center_y = float(np.mean([left_outer[1], right_outer[1]])
                            - 2.0 + float(offset_y))
    eye_z = float(np.mean([left_outer[2], right_outer[2]]))
    bridge_z = float(lm.get(168, lm.get(6, np.array([0, 0, eye_z])))[2])
    target_front_z = max(eye_z + 14.0, bridge_z + 11.0)

    raw = _auto_align_external_yaw(solid.verts.astype(np.float32))
    lo = raw.min(axis=0)
    hi = raw.max(axis=0)
    size = np.maximum(hi - lo, 1e-6)
    center = 0.5 * (lo + hi)
    sx = float(1.0 if scale_x is None else scale_x)
    sy = float(1.0 if scale_y is None else scale_y)
    scale = (eye_span * 1.18 * sx) / float(size[0])
    y_scale = (eye_span * 0.29 * sy) / float(size[1])
    depth_scale = scale * 0.42
    front_z = float(np.percentile(raw[:, 2], 94))

    out = raw.copy()
    out[:, 0] = target_center_x + (raw[:, 0] - center[0]) * scale
    out[:, 1] = target_center_y + (raw[:, 1] - center[1]) * y_scale
    out[:, 2] = target_front_z + (raw[:, 2] - front_z) * depth_scale
    return _SolidAccessory(solid.source, out.astype(np.float32), solid.tris)


def _merge_solid_accessories(
    source: str,
    solids: list[_SolidAccessory | None],
) -> _SolidAccessory | None:
    verts_all: list[np.ndarray] = []
    tris_all: list[np.ndarray] = []
    base = 0
    for solid in solids:
        if solid is None or len(solid.verts) == 0 or len(solid.tris) == 0:
            continue
        verts = solid.verts.astype(np.float32)
        tris = solid.tris.astype(np.int32)
        verts_all.append(verts)
        tris_all.append(tris + base)
        base += len(verts)
    if not verts_all or not tris_all:
        return None
    return _SolidAccessory(
        source,
        np.concatenate(verts_all, axis=0).astype(np.float32),
        np.concatenate(tris_all, axis=0).astype(np.int32),
    )


def _build_visible_temple_arms(
    basis: Basis,
    shape_coeffs: np.ndarray,
    scale_x: float | None,
    scale_y: float | None,
    offset_y: float,
) -> _SolidAccessory | None:
    lm = _landmark_points_3d(
        basis,
        shape_coeffs,
        (33, 133, 263, 362, 127, 162, 234, 356, 389, 454, 168, 6),
    )
    if any(k not in lm for k in (33, 263)):
        return None

    left_outer = lm[33]
    right_outer = lm[263]
    left_inner = lm.get(133, left_outer)
    right_inner = lm.get(362, right_outer)
    if left_outer[0] > right_outer[0]:
        left_outer, right_outer = right_outer, left_outer
        left_inner, right_inner = right_inner, left_inner

    sx = 1.0 if scale_x is None else float(scale_x)
    sy = 1.0 if scale_y is None else float(scale_y)
    eye_span = max(float(right_outer[0] - left_outer[0]), 1.0)
    eye_y = float(np.mean([
        left_outer[1], left_inner[1], right_inner[1], right_outer[1],
    ]) - 2.0 + float(offset_y))
    eye_z = float(np.mean([left_outer[2], right_outer[2]]))
    bridge_z = float(lm.get(168, lm.get(6, np.array([0, 0, eye_z])))[2])
    base_z = max(eye_z + 9.0, bridge_z - 3.0)

    left_side = [
        p for k, p in lm.items()
        if k in (127, 162, 234) and float(p[0]) < float(left_outer[0])
    ]
    right_side = [
        p for k, p in lm.items()
        if k in (356, 389, 454) and float(p[0]) > float(right_outer[0])
    ]
    left_edge_x = (
        min(float(p[0]) for p in left_side)
        if left_side else float(left_outer[0] - 0.24 * eye_span * sx)
    )
    right_edge_x = (
        max(float(p[0]) for p in right_side)
        if right_side else float(right_outer[0] + 0.24 * eye_span * sx)
    )

    hinge_y = eye_y + 0.047 * eye_span * sy
    end_y = eye_y + 0.005 * eye_span * sy
    inset = 0.025 * eye_span * sx
    reach = 0.86
    left_start_x = float(left_outer[0] - 0.075 * eye_span * sx)
    right_start_x = float(right_outer[0] + 0.075 * eye_span * sx)
    left_end_x = left_start_x + reach * (left_edge_x + inset - left_start_x)
    right_end_x = right_start_x + reach * (right_edge_x - inset - right_start_x)

    left_path = np.asarray([
        [left_start_x, hinge_y, base_z + 2.9],
        [0.56 * left_start_x + 0.44 * left_end_x,
         0.55 * hinge_y + 0.45 * end_y, base_z + 2.6],
        [left_end_x, end_y, base_z + 2.2],
    ], np.float32)
    right_path = np.asarray([
        [right_start_x, hinge_y, base_z + 2.9],
        [0.56 * right_start_x + 0.44 * right_end_x,
         0.55 * hinge_y + 0.45 * end_y, base_z + 2.6],
        [right_end_x, end_y, base_z + 2.2],
    ], np.float32)
    radius = max(1.10, 0.0140 * eye_span)
    verts, tris = _tube_mesh([(left_path, radius, False),
                              (right_path, radius, False)], sides=10)
    if len(verts) == 0:
        return None
    return _SolidAccessory("visible_temple_arms", verts, tris)


def _load_custom_gltf_accessory(
    asset_dir: str | Path | None,
    basis: Basis,
    shape_coeffs: np.ndarray,
    scale_x: float | None,
    scale_y: float | None,
    offset_y: float,
) -> _SolidAccessory | None:
    if asset_dir is None:
        return None
    resolved = resolve_glasses_mesh_asset(asset_dir)
    if resolved is None:
        return None
    solid = _load_external_gltf_mesh(str(resolved))
    if solid is None:
        return None
    fitted = _fit_external_goggle_mesh(
        solid, basis, shape_coeffs, scale_x, scale_y, offset_y
    )
    if fitted is None:
        return None
    arms = _build_visible_temple_arms(
        basis, shape_coeffs, scale_x, scale_y, offset_y
    )
    merged = _merge_solid_accessories(f"{fitted.source}+temple_arms",
                                      [fitted, arms])
    return merged or fitted


def _has_custom_mesh_asset(asset_dir: str | Path | None) -> bool:
    if asset_dir is None:
        return False
    p = resolve_glasses_mesh_asset(asset_dir)
    if p is None:
        return False
    if p.suffix.lower() in {".zip", ".gltf", ".fbx", ".obj", ".dae"}:
        return True
    return (
        _find_gltf_path(p) is not None
        or _find_convertible_model_path(p) is not None
    )


def _tube_mesh(
    paths: list[tuple[np.ndarray, float, bool]],
    sides: int = 10,
    depth_scale: float = 0.58,
) -> tuple[np.ndarray, np.ndarray]:
    verts: list[np.ndarray] = []
    tris: list[tuple[int, int, int]] = []
    depth_axis = np.array([0.0, 0.0, 1.0], np.float32)

    for path, radius, closed in paths:
        path = np.asarray(path, np.float32)
        if len(path) < 2 or radius <= 0:
            continue
        start = len(verts)
        n = len(path)
        closed = bool(closed)
        for i, p in enumerate(path):
            if closed:
                prev_p = path[(i - 1) % n]
                next_p = path[(i + 1) % n]
            else:
                prev_p = path[max(0, i - 1)]
                next_p = path[min(n - 1, i + 1)]
            tangent = (next_p - prev_p).astype(np.float32)
            tangent[2] = 0.0
            tlen = float(np.linalg.norm(tangent))
            if tlen < 1e-6:
                tangent = np.array([1.0, 0.0, 0.0], np.float32)
            else:
                tangent /= tlen
            normal = np.array([-tangent[1], tangent[0], 0.0], np.float32)
            for si in range(sides):
                theta = 2.0 * np.pi * si / sides
                verts.append(
                    p
                    + radius * np.cos(theta) * normal
                    + radius * depth_scale * np.sin(theta) * depth_axis
                )

        segments = n if closed else n - 1
        for i in range(segments):
            j = (i + 1) % n
            for si in range(sides):
                sj = (si + 1) % sides
                a = start + i * sides + si
                b = start + j * sides + si
                c = start + j * sides + sj
                d = start + i * sides + sj
                tris.append((a, b, c))
                tris.append((a, c, d))

        if not closed:
            for end_i, flip in ((0, True), (n - 1, False)):
                center_idx = len(verts)
                verts.append(path[end_i].copy())
                ring = start + end_i * sides
                for si in range(sides):
                    sj = (si + 1) % sides
                    if flip:
                        tris.append((center_idx, ring + sj, ring + si))
                    else:
                        tris.append((center_idx, ring + si, ring + sj))

    if not verts or not tris:
        return (
            np.zeros((0, 3), np.float32),
            np.zeros((0, 3), np.int32),
        )
    return np.stack(verts).astype(np.float32), np.asarray(tris, np.int32)


def _superellipse_points(
    center: np.ndarray,
    rx: float,
    ry: float,
    z: float,
    n: int = 80,
    px: float = 0.42,
    py: float = 0.58,
) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    c = np.cos(theta)
    s = np.sin(theta)
    x = rx * np.sign(c) * (np.abs(c) ** px)
    y = ry * np.sign(s) * (np.abs(s) ** py)
    pts = np.zeros((n, 3), np.float32)
    pts[:, 0] = center[0] + x
    pts[:, 1] = center[1] + y
    # A shallow front bow gives the baked rim a little dimensional variation
    # while staying close enough to the face for stable UV projection.
    pts[:, 2] = z + 1.15 * (1.0 - np.clip((x / max(rx, 1e-6)) ** 2, 0, 1))
    return pts


def _build_stock_glasses_mesh(
    basis: Basis,
    shape_coeffs: np.ndarray,
    style: str = "rectangular",
    scale_x: float | None = None,
    scale_y: float | None = None,
    offset_y: float = 0.0,
    rim_width: float = 1.0,
) -> _SolidAccessory | None:
    """Original procedural eyeglasses fitted to the player's 3D landmarks.

    License-clean replacement for the FaceGen Modeller accessory: lens rims
    are tube sweeps along style superellipses sized from the interpupillary
    distance, placed at the fitted mesh's eye centers, with a bridge and
    temple arms. Real frames are larger than the eye opening, so sizing comes
    from IPD, not from the eye-corner box.
    """
    lm = _landmark_points_3d(
        basis, shape_coeffs, (33, 133, 263, 362, 168, 6, 1, 468, 473)
    )
    needed = (33, 133, 263, 362)
    if any(k not in lm for k in needed):
        return None
    left_outer, left_inner = lm[33], lm[133]
    right_inner, right_outer = lm[362], lm[263]
    if left_outer[0] > right_outer[0]:
        left_outer, right_outer = right_outer, left_outer
        left_inner, right_inner = right_inner, left_inner
    left_c3 = lm.get(468, 0.5 * (left_outer + left_inner))
    right_c3 = lm.get(473, 0.5 * (right_inner + right_outer))
    if left_c3[0] > right_c3[0]:
        left_c3, right_c3 = right_c3, left_c3
    ipd = max(float(right_c3[0] - left_c3[0]), 1.0)

    sx = 1.0 if scale_x is None else float(scale_x)
    sy = 1.0 if scale_y is None else float(scale_y)
    # per-style: lens aspect (ry/rx), rim radius vs ipd, superellipse px/py
    spec = {
        "rectangular": (0.58, 0.050, 0.42, 0.58),
        "oval": (0.66, 0.045, 0.72, 0.80),
        "round": (0.88, 0.038, 0.86, 0.86),
        "sports_goggle": (0.62, 0.075, 0.50, 0.66),
    }
    aspect, rim_f, px, py = spec.get(style, spec["rectangular"])
    rx = 0.40 * ipd * sx
    ry = max(rx * aspect * sy, 3.0)
    eye_y = 0.5 * float(left_c3[1] + right_c3[1]) + float(offset_y)
    cy = eye_y + 0.10 * ry
    eye_z = 0.5 * float(left_c3[2] + right_c3[2])
    bridge_z = float(lm.get(168, lm.get(6, np.array([0, 0, eye_z])))[2])
    base_z = max(eye_z + 7.0, bridge_z - 4.0)

    left_c = np.array([float(left_c3[0]), cy, base_z], np.float32)
    right_c = np.array([float(right_c3[0]), cy, base_z], np.float32)
    left_lens = _superellipse_points(left_c, rx, ry, base_z, px=px, py=py)
    right_lens = _superellipse_points(right_c, rx, ry, base_z, px=px, py=py)

    rim = max(0.8, rim_f * ipd * float(rim_width))

    # shallow bridge arc between the inner rims, above the eye line
    bx0 = left_c[0] + 0.92 * rx
    bx1 = right_c[0] - 0.92 * rx
    xs = np.linspace(bx0, bx1, 8, dtype=np.float32)
    tt = np.linspace(0.0, np.pi, len(xs), dtype=np.float32)
    bridge = np.stack([
        xs,
        np.full_like(xs, cy + 0.45 * ry) + 0.18 * ry * np.sin(tt),
        np.full_like(xs, base_z + 1.5),
    ], axis=1)

    # temple arms: from the outer hinges sideways and slightly back
    hinge_y = cy + 0.35 * ry
    arm_len = 0.55 * ipd
    def _arm(x0: float, direction: float) -> np.ndarray:
        return np.stack([
            np.array([x0, hinge_y, base_z], np.float32),
            np.array([x0 + direction * 0.55 * arm_len, hinge_y + 0.06 * ry,
                      base_z - 6.0], np.float32),
            np.array([x0 + direction * arm_len, hinge_y + 0.16 * ry,
                      base_z - 14.0], np.float32),
        ])
    left_arm = _arm(left_c[0] - 0.98 * rx, -1.0)
    right_arm = _arm(right_c[0] + 0.98 * rx, 1.0)

    paths = [
        (left_lens, rim, True),
        (right_lens, rim, True),
        (bridge, rim * 0.9, False),
        (left_arm, rim * 0.8, False),
        (right_arm, rim * 0.8, False),
    ]
    if style == "sports_goggle":
        xs2 = np.linspace(left_c[0] - 0.90 * rx, right_c[0] + 0.90 * rx, 24,
                          dtype=np.float32)
        tt2 = np.linspace(np.pi, 0.0, len(xs2), dtype=np.float32)
        top_bar = np.stack([
            xs2,
            np.full_like(xs2, cy + 0.92 * ry) + 0.10 * ry * np.cos(tt2),
            np.full_like(xs2, base_z + 2.2),
        ], axis=1)
        paths.append((top_bar, rim * 1.15, False))

    verts, tris = _tube_mesh(paths, sides=10, depth_scale=0.62)
    if len(verts) == 0:
        return None
    return _SolidAccessory(f"procedural_{style}", verts, tris)


def _build_sports_goggle_mesh(
    basis: Basis,
    shape_coeffs: np.ndarray,
    scale_x: float | None,
    scale_y: float | None,
    offset_y: float,
    rim_width: float = 1.0,
) -> _SolidAccessory | None:
    lm = _landmark_points_3d(
        basis, shape_coeffs, (33, 133, 263, 362, 168, 6, 1, 468, 473)
    )
    needed = (33, 133, 263, 362)
    if any(k not in lm for k in needed):
        return None

    left_outer = lm[33]
    left_inner = lm[133]
    right_inner = lm[362]
    right_outer = lm[263]
    if left_outer[0] > right_outer[0]:
        left_outer, right_outer = right_outer, left_outer
        left_inner, right_inner = right_inner, left_inner

    sx = 1.0 if scale_x is None else float(scale_x)
    sy = 1.0 if scale_y is None else float(scale_y)
    y_offset = float(offset_y)
    eye_span = max(float(right_outer[0] - left_outer[0]), 1.0)
    left_w = max(float(left_inner[0] - left_outer[0]), eye_span * 0.18)
    right_w = max(float(right_outer[0] - right_inner[0]), eye_span * 0.18)
    eye_y = float(np.mean([left_outer[1], left_inner[1],
                           right_inner[1], right_outer[1]]) - 2.0 + y_offset)
    eye_z = float(np.mean([left_outer[2], left_inner[2],
                           right_inner[2], right_outer[2]]))
    bridge_z = float(lm.get(168, lm.get(6, np.array([0, 0, eye_z])))[2])
    base_z = max(eye_z + 7.0, bridge_z - 5.0)

    left_c = np.array([
        0.5 * (left_outer[0] + left_inner[0]) - 0.025 * eye_span * (sx - 1.0),
        eye_y - 0.010 * eye_span,
        base_z,
    ], np.float32)
    right_c = np.array([
        0.5 * (right_inner[0] + right_outer[0]) + 0.025 * eye_span * (sx - 1.0),
        eye_y - 0.010 * eye_span,
        base_z,
    ], np.float32)
    rx_l = max(left_w * 0.68 * sx, eye_span * 0.135)
    rx_r = max(right_w * 0.68 * sx, eye_span * 0.135)
    ry = max(eye_span * 0.074 * sy, 4.2)

    left_lens = _superellipse_points(left_c, rx_l, ry, base_z)
    right_lens = _superellipse_points(right_c, rx_r, ry, base_z)

    top_y = eye_y + 0.86 * ry
    top_x0 = left_c[0] - 0.90 * rx_l
    top_x1 = right_c[0] + 0.90 * rx_r
    xs = np.linspace(top_x0, top_x1, 24, dtype=np.float32)
    top_bar = np.stack([
        xs,
        np.full_like(xs, top_y) + 0.10 * ry * np.cos(
            np.linspace(np.pi, 0.0, len(xs), dtype=np.float32)
        ),
        np.full_like(xs, base_z + 2.6),
    ], axis=1)

    inner_l = np.array([left_c[0] + 0.72 * rx_l, eye_y + 0.35 * ry,
                        base_z + 8.0], np.float32)
    inner_r = np.array([right_c[0] - 0.72 * rx_r, eye_y + 0.35 * ry,
                        base_z + 8.0], np.float32)
    nose_front_z = max(
        base_z + 30.0,
        bridge_z + 24.0,
        float(lm.get(1, lm.get(6, np.array([0, 0, bridge_z])))[2]) - 7.0,
    )
    nose = np.array([
        0.5 * (inner_l[0] + inner_r[0]),
        eye_y - 2.08 * ry,
        nose_front_z,
    ], np.float32)
    bridge_v = np.stack([inner_l, nose, inner_r]).astype(np.float32)
    bridge_stem = np.stack([
        np.array([nose[0], eye_y + 0.50 * ry, base_z + 10.0], np.float32),
        nose,
    ]).astype(np.float32)

    left_arm = np.stack([
        np.array([top_x0, top_y, base_z + 2.0], np.float32),
        np.array([top_x0 - 0.15 * eye_span, eye_y - 0.33 * ry, base_z - 1.0],
                 np.float32),
    ])
    right_arm = np.stack([
        np.array([top_x1, top_y, base_z + 2.0], np.float32),
        np.array([top_x1 + 0.15 * eye_span, eye_y - 0.33 * ry, base_z - 1.0],
                 np.float32),
    ])

    rim = max(0.95, 1.45 * float(rim_width))
    heavy = max(1.15, 2.15 * float(rim_width))
    paths = [
        (left_lens, rim, True),
        (right_lens, rim, True),
        (top_bar, heavy, False),
        (bridge_v, heavy * 1.42, False),
        (bridge_stem, heavy * 0.72, False),
        (left_arm, rim * 0.86, False),
        (right_arm, rim * 0.86, False),
    ]
    verts, tris = _tube_mesh(paths, sides=10, depth_scale=0.62)
    if len(verts) == 0:
        return None
    return _SolidAccessory("procedural_sports_goggle", verts, tris)


def _bake_solid_accessory(
    D: np.ndarray,
    face: _FaceMaps,
    solid: _SolidAccessory,
    color_name: str,
    opacity: float,
) -> BakeResult:
    if len(solid.verts) == 0 or len(solid.tris) == 0:
        return BakeResult(False, source=solid.source, color_name=color_name,
                          reason="procedural mesh is empty")

    size = int(D.shape[0])
    canvas = int(face.rgb.shape[0])
    scr = render._project_with_params(solid.verts, canvas, "ootp",
                                      face.lo, face.scale)
    base_detail = D.astype(np.float32).copy()
    target_accum = np.zeros_like(base_detail)
    weight_accum = np.zeros(D.shape[:2], np.float32)
    frame_rgb = _frame_rgb(color_name)
    if color_name == "red":
        frame_rgb = np.array([170.0, 24.0, 21.0], np.float32)
    opacity = float(np.clip(opacity, 0.0, 1.0))

    tri_v = solid.verts[solid.tris]
    tri_s = scr[solid.tris]
    e1 = tri_v[:, 1] - tri_v[:, 0]
    e2 = tri_v[:, 2] - tri_v[:, 0]
    normals = np.cross(e1, e2)
    nlen = np.maximum(np.linalg.norm(normals, axis=1), 1e-9)
    nz = normals[:, 2] / nlen
    sign = 1.0 if (nz > 0).sum() >= (nz < 0).sum() else -1.0
    facing = nz * sign
    order = np.argsort(tri_v[:, :, 2].mean(1))

    # Real 3D shading is what separates a baked mesh from a drawn 2D line:
    # diffuse N.L rounds the rim tubes (bright top, dark underside) and a
    # tight specular adds the plastic/metal glint along the upper edge.
    n3 = (normals / nlen[:, None]) * sign
    light = np.array([0.25, 0.60, 0.76], np.float32)
    light /= np.linalg.norm(light)
    half = light + np.array([0.0, 0.0, 1.0], np.float32)
    half /= np.linalg.norm(half)
    diff = np.clip(n3 @ light, 0.0, 1.0)
    spec = np.clip(n3 @ half, 0.0, 1.0) ** 26

    acc_mask = np.zeros((canvas, canvas), np.float32)
    screen_pixels = 0
    for ti in order:
        if facing[ti] < -0.95:
            continue
        dst = tri_s[ti]
        x0, y0 = np.maximum(np.floor(dst.min(0)).astype(int), 0)
        x1 = min(int(np.ceil(dst[:, 0].max())) + 1, canvas)
        y1 = min(int(np.ceil(dst[:, 1].max())) + 1, canvas)
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        dst_local = (dst - [x0, y0]).astype(np.float32)
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillConvexPoly(mask, np.round(dst_local).astype(np.int32), 1)
        mb = mask.astype(bool)
        if not mb.any():
            continue
        zz = render._tri_depth(dst_local, tri_v[ti][:, 2], x1 - x0, y1 - y0)
        fz = face.zbuf[y0:y1, x0:x1]
        fvalid = face.valid[y0:y1, x0:x1]
        visible = mb & fvalid & (zz > fz - 1.8)
        if not visible.any():
            continue

        lit = 0.38 + 0.62 * float(diff[ti])
        target_rgb = np.clip(
            frame_rgb.reshape(1, 3) * lit + 230.0 * float(spec[ti]), 0, 255)
        face_rgb = np.maximum(face.rgb[y0:y1, x0:x1], 1.0)
        dxy = face.detail_xy[y0:y1, x0:x1]

        yy, xx = np.nonzero(visible)
        acc_mask[y0:y1, x0:x1][visible] = 1.0
        screen_pixels += int(len(xx))
        ix = np.clip(np.round(dxy[yy, xx, 0]).astype(int), 0, size - 1)
        iy = np.clip(np.round(dxy[yy, xx, 1]).astype(int), 0, size - 1)
        curD = np.maximum(base_detail[iy, ix], 1.0)
        alpha = np.full((len(xx), 1), opacity, np.float32)
        blended_rgb = (
            alpha * target_rgb
            + (1.0 - alpha) * face_rgb[yy, xx]
        )
        targetD = np.clip(curD * blended_rgb / face_rgb[yy, xx], 0, 255)
        weights = np.full(len(xx), opacity, np.float32)
        for ch in range(3):
            np.add.at(target_accum[..., ch], (iy, ix), targetD[:, ch] * weights)
        np.add.at(weight_accum, (iy, ix), weights)

    if screen_pixels == 0 or not weight_accum.any():
        return BakeResult(False, source=solid.source, color_name=color_name,
                          reason="procedural mesh did not project onto face")

    # Contact shadow: the frame floats above the skin, so it darkens the face
    # a little below it along the light direction — the depth cue a painted
    # frame lacks.
    shift = max(2, int(round(canvas * 0.008)))
    M = np.float32([[1, 0, 0.6 * shift], [0, 1, float(shift)]])
    sh = cv2.warpAffine(acc_mask, M, (canvas, canvas))
    sh = cv2.GaussianBlur(sh, (0, 0), max(1.5, canvas / 300.0))
    sh[acc_mask > 0.5] = 0.0
    sy, sx = np.nonzero((sh > 0.05) & face.valid)
    if len(sy):
        dxy_s = face.detail_xy[sy, sx]
        ix = np.clip(np.round(dxy_s[:, 0]).astype(int), 0, size - 1)
        iy = np.clip(np.round(dxy_s[:, 1]).astype(int), 0, size - 1)
        shv = sh[sy, sx].astype(np.float32)
        curD = np.maximum(base_detail[iy, ix], 1.0)
        targetD = np.clip(curD * (1.0 - 0.22 * shv[:, None]), 0, 255)
        w = 0.5 * shv
        for ch in range(3):
            np.add.at(target_accum[..., ch], (iy, ix), targetD[:, ch] * w)
        np.add.at(weight_accum, (iy, ix), w)

    target = base_detail.copy()
    hit = weight_accum > 1e-6
    target[hit] = target_accum[hit] / weight_accum[hit, None]
    alpha = np.clip(weight_accum, 0.0, 1.0)
    sigma = max(0.55, D.shape[0] / 1850.0)
    alpha_s = cv2.GaussianBlur(alpha, (0, 0), sigma)
    numer = cv2.GaussianBlur(target * alpha[..., None], (0, 0), sigma)
    denom = np.maximum(
        cv2.GaussianBlur(alpha, (0, 0), sigma)[..., None],
        1e-6,
    )
    target_s = numer / denom
    a = np.clip(alpha_s, 0.0, 1.0)[..., None]
    D[:] = np.clip(a * target_s + (1.0 - a) * base_detail,
                   0, 255).astype(np.uint8)
    return BakeResult(
        True,
        source=solid.source,
        color_name=color_name,
        screen_pixels=screen_pixels,
        detail_pixels=int((weight_accum > 0.03).sum()),
    )


def _solid_visible_screen_mask(
    face: _FaceMaps,
    solid: _SolidAccessory,
) -> tuple[np.ndarray, int]:
    canvas = int(face.rgb.shape[0])
    mask_full = np.zeros((canvas, canvas), np.uint8)
    if len(solid.verts) == 0 or len(solid.tris) == 0:
        return mask_full, 0

    scr = render._project_with_params(solid.verts, canvas, "ootp",
                                      face.lo, face.scale)
    tri_v = solid.verts[solid.tris]
    tri_s = scr[solid.tris]
    e1 = tri_v[:, 1] - tri_v[:, 0]
    e2 = tri_v[:, 2] - tri_v[:, 0]
    normals = np.cross(e1, e2)
    nlen = np.maximum(np.linalg.norm(normals, axis=1), 1e-9)
    nz = normals[:, 2] / nlen
    sign = 1.0 if (nz > 0).sum() >= (nz < 0).sum() else -1.0
    facing = nz * sign
    order = np.argsort(tri_v[:, :, 2].mean(1))

    screen_pixels = 0
    for ti in order:
        if facing[ti] < -0.95:
            continue
        dst = tri_s[ti]
        x0, y0 = np.maximum(np.floor(dst.min(0)).astype(int), 0)
        x1 = min(int(np.ceil(dst[:, 0].max())) + 1, canvas)
        y1 = min(int(np.ceil(dst[:, 1].max())) + 1, canvas)
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        dst_local = (dst - [x0, y0]).astype(np.float32)
        tri_mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillConvexPoly(tri_mask, np.round(dst_local).astype(np.int32), 1)
        mb = tri_mask.astype(bool)
        if not mb.any():
            continue
        zz = render._tri_depth(dst_local, tri_v[ti][:, 2], x1 - x0, y1 - y0)
        fz = face.zbuf[y0:y1, x0:x1]
        fvalid = face.valid[y0:y1, x0:x1]
        visible = mb & fvalid & (zz > fz - 1.8)
        if not visible.any():
            continue
        block = mask_full[y0:y1, x0:x1]
        block[visible] = 255
        screen_pixels += int(visible.sum())
    return mask_full, screen_pixels


def _bake_solid_accessory_outline(
    D: np.ndarray,
    face: _FaceMaps,
    solid: _SolidAccessory,
    color_name: str,
    opacity: float,
) -> BakeResult:
    mask, screen_pixels = _solid_visible_screen_mask(face, solid)
    if screen_pixels == 0 or not mask.any():
        return BakeResult(False, source=solid.source, color_name=color_name,
                          reason="custom mesh outline did not project onto face")

    canvas = int(face.rgb.shape[0])
    thick = max(4, int(round(canvas / 145.0)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * thick + 1, 2 * thick + 1),
    )
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    eroded = cv2.erode(closed, kernel, iterations=1)
    outline = cv2.subtract(closed, eroded)
    outline = cv2.GaussianBlur(outline, (0, 0), max(0.55, canvas / 2600.0))
    if not outline.any():
        return BakeResult(False, source=solid.source, color_name=color_name,
                          reason="custom mesh outline is empty")

    frame_rgb = _frame_rgb(color_name)
    if color_name == "red":
        frame_rgb = np.array([170.0, 24.0, 21.0], np.float32)
    s, d = _bake_screen_mask(
        D,
        face,
        outline.astype(np.float32) / 255.0,
        frame_rgb,
        float(np.clip(opacity, 0.0, 1.0)),
        sigma=max(0.70, D.shape[0] / 1500.0),
    )
    if d == 0:
        return BakeResult(False, source=solid.source, color_name=color_name,
                          reason="custom mesh outline did not bake to detail")
    return BakeResult(
        True,
        source=f"{solid.source}+outline",
        color_name=color_name,
        screen_pixels=s,
        detail_pixels=d,
    )


def bake_procedural_sports_goggle(
    D: np.ndarray,
    basis: Basis,
    shape_coeffs: np.ndarray,
    tex_coeffs: np.ndarray,
    color_name: str = "red",
    asset_dir: str | Path | None = None,
    opacity: float = 0.92,
    scale_x: float | None = None,
    scale_y: float | None = None,
    offset_y: float = 0.0,
    rim_width: float = 1.0,
    screen_scale: int = 2,
) -> BakeResult:
    if D.ndim != 3 or D.shape[2] != 3:
        return BakeResult(False, reason="detail map must be RGB")
    size = int(D.shape[0])
    canvas = max(size, int(size) * max(1, int(screen_scale)))
    face = _render_face_maps(basis, shape_coeffs, tex_coeffs, D, canvas)
    solid = _load_custom_gltf_accessory(
        asset_dir,
        basis,
        shape_coeffs,
        scale_x=scale_x,
        scale_y=scale_y,
        offset_y=offset_y,
    )
    if solid is None:
        solid = _build_sports_goggle_mesh(
            basis,
            shape_coeffs,
            scale_x=scale_x,
            scale_y=scale_y,
            offset_y=offset_y,
            rim_width=rim_width,
        )
    if solid is None:
        return BakeResult(False, color_name=color_name,
                          reason="could not build procedural sports goggle")
    return _bake_solid_accessory(D, face, solid, color_name, opacity)


def bake_custom_gltf_accessory(
    D: np.ndarray,
    basis: Basis,
    shape_coeffs: np.ndarray,
    tex_coeffs: np.ndarray,
    asset_dir: str | Path | None,
    color_name: str = "red",
    opacity: float = 0.90,
    scale_x: float | None = None,
    scale_y: float | None = None,
    offset_y: float = 0.0,
    screen_scale: int = 2,
    outline_only: bool = False,
) -> BakeResult:
    if D.ndim != 3 or D.shape[2] != 3:
        return BakeResult(False, reason="detail map must be RGB")
    size = int(D.shape[0])
    canvas = max(size, int(size) * max(1, int(screen_scale)))
    face = _render_face_maps(basis, shape_coeffs, tex_coeffs, D, canvas)
    solid = _load_custom_gltf_accessory(
        asset_dir,
        basis,
        shape_coeffs,
        scale_x=scale_x,
        scale_y=scale_y,
        offset_y=offset_y,
    )
    if solid is None:
        return BakeResult(False, color_name=color_name,
                          reason="custom glTF accessory not found")
    if outline_only:
        return _bake_solid_accessory_outline(
            D, face, solid, color_name, opacity
        )
    return _bake_solid_accessory(D, face, solid, color_name, opacity)

def _bake_screen_mask(
    D: np.ndarray,
    face: _FaceMaps,
    mask: np.ndarray,
    target_rgb: np.ndarray,
    opacity: float,
    sigma: float | None = None,
) -> tuple[int, int]:
    size = int(D.shape[0])
    base_detail = D.astype(np.float32).copy()
    alpha_src = np.clip(mask.astype(np.float32), 0.0, 1.0)
    visible = (alpha_src > 0.003) & face.valid
    if not visible.any():
        return 0, 0

    yy, xx = np.nonzero(visible)
    dxy = face.detail_xy[yy, xx]
    ok = (
        (dxy[:, 0] >= 0.0)
        & (dxy[:, 0] < size)
        & (dxy[:, 1] >= 0.0)
        & (dxy[:, 1] < size)
    )
    if not ok.any():
        return 0, 0

    yy = yy[ok]
    xx = xx[ok]
    dxy = dxy[ok]
    ix = np.clip(np.round(dxy[:, 0]).astype(int), 0, size - 1)
    iy = np.clip(np.round(dxy[:, 1]).astype(int), 0, size - 1)

    face_rgb = np.maximum(face.rgb[yy, xx], 1.0)
    curD = np.maximum(base_detail[iy, ix], 1.0)
    alpha = (alpha_src[yy, xx] * float(np.clip(opacity, 0.0, 1.0))).astype(
        np.float32
    )[:, None]
    target_rgb = np.asarray(target_rgb, np.float32).reshape(1, 3)
    blended_rgb = alpha * target_rgb + (1.0 - alpha) * face_rgb
    targetD = np.clip(curD * blended_rgb / face_rgb, 0, 255)
    weights = np.clip(alpha[:, 0], 0.0, 1.0)

    target_accum = np.zeros_like(base_detail)
    weight_accum = np.zeros(D.shape[:2], np.float32)
    for ch in range(3):
        np.add.at(target_accum[..., ch], (iy, ix), targetD[:, ch] * weights)
    np.add.at(weight_accum, (iy, ix), weights)
    if not weight_accum.any():
        return int(len(xx)), 0

    target = base_detail.copy()
    hit = weight_accum > 1e-6
    target[hit] = target_accum[hit] / weight_accum[hit, None]
    sigma = float(max(0.35, D.shape[0] / 2600.0) if sigma is None else sigma)
    alpha_s = cv2.GaussianBlur(np.clip(weight_accum, 0.0, 1.0), (0, 0), sigma)
    numer = cv2.GaussianBlur(target * np.clip(weight_accum, 0.0, 1.0)[..., None],
                             (0, 0), sigma)
    denom = np.maximum(
        cv2.GaussianBlur(np.clip(weight_accum, 0.0, 1.0), (0, 0), sigma)[..., None],
        1e-6,
    )
    target_s = numer / denom
    a = np.clip(alpha_s, 0.0, 1.0)[..., None]
    D[:] = np.clip(a * target_s + (1.0 - a) * base_detail, 0, 255).astype(
        np.uint8
    )
    return int(len(xx)), int((weight_accum > 0.03).sum())


def _bake_sports_goggle_overlay(
    D: np.ndarray,
    basis: Basis,
    shape_coeffs: np.ndarray,
    face: _FaceMaps,
    frame_rgb: np.ndarray,
    opacity: float,
) -> tuple[int, int]:
    eyes = _eye_screen_points(basis, shape_coeffs, face.lo, face.scale,
                              face.rgb.shape[0])
    if eyes is None:
        return 0, 0
    eyes = eyes[np.argsort(eyes[:, 0])]
    left, right = eyes[0], eyes[1]
    eye_d = float(np.linalg.norm(right - left))
    if eye_d < 8.0:
        return 0, 0

    canvas = int(face.rgb.shape[0])
    center = 0.5 * (left + right)
    lens_y = center[1] - 0.006 * eye_d
    rx = 0.32 * eye_d
    ry = 0.135 * eye_d
    thick = max(4, int(round(0.026 * eye_d)))

    def pt(p: np.ndarray | tuple[float, float]) -> tuple[int, int]:
        q = np.asarray(p, np.float32)
        return int(round(float(q[0]))), int(round(float(q[1])))

    rim = np.zeros((canvas, canvas), np.uint8)
    fill = np.zeros_like(rim)
    shade = np.zeros_like(rim)
    highlight = np.zeros_like(rim)

    lc = np.array([left[0], lens_y], np.float32)
    rc = np.array([right[0], lens_y], np.float32)
    for c in (lc, rc):
        # Keep the stock accessory as the real rim. Sports overlay only
        # reinforces the brow edge, avoiding a second full painted oval.
        cv2.ellipse(
            shade,
            pt(c + np.array([1.0, 1.5], np.float32)),
            (int(round(rx)), int(round(ry))),
            0,
            205,
            335,
            255,
            thick + 2,
            lineType=cv2.LINE_AA,
        )
        cv2.ellipse(
            rim,
            pt(c),
            (int(round(rx)), int(round(ry))),
            0,
            205,
            335,
            255,
            thick,
            lineType=cv2.LINE_AA,
        )

    top_y = lens_y - 0.38 * ry
    left_outer = np.array([lc[0] - 0.44 * rx, top_y], np.float32)
    right_outer = np.array([rc[0] + 0.44 * rx, top_y], np.float32)
    cv2.line(shade, pt(left_outer + [1.5, 2.0]),
             pt(right_outer + [1.5, 2.0]), 255, thick + 4,
             lineType=cv2.LINE_AA)
    cv2.line(rim, pt(left_outer), pt(right_outer), 255, thick + 1,
             lineType=cv2.LINE_AA)

    inner_l = np.array([center[0] - 0.115 * eye_d, lens_y - 0.018 * eye_d],
                       np.float32)
    inner_r = np.array([center[0] + 0.115 * eye_d, lens_y - 0.018 * eye_d],
                       np.float32)
    nose = np.array([center[0], lens_y + 0.195 * eye_d], np.float32)
    bridge_poly = np.round(np.stack([inner_l, inner_r, nose])).astype(np.int32)
    cv2.fillConvexPoly(fill, bridge_poly, 255, lineType=cv2.LINE_AA)
    cv2.polylines(rim, [bridge_poly], True, 255, thick + 1,
                  lineType=cv2.LINE_AA)

    cv2.line(highlight, pt(left_outer + [0.06 * eye_d, -0.32 * thick]),
             pt(right_outer - [0.06 * eye_d, 0.32 * thick]), 255,
             max(2, thick // 3), lineType=cv2.LINE_AA)

    frame_rgb = np.asarray(frame_rgb, np.float32)
    if frame_rgb[0] > frame_rgb[1] * 1.45 and frame_rgb[0] > frame_rgb[2] * 1.35:
        main_rgb = np.array([136.0, 23.0, 22.0], np.float32)
    else:
        main_rgb = np.clip(frame_rgb * 1.12, 0, 190)
    shadow_rgb = np.clip(frame_rgb * np.array([0.46, 0.42, 0.40], np.float32),
                         0, 90)
    hi_rgb = np.clip(0.70 * main_rgb + np.array([55.0, 48.0, 42.0], np.float32),
                     0, 190)

    total_screen = 0
    total_detail = 0
    for mask, rgb, alpha in (
        (shade, shadow_rgb, 0.18 * opacity),
        (fill, main_rgb, 0.22 * opacity),
        (rim, main_rgb, 0.44 * opacity),
        (highlight, hi_rgb, 0.14 * opacity),
    ):
        s, d = _bake_screen_mask(D, face, mask.astype(np.float32) / 255.0,
                                 rgb, alpha, sigma=max(0.70, D.shape[0] / 1500.0))
        total_screen += s
        total_detail += d
    return total_screen, total_detail


def bake_facegen_glasses(
    D: np.ndarray,
    basis: Basis,
    shape_coeffs: np.ndarray,
    tex_coeffs: np.ndarray,
    color_name: str = "brown",
    style_name: str = "rectangular",
    asset_dir: str | Path | None = None,
    opacity: float = 0.84,
    scale_x: float | None = None,
    scale_y: float | None = None,
    offset_y: float = 0.0,
    rim_width: float = 1.0,
    align_to_eyes: bool = True,
    sports_overlay: bool = False,
    screen_scale: int = 3,
) -> BakeResult:
    """Bake local FaceGen glasses accessory into ``D`` in place."""
    if _has_custom_mesh_asset(asset_dir):
        # solid bake keeps per-triangle geometry so the diffuse/specular
        # shading and contact shadow read as a real 3D frame; the outline
        # bake is a flat 2D contour and is no longer used
        custom = bake_custom_gltf_accessory(
            D,
            basis,
            shape_coeffs,
            tex_coeffs,
            asset_dir=asset_dir,
            color_name=color_name,
            opacity=max(float(opacity), 0.88),
            scale_x=scale_x,
            scale_y=scale_y,
            offset_y=offset_y,
            screen_scale=screen_scale,
        )
        if custom.applied or style_name != "sports_goggle":
            return custom

    # Default route is the license-clean procedural mesh; the proprietary
    # FaceGen Modeller accessory is used only when the user explicitly points
    # --glasses-mesh-assets at a Modeller Accessories folder.
    root = find_accessory_dir(asset_dir) if asset_dir else None
    if root is None:
        if D.ndim != 3 or D.shape[2] != 3:
            return BakeResult(False, reason="detail map must be RGB")
        solid = _build_stock_glasses_mesh(
            basis,
            shape_coeffs,
            style=style_name,
            scale_x=scale_x,
            scale_y=scale_y,
            offset_y=offset_y,
            rim_width=rim_width,
        )
        if solid is None and style_name == "sports_goggle":
            solid = _build_sports_goggle_mesh(
                basis, shape_coeffs, scale_x=scale_x, scale_y=scale_y,
                offset_y=offset_y, rim_width=rim_width,
            )
        if solid is None:
            return BakeResult(False, reason="could not build glasses mesh")
        size = int(D.shape[0])
        canvas = max(size, int(size) * max(1, int(screen_scale)))
        face = _render_face_maps(basis, shape_coeffs, tex_coeffs, D, canvas)
        return _bake_solid_accessory(D, face, solid, color_name,
                                     max(float(opacity), 0.88))
    if D.ndim != 3 or D.shape[2] != 3:
        return BakeResult(False, reason="detail map must be RGB")

    accessory = _load_accessory(str(root))
    size = int(D.shape[0])
    canvas = max(size, int(size) * max(1, int(screen_scale)))
    face = _render_face_maps(basis, shape_coeffs, tex_coeffs, D, canvas)
    verts = _shape(accessory.verts, accessory.modes, shape_coeffs)
    style_scale = {
        "sports_goggle": (1.04, 0.68),
        "rectangular": (0.94, 0.70),
        "oval": (0.96, 0.82),
        "round": (0.92, 0.96),
    }.get(style_name, (0.94, 0.76))
    sx = float(style_scale[0] if scale_x is None else scale_x)
    sy = float(style_scale[1] if scale_y is None else scale_y)
    center = 0.5 * (verts.min(0) + verts.max(0))
    verts = verts.copy()
    verts[:, 0] = center[0] + sx * (verts[:, 0] - center[0])
    verts[:, 1] = center[1] + sy * (verts[:, 1] - center[1])
    if align_to_eyes:
        eye_target = _eye_target_screen(basis, shape_coeffs, face.lo,
                                        face.scale, canvas)
        if eye_target is not None:
            lens_sel = (
                (np.abs(verts[:, 0] - center[0]) < 62.0)
                & (verts[:, 2] > np.percentile(verts[:, 2], 40))
            )
            if lens_sel.any():
                lens_scr = (
                    render._screen_plane(verts[lens_sel], "ootp")
                    - face.lo
                ) * face.scale + canvas * 0.06
                lens_center = np.median(lens_scr, axis=0)
                verts[:, 0] += float((eye_target[0] - lens_center[0]) / face.scale)
                verts[:, 1] += float((lens_center[1] - eye_target[1]) / face.scale)
    verts[:, 1] += float(offset_y)
    scr = render._project_with_params(verts, canvas, "ootp", face.lo, face.scale)

    base_detail = D.astype(np.float32).copy()
    target_accum = np.zeros_like(base_detail)
    weight_accum = np.zeros(D.shape[:2], np.float32)
    frame_rgb = _frame_rgb(color_name)
    # faint rims dissolve at the game's small portrait scale
    opacity = float(np.clip(max(opacity, 0.88), 0.0, 1.0))

    tri_v = verts[accessory.tris]
    tri_s = scr[accessory.tris]
    e1 = tri_v[:, 1] - tri_v[:, 0]
    e2 = tri_v[:, 2] - tri_v[:, 0]
    normals = np.cross(e1, e2)
    nlen = np.maximum(np.linalg.norm(normals, axis=1), 1e-9)
    nz = normals[:, 2] / nlen
    sign = 1.0 if (nz > 0).sum() >= (nz < 0).sum() else -1.0
    facing = nz * sign
    order = np.argsort(tri_v[:, :, 2].mean(1))
    order = order[(facing > 0.005)[order]]

    screen_pixels = 0
    for ti in order:
        dst = tri_s[ti]
        x0, y0 = np.maximum(np.floor(dst.min(0)).astype(int), 0)
        x1 = min(int(np.ceil(dst[:, 0].max())) + 1, canvas)
        y1 = min(int(np.ceil(dst[:, 1].max())) + 1, canvas)
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        dst_local = (dst - [x0, y0]).astype(np.float32)
        M = cv2.getAffineTransform(accessory.src_px[ti].astype(np.float32),
                                   dst_local)
        patch = cv2.warpAffine(accessory.rgba, M, (x1 - x0, y1 - y0),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(0, 0, 0, 0))
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillConvexPoly(mask, np.round(dst_local).astype(np.int32), 1)
        mb = mask.astype(bool)
        if not mb.any():
            continue
        zz = render._tri_depth(dst_local, tri_v[ti][:, 2], x1 - x0, y1 - y0)
        fz = face.zbuf[y0:y1, x0:x1]
        fvalid = face.valid[y0:y1, x0:x1]
        # Treat low-alpha TGA regions as transparent lens/background. The solid
        # frame uses alpha ~254 in FaceGen's stock accessory texture.
        a = np.clip((patch[..., 3] - 48.0) / 207.0, 0.0, 1.0)
        visible = mb & fvalid & (a > 0.02) & (zz > fz - 1.0)
        if not visible.any():
            continue

        lit = 0.66 + 0.34 * max(float(facing[ti]), 0.0)
        tex_luma = patch[..., :3] @ np.array([0.299, 0.587, 0.114], np.float32)
        tex_luma = np.clip(tex_luma / 170.0, 0.45, 1.15)[..., None]
        target_rgb = np.clip(frame_rgb.reshape(1, 1, 3) * tex_luma * lit,
                             0, 255)
        face_rgb = np.maximum(face.rgb[y0:y1, x0:x1], 1.0)
        dxy = face.detail_xy[y0:y1, x0:x1]

        yy, xx = np.nonzero(visible)
        screen_pixels += int(len(xx))
        ix = np.clip(np.round(dxy[yy, xx, 0]).astype(int), 0, size - 1)
        iy = np.clip(np.round(dxy[yy, xx, 1]).astype(int), 0, size - 1)
        curD = np.maximum(base_detail[iy, ix], 1.0)
        alpha = (a[yy, xx] * opacity).astype(np.float32)[:, None]
        blended_rgb = (
            alpha * target_rgb[yy, xx]
            + (1.0 - alpha) * face_rgb[yy, xx]
        )
        targetD = np.clip(curD * blended_rgb / face_rgb[yy, xx], 0, 255)
        weights = np.clip(alpha[:, 0], 0.0, 1.0)

        for ch in range(3):
            np.add.at(target_accum[..., ch], (iy, ix), targetD[:, ch] * weights)
        np.add.at(weight_accum, (iy, ix), weights)

    if screen_pixels == 0 or not weight_accum.any():
        return BakeResult(False, source=str(root), color_name=color_name,
                          reason="accessory did not project onto face")

    target = base_detail.copy()
    hit = weight_accum > 1e-6
    target[hit] = target_accum[hit] / weight_accum[hit, None]

    # The accessory is rasterized in screen space and scattered back to detail
    # UVs. Smooth the accumulated target/alpha together to remove subpixel
    # holes without widening the frame like a morphological stroke would.
    alpha = np.clip(weight_accum, 0.0, 1.0)
    sigma = max(0.6, D.shape[0] / 1536.0)
    alpha_s = cv2.GaussianBlur(alpha, (0, 0), sigma)
    numer = cv2.GaussianBlur(target * alpha[..., None], (0, 0), sigma)
    denom = np.maximum(
        cv2.GaussianBlur(alpha, (0, 0), sigma)[..., None],
        1e-6,
    )
    target_s = numer / denom
    a = np.clip(alpha_s, 0.0, 1.0)[..., None]
    out = a * target_s + (1.0 - a) * base_detail

    D[:] = np.clip(out, 0, 255).astype(np.uint8)
    overlay_screen = 0
    overlay_detail = 0
    if style_name == "sports_goggle" and sports_overlay:
        overlay_screen, overlay_detail = _bake_sports_goggle_overlay(
            D, basis, shape_coeffs, face, frame_rgb, opacity
        )
    return BakeResult(
        True,
        source=str(root),
        color_name=color_name,
        screen_pixels=screen_pixels + overlay_screen,
        detail_pixels=int((weight_accum > 0.03).sum()) + overlay_detail,
    )
