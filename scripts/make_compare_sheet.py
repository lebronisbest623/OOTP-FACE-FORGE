"""Create an original-photo vs FaceGen-preview contact sheet.

Inputs can be run folders, manifest files, or parent folders containing
`*.manifest.json`. If a run was built with --no-preview, this script renders the
FG file into a small cache next to the output sheet.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ootp_faceforge.fgformat import FgFile  # noqa: E402
from ootp_faceforge.render import render  # noqa: E402


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _resolve(path: str | Path, base: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def _find_manifests(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for path in paths:
        if path.is_file() and path.name.endswith(".manifest.json"):
            out.append(path)
        elif path.is_dir():
            if path.name == "meta":
                out.extend(sorted(path.glob("*.manifest.json")))
            else:
                out.extend(sorted(path.rglob("*.manifest.json")))
    seen: set[str] = set()
    deduped: list[Path] = []
    for path in out:
        key = str(path.resolve()).lower()
        if key not in seen:
            seen.add(key)
            deduped.append(path)
    return deduped


def _choose_photo(manifest: dict, project_root: Path) -> Path | None:
    raw = manifest.get("inputs", {}).get("photos")
    if not raw:
        return None
    root = _resolve(raw, project_root)
    if root.is_file() and root.suffix.lower() in IMAGE_EXTS:
        return root
    if not root.is_dir():
        return None

    tex = str(manifest.get("diagnostics", {}).get("texture_photo", ""))
    parts = tex.split(maxsplit=1)
    if len(parts) == 2:
        candidate = root / parts[1]
        if candidate.is_file():
            return candidate
    images = sorted(p for p in root.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    return images[0] if images else None


def _render_preview(manifest: dict, manifest_path: Path, out_dir: Path,
                    project_root: Path, size: int, aa: int) -> Path | None:
    outputs = manifest.get("outputs", {})
    preview = outputs.get("preview")
    if preview:
        path = _resolve(preview, project_root)
        if path.is_file():
            return path

    fg_raw = outputs.get("fg_export") or outputs.get("fg")
    if not fg_raw:
        return None
    fg_path = _resolve(fg_raw, project_root)
    if not fg_path.is_file():
        return None

    slug = manifest.get("player", {}).get("slug") or manifest_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{slug}_preview.png"
    if not out_path.exists() or out_path.stat().st_mtime < fg_path.stat().st_mtime:
        img, _ = render(FgFile.read(str(fg_path)), size=size, aa=aa)
        Image.fromarray(img).save(out_path)
    return out_path


def _thumb(path: Path, size: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img = ImageOps.contain(img, (size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (232, 232, 232))
    x = (size - img.width) // 2
    y = (size - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def _draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str,
               font: ImageFont.ImageFont, fill=(30, 30, 30)) -> None:
    safe = text.encode("ascii", errors="replace").decode("ascii")
    draw.text(xy, safe, fill=fill, font=font)


def make_sheet(manifests: list[Path], out: Path, project_root: Path,
               thumb_size: int, render_size: int, aa: int,
               max_rows: int) -> None:
    render_cache = out.parent / f"{out.stem}_renders"
    rows = []
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        photo = _choose_photo(manifest, project_root)
        preview = _render_preview(
            manifest, manifest_path, render_cache, project_root, render_size, aa
        )
        if photo is None or preview is None:
            continue
        rows.append((manifest, photo, preview))
        if max_rows and len(rows) >= max_rows:
            break

    if not rows:
        raise SystemExit("no comparable manifest rows found")

    font = ImageFont.load_default()
    margin = 18
    gap = 16
    label_h = 28
    header_h = 28
    row_h = thumb_size + label_h + gap
    width = margin * 2 + thumb_size * 2 + gap
    height = margin * 2 + header_h + row_h * len(rows)
    sheet = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(sheet)
    x0 = margin
    x1 = margin + thumb_size + gap
    y = margin
    _draw_text(draw, (x0, y), "Original", font)
    _draw_text(draw, (x1, y), "FaceGen", font)
    y += header_h

    for manifest, photo, preview in rows:
        name = manifest.get("player", {}).get("name") or photo.stem
        _draw_text(draw, (x0, y), str(name)[:44], font)
        diag = manifest.get("diagnostics", {})
        mf = str(diag.get("modeller_fit") or diag.get("identity") or "")
        _draw_text(draw, (x1, y), mf[:54], font, fill=(70, 70, 70))
        y += label_h
        sheet.paste(_thumb(photo, thumb_size), (x0, y))
        sheet.paste(_thumb(preview, thumb_size), (x1, y))
        y += thumb_size + gap

    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"wrote {out} rows={len(rows)}")
    if render_cache.exists():
        print(f"render cache: {render_cache}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("build/compare_sheet.png"))
    parser.add_argument("--thumb-size", type=int, default=256)
    parser.add_argument("--render-size", type=int, default=512)
    parser.add_argument("--aa", type=int, default=2)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    manifests = _find_manifests(args.paths)
    if not manifests:
        raise SystemExit("no manifest files found")
    make_sheet(
        manifests,
        args.out,
        project_root,
        args.thumb_size,
        args.render_size,
        args.aa,
        args.max_rows,
    )


if __name__ == "__main__":
    main()
