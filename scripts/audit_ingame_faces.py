#!/usr/bin/env python
"""Audit whether OOTP in-game portraits actually use the intended .fg faces.

For every roster player that has both an in-game person_picture and an intended
FaceForge .fg, render the .fg with the local preview renderer, embed both with
ArcFace, and score the similarity. High similarity means the game is showing
the intended face; low similarity means the game is showing something else
(face never imported, wrong lookup id, or a stale cached picture).

Outputs a CSV report plus an HTML page with side-by-side thumbnails, sorted
worst-first so re-import work can be prioritized.

  python scripts/audit_ingame_faces.py --save "...\\Ultimate_KBO.lg" ^
      --fg-dir "%USERPROFILE%\\FaceForgeWorkspace\\exports\\kbo_fg_lahman_source_20260704T185842" ^
      --out build\\ingame_audit
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ootp_faceforge import identity  # noqa: E402
from ootp_faceforge.fgformat import FgFile  # noqa: E402
from ootp_faceforge.landmarks import detect  # noqa: E402
from ootp_faceforge.render import render  # noqa: E402

MATCH_T = 0.40
MISMATCH_T = 0.22


def parse_roster(path: Path) -> list[dict[str, str]]:
    header, rows = None, []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line.startswith("//id"):
                header = [c.strip() for c in line[2:].split(",")]
                continue
            if not line.strip() or line.startswith("//"):
                continue
            if header is None:
                continue
            cells = next(csv.reader([line]))
            if cells and cells[-1] == "eol":
                cells = cells[:-1]
            if len(cells) < len(header):
                cells.extend([""] * (len(header) - len(cells)))
            rows.append(dict(zip(header, cells)))
    return rows


def embed(img: np.ndarray, id_model: str | None):
    try:
        return identity.embed(img, detect(img), id_model)
    except Exception:
        return None


def thumb_b64(img: np.ndarray, height: int = 150) -> str:
    im = Image.fromarray(img)
    im = im.resize((max(1, round(im.width * height / im.height)), height),
                   Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--save", type=Path, required=True,
                    help="OOTP .lg save folder")
    ap.add_argument("--fg-dir", type=Path, action="append", required=True,
                    help="Folder(s) with intended .fg files named by lahman_id "
                         "(pass more than once to search several)")
    ap.add_argument("--out", type=Path, default=ROOT / "build" / "ingame_audit")
    ap.add_argument("--limit", type=int, default=0,
                    help="Audit at most N players (0 = all)")
    ap.add_argument("--id-model", default=None)
    args = ap.parse_args()

    pics = args.save / "news" / "html" / "images" / "person_pictures"
    roster = args.save / "import_export" / "kbo_rosters.txt"
    fg_by_stem: dict[str, Path] = {}
    for d in args.fg_dir:
        for p in sorted(d.rglob("*.fg")):
            fg_by_stem.setdefault(p.stem.lower(), p)

    pairs = []
    for r in parse_roster(roster):
        lah = (r.get("lahman_id") or "").strip().lower()
        pid = (r.get("id") or "").strip()
        fg = fg_by_stem.get(lah)
        pic = pics / f"player_{pid}.png"
        if lah and pid and fg and pic.exists():
            name = f"{r.get('FirstName', '')} {r.get('LastName', '')}".strip()
            pairs.append((pid, lah, name, fg, pic))
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"auditing {len(pairs)} players "
          f"(fg sources: {sum(1 for _ in fg_by_stem)} files)")

    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for i, (pid, lah, name, fg_path, pic) in enumerate(pairs, 1):
        row = {"player_id": pid, "lahman_id": lah, "name": name,
               "fg": str(fg_path), "picture": str(pic),
               "sim": None, "verdict": "error"}
        try:
            game = np.asarray(Image.open(pic).convert("RGB"))
            gb = cv2.resize(game, (game.shape[1] * 4, game.shape[0] * 4),
                            interpolation=cv2.INTER_CUBIC)
            ours, _ = render(FgFile.read(fg_path), size=512, aa=1)
            e_g, e_o = embed(gb, args.id_model), embed(ours, args.id_model)
            if e_g is not None and e_o is not None:
                sim = float(e_g @ e_o)
                row["sim"] = round(sim, 4)
                row["verdict"] = ("match" if sim >= MATCH_T
                                  else "mismatch" if sim < MISMATCH_T
                                  else "uncertain")
                row["_ours"] = thumb_b64(ours)
                row["_game"] = thumb_b64(gb)
        except Exception as exc:  # noqa: BLE001
            row["verdict"] = f"error: {type(exc).__name__}"
        results.append(row)
        if i % 25 == 0:
            done = sum(1 for r in results if r["sim"] is not None)
            print(f"  {i}/{len(pairs)} scored={done}")

    scored = [r for r in results if r["sim"] is not None]
    scored.sort(key=lambda r: r["sim"])
    counts = {}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    print("verdicts:", json.dumps(counts, ensure_ascii=False))

    csv_path = args.out / "ingame_audit.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "player_id", "lahman_id", "name", "sim", "verdict",
            "fg", "picture"])
        writer.writeheader()
        for r in scored + [r for r in results if r["sim"] is None]:
            writer.writerow({k: r.get(k) for k in writer.fieldnames})

    rows_html = []
    for r in scored:
        color = {"match": "#2e7d4f", "mismatch": "#b3402a"}.get(
            r["verdict"], "#8a7a2c")
        rows_html.append(
            f"<tr><td><img src='data:image/jpeg;base64,{r['_ours']}'></td>"
            f"<td><img src='data:image/jpeg;base64,{r['_game']}'></td>"
            f"<td>{r['name']}<br><small>{r['lahman_id']} &rarr; "
            f"player_{r['player_id']}</small></td>"
            f"<td style='color:{color};font-weight:600'>{r['sim']:.2f}<br>"
            f"<small>{r['verdict']}</small></td></tr>")
    html = (
        "<meta charset='utf-8'><title>In-game face audit</title>"
        "<style>body{font-family:Segoe UI,sans-serif;background:#f4f4f2}"
        "table{border-collapse:collapse}td{padding:6px 10px;border-bottom:"
        "1px solid #ddd;vertical-align:middle}img{height:110px}</style>"
        f"<h2>In-game face audit — {json.dumps(counts)}</h2>"
        "<p>Sorted worst-first. 'mismatch' players show a different face "
        "in game than the intended .fg.</p>"
        "<table><tr><th>intended (.fg render)</th><th>in-game</th>"
        "<th>player</th><th>sim</th></tr>" + "".join(rows_html) + "</table>")
    (args.out / "ingame_audit.html").write_text(html, encoding="utf-8")
    print("wrote", csv_path)
    print("wrote", args.out / "ingame_audit.html")


if __name__ == "__main__":
    main()
