"""Likeness Lab: build several .fg candidates per player and keep the winner.

The pipeline's knobs and identity priors have no single best setting: id-refine
lifts some players dramatically and overfits others; the CUFP prior is gold
when the index holds the right person and noise when it does not. The lab
builds a small candidate set, renders each candidate with the OOTP-style
renderer, scores every render against *all* of the player's photos with
ArcFace, and keeps the best — converting per-player variance into quality.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import identity
from .fgformat import FgFile
from .landmarks import detect
from .render import render as render_fg

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# label -> extra pipeline flags appended after the profile/user flags
VARIANTS: list[tuple[str, list[str]]] = [
    ("default", []),
    ("refine", ["--id-refine", "8", "--refine-size", "160",
                "--refine-r-max", "1.2"]),
    ("refine-strong", ["--id-refine", "8", "--refine-size", "160",
                       "--refine-r-max", "1.2", "--detail-strength", "1.25",
                       "--likeness-detail", "0.75"]),
    ("no-prior-refine", ["--retrieval", "off", "--photofit", "off",
                         "--modeller-fit", "off", "--id-refine", "8",
                         "--refine-size", "160", "--refine-r-max", "1.2"]),
]


@dataclass
class Candidate:
    label: str
    fg_path: Path
    score: float | None = None
    per_photo: list[float] = field(default_factory=list)
    render: np.ndarray | None = None
    error: str | None = None


def photo_embeddings(photos: Path,
                     id_model: str | None = None) -> list[np.ndarray]:
    """ArcFace embeddings for every usable player photo."""
    if not identity.available(id_model):
        return []
    if photos.is_file():
        paths = [photos]
    else:
        paths = sorted(p for p in photos.iterdir()
                       if p.suffix.lower() in IMAGE_EXTS)
    refs: list[np.ndarray] = []
    for p in paths:
        try:
            img = np.asarray(Image.open(p).convert("RGB"))
            emb = identity.embed(img, detect(img), id_model)
        except Exception:
            continue
        if emb is not None:
            refs.append(emb)
    return refs


def score_candidate(cand: Candidate, refs: list[np.ndarray],
                    id_model: str | None = None, size: int = 384) -> None:
    """Render the candidate .fg and fill in its mean similarity to the photos."""
    try:
        img, _ = render_fg(FgFile.read(cand.fg_path), size=size, aa=1)
        cand.render = np.asarray(img)
        emb = identity.embed(cand.render, detect(cand.render), id_model)
        if emb is None:
            cand.error = "render embedding failed"
            return
        cand.per_photo = [float(r @ emb) for r in refs]
        cand.score = float(np.mean(cand.per_photo))
    except Exception as exc:  # noqa: BLE001 - one bad candidate must not kill the lab
        cand.error = f"{type(exc).__name__}: {exc}"


def contact_sheet(cands: list[Candidate], out_path: Path,
                  winner_label: str) -> None:
    """Side-by-side sheet of every rendered candidate with its score."""
    tiles = [c for c in cands if c.render is not None]
    if not tiles:
        return
    th = 320
    imgs = []
    for c in tiles:
        im = Image.fromarray(c.render)
        im = im.resize((max(1, round(im.width * th / im.height)), th),
                       Image.LANCZOS)
        imgs.append(im)
    pad, cap = 12, 46
    width = sum(im.width for im in imgs) + pad * (len(imgs) + 1)
    sheet = Image.new("RGB", (width, th + cap + pad * 2), (245, 245, 247))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("segoeui.ttf", 15)
    except Exception:
        font = ImageFont.load_default()
    x = pad
    for c, im in zip(tiles, imgs):
        sheet.paste(im, (x, pad))
        is_winner = c.label == winner_label
        if is_winner:
            draw.rectangle([x - 3, pad - 3, x + im.width + 2, pad + th + 2],
                           outline=(53, 116, 78), width=3)
        score = f"{c.score:.3f}" if c.score is not None else "failed"
        text = f"{c.label}  {score}" + ("  [BEST]" if is_winner else "")
        draw.text((x, pad + th + 8), text, fill=(32, 38, 31), font=font)
        x += im.width + pad
    sheet.save(out_path)
