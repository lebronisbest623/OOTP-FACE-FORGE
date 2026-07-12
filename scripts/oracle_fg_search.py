"""Oracle-search OOTP FaceGen coefficient space for one target photo.

This is intentionally not a production generator. It answers a narrower
question: does the available .fg coefficient space contain a render that scores
closer to the target photo than the normal fast pipeline result?
"""
from __future__ import annotations

import argparse
import heapq
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ootp_faceforge.basis import get_basis  # noqa: E402
from ootp_faceforge.fgformat import FgFile  # noqa: E402
from ootp_faceforge.identity import DEFAULT_MODEL, _RenderScorer, embed  # noqa: E402
from ootp_faceforge.landmarks import detect  # noqa: E402
from ootp_faceforge import photofit  # noqa: E402
from ootp_faceforge.render import render  # noqa: E402


@dataclass
class Hit:
    score: float
    sim: float
    label: str
    source: str
    coeff: np.ndarray
    seed_rms: float
    norm: float


def _shape_from_fg(fg: FgFile, n_sym: int, n_asym: int) -> np.ndarray:
    c = np.zeros(n_sym + n_asym, np.float32)
    c[: min(n_sym, len(fg.sym_shape))] = fg.sym_shape[:n_sym]
    a = min(n_asym, len(fg.asym_shape))
    c[n_sym : n_sym + a] = fg.asym_shape[:a]
    return c


def _make_fg(seed: FgFile, coeff: np.ndarray, n_sym: int) -> FgFile:
    return FgFile(
        geo_basis_version=seed.geo_basis_version,
        tex_basis_version=seed.tex_basis_version,
        sym_shape=coeff[:n_sym].astype(np.float32),
        asym_shape=coeff[n_sym:].astype(np.float32),
        sym_tex=seed.sym_tex.astype(np.float32),
        asym_tex=seed.asym_tex.astype(np.float32),
        detail_jpeg=seed.detail_jpeg,
    )


def _push_hit(heap: list[tuple[float, int, Hit]], hit: Hit, limit: int,
              seq: int) -> None:
    item = (hit.score, seq, hit)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif hit.score > heap[0][0]:
        heapq.heapreplace(heap, item)


def _draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    x, y = xy
    draw.rectangle([x, y, x + 236, y + 35], fill=(255, 255, 255))
    draw.text((x + 4, y + 3), text, fill=(20, 20, 20))


def _write_sheet(photo_path: Path, hits: list[Hit], preview_paths: list[Path],
                 out: Path, thumb: int) -> None:
    cols = 4
    label_h = 40
    pad = 14
    rows = 1 + int(np.ceil(len(preview_paths) / cols))
    w = cols * thumb + (cols + 1) * pad
    h = rows * (thumb + label_h) + (rows + 1) * pad
    sheet = Image.new("RGB", (w, h), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)

    src = Image.open(photo_path).convert("RGB")
    src.thumbnail((thumb, thumb))
    x = pad
    y = pad + label_h
    sheet.paste(src, (x + (thumb - src.width) // 2, y + (thumb - src.height) // 2))
    _draw_label(draw, (x, pad), "target photo")

    for i, (hit, path) in enumerate(zip(hits, preview_paths), start=1):
        slot = i
        col = slot % cols
        row = slot // cols
        x = pad + col * (thumb + pad)
        y0 = pad + row * (thumb + label_h + pad)
        img = Image.open(path).convert("RGB")
        img.thumbnail((thumb, thumb))
        sheet.paste(img, (x + (thumb - img.width) // 2,
                          y0 + label_h + (thumb - img.height) // 2))
        label = (
            f"#{i} {hit.label}\n"
            f"sim={hit.sim:.3f} rms={hit.seed_rms:.2f} norm={hit.norm:.1f}"
        )
        _draw_label(draw, (x, y0), label)

    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


def _score_coeff(scorer: _RenderScorer, target_emb: np.ndarray,
                 coeff: np.ndarray, seed_coeff: np.ndarray,
                 norm_penalty: float) -> tuple[float, float, float, float]:
    img = scorer.render(coeff)
    emb = scorer.embedding(img)
    if emb is None:
        return -1.0, -1.0, float("inf"), float(np.linalg.norm(coeff))
    sim = float(target_emb @ emb)
    diff = coeff[: scorer.basis.n_sym] - seed_coeff[: scorer.basis.n_sym]
    seed_rms = float(np.sqrt(np.mean(diff * diff)))
    norm = float(np.linalg.norm(coeff[: scorer.basis.n_sym]))
    score = sim - float(norm_penalty) * seed_rms
    return score, sim, seed_rms, norm


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("photo")
    p.add_argument("seed_fg")
    p.add_argument("--index", default=str(ROOT / "models" / "cufp_identity_index.npz"))
    p.add_argument("--out-dir", default=str(ROOT / "build" / "oracle_fg_search"))
    p.add_argument("--scan", type=int, default=8000,
                   help="Prototype rows to scan from the index; 0 means all rows.")
    p.add_argument("--prefilter", type=int, default=0,
                   help="Use photo embedding/geometry to preselect this many "
                        "index rows before expensive render scoring. 0 disables.")
    p.add_argument("--prefilter-geom-weight", type=float, default=0.18)
    p.add_argument("--random", type=int, default=4000,
                   help="Empirical random candidates sampled from index statistics.")
    p.add_argument("--cem-rounds", type=int, default=4)
    p.add_argument("--cem-pop", type=int, default=256)
    p.add_argument("--top", type=int, default=16)
    p.add_argument("--render-size", type=int, default=160)
    p.add_argument("--preview-size", type=int, default=512)
    p.add_argument("--sheet-thumb", type=int, default=192)
    p.add_argument("--seed", type=int, default=52001)
    p.add_argument("--norm-penalty", type=float, default=0.0)
    p.add_argument("--use-asym", action="store_true",
                   help="Let asym shape modes vary. Default freezes asymmetry.")
    p.add_argument("--progress-every", type=int, default=250)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    basis = get_basis()
    n_sym = basis.n_sym
    n_asym = basis.n_asym
    nm = n_sym + n_asym

    photo_path = Path(args.photo)
    seed_path = Path(args.seed_fg)
    seed_fg = FgFile.read(str(seed_path))
    seed_coeff = _shape_from_fg(seed_fg, n_sym, n_asym)

    photo = np.asarray(Image.open(photo_path).convert("RGB"))
    lms = detect(photo)
    target_feature = photofit.feature_from_image(photo, lms, DEFAULT_MODEL)
    if target_feature is None:
        raise SystemExit("target embedding failed")
    target_emb = target_feature.embedding

    scorer = _RenderScorer(basis, seed_fg.sym_tex, seed_fg.detail_jpeg,
                           args.render_size, DEFAULT_MODEL)
    if not scorer.realign(scorer.render(seed_coeff)):
        raise SystemExit("render embedding alignment failed")

    index_path = Path(args.index)
    data = np.load(index_path, allow_pickle=False)
    coeffs = data["coeffs"].astype(np.float32)
    names = data["names"].astype(str) if "names" in data else np.array(
        [f"row{i}" for i in range(len(coeffs))]
    )
    shapes = coeffs[:, :nm].copy()
    if not args.use_asym:
        shapes[:, n_sym:] = seed_coeff[n_sym:]

    index_rows = np.arange(len(shapes))
    if int(args.prefilter) > 0 and "embeddings" in data and "geom" in data:
        n_pref = min(int(args.prefilter), len(shapes))
        emb_scores = data["embeddings"].astype(np.float32) @ target_emb
        geom = (
            target_feature.feature[512:].astype(np.float32)
            - data["geom_mean"].astype(np.float32)
        ) / np.maximum(data["geom_scale"].astype(np.float32), 1e-6)
        gn = float(np.linalg.norm(geom))
        if gn > 1e-9:
            geom = geom / gn
        geom_scores = data["geom"].astype(np.float32) @ geom
        pre_scores = emb_scores + float(args.prefilter_geom_weight) * geom_scores
        index_rows = np.argpartition(-pre_scores, n_pref - 1)[:n_pref]
        index_rows = index_rows[np.argsort(-pre_scores[index_rows])]
        print(
            f"prefilter rows={len(index_rows)} "
            f"best={float(pre_scores[index_rows[0]]):.4f} "
            f"name={names[index_rows[0]]}",
            flush=True,
        )

    lo = np.percentile(shapes[:, :n_sym], 0.25, axis=0)
    hi = np.percentile(shapes[:, :n_sym], 99.75, axis=0)
    mu = shapes[:, :n_sym].mean(axis=0)
    sd = np.maximum(shapes[:, :n_sym].std(axis=0), 0.15)

    heap: list[tuple[float, int, Hit]] = []
    seq = 0
    evals = 0
    t0 = time.time()

    def consider(coeff: np.ndarray, label: str, source: str) -> None:
        nonlocal seq, evals
        coeff = coeff.astype(np.float32, copy=True)
        coeff[:n_sym] = np.clip(coeff[:n_sym], lo, hi)
        coeff[:n_sym] = np.clip(coeff[:n_sym], -4.0, 4.0)
        if not args.use_asym:
            coeff[n_sym:] = seed_coeff[n_sym:]
        else:
            coeff[n_sym:] = np.clip(coeff[n_sym:], -4.0, 4.0)
        score, sim, seed_rms, norm = _score_coeff(
            scorer, target_emb, coeff, seed_coeff, args.norm_penalty
        )
        hit = Hit(score, sim, label, source, coeff, seed_rms, norm)
        _push_hit(heap, hit, max(args.top, 32), seq)
        seq += 1
        evals += 1
        if args.progress_every > 0 and evals % args.progress_every == 0:
            best = max(heap, key=lambda item: item[0])[2]
            dt = time.time() - t0
            print(
                f"progress evals={evals} best={best.sim:.4f} "
                f"label={best.label} source={best.source} elapsed={dt:.1f}s",
                flush=True,
            )

    consider(seed_coeff, "seed", "seed")

    total = len(index_rows)
    if args.scan == 0 or args.scan >= total:
        scan_idx = index_rows
    else:
        picked = rng.choice(total, size=max(args.scan, 0), replace=False)
        scan_idx = index_rows[picked]
    for row in scan_idx:
        consider(shapes[int(row)], str(names[int(row)]), "prototype")

    for i in range(max(args.random, 0)):
        coeff = seed_coeff.copy()
        coeff[:n_sym] = rng.normal(mu, sd)
        consider(coeff, f"rand{i:05d}", "empirical_random")

    for round_i in range(max(args.cem_rounds, 0)):
        ranked = sorted((item[2] for item in heap), key=lambda h: h.score,
                        reverse=True)
        elite = ranked[: max(4, min(12, len(ranked)))]
        elite_c = np.stack([h.coeff[:n_sym] for h in elite])
        elite_scores = np.asarray([h.score for h in elite], np.float64)
        logits = elite_scores - float(elite_scores.max())
        weights = np.exp(np.clip(logits / 0.03, -50.0, 0.0))
        weights = weights / max(float(weights.sum()), 1e-9)
        cem_mu = (elite_c * weights[:, None]).sum(axis=0)
        cem_sd = np.sqrt(
            ((elite_c - cem_mu) ** 2 * weights[:, None]).sum(axis=0)
        )
        floor = max(0.08, 0.32 * (0.72 ** round_i))
        cem_sd = np.maximum(cem_sd, floor)
        for j in range(max(args.cem_pop, 0)):
            coeff = seed_coeff.copy()
            coeff[:n_sym] = rng.normal(cem_mu, cem_sd)
            consider(coeff, f"cem{round_i:02d}_{j:04d}", "cem")
        best = max(heap, key=lambda item: item[0])[2]
        print(
            f"round {round_i + 1}/{args.cem_rounds} best={best.sim:.4f} "
            f"label={best.label} source={best.source}",
            flush=True,
        )

    hits = sorted((item[2] for item in heap), key=lambda h: h.score,
                  reverse=True)[: args.top]
    preview_paths: list[Path] = []
    fg_paths: list[Path] = []
    for i, hit in enumerate(hits, start=1):
        fg = _make_fg(seed_fg, hit.coeff, n_sym)
        fg_path = out_dir / f"oracle_{i:02d}_{hit.source}_{hit.label}.fg"
        safe_fg_path = Path(str(fg_path).replace(":", "_").replace("/", "_"))
        fg.write(str(safe_fg_path))
        fg_paths.append(safe_fg_path)
        img, _ = render(fg, size=args.preview_size, shade=True, aa=2)
        preview_path = out_dir / f"oracle_{i:02d}_{hit.source}_{hit.label}.png"
        safe_preview_path = Path(
            str(preview_path).replace(":", "_").replace("/", "_")
        )
        Image.fromarray(img).save(safe_preview_path)
        preview_paths.append(safe_preview_path)

    sheet_path = out_dir / "oracle_sheet.png"
    _write_sheet(photo_path, hits, preview_paths, sheet_path, args.sheet_thumb)

    report = {
        "photo": str(photo_path),
        "seed_fg": str(seed_path),
        "index": str(index_path),
        "evals": evals,
        "elapsed_sec": round(time.time() - t0, 3),
        "render_size": args.render_size,
        "scan": int(len(scan_idx)),
        "random": int(max(args.random, 0)),
        "cem_rounds": int(max(args.cem_rounds, 0)),
        "cem_pop": int(max(args.cem_pop, 0)),
        "use_asym": bool(args.use_asym),
        "norm_penalty": float(args.norm_penalty),
        "sheet": str(sheet_path),
        "top": [
            {
                "rank": i,
                "score": h.score,
                "sim": h.sim,
                "label": h.label,
                "source": h.source,
                "seed_rms": h.seed_rms,
                "sym_norm": h.norm,
                "fg": str(fg_paths[i - 1]),
                "preview": str(preview_paths[i - 1]),
            }
            for i, h in enumerate(hits, start=1)
        ],
    }
    report_path = out_dir / "oracle_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(
        f"done evals={evals} elapsed={report['elapsed_sec']:.1f}s "
        f"best={hits[0].sim:.4f} sheet={sheet_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
