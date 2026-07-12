"""Build/evaluate a render-domain FaceGen retrieval index.

The expensive work happens here: each .fg/coefficient row is rendered with the
same OOTP renderer used at runtime, landmarked, embedded, and stored. Runtime
then needs only one photo feature extraction plus a vector lookup.

Usage:
  python scripts/build_render_index.py build --official-limit 500 --cufp-limit 500
  python scripts/build_render_index.py eval --limit 100
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ootp_faceforge import photofit, render, render_retrieval  # noqa: E402
from ootp_faceforge.fgformat import FgFile  # noqa: E402
from ootp_faceforge.landmarks import detect  # noqa: E402
from ootp_faceforge.paths import get_ootp_3d_path  # noqa: E402


DEFAULT_CUFP_DIR = Path(r"C:\Users\user\facegen\data\cufp_photo_dataset")
N_SYM, N_ASYM, N_TEX = 50, 30, 50


@dataclass(frozen=True)
class SourceRow:
    name: str
    source: str
    coeff: np.ndarray
    fg: FgFile


def _unit_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-9)


def _coeff_from_fg(fg: FgFile) -> np.ndarray | None:
    if (
        len(fg.sym_shape) != N_SYM
        or len(fg.asym_shape) != N_ASYM
        or len(fg.sym_tex) != N_TEX
    ):
        return None
    return np.concatenate([fg.sym_shape, fg.asym_shape, fg.sym_tex]).astype(np.float32)


def _fg_from_coeff(coeff: np.ndarray) -> FgFile:
    coeff = np.asarray(coeff, np.float32).reshape(-1)
    if coeff.shape[0] < N_SYM + N_ASYM + N_TEX:
        raise ValueError(f"coeff too short: {coeff.shape}")
    a = N_SYM
    b = a + N_ASYM
    c = b + N_TEX
    return FgFile(
        sym_shape=coeff[:a],
        asym_shape=coeff[a:b],
        sym_tex=coeff[b:c],
        asym_tex=np.zeros(0),
        detail_jpeg=None,
    )


def _default_official_fg_dir() -> Path:
    return get_ootp_3d_path().parents[1] / "fg_files"


def _iter_official_rows(fg_dir: Path, limit: int, seed: int):
    paths = sorted(Path(fg_dir).glob("*.fg"))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(paths)) if paths else []
    if limit:
        order = order[:limit]
    for i in order:
        path = paths[int(i)]
        try:
            fg = FgFile.read(str(path))
            coeff = _coeff_from_fg(fg)
            if coeff is None:
                continue
        except Exception:
            continue
        yield SourceRow(path.stem, "official", coeff, fg)


def _load_cufp_labels(data_dir: Path):
    labels_path = data_dir / "labels.npz"
    if not labels_path.is_file():
        raise SystemExit(f"missing labels: {labels_path}")
    data = np.load(labels_path, allow_pickle=True)
    required = {"names", "coeffs", "is_val"}
    missing = required - set(data.files)
    if missing:
        raise SystemExit(f"labels missing keys: {sorted(missing)}")
    return data


def _iter_cufp_rows(data_dir: Path, split: str, limit: int, seed: int):
    labels = _load_cufp_labels(data_dir)
    names = labels["names"].astype(str)
    coeffs = labels["coeffs"].astype(np.float32)
    is_val = labels["is_val"].astype(bool)
    if split == "train":
        idx = np.nonzero(~is_val)[0]
    elif split == "val":
        idx = np.nonzero(is_val)[0]
    elif split == "all":
        idx = np.arange(len(is_val))
    else:
        raise ValueError(f"unknown split: {split}")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(idx)
    if limit:
        idx = idx[:limit]
    for i in idx:
        coeff = coeffs[int(i)]
        if coeff.shape[0] < N_SYM + N_ASYM + N_TEX:
            continue
        yield SourceRow(str(names[int(i)]), "cufp", coeff[:N_SYM + N_ASYM + N_TEX], _fg_from_coeff(coeff))


def _feature_for_fg(fg: FgFile, size: int, aa: int, id_model: str | None):
    img, _ = render.render(fg, size=size, aa=aa, shade=True, include_eyes=True)
    lms = detect(img)
    feat = photofit.feature_from_image(img, lms, id_model)
    if feat is None:
        raise RuntimeError("embedding failed")
    return feat


def stage_build(args: argparse.Namespace) -> None:
    out_names = []
    out_sources = []
    out_coeffs = []
    out_emb = []
    out_geom = []
    skipped: list[dict[str, str]] = []
    rows: list[SourceRow] = []
    sources = set(args.sources.split(","))

    if "official" in sources:
        rows.extend(_iter_official_rows(
            Path(args.official_fg_dir or _default_official_fg_dir()),
            args.official_limit,
            args.seed,
        ))
    if "cufp" in sources:
        rows.extend(_iter_cufp_rows(
            Path(args.cufp_dir),
            args.cufp_split,
            args.cufp_limit,
            args.seed + 1,
        ))

    if not rows:
        raise SystemExit("no source rows")

    t0 = time.time()
    for n_done, row in enumerate(rows, 1):
        try:
            feat = _feature_for_fg(row.fg, args.size, args.aa, args.id_model)
        except Exception as exc:  # noqa: BLE001 - offline index should continue
            skipped.append({
                "name": row.name,
                "source": row.source,
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue
        out_names.append(row.name)
        out_sources.append(row.source)
        out_coeffs.append(row.coeff)
        out_emb.append(feat.embedding)
        out_geom.append(feat.feature[512:])
        if args.report_every and n_done % args.report_every == 0:
            print(
                f"{n_done}/{len(rows)} indexed={len(out_names)} "
                f"skipped={len(skipped)} elapsed={time.time() - t0:.1f}s",
                flush=True,
            )

    if not out_names:
        raise SystemExit("no render index rows generated")

    emb = _unit_rows(np.stack(out_emb).astype(np.float32))
    geom_raw = np.stack(out_geom).astype(np.float32)
    geom_mean = geom_raw.mean(0).astype(np.float32)
    geom_scale = geom_raw.std(0).astype(np.float32)
    geom_scale[geom_scale < 1e-6] = 1.0
    geom = _unit_rows((geom_raw - geom_mean) / geom_scale).astype(np.float32)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        schema="fg_render_identity_index_v1",
        names=np.asarray(out_names),
        sources=np.asarray(out_sources),
        coeffs=np.stack(out_coeffs).astype(np.float32),
        embeddings=emb,
        geom=geom,
        geom_mean=geom_mean,
        geom_scale=geom_scale,
        n_sym=N_SYM,
        n_asym=N_ASYM,
        n_tex=N_TEX,
        render_size=int(args.size),
        render_aa=int(args.aa),
        feature_lms=np.asarray(photofit.FEATURE_LMS, np.int32),
        feature_names=np.asarray(photofit.feature_names()),
    )
    report = {
        "out": str(out),
        "sources": sorted(sources),
        "requested": int(len(rows)),
        "indexed": int(len(out_names)),
        "skipped": int(len(skipped)),
        "elapsed_sec": round(time.time() - t0, 3),
        "render_size": int(args.size),
        "render_aa": int(args.aa),
        "skip_examples": skipped[:20],
    }
    report_path = out.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}: indexed={len(out_names)} skipped={len(skipped)}")
    print(f"wrote {report_path}")


def _cufp_photo_path(data_dir: Path, labels, row_idx: int) -> Path:
    names = labels["names"].astype(str)
    photo_paths = labels["photo_paths"].astype(str) if "photo_paths" in labels.files else names
    name = str(names[row_idx])
    filename = Path(str(photo_paths[row_idx]).replace("\\", "/")).name
    candidates = [
        data_dir / "images" / filename,
        data_dir / "images" / f"{name}.jpg",
        data_dir / str(photo_paths[row_idx]),
    ]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def stage_eval(args: argparse.Namespace) -> None:
    from PIL import Image

    data_dir = Path(args.cufp_dir)
    labels = _load_cufp_labels(data_dir)
    names = labels["names"].astype(str)
    coeffs = labels["coeffs"].astype(np.float32)
    is_val = labels["is_val"].astype(bool)
    if args.split == "train":
        idx = np.nonzero(~is_val)[0]
    elif args.split == "val":
        idx = np.nonzero(is_val)[0]
    else:
        idx = np.arange(len(is_val))
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(idx)
    if args.limit:
        idx = idx[:args.limit]

    index = render_retrieval.RenderIndex(args.index)
    rows = []
    skipped = 0
    t0 = time.time()
    for row_idx in idx:
        name = str(names[int(row_idx)])
        path = _cufp_photo_path(data_dir, labels, int(row_idx))
        try:
            img = np.asarray(Image.open(path).convert("RGB"))
            lms = detect(img)
            feat = photofit.feature_from_image(img, lms, args.id_model)
            if feat is None:
                raise RuntimeError("embedding failed")
            pred = index.predict_feature(
                feat.feature,
                top_k=args.top_k,
                geom_weight=args.geom_weight,
                temperature=args.temperature,
            )
        except Exception:
            skipped += 1
            continue
        err = pred.raw - coeffs[int(row_idx), :N_SYM + N_ASYM + N_TEX]
        top = pred.hits[0]
        rows.append({
            "name": name,
            "top1": top.name,
            "top1_source": top.source,
            "top1_score": top.score,
            "top1_emb": top.emb_score,
            "top1_geom": top.geom_score,
            "confidence": pred.confidence,
            "coeff_mse": float(np.mean(err ** 2)),
            "shape_mse": float(np.mean(err[:N_SYM + N_ASYM] ** 2)),
            "self_in_topk": any(hit.name == name for hit in pred.hits),
        })

    if not rows:
        raise SystemExit("no eval rows")
    coeff_mse = np.asarray([row["coeff_mse"] for row in rows])
    shape_mse = np.asarray([row["shape_mse"] for row in rows])
    self_hits = np.asarray([row["self_in_topk"] for row in rows], bool)
    report = {
        "index": str(args.index),
        "data_dir": str(data_dir),
        "split": args.split,
        "n": len(rows),
        "skipped": skipped,
        "top_k": args.top_k,
        "coeff_mse_mean": float(coeff_mse.mean()),
        "shape_mse_mean": float(shape_mse.mean()),
        "self_recall_at_k": float(self_hits.mean()),
        "elapsed_sec": round(time.time() - t0, 3),
        "examples": rows[:25],
    }
    out = Path(args.out or Path(args.index).with_suffix(".eval.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"eval n={len(rows)} skipped={skipped} "
        f"coeff_mse={report['coeff_mse_mean']:.3f} "
        f"shape_mse={report['shape_mse_mean']:.3f} "
        f"self@{args.top_k}={report['self_recall_at_k']:.3f}"
    )
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="stage", required=True)

    build = sub.add_parser("build")
    build.add_argument("--out", type=Path, default=render_retrieval.DEFAULT_INDEX)
    build.add_argument("--sources", default="official,cufp",
                       help="Comma-separated: official,cufp")
    build.add_argument("--official-fg-dir", type=Path)
    build.add_argument("--official-limit", type=int, default=0)
    build.add_argument("--cufp-dir", type=Path, default=DEFAULT_CUFP_DIR)
    build.add_argument("--cufp-split", choices=("train", "val", "all"), default="all")
    build.add_argument("--cufp-limit", type=int, default=0)
    build.add_argument("--seed", type=int, default=0)
    build.add_argument("--size", type=int, default=224)
    build.add_argument("--aa", type=int, default=1)
    build.add_argument("--id-model")
    build.add_argument("--report-every", type=int, default=250)
    build.set_defaults(func=stage_build)

    ev = sub.add_parser("eval")
    ev.add_argument("--index", type=Path, default=render_retrieval.DEFAULT_INDEX)
    ev.add_argument("--cufp-dir", type=Path, default=DEFAULT_CUFP_DIR)
    ev.add_argument("--split", choices=("train", "val", "all"), default="val")
    ev.add_argument("--limit", type=int, default=200)
    ev.add_argument("--seed", type=int, default=1)
    ev.add_argument("--id-model")
    ev.add_argument("--top-k", type=int, default=16)
    ev.add_argument("--geom-weight", type=float, default=0.12)
    ev.add_argument("--temperature", type=float, default=0.06)
    ev.add_argument("--out", type=Path)
    ev.set_defaults(func=stage_eval)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
