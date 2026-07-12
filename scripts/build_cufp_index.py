"""Build/evaluate a CUFP nearest-neighbour FaceGen prior index.

This intentionally does not train a photo->coeff regressor. It stores robust
identity/geometry features for each CUFP photo/FG pair so runtime builds can
retrieve and blend real FaceGen coefficient prototypes.

Usage:
  python scripts/build_cufp_index.py build --limit 500 --out models/cufp_identity_index_pilot.npz
  python scripts/build_cufp_index.py eval --index models/cufp_identity_index_pilot.npz --limit 100
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ootp_faceforge import photofit, retrieval  # noqa: E402
from ootp_faceforge.landmarks import detect  # noqa: E402


DEFAULT_DATA_DIR = Path(r"C:\Users\user\facegen\data\cufp_photo_dataset")
N_SYM, N_ASYM, N_TEX = 50, 30, 50


def _load_labels(data_dir: Path):
    labels_path = data_dir / "labels.npz"
    if not labels_path.is_file():
        raise SystemExit(f"missing labels: {labels_path}")
    data = np.load(labels_path, allow_pickle=True)
    required = {"names", "coeffs", "is_val", "photo_paths"}
    missing = required - set(data.files)
    if missing:
        raise SystemExit(f"labels missing keys: {sorted(missing)}")
    return data


def _image_path(data_dir: Path, photo_path: str, name: str) -> Path:
    filename = Path(str(photo_path).replace("\\", "/")).name
    candidates = [
        data_dir / "images" / filename,
        data_dir / "images" / f"{name}.jpg",
        data_dir / str(photo_path),
    ]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def _row_indices(is_val: np.ndarray, split: str, limit: int, seed: int) -> np.ndarray:
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
    return idx


def _feature_for_image(path: Path, id_model: str | None):
    img = np.asarray(Image.open(path).convert("RGB"))
    lms = detect(img)
    return photofit.feature_from_image(img, lms, id_model)


def _unit_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-9)


def stage_build(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    labels = _load_labels(data_dir)
    names = labels["names"].astype(str)
    coeffs = labels["coeffs"].astype(np.float32)
    is_val = labels["is_val"].astype(bool)
    photo_paths = labels["photo_paths"].astype(str)
    idx = _row_indices(is_val, args.split, args.limit, args.seed)

    out_names = []
    out_coeffs = []
    out_emb = []
    out_geom = []
    out_photo_paths = []
    skipped: list[dict[str, str]] = []
    t0 = time.time()
    for n_done, row_idx in enumerate(idx, 1):
        name = str(names[row_idx])
        path = _image_path(data_dir, photo_paths[row_idx], name)
        try:
            feat = _feature_for_image(path, args.id_model)
            if feat is None:
                raise RuntimeError("embedding failed")
        except Exception as exc:  # noqa: BLE001 - index build should continue
            skipped.append({
                "name": name,
                "photo": str(path),
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue
        out_names.append(name)
        out_coeffs.append(coeffs[row_idx])
        out_emb.append(feat.embedding)
        out_geom.append(feat.feature[512:])
        out_photo_paths.append(str(path))
        if args.report_every and n_done % args.report_every == 0:
            print(
                f"{n_done}/{len(idx)} indexed={len(out_names)} "
                f"skipped={len(skipped)} elapsed={time.time() - t0:.1f}s",
                flush=True,
            )

    if not out_names:
        raise SystemExit("no CUFP index rows generated")

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
        schema="cufp_identity_index_v1",
        names=np.asarray(out_names),
        coeffs=np.stack(out_coeffs).astype(np.float32),
        embeddings=emb,
        geom=geom,
        geom_mean=geom_mean,
        geom_scale=geom_scale,
        photo_paths=np.asarray(out_photo_paths),
        n_sym=N_SYM,
        n_asym=N_ASYM,
        n_tex=N_TEX,
        source_data_dir=str(data_dir),
        split=args.split,
        feature_lms=np.asarray(photofit.FEATURE_LMS, np.int32),
        feature_names=np.asarray(photofit.feature_names()),
    )
    report = {
        "out": str(out),
        "source_data_dir": str(data_dir),
        "split": args.split,
        "requested": int(len(idx)),
        "indexed": int(len(out_names)),
        "skipped": int(len(skipped)),
        "elapsed_sec": round(time.time() - t0, 3),
        "skip_examples": skipped[:20],
    }
    report_path = out.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}: indexed={len(out_names)} skipped={len(skipped)}")
    print(f"wrote {report_path}")


def stage_eval(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    labels = _load_labels(data_dir)
    names = labels["names"].astype(str)
    coeffs = labels["coeffs"].astype(np.float32)
    is_val = labels["is_val"].astype(bool)
    photo_paths = labels["photo_paths"].astype(str)
    idx = _row_indices(is_val, args.split, args.limit, args.seed)
    index = retrieval.CufpIndex(args.index)

    rows = []
    skipped = 0
    for row_idx in idx:
        name = str(names[row_idx])
        path = _image_path(data_dir, photo_paths[row_idx], name)
        try:
            feat = _feature_for_image(path, args.id_model)
            if feat is None:
                raise RuntimeError("embedding failed")
            pred = index.predict_feature(
                feat.feature,
                top_k=args.top_k,
                geom_weight=args.geom_weight,
                temperature=args.temperature,
                exclude_names={name} if args.exclude_self else None,
                render_index_path=args.render_index,
                render_top_n=args.render_top_n,
                render_weight=args.render_weight,
                render_geom_weight=args.render_geom_weight,
                render_min_matches=args.render_min_matches,
            )
        except Exception:
            skipped += 1
            continue
        err = pred.raw - coeffs[row_idx]
        top = pred.hits[0]
        rows.append({
            "name": name,
            "top1": top.name,
            "top1_score": top.score,
            "top1_emb": top.emb_score,
            "top1_geom": top.geom_score,
            "top1_render_score": top.render_score,
            "render_reranked": pred.reranked,
            "render_matches": pred.render_matches,
            "coeff_mse": float(np.mean(err ** 2)),
            "shape_mse": float(np.mean(err[:N_SYM + N_ASYM] ** 2)),
            "self_in_topk": any(hit.name == name for hit in pred.hits),
        })

    if not rows:
        raise SystemExit("no eval rows")
    coeff_mse = np.asarray([row["coeff_mse"] for row in rows])
    shape_mse = np.asarray([row["shape_mse"] for row in rows])
    self_hits = np.asarray([row["self_in_topk"] for row in rows], bool)
    render_reranked = np.asarray([row["render_reranked"] for row in rows], bool)
    report = {
        "index": str(args.index),
        "data_dir": str(data_dir),
        "split": args.split,
        "n": len(rows),
        "skipped": skipped,
        "top_k": args.top_k,
        "exclude_self": bool(args.exclude_self),
        "coeff_mse_mean": float(coeff_mse.mean()),
        "shape_mse_mean": float(shape_mse.mean()),
        "self_recall_at_k": float(self_hits.mean()),
        "render_rerank_rate": float(render_reranked.mean()),
        "examples": rows[:25],
    }
    out = Path(args.out or Path(args.index).with_suffix(".eval.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"eval n={len(rows)} skipped={skipped} "
        f"coeff_mse={report['coeff_mse_mean']:.3f} "
        f"shape_mse={report['shape_mse_mean']:.3f} "
        f"self@{args.top_k}={report['self_recall_at_k']:.3f} "
        f"render_rerank={report['render_rerank_rate']:.3f}"
    )
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="stage", required=True)

    build = sub.add_parser("build")
    build.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    build.add_argument("--out", type=Path, default=retrieval.DEFAULT_INDEX)
    build.add_argument("--split", choices=("train", "val", "all"), default="train")
    build.add_argument("--limit", type=int, default=0)
    build.add_argument("--seed", type=int, default=0)
    build.add_argument("--id-model")
    build.add_argument("--report-every", type=int, default=250)
    build.set_defaults(func=stage_build)

    ev = sub.add_parser("eval")
    ev.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ev.add_argument("--index", type=Path, default=retrieval.DEFAULT_INDEX)
    ev.add_argument("--split", choices=("train", "val", "all"), default="val")
    ev.add_argument("--limit", type=int, default=200)
    ev.add_argument("--seed", type=int, default=1)
    ev.add_argument("--id-model")
    ev.add_argument("--top-k", type=int, default=12)
    ev.add_argument("--geom-weight", type=float, default=0.18)
    ev.add_argument("--temperature", type=float, default=0.055)
    ev.add_argument("--render-index", type=Path)
    ev.add_argument("--render-top-n", type=int, default=64)
    ev.add_argument("--render-weight", type=float, default=0.05)
    ev.add_argument("--render-geom-weight", type=float, default=0.12)
    ev.add_argument("--render-min-matches", type=int, default=3)
    ev.add_argument("--exclude-self", action="store_true")
    ev.add_argument("--out", type=Path)
    ev.set_defaults(func=stage_eval)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
