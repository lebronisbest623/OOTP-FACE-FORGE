"""Download the optional ONNX models used by the pipeline.

- GFPGAN v1.4 (--restore stage; Apache-2.0, TencentARC, ONNX conversion
  hosted by the facefusion project).
- ArcFace w600k_r50 (--id-refine stage; InsightFace model-zoo weights,
  distributed for non-commercial research use).

Neither model is bundled with this repository.

Usage:
  python scripts/download_restore_model.py [--models-dir models]
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/facefusion/models-3.0.0/resolve/main/"
MODELS = {
    "GFPGANv1.4.onnx": BASE + "gfpgan_1.4.onnx",
    "arcface_w600k_r50.onnx": BASE + "arcface_w600k_r50.onnx",
}
DEFAULT_DIR = Path(__file__).resolve().parents[1] / "models"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=str(DEFAULT_DIR))
    args = ap.parse_args()
    out_dir = Path(args.models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, url in MODELS.items():
        out = out_dir / name
        if out.exists():
            print(f"already present: {out} ({out.stat().st_size} bytes)")
            continue
        print(f"downloading {url}\n -> {out}")
        urllib.request.urlretrieve(url, out)
        print(f"done ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
