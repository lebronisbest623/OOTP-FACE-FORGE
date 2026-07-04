"""Product-style CLI for OOTP FaceForge."""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import importlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .paths import workspace_root


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent.parent
WORKSPACE_ROOT = workspace_root()
WORKSPACE_PHOTOS = WORKSPACE_ROOT / "photos"
WORKSPACE_EXPORTS = WORKSPACE_ROOT / "exports"
WORKSPACE_DEBUG = WORKSPACE_ROOT / "debug"
WORKSPACE_MODELS = WORKSPACE_ROOT / "models"
DEFAULT_OUT = WORKSPACE_ROOT / "runs"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class BatchCancelled(RuntimeError):
    """Raised when a GUI/user cancellation stops a batch build."""

PROFILES: dict[str, list[str]] = {
    "identity": [],
    "clean": [
        "--tex-lam", "7",
        "--detail-strength", "0.55",
        "--eye-detail-strength", "0.15",
    ],
    "mouth-soft": [
        "--tex-lam", "10",
        "--detail-strength", "0.4",
        "--eye-detail-strength", "0.12",
    ],
    "strict-front": [
        "--max-yaw", "0.04",
    ],
}

INTERNAL_MODULES = {
    "ootp_faceforge.pipeline",
    "ootp_faceforge.render",
}


def slugify(value: str) -> str:
    value = value.strip().replace("\\", "/").split("/")[-1]
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return value.lower() or "player"


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def run_command(args: list[str], cwd: Path) -> dict[str, Any]:
    if len(args) >= 3 and args[0] == sys.executable and args[1] == "-m":
        module_name = args[2]
        if module_name in INTERNAL_MODULES:
            stdout = io.StringIO()
            stderr = io.StringIO()
            old_cwd = Path.cwd()
            try:
                os.chdir(cwd)
                module = importlib.import_module(module_name)
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    module.main(args[3:])
                return {
                    "args": args,
                    "returncode": 0,
                    "stdout": stdout.getvalue(),
                    "stderr": stderr.getvalue(),
                }
            except SystemExit as exc:
                code = exc.code if isinstance(exc.code, int) else 1
                if exc.code not in (None, 0) and not isinstance(exc.code, int):
                    print(exc.code, file=stderr)
                return {
                    "args": args,
                    "returncode": code,
                    "stdout": stdout.getvalue(),
                    "stderr": stderr.getvalue(),
                }
            except Exception as exc:
                print(f"{type(exc).__name__}: {exc}", file=stderr)
                return {
                    "args": args,
                    "returncode": 1,
                    "stdout": stdout.getvalue(),
                    "stderr": stderr.getvalue(),
                }
            finally:
                os.chdir(old_cwd)

    env = os.environ.copy()
    src_path = str(PROJECT_ROOT / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not existing else src_path + os.pathsep + existing
    proc = subprocess.run(
        args,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return {
        "args": args,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def require_ok(result: dict[str, Any], label: str) -> None:
    if result["returncode"] == 0:
        return
    sys.stderr.write(result["stdout"])
    sys.stderr.write(result["stderr"])
    raise SystemExit(f"{label} failed with exit code {result['returncode']}")


def add_optional_arg(cmd: list[str], name: str, value: Any) -> None:
    if value is None:
        return
    cmd.extend([f"--{name.replace('_', '-')}", str(value)])


def parse_pipeline_summary(stdout: str) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "photos": [],
    }
    for line in stdout.splitlines():
        if line.startswith("photo "):
            summary["photos"].append(line)
        elif line.startswith("skip "):
            summary.setdefault("skipped", []).append(line)
        elif line.startswith("texture photo:"):
            summary["texture_photo"] = line.partition(":")[2].strip()
        elif line.startswith("texture mode:"):
            summary["texture_mode"] = line.partition(":")[2].strip()
        elif line.startswith("texture fusion:"):
            summary["texture_fusion"] = line
        elif line.startswith("identity:"):
            summary["identity"] = line
        elif line.startswith("multi shape:"):
            summary["shape"] = line
        elif line.startswith("exposure gain:"):
            summary["exposure_gain"] = line.partition(":")[2].strip()
        elif line.startswith("tex fit:"):
            summary["texture_fit"] = line
        elif line.startswith("detail fusion:"):
            summary["detail_fusion"] = line
        elif line.startswith("detail coverage:"):
            summary["detail_coverage"] = line.partition(":")[2].strip()
    return summary


def _has_images(path: Path) -> bool:
    return path.is_dir() and any(
        p.is_file() and p.suffix.lower() in IMAGE_EXTS
        for p in path.iterdir()
    )


def _dedupe_batch_slugs(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, int] = {}
    out: list[dict[str, Any]] = []
    for item in items:
        merged = dict(item)
        base = slugify(str(merged.get("slug") or merged.get("name") or "player"))
        seen[base] = seen.get(base, 0) + 1
        merged["slug"] = base if seen[base] == 1 else f"{base}_{seen[base]}"
        out.append(merged)
    return out


def expand_path_batch_items(inputs: list[str | Path]) -> list[dict[str, Any]]:
    """Turn selected image files or folders into automatic batch build items."""
    items: list[dict[str, Any]] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            items.append({"name": path.stem, "photos": str(path)})
            continue
        if not path.is_dir():
            continue
        if _has_images(path):
            items.append({"name": path.name, "photos": str(path)})
            continue
        children = sorted(p for p in path.iterdir() if p.is_dir() and _has_images(p))
        items.extend({"name": child.name, "photos": str(child)} for child in children)
    if not items:
        raise ValueError("no image files or photo folders found")
    return _dedupe_batch_slugs(items)


def load_batch_items(inputs: list[str | Path]) -> list[dict[str, Any]]:
    if len(inputs) == 1 and Path(inputs[0]).suffix.lower() == ".json":
        data = json.loads(Path(inputs[0]).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("batch file must be a JSON list")
        return data
    return expand_path_batch_items(inputs)


def build_pipeline_args(args: argparse.Namespace, fg_path: Path) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "ootp_faceforge.pipeline",
        str(Path(args.photos)),
        str(fg_path),
    ]
    cmd.extend(PROFILES[args.profile])
    for option in (
        "texture_photo",
        "texture_mode",
        "max_yaw",
        "shape_lam",
        "asym_lam",
        "dense_weight",
        "tex_lam",
        "tex_erode",
        "exposure_lo",
        "exposure_hi",
        "detail_size",
        "detail_strength",
        "detail_chroma_strength",
        "detail_edge_strength",
        "detail_flat_neutralize",
        "detail_shadow_neutralize",
        "detail_jpeg_quality",
        "eye_detail_strength",
        "detail_min_cos",
        "restore",
        "restore_model",
        "id_refine",
        "id_model",
        "refine_size",
        "refine_r_max",
        "debug_dir",
    ):
        add_optional_arg(cmd, option, getattr(args, option, None))
    return cmd


def build_player(args: argparse.Namespace) -> dict[str, Any]:
    photos = Path(args.photos)
    name = getattr(args, "name", None) or photos.name
    slug = slugify(getattr(args, "slug", None) or name)
    player_dir = Path(args.out_dir) / slug
    fg_dir = player_dir / "facegen"
    preview_dir = player_dir / "preview"
    meta_dir = player_dir / "meta"
    log_dir = player_dir / "logs"
    for directory in (fg_dir, preview_dir, meta_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)

    fg_path = fg_dir / f"{slug}.fg"
    preview_path = preview_dir / f"{slug}_ootp.png"
    manifest_path = meta_dir / f"{slug}.manifest.json"
    appearance_path = meta_dir / f"{slug}.appearance.json"
    log_path = log_dir / "build.log"

    pipeline_cmd = build_pipeline_args(args, fg_path)
    pipeline = run_command(pipeline_cmd, PROJECT_ROOT)
    require_ok(pipeline, "pipeline")

    render_cmd = [
        sys.executable,
        "-m",
        "ootp_faceforge.render",
        str(fg_path),
        str(preview_path),
        "--size",
        str(args.size),
        "--aa",
        str(args.aa),
    ]
    render = run_command(render_cmd, PROJECT_ROOT)
    require_ok(render, "render")

    log_path.write_text(
        "\n".join([
            "$ " + " ".join(pipeline_cmd),
            pipeline["stdout"],
            pipeline["stderr"],
            "$ " + " ".join(render_cmd),
            render["stdout"],
            render["stderr"],
        ]),
        encoding="utf-8",
    )

    appearance = {
        "schema_version": 1,
        "player": name,
        "slug": slug,
        "hair": None,
        "hair_color": None,
        "facial_hair": None,
        "cap": None,
        "notes": "OOTP appearance is separate from the .fg face file.",
    }
    if not appearance_path.exists() or getattr(args, "overwrite_meta", False):
        appearance_path.write_text(
            json.dumps(appearance, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    manifest = {
        "schema_version": 1,
        "product": "OOTP FaceForge",
        "generated_at": now_iso(),
        "player": {
            "name": name,
            "slug": slug,
        },
        "inputs": {
            "photos": str(photos),
            "profile": args.profile,
            "texture_mode": getattr(args, "texture_mode", None) or "fuse",
            "texture_photo": getattr(args, "texture_photo", None),
        },
        "outputs": {
            "fg": str(fg_path),
            "preview": str(preview_path),
            "appearance": str(appearance_path),
            "log": str(log_path),
        },
        "diagnostics": parse_pipeline_summary(pipeline["stdout"]),
        "commands": {
            "pipeline": pipeline_cmd,
            "render": render_cmd,
        },
    }
    if getattr(args, "flat_copy", False):
        flat_fg = Path(args.out_dir) / f"{slug}.fg"
        flat_preview = Path(args.out_dir) / f"{slug}.png"
        shutil.copy2(fg_path, flat_fg)
        shutil.copy2(preview_path, flat_preview)
        manifest["outputs"]["flat_fg"] = str(flat_fg)
        manifest["outputs"]["flat_preview"] = str(flat_preview)

    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"built {name}")
    print(f"  fg:       {fg_path}")
    print(f"  preview:  {preview_path}")
    print(f"  manifest: {manifest_path}")
    return manifest


def render_only(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "-m",
        "ootp_faceforge.render",
        str(Path(args.fg)),
        str(Path(args.out)),
        "--size",
        str(args.size),
        "--aa",
        str(args.aa),
    ]
    if args.no_eyes:
        cmd.append("--no-eyes")
    if args.flat:
        cmd.append("--flat")
    result = run_command(cmd, PROJECT_ROOT)
    require_ok(result, "render")
    sys.stdout.write(result["stdout"])
    sys.stderr.write(result["stderr"])


def default_jobs() -> int:
    """Worker count for parallel batch: most cores, leaving headroom."""
    n = os.cpu_count() or 4
    return max(1, min(n - 1, 12))


def _batch_failure_message(exc: BaseException, output: str) -> str:
    message = str(exc) or type(exc).__name__
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return message
    detail = "\n".join(lines[-12:])
    if len(detail) > 1800:
        detail = "..." + detail[-1797:]
    if message in detail:
        return detail
    return f"{message}\n\nDetails:\n{detail}"


def _ensure_worker_path() -> None:
    src = str(PROJECT_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def _batch_build_one(
    ns: argparse.Namespace,
) -> tuple[str, str | None, dict[str, Any] | None]:
    """Build one player in a worker process; capture output, return status."""
    _ensure_worker_path()
    name = getattr(ns, "name", None) or Path(str(ns.photos)).name
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            manifest = build_player(ns)
        return name, None, manifest
    except SystemExit as exc:
        return name, _batch_failure_message(exc, buf.getvalue()), None
    except Exception as exc:  # noqa: BLE001 - report, do not abort the batch
        return name, _batch_failure_message(exc, buf.getvalue()), None


def _cancel_requested(cancel_event) -> bool:
    return bool(cancel_event is not None and cancel_event.is_set())


def _terminate_process_pool(executor) -> None:
    processes = getattr(executor, "_processes", None) or {}
    for proc in list(processes.values()):
        if proc is not None and proc.is_alive():
            with contextlib.suppress(Exception):
                proc.terminate()
    deadline = time.monotonic() + 2.0
    for proc in list(processes.values()):
        if proc is None:
            continue
        remaining = max(0.0, deadline - time.monotonic())
        with contextlib.suppress(Exception):
            proc.join(remaining)
        if proc.is_alive():
            with contextlib.suppress(Exception):
                proc.kill()


def batch_namespaces(args: argparse.Namespace,
                     items: list[dict[str, Any]]) -> list[argparse.Namespace]:
    out: list[argparse.Namespace] = []
    for item in items:
        if not isinstance(item, dict):
            raise SystemExit("each batch item must be an object")
        merged = argparse.Namespace(**vars(args))
        for key, value in item.items():
            setattr(merged, key.replace("-", "_"), value)
        if not getattr(merged, "photos", None):
            raise SystemExit("batch item missing photos")
        if not getattr(merged, "profile", None):
            merged.profile = "identity"
        out.append(merged)
    return out


def run_batch(namespaces: list[argparse.Namespace], jobs: int | None = None,
              progress=None, completed=None, cancel_event=None,
              force_processes: bool = False) -> list[tuple[str, str]]:
    """Build many players, in parallel processes when it pays off.

    progress(done, total, name, error|None) is called as each finishes.
    completed(done, total, name, error|None, manifest|None) receives the full result.
    Returns the list of (name, error) failures."""
    total = len(namespaces)
    jobs = default_jobs() if jobs is None else max(1, int(jobs))
    failures: list[tuple[str, str]] = []

    def report(done: int, name: str, err: str | None,
               manifest: dict[str, Any] | None) -> None:
        if err:
            failures.append((name, err))
        if progress:
            progress(done, total, name, err)
        if completed:
            completed(done, total, name, err, manifest)

    # Worker spawn re-imports the package + heavy deps (mediapipe/cv2), so
    # small batches are faster sequentially.
    if not force_processes and (jobs <= 1 or total <= 2):
        for i, ns in enumerate(namespaces, 1):
            if _cancel_requested(cancel_event):
                raise BatchCancelled("cancelled")
            name, err, manifest = _batch_build_one(ns)
            report(i, name, err, manifest)
        return failures

    # Spawned workers rebuild sys.path from the environment, so make sure the
    # package is importable there even under the local `python ootp_facegen.py`.
    src = str(PROJECT_ROOT / "src")
    existing = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = src if not existing else src + os.pathsep + existing

    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

    done = 0
    executor = ProcessPoolExecutor(max_workers=min(max(jobs, 1), total))
    futures = set()
    did_shutdown = False
    try:
        for ns in namespaces:
            if _cancel_requested(cancel_event):
                raise BatchCancelled("cancelled")
            futures.add(executor.submit(_batch_build_one, ns))
        while futures:
            if _cancel_requested(cancel_event):
                raise BatchCancelled("cancelled")
            finished, futures = wait(
                futures,
                timeout=0.2,
                return_when=FIRST_COMPLETED,
            )
            if not finished:
                continue
            for fut in finished:
                name, err, manifest = fut.result()
                done += 1
                report(done, name, err, manifest)
    except BatchCancelled:
        for fut in futures:
            fut.cancel()
        _terminate_process_pool(executor)
        executor.shutdown(wait=False, cancel_futures=True)
        did_shutdown = True
        raise
    finally:
        if not did_shutdown:
            executor.shutdown(wait=True, cancel_futures=True)
    return failures


def batch(args: argparse.Namespace) -> None:
    try:
        items = load_batch_items(args.inputs)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    namespaces = batch_namespaces(args, items)
    jobs = getattr(args, "jobs", None) or default_jobs()
    total = len(namespaces)
    print(f"batch: {total} players, {min(jobs, total)} parallel workers")

    def report(done: int, total: int, name: str, err: str | None) -> None:
        if err:
            print(f"[{done}/{total}] skipped {name}: {err}", file=sys.stderr)
        else:
            print(f"[{done}/{total}] built {name}")

    failures = run_batch(namespaces, jobs=jobs, progress=report)
    if failures:
        raise SystemExit(f"batch completed with {len(failures)} skipped")


def launch_gui(args: argparse.Namespace) -> None:
    from .gui import main as gui_main

    gui_main()


def add_build_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("photos", help="Photo folder or single image.")
    p.add_argument("--name", help="Display name for manifests.")
    p.add_argument("--slug", help="Output slug. Defaults to the name/photo folder.")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--profile", choices=sorted(PROFILES), default="identity")
    p.add_argument("--texture-photo",
                   help="Substring/path of the texture/detail photo to force.")
    p.add_argument("--texture-mode", choices=("fuse", "best"), default="fuse",
                   help="Fuse all usable photos or use the single best texture photo.")
    p.add_argument("--max-yaw", type=float)
    p.add_argument("--shape-lam", type=float)
    p.add_argument("--asym-lam", type=float)
    p.add_argument("--dense-weight", type=float)
    p.add_argument("--tex-lam", type=float)
    p.add_argument("--tex-erode", type=int)
    p.add_argument("--exposure-lo", type=float)
    p.add_argument("--exposure-hi", type=float)
    p.add_argument("--detail-size", type=int)
    p.add_argument("--detail-strength", type=float)
    p.add_argument("--detail-chroma-strength", type=float)
    p.add_argument("--detail-edge-strength", type=float)
    p.add_argument("--detail-flat-neutralize", type=float)
    p.add_argument("--detail-shadow-neutralize", type=float)
    p.add_argument("--detail-jpeg-quality", type=int)
    p.add_argument("--eye-detail-strength", type=float)
    p.add_argument("--detail-min-cos", type=float)
    p.add_argument("--restore", choices=("auto", "off", "force"))
    p.add_argument("--restore-model")
    p.add_argument("--id-refine", type=int,
                   help="0 disables; positive values enable slow experimental identity search.")
    p.add_argument("--id-model")
    p.add_argument("--refine-size", type=int)
    p.add_argument("--refine-r-max", type=float)
    p.add_argument("--debug-dir",
                   help="Optional directory for intermediate texture/debug images.")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--aa", type=int, default=2)
    p.add_argument("--flat-copy", action="store_true",
                   help="Also copy .fg and preview directly under --out-dir.")
    p.add_argument("--overwrite-meta", action="store_true",
                   help="Rewrite existing appearance sidecar.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ootp-faceforge",
        description="Build OOTP FaceGen .fg files from player photo folders.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Build one player package.")
    add_build_options(build)
    build.set_defaults(func=build_player)

    render = sub.add_parser("render", help="Render an existing .fg preview.")
    render.add_argument("fg")
    render.add_argument("out")
    render.add_argument("--size", type=int, default=512)
    render.add_argument("--aa", type=int, default=2)
    render.add_argument("--no-eyes", action="store_true")
    render.add_argument("--flat", action="store_true")
    render.set_defaults(func=render_only)

    batch_cmd = sub.add_parser(
        "batch",
        help="Build selected image files, photo folders, parent folders, or a JSON list.",
    )
    batch_cmd.add_argument("inputs", nargs="+")
    batch_cmd.add_argument("-j", "--jobs", type=int, default=None,
                           help="Parallel worker processes (default: most cores).")
    batch_cmd.add_argument("--out-dir", default=str(DEFAULT_OUT))
    batch_cmd.add_argument("--size", type=int, default=512)
    batch_cmd.add_argument("--aa", type=int, default=2)
    batch_cmd.add_argument("--texture-mode", choices=("fuse", "best"), default="fuse")
    batch_cmd.add_argument("--flat-copy", action="store_true")
    batch_cmd.add_argument("--overwrite-meta", action="store_true")
    batch_cmd.set_defaults(func=batch)

    gui = sub.add_parser("gui", help="Open the desktop GUI.")
    gui.set_defaults(func=launch_gui)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        args.func(args)
    finally:
        from .landmarks import close_landmarker

        close_landmarker()


if __name__ == "__main__":
    main()
