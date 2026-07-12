#!/usr/bin/env python
"""Install a flat FaceGen package with backup, atomic writes, and SHA-256 proof.

The command is a dry run unless ``--apply`` is passed.  Existing destination
files that would change are copied to a timestamped backup before any target is
replaced.  Every installed file is verified against the package hash.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BACKUP_ROOT = PROJECT_ROOT / "install_backups"
DEFAULT_LOG_ROOT = PROJECT_ROOT / "install_logs"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def safe_label(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._")
    return cleaned or "ootp_fg_install"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_flat_fg_files(root: Path, *, allow_empty: bool = False) -> dict[str, Path]:
    if not root.is_dir():
        raise SystemExit(f"fg directory does not exist: {root}")
    files: dict[str, Path] = {}
    for path in sorted(root.glob("*.fg"), key=lambda item: item.name.lower()):
        key = path.name.lower()
        if key in files:
            raise SystemExit(f"case-insensitive duplicate fg filename: {files[key]}, {path}")
        files[key] = path
    if not files and not allow_empty:
        raise SystemExit(f"no .fg files found: {root}")
    return files


def ootp27_running() -> bool:
    if os.name != "nt":
        return False
    result = subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq ootp27.exe", "/FO", "CSV", "/NH"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return "ootp27.exe" in result.stdout.lower()


def atomic_copy(source: Path, target: Path, expected_sha256: str) -> None:
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copy2(source, temporary)
        copied_sha256 = sha256(temporary)
        if copied_sha256 != expected_sha256:
            raise RuntimeError(
                f"temporary copy hash mismatch: {source} -> {temporary}: "
                f"{expected_sha256} != {copied_sha256}"
            )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "destination_index",
        "destination",
        "action",
        "package_fg",
        "target_fg",
        "backup_fg",
        "package_bytes",
        "previous_bytes",
        "package_sha256",
        "previous_sha256",
        "post_sha256",
        "verified",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_plan(package_dir: Path, destinations: list[Path]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    package_files = collect_flat_fg_files(package_dir)
    package_hashes = {key: sha256(path) for key, path in package_files.items()}
    rows: list[dict[str, Any]] = []

    for destination_index, destination in enumerate(destinations, 1):
        if not destination.is_dir():
            raise SystemExit(f"destination fg directory does not exist: {destination}")
        existing = collect_flat_fg_files(destination, allow_empty=True)
        for key, package_fg in package_files.items():
            current = existing.get(key)
            target_fg = current if current else destination / package_fg.name
            previous_sha256 = sha256(current) if current else ""
            package_sha256 = package_hashes[key]
            action = "skip_identical" if current and previous_sha256 == package_sha256 else "replace" if current else "add"
            rows.append(
                {
                    "destination_index": destination_index,
                    "destination": str(destination),
                    "action": action,
                    "package_fg": str(package_fg),
                    "target_fg": str(target_fg),
                    "backup_fg": "",
                    "package_bytes": package_fg.stat().st_size,
                    "previous_bytes": current.stat().st_size if current else "",
                    "package_sha256": package_sha256,
                    "previous_sha256": previous_sha256,
                    "post_sha256": previous_sha256 if action == "skip_identical" else "",
                    "verified": action == "skip_identical",
                }
            )
    return rows, package_hashes


def summarize_destinations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    destination_indexes = sorted({int(row["destination_index"]) for row in rows})
    for destination_index in destination_indexes:
        grouped = [row for row in rows if int(row["destination_index"]) == destination_index]
        counts = Counter(str(row["action"]) for row in grouped)
        summaries.append(
            {
                "destination_index": destination_index,
                "destination": grouped[0]["destination"],
                "package_targets": len(grouped),
                "action_counts": dict(counts),
                "backup_bytes": sum(int(row["previous_bytes"] or 0) for row in grouped if row["action"] == "replace"),
                "added_bytes": sum(int(row["package_bytes"]) for row in grouped if row["action"] == "add"),
            }
        )
    return summaries


def validate_unchanged_since_plan(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        target = Path(row["target_fg"])
        if row["action"] == "add":
            if target.exists():
                raise RuntimeError(f"target appeared after planning: {target}")
            continue
        if not target.exists():
            raise RuntimeError(f"target disappeared after planning: {target}")
        current_sha256 = sha256(target)
        if current_sha256 != row["previous_sha256"]:
            raise RuntimeError(f"target changed after planning: {target}")


def apply_plan(rows: list[dict[str, Any]], backup_session: Path) -> None:
    changed = [row for row in rows if row["action"] in {"replace", "add"}]
    validate_unchanged_since_plan(changed)

    for row in changed:
        if row["action"] != "replace":
            continue
        destination_index = int(row["destination_index"])
        target = Path(row["target_fg"])
        backup_dir = backup_session / f"destination_{destination_index:02d}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup = backup_dir / target.name
        atomic_copy(target, backup, row["previous_sha256"])
        row["backup_fg"] = str(backup)

    installed: list[dict[str, Any]] = []
    try:
        for row in changed:
            package_fg = Path(row["package_fg"])
            target = Path(row["target_fg"])
            atomic_copy(package_fg, target, row["package_sha256"])
            installed.append(row)
            row["post_sha256"] = sha256(target)
            row["verified"] = row["post_sha256"] == row["package_sha256"]
            if not row["verified"]:
                raise RuntimeError(f"installed target hash mismatch: {target}")
    except Exception:
        rollback_errors: list[str] = []
        for row in reversed(installed):
            target = Path(row["target_fg"])
            try:
                if row["action"] == "replace":
                    atomic_copy(Path(row["backup_fg"]), target, row["previous_sha256"])
                elif row["action"] == "add":
                    target.unlink(missing_ok=True)
            except Exception as rollback_error:  # pragma: no cover - last-resort reporting
                rollback_errors.append(f"{target}: {rollback_error}")
        if rollback_errors:
            raise RuntimeError("install failed and rollback had errors: " + "; ".join(rollback_errors))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True, help="Flat lookup-ready .fg package.")
    parser.add_argument("--destination", type=Path, action="append", required=True, help="OOTP fg_files directory; repeat for mirrors.")
    parser.add_argument("--backup-root", type=Path, default=DEFAULT_BACKUP_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--label", default="", help="Label used for timestamped backup and log folders.")
    parser.add_argument("--apply", action="store_true", help="Apply the plan. Without this flag the command is read-only.")
    parser.add_argument("--allow-ootp-running", action="store_true", help="Allow writes while ootp27.exe is running.")
    args = parser.parse_args()

    package_dir = args.package_dir.resolve()
    destinations = [path.resolve() for path in args.destination]
    if len({str(path).lower() for path in destinations}) != len(destinations):
        raise SystemExit("duplicate destination directories")

    running = ootp27_running()
    rows, _ = build_plan(package_dir, destinations)
    destination_summaries = summarize_destinations(rows)
    label = safe_label(args.label or package_dir.name)
    run_stamp = timestamp()
    backup_session = args.backup_root.resolve() / f"{label}_before_{run_stamp}"
    log_session = args.log_root.resolve() / f"{label}_{run_stamp}"

    summary: dict[str, Any] = {
        "created_at": now_iso(),
        "mode": "apply" if args.apply else "dry_run",
        "status": "planned",
        "ootp27_running": running,
        "package_dir": str(package_dir),
        "package_file_count": len({row["package_fg"] for row in rows}),
        "destinations": destination_summaries,
        "backup_session": str(backup_session) if args.apply else "",
        "log_session": str(log_session) if args.apply else "",
    }

    if not args.apply:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    if running and not args.allow_ootp_running:
        raise SystemExit("ootp27.exe is running; close OOTP before applying, or pass --allow-ootp-running")
    if backup_session.exists() or log_session.exists():
        raise SystemExit(f"timestamped output already exists: {backup_session} or {log_session}")

    backup_session.mkdir(parents=True)
    log_session.mkdir(parents=True)
    try:
        apply_plan(rows, backup_session)
        summary["status"] = "installed"
        summary["completed_at"] = now_iso()
        summary["verified_file_count"] = sum(bool(row["verified"]) for row in rows)
    except Exception as error:
        summary["status"] = "failed"
        summary["failed_at"] = now_iso()
        summary["error"] = str(error)
        write_csv(log_session / "install_manifest.csv", rows)
        write_json(log_session / "summary.json", summary)
        raise

    write_csv(log_session / "install_manifest.csv", rows)
    write_json(log_session / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130)
