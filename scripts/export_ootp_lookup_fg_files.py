#!/usr/bin/env python
"""Build an OOTP lookup-ready flat .fg folder from generated FaceForge output.

OOTP prefers the historical minor id (bbrefminors_id) when resolving FaceGen
files, then falls back to the historical id.  Our source folders usually name
files by the stable project id in lahman_id, so this script copies those files
and adds aliases using OOTP's lookup id.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def file_safe_id(value: str | None) -> str:
    return (value or "").strip().replace(":", "_")


def parse_roster_export(path: Path) -> list[dict[str, str]]:
    header: list[str] | None = None
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line.startswith("//id"):
                header = [cell.strip() for cell in line[2:].split(",")]
                continue
            if not line.strip() or line.startswith("//"):
                continue
            if header is None:
                raise SystemExit(f"roster header not found before data row in {path}")
            cells = next(csv.reader([line]))
            if cells and cells[-1] == "eol":
                cells = cells[:-1]
            if len(cells) < len(header):
                cells.extend([""] * (len(header) - len(cells)))
            rows.append(dict(zip(header, cells)))
    if header is None:
        raise SystemExit(f"roster header not found: {path}")
    required = {"id", "LastName", "FirstName", "lahman_id", "bbrefminors_id"}
    missing = sorted(required.difference(header))
    if missing:
        raise SystemExit(f"roster missing required fields {missing}: {path}")
    return rows


def collect_fg_files(source_dirs: list[Path]) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    by_stem: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = defaultdict(list)
    for source_dir in source_dirs:
        if not source_dir.exists():
            raise SystemExit(f"source fg directory does not exist: {source_dir}")
        for path in source_dir.rglob("*.fg"):
            stem = path.stem.lower()
            if stem in by_stem:
                duplicates[stem].append(path)
                continue
            by_stem[stem] = path
    return by_stem, duplicates


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_plan(roster_rows: list[dict[str, str]], fg_by_stem: dict[str, Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    alias_plan: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    lookup_groups: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in roster_rows:
        lahman_id = file_safe_id(row.get("lahman_id"))
        if not lahman_id:
            continue
        bbrefminors_id = file_safe_id(row.get("bbrefminors_id"))
        lookup_id = bbrefminors_id or lahman_id
        lookup_kind = "bbrefminors_id" if bbrefminors_id else "lahman_id_fallback"
        lookup_groups[lookup_id.lower()].append(row)

        source_fg = fg_by_stem.get(lahman_id.lower())
        has_source = source_fg is not None
        validation_rows.append(
            {
                "player_id": row.get("id", ""),
                "last_name": row.get("LastName", ""),
                "first_name": row.get("FirstName", ""),
                "lahman_id": lahman_id,
                "bbrefminors_id": bbrefminors_id,
                "ootp_lookup_id": lookup_id,
                "lookup_kind": lookup_kind,
                "has_lahman_source_fg": has_source,
                "source_fg": str(source_fg or ""),
            }
        )
        if not has_source or lookup_id.lower() == lahman_id.lower():
            continue
        alias_plan.append(
            {
                "player_id": row.get("id", ""),
                "last_name": row.get("LastName", ""),
                "first_name": row.get("FirstName", ""),
                "lahman_id": lahman_id,
                "bbrefminors_id": bbrefminors_id,
                "ootp_lookup_id": lookup_id,
                "lookup_kind": lookup_kind,
                "source_fg": str(source_fg),
                "alias_fg_name": f"{lookup_id}.fg",
            }
        )

    duplicate_lookup_rows: list[dict[str, Any]] = []
    for lookup_id, grouped in sorted(lookup_groups.items()):
        if len(grouped) <= 1:
            continue
        duplicate_lookup_rows.append(
            {
                "ootp_lookup_id": lookup_id,
                "count": len(grouped),
                "players": json.dumps(
                    [
                        {
                            "player_id": row.get("id", ""),
                            "name": f"{row.get('LastName', '')},{row.get('FirstName', '')}",
                            "lahman_id": file_safe_id(row.get("lahman_id")),
                            "bbrefminors_id": file_safe_id(row.get("bbrefminors_id")),
                        }
                        for row in grouped
                    ],
                    ensure_ascii=False,
                ),
            }
        )
    return alias_plan, validation_rows, duplicate_lookup_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", type=Path, required=True, help="OOTP roster export txt, e.g. import_export/kbo_rosters.txt")
    parser.add_argument("--source-fg-dir", type=Path, action="append", required=True, help="Generated .fg folder; can be passed more than once. Earlier dirs win for duplicate stems.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Destination flat .fg folder.")
    parser.add_argument(
        "--lookup-only",
        action="store_true",
        help="Write exactly one lookup-ready file per source: bbrefminors_id when present, otherwise lahman_id.",
    )
    parser.add_argument("--allow-duplicate-lookup", action="store_true", help="Do not fail when multiple roster rows resolve to the same OOTP lookup id.")
    args = parser.parse_args()

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise SystemExit(f"out-dir already exists and is not empty: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    roster_rows = parse_roster_export(args.roster)
    fg_by_stem, duplicate_source_stems = collect_fg_files(args.source_fg_dir)
    alias_plan, validation_rows, duplicate_lookup_rows = build_plan(roster_rows, fg_by_stem)
    if duplicate_lookup_rows and not args.allow_duplicate_lookup:
        write_csv(args.out_dir / "duplicate_lookup_ids.csv", duplicate_lookup_rows, ["ootp_lookup_id", "count", "players"])
        raise SystemExit(f"duplicate OOTP lookup ids found: {len(duplicate_lookup_rows)}; see duplicate_lookup_ids.csv")

    copy_manifest: list[dict[str, Any]] = []
    if args.lookup_only:
        alias_by_source: dict[str, dict[str, Any]] = {}
        for alias in alias_plan:
            source_stem = Path(alias["source_fg"]).stem.lower()
            previous = alias_by_source.get(source_stem)
            if previous and previous["alias_fg_name"].lower() != alias["alias_fg_name"].lower():
                raise SystemExit(
                    f"source fg maps to multiple lookup ids: {source_stem}: "
                    f"{previous['alias_fg_name']}, {alias['alias_fg_name']}"
                )
            alias_by_source[source_stem] = alias

        lookup_plan: list[tuple[Path, Path, str, dict[str, Any] | None]] = []
        planned_destinations: dict[str, Path] = {}
        for stem, source_fg in sorted(fg_by_stem.items()):
            alias = alias_by_source.get(stem)
            dest_name = alias["alias_fg_name"] if alias else source_fg.name
            dest = args.out_dir / dest_name
            dest_key = dest.name.lower()
            previous_source = planned_destinations.get(dest_key)
            if previous_source and previous_source != source_fg:
                raise SystemExit(
                    f"multiple source fg files map to one lookup filename: "
                    f"{previous_source}, {source_fg} -> {dest.name}"
                )
            planned_destinations[dest_key] = source_fg
            lookup_plan.append((source_fg, dest, "copy_lookup_alias" if alias else "copy_source", alias))

        for source_fg, dest, action, alias in lookup_plan:
            shutil.copy2(source_fg, dest)
            copy_manifest.append(
                {
                    "action": action,
                    "source_fg": str(source_fg),
                    "dest_fg": str(dest),
                    "source_stem": source_fg.stem,
                    "dest_stem": dest.stem,
                    "player_id": alias["player_id"] if alias else "",
                    "lahman_id": alias["lahman_id"] if alias else "",
                    "bbrefminors_id": alias["bbrefminors_id"] if alias else "",
                    "ootp_lookup_id": alias["ootp_lookup_id"] if alias else "",
                }
            )
    else:
        for stem, source_fg in sorted(fg_by_stem.items()):
            dest = args.out_dir / f"{source_fg.stem}.fg"
            shutil.copy2(source_fg, dest)
            copy_manifest.append(
                {
                    "action": "copy_source",
                    "source_fg": str(source_fg),
                    "dest_fg": str(dest),
                    "source_stem": source_fg.stem,
                    "dest_stem": dest.stem,
                    "player_id": "",
                    "lahman_id": "",
                    "bbrefminors_id": "",
                    "ootp_lookup_id": "",
                }
            )

        for alias in alias_plan:
            source_fg = Path(alias["source_fg"])
            dest = args.out_dir / alias["alias_fg_name"]
            action = "copy_alias"
            if dest.exists():
                action = "overwrite_lookup_alias"
            shutil.copy2(source_fg, dest)
            copy_manifest.append(
                {
                    "action": action,
                    "source_fg": str(source_fg),
                    "dest_fg": str(dest),
                    "source_stem": source_fg.stem,
                    "dest_stem": dest.stem,
                    "player_id": alias["player_id"],
                    "lahman_id": alias["lahman_id"],
                    "bbrefminors_id": alias["bbrefminors_id"],
                    "ootp_lookup_id": alias["ootp_lookup_id"],
                }
            )

    output_stems = {path.stem.lower() for path in args.out_dir.glob("*.fg")}
    missing_lookup = []
    for row in validation_rows:
        if row["has_lahman_source_fg"] and row["ootp_lookup_id"].lower() not in output_stems:
            missing_lookup.append(row)
        row["lookup_fg_exists_after_export"] = row["ootp_lookup_id"].lower() in output_stems

    write_csv(args.out_dir / "copy_manifest.csv", copy_manifest, ["action", "source_fg", "dest_fg", "source_stem", "dest_stem", "player_id", "lahman_id", "bbrefminors_id", "ootp_lookup_id"])
    write_csv(args.out_dir / "alias_plan.csv", alias_plan, ["player_id", "last_name", "first_name", "lahman_id", "bbrefminors_id", "ootp_lookup_id", "lookup_kind", "source_fg", "alias_fg_name"])
    write_csv(args.out_dir / "lookup_validation.csv", validation_rows, ["player_id", "last_name", "first_name", "lahman_id", "bbrefminors_id", "ootp_lookup_id", "lookup_kind", "has_lahman_source_fg", "source_fg", "lookup_fg_exists_after_export"])
    if duplicate_lookup_rows:
        write_csv(args.out_dir / "duplicate_lookup_ids.csv", duplicate_lookup_rows, ["ootp_lookup_id", "count", "players"])

    summary = {
        "created_at": now_iso(),
        "roster": str(args.roster),
        "source_fg_dirs": [str(path) for path in args.source_fg_dir],
        "out_dir": str(args.out_dir),
        "source_unique_fg": len(fg_by_stem),
        "source_duplicate_stem_count": len(duplicate_source_stems),
        "lookup_only": args.lookup_only,
        "alias_rows": len(alias_plan),
        "copied_files": len(list(args.out_dir.glob("*.fg"))),
        "copy_action_counts": dict(Counter(row["action"] for row in copy_manifest)),
        "validation_rows": len(validation_rows),
        "missing_lookup_count": len(missing_lookup),
        "duplicate_lookup_id_count": len(duplicate_lookup_rows),
        "rule": "ootp_lookup_id = file_safe(bbrefminors_id) if non-empty else file_safe(lahman_id)",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
