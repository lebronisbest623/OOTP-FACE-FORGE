from __future__ import annotations

import argparse
import csv
import io
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image


CDN_URL = "https://6ptotvmi5753.edge.naverncp.com/KBO_IMAGE/person/middle/{year}/{pid}.jpg"
IMAGE_RE = re.compile(r"^kbo_(\d+)$")
DEFAULT_YEARS = list(range(2026, 2000, -1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download official KBO profile headshots into OOTP FaceForge photo folders."
    )
    parser.add_argument(
        "--index",
        default=str(
            Path.home()
            / "FaceForgeWorkspace"
            / "photos"
            / "_priority_facegen_core_1200"
            / "_priority_facegen_core_1200_index.csv"
        ),
        help="CSV index with kbo_id and target_path columns.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Output CSV report path. Defaults next to the index file.",
    )
    parser.add_argument(
        "--base-root",
        default=None,
        help="Base folder for indexes that only contain a folder column. Defaults to the index parent.",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--sleep", type=float, default=0.0, help="Optional delay per successful request.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing official_kbo_profile jpg.")
    parser.add_argument("--start-year", type=int, default=2026)
    parser.add_argument("--end-year", type=int, default=2001)
    return parser.parse_args()


def row_value(row: dict[str, str], *names: str) -> str:
    for name in names:
        if row.get(name):
            return row[name].strip()
    return ""


def image_dimensions(data: bytes) -> tuple[int, int] | None:
    try:
        image = Image.open(io.BytesIO(data))
        image.verify()
        return image.size
    except Exception:
        return None


def fetch(pid: str, year: int) -> tuple[bytes, tuple[int, int]] | None:
    url = CDN_URL.format(year=year, pid=pid)
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(req, timeout=10) as response:
            content_type = response.headers.get("Content-Type", "")
            data = response.read()
    except (HTTPError, URLError, TimeoutError, OSError):
        return None
    if "image" not in content_type.lower() and not data.startswith(b"\xff\xd8"):
        return None
    dims = image_dimensions(data)
    if not dims:
        return None
    width, height = dims
    if width < 40 or height < 40 or len(data) < 1000:
        return None
    return data, dims


def existing_profile_image(target: Path) -> Path | None:
    matches = sorted(target.glob("official_kbo_profile_*.jpg"))
    return matches[0] if matches else None


def resolve_target(row: dict[str, str], base_root: Path) -> Path:
    explicit = row_value(row, "target_path", "queue_path")
    if explicit:
        return Path(explicit)
    folder = row_value(row, "folder")
    if folder:
        return base_root / folder
    return Path()


def download_one(
    row: dict[str, str],
    years: list[int],
    force: bool,
    sleep: float,
    base_root: Path,
) -> dict[str, str]:
    kbo_id = row_value(row, "kbo_id", "lahman_id")
    match = IMAGE_RE.match(kbo_id)
    target = resolve_target(row, base_root)
    result = {
        "status": "skipped_non_kbo_id",
        "ootp_id": row_value(row, "ootp_id"),
        "kbo_id": kbo_id,
        "display_name": row_value(row, "display_name"),
        "team_name": row_value(row, "team_name"),
        "league_name": row_value(row, "league_name"),
        "folder": row_value(row, "folder"),
        "target_path": str(target),
        "saved_path": "",
        "source_url": "",
        "source_year": "",
        "width": "",
        "height": "",
        "bytes": "",
    }
    if not match:
        return result
    if not target.exists():
        result["status"] = "missing_target_folder"
        return result
    existing = existing_profile_image(target)
    if existing and not force:
        result["status"] = "already_exists"
        result["saved_path"] = str(existing)
        return result

    pid = match.group(1)
    for year in years:
        hit = fetch(pid, year)
        if not hit:
            continue
        data, dims = hit
        out_path = target / f"official_kbo_profile_{year}_{pid}.jpg"
        out_path.write_bytes(data)
        if sleep:
            time.sleep(sleep)
        result.update(
            {
                "status": "downloaded",
                "saved_path": str(out_path),
                "source_url": CDN_URL.format(year=year, pid=pid),
                "source_year": str(year),
                "width": str(dims[0]),
                "height": str(dims[1]),
                "bytes": str(len(data)),
            }
        )
        return result

    result["status"] = "not_found"
    return result


def main() -> None:
    args = parse_args()
    index_path = Path(args.index)
    years = list(range(args.start_year, args.end_year - 1, -1))
    report_path = Path(args.report) if args.report else index_path.with_name(index_path.stem + "_kbo_profile_download_report.csv")
    base_root = Path(args.base_root) if args.base_root else index_path.parent
    rows = list(csv.DictReader(index_path.open(encoding="utf-8-sig")))

    results: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(download_one, row, years, args.force, args.sleep, base_root) for row in rows]
        for future in as_completed(futures):
            results.append(future.result())

    order = {row_value(row, "ootp_id"): i for i, row in enumerate(rows)}
    results.sort(key=lambda r: order.get(r["ootp_id"], 10**9))

    fieldnames = [
        "status",
        "ootp_id",
        "kbo_id",
        "display_name",
        "team_name",
        "league_name",
        "folder",
        "target_path",
        "saved_path",
        "source_url",
        "source_year",
        "width",
        "height",
        "bytes",
    ]
    with report_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    counts: dict[str, int] = {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    print(f"report={report_path}")
    for status, count in sorted(counts.items()):
        print(f"{status}={count}")


if __name__ == "__main__":
    main()
