"""Workspace and local OOTP asset path helpers."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ENV_WORKSPACE = "OOTP_FACEFORGE_WORKSPACE"
ENV_OOTP_3D = "OOTP_FACEFORGE_OOTP_3D"
CONFIG_NAME = "config.json"
CONFIG_OOTP_3D = "ootp_3d"

REQUIRED_OOTP_ASSETS = (
    "face_hi.tri",
    "face_hi.egm",
    "face_hi.egt",
    "face_hi.png",
    "face_hi.fim",
    "eyer_hi.tri",
    "eyer_hi.egm",
    "eyer_hi.egt",
    "eyer_hi_brown.png",
    "eyel_hi.tri",
    "eyel_hi.egm",
    "eyel_hi.egt",
    "eyel_hi_brown.png",
)


@dataclass(frozen=True)
class OOTPPathResult:
    path: Path
    source: str


class MissingOOTPAssets(FileNotFoundError):
    """Raised when the local OOTP facegen assets cannot be found."""

    def __init__(
        self,
        path: Path | None = None,
        *,
        tried: Iterable[Path] = (),
    ) -> None:
        self.path = path
        self.tried = tuple(tried)
        super().__init__(_missing_ootp_message(path, self.tried))


def workspace_root() -> Path:
    raw = os.environ.get(ENV_WORKSPACE)
    if raw:
        return Path(raw).expanduser()
    return Path.home() / "FaceForgeWorkspace"


def config_path() -> Path:
    return workspace_root() / CONFIG_NAME


def load_settings() -> dict[str, Any]:
    path = config_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_settings(settings: dict[str, Any]) -> None:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")


def save_ootp_3d_path(path: str | Path) -> Path:
    root = normalize_ootp_3d_path(path)
    settings = load_settings()
    settings[CONFIG_OOTP_3D] = str(root)
    save_settings(settings)
    return root


def missing_ootp_assets(path: str | Path) -> list[str]:
    root = Path(path).expanduser()
    return [name for name in REQUIRED_OOTP_ASSETS if not (root / name).exists()]


def is_ootp_3d_path(path: str | Path) -> bool:
    root = Path(path).expanduser()
    return root.is_dir() and not missing_ootp_assets(root)


def normalize_ootp_3d_path(path: str | Path) -> Path:
    root = Path(path).expanduser()
    tried = list(_candidate_shapes(root))
    for candidate in tried:
        if is_ootp_3d_path(candidate):
            return candidate.resolve()
    raise MissingOOTPAssets(root, tried=tried)


def detect_ootp_3d_path() -> OOTPPathResult | None:
    seen: set[str] = set()
    for source, raw in iter_ootp_3d_candidates():
        for candidate in _candidate_shapes(raw):
            key = os.path.normcase(str(candidate))
            if key in seen:
                continue
            seen.add(key)
            if is_ootp_3d_path(candidate):
                return OOTPPathResult(candidate.resolve(), source)
    return None


def get_ootp_3d_path() -> Path:
    result = detect_ootp_3d_path()
    if result is not None:
        return result.path
    raise MissingOOTPAssets(tried=(path for _, path in iter_ootp_3d_candidates()))


def iter_ootp_3d_candidates() -> Iterable[tuple[str, Path]]:
    env_path = os.environ.get(ENV_OOTP_3D)
    if env_path:
        yield ENV_OOTP_3D, Path(env_path).expanduser()

    configured = load_settings().get(CONFIG_OOTP_3D)
    if isinstance(configured, str) and configured.strip():
        yield "saved settings", Path(configured).expanduser()

    for common in _steam_common_dirs():
        yield "Steam library", common / "Out of the Park Baseball 27" / "data" / "facegen" / "3d"
        if common.exists():
            for game_dir in sorted(common.glob("Out of the Park Baseball*"), reverse=True):
                yield "Steam library", game_dir / "data" / "facegen" / "3d"

    for base in _program_files_dirs():
        yield (
            "Program Files",
            base / "Out of the Park Developments" / "OOTP Baseball 27" / "data" / "facegen" / "3d",
        )
        yield (
            "Program Files",
            base / "Out of the Park Baseball 27" / "data" / "facegen" / "3d",
        )
        yield (
            "Program Files",
            base / "OOTP Baseball 27" / "data" / "facegen" / "3d",
        )


def default_ootp_dialog_dir() -> Path:
    result = detect_ootp_3d_path()
    if result is not None:
        return result.path
    for common in _steam_common_dirs():
        if common.exists():
            return common
    for directory in _program_files_dirs():
        if directory.exists():
            return directory
    return Path.home()


def _candidate_shapes(path: Path) -> Iterable[Path]:
    yield path
    yield path / "3d"
    yield path / "facegen" / "3d"
    yield path / "data" / "facegen" / "3d"
    if path.name.lower() == "fg_files":
        yield path.parent / "facegen" / "3d"
    for common in _manual_steam_common_dirs(path):
        yield common / "Out of the Park Baseball 27" / "data" / "facegen" / "3d"
        if common.exists():
            for game_dir in sorted(common.glob("Out of the Park Baseball*"), reverse=True):
                yield game_dir / "data" / "facegen" / "3d"
    for game_dir in _manual_game_dirs(path):
        yield game_dir / "data" / "facegen" / "3d"


def _manual_steam_common_dirs(path: Path) -> Iterable[Path]:
    name = path.name.lower()
    if name == "common" and path.parent.name.lower() == "steamapps":
        yield path
    if name == "steamapps":
        yield path / "common"
    if name == "steam":
        yield path / "steamapps" / "common"
    yield path / "Steam" / "steamapps" / "common"


def _manual_game_dirs(path: Path) -> Iterable[Path]:
    if path.exists():
        yield from sorted(path.glob("Out of the Park Baseball*"), reverse=True)
    yield path / "Out of the Park Developments" / "OOTP Baseball 27"
    yield path / "Out of the Park Baseball 27"
    yield path / "OOTP Baseball 27"


def _program_files_dirs() -> Iterable[Path]:
    raw_dirs = (
        os.environ.get("PROGRAMFILES(X86)") or os.environ.get("ProgramFiles(x86)"),
        os.environ.get("PROGRAMFILES") or os.environ.get("ProgramFiles"),
        r"C:\Program Files (x86)",
        r"C:\Program Files",
    )
    seen: set[str] = set()
    for raw in raw_dirs:
        if not raw:
            continue
        path = Path(raw)
        key = os.path.normcase(str(path))
        if key not in seen:
            seen.add(key)
            yield path


def _steam_common_dirs() -> Iterable[Path]:
    roots = list(_steam_roots())
    seen: set[str] = set()
    for root in roots:
        common = root / "steamapps" / "common"
        key = os.path.normcase(str(common))
        if key not in seen:
            seen.add(key)
            yield common


def _steam_roots() -> Iterable[Path]:
    default = (
        Path(os.environ.get("PROGRAMFILES(X86)") or r"C:\Program Files (x86)")
        / "Steam"
    )
    roots = [default]
    library_file = default / "steamapps" / "libraryfolders.vdf"
    roots.extend(_parse_steam_libraries(library_file))
    seen: set[str] = set()
    for root in roots:
        key = os.path.normcase(str(root))
        if key not in seen:
            seen.add(key)
            yield root


def _parse_steam_libraries(path: Path) -> list[Path]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    out: list[Path] = []
    for raw in re.findall(r'"path"\s+"([^"]+)"', text):
        out.append(Path(raw.replace("\\\\", "\\")))
    for raw in re.findall(r'"\d+"\s+"([^"]+)"', text):
        if not raw.startswith("{"):
            out.append(Path(raw.replace("\\\\", "\\")))
    return out


def _missing_ootp_message(path: Path | None, tried: tuple[Path, ...]) -> str:
    lines = [
        "OOTP FaceGen assets were not found.",
        r"Choose the folder that contains face_hi.tri, usually:",
        r"  ...\Out of the Park Baseball 27\data\facegen\3d",
        f"Advanced users can set {ENV_OOTP_3D} to that 3d folder.",
    ]
    if path is not None:
        lines.insert(1, f"Selected folder: {path}")
        if path.name.lower() == "fg_files":
            lines.insert(
                2,
                r"That looks like fg_files, which stores player .fg files. "
                r"Choose the sibling data\facegen\3d asset folder instead.",
            )
    if tried:
        lines.append("Checked:")
        for candidate in tried[:8]:
            lines.append(f"  {candidate}")
        if len(tried) > 8:
            lines.append(f"  ... and {len(tried) - 8} more")
    return "\n".join(lines)
