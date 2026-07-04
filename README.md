# OOTP FaceForge

OOTP FaceForge builds OOTP-compatible FaceGen `.fg` files from player photo
folders. It can also render an OOTP-style preview image and write a small
manifest describing how each face was built.

This is an unofficial compatibility tool. It is not affiliated with, endorsed
by, or sponsored by Out of the Park Developments, Singular Inversions, Google,
MLB, or MLB Players, Inc.

## What It Does

- Fits one shared FaceGen shape from multiple photos.
- Fuses usable photos into shared texture/detail maps so additional good
  references can improve the final face instead of only choosing one image.
- Scores photos for yaw, mouth opening, cap shadow, and texture usefulness.
- Builds OOTP `.fg` files with FaceGen shape, texture coefficients, and detail
  maps.
- Renders a preview using the user's local OOTP `face_hi` and eye assets.
- Writes a per-player package with logs, diagnostics, and editable appearance
  metadata for hair, cap, and facial hair.

## What Is Not Included

This repository intentionally does not include:

- Player photos.
- Generated `.fg` files or preview renders.
- OOTP game assets.
- FaceGen Modeller or FaceGen SDK assets.
- FaceGen `.tri`, `.egm`, `.egt`, or `.fim` model files.

Users must provide their own licensed local OOTP and FaceGen installations.

## Requirements

- Windows.
- Python 3.11+.
- OOTP 27 installed in the default Steam path, with FaceGen assets under:
  `C:\Program Files (x86)\Steam\steamapps\common\Out of the Park Baseball 27\data\facegen`
- FaceGen Modeller Demo 3 installed with statistical model data under:
  `C:\Program Files\FaceGen\Modeller Demo 3\data\photofit`
- The bundled MediaPipe Face Landmarker model:
  `src\ootp_faceforge\face_landmarker.task`

Install the Python package in editable mode:

```powershell
pip install -e .
```

For local development without installation, the compatibility launcher still
works:

```powershell
python ootp_facegen.py --help
```

## Quick Start

Open the desktop app:

```powershell
ootp-faceforge-gui
```

Or, without installing the package:

```powershell
python ootp_facegen.py gui
```

Put several photos for one player in the workspace photos folder:

```text
%USERPROFILE%\FaceForgeWorkspace\photos\
  park_yongtaek/
    front.jpg
    official.webp
    award.jpg
```

Build a player package:

```powershell
ootp-faceforge build $env:USERPROFILE\FaceForgeWorkspace\photos\park_yongtaek --name "Park Yong-taek"
```

By default, output is written outside the source repo:

```text
%USERPROFILE%\FaceForgeWorkspace\runs\
  park_yong-taek/
    facegen/park_yong-taek.fg
    preview/park_yong-taek_ootp.png
    meta/park_yong-taek.manifest.json
    meta/park_yong-taek.appearance.json
    logs/build.log
```

Set `OOTP_FACEFORGE_WORKSPACE` to use a different workspace root, or pass
`--out-dir` for one build. Generated output inside the repo is ignored by Git by
default, but keeping runs outside the repo makes development and release folders
less noisy.

## Useful Build Profiles

```powershell
# Default OOTP in-game pipeline.
ootp-faceforge build $env:USERPROFILE\FaceForgeWorkspace\photos\park_yongtaek --name "Park Yong-taek"

# Strict front-only fitting for unstable photo sets.
ootp-faceforge build $env:USERPROFILE\FaceForgeWorkspace\photos\park_yongtaek --profile strict-front

# Softer mouth/detail handling for photos with visible teeth.
ootp-faceforge build $env:USERPROFILE\FaceForgeWorkspace\photos\park_yongtaek --profile mouth-soft
```

You can force a texture photo with any filename substring:

```powershell
ootp-faceforge build $env:USERPROFILE\FaceForgeWorkspace\photos\park_yongtaek --texture-photo official
```

By default, texture/detail are fused from every usable photo. To compare against
the older single-photo path, run:

```powershell
ootp-faceforge build $env:USERPROFILE\FaceForgeWorkspace\photos\park_yongtaek --texture-mode best
```

Low-resolution GFPGAN restoration is opt-in because it can add time and
hallucinated detail:

```powershell
ootp-faceforge build $env:USERPROFILE\FaceForgeWorkspace\photos\park_yongtaek --restore auto
```

## Batch Builds

The GUI batch button can build from selected photos or from folders. In the CLI,
pass image files, player photo folders, or a parent folder containing player
folders:

```powershell
# Build one .fg per selected image file, using each file stem as the name.
ootp-faceforge batch $env:USERPROFILE\FaceForgeWorkspace\photos\headshots\player_a.jpg $env:USERPROFILE\FaceForgeWorkspace\photos\headshots\player_b.png

# Build one .fg from a player photo folder, using the folder name.
ootp-faceforge batch $env:USERPROFILE\FaceForgeWorkspace\photos\park_yongtaek

# Build every immediate child folder that contains photos.
ootp-faceforge batch $env:USERPROFILE\FaceForgeWorkspace\photos
```

JSON batch files are still accepted for advanced scripted builds:

```powershell
ootp-faceforge batch examples\park_yongtaek.json
```

The example file names a real sample slot, but the repository does not include
player photos.

## Appearance Sidecar

OOTP FaceGen `.fg` files store face shape, face texture, and detail texture.
Hair, caps, and facial hair are separate OOTP appearance layers. FaceForge writes
an editable sidecar:

```json
{
  "hair": null,
  "hair_color": null,
  "facial_hair": null,
  "cap": null
}
```

## Project Layout

```text
src/ootp_faceforge/
  cli.py              # command line interface
  gui.py              # Tkinter desktop app
  pipeline.py         # photo folder -> .fg
  render.py           # .fg -> OOTP-style preview PNG
  basis.py            # FaceGen basis loading from local Modeller data
  fgformat.py         # .fg/.tri/.egm/.egt readers and writers
  fit.py              # shape fitting
  landmarks.py        # MediaPipe landmark detection and scoring helpers
  texture.py          # texture/detail-map fitting
  assets/icon.svg
  assets/icon.ico
  face_landmarker.task
```

## Legal Notes

The project code is licensed under the MIT License. See `LICENSE`.

The bundled MediaPipe Face Landmarker model is covered separately under
Apache License 2.0. See `THIRD_PARTY_NOTICES.md` and
`LICENSES/Apache-2.0.txt`.

OOTP FaceForge reads OOTP and FaceGen assets from the user's local installation
at runtime. Those assets are not redistributed here. Do not publish player
photos, generated face packs, or third-party game/model assets unless you have
the rights to do so.

## Roadmap

- Add hair, cap, and facial-hair preview overlays.
- Add a photo contact sheet and automatic best-texture explanation.
- Add a QA score for mouth artifacts, cap shadows, skin-tone drift, and landmark
  confidence.
- Add export helpers for OOTP saved-game facegen folders.
- Replace hard-coded local asset paths with a settings file.
