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

Users must provide their own licensed local OOTP installation.

## Requirements

- Windows.
- Python 3.11+.
- OOTP 27 installed locally. FaceForge reads FaceGen assets from the user's
  installation at runtime.
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

## OOTP Asset Path

The app checks these locations in order:

- `OOTP_FACEFORGE_OOTP_3D`, if set.
- The saved GUI setting in `%USERPROFILE%\FaceForgeWorkspace\config.json`.
- Steam library folders, including the default Steam install path.
- Common non-Steam Program Files install folders.

If the app cannot find OOTP automatically, click **Choose OOTP Folder** in the
GUI and select either:

```text
...\Out of the Park Baseball 27\data\facegen
```

or the inner `3d` folder:

```text
...\Out of the Park Baseball 27\data\facegen\3d
```

The selected path is remembered for future GUI, CLI, and batch runs. Advanced
users can also set `OOTP_FACEFORGE_OOTP_3D` directly to the `3d` folder.

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

By default, finished `.fg` files are collected in one user-facing folder:

```text
%USERPROFILE%\FaceForgeWorkspace\fg_files\
  park_yong-taek.fg
```

The detailed build output is still kept separately for previews and logs:

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
`--fg-dir` for a one-off export folder. `--out-dir` only changes the detailed
run folders. Generated output inside the repo is ignored by Git by default, but
keeping runs outside the repo makes development and release folders less noisy.

## Useful Build Profiles

```powershell
# Default OOTP in-game pipeline.
ootp-faceforge build $env:USERPROFILE\FaceForgeWorkspace\photos\park_yongtaek --name "Park Yong-taek"

# Strict front-only fitting for unstable photo sets.
ootp-faceforge build $env:USERPROFILE\FaceForgeWorkspace\photos\park_yongtaek --profile strict-front

# Softer mouth/detail handling for photos with visible teeth.
ootp-faceforge build $env:USERPROFILE\FaceForgeWorkspace\photos\park_yongtaek --profile mouth-soft

# Fast direct likeness pass. Uses optional ArcFace/emb2shape models when present.
ootp-faceforge build $env:USERPROFILE\FaceForgeWorkspace\photos\park_yongtaek --profile likeness-fast

# Fast .fg-only batch path. Skips preview rendering to stay in the 2-5s target band.
ootp-faceforge build $env:USERPROFILE\FaceForgeWorkspace\photos\park_yongtaek --profile likeness-fast --no-preview

# Practical likeness pass. Adds a small identity-search cleanup after direct fit.
ootp-faceforge build $env:USERPROFILE\FaceForgeWorkspace\photos\park_yongtaek --profile likeness

# Slow research/hero-player pass. Uses restoration and a deeper identity search.
ootp-faceforge build $env:USERPROFILE\FaceForgeWorkspace\photos\park_yongtaek --profile likeness-max
```

`likeness-fast` is the Modeller-style target path: one detection/embedding pass,
then direct coefficient seeding. If `models\cufp_identity_index.npz` exists, it
retrieves real CUFP FaceGen prototypes by photo embedding and face geometry. If
`models\fg_render_identity_index.npz` also exists, FaceForge uses it only to
rerank those CUFP photo candidates, not as a global search source. If no index is
present it falls back to the optional `photofit`/`emb2shape` direct priors. The
selected identity prior is then fed into a Modeller-style direct shape solve, so
the prior influences the landmark fit instead of overwriting it after the fact.
`--no-preview` skips the final PNG render when you only need the `.fg`.
`likeness-max` is intentionally slower and is best treated as a research or
hero-player QA pass.

## Render Retrieval Index

The render-domain index pre-renders known `.fg`/coefficient rows with the local
OOTP renderer and stores ArcFace/geometry features from those renders. It is not
used as a global search by default; it reranks the top CUFP photo-retrieval
candidates inside that candidate set:

```powershell
python scripts\build_render_index.py build --out models\fg_render_identity_index.npz --sources cufp --cufp-split all
python scripts\build_render_index.py eval --index models\fg_render_identity_index.npz --limit 200
```

This is an offline job. A small pilot is useful before committing to a full
index:

```powershell
python scripts\build_render_index.py build --out build\fg_render_identity_index_pilot.npz --sources cufp --cufp-split val --cufp-limit 100
```

## CUFP Retrieval Index

The preferred fast likeness prior is not another global photo-to-coeff
regressor. CUFP photo/FG pairs are better used as a nearest-neighbour bank:

```powershell
python scripts\build_cufp_index.py build --out models\cufp_identity_index.npz
python scripts\build_cufp_index.py eval --index models\cufp_identity_index.npz --limit 200
```

At build time FaceForge compares the player's photo embedding and normalized
face geometry against that index, blends the top matching real FaceGen
coefficients, then lets the normal texture/detail pipeline finish the face.

## Photofit Training

The direct photofit model is still local and optional, but it is a fallback path.
It learns from the user's OOTP `.fg` library by rendering known FaceGen files,
applying photo-like augmentation, and training a small numpy ridge model:

```powershell
python scripts\train_photofit.py gen --out models\photofit_data.npz --augs 4
python scripts\train_photofit.py train --data models\photofit_data.npz --model models\photofit.npz
python scripts\train_photofit.py eval --data models\photofit_data.npz --model models\photofit.npz --out-dir models
```

The eval step writes `photofit_eval_report.json` plus contact-sheet PNGs comparing
the true render, photofit prediction, optional `emb2shape` baseline, and mean
face. Use those reports before promoting a trained model into batch work.

You can force a texture photo with any filename substring:

```powershell
ootp-faceforge build $env:USERPROFILE\FaceForgeWorkspace\photos\park_yongtaek --texture-photo official
```

By default, texture/detail are fused from every usable photo. To compare against
the older single-photo path, run:

```powershell
ootp-faceforge build $env:USERPROFILE\FaceForgeWorkspace\photos\park_yongtaek --texture-mode best
```

### Delighting

Photos carry their own studio/stadium lighting, and OOTP's renderer adds its
own light again. By default FaceForge removes baked-in photo lighting with a
geometry-based spherical-harmonics fit before texture/detail baking: the fitted
mesh's surface normals predict the photo's smooth shading, and dividing it out
leaves approximate albedo. This keeps identity detail (stubble, brows, lip
lines, skin color variation) that the older blur-based shadow neutralizers used
to erase along with the lighting.

```powershell
# default: SH delighting (falls back to mirror correction when the fit fails)
ootp-faceforge build ...\park_yongtaek

# older midline mirror correction, or no lighting correction at all
ootp-faceforge build ...\park_yongtaek --delight mirror
ootp-faceforge build ...\park_yongtaek --delight off
```

After the SH division a light mirror pass still runs: SH models smooth attached
shading only, so hard one-sided cast shadows (a nose wedge from stage lighting)
are cleaned up by midline symmetrization, which stays near a no-op on photos
that were already evenly lit.

When every photo in a build was successfully delit, the detail clean-up relaxes
automatically: `--detail-chroma-strength` defaults to 0.2 (instead of 0.08),
`--detail-shadow-neutralize` to 0.55 (instead of 0.8), and the new
`--detail-dark-keep` to 0.6, which lets sub-neutral (dark) albedo detail such
as facial-hair shading survive into the OOTP detail map. Explicit flags always
win over these adaptive defaults.

Two more texture changes support the delit detail: the default `--detail-size`
is now 512, and multi-photo detail fusion splits each map into frequency bands
— tone (low band) is still blended from every photo, but crisp detail (high
band) comes almost entirely from the best-scoring photo, so cross-photo
misalignment no longer averages stubble and brow edges into mush.

Because delit photos tolerate a much stronger bake, the default detail knobs
moved up as well: `--detail-strength` 1.15, `--detail-edge-strength` 1.1,
`--eye-detail-strength` 1.0, `--likeness-detail` 0.65. The default build also
now enables `--retrieval auto`, `--photofit auto`, and `--modeller-fit auto`,
so the CUFP identity prior and Modeller-style shape refit activate
automatically whenever the models/index files are present in `models/`. On a
three-player benchmark (multi-photo ArcFace similarity) the combined pipeline
scores rose 28-82% over the previous defaults.

Low-resolution GFPGAN restoration is opt-in because it can add time and
hallucinated detail:

```powershell
ootp-faceforge build $env:USERPROFILE\FaceForgeWorkspace\photos\park_yongtaek --restore auto
```

### Eyeglasses

FaceGen `.fg` files do not carry a separate glasses/accessory mesh in the final
file, and OOTP ships no glasses assets at all, so the detail map is the only
channel — FaceForge bakes glasses into it. The default `--glasses-method auto`
picks the best available route:

- **mesh** (default route): a real 3D glasses mesh is rendered in screen
  space with a z-test against the fitted face and scattered back onto the
  detail texture through the camera rays, so the baked frame keeps rigid
  screen-space lines (lenses, bridge, temple arms) instead of hugging the
  skin. The mesh itself comes from the first available source:
  1. the template registry (`src/ootp_faceforge/assets/glasses_templates.json`)
     auto-picks the best CC-licensed converted mesh for the detected style
     from `<workspace>\models\glasses\` (these user-provided Sketchfab
     conversions are not redistributed with the repo);
  2. a license-clean procedural glasses mesh generated from the player's own
     3D landmarks (tube-swept rims sized from the interpupillary distance) —
     always available, fully open source;
  3. a FaceGen Modeller `Accessories` folder, only when explicitly passed via
     `--glasses-mesh-assets`.
  When glasses are actually baked, the detail map is upscaled to 1024px /
  JPEG 94 so the rims stay crisp; bare-face builds keep the normal 512px
  output.
- **parametric** (2D fallback): removes the raw source glasses from the face
  texture, then redraws a clean layered vector frame from the detected
  placement, style, and color. Detected `sports_goggle` frames draw bolder
  automatically (strength/rim-width floors) so they stay legible at OOTP's
  small in-game portrait scale.

Frame colors are stored as multiplicative detail-space ratios and the detail
JPEG is written with 4:4:4 chroma, so a red goggle renders vivid red instead
of washed-out brick.

The optional BiSeNet face-parser model
(`python scripts/download_restore_model.py`) is still used to detect whether the
source face wears glasses and to infer frame style/color. Without it, force
glasses with explicit style/color options.

Composite methods:

- `mesh` (recommended): bake a 3D accessory into the detail texture. Normal
  glasses use the local FaceGen accessory; `sports_goggle` uses FaceForge's
  generated 3D goggle mesh instead of painted ovals.
- `parametric` (default fallback): remove the raw source glasses from the face
  texture, then redraw a clean layered vector frame using detected placement and
  color.
- `frame`: composite the source glasses frame contour/color onto the finished
  detail map. This follows the photo more literally, but can look like a colored
  smear when the source mask includes lens tint or shadow.
- `draw`: fit an ellipse to each detected lens and draw a clean generic frame
  onto the finished detail map. Rim darkness: `--glasses-frame-strength` (0..1).
- `protect`: keep the whole warped glasses region through the neutralizers
  (`--glasses-strength` for its contrast).
- `suppress`: remove detected source-photo glasses from the baked face detail.

```powershell
# recommended: bake a 3D glasses/goggle accessory
ootp-faceforge build ...\park_yongtaek --glasses-method mesh

# optional explicit accessory folder
ootp-faceforge build ...\park_yongtaek --glasses-method mesh --glasses-mesh-assets "C:\Program Files\FaceGen\Modeller Demo 3\data\csam\Animate\Accessories"

# custom Sketchfab/Fab-style glTF/FBX/OBJ/DAE download, extracted or zipped
ootp-faceforge build ...\yang_hyeonjong --glasses-method mesh --glasses-style rectangular --glasses-color red --glasses-mesh-assets "C:\Users\user\Downloads\glasses.zip"

# named local template alias
ootp-faceforge build ...\yang_hyeonjong --glasses-method mesh --glasses-mesh-assets sports_wrap_outline_v1

# alternative methods, or force glasses handling off
ootp-faceforge build ...\park_yongtaek --glasses-method parametric
ootp-faceforge build ...\park_yongtaek --glasses-method frame
ootp-faceforge build ...\park_yongtaek --glasses-method draw
ootp-faceforge build ...\park_yongtaek --glasses-method protect --glasses-strength 1.5
ootp-faceforge build ...\park_yongtaek --glasses-method suppress
ootp-faceforge build ...\park_yongtaek --glasses off

# force a specific template when auto inference is not enough
ootp-faceforge build ...\yang_hyeonjong --glasses-method mesh --glasses-style sports_goggle --glasses-color red
ootp-faceforge build ...\yang_hyeonjong --glasses-rim-width 1.12 --glasses-lens-width 1.38 --glasses-lens-height 0.74 --glasses-bridge thick
```

Built-in local template aliases are defined in
`src/ootp_faceforge/assets/glasses_templates.json`:

- `sports_wrap_outline_v1`: best current red sports-goggle candidate.
- `round_wire_v1`: thin round wire-frame glasses.
- `rect_fullrim_v1`: heavier rectangular full-rim glasses.
- `rect_light_v1`: lighter rectangular glasses.
- `browline_thin_v1`: weak top-rim candidate kept for comparison.

The `mesh` and `parametric` methods support `auto`, `sports_goggle`,
`rectangular`, `round`, and `oval` frame templates plus `auto`, `red`, `black`,
`brown`, `blue`, and `silver` frame colors. `--glasses-mesh-scale-x`,
`--glasses-mesh-scale-y`, `--glasses-mesh-offset-y`, and `--glasses-rim-width`
also tune generated or custom glTF/FBX/OBJ/DAE meshes. For custom
glTF/FBX/OBJ/DAE/zip assets,
transparent lens materials are skipped and opaque frame/handle geometry is baked
as the accessory. FBX/OBJ/DAE assets are converted through Blender into cached glTF
on first use; nested Sketchfab zip downloads are unpacked in the cache. Mesh
bakes raise detail output to 1024px / JPEG quality 94
automatically, because smaller detail maps make the rims break into blotches.

## Likeness Lab

The pipeline's identity priors and refine passes have no single best setting
per player, so `--lab` builds four candidates (default, id-refine, id-refine
with stronger detail, and no-prior id-refine), renders each with the local
OOTP renderer, scores every render against all of the player's photos with
ArcFace, and keeps the winner:

```powershell
ootp-faceforge build $env:USERPROFILE\FaceForgeWorkspace\photos\park_yongtaek --lab
```

The run folder gains `lab/` with every candidate `.fg` and
`preview/<slug>_lab.png`, a labeled contact sheet with per-candidate scores.
The manifest records each candidate's score so the choice is auditable. In the
GUI, enable the "Likeness Lab (best of 4, slower)" checkbox. A lab build is
roughly 4x slower than a normal build; when the ArcFace model is missing it
falls back to a normal single build.

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
  basis.py            # OOTP face_hi basis loading for fitting
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

OOTP FaceForge reads OOTP assets from the user's local installation at runtime.
Those assets are not redistributed here. Do not publish player photos,
generated face packs, or third-party game/model assets unless you have the
rights to do so.

## Roadmap

- Add hair, cap, and facial-hair preview overlays.
- Add a likeness lab that builds several photofit candidates per player and
  ranks them by render/photo identity similarity.
- Add a photo contact sheet and automatic best-texture explanation.
- Add a QA score for mouth artifacts, cap shadows, skin-tone drift, and landmark
  confidence.
- Add export helpers for OOTP saved-game facegen folders.
