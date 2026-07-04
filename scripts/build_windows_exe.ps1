$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$releaseRoot = Join-Path $repoRoot "release"
$distRoot = Join-Path $releaseRoot "windows"
$buildRoot = Join-Path $repoRoot "build"
$pyinstallerWork = Join-Path $buildRoot "pyinstaller"
$iconPath = Join-Path $buildRoot "faceforge.ico"
$entryPath = Join-Path $repoRoot "scripts\faceforge_gui_entry.py"

New-Item -ItemType Directory -Force -Path $releaseRoot, $distRoot, $buildRoot | Out-Null

$iconScript = @'
from pathlib import Path
from PIL import Image, ImageDraw

out = Path(r"__ICON_PATH__")
out.parent.mkdir(parents=True, exist_ok=True)

sizes = [16, 24, 32, 48, 64, 128, 256]
images = []
for size in sizes:
    scale = size / 128
    img = Image.new("RGBA", (size, size), "#f5f5f700")
    d = ImageDraw.Draw(img)

    def xy(values):
        return [round(v * scale) for v in values]

    d.ellipse(xy([18, 20, 112, 114]), fill=(22, 32, 51, 45))
    d.ellipse(xy([16, 12, 112, 108]), fill="#fbfaf6", outline="#1f2a3a",
              width=max(1, round(4 * scale)))
    d.arc(xy([-8, 8, 74, 112]), start=-58, end=58, fill="#c92335",
          width=max(1, round(5 * scale)))
    d.arc(xy([54, 8, 136, 112]), start=122, end=238, fill="#c92335",
          width=max(1, round(5 * scale)))
    stitch_w = max(1, round(3 * scale))
    for points in ([38, 32, 30, 36], [43, 43, 34, 46], [45, 55, 35, 56],
                   [45, 67, 35, 66], [43, 79, 34, 76], [38, 90, 30, 86],
                   [90, 32, 98, 36], [85, 43, 94, 46], [83, 55, 93, 56],
                   [83, 67, 93, 66], [85, 79, 94, 76], [90, 90, 98, 86]):
        d.line(xy(points), fill="#c92335", width=stitch_w)
    d.arc(xy([38, 16, 96, 50]), start=200, end=340, fill="#ffffff",
          width=max(1, round(3 * scale)))
    images.append(img)

images[-1].save(out, sizes=[(s, s) for s in sizes], append_images=images[:-1])
'@

$iconScript = $iconScript.Replace("__ICON_PATH__", $iconPath.Replace("\", "\\"))
$tmpIconScript = Join-Path $buildRoot "make_icon.py"
Set-Content -Path $tmpIconScript -Value $iconScript -Encoding UTF8
python $tmpIconScript

$env:MPLBACKEND = "Agg"

python -m PyInstaller `
  --noconfirm `
  --clean `
  --windowed `
  --name "OOTP-FaceForge" `
  --distpath $distRoot `
  --workpath $pyinstallerWork `
  --specpath $buildRoot `
  --icon $iconPath `
  --paths (Join-Path $repoRoot "src") `
  --add-data "$(Join-Path $repoRoot 'src\ootp_faceforge\face_landmarker.task');ootp_faceforge" `
  --add-data "$(Join-Path $repoRoot 'src\ootp_faceforge\assets\icon.svg');ootp_faceforge\assets" `
  --add-data "$(Join-Path $repoRoot 'src\ootp_faceforge\assets\icon.ico');ootp_faceforge\assets" `
  --collect-data "mediapipe" `
  --collect-binaries "mediapipe" `
  --hidden-import "ootp_faceforge.pipeline" `
  --hidden-import "ootp_faceforge.render" `
  --hidden-import "ootp_faceforge.calibrate" `
  --hidden-import "ootp_faceforge.emb2shape" `
  --hidden-import "ootp_faceforge.identity" `
  --hidden-import "ootp_faceforge.restore" `
  --hidden-import "mediapipe" `
  --hidden-import "mediapipe.tasks" `
  --hidden-import "mediapipe.tasks.python" `
  --hidden-import "mediapipe.tasks.python.core.base_options" `
  --hidden-import "mediapipe.tasks.python.vision" `
  --hidden-import "PIL._tkinter_finder" `
  --exclude-module "PyQt5" `
  --exclude-module "PyQt6" `
  --exclude-module "PySide2" `
  --exclude-module "PySide6" `
  --exclude-module "IPython" `
  --exclude-module "pytest" `
  --exclude-module "pygame" `
  --exclude-module "scipy" `
  --exclude-module "sounddevice" `
  --exclude-module "tensorflow" `
  --exclude-module "jax" `
  --exclude-module "torch" `
  $entryPath

$exePath = Join-Path $distRoot "OOTP-FaceForge\OOTP-FaceForge.exe"
if (!(Test-Path $exePath)) {
  throw "Expected exe was not created: $exePath"
}

Write-Host "Built $exePath"
