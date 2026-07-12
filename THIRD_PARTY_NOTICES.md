# Third-Party Notices

This project is an unofficial compatibility tool. It is not affiliated with,
endorsed by, or sponsored by Out of the Park Developments, Singular Inversions,
Google, MLB, or MLB Players, Inc.

## Runtime Dependencies

OOTP FaceForge depends on these Python packages:

- MediaPipe, licensed under Apache License 2.0.
- NumPy, licensed under a BSD-style license.
- OpenCV Python packages, licensed under Apache License 2.0.
- Pillow, licensed under the HPND license.

See each package distribution for its full license text and bundled dependency
notices.

## Bundled MediaPipe Model

The repository includes `src/ootp_faceforge/face_landmarker.task`, the
MediaPipe Face Landmarker model bundle. The related MediaPipe model cards
identify the included face detector, face mesh, and blendshape models as
licensed under Apache License 2.0. A copy of Apache License 2.0 is included at
`LICENSES/Apache-2.0.txt`.

Reference documentation:

- https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker
- https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Model%20Card%20%28Short%20Range%29.pdf
- https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf
- https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Blendshape%20V2.pdf

## Optional Models in Windows Release Builds

The source repository does not bundle restoration/identity/parsing models; use
`scripts/download_restore_model.py` to fetch them locally. Prebuilt Windows
release packages may include `bisenet_resnet_34.onnx`, a BiSeNet ResNet-34 face
parser used to detect eyeglasses. The ONNX conversion is redistributed by the
facefusion project (https://huggingface.co/facefusion/models-3.0.0); the
underlying face-parsing model derives from CelebAMask-HQ training data, which
is provided for non-commercial research use. GFPGAN and ArcFace models are
never bundled and must be downloaded by the user.

## OOTP Assets

This repository does not include OOTP game assets. Users must provide their own
licensed local OOTP installation. The default pipeline reads those local OOTP
assets at runtime to build compatible `.fg` files and preview renders.

Do not commit player photos, generated `.fg` files, OOTP asset files, FaceGen
model files such as `.tri`, `.egm`, `.egt`, `.fim`, or generated preview images
unless you have the rights to redistribute them.
