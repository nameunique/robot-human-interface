# Asset provenance

## MediaPipe model

`models/pose_landmarker_full.task` is the official MediaPipe Pose Landmarker
Full float16 bundle downloaded from:

`https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task`

- Size: 9,398,198 bytes
- SHA-256: `4eaa5eb7a98365221087693fcc286334cf0858e2eb6e15b506aa4a7ecdcec4ad`

The model is kept as a separate runtime asset so the MediaPipe package and the
model bundle can be upgraded and evaluated independently.

## Bundled MP4 pose demo

`videos/jumping_jacks_demo.mp4` is an unmodified test clip used to
exercise the camera-equivalent MP4 path. It shows one person performing jumping
jacks and is detected by the bundled MediaPipe model on all 194 frames.

- Creator: Gustavo Fring
- Source: `https://www.pexels.com/video/woman-doing-jumping-jacks-outdoors-4767084/`
- License: Pexels License, `https://www.pexels.com/license/`
- Downloaded: 2026-08-10
- Media: MP4, 1280 x 720, 29.97 FPS, 194 frames, about 6.47 seconds
- Size: 2,017,366 bytes
- SHA-256: `323c0bc5b458f3139d8b9a9bf6f7032ef0a198d521fff1d8f659e1e534a63569`

The Pexels License permits free use and modification and does not require
attribution; the credit is retained here for provenance. Do not imply that the
creator or depicted person endorses this project.

## Bundled slow balance demo

`videos/slow_balance_demo.mp4` is the default MP4 test. It combines two
public-domain DVIDS exercise demonstrations featuring one full-body subject and
a stationary 16:9 camera:

- `Arm Circles`, Capt. Matthew Holfinger, U.S. Marine Corps Training and
  Education Command: `https://www.dvidshub.net/video/551356/arm-circles`
  - 1024 x 576 source stream:
    `https://d34w7g4gy10iej.cloudfront.net/video/1709/DOD_104840680/DOD_104840680-1024x576-1769k.mp4`
- `Single Leg Balance`, Capt. Matthew Holfinger, U.S. Marine Corps Training and
  Education Command: `https://www.dvidshub.net/video/551827/single-leg-balance`
  - 1024 x 576 source stream:
    `https://d34w7g4gy10iej.cloudfront.net/video/1709/DOD_104849764/DOD_104849764-1024x576-1769k.mp4`
- Public-use notice: `https://www.dvidshub.net/about/copyright`

The reproducible builder is `tools/build_slow_demo.py`. It removes the long
static introductions and fade-outs, keeps a two-second neutral acquisition
pose, slows every retained motion frame by exactly 2x through frame duplication,
and adds a 1.5-second neutral transition between the arm and balance sections.
The result contains slow arm elevation/circles followed by balance on each leg.

- Downloaded and derived: 2026-08-11
- Media: MP4 (MPEG-4 Part 2), 1024 x 576, 29.97 FPS, 1,961 frames,
  about 65.43 seconds, no audio
- Size: 13,875,839 bytes
- SHA-256: `5a91b2082043006920eb3d35d0799db5cf1d2657b6d3ec99b9b03e1daaeea222`

The appearance of U.S. Department of War (DoW) visual information does not
imply or constitute DoW endorsement.

## Unity visual sources

`unity_fbx/` contains byte-for-byte copies of the 21 FBX geometry files used by
the active `humanoid_v4` prefab in the read-only Unity project. MuJoCo does not
load FBX directly, so a separate batch-only Unity project converts them into 21
Wavefront OBJ files under `models/humanoid/meshes/`.

The conversion project is `tools/fbx_converter_unity/`. It imports only the FBX
copies from this repository and never opens or modifies the source Unity project.
The checked-in OBJ files contain 259,480 render vertices and 122,251 triangles;
they are platform-independent and load unchanged on Windows and Ubuntu.

The Unity-to-MuJoCo conversion applied to every imported Unity-space vertex is:

`p_mj = 0.35 * (-p_unity.z, -p_unity.x, p_unity.y) + (-0.035, 0, 0)`

OBJ geoms are visual-only (`contype=0`, `conaffinity=0`, `mass=0`). Dynamics,
inertia and contacts continue to use separately marked provisional primitive
geoms, so changing the rendered view cannot change the simulation.

Source root at extraction time:

`C:/Users/k_desktop/Desktop/AR_Projects/humanoidconfigurator/Assets/models/new_humanoid/parsed`

Active prefab SHA-256 at extraction time:

`f2c571cf4d97198b13fb429bd811103faa16af815754f1d06885b3059fe907ed`

The source Unity project was not modified or launched during extraction or OBJ
conversion. Unity Editor 6000.0.43f1 was used only on the isolated batch project.
