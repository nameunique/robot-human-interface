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

## External full-body retargeting audit clips

`videos/external/` contains four unmodified DVIDS exercise demonstrations for
checking retargeting against independent internet videos. They use a stationary
camera, one unobstructed subject, and a full-body view. Each source page marks
the work **PUBLIC DOMAIN** and identifies Capt. Matthew Holfinger and the U.S.
Marine Corps Training and Education Command. The applicable public-use and
non-endorsement notice is:

`https://www.dvidshub.net/about/copyright`

All files were downloaded from the 1024 x 576 MP4 stream exposed by the
corresponding official DVIDS embed page on 2026-08-11. The files have not been
trimmed, slowed, recompressed, or otherwise modified.

### Arms: `dvids_arm_circles.mp4`

- Source page: `https://www.dvidshub.net/video/551356/arm-circles`
- Direct media: `https://d34w7g4gy10iej.cloudfront.net/video/1709/DOD_104840680/DOD_104840680-1024x576-1769k.mp4`
- Test coverage: bilateral shoulder elevation and circular arm motion while the
  feet and torso remain nearly stationary
- Media: MP4, 1024 x 576, 29.97 FPS, 796 frames, about 26.56 seconds
- Size: 6,318,191 bytes
- SHA-256: `b44f9ea5707c091863755a9c65231e06a3a790219bbaa41eb0be04c1b622a3e2`

### Single leg: `dvids_frontal_leg_swing.mp4`

- Source page: `https://www.dvidshub.net/video/551381/frontal-leg-swing`
- Direct media: `https://d34w7g4gy10iej.cloudfront.net/video/1709/DOD_104840801/DOD_104840801-1024x576-1769k.mp4`
- Test coverage: one-leg support and repeated hip abduction/adduction; the
  subject uses the fixed rack for balance
- Media: MP4, 1024 x 576, 29.97 FPS, 836 frames, about 27.89 seconds
- Size: 6,634,218 bytes
- SHA-256: `a936684973411c674f86fb7cc1f8003137593c6aafa223df6b1b8ad46f60a967`

### Both legs: `dvids_stationary_squat.mp4`

- Source page: `https://www.dvidshub.net/video/551837/stationary-squat`
- Direct media: `https://d34w7g4gy10iej.cloudfront.net/video/1709/DOD_104849840/DOD_104849840-1024x576-1769k.mp4`
- Test coverage: symmetric hip and knee flexion with arms moving forward for
  balance
- Media: MP4, 1024 x 576, 29.97 FPS, 817 frames, about 27.26 seconds
- Size: 6,471,387 bytes
- SHA-256: `3b82ead850f106e0691385008fa8af0fe48b1f973a68ac098b7f3acf49e6f340`

### Head and trunk: `dvids_trunk_circles.mp4`

- Source page: `https://www.dvidshub.net/video/551844/trunk-circles`
- Direct media: `https://d34w7g4gy10iej.cloudfront.net/video/1709/DOD_104849856/DOD_104849856-1024x576-1769k.mp4`
- Test coverage: head pitch and translation coupled to a large circular trunk
  motion. This is not an isolated neck-yaw test and must not be treated as one.
- Media: MP4, 1024 x 576, 29.97 FPS, 867 frames, about 28.93 seconds
- Size: 6,871,249 bytes
- SHA-256: `5095be71e318228c0cc5686d1fc52696368ac71577bea5f032872635a39ceff6`

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

The source prefab's named `FRONT` view places its camera on Unity +Z, and its
named `front_motor` plus the longer toe edge are also on the +Z side.  Thus the
asset front maps to MuJoCo -X; right/left/up map to -Y/+Y/+Z.  This named Unity
evidence is authoritative when the untextured front and rear panels look
ambiguous in a MuJoCo render.

OBJ geoms are visual-only (`contype=0`, `conaffinity=0`, `mass=0`). Dynamics,
inertia and contacts continue to use separately marked provisional primitive
geoms, so changing the rendered view cannot change the simulation.

Source root at extraction time:

`C:/Users/k_desktop/Desktop/AR_Projects/humanoidconfigurator/Assets/models/new_humanoid/parsed`

Active prefab SHA-256 at extraction time:

`f2c571cf4d97198b13fb429bd811103faa16af815754f1d06885b3059fe907ed`

The source Unity project was not modified or launched during extraction or OBJ
conversion. Unity Editor 6000.0.43f1 was used only on the isolated batch project.
