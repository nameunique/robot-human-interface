# Asset provenance

## MediaPipe model

`models/pose_landmarker_full.task` is the official MediaPipe Pose Landmarker
Full float16 bundle downloaded from:

`https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task`

- Size: 9,398,198 bytes
- SHA-256: `4eaa5eb7a98365221087693fcc286334cf0858e2eb6e15b506aa4a7ecdcec4ad`

The model is kept as a separate runtime asset so the MediaPipe package and the
model bundle can be upgraded and evaluated independently.

## Unity visual sources

`unity_fbx/` contains byte-for-byte copies of the 21 FBX geometry files used by
the active `humanoid_v4` prefab in the read-only Unity project. They are copied
for future visual conversion; the first MuJoCo model deliberately uses primitive
visual and collision geoms so that dynamics do not depend on FBX conversion.

Source root at extraction time:

`C:/Users/k_desktop/Desktop/AR_Projects/humanoidconfigurator/Assets/models/new_humanoid/parsed`

Active prefab SHA-256 at extraction time:

`f2c571cf4d97198b13fb429bd811103faa16af815754f1d06885b3059fe907ed`

The Unity project was not modified or launched during extraction.
