# Verification summary

Verified on Windows with Python 3.12.13 on 2026-08-10.

- MuJoCo 3.11.0, MediaPipe 0.10.35, OpenCV 5.0.0.
- `pip check`: no broken requirements.
- `compileall`: successful for `src` and `tests`.
- Full pytest suite: **38 passed**.
- Synthetic headless end-to-end: 30 frames, 30 skeleton frames, 0 stale
  commands, 20.872° command span, fixed base height 0.9275 m.
- Fixed/free MJCF scenes load with 20 canonical actuators.
- Both scenes remain finite for 1000 headless steps.
- Fixed home reset unintended self-contacts: 0.
- Maximum home tracking error after 1000 steps: 1.2015°.
- MuJoCo total model mass: 2.933134 kg.
- Native interactive run stayed alive and exposed both visible windows:
  `MuJoCo : humanoid_v4_proxy_fixed` and
  `Robot human interface - camera skeleton`.
- Official MediaPipe Full bundle opened through the Tasks API and completed an
  inference call.
- 21 copied FBX files: 0 SHA-256 mismatches against Unity sources.
- Unity worktree remained clean on `develop...origin/develop`; active prefab
  SHA-256 stayed
  `f2c571cf4d97198b13fb429bd811103faa16af815754f1d06885b3059fe907ed`.

The host reported zero Windows PnP devices in Camera/Image classes. A physical
camera frame could therefore not be captured on this machine. The real camera
path returns an actionable error; MediaPipe initialization and the complete
camera-free pipeline were verified independently.

Raw PowerShell check output is in `verification.log`; the rendered fixed-base
home scene is in `mujoco_fixed_home.png`.
