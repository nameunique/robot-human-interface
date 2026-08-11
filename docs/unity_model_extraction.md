# Unity model extraction record

This record was produced by static, read-only inspection of
`Assets/Prefabs/Resourses/humanoid/humanoid_v4.prefab` in the Unity project.
The active object is `control` with the `humanoid_control` component. Service
duplicates named `model_hide*` were intentionally excluded.

- Prefab SHA-256: `f2c571cf4d97198b13fb429bd811103faa16af815754f1d06885b3059fe907ed`
- Unity project status before and after extraction: clean `develop...origin/develop`
- Active torso mass: 1.0 kg; active torso is kinematic in Unity
- Sum of torso and 20 active child rigid bodies: 2.933134 kg
- Hinge configuration in Unity: spring 100000, damper 0, motor disabled,
  limits enabled, connected-body collision disabled

The anchors below are serialized in Unity model coordinates. Every active FBX
root and organizing parent has an identity transform, so those coordinates can
be used as model-space joint locations. Unity units are assumed to be metres by
the initial proxy, but this assumption must be checked against the physical
robot: the model is approximately 3.5 units tall.

| # | Joint | Parent | Unity axis | Limits (deg) | Start (deg) | Anchor (Unity x,y,z) | Mass (kg) |
|---:|---|---|---|---:|---:|---|---:|
| 0 | shoulder_rh | torso | -X | -180..180 | 30 | 0.58, 3.127, -0.01 | 0.0107 |
| 1 | shoulder_lh | torso | -X | -180..180 | 30 | -0.58, 3.127, -0.01 | 0.0107 |
| 2 | elbow_rh | shoulder_rh | +Z | -50..160 | 15 | 0.83, 2.947, 0.20 | 0.0886 |
| 3 | elbow_lh | shoulder_lh | -Z | -50..160 | 15 | -0.83, 2.947, 0.18 | 0.0921 |
| 4 | wrist_rh | elbow_rh | -Z | -90..90 | 15 | 0.83, 2.242, 0.20 | 0.0921 |
| 5 | wrist_lh | elbow_lh | +Z | -90..90 | 15 | -0.83, 2.242, 0.18 | 0.0921 |
| 6 | rotat_axis_rl | torso | +Y | -40..40 | 0 | 0.40, 2.000, 0.00 | 0.0142 |
| 7 | rotat_axis_ll | torso | -Y | -40..40 | 0 | -0.40, 2.000, 0.00 | 0.0142 |
| 8 | motors_thigh_rl | rotat_axis_rl | +Z | -40..40 | 5 | 0.40, 1.747, -0.210 | 0.143017 |
| 9 | motors_thigh_ll | rotat_axis_ll | -Z | -40..40 | 5 | -0.40, 1.747, -0.212 | 0.143017 |
| 10 | knee_rl | motors_thigh_rl | -X | -30..90 | 28 | 0.60, 1.747, 0.00 | 0.0334 |
| 11 | knee_ll | motors_thigh_ll | -X | -30..90 | 28 | -0.60, 1.747, 0.00 | 0.0334 |
| 12 | shin_rl | knee_rl | +X | -10..150 | 45 | 0.60, 0.953, -0.10 | 0.0972 |
| 13 | shin_ll | knee_ll | +X | -10..150 | 45 | -0.60, 0.953, -0.10 | 0.0972 |
| 14 | motors_feet_rl | shin_rl | -X | -75..75 | 20 | 0.61, 0.160, 0.00 | 0.142 |
| 15 | motors_feet_ll | shin_ll | -X | -75..75 | 20 | -0.60, 0.160, 0.00 | 0.142 |
| 16 | foot_rl | motors_feet_rl | +Z | -60..45 | -5 | 0.42, 0.170, -0.21 | 0.200 |
| 17 | foot_ll | motors_feet_ll | -Z | -60..45 | -5 | -0.40, 0.160, -0.21 | 0.200 |
| 18 | neck | torso | -Y | -180..180 | 0 | 0.00, 3.280, -0.148 | 0.0107 |
| 19 | head | neck | -X | -25..70 | 0 | -0.20, 3.543, -0.1479 | 0.2765 |

The start pose comes from
`Assets/StreamingAssets/zero_poses/start_pose_hum_1.json`. The Unity controller
uses only its first 20 values:

```text
[30, 30, 15, 15, 15, 15, 0, 0, 5, 5,
 28, 28, 45, 45, 20, 20, -5, -5, 0, 0]
```

No center of mass or inertia tensor is serialized for the active bodies. Unity
computes them from convex mesh colliders at runtime. Consequently, MuJoCo proxy
inertias and collision geometry are provisional rather than extracted facts.

Unity uses a Y-up left-handed convention, while this project uses a Z-up
right-handed MuJoCo convention. Positions and axes must therefore pass through
the same explicit coordinate transform, followed by positive-direction tests
for every actuator. Raw numeric axes must not be copied across conventions
without that verification.

Front-direction verification uses semantic source evidence, not panel
appearance: Unity's `FRONT` view is at +Z, the prefab's `front_motor` lies on
the +Z side, and the foot mesh has its longer toe extent toward +Z.  Under the
checked converter transform this is MuJoCo -X front; -Y/+Y are anatomical
right/left and +Z is up.
