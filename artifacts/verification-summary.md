# Verification summary

Verified on Windows on 2026-08-15 at revision
`8349fcdcb2d51f43d963077ed624d01a9b894695`.

- Python 3.12.13; MuJoCo 3.11.0; MediaPipe 0.10.35; OpenCV 5.0.0.
- Full pytest suite: **793 passed in 103.20 s**.
- `compileall` and `git diff --check`: passed.
- Raw five-video IK measurement: complete, but diagnostic-only; no fidelity
  acceptance threshold is defined.
- Free-base stability: **FAIL, 4/6 clips**. Slow-balance narrowly exceeds
  landing/slip limits; frontal-leg-swing fails closed on stale support intent.
- Final safe-pose fidelity: **FAIL, 3/6 clips**. Jumping-jacks legs,
  stationary-squat, and an unfinished frontal support cycle remain failures.
- Robustness schema 3: critical **10/10**, broad randomized **16/20**, and
  independent single-support holdout **14/40** (right `9/20`, left `5/20`).
  Eight holdout trials fall; the per-cohort Wilson gates fail.

Machine-readable evidence is in:

- `pose-fidelity.json`
- `freebase-stability.json`
- `safe-pose-fidelity.json`
- `freebase-robustness.json`

The current result is a research/proxy milestone, not a full acceptance pass or
a physical-robot safety certificate. Video-specific jumping/squat candidates
and local force/CoP landing overlays were rejected after independent holdouts;
no acceptance threshold was weakened. The optional WebSocket path has no real
encoder/IMU/foot-load feedback, onboard watchdog, or hardware E-stop.
