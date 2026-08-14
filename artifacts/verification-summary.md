# Verification summary

Verified on Windows on 2026-08-14 at revision
`d091d1e48c920a316f5823d2fd96e333d0aad7d6`.

- Python 3.12.13; MuJoCo 3.11.0; MediaPipe 0.10.35; OpenCV 5.0.0.
- Full pytest suite: **789 passed in 105.81 s**.
- `compileall` and `git diff --check`: passed.
- Raw five-video IK measurement: complete, but diagnostic-only; no fidelity
  acceptance threshold is defined.
- Free-base stability: **FAIL, 4/6 clips**. Slow-balance narrowly exceeds
  landing/slip limits; frontal-leg-swing fails closed on stale support intent.
- Final safe-pose fidelity: **FAIL, 3/6 clips**. Jumping-jacks legs,
  stationary-squat, and an unfinished frontal support cycle remain failures.
- Robustness: all **10/10** nominal/perturbation cases pass, but randomized
  coverage is **16/20** (Wilson lower 95% `0.58398`, required `0.80`).

Machine-readable evidence is in:

- `pose-fidelity.json`
- `freebase-stability.json`
- `safe-pose-fidelity.json`
- `freebase-robustness.json`

The current result is a research/proxy milestone, not a full acceptance pass or
a physical-robot safety certificate. The optional WebSocket path has no real
encoder/IMU/foot-load feedback, onboard watchdog, or hardware E-stop.
