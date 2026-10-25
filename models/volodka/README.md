# Volodka physical-model snapshot

This folder contains a portable, physics-only export from the user-supplied
`humanoid_model` project. No upstream GUI, planner, network transport or Python
runtime package is needed to load the exported model. Keep all `meshes/*.stl`
files with the XML; paths are relative, not tied to the original computer.

## Files and regeneration

- `source_scene.xml`: original canonical coordinates, body/joint names, explicit
  inertials, floor contacts and the upstream numerical position-drive reference.
- `physical_snapshot.json`: typed passport, provenance, geometry, contact and
  numerical profiles serialized as ordinary JSON, without pickle.
- `meshes/manifest.json`: the original CAD mesh manifest; all 21 binary STL
  files retain their original SHA-256 hashes.
- `import_provenance.json`: hashes of the upstream modules actually used by the
  importer and the exported artifacts.

To repeat the import, retain the original reviewed source archive and run from
the repository root (the application does **not** need that archive):

```powershell
.\.venv\Scripts\python.exe -B tools/import_humanoid_model.py --source C:\path\to\humanoid_model
```

On Linux/macOS use `.venv/bin/python` with the same arguments. Only NumPy and
MuJoCo are required by the import. This command executes Python from the source
archive, so it is a developer operation for trusted/reviewed sources.

## What is and is not established

The 21 modeled bodies total **3.0942 kg**. The upstream CAD mass ledger gives
3.0966 kg for the complete assembly and explicitly leaves 0.0024 kg unassigned.
Masses and centers of mass have upstream CAD provenance; the original drawing
and STEP assembly are not present in the supplied archive, so that provenance
has not been independently verified here.

Inertia tensors are finite rod/box approximations, not exact CAD inertia
tensors. CAD STL meshes are **visual only**. Separate body-envelope boxes and
sole patches generate contacts with the floor. Body/body self-collisions are
not included in this faithful source snapshot.

The original drive reference uses built-in position actuators and a **1.5 N m**
torque cap, but is explicitly a numerical fixture, **not** an identified model
of the real AR-S430-01 servos. Backlash, thermal effects, current, battery sag,
communication delay and physical drive response have not been calibrated.
Friction and joint ranges are provisional; CAD visual alignment is marked
`provisional_visual_mapping` upstream. Successful simulated motion is not a
hardware safety or sim-to-real validation.

The source model is Z-up with its own canonical axes, joint ordering and zero
pose. Existing Unity-order commands require an explicit adapter; do not send
these source joint values directly to the physical robot.
