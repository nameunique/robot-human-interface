[CmdletBinding()]
param(
    [switch]$FullFreeBaseAcceptance
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$env:MPLCONFIGDIR = Join-Path $projectRoot ".cache\matplotlib"
$env:PIP_CACHE_DIR = Join-Path $projectRoot ".cache\pip"
New-Item -ItemType Directory -Force -Path $env:MPLCONFIGDIR, $env:PIP_CACHE_DIR | Out-Null

if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    & (Join-Path $PSScriptRoot "setup_windows.ps1")
}

Write-Host "Checking native Python dependencies..."
& $venvPython -c "import cv2, mediapipe, mujoco, numpy, websocket, robot_human_interface; assert callable(websocket.create_connection); print(f'Python imports OK; MuJoCo {mujoco.__version__}, MediaPipe {mediapipe.__version__}')"
if ($LASTEXITCODE -ne 0) {
    throw "Dependency import/version smoke test failed."
}

Write-Host "Running the test suite..."
& $venvPython -m pytest
if ($LASTEXITCODE -ne 0) {
    throw "pytest failed with exit code $LASTEXITCODE."
}

Write-Host "Running a finite camera-free, display-free free-base smoke test..."
& $venvPython -m robot_human_interface.app.teleop `
    --source synthetic `
    --headless `
    --free-base `
    --max-frames 30
if ($LASTEXITCODE -ne 0) {
    throw "Headless synthetic teleoperation failed with exit code $LASTEXITCODE."
}

if ($FullFreeBaseAcceptance) {
    Write-Host "Running the six-video free-base stability acceptance matrix..."
    & $venvPython (Join-Path $projectRoot "tools\evaluate_freebase_stability.py")
    if ($LASTEXITCODE -ne 0) {
        throw "Free-base stability acceptance failed with exit code $LASTEXITCODE."
    }

    Write-Host "Running final safe-command pose fidelity acceptance..."
    & $venvPython (Join-Path $projectRoot "tools\evaluate_safe_pose_fidelity.py")
    if ($LASTEXITCODE -ne 0) {
        throw "Safe pose fidelity acceptance failed with exit code $LASTEXITCODE."
    }

    Write-Host "Running free-base perturbation/domain-randomization acceptance..."
    & $venvPython (Join-Path $projectRoot "tools\evaluate_freebase_robustness.py")
    if ($LASTEXITCODE -ne 0) {
        throw "Free-base robustness acceptance failed with exit code $LASTEXITCODE."
    }
}

Write-Host "All checks passed."
