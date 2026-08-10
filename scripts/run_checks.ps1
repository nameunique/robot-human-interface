[CmdletBinding()]
param()

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
& $venvPython -c "import cv2, mediapipe, mujoco, numpy, robot_human_interface; print(f'Python imports OK; MuJoCo {mujoco.__version__}, MediaPipe {mediapipe.__version__}')"
if ($LASTEXITCODE -ne 0) {
    throw "Dependency import/version smoke test failed."
}

Write-Host "Running the test suite..."
& $venvPython -m pytest
if ($LASTEXITCODE -ne 0) {
    throw "pytest failed with exit code $LASTEXITCODE."
}

Write-Host "Running a finite camera-free, display-free teleoperation smoke test..."
& $venvPython -m robot_human_interface.app.teleop `
    --source synthetic `
    --headless `
    --max-frames 30
if ($LASTEXITCODE -ne 0) {
    throw "Headless synthetic teleoperation failed with exit code $LASTEXITCODE."
}

Write-Host "All checks passed."
