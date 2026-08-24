[CmdletBinding()]
param(
    [switch]$FullFreeBaseAcceptance
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$uvCommand = Get-Command "uv" -ErrorAction SilentlyContinue
if (-not $uvCommand) {
    $uvCommand = Get-Command (Join-Path $projectRoot ".venv\Scripts\uv.exe") -ErrorAction SilentlyContinue
}
if (-not $uvCommand -or -not (Test-Path -LiteralPath (Join-Path $projectRoot ".venv"))) {
    & (Join-Path $PSScriptRoot "setup_windows.ps1")
    $uvCommand = Get-Command "uv" -ErrorAction SilentlyContinue
    if (-not $uvCommand) {
        $uvCommand = Get-Command (Join-Path $projectRoot ".venv\Scripts\uv.exe") -ErrorAction Stop
    }
}

$previousQtPlatform = $env:QT_QPA_PLATFORM
$env:QT_QPA_PLATFORM = "offscreen"

Push-Location $projectRoot
try {
    & $uvCommand.Source sync --locked --group dev
    if ($LASTEXITCODE -ne 0) {
        throw "uv sync failed with exit code $LASTEXITCODE."
    }

    Write-Host "Checking Qt before OpenCV/MediaPipe imports..."
    & $uvCommand.Source run --locked --no-sync python -c `
        "from PyQt6.QtWidgets import QApplication; app = QApplication([]); assert app is not None; app.quit(); print('PyQt6 offscreen OK')"
    if ($LASTEXITCODE -ne 0) {
        throw "PyQt6 offscreen smoke test failed."
    }

    Write-Host "Running the complete pytest suite..."
    & $uvCommand.Source run --locked --no-sync pytest
    if ($LASTEXITCODE -ne 0) {
        throw "pytest failed with exit code $LASTEXITCODE."
    }

    Write-Host "Running a finite camera-free, display-free pipeline smoke test..."
    & $uvCommand.Source run --locked --no-sync robot-camera-teleop `
        --source synthetic `
        --headless `
        --free-base `
        --max-frames 30
    if ($LASTEXITCODE -ne 0) {
        throw "Headless synthetic teleoperation failed with exit code $LASTEXITCODE."
    }

    Write-Host "Running a MuJoCo model/step smoke test..."
    & $uvCommand.Source run --locked --no-sync python -c `
        "from robot_human_interface.simulation import HumanoidSimulation; sim = HumanoidSimulation('fixed'); state = sim.step(2); assert state.is_finite; sim.close(); print('MuJoCo smoke OK')"
    if ($LASTEXITCODE -ne 0) {
        throw "MuJoCo smoke test failed with exit code $LASTEXITCODE."
    }

    if ($FullFreeBaseAcceptance) {
        Write-Host "Running the six-video free-base stability acceptance matrix..."
        & $uvCommand.Source run --locked --no-sync python tools\evaluate_freebase_stability.py
        if ($LASTEXITCODE -ne 0) {
            throw "Free-base stability acceptance failed with exit code $LASTEXITCODE."
        }

        Write-Host "Running final safe-command pose fidelity acceptance..."
        & $uvCommand.Source run --locked --no-sync python tools\evaluate_safe_pose_fidelity.py
        if ($LASTEXITCODE -ne 0) {
            throw "Safe pose fidelity acceptance failed with exit code $LASTEXITCODE."
        }

        Write-Host "Running free-base perturbation/domain-randomization acceptance..."
        & $uvCommand.Source run --locked --no-sync python tools\evaluate_freebase_robustness.py
        if ($LASTEXITCODE -ne 0) {
            throw "Free-base robustness acceptance failed with exit code $LASTEXITCODE."
        }
    }
}
finally {
    if ($null -eq $previousQtPlatform) {
        Remove-Item Env:QT_QPA_PLATFORM -ErrorAction SilentlyContinue
    }
    else {
        $env:QT_QPA_PLATFORM = $previousQtPlatform
    }
    Pop-Location
}

Write-Host "All checks passed."
