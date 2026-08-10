[CmdletBinding()]
param(
    [ValidateSet("camera", "synthetic", "replay")]
    [string]$Source = "camera",

    [string]$ReplayPath = "",
    [int]$CameraIndex = 0,
    [ValidateSet("auto", "dshow", "msmf", "v4l2", "avfoundation", "gstreamer")]
    [string]$CameraBackend = "auto",
    [switch]$MirrorInput,
    [string]$PoseModel = "",
    [switch]$FreeBase,
    [switch]$Headless,
    [int]$MaxFrames = 0,
    [string]$PythonExe = "",

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$AdditionalArguments = @()
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$setupScript = Join-Path $PSScriptRoot "setup_windows.ps1"
$env:MPLCONFIGDIR = Join-Path $projectRoot ".cache\matplotlib"
$env:PIP_CACHE_DIR = Join-Path $projectRoot ".cache\pip"
New-Item -ItemType Directory -Force -Path $env:MPLCONFIGDIR, $env:PIP_CACHE_DIR | Out-Null

$needsSetup = -not (Test-Path -LiteralPath $venvPython -PathType Leaf)
if (-not $needsSetup) {
    & $venvPython -c "import cv2, mediapipe, mujoco, robot_human_interface" 2>$null
    $needsSetup = $LASTEXITCODE -ne 0
}

if ($needsSetup) {
    Write-Host "The project environment is not ready; running Windows setup..."
    if ($PythonExe) {
        & $setupScript -PythonExe $PythonExe
    }
    else {
        & $setupScript
    }
}

if ($Source -eq "replay" -and [string]::IsNullOrWhiteSpace($ReplayPath)) {
    throw "-ReplayPath is required when -Source replay is selected."
}
if ($CameraIndex -lt 0) {
    throw "-CameraIndex must be non-negative."
}
if ($MaxFrames -lt 0) {
    throw "-MaxFrames must be non-negative."
}

$teleopArguments = @(
    "-m", "robot_human_interface.app.teleop",
    "--source", $Source,
    "--camera-index", "$CameraIndex"
    "--camera-backend", $CameraBackend
)

if ($ReplayPath) {
    $teleopArguments += @("--replay-path", $ReplayPath)
}
if ($PoseModel) {
    $teleopArguments += @("--pose-model", $PoseModel)
}
if ($MirrorInput) {
    $teleopArguments += "--mirror-input"
}
if ($FreeBase) {
    $teleopArguments += "--free-base"
}
if ($Headless) {
    $teleopArguments += "--headless"
}
if ($MaxFrames -gt 0) {
    $teleopArguments += @("--max-frames", "$MaxFrames")
}
if ($AdditionalArguments) {
    $teleopArguments += $AdditionalArguments
}

& $venvPython @teleopArguments
if ($LASTEXITCODE -ne 0) {
    throw "Camera teleoperation exited with code $LASTEXITCODE."
}
