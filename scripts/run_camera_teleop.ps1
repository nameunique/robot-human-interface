[CmdletBinding()]
param(
    [ValidateSet("camera", "mp4", "synthetic", "replay")]
    [string]$Source = "camera",

    [ValidateSet("jumping-jacks", "slow-balance")]
    [string]$DemoVideo = "slow-balance",

    [Alias("VideoPath")]
    [string]$ReplayPath = "",
    [switch]$LoopReplay,
    [int]$CameraIndex = 0,
    [ValidateSet("auto", "dshow", "msmf", "v4l2", "avfoundation", "gstreamer")]
    [string]$CameraBackend = "auto",
    [switch]$MirrorInput,
    [string]$PoseModel = "",
    [switch]$FreeBase,
    [ValidateSet("visual", "joints")]
    [string]$ViewerMode = "visual",
    [string]$RobotWebSocketUrl = "",
    [double]$RobotWebSocketTimeoutSeconds = 0.5,
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
    & $venvPython -c "import cv2, mediapipe, mujoco, websocket, robot_human_interface; assert callable(websocket.create_connection)" 2>$null
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
    throw "-VideoPath is required when -Source replay is selected."
}
if ($CameraIndex -lt 0) {
    throw "-CameraIndex must be non-negative."
}
if ($MaxFrames -lt 0) {
    throw "-MaxFrames must be non-negative."
}
if ($RobotWebSocketTimeoutSeconds -le 0.0) {
    throw "-RobotWebSocketTimeoutSeconds must be positive."
}
if ($RobotWebSocketUrl -and -not $FreeBase) {
    throw "-RobotWebSocketUrl requires -FreeBase so the motor-angle safety layer is active."
}

$teleopArguments = @(
    "-m", "robot_human_interface.app.teleop",
    "--source", $Source,
    "--demo-video", $DemoVideo,
    "--camera-index", "$CameraIndex"
    "--camera-backend", $CameraBackend
    "--viewer-mode", $ViewerMode
)

if ($ReplayPath) {
    if (-not [System.IO.Path]::IsPathRooted($ReplayPath)) {
        $ReplayPath = Join-Path $projectRoot $ReplayPath
    }
    $ReplayPath = [System.IO.Path]::GetFullPath($ReplayPath)
    if (-not (Test-Path -LiteralPath $ReplayPath -PathType Leaf)) {
        throw "Video file does not exist: $ReplayPath"
    }
    $teleopArguments += @("--video-path", $ReplayPath)
}
if ($LoopReplay) {
    $teleopArguments += "--loop-replay"
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
if ($RobotWebSocketUrl) {
    $teleopArguments += @(
        "--robot-websocket-url", $RobotWebSocketUrl,
        "--robot-websocket-timeout-s", "$RobotWebSocketTimeoutSeconds"
    )
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
