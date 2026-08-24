[CmdletBinding()]
param(
    [ValidateSet("camera", "mp4", "synthetic", "replay")]
    [string]$Source = "camera",

    [ValidateSet("jumping-jacks", "slow-balance")]
    [string]$DemoVideo = "slow-balance",

    [Alias("VideoPath")]
    [string]$ReplayPath = "",
    [string]$CalibrationVideo = "",
    [Nullable[int]]$CalibrationFrame = $null,
    [switch]$LoopReplay,
    [int]$CameraIndex = 0,
    [ValidateSet("auto", "dshow", "msmf", "v4l2", "avfoundation", "gstreamer")]
    [string]$CameraBackend = "auto",
    [switch]$MirrorInput,
    [string]$PoseModel = "",
    [switch]$FreeBase,
    [switch]$FixedBase,
    [ValidateSet("ik", "geometric")]
    [string]$Retargeting = "ik",
    [ValidateSet("visual", "joints")]
    [string]$ViewerMode = "visual",
    [string]$RobotWebSocketUrl = "",
    [double]$RobotWebSocketTimeoutSeconds = 0.5,
    [switch]$Headless,
    [int]$MaxFrames = 0,
    [double]$SettleSeconds = 0.0,
    [double]$SettleTimeoutSeconds = 20.0,
    [string]$PythonExe = "",

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$AdditionalArguments = @()
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$setupScript = Join-Path $PSScriptRoot "setup_windows.ps1"
$uvCommand = Get-Command "uv" -ErrorAction SilentlyContinue
if (-not $uvCommand) {
    $uvCommand = Get-Command (Join-Path $projectRoot ".venv\Scripts\uv.exe") -ErrorAction SilentlyContinue
}

$needsSetup = -not $uvCommand -or -not (Test-Path -LiteralPath (Join-Path $projectRoot ".venv"))
if (-not $needsSetup) {
    Push-Location $projectRoot
    try {
        & $uvCommand.Source run --locked --no-sync python -c `
            "import cv2, mediapipe, mujoco, websocket, robot_human_interface; assert callable(websocket.create_connection)" 2>$null
        $needsSetup = $LASTEXITCODE -ne 0
    }
    finally {
        Pop-Location
    }
}

if ($needsSetup) {
    Write-Host "The project environment is not ready; running Windows setup..."
    if ($PythonExe) {
        & $setupScript -PythonExe $PythonExe
    }
    else {
        & $setupScript
    }
    $uvCommand = Get-Command "uv" -ErrorAction SilentlyContinue
    if (-not $uvCommand) {
        $uvCommand = Get-Command (Join-Path $projectRoot ".venv\Scripts\uv.exe") -ErrorAction Stop
    }
}

if ($Source -eq "replay" -and [string]::IsNullOrWhiteSpace($ReplayPath)) {
    throw "-VideoPath is required when -Source replay is selected."
}
$hasCalibrationVideo = -not [string]::IsNullOrWhiteSpace($CalibrationVideo)
$hasCalibrationFrame = $null -ne $CalibrationFrame
if ($hasCalibrationFrame -and $CalibrationFrame -lt 0) {
    throw "-CalibrationFrame must be non-negative."
}
if ($hasCalibrationVideo -ne $hasCalibrationFrame) {
    throw "-CalibrationVideo and a non-negative -CalibrationFrame must be supplied together."
}
if ($hasCalibrationVideo -and $Source -notin @("mp4", "replay")) {
    throw "Controlled replay calibration is supported only with -Source mp4 or replay."
}
if ($CameraIndex -lt 0) {
    throw "-CameraIndex must be non-negative."
}
if ($MaxFrames -lt 0) {
    throw "-MaxFrames must be non-negative."
}
if ($SettleSeconds -lt 0.0) {
    throw "-SettleSeconds must be non-negative."
}
if ($SettleTimeoutSeconds -le 0.0 -or $SettleSeconds -gt $SettleTimeoutSeconds) {
    throw "-SettleTimeoutSeconds must be positive and at least -SettleSeconds."
}
if ($FreeBase -and $FixedBase) {
    throw "-FreeBase and -FixedBase are mutually exclusive."
}
if ($RobotWebSocketTimeoutSeconds -le 0.0) {
    throw "-RobotWebSocketTimeoutSeconds must be positive."
}
$effectiveFreeBase = -not $FixedBase
if ($RobotWebSocketUrl -and -not $effectiveFreeBase) {
    throw "-RobotWebSocketUrl requires free-base mode and cannot be used with -FixedBase."
}

$teleopArguments = @(
    "-m", "robot_human_interface.app.teleop",
    "--source", $Source,
    "--demo-video", $DemoVideo,
    "--camera-index", "$CameraIndex"
    "--camera-backend", $CameraBackend
    "--retargeting", $Retargeting
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
if ($hasCalibrationVideo) {
    if (-not [System.IO.Path]::IsPathRooted($CalibrationVideo)) {
        $CalibrationVideo = Join-Path $projectRoot $CalibrationVideo
    }
    $CalibrationVideo = [System.IO.Path]::GetFullPath($CalibrationVideo)
    if (-not (Test-Path -LiteralPath $CalibrationVideo -PathType Leaf)) {
        throw "Calibration video does not exist: $CalibrationVideo"
    }
    $teleopArguments += @(
        "--calibration-video", $CalibrationVideo,
        "--calibration-frame", "$CalibrationFrame"
    )
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
if ($FixedBase) {
    $teleopArguments += "--fixed-base"
}
elseif ($FreeBase) {
    # Backward-compatible explicit spelling; free-base is already the default.
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
if ($SettleSeconds -gt 0.0) {
    $teleopArguments += @(
        "--settle-seconds", "$SettleSeconds",
        "--settle-timeout-s", "$SettleTimeoutSeconds"
    )
}
if ($AdditionalArguments) {
    $teleopArguments += $AdditionalArguments
}

Push-Location $projectRoot
try {
    & $uvCommand.Source run --locked --no-sync python @teleopArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Camera teleoperation exited with code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
