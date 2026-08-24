[CmdletBinding()]
param(
    [string]$PythonExe = "",
    [switch]$NoDev
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$lockFile = Join-Path $projectRoot "uv.lock"
$uvCommand = Get-Command "uv" -ErrorAction SilentlyContinue
if (-not $uvCommand) {
    $localUv = Join-Path $projectRoot ".venv\Scripts\uv.exe"
    if (Test-Path -LiteralPath $localUv -PathType Leaf) {
        $uvCommand = Get-Command $localUv -ErrorAction Stop
    }
}

if (-not $uvCommand) {
    throw @"
uv is required but was not found on PATH.
Install it with:
  winget install --id astral-sh.uv -e
Open a new PowerShell window, then rerun this script.
"@
}

$minimumUvVersion = [version]"0.12.5"
$uvVersionOutput = (& $uvCommand.Source --version 2>&1 | Select-Object -First 1)
$uvVersionMatch = [regex]::Match("$uvVersionOutput", '^uv\s+(\d+\.\d+\.\d+)')
if ($LASTEXITCODE -ne 0 -or -not $uvVersionMatch.Success) {
    throw "Could not determine the installed uv version from: $uvVersionOutput"
}
$uvVersion = [version]$uvVersionMatch.Groups[1].Value
if ($uvVersion -lt $minimumUvVersion) {
    throw @"
uv $uvVersion is too old; this project requires uv $minimumUvVersion or newer.
Upgrade it with:
  winget upgrade --id astral-sh.uv -e
Then open a new PowerShell window and rerun this script.
"@
}

if (-not (Test-Path -LiteralPath $lockFile -PathType Leaf)) {
    throw @"
The universal dependency lock is missing: $lockFile
Run 'uv lock' on a maintainer machine and commit uv.lock before setup.
"@
}

$pythonRequest = "3.12"
if (-not [string]::IsNullOrWhiteSpace($PythonExe)) {
    $pythonRequest = [System.IO.Path]::GetFullPath($PythonExe)
    if (-not (Test-Path -LiteralPath $pythonRequest -PathType Leaf)) {
        throw "Python executable does not exist: $pythonRequest"
    }
}

$syncArguments = @("sync", "--locked", "--python", $pythonRequest)
if ($NoDev) {
    $syncArguments += "--no-dev"
}
else {
    $syncArguments += @("--group", "dev")
}

Push-Location $projectRoot
try {
    Write-Host "Synchronizing the locked CPython 3.12 environment..."
    & $uvCommand.Source @syncArguments
    if ($LASTEXITCODE -ne 0) {
        throw "uv sync failed with exit code $LASTEXITCODE."
    }

    Write-Host "Checking runtime imports..."
    & $uvCommand.Source run --locked --no-sync python -c `
        "import PyQt6, mediapipe, mujoco, websocket, robot_human_interface; assert callable(websocket.create_connection); print(f'Environment ready: MuJoCo {mujoco.__version__}, MediaPipe {mediapipe.__version__}')"
    if ($LASTEXITCODE -ne 0) {
        throw "The environment was installed but its import smoke test failed."
    }
}
finally {
    Pop-Location
}

Write-Host "Windows setup complete."
Write-Host "Run the operator GUI: & `"$PSScriptRoot\run_gui.ps1`""
Write-Host "Run the legacy CLI: & `"$PSScriptRoot\run_camera_teleop.ps1`" -Source synthetic"
