[CmdletBinding()]
param(
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$venvDir = Join-Path $projectRoot ".venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"
$lockFile = Join-Path $projectRoot "requirements.lock.txt"
$env:MPLCONFIGDIR = Join-Path $projectRoot ".cache\matplotlib"
$env:PIP_CACHE_DIR = Join-Path $projectRoot ".cache\pip"
New-Item -ItemType Directory -Force -Path $env:MPLCONFIGDIR, $env:PIP_CACHE_DIR | Out-Null

function Get-PythonMinorVersion {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [string[]]$PrefixArguments = @()
    )

    try {
        $arguments = @($PrefixArguments) + @(
            "-c",
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        )
        $version = (& $Executable @arguments 2>$null | Select-Object -Last 1)
        if ($LASTEXITCODE -ne 0) {
            return $null
        }
        return "$version".Trim()
    }
    catch {
        return $null
    }
}

function Find-Python312 {
    $candidates = [System.Collections.Generic.List[object]]::new()

    if ($PythonExe) {
        $candidates.Add([pscustomobject]@{
            Executable = $PythonExe
            PrefixArguments = @()
            Description = "-PythonExe"
        })
    }

    $pyLauncher = Get-Command "py" -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        $candidates.Add([pscustomobject]@{
            Executable = $pyLauncher.Source
            PrefixArguments = @("-3.12")
            Description = "Python Launcher (py -3.12)"
        })
    }

    foreach ($name in @("python3.12", "python")) {
        $command = Get-Command $name -ErrorAction SilentlyContinue
        if ($command) {
            $candidates.Add([pscustomobject]@{
                Executable = $command.Source
                PrefixArguments = @()
                Description = $name
            })
        }
    }

    foreach ($candidate in $candidates) {
        $version = Get-PythonMinorVersion `
            -Executable $candidate.Executable `
            -PrefixArguments $candidate.PrefixArguments
        if ($version -eq "3.12") {
            return $candidate
        }
    }

    throw @"
Python 3.12 was not found. Install native 64-bit Python 3.12 for Windows or
rerun this script with -PythonExe C:\path\to\python.exe.
"@
}

if (-not (Test-Path -LiteralPath $lockFile -PathType Leaf)) {
    throw "Dependency lock file is missing: $lockFile"
}

if (Test-Path -LiteralPath $venvPython -PathType Leaf) {
    $venvVersion = Get-PythonMinorVersion -Executable $venvPython
    if ($venvVersion -ne "3.12") {
        throw @"
The existing project environment uses Python $venvVersion instead of 3.12:
$venvDir
Move or remove that directory, then rerun this setup script.
"@
    }
    Write-Host "Using existing Python 3.12 environment: $venvDir"
}
else {
    if (Test-Path -LiteralPath $venvDir) {
        throw @"
$venvDir exists but is not a valid Windows virtual environment.
Move or remove it, then rerun this setup script.
"@
    }

    $python = Find-Python312
    Write-Host "Creating .venv with $($python.Description)..."
    $venvArguments = @($python.PrefixArguments) + @("-m", "venv", $venvDir)
    & $python.Executable @venvArguments
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $venvPython)) {
        throw "Failed to create the project virtual environment."
    }
}

Write-Host "Installing locked dependencies..."
& $venvPython -m pip install --disable-pip-version-check -r $lockFile
if ($LASTEXITCODE -ne 0) {
    throw "Installing requirements.lock.txt failed with exit code $LASTEXITCODE."
}

Write-Host "Installing this project in editable mode without changing the lock..."
& $venvPython -m pip install `
    --disable-pip-version-check `
    --no-deps `
    --no-build-isolation `
    -e $projectRoot
if ($LASTEXITCODE -ne 0) {
    throw "Installing robot-human-interface failed with exit code $LASTEXITCODE."
}

& $venvPython -c "import mediapipe, mujoco, websocket, robot_human_interface; assert callable(websocket.create_connection); print(f'Environment ready: MuJoCo {mujoco.__version__}')"
if ($LASTEXITCODE -ne 0) {
    throw "The environment was installed but its import smoke test failed."
}

Write-Host "Windows setup complete."
Write-Host "Run: .\scripts\run_camera_teleop.ps1"
