[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$AdditionalArguments = @()
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$venvDir = Join-Path $projectRoot ".venv"
$guiEntrypoint = Join-Path $venvDir "Scripts\humanoid-interface.exe"
$uvCommand = Get-Command "uv" -ErrorAction SilentlyContinue
if (-not $uvCommand) {
    $uvCommand = Get-Command (Join-Path $projectRoot ".venv\Scripts\uv.exe") -ErrorAction SilentlyContinue
}
$needsSetup = -not $uvCommand `
    -or -not (Test-Path -LiteralPath $venvDir -PathType Container) `
    -or -not (Test-Path -LiteralPath $guiEntrypoint -PathType Leaf)
if (-not $needsSetup) {
    Push-Location $projectRoot
    try {
        & $uvCommand.Source run --locked --no-sync python -c `
            "import PyQt6, robot_human_interface; from robot_human_interface.gui.app import main; assert callable(main)" 2>$null
        $needsSetup = $LASTEXITCODE -ne 0
    }
    finally {
        Pop-Location
    }
}
if ($needsSetup) {
    & (Join-Path $PSScriptRoot "setup_windows.ps1")
    $uvCommand = Get-Command "uv" -ErrorAction SilentlyContinue
    if (-not $uvCommand) {
        $uvCommand = Get-Command (Join-Path $projectRoot ".venv\Scripts\uv.exe") -ErrorAction Stop
    }
}

Push-Location $projectRoot
try {
    & $uvCommand.Source run --locked --no-sync humanoid-interface @AdditionalArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Humanoid Interface exited with code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
