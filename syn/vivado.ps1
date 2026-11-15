# TinyTPU Vivado helper for Windows PowerShell.
#
#   .\syn\vivado.ps1                       # build loopback_top
#   .\syn\vivado.ps1 -Top loopback_top     # build a given top
#   .\syn\vivado.ps1 -Program              # program the board with the last build of -Top
#   .\syn\vivado.ps1 -Ooc -Top pe          # out-of-context synthesis + resource checks (syn/ooc.tcl)
#
# A full build of tinytpu_top also checks the AGENTS.md resource ceilings (in syn/build.tcl) and
# then compares <top>_metrics.json with syn/baseline.json (python -m tinytpu.baseline).
#
# Builds go to %LOCALAPPDATA%\tinytpu\vivado\<top> (OOC runs: ...\<top>_ooc), a local disk,
# never the repo. Vivado version and path come from tools/versions.env (VIVADO_VERSION).

param(
    [string]$Top = "loopback_top",
    [switch]$Program,
    [switch]$Ooc
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$version = (Select-String -Path (Join-Path $repo "tools\versions.env") -Pattern '^VIVADO_VERSION=(.+)$').Matches[0].Groups[1].Value
$vivado = "C:\AMDDesignTools\$version\Vivado\bin\vivado.bat"
if (-not (Test-Path $vivado)) { throw "Vivado $version not found at $vivado" }

$out = Join-Path $env:LOCALAPPDATA ("tinytpu\vivado\$Top" + $(if ($Ooc) { "_ooc" } else { "" }))
New-Item -ItemType Directory -Force -Path $out | Out-Null
Push-Location $out
try {
    if ($Ooc) {
        & $vivado -mode batch -nojournal -log (Join-Path $out "vivado.log") -source (Join-Path $repo "syn\ooc.tcl") -tclargs $Top $out
    } elseif ($Program) {
        & $vivado -mode batch -nojournal -nolog -source (Join-Path $repo "syn\program.tcl") -tclargs (Join-Path $out "$Top.bit")
    } else {
        & $vivado -mode batch -nojournal -log (Join-Path $out "vivado.log") -source (Join-Path $repo "syn\build.tcl") -tclargs $Top $out
    }
    if ($LASTEXITCODE -ne 0) { throw "Vivado failed (exit $LASTEXITCODE); see $out\vivado.log" }
} finally {
    Pop-Location
}

# After a full build of tinytpu_top: compare its metrics with syn/baseline.json (A5.2), once a
# baseline exists. uv must use the Windows host venv (UV_PROJECT_ENVIRONMENT, tools/setup.md),
# never the repo's .venv, which belongs to WSL.
$metrics = Join-Path $out "${Top}_metrics.json"
$baseline = Join-Path $repo "syn\baseline.json"
if (-not $Ooc -and -not $Program -and $Top -eq "tinytpu_top" -and (Test-Path $metrics)) {
    if (-not (Test-Path $baseline)) {
        Write-Host "No syn\baseline.json yet: baseline check skipped (metrics: $metrics)"
    } elseif (-not $env:UV_PROJECT_ENVIRONMENT) {
        Write-Host "UV_PROJECT_ENVIRONMENT is not set: baseline check skipped. Run it with:"
        Write-Host "  uv run --no-default-groups python -m tinytpu.baseline `"$metrics`""
    } else {
        & uv run --no-default-groups --project $repo python -m tinytpu.baseline $metrics --baseline $baseline
        if ($LASTEXITCODE -ne 0) { throw "baseline check failed (see the table above)" }
    }
}
