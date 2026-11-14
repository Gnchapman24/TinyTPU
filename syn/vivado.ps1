# TinyTPU Vivado helper for Windows PowerShell.
#
#   .\syn\vivado.ps1                       # build loopback_top
#   .\syn\vivado.ps1 -Top loopback_top     # build a given top
#   .\syn\vivado.ps1 -Program              # program the board with the last build of -Top
#
# Builds go to %LOCALAPPDATA%\tinytpu\vivado\<top> (a local disk, never the repo).
# Vivado version and path come from tools/versions.env (VIVADO_VERSION).

param(
    [string]$Top = "loopback_top",
    [switch]$Program
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$version = (Select-String -Path (Join-Path $repo "tools\versions.env") -Pattern '^VIVADO_VERSION=(.+)$').Matches[0].Groups[1].Value
$vivado = "C:\AMDDesignTools\$version\Vivado\bin\vivado.bat"
if (-not (Test-Path $vivado)) { throw "Vivado $version not found at $vivado" }

$out = Join-Path $env:LOCALAPPDATA "tinytpu\vivado\$Top"
New-Item -ItemType Directory -Force -Path $out | Out-Null
Push-Location $out
try {
    if ($Program) {
        & $vivado -mode batch -nojournal -nolog -source (Join-Path $repo "syn\program.tcl") -tclargs (Join-Path $out "$Top.bit")
    } else {
        & $vivado -mode batch -nojournal -log (Join-Path $out "vivado.log") -source (Join-Path $repo "syn\build.tcl") -tclargs $Top $out
    }
    if ($LASTEXITCODE -ne 0) { throw "Vivado failed (exit $LASTEXITCODE); see $out\vivado.log" }
} finally {
    Pop-Location
}
