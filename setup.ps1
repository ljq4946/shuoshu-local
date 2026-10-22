param([switch]$Models, [switch]$CoreOnly)
$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'scripts/Setup-Environment.ps1') @PSBoundParameters
