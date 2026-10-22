param(
    [ValidateSet('All','Prepare','Extract','Train','Verify','Status')]
    [string]$Stage = 'All',
    [string]$Dataset = 'training/qiong1_20261022'
)
$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'scripts/Train-Voice.ps1') @PSBoundParameters
