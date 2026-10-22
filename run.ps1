param([Parameter(ValueFromRemainingArguments=$true)][string[]]$CliArgs)
$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'scripts/Run-Storyteller.ps1') -CliArgs $CliArgs
exit $LASTEXITCODE
