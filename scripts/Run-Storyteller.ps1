param([Parameter(ValueFromRemainingArguments=$true)][string[]]$CliArgs)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Common.ps1')
$projectRoot = Get-StorytellerProjectRoot
$pythonPath = Get-StorytellerCorePython -AllowManagedFallback
Initialize-StorytellerPython
Push-Location -LiteralPath $projectRoot
try {
    & $pythonPath -m storyteller @CliArgs
    $result = $LASTEXITCODE
} finally { Pop-Location }
exit $result
