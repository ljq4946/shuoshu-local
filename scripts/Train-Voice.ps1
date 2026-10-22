param(
    [ValidateSet('All','Prepare','Extract','Train','Verify','Status')]
    [string]$Stage = 'All',
    [string]$Dataset = 'training/qiong1_20261022'
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Common.ps1')
$projectRoot = Get-StorytellerProjectRoot
$trainingPython = Join-Path $projectRoot '.runtime/index-tts/.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $trainingPython)) { throw 'Run setup.ps1 first.' }
if (-not [IO.Path]::IsPathRooted($Dataset)) { $Dataset = Join-Path $projectRoot $Dataset }
$trainingRun = Join-Path $Dataset 'runs/lora_rank16'
Initialize-StorytellerPython
$env:PYTHONUNBUFFERED = '1'

function Invoke-VoiceModule([string]$ModuleName, [string[]]$ModuleArgs) {
    Invoke-Checked $trainingPython (@('-m', $ModuleName) + $ModuleArgs)
}

Push-Location -LiteralPath $projectRoot
try {
    if ($Stage -eq 'Status') {
        foreach ($statusPath in @((Join-Path $Dataset 'extraction_status.json'), (Join-Path $trainingRun 'status.json'))) {
            if (Test-Path -LiteralPath $statusPath) { Get-Content -Encoding UTF8 -LiteralPath $statusPath }
        }
        return
    }
    if ($Stage -eq 'Prepare' -or ($Stage -eq 'All' -and -not (Test-Path -LiteralPath (Join-Path $Dataset 'dataset_report.json')))) {
        Invoke-VoiceModule 'storyteller.finetune' @('prepare','--dataset',$Dataset)
    }
    if ($Stage -in @('All','Verify')) {
        Invoke-VoiceModule 'storyteller.finetune' @('verify','--dataset',$Dataset)
    }
    if ($Stage -in @('All','Extract')) {
        Invoke-VoiceModule 'storyteller.finetune' @('extract','--dataset',$Dataset)
    }
    if ($Stage -in @('All','Train') -and -not (Test-Path -LiteralPath (Join-Path $trainingRun 'summary.json'))) {
        $voiceArguments = @('train','--dataset',$Dataset,'--run',$trainingRun)
        if (Test-Path -LiteralPath (Join-Path $trainingRun 'last.pt')) { $voiceArguments += '--resume' }
        Invoke-VoiceModule 'storyteller.finetune' $voiceArguments
    }
} finally { Pop-Location }
