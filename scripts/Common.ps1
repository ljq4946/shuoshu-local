# Shared setup for the project workflows. Dot-source from scripts/*.ps1.
$script:StorytellerProjectRoot = Split-Path -Parent $PSScriptRoot

function Get-StorytellerProjectRoot {
    return $script:StorytellerProjectRoot
}

function Initialize-StorytellerPython {
    $env:PYTHONUTF8 = '1'
    $env:PYTHONIOENCODING = 'utf-8'
}

function Get-StorytellerCorePython {
    param([switch]$AllowManagedFallback)
    $projectRoot = Get-StorytellerProjectRoot
    $pythonPath = Join-Path $projectRoot '.runtime/core/Scripts/python.exe'
    if (Test-Path -LiteralPath $pythonPath) { return $pythonPath }
    if ($AllowManagedFallback) {
        $candidates = @(Get-ChildItem -LiteralPath (Join-Path $projectRoot '.runtime/python') -Directory -ErrorAction SilentlyContinue)
        foreach ($candidate in $candidates) {
            $possible = Join-Path $candidate.FullName 'python.exe'
            if (Test-Path -LiteralPath $possible) { return $possible }
        }
    }
    throw 'Python is not ready. Run setup.ps1 first.'
}

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $Program $($Arguments -join ' ')" }
}
