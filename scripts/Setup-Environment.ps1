param([switch]$Models, [switch]$CoreOnly)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Common.ps1')
$root = Get-StorytellerProjectRoot
$env:UV_CACHE_DIR = Join-Path $root '.runtime/cache/uv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $root '.runtime/python'
$env:UV_PYTHON_BIN_DIR = Join-Path $root '.runtime/bin'
Initialize-StorytellerPython
$env:HF_HOME = Join-Path $root '.runtime/cache/huggingface'
$env:NLTK_DATA = Join-Path $root '.runtime/nltk_data'
$env:GIT_LFS_SKIP_SMUDGE = '1'
Push-Location -LiteralPath $root
try {
    New-Item -ItemType Directory -Force -Path '.tools', '.runtime/bin' | Out-Null
    $uv = Join-Path $root '.tools/bootstrap/bin/uv.exe'
    if (-not (Test-Path -LiteralPath $uv)) {
        $uv = Join-Path $root '.tools/uv/uv.exe'
        if (-not (Test-Path -LiteralPath $uv)) {
            $archive = Join-Path $root '.tools/uv.zip'
            Invoke-WebRequest -Uri 'https://github.com/astral-sh/uv/releases/download/0.13.4/uv-x86_64-pc-windows-msvc.zip' -OutFile $archive -UseBasicParsing
            Expand-Archive -LiteralPath $archive -DestinationPath (Join-Path $root '.tools/uv') -Force
        }
    }
    Invoke-Checked $uv @('python', 'install', '3.11.17', '--no-bin', '--no-registry')
    $python = (& $uv python find --managed-python --no-project 3.11.17).Trim()
    if ($LASTEXITCODE -ne 0) { throw 'Managed Python was not found.' }
    if (-not (Test-Path -LiteralPath '.runtime/core/Scripts/python.exe')) {
        Invoke-Checked $uv @('venv', '.runtime/core', '--python', $python)
    }
    $core = Join-Path $root '.runtime/core/Scripts/python.exe'
    Invoke-Checked $uv @('pip', 'install', '--python', $core, '-r', 'requirements/core.txt')
    $ffmpeg = (& $core -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())').Trim()
    Copy-Item -LiteralPath $ffmpeg -Destination '.runtime/bin/ffmpeg.exe' -Force
    if (-not $CoreOnly) {
        $revision = 'd9e41aac89fd00b3d71497fddb287b7f24613712'
        if (-not (Test-Path -LiteralPath '.runtime/index-tts/.git')) {
            Invoke-Checked 'git' @('-c', 'core.longpaths=true', 'clone', 'https://github.com/index-tts/index-tts.git', '.runtime/index-tts')
        }
        $actual = (& git -C .runtime/index-tts rev-parse HEAD).Trim()
        if ($actual -ne $revision) {
            $changes = & git -C .runtime/index-tts status --porcelain
            if ($changes) { throw 'IndexTTS contains local changes. Preserve them before changing its version.' }
            Invoke-Checked 'git' @('-C', '.runtime/index-tts', 'fetch', 'origin', $revision)
            Invoke-Checked 'git' @('-C', '.runtime/index-tts', 'checkout', '--detach', $revision)
        }
        Invoke-Checked $core @('scripts/bootstrap/prepare_torch.py')
        if (-not (Test-Path -LiteralPath '.runtime/index-tts/.venv/Scripts/python.exe')) {
            Invoke-Checked $uv @('venv', '.runtime/index-tts/.venv', '--python', $python)
        }
        Invoke-Checked $uv @('pip', 'install', '--python', '.runtime/index-tts/.venv/Scripts/python.exe', '--no-deps', '--no-index', '--find-links', '.runtime/downloads', 'torch==2.8.0+cu128')
        Invoke-Checked $uv @('sync', '--project', '.runtime/index-tts', '--python', $python, '--frozen', '--no-default-groups')
        if (-not (Test-Path -LiteralPath '.runtime/asr/Scripts/python.exe')) {
            Invoke-Checked $uv @('venv', '.runtime/asr', '--python', $python)
        }
        $asr = Join-Path $root '.runtime/asr/Scripts/python.exe'
        Invoke-Checked $uv @('pip', 'install', '--python', $asr, '--no-deps', '--no-index', '--find-links', '.runtime/downloads', 'torch==2.8.0+cu128')
        Invoke-Checked $uv @('pip', 'install', '--python', $asr, 'torch==2.8.0', 'torchaudio==2.8.0', '--index-url', 'https://download.pytorch.org/whl/cu128')
        Invoke-Checked $uv @('pip', 'install', '--python', $asr, '-r', 'requirements/asr.txt')
        & $uv pip freeze --python $asr | Set-Content -LiteralPath '.runtime/asr-installed.txt' -Encoding UTF8
        New-Item -ItemType Directory -Force -Path $env:NLTK_DATA | Out-Null
        Invoke-Checked (Join-Path $root '.runtime/index-tts/.venv/Scripts/python.exe') @('-c', 'import nltk, os; [nltk.download(x, download_dir=os.environ["NLTK_DATA"], raise_on_error=True) for x in ("averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "cmudict", "punkt", "punkt_tab")]')
    }
    if ($Models) { Invoke-Checked $core @('scripts/bootstrap/prepare_models.py') }
    Write-Host 'Setup stage finished. Run: .\run.ps1 doctor --load-models'
} finally { Pop-Location }
