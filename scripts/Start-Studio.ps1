$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Common.ps1')
$root = Get-StorytellerProjectRoot
$python = Join-Path $root '.runtime/core/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw '请先在此目录运行：powershell -ExecutionPolicy Bypass -File setup.ps1 -Models' }
Initialize-StorytellerPython
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
Add-Type -AssemblyName System.Windows.Forms
$lastFile = Join-Path $root '.runtime/last-project.txt'
Write-Host '说书人 · 离线有声书'
Write-Host '1. 导入新书，检查章节并试读'
Write-Host '2. 继续上一次制作（本地面板）'
Write-Host '3. 选择已有项目（本地面板）'
Write-Host '4. 打开上一次制作与复听面板'
$choice = Read-Host '请选择（默认 1）'
Push-Location -LiteralPath $root
try {
    if ($choice -eq '2' -or $choice -eq '4') {
        if (-not (Test-Path -LiteralPath $lastFile)) { throw '尚无上次项目，请选择新建或选择已有项目。' }
        $bookProject = (Get-Content -LiteralPath $lastFile -Encoding UTF8 -Raw).Trim()
    } elseif ($choice -eq '3') {
        $folder = New-Object System.Windows.Forms.FolderBrowserDialog
        $folder.Description = '选择包含 project.json 的项目目录'
        if ($folder.ShowDialog() -ne 'OK') { exit 0 }
        $bookProject = $folder.SelectedPath
    } else {
        $book = New-Object System.Windows.Forms.OpenFileDialog
        $book.Title = '选择 TXT 书稿'
        $book.Filter = 'TXT 书稿|*.txt'
        if ($book.ShowDialog() -ne 'OK') { exit 0 }
        $voice = New-Object System.Windows.Forms.OpenFileDialog
        $voice.Title = '选择参考声音（建议 10–15 秒，默认取开头最多 15 秒）'
        $voice.Filter = '音频|*.wav;*.mp3;*.flac;*.m4a;*.ogg'
        if ($voice.ShowDialog() -ne 'OK') { exit 0 }
        $name = [System.IO.Path]::GetFileNameWithoutExtension($book.FileName)
        $safeName = $name -replace '[<>:"/\\|?*]', '_'
        $bookProject = Join-Path $root ('books/' + $safeName + '_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
        & $python -m storyteller init $book.FileName --voice $voice.FileName --out $bookProject
        if ($LASTEXITCODE -ne 0) { throw '导入失败，请检查上方提示。' }
    }
    Set-Content -LiteralPath $lastFile -Value $bookProject -Encoding UTF8
    Write-Host '正在打开本地制作面板。新书先检查目录、生成试读，再确认开始全书。'
    & $python -m storyteller studio $bookProject
    if ($LASTEXITCODE -notin @(0, 130)) { Write-Host '面板尚未正常结束，请检查上方提示。' }
} finally { Pop-Location }
