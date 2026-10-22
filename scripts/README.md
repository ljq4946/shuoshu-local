# 脚本索引

日常使用根目录入口；这里保存实际实现。修改工作流时修改本目录对应脚本。

| 根目录入口 | 实际实现 | 用途 |
|---|---|---|
| `start.cmd` → `start.ps1` | [Start-Studio.ps1](Start-Studio.ps1) | 选择书稿、参考声音或已有项目，启动面板 |
| `run.ps1` | [Run-Storyteller.ps1](Run-Storyteller.ps1) | 调用 `python -m storyteller`，传递参数和退出码 |
| `setup.ps1` | [Setup-Environment.ps1](Setup-Environment.ps1) | 安装三个隔离环境、FFmpeg 和模型 |
| `train_voice.ps1` | [Train-Voice.ps1](Train-Voice.ps1) | 准备、核验、提取、训练及查看状态 |

[Common.ps1](Common.ps1) 提供项目根目录定位、UTF-8 环境、控制端 Python 选择和外部命令失败检查。各工作流进入项目根目录执行，并在结束时恢复原工作目录。

## 安装辅助工具

| 工具 | 用途 |
|---|---|
| [bootstrap/prepare_torch.py](bootstrap/prepare_torch.py) | 下载固定版本及 SHA-256 的 CUDA PyTorch wheel |
| [bootstrap/prepare_models.py](bootstrap/prepare_models.py) | 准备模型、记录固定版本和资产哈希 |
| [bootstrap/ranged_download.py](bootstrap/ranged_download.py) | 分块续传与下载校验，被前两项调用 |

安装工具由 `setup.ps1` 按顺序调用。依赖版本统一维护在 [requirements/](../requirements/README.md)。PowerShell 脚本保存为带 BOM 的 UTF-8，兼容 Windows PowerShell 5.1 的中文文本。

## 常用命令

在项目根目录执行：

```powershell
.\run.ps1 --help
.\run.ps1 doctor
.\train_voice.ps1 -Stage Status
.\train_voice.ps1 -Stage Verify
```

安装使用 `powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1 -Models`；仅准备控制端和 FFmpeg 时使用 `-CoreOnly`。完整制作说明见 [README](../README.md)，训练参数见 [训练目录说明](../training/README.md)。
