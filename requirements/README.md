# 环境与依赖

| 环境 | 清单 | 说明 |
|---|---|---|
| 控制端 `.runtime/core` | [core.txt](core.txt) | FFmpeg 和模型安装辅助工具；业务控制代码使用 Python 标准库 |
| ASR `.runtime/asr` | [asr.txt](asr.txt) | 现有 SenseVoice 环境的固定版本清单 |
| TTS `.runtime/index-tts/.venv` | 上游 `.runtime/index-tts/uv.lock` | 按 `Setup-Environment.ps1` 中固定的源码版本和上游锁文件安装 |

三个环境分别安装。CUDA PyTorch wheel 在安装流程中先行准备，ASR 的 torch 与 torchaudio 使用 CUDA 源；完整流程由 [Setup-Environment.ps1](../scripts/Setup-Environment.ps1) 维护。

根目录 [requirements-asr.txt](../requirements-asr.txt) 通过 `-r requirements/asr.txt` 保留旧安装路径。新增或调整 ASR 依赖时修改本目录的 `asr.txt`。本次整理保留原有版本，未重新安装依赖。
