# 开发与维护

本项目从当前目录直接运行。入口、配置和模型路径都相对于项目根目录解析，Python 包维持 `storyteller/` 布局。目录与模块导航见 [项目架构](ARCHITECTURE.md)。

## 环境与依赖

- `.runtime/core/`：控制端、CLI、面板和安装辅助工具。
- `.runtime/index-tts/.venv/`：TTS 与独立训练使用的环境。
- `.runtime/asr/`：语音识别校对环境。
- `.runtime/models/`：已下载的模型资产；`.runtime/index-tts/` 同时保存固定版本的上游源码。
- `requirements/core.txt` 与 `requirements/asr.txt`：控制端安装辅助依赖和 ASR 固定版本；TTS 使用上游锁文件。

安装或补齐环境仍使用根目录的 `setup.ps1`，详见 [使用说明](../README.md) 和 [依赖说明](../requirements/README.md)。本机路径配置保留在 `config.local.json`，示例位于 `config.example.json`。

## 修改后的检查

在项目根目录执行：

```powershell
# 参数与文件路径检查，不加载模型。
.\run.ps1 --help
.\run.ps1 doctor

# 核心回归：数值校对、关键字、生成停顿、面板会话、分章和续跑。
& '.\.runtime\core\Scripts\python.exe' -B -X utf8 -m unittest discover -s tests -v

# 源码语法检查。
& '.\.runtime\core\Scripts\python.exe' -m compileall -q storyteller scripts/bootstrap

# 核验既有训练素材，没有训练或重新划分数据。
.\train_voice.ps1 -Stage Verify
```

涉及模型安装、模型加载或推理代码变动时，再运行：

```powershell
.\run.ps1 doctor --verify-hashes --load-models
```

该命令会实际读取模型并加载到显卡。涉及生成质量的改动还需要指定文本和参考声音的实测、复听。

PowerShell 脚本使用带 BOM 的 UTF-8，兼容 Windows PowerShell 5.1。新增工作流使用 `scripts/Common.ps1` 的项目根目录定位和编码初始化，保持根目录入口的参数行为。

`tests/` 使用标准库 unittest，模型阶段采用可控工作进程模拟；音频导出与 HTTP 会话测试使用本地 FFmpeg 和回环服务，不加载 GPU 模型。真实模型检查和样例生成另行验证。

## 已保存的验证结果

2026-10-22 的五项修复覆盖数值结构、关键字规范化、固定内部生成停顿、面板会话隔离和正文误分章。操作与配置说明已并入根目录 README，源码职责见项目架构。验证原始记录保存在 `docs/audits/2026-10-22/fixes/`：

| 记录 | 当时验证结果 |
|---|---|
| [回归测试](audits/2026-10-22/fixes/tests.log) | 21 项全部通过 |
| [模型检查](audits/2026-10-22/fixes/doctor.json) | 40 个模型文件校验及 TTS、ASR/VAD 真实离线加载通过 |
| [真实语音样例](audits/2026-10-22/fixes/model-smoke.json) | 两段首轮校对通过，CER 均为 0，WAV/MP3 导出正常 |
| [数据与兼容性](audits/2026-10-22/fixes/integrity.json) | 书籍、音频、训练数据及权重核验通过，现有 85 章保持一致 |

这些记录是验证时的快照；当前代码的检查使用上面的命令。短样例和自动校对不替代连续章节的人工听感验收。

## 备份与恢复

项目源码、脚本、测试和文档已纳入 Git 版本管理。模型和运行环境、原始录音与标注、训练数据与权重、书籍成品、审计生成的样例目录及 `config.local.json` 保留在本机，由 `.gitignore` 排除，需单独备份。

此前目录整理、源码修复和文档清理的恢复材料按操作分别保存在本机 `archive/`，入口见 [归档索引](../archive/README.md)。Git 仅保留归档索引，不包含这些恢复快照。各清单记录原相对路径、备份位置及 SHA-256。

恢复时对照对应操作的 `manifest.json`，将需要的 `before/` 文件复制回原相对路径。归档中的代码是历史快照，当前运行源码位于 `storyteller/` 和 `scripts/`。
