# 训练数据与权重

本目录保存独立训练实验。当前日常有声书配置 `config.local.json` 的 `tts_adapter` 为空，使用 IndexTTS 原模型；训练产物不会自动成为制作默认值。

## 当前数据集目录

```text
training/qiong1_20261022/
├─ sources.json             原始标注、音频哈希与来源记录
├─ dataset_report.json      筛选结果、数据划分和限制说明
├─ train.jsonl              训练清单
├─ validation.jsonl         验证清单
├─ test.jsonl               测试清单
├─ excluded.jsonl           排除条目及原因
├─ extraction_status.json   特征提取进度
├─ features/                模型训练特征
├─ runs/lora_rank16/         检查点、训练状态、指标和总结
└─ config.trained.json       显式选择本次适配器的配置
```

原始输入保存在项目根目录 `01_语音/` 和 `02_标注/qiong1.list`。清单引用原始音频的绝对路径；训练代码位于 [training_data.py](../storyteller/training_data.py)、[finetune.py](../storyteller/finetune.py) 和 [lora.py](../storyteller/lora.py)。

## 操作入口

在项目根目录执行：

```powershell
.\train_voice.ps1 -Stage Status
.\train_voice.ps1 -Stage Verify
```

`Stage` 支持 `Prepare`、`Extract`、`Train`、`Verify`、`Status` 和 `All`，默认 `All`。默认数据集是 `training/qiong1_20261022`；其他目录通过 `-Dataset` 指定。`Status` 读取已保存的进度，`Verify` 检查原始标注和音频哈希，`Prepare` 会重新生成数据划分，`Extract` 和 `Train` 使用显卡。

独立命令行参数可通过以下命令查看：

```powershell
& '.\.runtime\index-tts\.venv\Scripts\python.exe' -m storyteller.finetune --help
```

`config.trained.json` 供需要适配器的实验显式选择。日文语音与日文转写用于本次训练；训练损失不能证明中文发音或长篇自然度改善，这些结论仍需要实际试听。
