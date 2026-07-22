---
name: lora:cook
version: 2.5.0
description: >-
  Streamlined LoRA fine-tuning: same pipeline as analyze but concise output.
  Optional --auto flag starts training after confirmation. Use when the user
  is experienced and wants minimal output. Trigger: /lora:cook, "一键微调",
  "直接帮我炼丹", "cook this data".
argument-hint: <data-path> [model] [task] [--auto]
disable-model-invocation: true
allowed-tools: Read, Write, Bash(python *), Glob, Grep
---

# /lora:cook — 一键炼丹

和 analyze 一样的流程，但输出精简。适合已经知道自己在干什么的人。

## 参数

- `$ARGUMENTS[0]` — 数据路径，必填
- `$ARGUMENTS[1]` — 模型名，选填
- `$ARGUMENTS[2]` — 任务类型，选填
- `$ARGUMENTS[3]` — `--auto`，选填。确认后自动开始训练。

## 示例

```
# 快速炼丹：精简输出
/lora:cook ./data/train.jsonl qwen2-7b chat

# 自动模式：确认后直接开练
/lora:cook ./data/train.jsonl qwen2-7b chat --auto

# 只给数据
/lora:cook ./data/train.jsonl

# 代码任务
/lora:cook ./data/code.jsonl deepseek-7b code

# 自然语言
"一键微调 ./data/train.jsonl，用 qwen2-7b"
"帮我炼丹，数据在 ./data/chat.jsonl"
```

## 执行

加载 lora-trainer skill，跳过冗长解释，直接给数据报告摘要 + 参数推荐表 + 确认提示。
`--auto` 模式下：用户确认后 → 生成脚本 → 运行训练（先检查 `nvidia-smi` 有 GPU 再跑）。
