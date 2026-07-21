---
name: lora:cook
description: 一键炼丹 — 数据分析 + 参数推荐 + 脚本生成，可选自动开始训练
argument-hint: <data-path> [--model <name>] [--task <type>] [--auto]
allowed-tools: Read, Write, Bash(python *), Glob, Grep
---

# /lora:cook

和 analyze 做一样的事，但输出精简。适合已经知道自己在干什么的人。

加了 `--auto` 的话，生成完会问你要不要直接开始训练。

## 参数

- `$1` — 数据路径，必填
- `$2` — 模型名，选填
- `$3` — 任务类型，选填
- `$4` — `--auto`，选填

## 示例

```
/lora:cook ./data/train.jsonl qwen2-7b chat
/lora:cook ./data/train.jsonl --auto
```
