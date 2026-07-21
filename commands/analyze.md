---
name: lora:analyze
description: 完整分析微调数据 — 数据诊断 + 显存计算 + LoRA 参数推荐 + 生成训练脚本
argument-hint: <data-path> [--model <name>] [--task <type>] [--gpu <gb>]
allowed-tools: Read, Write, Bash(python *), Glob, Grep
---

# /lora:analyze

给你一条训练数据和一个模型名，帮你跑完整分析：数据质量 → 显存评估 → 参数推荐 → 生成脚本。

## 参数

- `$1` — 数据路径 (JSONL)，必填
- `$2` — 模型名（qwen2-7b / llama3-8b / ...），选填
- `$3` — 任务类型（chat / code / math / roleplay），选填，默认 chat
- `$4` — 显存 (GB)，选填，默认 24

## 示例

```
/lora:analyze ./data/train.jsonl qwen2-7b chat 24
/lora:analyze ./data/train.jsonl
/lora:analyze ./data/train.jsonl --task code
```
