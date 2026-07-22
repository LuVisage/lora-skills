---
name: lora:analyze
version: 2.5.0
description: >-
  Full LoRA fine-tuning pipeline: data audit → VRAM estimation → hyperparameter
  recommendation → training script generation. Use when the user provides
  training data and wants end-to-end analysis. Trigger: /lora:analyze,
  "analyze my data", "帮我分析微调数据", "看看这个数据能不能微调".
argument-hint: <data-path> [model] [task] [gpu-gb]
disable-model-invocation: true
allowed-tools: Read, Write, Bash(python *), Glob, Grep
---

# /lora:analyze — 完整分析

数据 → 显存 → 参数 → 脚本，一步到位。

## 参数

- `$ARGUMENTS[0]` — 数据路径 (JSONL)，必填
- `$ARGUMENTS[1]` — 模型名（qwen2-7b / llama3-8b / ...），选填
- `$ARGUMENTS[2]` — 任务类型（chat / code / math / roleplay），选填，默认 chat
- `$ARGUMENTS[3]` — 显存 (GB)，选填，自动检测

## 示例

```
# 完整分析：数据 + 显存 + 参数 + 脚本
/lora:analyze ./data/train.jsonl qwen2-7b chat 24

# 只给数据，其他我来猜
/lora:analyze ./data/train.jsonl

# 指定任务类型
/lora:analyze ./data/code.jsonl deepseek-7b code

# 角色扮演数据
/lora:analyze ./data/rp.jsonl llama3-8b roleplay

# 自然语言也行
"帮我分析 ./data/train.jsonl，用 qwen2-7b 做聊天微调"
"看看这个数据适不适合微调 ./data/chat.jsonl"
```

## 执行

加载 lora-trainer skill，执行完整五步流程。参考 SKILL.md 中的示例了解输出格式。
