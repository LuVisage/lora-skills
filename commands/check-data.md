---
name: lora:check-data
description: 快速检查微调数据质量 — 空回复、重复、长度分布、格式检测
argument-hint: <data-path>
allowed-tools: Read, Bash(python *), Glob
---

# /lora:check-data

看一眼数据质量。不做分析，不推荐参数，不生成脚本。

## 参数

- `$1` — 数据路径 (JSONL)，必填

## 示例

```
/lora:check-data ./data/train.jsonl
```

## 输出

会告诉你：多少条、什么格式、长度分布、有没有空回复、有没有重复。
有问题会直接指出来，附带建议。
