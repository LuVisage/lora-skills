---
name: lora:check-data
version: 2.1.0
description: >-
  Quick training data quality scan: sample count, length distribution, empty
  responses, duplicates, format detection. No parameter recommendation or script
  generation. Use when the user just wants to check data quality before deciding
  whether to fine-tune. Trigger: /lora:check-data, "检查数据", "看看数据质量",
  "check my data".
argument-hint: <data-path>
disable-model-invocation: true
allowed-tools: Read, Bash(python *), Glob
---

# /lora:check-data — 数据质量扫描

只看数据质量，不做参数推荐，不生成脚本。适合在决定微调前快速摸底。

## 参数

- `$ARGUMENTS[0]` — 数据路径 (JSONL)，必填

## 示例

```
/lora:check-data ./data/train.jsonl
```

## 输出

- 样本总量、格式类型
- 长度分布（min / avg / p95 / max）
- 空回复数及占比（> 5% 红色警告）
- 重复对数量（> 10% 红色警告）
- 质量问题清单 + 修复建议

## 执行

加载 lora-trainer skill 的数据分析能力，但不执行 Step 3-5（显存、参数、脚本）。
