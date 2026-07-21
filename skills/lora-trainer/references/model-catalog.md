# 常见大模型规格目录

纯数据参考。行为规则在 SKILL.md 中。

⚠️ **MoE 模型注意：** 显存按激活参数计算，不是总参数。如 Qwen2-57B 总参数 57B 但激活仅 14B。

## Qwen2 系列

```
qwen2-0.5b  → params: 0.5B,  hidden: 896,   layers: 24, vocab: 151936, ctx: 32K
qwen2-1.5b  → params: 1.5B,  hidden: 1536,  layers: 28, vocab: 151936, ctx: 32K
qwen2-7b    → params: 7B,    hidden: 3584,  layers: 28, vocab: 151936, ctx: 128K
qwen2-14b   → params: 14B,   hidden: 5120,  layers: 40, vocab: 151936, ctx: 128K
qwen2-32b   → params: 32B,   hidden: 7168,  layers: 64, vocab: 151936, ctx: 128K
qwen2-57b   → params: 57B (14B active, MoE), hidden: 4096, layers: 48, vocab: 151936, ctx: 64K
qwen2-72b   → params: 72B,   hidden: 8192,  layers: 80, vocab: 151936, ctx: 128K
```

## Qwen2.5 系列

```
qwen2.5-0.5b  → params: 0.5B,  hidden: 896,   layers: 24, vocab: 151936, ctx: 32K
qwen2.5-1.5b  → params: 1.5B,  hidden: 1536,  layers: 28, vocab: 151936, ctx: 32K
qwen2.5-3b    → params: 3B,    hidden: 2048,  layers: 36, vocab: 151936, ctx: 32K
qwen2.5-7b    → params: 7B,    hidden: 3584,  layers: 28, vocab: 151936, ctx: 128K
qwen2.5-14b   → params: 14B,   hidden: 5120,  layers: 40, vocab: 151936, ctx: 128K
qwen2.5-32b   → params: 32B,   hidden: 7168,  layers: 64, vocab: 151936, ctx: 128K
qwen2.5-72b   → params: 72B,   hidden: 8192,  layers: 80, vocab: 151936, ctx: 128K
```

## LLaMA3 系列

```
llama3-8b   → params: 8B,   hidden: 4096,  layers: 32, vocab: 128000, ctx: 8K
llama3-70b  → params: 70B,  hidden: 8192,  layers: 80, vocab: 128000, ctx: 8K
```

## LLaMA3.1/3.2 系列

```
llama3.1-8b    → params: 8B,    hidden: 4096,  layers: 32,  vocab: 128000, ctx: 128K
llama3.1-70b   → params: 70B,   hidden: 8192,  layers: 80,  vocab: 128000, ctx: 128K
llama3.1-405b  → params: 405B,  hidden: 16384, layers: 126, vocab: 128000, ctx: 128K
llama3.2-1b    → params: 1B,    hidden: 2048,  layers: 16,  vocab: 128000, ctx: 128K
llama3.2-3b    → params: 3B,    hidden: 3072,  layers: 28,  vocab: 128000, ctx: 128K
```

## LLaMA2 系列

```
llama2-7b   → params: 7B,   hidden: 4096,  layers: 32, vocab: 32000,  ctx: 4K
llama2-13b  → params: 13B,  hidden: 5120,  layers: 40, vocab: 32000,  ctx: 4K
llama2-70b  → params: 70B,  hidden: 8192,  layers: 80, vocab: 32000,  ctx: 4K
```

## Mistral 系列

```
mistral-7b      → params: 7.3B,  hidden: 4096,  layers: 32, vocab: 32000, ctx: 8K
mistral-nemo    → params: 12B,   hidden: 5120,  layers: 40, vocab: 128000, ctx: 128K
mistral-large   → params: 123B,  hidden: 12288, layers: 88, vocab: 128000, ctx: 128K
mixtral-8x7b    → params: 47B (13B active, MoE), hidden: 4096, layers: 32, vocab: 32000, ctx: 32K
```

## DeepSeek 系列

```
deepseek-7b    → params: 7B,   hidden: 4096, layers: 30, vocab: 102400, ctx: 4K
deepseek-67b   → params: 67B,  hidden: 8192, layers: 95, vocab: 102400, ctx: 4K
deepseek-v2    → params: 236B (21B active, MoE),  hidden: 5120, layers: 60, vocab: 102400, ctx: 128K
deepseek-v3    → params: 671B (37B active, MoE),  hidden: 7168, layers: 61, vocab: 129280, ctx: 128K
```

## ChatGLM 系列

```
chatglm3-6b  → params: 6B,  hidden: 4096, layers: 28, vocab: 65024,  ctx: 8K
chatglm4-9b  → params: 9B,  hidden: 4096, layers: 40, vocab: 65024,  ctx: 128K
```

## Yi 系列

```
yi-6b   → params: 6B,  hidden: 4096, layers: 32, vocab: 64000, ctx: 4K
yi-34b  → params: 34B, hidden: 7168, layers: 60, vocab: 64000, ctx: 4K
```

## Baichuan2 系列

```
baichuan2-7b   → params: 7B,  hidden: 4096, layers: 32, vocab: 125696, ctx: 4K
baichuan2-13b  → params: 13B, hidden: 5120, layers: 40, vocab: 125696, ctx: 4K
```

## Phi-3 系列

```
phi-3-mini    → params: 3.8B, hidden: 3072, layers: 32, vocab: 32064,  ctx: 128K
phi-3-small   → params: 7B,   hidden: 4096, layers: 32, vocab: 100352, ctx: 128K
phi-3-medium  → params: 14B,  hidden: 5120, layers: 40, vocab: 100352, ctx: 128K
```

## Gemma 系列

```
gemma-2b  → params: 2B, hidden: 2048, layers: 18, vocab: 256000, ctx: 8K
gemma-7b  → params: 7B, hidden: 3072, layers: 28, vocab: 256000, ctx: 8K
```

## InternLM2 系列

```
internlm2-7b   → params: 7B,  hidden: 4096, layers: 32, vocab: 92544, ctx: 200K
internlm2-20b  → params: 20B, hidden: 6144, layers: 48, vocab: 92544, ctx: 200K
```

## 显存速算公式

QLoRA 4-bit, seq=2048, bs=4, r=8 条件下：

```
Dense 模型: 显存(GB) ≈ params(B) × 0.5 + hidden × layers × seq × bs × 2 / 1e9 × 0.3
MoE 模型:   显存(GB) ≈ active_params(B) × 0.5 + hidden × layers × seq × bs × 2 / 1e9 × 0.3
```

精确值请用 `scripts/memory_calc.py` 计算。MoE 模型会自动使用激活参数。
