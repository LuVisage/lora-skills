# 显存快速参考

以下为 QLoRA 4-bit 量化、seq_length=2048、batch_size=4、r=8 配置下的显存占用参考值。

## 7B-9B 模型

```
Qwen2-7B        ~4.6 GB
LLaMA3-8B       ~5.2 GB
Mistral-7B      ~5.2 GB
ChatGLM3-6B     ~4.2 GB
Gemma-7B        ~5.0 GB
InternLM2-7B    ~4.8 GB
DeepSeek-7B     ~4.8 GB
Yi-6B           ~4.0 GB
```

## 13B-14B 模型

```
LLaMA2-13B      ~8.5 GB
Qwen2-14B       ~9.0 GB
Baichuan2-13B   ~8.5 GB
InternLM2-20B   ~13.0 GB
```

## 30B-40B 模型

```
Yi-34B          ~20.0 GB
DeepSeek-33B    ~19.0 GB
Qwen2-32B       ~19.0 GB
Mixtral-8x7B    ~22.0 GB
```

## 65B-72B 模型

```
Qwen2-72B       ~35.0 GB
LLaMA3-70B      ~38.0 GB
LLaMA2-70B      ~37.0 GB
DeepSeek-67B    ~35.0 GB
```

## 小模型 (< 4B)

```
Qwen2-0.5B      ~0.8 GB
Qwen2-1.5B      ~1.5 GB
Phi-3-mini      ~1.8 GB
Gemma-2B        ~1.8 GB
Qwen2-4B        ~3.0 GB
Phi-3-small     ~4.0 GB
```

## 近似估算公式

实际显存 ≈ 参考值 × (seq_length / 2048) × (batch_size / 4)

示例：Qwen2-7B、seq=4096、bs=2 → 4.6 × 2 × 0.5 = 4.6 GB

**注意：** 精确值必须用 `scripts/memory_calc.py` 计算。参考值不含 KV cache 和 gradient checkpointing 优化效果。
