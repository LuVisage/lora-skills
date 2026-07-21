# LoRA 预置配方

为常见场景预调好的参数组合。可用作快速起点。

## 通用聊天 (Chat)

```
r=8, alpha=16, modules=[q_proj, v_proj], dropout=0.05, lr=2e-4, epochs=3
```

适用：日常对话、客服、通用助手。最常用的配方。

## 代码生成 (Code)

```
r=16, alpha=32, modules=[q_proj, k_proj, v_proj, o_proj], dropout=0.05, lr=2e-4, epochs=3
```

适用：代码补全、Bug 修复、代码翻译、SQL 生成。

## 数学推理 (Math)

```
r=16, alpha=16, modules=[q_proj, v_proj, up_proj, down_proj, gate_proj], dropout=0.05, lr=1.5e-4, epochs=2
```

适用：数学解题、逻辑推理、定理证明。lr 较低以保护推理链。

## 角色扮演 (Roleplay)

```
r=8, alpha=32, modules=[v_proj], dropout=0.05, lr=3e-4, epochs=3
```

适用：角色扮演、风格模仿、创意写作。仅训练 V 层以保留基础语言能力。

## Qwen2-7B 专用

```
建议 max_length=2048（中文 token 效率高）
中文对话场景无需调整默认配方
```

## LLaMA3-8B 中文场景

```
建议 r=16（稍大 rank 弥补中文能力不足）
建议 max_length=4096（英文 token 效率低，需更长上下文覆盖）
```

## 极小数据集 (< 200 条)

```
r=4, alpha=8-16, dropout=0.15, epochs=5, lr=1e-4
```

关键：高 dropout + 低 rank + 多 epochs。宁愿欠拟合也不要过拟合。
