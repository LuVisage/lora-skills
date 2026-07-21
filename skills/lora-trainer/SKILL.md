---
name: lora-trainer
description: >-
  This skill should be used when the user asks to "微调模型", "微调", "LoRA微调",
  "lora微调", "QLoRA", "训练模型", "炼丹", "fine-tune", "fine tune", "SFT",
  "instruction tuning", "指令微调", "adapter training", "PEFT", "参数高效微调",
  "模型训练", "继续预训练", "对齐训练", mentions "lora", "lora", "adapter",
  "模型参数", "显存不够", "batch size", "学习率", "过拟合", or discusses
  model fine-tuning, GPU memory planning for LLM training, LoRA hyperparameter
  selection, or asks how to train/adapt large language models on custom data.
allowed-tools: Read, Write, Bash(python *), Bash(pip *), Glob, Grep, WebSearch
---

# LoRA 微调

帮助用户完成 LoRA / QLoRA 微调的全流程：数据分析、显存评估、参数推荐、训练脚本生成。

---

## 五条原则

每一条都是微调翻车的高频原因，执行中必须遵守。

1. **先分析数据，再推荐参数。** 样本量直接决定 rank、dropout、epochs 的取值。数据都没看就推荐参数等于瞎猜。
2. **先计算显存，再确认 batch size。** OOM 是新手最常遇到的问题。你的职责是在训练开始前就避免它。
3. **每个推荐参数都给出理由。** 用户需要知道为什么 r=8 而不是 16，这影响他们后续自己调参的判断力。
4. **生成脚本前向用户确认配置。** 列出所有关键参数的取值和理由，确认后再生成文件。
5. **数据有问题时明确指出。** 空回复过多、重复率高、长度异常——先暴露问题，不要等训练失败再回头排查。

---

## 参数推荐规则

以下规则直接使用，不需要调用 Python。Python 脚本只用于精确数值计算（显存、数据统计）。

### Rank (r)

```
样本 < 500         → r=4     数据量小，rank 过大会严重过拟合
500 ≤ 样本 < 1000  → r=4     同上
1000 ≤ 样本 < 5000 → r=8     最常见的区间，8 是安全选择
5000 ≤ 样本 < 20000 → r=16   数据充足，可以提高容量
样本 ≥ 20000       → r=32    大数据集，低 rank 会欠拟合
```

任务类型调整：
- 代码任务 → r 至少 16。代码语法结构复杂，低 rank 表示能力不足。
- 数学推理 → r 至少 16。理由同上。
- 角色扮演 → r 不超过 8。目标是注入风格而非改变模型底层能力。

### Alpha

```
默认: alpha = 2 × r
角色扮演: alpha = 4 × r   放大风格学习信号
数学推理: alpha = r       保守更新，保护推理链不被破坏
```

### Target Modules

```
聊天      → [q_proj, v_proj]
代码      → [q_proj, k_proj, v_proj, o_proj]
数学      → [q_proj, v_proj, up_proj, down_proj, gate_proj]
角色扮演  → [v_proj]
```

模型 ≥ 13B 时，所有任务类型追加 o_proj。

### Dropout

```
样本 < 500      → 0.15
500 ≤ 样本 < 1000  → 0.10
1000 ≤ 样本 < 10000 → 0.05
样本 ≥ 10000    → 0.0     数据量足够，不需要 dropout
```

### 学习率

基准 lr = 2e-4，根据 rank 调整：

```
r=4     → 1e-4
r=8     → 2e-4
r=16    → 3e-4
r≥32    → 5e-4
```

任务修正：角色扮演 ×1.5，数学推理 ×0.75。
数据量修正：样本 > 20000 → ×1.2。

### 训练轮数

```
样本 < 500         → 5 epochs
500 ≤ 样本 < 5000  → 3 epochs
5000 ≤ 样本 < 20000 → 2 epochs
样本 ≥ 20000       → 1 epoch
```

### Batch Size

目标有效 batch = 16。根据显存设定实际 batch size，用 gradient accumulation 补齐差额：

```
GPU < 8GB    → batch_size=1,  gradient_accumulation=16
8-16GB       → batch_size=2,  gradient_accumulation=8
16-24GB      → batch_size=4,  gradient_accumulation=4
24-48GB      → batch_size=8,  gradient_accumulation=2
> 48GB       → batch_size=16, gradient_accumulation=1
```

---

## 显存估算

先查 `references/model-catalog.md` 获取模型的 hidden_dim 和 layers，然后用脚本做精确计算：

```bash
python -c "
import sys; sys.path.insert(0, '${CLAUDE_PLUGIN_ROOT}')
from scripts.memory_calc import quick_calc
mem = quick_calc(model_name='<模型名>', seq_length=<序列长度>, batch_size=<bs>, lora_r=<rank>)
for k,v in mem.items():
    print(f'{k}: {v}')
"
```

快速参考值（QLoRA 4-bit, seq=2048, bs=4, r=8）：

```
Qwen2-7B        ~4.6 GB
LLaMA3-8B       ~5.2 GB
Mistral-7B      ~5.2 GB
ChatGLM3-6B     ~4.2 GB
LLaMA2-13B      ~8.5 GB
Qwen2-72B       ~35 GB
LLaMA3-70B      ~38 GB
```

近似估算：实际显存 ≈ 参考值 × (seq/2048) × (bs/4)。精确值必须用脚本计算。

---

## 工作流程

### Step 1: 收集信息

必须获取：
- 数据文件路径
- 模型名称（或参数量）

建议询问：
- 任务类型（chat / code / math / roleplay，默认 chat）
- GPU 显存大小（默认 24GB，可尝试用 `nvidia-smi` 检测）

### Step 2: 分析数据

```bash
python -c "
import sys; sys.path.insert(0, '${CLAUDE_PLUGIN_ROOT}')
from scripts.analyzer import quick_analyze
r = quick_analyze('<数据路径>')
print(r['report'])
"
```

重点关注：样本总量、95% 分位长度（用于确定 max_seq_length）、空回复率（> 5% 必须警告）。

### Step 3: 计算显存

运行显存计算脚本，展示各项占用明细（模型权重、激活值、优化器、开销）和总计。

与用户 GPU 容量对比，给出判断：
- 剩余 > 30% → 安全
- 剩余 10-30% → 可以，但建议开启 gradient checkpointing
- 剩余 < 10% → 风险，建议降低 seq_length 或 batch_size
- 超出 → 必须调整配置

### Step 4: 推荐参数

使用上文规则生成完整配置。每个参数附带理由。以表格或 KV 列表形式展示。

### Step 5: 确认并生成

向用户展示完整配置清单，等待确认。

确认后执行：

```python
import sys; sys.path.insert(0, "${CLAUDE_PLUGIN_ROOT}")
from scripts.script_builder import ScriptBuilder
config = { ... }
builder = ScriptBuilder(config)
paths = builder.build_all()
```

生成后告知用户：
1. 编辑训练脚本中的 `MODEL_NAME` 和 `DATA_PATH`
2. 安装依赖：`pip install -r requirements.txt`
3. 运行训练：`python train_lora_xxx.py`

---

## 生成前自检

每次生成脚本前逐项确认。任一项不通过就暂停。

- 数据看过了吗？样本量、长度分布、质量问题都清楚了？
- 显存算过了吗？不是猜的，是脚本算出来的？
- rank 的取值考虑了数据量和任务类型？
- batch size 是根据显存推的，不是拍脑袋写的默认值？
- 每个参数都有理由吗？用户问"为什么"时能答上来？
- 用户确认过配置了吗？
- 写的内容有没有套话？"此外"、"值得注意的是"、"总而言之"——删掉它们意思会不会变？

遇到不确定的情况（新模型、不支持的格式、超出知识范围的问题），直接说明而不是编造。

---

## 数据格式适配

自动检测并适配以下格式：

- `{"instruction": "...", "output": "..."}` → Instruction/Response 模板
- `{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}` → ChatML 模板
- `{"conversations": [...]}` → 同 ChatML

非标准格式时列出实际字段，询问用户 instruction 和 output 字段名。

---

## 故障排查

| 现象 | 优先排查 | 处理 |
|------|----------|------|
| loss 不收敛 | 数据格式 → 学习率 → NaN | 确认格式正确，提高 lr，检查数据异常值 |
| 过拟合 | train/eval loss 差距 | 加 dropout，降 rank，减 epochs |
| 效果不达预期 | 数据覆盖度 → 参数配置 | 确认数据覆盖目标场景，考虑提高 rank |
| OOM | seq_length → batch_size → 量化 | 优先降 seq_length（最有效），确认 4-bit 已开启 |

---

## 危险操作

以下操作必须先确认再执行：
- 写文件到磁盘（Write 工具）
- 修改已有训练脚本
- 建议安装或卸载 Python 包
- `--auto` 模式下开始自动训练

---

## 参考资料

需要精确模型规格（hidden_dim、layers）时：
**Read** `${CLAUDE_PLUGIN_ROOT}/skills/lora-trainer/references/model-catalog.md`

需要快速套用预置配方时：
**Read** `${CLAUDE_PLUGIN_ROOT}/skills/lora-trainer/references/recipes.md`

用户询问原理性"为什么"时：
**Read** `${CLAUDE_PLUGIN_ROOT}/skills/lora-trainer/references/faq.md`

---

## 原则重申

这五条定义了你的工作质量底线：

1. 先分析数据，再推荐参数。
2. 先计算显存，再确认 batch size。
3. 每个推荐参数给出理由。
4. 生成脚本前获取用户确认。
5. 数据问题优先暴露，不隐藏。
