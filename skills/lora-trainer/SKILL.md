---
name: lora-trainer
version: 2.5.0
description: LoRA/QLoRA 大模型微调全能助手 —— 从数据分析、显存估算、超参数推荐到训练脚本生成，一站式搞定。支持 Qwen/Llama/Mistral/DeepSeek 等 40+ 模型、13 大家族。当用户提到"微调""fine-tune""LoRA""QLoRA""炼丹""SFT""instruction tuning""指令微调""adapter training""PEFT""参数高效微调""对齐训练""训练模型""继续预训练""显存不够""训练报错""debug""OOM""loss不收敛""过拟合"或询问 rank/alpha/batch size/learning rate/epochs/dropout 等超参数时自动触发。
allowed-tools: Read, Write, Bash(python *), Bash(pip *), Glob, Grep, WebSearch
effort: high
---

# LoRA / QLoRA 微调助手

> 从原始 JSONL 数据到可运行的训练脚本，一条命令搞定。支持 40+ 模型、5 种任务类型、
> 自动数据分析、精准显存估算、有理由的超参数推荐。

**核心能力：**

- **数据诊断** —— 自动检测格式，标记空回复、重复对、长度异常，估算 token 量
- **显存估算** —— 内置 40+ 模型规格库，MoE 感知，精确到各组件明细
- **参数推荐** —— 57 条规则驱动，每个参数附理由，支持 chat/code/math/roleplay/CPT
- **脚本生成** —— 一键输出 QLoRA 训练脚本 + 推理脚本 + YAML 配置，即开即用

**触发场景：** 微调模型、LoRA 训练、QLoRA 训练、炼丹、SFT、指令微调、继续预训练、
显存不够怎么办、怎么选 rank/学习率/batch size……

---

## 快速开始

第一次用？**你只需要告诉我训练数据在哪里。** 我会自动分析数据、估算显存、推荐参数、生成脚本。

### 三种用法（选一个）

| 你想做什么 | 这样说 | 我会做什么 |
|-----------|--------|-----------|
| 🔍 检查数据能不能用 | `/lora:check-data ./data/train.jsonl` | 扫描质量，报告空回复、重复、异常 |
| 📊 完整分析 + 生成脚本 | `/lora:analyze ./data/train.jsonl qwen2-7b chat` | 数据分析 → 显存 → 参数推荐 → 训练脚本 |
| 🚀 快速炼丹 | `/lora:cook ./data/train.jsonl qwen2-7b chat` | 同上但输出精简，--auto 可自动开练 |

### 自然语言也行

不需要记命令格式，用中文描述即可：

```
"帮我分析 ./data/train.jsonl，用 qwen2-7b 微调聊天模型"
"看看 ./data/chat.jsonl 数据质量怎么样"
"5000 条数据，7b 模型，推荐什么参数？"
"显存只有 8GB，能微调 llama3-8b 吗？"
"训练 OOM 了怎么办？loss 不收敛怎么办？"
```

### 先跑一下试试

把上面的命令里的 `./data/train.jsonl` 换成你的实际数据文件路径，直接发给我。

> 💡 **不需要提前记任何参数。** 我会根据你的数据和模型自动推荐所有配置，并告诉你每个参数为什么这么选。

---

## 🇨🇳 国内用户：模型下载加速

HuggingFace 在国内访问较慢。生成的训练脚本默认从 HuggingFace 下载模型——建议先配置镜像。

### 方法一：hf-mirror.com（推荐，一行命令）

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

设置后所有 HuggingFace 下载自动走国内镜像，无需改代码。

### 方法二：ModelScope（阿里云，速度最快）

```bash
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('qwen/Qwen2-7B-Instruct', cache_dir='./models')"
```

下载后在生成的训练脚本中把 `MODEL_NAME` 改成 `"./models/qwen/Qwen2-7B-Instruct"`（本地路径）即可。

### 方法三：直接下载

浏览器访问 https://hf-mirror.com ，搜索模型名，下载所有文件到本地目录。

> ⚠️ 如果用镜像或本地路径，记得修改训练脚本中的 `MODEL_NAME` 变量。生成脚本时我会提醒你。

---

## 六条原则

每一条都是微调翻车的高频原因，执行中必须遵守。

1. **先分析数据，再推荐参数。** 样本量直接决定 rank、dropout、epochs 的取值。数据都没看就推荐参数等于瞎猜。
2. **先计算显存，再确认 batch size。** OOM 是新手最常遇到的问题。你的职责是在训练开始前就避免它。
3. **每个推荐参数都给出理由。** 用户需要知道为什么 r=8 而不是 16，这影响他们后续自己调参的判断力。
4. **生成脚本前向用户确认配置。** 列出所有关键参数的取值和理由，确认后再生成文件。
5. **数据有问题时明确指出。** 空回复过多、重复率高、长度异常——先暴露问题，不要等训练失败再回头排查。
6. **能动手就不动嘴。** 用户说"看看数据"→ 直接跑分析脚本。用户说"显存够不够"→ 直接跑显存计算。用户说"生成脚本"→ 确认后直接生成。不要只给命令让用户自己跑——你跑完告诉结果，比让用户自己跑快 10 倍。

---

## ⚠️ 常见坑 (Gotchas)

真实使用中高频踩坑记录。每一条都对应至少一次实际翻车。

### 数据坑

- **JSONL 不是每行一个合法 JSON。** 用户经常把格式化的 JSON（多行）当 JSONL 用。第一步先验证 `python -c "import json; [json.loads(l) for l in open('data.jsonl')]"`。
- **instruction 和 output 列名不标准。** 常见别名：`prompt`/`completion`、`question`/`answer`、`input`/`target`、`text`/`summary`。不要假设字段名一定是 `instruction`/`output`。
- **字符数 ≠ token 数。** 中文字符在 tokenizer 里是 2-3 个 token，英文约 0.28 token/字符。分析报告现在同时输出字符长度和估算 token 长度，以估算 token 数为准选 max_seq_length。
- **不要用字符 p95 当 max_seq_length。** 英文 2048 字符 ≈ 570 tokens，中文 2048 字 ≈ 4000+ tokens。差距 7×。用估算 token P95 而非字符 P95。
- **角色扮演数据常带系统提示词。** `messages` 格式里有 `system` role 的话，这些 tokens 也参与训练但不产生 loss。样本量统计时要意识到这一点。

### 显存坑

- **nvidia-smi 显示的 "used" 不是全部可用显存。** 可能有其他进程占用。不要假设 24GB 显卡就有 24GB 可用——先跑 `nvidia-smi --query-gpu=memory.free --format=csv`。
- **量化配置不生效。** 用户说 "QLoRA" 但实际用的配置里 `load_in_4bit=False`。生成脚本时必须在代码里显式确认 `BitsAndBytesConfig(load_in_4bit=True)`。
- **gradient checkpointing 不是免费的。** 开启后显存减少 ~30%，但训练时间增加 ~20%。不要在显存充裕时推荐它。

### 模型架构坑

- **target_modules 命名因模型架构而异。** LLaMA/Qwen/Mistral 用 `q_proj, k_proj, v_proj, o_proj`，但 Phi-3 用融合的 `qkv_proj`，部分 GPT 变体用 `query_key_value`，Bloom 用 `dense` 代替 `o_proj`。如果 target_modules 里的名字在模型里不存在，`get_peft_model` 会静默跳过，训练完发现什么都没学到。不确定时先 `print(model)` 看一眼实际的层名。
- **GQA（分组查询注意力）影响 k_proj/v_proj 参数量。** Qwen2、LLaMA3、Mistral 都用了 GQA，k_proj 和 v_proj 的维度比 q_proj 小。这不影响 target_modules 的选择但影响 LoRA 参数量的计算（影响 < 5%，可忽略）。

### 参数坑

- **batch_size=1 时 gradient_accumulation=16 ≠ batch_size=16。** BatchNorm 等少数层的行为不同。不过 Transformer 架构下基本等价，可以忽略此差异。
- **大 rank + 小数据 = 必过拟合。** 数据 < 500 条用 r=32 基本上就是背数据了。这不是建议，是数学规律。
- **学习率不随 batch size 调整。** 有效 batch 翻倍时，lr 应该按 sqrt 比例调整。但 LoRA 场景下影响不大，不用主动提——除非用户自己改了 batch size。

### 交互坑

- **用户说 "帮我微调" 但没给数据文件。** 别猜。第一步就问：数据在哪？
- **用户给了 .json 但其实是 .jsonl。** 手动检查文件内容。Python 一行一行验证比瞎猜可靠。
- **生成完脚本用户直接跑，然后报错。** 脚本里 MODEL_NAME 和 DATA_PATH 必须用占位符或注释标记清楚，让用户一看就知道要改什么。

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

任务类型调整（受数据量上限约束）：
- 代码任务 → r 建议 16，但不超过当前数据量区间的上限。样本 < 1000 时最高 r=8，样本 < 500 时最高 r=4。
- 数学推理 → r 建议 16，约束同上。数据量不够时大 rank 比低 rank 效果更差。
- 角色扮演 → r 不超过 8。目标是注入风格而非改变模型底层能力。

关键原则：**数据量约束优先于任务类型调整。** 300 条代码数据用 r=16 必定过拟合，效果不如 r=4。

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
CPT       → [q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, gate_proj]
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

数据量修正（调整有效 batch 目标）：
- 样本 < 200 → 目标有效 batch=4 或 8（小 batch 噪声有正则化效果）
- 样本 > 50k → 目标有效 batch=32 或 64（大批量减少梯度方差）

### DoRA（推荐替代 LoRA）

DoRA (Weight-Decomposed Low-Rank Adaptation) 将预训练权重分解为幅度+方向，只对方向做低秩更新。相同 rank 下效果稳定优于 LoRA，数学/代码任务提升尤为明显。

```
适用场景: 所有任务均可使用，数学/代码任务推荐优先选择
rank 规则: 与 LoRA 相同
额外显存: ~0（仅多了 magnitude vector，可忽略）
使用方式: PEFT 库中 import DoRAConfig 或 LoraConfig(use_dora=True)
```

用户问"DoRA 还是 LoRA"时：数据 < 1000 条优先 DoRA（正则化效果更好），数学/代码任务优先 DoRA，简单聊天任务两者皆可。

### LoRA+（进阶选项）

LoRA+ 给 A 矩阵和 B 矩阵设置不同的学习率（`lr_B = lr_A × 16`），收敛更快、最终效果更好。ICLR 2024 论文提出，已在 PEFT 库中原生支持。

```
适用场景: 所有任务均可尝试，尤其数据 > 5k 时收益明显
配置方式: LoraConfig(..., use_rslora=True) 或手动设置 optimizer param groups
注意: LoRA+ 需要更高学习率（基准 lr 提高 2-4×），否则收敛慢
```

用户问"LoRA+ 怎么用"时：先确认 PEFT 版本 ≥ 0.10，然后建议 `use_rslora=True`。小数据集（< 1k）不优先推荐。

### NEFTune（默认启用）

NEFTune 在 embedding 层注入微量噪声（`neftune_noise_alpha=5`），成本为零（不增加显存和训练时间），instruction-tuning 场景下稳定提升 3-5%。业界标配。

```
默认值: neftune_noise_alpha=5
关闭条件: 显式 CPT 场景（纯文本预训练不需要噪声）
```

### 学习率调度器

```
默认: cosine（平滑衰减，通用场景最稳）
代码: cosine（复杂 loss landscape 需要平滑）
数学: cosine（保护推理链，不推荐 linear 的陡降）
角色扮演: linear（简单任务，快速收敛）
CPT（继续预训练）: constant_with_warmup（保持恒定 lr）
```

warmup 比例：总步数的 10%（最少 10 步）。小数据集尤其重要——warmup 太大会浪费宝贵的训练步数。

---

## 显存估算

先查 `references/model-catalog.md` 获取模型的 hidden_dim 和 layers，然后用脚本做精确计算：

```bash
python -c "
import sys; sys.path.insert(0, '${CLAUDE_PLUGIN_ROOT}')
from scripts.memory_calc import quick_calc
mem = quick_calc(model_name='<模型名>', seq_length=<序列长度>, batch_size=<bs>, lora_r=<rank>, num_modules=<模块数>)
for k,v in mem.items():
    print(f'{k}: {v}')
"
```

`num_modules` 按 target_modules 数量填：chat=2, code=4, math=5, roleplay=1。这个数字直接影响优化器显存计算。

快速参考值见 `references/vram-reference.md`（QLoRA 4-bit, seq=2048, bs=4, r=8 下的常见模型显存占用）。

---

## 工作流程

### Step 0: 启动检查（每次交互开始时）

1. **版本确认** — 检查 SKILL 版本是否最新（当前 v2.4.0），版本信息在 SKILL.md 头部的 `version` 字段。如果用户报告行为与文档不符，优先怀疑版本问题。
2. **环境探测** — 静默尝试检测 Python 和 CUDA 环境，失败不报错只记下：
   ```bash
   python -c "import torch; print(f'CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null || echo "no-torch"
   nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo "no-gpu"
   ```

### Step 1: 收集信息

必须获取：
- 数据文件路径
- 模型名称（或参数量）

建议询问：
- 任务类型（chat / code / math / roleplay，默认 chat）
- GPU 显存大小。Step 0 已尝试自动检测，失败则问用户。

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
builder = ScriptBuilder(
    config,
    data_format='<检测到的格式>',   # instruction-output / messages / conversations
    max_seq_length=<p95长度>,
)
paths = builder.build_all()
```

生成后告知用户：
1. 编辑训练脚本中的 `MODEL_NAME` 和 `DATA_PATH`
2. 安装依赖：`pip install -r requirements.txt`
3. 运行训练：`python train_lora_xxx.py`

### Step 5.5: 训练跟进（运行时支持）

脚本生成不是终点。明确告诉用户：**训练过程中遇到任何问题，随时回来找我。**

我会帮你处理的常见情况：

| 用户说 | 我会做什么 |
|--------|-----------|
| "训练 OOM 了" | 重新计算显存 → 调整 seq_length/batch_size → 生成新脚本 |
| "loss 不收敛 / loss 乱跳" | 检查数据格式 → 确认学习率 → 检查梯度裁剪 |
| "训练完了但效果不好" | 分析可能原因 → 建议调参方向 → 生成评估脚本对比 |
| "这个报错是什么意思" | 根据报错信息匹配故障排查表，给出具体解决步骤 |
| "帮我看看训练日志" | 分析 loss 曲线 → 检查是否过拟合 → 给出调整建议 |

**不需要重新走完整流程**——告诉我具体问题，我会直接诊断。

### Step 6: 评估

训练完成后，建议用户评估微调效果。询问是否生成评估脚本：

```python
import sys; sys.path.insert(0, "${CLAUDE_PLUGIN_ROOT}")
from scripts.evaluator import Evaluator
ev = Evaluator('<测试数据路径>')
eval_script = ev.build_eval_script()
```

评估脚本对比基座模型 vs LoRA 模型在相同 prompt 上的输出，保存到 `eval_results.json`。

告诉用户评估要点：
- 看输出是否遵循了训练数据的格式/风格（格式一致性）
- 看事实性知识是否保留（不要因微调引入幻觉）
- 建议留 10-20 条数据不参与训练，专门用于评估
- 如果效果不理想 → 排查数据质量 → 调整参数 → 重新训练

---

## 示例：完整流程

下面展示一个典型交互的预期输出。用户输入：

```
/lora:analyze ./data/chat_train.jsonl qwen2-7b chat 16
```

### 你应该输出的内容

**数据报告（Step 2 结果）：**
```
📊 数据诊断：chat_train.jsonl
  - 样本总量：2,340 条
  - 质量等级：✅ 良好
  - 平均长度：342 tokens | P95：1,024 tokens | 最大：2,856 tokens
  - 空回复：3 条 (0.1%) ✅
  - 重复对：12 条 (0.5%) ✅
  - 格式：instruction-output
```

**显存评估（Step 3 结果）：**
```
💾 显存估算：Qwen2-7B + QLoRA 4-bit
  - 模型权重：3.5 GB | LoRA 参数：0.02 GB
  - 激活值 (seq=1024, bs=4)：1.8 GB
  - 优化器状态：0.3 GB | 系统开销：0.6 GB
  - 总计：6.2 GB

你的显卡：16 GB → 剩余 9.8 GB (61%) ✅ 安全
可以开 bs=4, gradient_accumulation=4。
```

**参数推荐（Step 4 结果）：**
```
⚙️ 推荐配置：
  rank=8         ← 2,340 条数据，常规区间，8 是安全选择
  alpha=16       ← 标准 2×r，平衡学习强度
  target_modules=[q_proj, v_proj]  ← 聊天任务，Q/V 足够了
  dropout=0.05   ← 1k-10k 区间标准值
  lr=2e-4        ← r=8 基准值，聊天任务不调整
  epochs=3       ← 500-5k 区间，3 轮稳妥
  batch_size=4   ← 16GB 显存对应配置
  gradient_accumulation=4  ← 目标有效 batch=16
```

然后等待用户确认再生成脚本。

**常见反馈的回应：**
- 用户说 "rank 能不能大一点" → "可以调到 16，但 2,340 条数据用 16 风险不大。如果你想要更强效果可以试。"
- 用户说 "学习率太高了吧" → "2e-4 是 LoRA 的行业基准。想保守的话可以降到 1e-4，但收敛会慢一些。"
- 用户说 "就按这个来" → 调用 ScriptBuilder 生成文件，列出生成的文件路径和下一步操作。生成后询问："训练完成后需要我帮你生成评估脚本吗？可以对比基座模型和微调模型的效果差异。"

---

## 继续预训练 (CPT)

CPT 和 SFT 有本质区别：CPT 是教模型新领域的知识（如医学、法律），SFT 是教模型按特定格式回答问题。

### CPT vs SFT 参数差异

```
参数          SFT（指令微调）          CPT（继续预训练）
─────────────────────────────────────────────────────
rank          r=4-32（数据驱动）       r=16-64（需要更强知识注入）
alpha         2×r                      2×r（标准比例不变）
target_mods   [q,v] 或 [q,k,v,o]      [q,k,v,o,up,down,gate]（全层训练）
lr            1e-4 ~ 3e-4              5e-5 ~ 1e-4（更保守）
epochs        1-5                      1-3（数据通常较多）
dropout       0.05-0.15                0.05（CPT 数据量大，不需要高 dropout）
数据格式       instruction/output 对    纯文本段落（JSONL 每行一个 text 字段）
模板          需要 instruction 模板     不需要模板，直接拼接文本
packing       不必须                    强烈建议（短文本拼成长序列）
目标有效batch 16                       32-64（更大 batch 稳定训练）
```

### CPT 数据格式

```
{"text": "第一章：细胞生物学概述\n\n细胞是生命活动的基本单位..."}
{"text": "深度学习在医疗影像分析中的应用日益广泛..."}
```

### CPT 执行流程

1. 确认用户意图是 CPT 而非 SFT（关键区分：用户是否想在特定领域注入知识？）
2. 数据检查：确认每行有 `text` 字段、估算总 token 量（CPT 建议 > 10M tokens）
3. 显存计算：target_modules 数量多（7 个），优化器显存比 SFT 高 3-4×
4. 参数推荐：用 CPT 专用规则（上表），启用 packing
5. 生成脚本：使用纯文本拼接模板，不添加 instruction 前缀

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

### 训练前：环境检查（30 秒搞定）

在跑训练脚本之前，先确认这三项：

```bash
# 1. CUDA 可用吗？
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0)}')"

# 2. 显存够吗？
nvidia-smi --query-gpu=memory.free --format=csv,noheader

# 3. 数据格式对吗？
python -c "import json; [json.loads(l) for l in open('你的数据.jsonl')]"
```

第三项如果报 `JSONDecodeError`，说明文件不是合法的 JSONL。用 `/lora:check-data` 诊断。

### 训练中：常见报错速查

| 报错信息（关键词） | 原因 | 解决方法（按优先级） |
|-------------------|------|---------------------|
| `CUDA out of memory` | 显存不够 | ①降 `max_seq_length`（最有效，砍半显存减 ~40%） ②降 `batch_size` 到 1 ③确认 `load_in_4bit=True` ④设 `gradient_checkpointing=True` |
| `No module named 'torch'` | PyTorch 没装 | `pip install lora-trainer[train]` 一键安装全部依赖 |
| `No module named 'transformers'` | transformers 没装 | 同上，或 `pip install transformers` |
| `No module named 'bitsandbytes'` | bitsandbytes 没装 | `pip install bitsandbytes`；Windows 用户可能需要预编译 wheel |
| `CUDA error: no kernel image` | CUDA 版本与 PyTorch 不匹配 | `nvcc --version` 看 CUDA 版本，去 pytorch.org 装对应版本 |
| `tokenizer.pad_token is None` | tokenizer 缺 pad_token | 在脚本中加 `tokenizer.pad_token = tokenizer.eos_token` |
| `expected str instance, NoneType found` | 数据中有空字段 | 用 `/lora:check-data` 找空字段行号，过滤或修复 |
| `target_modules not found` | LoRA 目标模块名不匹配 | `print(model)` 查实际层名，修正 `target_modules` |
| `DataLoader worker died` | 内存不足或数据损坏 | 设 `dataloader_num_workers=0` 排查，检查 JSONL 完整性 |
| `CUDA error: device-side assert` | 标签越界或 NaN | 检查 tokenizer vocab size，确认 labels 无负值或越界 |
| `RuntimeError: expected scalar type Half but found Float` | 精度不匹配 | 确保模型和输入都是 fp16/bf16，或统一用 fp32 |

### 训练效果问题

| 现象 | 怎么诊断 | 怎么修 |
|------|---------|--------|
| loss 不下降 | 看 tensorboard：`tensorboard --logdir ./output/logs` | ①确认数据格式正确 ②lr 太低 → 当前 lr×2 ③数据有 NaN → 清洗 |
| loss 下降但输出乱码 | 对比训练前后模型输出 | ①lr 太高 → 降为 1/2 ②数据质量差 ③epochs 太多 |
| loss 中途突然 spike | 找到 spike 的 step，定位对应 batch | ①降 lr ②确认 `max_grad_norm=1.0` 已设 ③检查数据中是否有超长/乱码样本 |
| 过拟合（train loss ≪ eval loss） | 留 5-10% 数据做 eval | ①dropout +0.05 ②rank 减半 ③epochs -1 |
| 效果不达预期 | 检查输出格式是否遵循训练数据 | ①数据覆盖目标场景？ ②提高 rank ③增加数据量 |
| 微调后幻觉增多 | 对比基座模型输出 | ①降 lr（×0.5） ②降 rank ③换 DoRA ④增加数据多样性 |

### 训练完成后用这个诊断

如果训练完了但效果不好，直接：

```
/lora:debug ./output/logs
```

或者把训练日志/报错贴给我，我会分析 loss 曲线、梯度变化、学习率调度，帮你定位问题。

---

## 危险操作

以下操作必须先确认再执行：
- 写文件到磁盘（Write 工具）
- 修改已有训练脚本
- 建议安装或卸载 Python 包
- `--auto` 模式下开始自动训练

---

## 训练时间估算

给用户一个大致预期，避免"要训练多久"被问两遍。

```
硬件假设: RTX 4090 (24GB), QLoRA 4-bit, 7B 模型, FP16

1,000 条, seq=2048, bs=4, 3 epochs → ~30-60 分钟
5,000 条, seq=2048, bs=4, 2 epochs → ~2-4 小时
20,000 条, seq=2048, bs=4, 1 epoch → ~3-6 小时
50,000 条, seq=2048, bs=8, 1 epoch → ~5-10 小时

CPU 推断公式: (样本数 × 平均长度 × epochs) / (GPU算力系数 × batch_size)
              RTX 4090 算力系数 ≈ 2000 tokens/sec (7B QLoRA)
```

## 多卡训练

用户有多张 GPU 时，给出简要指引，不改动生成的脚本。

- **2× GPU, 相同型号** → 推荐 DeepSpeed ZeRO-2。启动：`deepspeed --num_gpus=2 train.py --deepspeed ds_config.json`
- **4× GPU+** → 推荐 DeepSpeed ZeRO-3。显存几乎线性扩展。
- **单卡就够了** → 不要推荐多卡。7B QLoRA 用 24GB 单卡绰绰有余。
- **70B+ 模型** → 必须多卡或量化到 4-bit + 单卡大显存（48GB+）。

多卡训练的 batch_size 计算：`per_device_batch_size × num_gpus × gradient_accumulation`。

## Packing（序列打包）

把多个短训练样本拼成一条长序列，提升 GPU 利用率。

```
适用场景:
  ✅ CPT（文本段落通常 < max_length）
  ✅ 短问答（平均 < 500 tokens, max_length=4096）
  ❌ 长文本（平均 > 2048 tokens, 浪费 token 去拼接）
  ❌ 严格需要 attention mask 的场景（packing 会干扰 mask）

效果:  短数据场景下吞吐量提升 3-10×
代价:  实现复杂度增加（需要正确构造 position_ids 和 attention_mask）
PEFT 库方式: 使用 ConstantLengthDataset 或 SFTTrainer 的 packing=True
```

---

## 参考资料

四个参考文件，按需加载：

- **model-catalog.md** — 精确模型规格（hidden_dim、layers、vocab）。显存计算前必须查。
- **recipes.md** — 预置配方（chat/code/math/roleplay）。用户不想调参时直接套用。
- **vram-reference.md** — 常见模型显存快速参考。给用户一个大致数字时用。
- **faq.md** — 原理性问答。用户问"为什么 r=8"、"LoRA 和全量微调什么区别"时查。

加载方式：
**Read** `${CLAUDE_PLUGIN_ROOT}/skills/lora-trainer/references/<文件名>`

---

## 原则重申

这六条定义了你的工作质量底线：

1. 先分析数据，再推荐参数。
2. 先计算显存，再确认 batch size。
3. 每个推荐参数给出理由。
4. 生成脚本前获取用户确认。
5. 数据问题优先暴露，不隐藏。
6. 能动手就不动嘴——直接帮用户跑分析、算显存、生成脚本，不要只给命令让用户自己执行。
