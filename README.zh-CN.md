<div align="center">

<img src="https://img.shields.io/pypi/v/lora-trainer?style=flat-square&color=blue" alt="PyPI version">
<img src="https://img.shields.io/npm/v/lora-trainer?style=flat-square&color=blue" alt="npm version">
<img src="https://img.shields.io/github/license/LuVisage/lora-skills?style=flat-square" alt="license">
<img src="https://img.shields.io/github/stars/LuVisage/lora-skills?style=flat-square" alt="stars">
<img src="https://img.shields.io/badge/platform-CLI%20%7C%20Claude%20Code%20%7C%20Cursor%20%7C%20Codex-orange?style=flat-square" alt="platforms">
<img src="https://img.shields.io/badge/python-≥3.10-blue?style=flat-square" alt="python">

</div>

# lora-trainer

> 数据分析、显存规划、有理有据的超参数推荐、可运行的训练脚本 — 一条命令全搞定。独立 CLI 工具，也是 Claude Code / Cursor / Codex 的 AI Agent 技能。

[English](README.md) | **简体中文**

## 为什么你需要这个工具？

用 LoRA 微调大模型看似简单，坑却很多：显存拍脑袋估算，一跑就 OOM；超参数靠猜，GPU 小时白白浪费；从过时的教程拼凑训练脚本，引入各种隐蔽 bug。

**lora-trainer** 帮你避开这些坑。它既是**独立 CLI 工具**（`pip install lora-trainer`），也是 **Claude Code / Cursor / Codex 插件**。给它 JSONL 数据，剩下的全自动：格式检测、基于 40+ 模型规格库的精准显存计算、带理由的超参数推荐、以及一份可直接运行的训练脚本。

## 快速开始

### CLI 模式（推荐）

```bash
pip install lora-trainer
lora-trainer cook --data ./data/train.jsonl --model qwen2-7b --task chat
```

搞定。打印数据报告 → 显存明细 → 参数推荐表（含理由）→ 在 `./output/` 下生成三个文件。

### Claude Code / Cursor / Codex 模式

```bash
npx skills add LuVisage/lora-skills
```

然后：

```
/lora:analyze ./data/train.jsonl qwen2-7b chat
```

## 安装方式

```bash
# pip 安装（独立 CLI）
pip install lora-trainer             # CLI + 分析 + 推荐
pip install lora-trainer[train]      # 完整版，含 torch/transformers

# npm / npx 安装（Claude Code / Cursor / Codex）
npx skills add LuVisage/lora-skills

# Claude Code 插件安装
claude plugin install https://github.com/LuVisage/lora-skills
```

## CLI 命令

```
lora-trainer
├── analyze <数据>           📊 数据质量诊断
│   └── --json               机器可读输出
├── recommend                ⚙️ 超参数推荐
│   └── -n 样本数 -m 模型 -t 任务 -g 显存
├── memory <模型>            💾 显存估算
│   └── -s 序列长度 -b batch -r rank --modules N
├── cook                     🚀 一键生成训练脚本
│   ├── -d 数据 -m 模型 -t 任务 -g 显存
│   ├── --interactive        逐项确认参数
│   └── --rank/--lr/--epochs 覆盖推荐值
└── evaluate <测试数据>      📏 评估脚本生成
```

### 使用示例

```bash
# 分析训练数据质量
lora-trainer analyze ./data/train.jsonl
lora-trainer analyze ./data/train.jsonl --json | jq .quality

# 获取超参数推荐
lora-trainer recommend --samples 2340 --model 7b --task chat
lora-trainer recommend -n 5000 -m 13b -t code -g 16

# 估算任意模型的显存需求
lora-trainer memory qwen2-7b
lora-trainer memory llama3-8b --seq-length 4096 --batch-size 8
lora-trainer memory deepseek-v3 --modules 7

# 一键生成训练脚本（自动检测 GPU、数据格式、最优参数）
lora-trainer cook --data ./data/train.jsonl --model qwen2-7b --task chat
lora-trainer cook -d ./data/code.jsonl -t code --rank 16 --lr 3e-4

# 生成评估脚本
lora-trainer evaluate ./data/test.jsonl
lora-trainer evaluate ./data/test.jsonl --format messages
```

## 功能

- **数据诊断** — 自动识别 JSONL 格式（instruction-output / messages / conversations），标记空回复、重复对、控制字符、长度异常值和中英混合比例
- **显存计算器** — 内置 40+ 模型、13 大家族的规格库（Qwen2/2.5、LLaMA2/3/3.1/3.2、Mistral、Mixtral、ChatGLM、DeepSeek、Yi、Baichuan2、Phi-3、Gemma、InternLM2），MoE 感知，精确估算模型权重、激活值、优化器状态和额外开销
- **超参数推荐** — rank、alpha、target modules、dropout、learning rate、epochs、batch size、gradient accumulation，每个值都附带推荐理由；任务感知（chat/code/math/roleplay/CPT）
- **脚本生成** — 直接输出完整 QLoRA 训练脚本（Transformers + PEFT + BitsAndBytes + flash_attention_2）、推理脚本和 YAML 配置文件；只需填入模型名和数据路径即可运行
- **模型评估** — 生成基座模型 vs LoRA 模型的对比评估脚本
- **中文优先** — 全中文界面 + emoji，面向非技术用户；同时支持 `--json` 参数导出机器可读输出

## 适用场景

lora-trainer 适用于以下情况：

- 想微调模型但不确定该选什么超参数
- 想在训练前确认显存是否够用（而不是跑起来才发现 OOM）
- 有一份 JSONL 数据集，想在训练前做质量检查
- 想以最快速度从原始数据到可运行的训练脚本
- 正在做 SFT、CPT（继续预训练）或 LoRA/QLoRA 领域适配
- 需要评估微调是否真的提升了模型效果

支持任意 NVIDIA GPU（7B 模型 QLoRA 微调建议 8 GB 以上显存）。

## 支持模型

Qwen2 (0.5B–72B)、Qwen2.5 (0.5B–72B)、LLaMA3 (8B–70B)、LLaMA3.1/3.2 (1B–405B)、LLaMA2 (7B–70B)、Mistral 7B、Mistral Nemo 12B、Mistral Large 123B、Mixtral 8×7B、ChatGLM3/4、DeepSeek 7B/67B/V2/V3、Yi (6B–34B)、Baichuan2 (7B–13B)、Phi-3 (mini/small/medium)、Gemma (2B–7B)、InternLM2 (7B–20B)。

缺少你需要的模型？在 `references/model-catalog.md` 加一行，或直接在 `scripts/memory_calc.py` 的 `MODEL_DB` 字典里加一条。

## 项目结构

```
lora-trainer/
├── .claude-plugin/                # Claude Code 插件清单
│   └── plugin.json
├── skills/lora-trainer/           # 🔥 核心技能（AI Agent 大脑）
│   ├── SKILL.md                   # 全部推荐规则
│   └── references/                # 纯数据，按需加载
│       ├── model-catalog.md       # 模型规格（40+）
│       ├── recipes.md             # 预设配置方案
│       ├── vram-reference.md      # 显存快速查询表
│       └── faq.md                 # 常见问题
├── commands/                      # 斜杠命令定义（4 个）
│   ├── analyze.md                 # /lora:analyze
│   ├── cook.md                    # /lora:cook
│   ├── check-data.md              # /lora:check-data
│   └── debug.md                   # /lora:debug（新增）
├── agents/                        # 子 Agent 定义（3 个）
├── hooks/                         # 会话生命周期钩子
├── cli/                           # 独立 CLI（pip install 入口）
│   ├── main.py
│   └── commands/
├── scripts/                       # Python 计算层
│   ├── analyzer.py
│   ├── memory_calc.py
│   ├── lora_advisor.py
│   ├── script_builder.py
│   └── evaluator.py
├── templates/                     # 训练脚本模板
├── examples/                      # 示例 JSONL 训练数据
├── docs/                          # 博客文章 & 文档
├── pyproject.toml                 # Python 包配置
├── package.json                   # npm 包配置
└── requirements.txt               # Python 依赖
```

## 设计理念

- **Agent 是大脑** — 所有推荐规则和判断逻辑都在 `SKILL.md` 中。Agent 做决策，脚本做计算。
- **CLI 是界面** — `click` + `rich` 提供美观的终端交互体验。每个命令都支持 `--json` 标志方便脚本调用。
- **脚本是双手** — Python 只处理精确数值工作：显存计算、数据统计和模板渲染。Python 中不存在 `if/else` 决策逻辑。
- **渐进式披露** — SKILL.md 约 520 行。深度参考数据放在 `references/` 中，按需加载。

## 环境要求

```bash
pip install lora-trainer             # CLI + 核心依赖 (click, rich, numpy, pyyaml)
pip install lora-trainer[train]      # 完整训练栈 (torch, transformers, peft, ...)
```

训练需要 NVIDIA GPU（7B 模型 QLoRA 微调建议 8GB+ 显存）、CUDA Toolkit 和 CUDA 版 PyTorch。

## 参与贡献

欢迎提 Issue 和 PR。提交前请运行冒烟测试：

```bash
# Python 模块测试
python -c "
from scripts.analyzer import quick_analyze
from scripts.memory_calc import quick_calc
from scripts.lora_advisor import quick_recommend
from cli.main import main
r = quick_analyze('examples/sample_data.jsonl')
m = quick_calc('qwen2-7b')
c = quick_recommend(r['length']['total_samples'], '7b', 'chat')
print('✅ All OK')
"

# CLI 冒烟测试
python -m cli.main --help
python -m cli.main analyze examples/sample_data.jsonl
python -m cli.main recommend -n 2340 -m 7b -t chat
python -m cli.main memory qwen2-7b
```

## 许可证

MIT
