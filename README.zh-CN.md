<div align="center">

<img src="https://img.shields.io/npm/v/lora-trainer?style=flat-square&color=blue" alt="npm version">
<img src="https://img.shields.io/github/license/LuVisage/lora-skills?style=flat-square" alt="license">
<img src="https://img.shields.io/github/stars/LuVisage/lora-skills?style=flat-square" alt="stars">
<img src="https://img.shields.io/badge/platform-Claude%20Code%20%7C%20Cursor%20%7C%20Codex-orange?style=flat-square" alt="platforms">

</div>

# lora-trainer

> LoRA / QLoRA 大模型微调智能助手，为 AI 编程 Agent 而生。
> 一行命令，从原始数据到可运行的训练脚本。

[English](README.md) | **简体中文**

## 为什么你需要这个工具？

用 LoRA 微调大语言模型，不应该需要读五篇博客、手动算显存、然后从 2023 年的 Colab 笔记里复制粘贴训练代码。但现状就是这样：凭感觉选 rank，在 Excel 里估算显存，祈祷代码能跑通。

**lora-trainer** 把决策规则内置到了你的 AI Agent 里。它会分析你的数据，结合 20+ 模型的精确规格计算显存需求，给出有理有据的超参数建议，然后生成一份语法正确的完整训练脚本 —— 全程通过斜杠命令完成。

## 快速开始

```bash
npx skills add LuVisage/lora-skills
```

然后在 Claude Code、Cursor 或 Codex 中：

```
/lora:analyze ./data/train.jsonl qwen2-7b chat
```

搞定。它会打印数据报告、显存分解、推荐 LoRA 参数及理由，并在 `./output/` 下生成三个文件。

## 安装方式

```bash
# npx 安装（自动检测 Claude Code / Cursor / Codex）
npx skills add LuVisage/lora-skills

# Claude Code 插件安装
claude plugin install https://github.com/LuVisage/lora-skills

# npm 安装
npm install lora-trainer
```

## 命令

| 命令 | 功能 |
|---------|-------------|
| `/lora:analyze <数据> [模型] [任务] [显存]` | 全流程：数据诊断 → 显存估算 → 超参数推荐 → 脚本生成 |
| `/lora:cook <数据> [模型] [任务] [--auto]` | 同 analyze，但输出更精简；`--auto` 确认后自动开始训练 |
| `/lora:check-data <数据>` | 快速扫描：样本数、长度分布、空响应、重复项 |

支持自然语言触发：说"帮我看这个数据能不能微调"，技能自动激活。

## 功能

- **数据诊断** — 自动识别格式（instruction-output / messages / conversations），标记空响应、重复、控制字符和长度异常值
- **显存计算器** — 内置 20+ 模型规格库（Qwen2、LLaMA3、Mistral、ChatGLM、DeepSeek、Yi、Baichuan2、Phi-3、Gemma、InternLM2），精确估算模型权重、激活值、优化器状态和额外开销
- **超参数推荐** — rank、alpha、target modules、dropout、learning rate、epochs、batch size，每个值都附带推荐理由
- **脚本生成** — 直接输出完整 QLoRA 训练脚本（Transformers + PEFT + BitsAndBytes）、推理脚本和 YAML 配置文件
- **内置知识** — SKILL.md 中内置 57 条参数规则；Python 脚本只负责精确数值计算；参考数据全部在 `references/` 中

## 支持模型

Qwen2 (0.5B–72B)、LLaMA3 (8B–70B)、LLaMA2 (7B–70B)、Mistral 7B、Mixtral 8×7B、ChatGLM3/4、DeepSeek (7B–67B)、Yi (6B–34B)、Baichuan2 (7B–13B)、Phi-3 (mini/small/medium)、Gemma (2B–7B)、InternLM2 (7B–20B)。

缺少你需要的模型？在 `references/model-catalog.md` 里加一行就行。

## 项目结构

```
lora-trainer/
├── .claude-plugin/plugin.json   # 插件清单
├── commands/                    # 斜杠命令
│   ├── analyze.md               # /lora:analyze
│   ├── cook.md                  # /lora:cook
│   └── check-data.md            # /lora:check-data
├── agents/                      # 子 Agent
│   ├── data-analyzer.md
│   ├── memory-estimator.md
│   └── script-generator.md
├── skills/lora-trainer/         # 自动激活技能
│   ├── SKILL.md                 # 全部推荐规则（行为层）
│   └── references/              # 纯数据，按需加载
│       ├── model-catalog.md     # 模型规格（20+）
│       ├── recipes.md           # 预设配置方案
│       └── faq.md               # 常见问题
├── scripts/                     # Python 计算层
│   ├── analyzer.py              # 数据统计
│   ├── memory_calc.py           # 显存估算
│   ├── lora_advisor.py          # 参数推荐引擎
│   ├── script_builder.py        # 训练脚本生成
│   └── evaluator.py             # 模型评估
├── templates/                   # 代码模板
├── examples/                    # 示例 JSONL 数据
└── requirements.txt
```

## 设计理念

- **Agent 是大脑** — 所有推荐规则和判断逻辑都在 `SKILL.md` 中。Agent 做决策，脚本做计算。
- **脚本是双手** — Python 只处理精确数值工作：显存计算、数据统计和模板渲染。Python 中不存在 `if/else` 决策逻辑。
- **渐进式披露** — SKILL.md 控制在 500 行以内。深度参考数据放在 `references/` 中，按需加载。

## 环境要求

```bash
pip install -r requirements.txt
```

训练需要 NVIDIA GPU（7B 模型 QLoRA 微调建议 8GB+ 显存）、CUDA Toolkit 和 CUDA 版 PyTorch。

## 参与贡献

欢迎提 Issue 和 PR。提交前请运行冒烟测试：

```bash
python -c "
from scripts.analyzer import quick_analyze
from scripts.memory_calc import quick_calc
from scripts.lora_advisor import quick_recommend
r = quick_analyze('examples/sample_data.jsonl')
m = quick_calc('qwen2-7b')
c = quick_recommend(r['length']['total_samples'], '7b', 'chat')
print('OK')
"
```

## 许可证

MIT
