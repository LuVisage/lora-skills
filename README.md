# 🎯 LoRA Trainer — Claude Code Plugin

> 大模型 LoRA/QLoRA 微调智能助手。
> 在 Claude Code 里输入 `/lora:analyze`，AI 帮你从数据分析到训练脚本生成一条龙搞定。

## 🚀 安装

```bash
# 安装到 Claude Code
claude plugin install https://github.com/LuVisage/lora-skills --scope user
```

## 🎮 使用

```
# 完整分析（数据 → 显存 → 参数 → 脚本）
/lora:analyze ./data/train.jsonl qwen2-7b chat

# 一键炼丹（精简输出）
/lora:cook ./data/train.jsonl --model qwen2-7b --task chat

# 快速检查数据质量
/lora:check-data ./data/train.jsonl

# 自然语言也能触发
"帮我看看这个数据集，我想微调 Qwen2-7B"
```

## 📂 项目结构

```
lora-trainer/
├── .claude-plugin/plugin.json   # Plugin 元数据
├── commands/                    # 斜杠命令
│   ├── analyze.md               # /lora:analyze
│   ├── cook.md                  # /lora:cook
│   └── check-data.md            # /lora:check-data
├── agents/                      # 专用子 Agent
│   ├── data-analyzer.md         # 数据诊断
│   ├── memory-estimator.md      # 显存评估
│   └── script-generator.md      # 脚本生成
├── skills/lora-trainer/         # 自动激活 Skill
│   ├── SKILL.md                 # 核心！内含所有推荐规则
│   └── references/              # 参考资料（纯数据）
│       ├── model-catalog.md     # 20+ 模型规格
│       ├── recipes.md           # 预置配方
│       └── faq.md               # 常见问题
├── scripts/                     # Python 计算引擎（不做判断）
│   ├── analyzer.py
│   ├── memory_calc.py
│   ├── lora_advisor.py
│   ├── script_builder.py
│   └── evaluator.py
├── templates/                   # 代码模板
├── examples/                    # 示例数据
└── requirements.txt
```

## 🧠 设计理念

- **Claude 是大脑** — 所有推荐规则和判断逻辑在 SKILL.md 中
- **Python 是手脚** — 只做精确数值计算（显存、数据统计、脚本模板填充）
- **渐进式披露** — SKILL.md < 500 行，详细数据在 references/

## ⚙️ 依赖

```bash
pip install -r requirements.txt
```

实际训练还需要：
- NVIDIA GPU (建议 8GB+)
- CUDA Toolkit
- PyTorch with CUDA support

## 📄 License

MIT
