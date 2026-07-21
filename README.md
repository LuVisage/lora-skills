<div align="center">

<img src="https://img.shields.io/npm/v/lora-trainer?style=flat-square&color=blue" alt="npm version">
<img src="https://img.shields.io/github/license/LuVisage/lora-skills?style=flat-square" alt="license">
<img src="https://img.shields.io/github/stars/LuVisage/lora-skills?style=flat-square" alt="stars">
<img src="https://img.shields.io/badge/platform-Claude%20Code%20%7C%20Cursor%20%7C%20Codex-orange?style=flat-square" alt="platforms">

</div>

# lora-trainer

> LoRA / QLoRA fine-tuning assistant for AI coding agents.
> One command from raw data to a runnable training script.

## Why?

Fine-tuning a large language model with LoRA shouldn't require reading five blog posts and memorizing the PEFT API. But today it does: you pick a rank by guess, calculate VRAM in a spreadsheet, and copy-paste training snippets from a 2023 Colab notebook.

**lora-trainer** puts the decision rules inside your AI agent. It analyzes your data, calculates exact memory requirements against a catalog of 20+ model specs, recommends hyperparameters with reasoning, and generates a complete, syntax-checked training script — all through a slash command.

## Quick Start

```bash
npx skills add LuVisage/lora-skills
```

Then in Claude Code, Cursor, or Codex:

```
/lora:analyze ./data/train.jsonl qwen2-7b chat
```

That's it. It will print a data report, a VRAM breakdown, recommended LoRA parameters with explanations, and save three files to `./output/`.

## Install

```bash
# npx (auto-detects Claude Code / Cursor / Codex)
npx skills add LuVisage/lora-skills

# Claude Code plugin
claude plugin install https://github.com/LuVisage/lora-skills

# npm
npm install lora-trainer
```

## Commands

| Command | What it does |
|---------|-------------|
| `/lora:analyze <data> [model] [task] [gpu]` | Full pipeline: data audit → VRAM estimate → hyperparameter recommendation → script generation |
| `/lora:cook <data> [model] [task] [--auto]` | Same as analyze but concise output; `--auto` starts training after confirmation |
| `/lora:check-data <data>` | Quick scan: sample count, length distribution, empty responses, duplicates |

Auto-triggers on natural language: say "帮我看看这个数据能不能微调" and the skill activates automatically.

## Features

- **Data audit** — detects format (instruction-output, messages, conversations), flags empty responses, duplicates, control characters, and length outliers
- **VRAM calculator** — built-in spec catalog for 20+ models (Qwen2, LLaMA3, Mistral, ChatGLM, DeepSeek, Yi, Baichuan2, Phi-3, Gemma, InternLM2); estimates model weights, activations, optimizer states, and overhead
- **Hyperparameter recommendation** — rank, alpha, target modules, dropout, learning rate, epochs, batch size; every value comes with a reason
- **Script generation** — produces a complete QLoRA training script (Transformers + PEFT + BitsAndBytes), an inference script, and a YAML config
- **Built-in knowledge** — 57 parameter rules direct in SKILL.md; Python scripts handle only exact numeric computation; all reference data in `references/`

## Supported Models

Qwen2 (0.5B–72B), LLaMA3 (8B–70B), LLaMA2 (7B–70B), Mistral 7B, Mixtral 8×7B, ChatGLM3/4, DeepSeek (7B–67B), Yi (6B–34B), Baichuan2 (7B–13B), Phi-3 (mini/small/medium), Gemma (2B–7B), InternLM2 (7B–20B).

Missing a model? Add one line to `references/model-catalog.md`.

## Project Structure

```
lora-trainer/
├── .claude-plugin/plugin.json   # Plugin manifest
├── commands/                    # Slash commands
│   ├── analyze.md               # /lora:analyze
│   ├── cook.md                  # /lora:cook
│   └── check-data.md            # /lora:check-data
├── agents/                      # Sub-agents
│   ├── data-analyzer.md
│   ├── memory-estimator.md
│   └── script-generator.md
├── skills/lora-trainer/         # Auto-activating skill
│   ├── SKILL.md                 # All recommendation rules (behavioral)
│   └── references/              # Pure data, loaded on demand
│       ├── model-catalog.md     # Model specs (20+)
│       ├── recipes.md           # Preset configurations
│       └── faq.md               # Common questions
├── scripts/                     # Python computation layer
│   ├── analyzer.py              # Data statistics
│   ├── memory_calc.py           # VRAM estimation
│   ├── lora_advisor.py          # Parameter recommendation engine
│   ├── script_builder.py        # Training script generator
│   └── evaluator.py             # Model evaluation
├── templates/                   # Code templates
├── examples/                    # Sample JSONL data
└── requirements.txt
```

## Design

- **Agent is the brain** — all recommendation rules and judgment live in `SKILL.md`. The agent decides; scripts compute.
- **Scripts are the hands** — Python handles exact numeric work: VRAM math, data statistics, and template rendering. No `if/else` decision logic in Python.
- **Progressive disclosure** — SKILL.md is under 500 lines. Deep reference data stays in `references/` and loads only when needed.

## Requirements

```bash
pip install -r requirements.txt
```

Training requires an NVIDIA GPU (8 GB+ recommended for 7B models with QLoRA), CUDA Toolkit, and PyTorch with CUDA.

## Contributing

Issues and PRs welcome. Before submitting, run the smoke test:

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

## License

MIT
