<div align="center">

<img src="https://img.shields.io/pypi/v/lora-trainer?style=flat-square&color=blue" alt="PyPI version">
<img src="https://img.shields.io/npm/v/lora-trainer?style=flat-square&color=blue" alt="npm version">
<img src="https://img.shields.io/github/license/LuVisage/lora-skills?style=flat-square" alt="license">
<img src="https://img.shields.io/github/stars/LuVisage/lora-skills?style=flat-square" alt="stars">
<img src="https://img.shields.io/badge/platform-CLI%20%7C%20Claude%20Code%20%7C%20Cursor%20%7C%20Codex-orange?style=flat-square" alt="platforms">
<img src="https://img.shields.io/badge/python-≥3.10-blue?style=flat-square" alt="python">

</div>

# lora-trainer

> Analyze data, plan VRAM, get hyperparameter recommendations with reasoning, and generate a runnable training script — all in one command. Works as a standalone CLI and as an AI agent skill for Claude Code, Cursor, and Codex.

**English** | [简体中文](README.zh-CN.md)

## Why?

Fine-tuning an LLM with LoRA is powerful but full of hidden pitfalls: guessing VRAM leads to OOM errors, picking hyperparameters by trial and error wastes GPU hours, and stitching together training scripts from outdated tutorials introduces silent bugs.

**lora-trainer** removes the guesswork. It is both a **standalone CLI tool** (`pip install lora-trainer`) and a **Claude Code / Cursor / Codex plugin**. Point it at your JSONL data and it handles everything: automatic format detection, exact VRAM calculation against a catalog of 40+ model specs, hyperparameter recommendations with reasoning, and a complete, runnable training script.

## Quick Start

### CLI (recommended)

```bash
pip install lora-trainer
lora-trainer cook --data ./data/train.jsonl --model qwen2-7b --task chat
```

That's it. It prints a data report, a VRAM breakdown, a parameter table with reasoning, and generates three files in `./output/`.

### Claude Code / Cursor / Codex

```bash
npx skills add LuVisage/lora-skills
```

Then:

```
/lora:analyze ./data/train.jsonl qwen2-7b chat
```

## Install

```bash
# pip (standalone CLI)
pip install lora-trainer             # CLI + analysis + recommendations
pip install lora-trainer[train]      # full stack with torch/transformers

# npm / npx (Claude Code / Cursor / Codex)
npx skills add LuVisage/lora-skills

# Claude Code plugin
claude plugin install https://github.com/LuVisage/lora-skills
```

## CLI Commands

```
lora-trainer
├── analyze <data>           📊 Data quality audit
│   └── --json               Machine-readable output
├── recommend                ⚙️ Hyperparameter recommendation
│   └── -n SAMPLES -m MODEL -t TASK -g GPU
├── memory <model>           💾 VRAM estimation
│   └── -s SEQ -b BATCH -r RANK --modules N
├── cook                     🚀 One-command script generation
│   ├── -d DATA -m MODEL -t TASK -g GPU
│   ├── --interactive        Confirm each parameter
│   └── --rank/--lr/--epochs Override recommendations
└── evaluate <test.jsonl>    📏 Evaluation script
```

### Examples

```bash
# Analyze your training data
lora-trainer analyze ./data/train.jsonl
lora-trainer analyze ./data/train.jsonl --json | jq .quality

# Get hyperparameter recommendations
lora-trainer recommend --samples 2340 --model 7b --task chat
lora-trainer recommend -n 5000 -m 13b -t code -g 16

# Estimate VRAM for any model
lora-trainer memory qwen2-7b
lora-trainer memory llama3-8b --seq-length 4096 --batch-size 8
lora-trainer memory deepseek-v3 --modules 7

# Generate training scripts (auto-detects GPU, format, and optimal params)
lora-trainer cook --data ./data/train.jsonl --model qwen2-7b --task chat
lora-trainer cook -d ./data/code.jsonl -t code --rank 16 --lr 3e-4

# Generate evaluation script
lora-trainer evaluate ./data/test.jsonl
lora-trainer evaluate ./data/test.jsonl --format messages
```

## Features

- **Data audit** — auto-detects JSONL format (instruction-output, messages, conversations), flags empty responses, duplicates, control characters, length outliers, and bilingual ratio
- **VRAM calculator** — built-in spec catalog covering 40+ models across 12 families (Qwen2/2.5, LLaMA2/3/3.1/3.2, Mistral, Mixtral, ChatGLM, DeepSeek, Yi, Baichuan2, Phi-3, Gemma, InternLM2); MoE-aware; breaks down model weights, activations, optimizer states, and overhead
- **Hyperparameter recommendation** — rank, alpha, target modules, dropout, learning rate, epochs, batch size, gradient accumulation; every value comes with a reason; task-aware across chat, code, math, roleplay, and CPT
- **Script generation** — produces a complete QLoRA training script (Transformers + PEFT + BitsAndBytes + flash_attention_2), an inference script, and a YAML config; just fill in your model name and data path
- **Evaluation** — generates side-by-side comparison scripts (base model vs LoRA) for post-training assessment
- **Chinese-first UX** — all output in Chinese with emoji, designed for non-technical users; `--json` flag available for programmatic use

## When to Use

Use lora-trainer when you:

- Want to fine-tune a model but don't know which hyperparameters to pick
- Need to know if your GPU has enough VRAM *before* starting training
- Have a JSONL dataset and want a quality check before training
- Want the fastest path from raw data to a running training script
- Are doing SFT, CPT (continued pre-training), or domain adaptation with LoRA/QLoRA
- Need to evaluate whether fine-tuning actually improved your model

Works with any NVIDIA GPU (8 GB+ recommended for 7B models with QLoRA).

## Supported Models

Qwen2 (0.5B–72B), Qwen2.5 (0.5B–72B), LLaMA3 (8B–70B), LLaMA3.1/3.2 (1B–405B), LLaMA2 (7B–70B), Mistral 7B, Mistral Nemo 12B, Mistral Large 123B, Mixtral 8×7B, ChatGLM3/4, DeepSeek 7B/67B/V2/V3, Yi (6B–34B), Baichuan2 (7B–13B), Phi-3 (mini/small/medium), Gemma (2B–7B), InternLM2 (7B–20B).

Missing a model? Add one line to `references/model-catalog.md` or directly to the `MODEL_DB` dict in `scripts/memory_calc.py`.

## Project Structure

```
lora-trainer/
├── cli/                          # NEW: Standalone CLI (pip install)
│   ├── main.py                   # Entry point, CLI group
│   └── commands/
│       ├── analyze.py            # lora-trainer analyze
│       ├── recommend.py          # lora-trainer recommend
│       ├── memory.py             # lora-trainer memory
│       ├── cook.py               # lora-trainer cook
│       └── evaluate.py           # lora-trainer evaluate
├── scripts/                      # Python computation layer
│   ├── analyzer.py               # Data statistics & quality checks
│   ├── memory_calc.py            # VRAM estimation (40+ model DB)
│   ├── lora_advisor.py           # Parameter recommendation engine
│   ├── script_builder.py         # Training/inference script generator
│   └── evaluator.py              # Model evaluation & comparison
├── skills/lora-trainer/          # Claude Code skill (AI agent)
│   ├── SKILL.md                  # All recommendation rules (~522 lines)
│   └── references/               # Pure data, loaded on demand
│       ├── model-catalog.md      # Model specs (40+)
│       ├── recipes.md            # Preset configurations
│       ├── vram-reference.md     # Quick VRAM lookup table
│       └── faq.md                # Frequently asked questions
├── agents/                       # Sub-agent definitions
├── commands/                     # Slash command definitions
├── .claude-plugin/               # Claude Code plugin manifest
├── examples/                     # Sample JSONL training data
├── templates/                    # Code template stubs
├── pyproject.toml                # Python package config
├── package.json                  # npm package config
└── requirements.txt              # Python dependencies
```

## Design

- **Agent is the brain** — all recommendation rules and judgment live in `SKILL.md`. The agent decides; scripts compute.
- **CLI is the interface** — `click` + `rich` provide a beautiful, emoji-rich terminal experience. Every command has a `--json` flag for scripting.
- **Scripts are the hands** — Python handles exact numeric work: VRAM math, data statistics, and template rendering. No `if/else` decision logic in Python.
- **Progressive disclosure** — SKILL.md is ~520 lines. Deep reference data stays in `references/` and loads only when needed.

## Requirements

```bash
pip install lora-trainer        # CLI + core (click, rich, numpy, pyyaml)
pip install lora-trainer[train] # + torch, transformers, peft, datasets, etc.
```

Training requires an NVIDIA GPU (8 GB+ recommended for 7B models with QLoRA), CUDA Toolkit, and PyTorch with CUDA.

## Contributing

Issues and PRs welcome. Before submitting, run the smoke test:

```bash
# Python modules
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

# CLI smoke test
python -m cli.main --help
python -m cli.main analyze examples/sample_data.jsonl
python -m cli.main recommend -n 2340 -m 7b -t chat
python -m cli.main memory qwen2-7b
```

## License

MIT
