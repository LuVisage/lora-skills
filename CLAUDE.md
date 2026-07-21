# CLAUDE.md — lora-trainer Project Context

## What This Project Is

**lora-trainer** is both a standalone CLI tool (`pip install lora-trainer`) and a Claude Code plugin for LoRA/QLoRA fine-tuning. It helps users go from raw JSONL training data to a runnable training script in one command.

## Tech Stack

- **CLI**: Python 3.10+, Click, Rich (terminal formatting)
- **ML Engine**: PyTorch, Transformers, PEFT, BitsAndBytes, Datasets, Accelerate
- **Plugin**: npm package (for Claude Code / Cursor / Codex distribution)
- **Packaging**: `pyproject.toml` (pip) + `package.json` (npm)

## Architecture

```
cli/          → CLI layer (pip install entry point), wraps scripts/
scripts/      → Pure computation — no decision logic, just math + template rendering
skills/       → AI agent behavioral rules (SKILL.md), loaded by Claude Code
agents/       → Sub-agent definitions (Haiku model, fork context)
commands/     → Slash command definitions (/lora:analyze, /lora:cook, /lora:check-data)
references/   → Pure data loaded on demand (model specs, recipes, VRAM table, FAQ)
hooks/        → Session lifecycle hooks (session-start bootstrap)
```

### Key Design Principle

**Agent is the brain, scripts are the hands.** All decision rules live in `skills/lora-trainer/SKILL.md`. Python scripts (`scripts/*.py`) do exact numeric computation only — no `if/else` decision logic.

## Commands

```bash
# Dev: run CLI locally
python -m cli.main --help
python -m cli.main analyze examples/sample_data.jsonl
python -m cli.main recommend -n 2340 -m 7b -t chat
python -m cli.main memory qwen2-7b
python -m cli.main cook -d examples/sample_data.jsonl -m 7b -t chat -o ./test-output
python -m cli.main evaluate examples/sample_data.jsonl

# Install locally for testing
pip install -e .
lora-trainer --help

# Full stack
pip install -e ".[train]"
```

## Testing

```bash
# Smoke test (all modules + CLI)
python -c "from scripts.analyzer import quick_analyze; from scripts.memory_calc import quick_calc; from scripts.lora_advisor import quick_recommend; from cli.main import main; r = quick_analyze('examples/sample_data.jsonl'); m = quick_calc('qwen2-7b'); c = quick_recommend(r['length']['total_samples'], '7b', 'chat'); print('✅ OK')"

# npm test
npm test
```

## Versioning

- Version in 4 places must stay in sync: `pyproject.toml`, `package.json`, `.claude-plugin/plugin.json`, `cli/__init__.py`
- Follow semver: MAJOR.MINOR.PATCH

## Conventions

- Python: no type annotations that require `from __future__ import annotations` (compat with 3.10+)
- Chinese-first UX: all user-facing CLI output in Chinese with emoji
- Every command supports `--json` flag for programmatic use
- Don't modify `scripts/*.py` unless fixing a computation bug — they're the stable core
- When adding a new CLI command: create `cli/commands/<name>.py`, then register in `cli/main.py`
- When adding a new slash command: create `commands/<name>.md`, reference in `skills/lora-trainer/SKILL.md`

## Protected Files

- `skills/lora-trainer/SKILL.md` — the behavioral core. Changes require careful review.
- `scripts/*.py` — the computation core. Don't refactor without running full test suite.
- `pyproject.toml` — version and dependencies. Don't add heavy deps without discussion.
