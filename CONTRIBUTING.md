# Contributing to lora-trainer

Thanks for your interest in contributing! This document covers how to get started, what makes a good contribution, and the review process.

## Ways to Contribute

- **Add a model** — add an entry to `scripts/memory_calc.py`'s `MODEL_DB` dict and `skills/lora-trainer/references/model-catalog.md`
- **Add a recipe** — contribute a preset configuration to `skills/lora-trainer/references/recipes.md`
- **Fix a bug** — open an issue first, then a PR with a fix
- **Improve docs** — clarify explanations, fix typos, add examples
- **New feature** — open an issue to discuss before writing code

## Project Philosophy

1. **Agent is the brain.** Rules and decision logic go in `skills/lora-trainer/SKILL.md`, not in Python.
2. **Scripts are the hands.** `scripts/*.py` does exact computation only. No `if/else` judgment calls.
3. **Chinese-first UX.** All user-facing CLI output is in Chinese with emoji. Programmatic output (`--json`) is in English.
4. **Progressive disclosure.** Core content in SKILL.md; deep reference data in `references/`, loaded on demand.

## Development Setup

```bash
git clone https://github.com/LuVisage/lora-skills.git
cd lora-skills
pip install -e ".[train,dev]"
```

## Before Submitting a PR

```bash
# 1. Run the smoke test
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

# 2. Run all CLI commands
python -m cli.main --help
python -m cli.main analyze examples/sample_data.jsonl
python -m cli.main recommend -n 2340 -m 7b -t chat
python -m cli.main memory qwen2-7b
python -m cli.main evaluate examples/sample_data.jsonl -o /tmp/eval_test.py

# 3. Verify version consistency
grep version pyproject.toml
grep version package.json
grep version .claude-plugin/plugin.json
grep __version__ cli/__init__.py
```

## Versioning

Follow [SemVer](https://semver.org/). Version is stored in 4 files — all must be updated together:
- `pyproject.toml` → `version = "X.Y.Z"`
- `package.json` → `"version": "X.Y.Z"`
- `.claude-plugin/plugin.json` → `"version": "X.Y.Z"`
- `cli/__init__.py` → `__version__ = "X.Y.Z"`

## Review Process

1. Open an issue describing the change (bug report or feature request)
2. Fork and create a feature branch
3. Write your change. Follow existing code style.
4. Run the smoke tests above
5. Open a PR against `main`
6. A maintainer will review within 48 hours

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
