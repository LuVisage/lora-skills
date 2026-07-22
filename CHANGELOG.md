# Changelog

All notable changes to lora-trainer.

## [2.5.0] — 2026-07-22

### Added

- **Executable Python scripts** in `skills/lora-trainer/scripts/` — addressing "only docs, no code" feedback
  - `train_qlora.py` — Complete QLoRA training script (change 2 lines, run)
  - `inference.py` — LoRA model inference (interactive / single / batch)
  - `check_env.py` — One-click environment check (CUDA, PyTorch, deps, GPU, mirror)
- **`TRIGGERS.md`** — Exhaustive invocation catalog (slash commands + natural language, CN + EN)
- **`/lora:setup` command** — Environment check + dependency installation
- **Code snippets** for DoRA, LoRA+, NEFTune in SKILL.md — complete `LoraConfig(...)` examples, not just descriptions
- **Quick Start section** in SKILL.md — prominent copy-paste examples
- **Domestic mirror guide** (🇨🇳 国内用户模型下载加速) — hf-mirror.com, ModelScope, and manual download methods for Chinese users
- **Principle #6: 能动手就不动嘴** — agent proactively runs analysis/VRAM/script generation instead of telling users to run commands themselves
- **`/lora:debug` command** — post-training failure diagnosis from logs, error messages, or symptoms. Matches error signatures → solutions
- **Step 0: Startup check** — version verification and environment probing at each interaction start
- **Step 5.5: Runtime support** — explicit post-generation follow-up flow; agent commits to helping with OOM, loss issues, bad results after training
- **Enhanced troubleshooting** — expanded from 6 rows to 17 rows across three tables (environment checks, error messages, training quality). Each entry has specific diagnostic commands and prioritized fixes
- **FAQ additions** — checkpoint recovery, training monitoring via tensorboard, domestic model download guide

### Changed

- **Five principles → Six principles** — added "能动手就不动嘴" principle
- **Command files** (analyze/cook/check-data) now include richer invocation examples with natural language variants
- **Troubleshooting table** restructured into three categories: pre-training checks, error messages, and training quality issues

## [2.4.0] — 2026-07-21

### Added

- **Standalone CLI tool** (`cli/` package) — `pip install lora-trainer` support
- 5 CLI commands: `analyze`, `recommend`, `memory`, `cook`, `evaluate`
- `pyproject.toml` — modern Python packaging with `[train]` extras
- Rich terminal formatting with Chinese-first UX and emoji
- `--json` flag on all commands for programmatic use
- `--interactive` mode on `cook` command
- CLAUDE.md, CONTRIBUTING.md, CHANGELOG.md, CODE_OF_CONDUCT.md, SECURITY.md
- `.github/` directory with CODEOWNERS and PR template
- `hooks/` directory with session-start bootstrap

## [2.3.0] — 2026-07-20

### Added

- CPT (Continual Pre-Training) full support across all modules
- Task-aware gradient accumulation targets (CPT=32, small_data=8, default=16)
- Task-aware learning rate schedulers (constant_with_warmup for CPT, linear for roleplay)
- NEFTune noise auto-disabled for CPT
- Published to npm as `lora-trainer@2.3.0`

### Fixed

- Messages/ChatML format data quality checks (format-aware text extraction)
- Auto-split logic: `"test" not in dataset` → `"test" not in dataset and "eval" not in dataset`
- `total_steps` f-string NameError in generated training script
- `{reply}` f-string escaping in inference script template

## [2.2.1] — 2026-07-19

### Added

- DoRA, LoRA+, NEFTune recommendations in SKILL.md
- Learning rate scheduler recommendations (cosine/linear/constant_with_warmup)
- Packing guidance for CPT and short-QA scenarios

### Fixed

- `_lookup_model` prefers 7B models on ambiguous fuzzy matches
- `recommend()` uses `quantized` parameter in output text
- Messages/auto formatter: removed trailing `<|assistant|>\n`

## [2.1.0] — 2026-07-18

### Added

- 40+ model specs in MODEL_DB and model-catalog.md
- MoE model support (Qwen2-57B, Mixtral, DeepSeek-v2/v3)
- VRAM quick reference table
- CJK token estimation (2.0 tokens/char)
- Data formatter for messages, conversations, and instruction-output
- FAQ with 19 Q&A entries

## [2.0.0] — 2026-07-17

### Added

- Initial public release
- Claude Code plugin with 3 slash commands (`/lora:analyze`, `/lora:cook`, `/lora:check-data`)
- 3 sub-agents (data-analyzer, memory-estimator, script-generator)
- 57 parameter recommendation rules in SKILL.md
- 4 reference files (model-catalog, recipes, vram-reference, faq)
- npm package publishing
