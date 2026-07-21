# Changelog

All notable changes to lora-trainer.

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
