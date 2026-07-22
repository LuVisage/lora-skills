# CLAUDE.md — lora-trainer Skill

## What This Is

**lora-trainer** is a Claude Code skill for LoRA/QLoRA fine-tuning. Point it at JSONL training data and it handles: data audit, VRAM estimation, hyperparameter recommendation (with reasoning), and training script generation. Supports 40+ models across 13 families.

## Skill Architecture

```
skills/lora-trainer/          ← Core: AI agent brain
├── SKILL.md                  ← All recommendation rules & workflow
└── references/               ← Pure data, loaded on demand
commands/                     ← Slash commands (4)
agents/                       ← Sub-agents (3)
hooks/                        ← Lifecycle hooks
```

## Design Principles

1. **Agent is the brain.** All rules live in SKILL.md. Agent decides; scripts compute.
2. **Data first, parameters second.** Never recommend blind.
3. **VRAM first, batch size second.** OOM is the #1 beginner failure.
4. **Every parameter comes with a reason.** Users learn why.
5. **Act, don't just explain.** Run the analyzer directly, don't give commands.
6. **Progressive disclosure.** Deep reference data loads on demand.

## Version

Current: 2.5.0. Keep plugin.json, package.json, and SKILL.md version in sync.
