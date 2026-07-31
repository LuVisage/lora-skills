#!/usr/bin/env bash
# lora-trainer session-start hook
# Injects skill bootstrap context at the start of every Claude Code session.
set -euo pipefail

HOOK_NAME="${1:-}"

case "$HOOK_NAME" in
  session-start)
    # Read the skill's auto-activation description for the agent's context
    HOOK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
    # Support both source layout (skills/lora-trainer/) and installed layout (flat root)
    if [ -f "$HOOK_ROOT/skills/lora-trainer/SKILL.md" ]; then
      SKILL_DIR="$HOOK_ROOT/skills/lora-trainer"
    elif [ -f "$HOOK_ROOT/SKILL.md" ]; then
      SKILL_DIR="$HOOK_ROOT"
    else
      SKILL_DIR="$HOOK_ROOT/skills/lora-trainer"  # default: source layout
    fi
    if [ -f "$SKILL_DIR/SKILL.md" ]; then
      # Extract YAML frontmatter description as context hint
      DESCRIPTION=$(sed -n '/^---$/,/^---$/p' "$SKILL_DIR/SKILL.md" | grep "^description:" | sed 's/^description:\s*//' | sed 's/>-//' | tr '\n' ' ' | sed 's/  */ /g')
      echo '{"hookSpecificOutput":{"additionalContext":"<EXTREMELY_IMPORTANT> This project has a lora-trainer skill available. When the user asks about LoRA fine-tuning, model training, or 微调, use the /lora:analyze or /lora:cook commands. Skill description: '"${DESCRIPTION:-LoRA fine-tuning assistant}"'</EXTREMELY_IMPORTANT>"}}'
    else
      echo '{"hookSpecificOutput":{}}'
    fi
    ;;
  *)
    echo '{"hookSpecificOutput":{}}'
    ;;
esac
