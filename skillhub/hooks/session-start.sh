#!/usr/bin/env bash
# lora-trainer session-start hook
# Injects skill bootstrap context at the start of every Claude Code session.
set -euo pipefail

HOOK_NAME="${1:-}"

case "$HOOK_NAME" in
  session-start)
    # Read the skill's auto-activation description for the agent's context
    SKILL_DIR="$(cd "$(dirname "$0")/.." && pwd)/skills/lora-trainer"
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
