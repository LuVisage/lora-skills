#!/usr/bin/env bash
# Polyglot wrapper — works on Windows (Git Bash) and Unix
# Usage: run-hook.sh <hook-name>
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "$SCRIPT_DIR/session-start.sh" "$@"
