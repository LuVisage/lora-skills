#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build script for lora-trainer three-platform publication.

Usage:
  python build.py all       Build all three dist targets
  python build.py github    Sync root -> dist/github
  python build.py skillhub  Sync skills/ -> skillhub/ -> dist/skillhub/
  python build.py hf        Validate dist/huggingface/ is ready
  python build.py check     Dry-run: report what would change without writing
"""

import os
import sys
import shutil
import filecmp
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Files/dirs included in npm package (from package.json "files")
NPM_FILES = [
    ".claude-plugin",
    ".github",
    "commands",
    "agents",
    "skills",
    "hooks",
    "cli",
    "scripts",
    "templates",
    "examples",
    "pyproject.toml",
    "CLAUDE.md",
    "README.md",
    "README.zh-CN.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "SECURITY.md",
    "LICENSE",
    "requirements.txt",
]

# Always exclude from dist copies
ALWAYS_EXCLUDE = {
    "__pycache__", ".git", "node_modules", "dist", "build",
    ".claude", ".playwright-mcp", ".venv", "venv", "env",
    ".DS_Store", "Thumbs.db", "*.pyc", "*.pyo", "*.egg-info",
}

# Files/dirs that don't belong in SkillHub submissions
SKILLHUB_EXCLUDE_SOURCE = {
    "CHANGELOG.md", "README.md", "README.zh-CN.md",
    "examples", "docs", ".github", "cli", "templates",
    "pyproject.toml", "LICENSE", "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md", "SECURITY.md",
}

SKILLHUB_EXCLUDE_DIST = SKILLHUB_EXCLUDE_SOURCE | {"build.py", "skillhub"}


def should_exclude(name, exclude_set):
    """Check if a file/dir name should be excluded."""
    if name in exclude_set:
        return True
    if name.endswith(".pyc") or name.endswith(".pyo"):
        return True
    if name == "__pycache__":
        return True
    return False


def copy_tree(src, dst, exclude_set=None, dry_run=False):
    """Copy directory tree from src to dst, excluding specified names.
    Returns list of (action, path) tuples."""
    if exclude_set is None:
        exclude_set = set()
    actions = []
    if not src.exists():
        return actions
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if should_exclude(item.name, exclude_set):
            continue
        rel = item.relative_to(ROOT)
        if item.is_dir():
            sub_actions = copy_tree(item, dst / item.name, exclude_set, dry_run)
            actions.extend(sub_actions)
        else:
            dst_file = dst / item.name
            if not dst_file.exists() or not filecmp.cmp(str(item), str(dst_file), shallow=False):
                actions.append(("COPY", str(rel)))
                if not dry_run:
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(item), str(dst_file))
    return actions


def remove_extra(dst, exclude_set, dry_run=False):
    """Remove files/dirs in dst that should be excluded.
    Returns list of (action, path) tuples."""
    actions = []
    if not dst.exists():
        return actions
    for item in dst.iterdir():
        if should_exclude(item.name, exclude_set):
            rel = item.relative_to(ROOT)
            actions.append(("DELETE", str(rel)))
            if not dry_run:
                if item.is_dir():
                    shutil.rmtree(str(item))
                else:
                    item.unlink()
    return actions


def build_github(dry_run=False):
    """Sync root -> dist/github (full repo snapshot)."""
    print("\n[github] Building dist/github/ ...")
    src = ROOT
    dst = ROOT / "dist" / "github"
    all_actions = []

    for item_name in NPM_FILES:
        src_item = src / item_name
        if not src_item.exists():
            continue
        if src_item.is_dir():
            actions = copy_tree(src_item, dst / item_name, ALWAYS_EXCLUDE, dry_run)
        else:
            actions = []
            dst_file = dst / item_name
            if not dst_file.exists() or not filecmp.cmp(str(src_item), str(dst_file), shallow=False):
                rel = src_item.relative_to(ROOT)
                actions.append(("COPY", str(rel)))
                if not dry_run:
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(src_item), str(dst_file))
        all_actions.extend(actions)

    # Copy package.json (npm) and root configs
    for extra in ["package.json", ".gitignore", ".npmignore", "docs"]:
        extra_item = src / extra
        if extra_item.exists():
            if extra_item.is_dir():
                all_actions.extend(copy_tree(extra_item, dst / extra, ALWAYS_EXCLUDE, dry_run))
            else:
                dst_file = dst / extra
                if not dst_file.exists() or not filecmp.cmp(str(extra_item), str(dst_file), shallow=False):
                    all_actions.append(("COPY", str(extra_item.relative_to(ROOT))))
                    if not dry_run:
                        shutil.copy2(str(extra_item), str(dst_file))

    if all_actions:
        for action, path in all_actions:
            print(f"  [{action}] {path}")
        print(f"  -> {len(all_actions)} change(s)")
    else:
        print("  -> Up to date")
    return all_actions


# ---------------------------------------------------------------------------
# SkillHub path transformation
# ---------------------------------------------------------------------------
# In the GitHub source (project context), paths use:
#   ${CLAUDE_PLUGIN_ROOT}/skills/lora-trainer/scripts/...
# In the SkillHub install (flat context), paths must use:
#   ${CLAUDE_PLUGIN_ROOT}/scripts/...
#
# TRANSFORM_MAP rewrites source-relative paths to installed-relative paths.

TRANSFORM_MAP = {
    "${CLAUDE_PLUGIN_ROOT}/skills/lora-trainer/scripts/": "${CLAUDE_PLUGIN_ROOT}/scripts/",
    "${CLAUDE_PLUGIN_ROOT}/skills/lora-trainer/references/": "${CLAUDE_PLUGIN_ROOT}/references/",
    "skills/lora-trainer/scripts/": "scripts/",
    "skills/lora-trainer/references/": "references/",
}

# Project-level computation scripts that the skill agents depend on.
# These live at repo-root scripts/ but must be included in the skill package
# so that agents can call them in the installed (SkillHub) context.
AGENT_SCRIPTS = [
    "analyzer.py",
    "memory_calc.py",
    "lora_advisor.py",
    "script_builder.py",
    "evaluator.py",
]


def _transform_text(content):
    """Apply SkillHub path transformations to text content."""
    for old, new in TRANSFORM_MAP.items():
        content = content.replace(old, new)
    return content


def copy_text_file_transformed(src, dst, dry_run=False):
    """Copy a text file with SkillHub path transformations.
    Returns list of (action, path) tuples."""
    if not src.exists():
        return []
    with open(str(src), "r", encoding="utf-8") as f:
        original = f.read()
    transformed = _transform_text(original)
    if dst.exists():
        with open(str(dst), "r", encoding="utf-8") as f:
            existing = f.read()
        if existing == transformed:
            return []
    rel = dst.relative_to(ROOT)
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(str(dst), "w", encoding="utf-8", newline="\n") as f:
            f.write(transformed)
    return [("COPY+TRANSFORM", str(rel))]


def copy_single_file(src, dst, dry_run=False):
    """Copy a single file from src to dst if content differs.
    Returns list of (action, path) tuples."""
    if not src.exists():
        return []
    if dst.exists() and filecmp.cmp(str(src), str(dst), shallow=False):
        return []
    rel = dst.relative_to(ROOT)
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
    return [("COPY", str(rel))]


def build_skillhub(dry_run=False):
    """Flatten skills/lora-trainer into skillhub/ root, then skillhub -> dist/skillhub.

    SkillHub requires a flat structure with SKILL.md at root.  We flatten the
    nested skills/lora-trainer/ layout so that scripts/, references/, and
    TRIGGERS.md sit alongside commands/, agents/, and hooks/ at the root.

    Paths in text files are transformed so that ${CLAUDE_PLUGIN_ROOT} references
    work correctly both in the GitHub source (nested) and installed (flat) contexts.
    """
    all_actions = []
    src_skill = ROOT / "skills" / "lora-trainer"
    src_scripts = ROOT / "scripts"
    dst_root = ROOT / "skillhub"

    # Step 1: Copy SKILL.md + TRIGGERS.md to root with path transformations
    print("\n[skillhub] Step 1: Flattening SKILL.md + TRIGGERS.md to root (with path transform) ...")
    for stem in ["SKILL.md", "TRIGGERS.md"]:
        actions = copy_text_file_transformed(src_skill / stem, dst_root / stem, dry_run)
        all_actions.extend(actions)
        if actions:
            for action, path in actions:
                print(f"  [{action}] {path}")
        else:
            print(f"  -> {stem} up to date")

    # Step 2: Copy references/ and skill-level scripts/ to skillhub root
    print("\n[skillhub] Step 2: Flattening references/ + scripts/ to root ...")
    for sub in ["references"]:
        src_sub = src_skill / sub
        if not src_sub.exists():
            continue
        actions = copy_tree(src_sub, dst_root / sub, {"__pycache__"}, dry_run)
        all_actions.extend(actions)
        if actions:
            for action, path in actions:
                print(f"  [{action}] {path}")
            print(f"  -> {len(actions)} change(s) in {sub}/")
        else:
            print(f"  -> {sub}/ up to date")

    # Copy skill-level scripts
    actions = copy_tree(src_skill / "scripts", dst_root / "scripts", {"__pycache__"}, dry_run)
    all_actions.extend(actions)
    if actions:
        for action, path in actions:
            print(f"  [{action}] {path}")
        print(f"  -> {len(actions)} change(s) in scripts/")
    else:
        print("  -> scripts/ up to date")

    # Step 2b: Copy project-level computation scripts needed by agents
    print("\n[skillhub] Step 2b: Adding agent computation scripts ...")
    for script_name in AGENT_SCRIPTS:
        src_file = src_scripts / script_name
        dst_file = dst_root / "scripts" / script_name
        if not src_file.exists():
            continue
        actions = copy_single_file(src_file, dst_file, dry_run)
        all_actions.extend(actions)
        if actions:
            for action, path in actions:
                print(f"  [{action}] {path}")
        else:
            print(f"  -> {script_name} up to date")

    # Step 3: Sync commands/, agents/ (with path transform), hooks/, .claude-plugin/
    print("\n[skillhub] Step 3: Syncing commands/agents/hooks/plugin (with path transform) ...")
    # Commands and agents need path transformation
    for comp in ["commands", "agents"]:
        src_comp = ROOT / comp
        if not src_comp.exists():
            continue
        dst_comp = dst_root / comp
        for item in src_comp.iterdir():
            if item.is_file() and item.suffix in (".md", ".json"):
                actions = copy_text_file_transformed(item, dst_comp / item.name, dry_run)
                all_actions.extend(actions)
                if actions:
                    for action, path in actions:
                        print(f"  [{action}] {path}")

    # Hooks and plugin config are copied as-is (no path transform needed)
    for comp in ["hooks", ".claude-plugin"]:
        src_comp = ROOT / comp
        if src_comp.exists():
            actions = copy_tree(src_comp, dst_root / comp, ALWAYS_EXCLUDE, dry_run)
            all_actions.extend(actions)
            if actions:
                for action, path in actions:
                    print(f"  [{action}] {path}")

    # Step 4: Remove stale skills/ subdirectory (legacy nesting) and excluded files
    print("\n[skillhub] Step 4: Cleaning skillhub/ ...")
    legacy_skills = dst_root / "skills"
    if legacy_skills.exists():
        rel = legacy_skills.relative_to(ROOT)
        all_actions.append(("DELETE", str(rel)))
        if not dry_run:
            shutil.rmtree(str(legacy_skills))
        print(f"  [DELETE] {rel}")
    actions = remove_extra(dst_root, SKILLHUB_EXCLUDE_SOURCE, dry_run)
    all_actions.extend(actions)
    if actions:
        for action, path in actions:
            print(f"  [{action}] {path}")
    else:
        print("  -> Clean")

    # Step 5: Copy skillhub -> dist/skillhub
    print("\n[skillhub] Step 5: Building dist/skillhub/ ...")
    dst_dist = ROOT / "dist" / "skillhub"

    # Remove stale files in dist/skillhub
    actions = remove_extra(dst_dist, SKILLHUB_EXCLUDE_DIST, dry_run)
    if actions:
        for action, path in actions:
            print(f"  [CLEAN] {path}")

    # Also remove legacy nested skills/ from dist
    legacy_dist_skills = dst_dist / "skills"
    if legacy_dist_skills.exists():
        rel = legacy_dist_skills.relative_to(ROOT)
        if not dry_run:
            shutil.rmtree(str(legacy_dist_skills))
        print(f"  [CLEAN] {rel}")

    actions = copy_tree(dst_root, dst_dist, {"__pycache__"}, dry_run)
    all_actions.extend(actions)
    if actions:
        for action, path in actions:
            print(f"  [{action}] {path}")
        print(f"  -> {len(actions)} change(s)")
    else:
        print("  -> Up to date")

    return all_actions


def build_hf(dry_run=False):
    """Validate dist/huggingface/ is ready."""
    print("\n[huggingface] Validating dist/huggingface/ ...")
    hf_dir = ROOT / "dist" / "huggingface"
    issues = []

    required = ["app.py", "requirements.txt", "README.md", "scripts"]
    for name in required:
        if not (hf_dir / name).exists():
            issues.append(f"Missing: {name}")

    # Check scripts have the core modules
    if (hf_dir / "scripts").exists():
        core_modules = ["analyzer.py", "memory_calc.py", "lora_advisor.py", "script_builder.py"]
        for mod in core_modules:
            if not (hf_dir / "scripts" / mod).exists():
                issues.append(f"Missing script: scripts/{mod}")

    if issues:
        print("  [ISSUES]")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  -> Ready for HuggingFace Space deployment")
    return issues


def build_all(dry_run=False):
    """Build all three dist targets."""
    action = "CHECK" if dry_run else "BUILD"
    print("=" * 60)
    print(f"  lora-trainer Build Script -- {action}")
    print("=" * 60)

    github_actions = build_github(dry_run)
    skillhub_actions = build_skillhub(dry_run)
    hf_issues = build_hf(dry_run)

    total = len(github_actions) + len(skillhub_actions) + len(hf_issues)
    print("\n" + "=" * 60)
    if dry_run:
        print(f"  CHECK COMPLETE: {total} change(s) needed")
    else:
        print(f"  BUILD COMPLETE: {total} file(s) updated")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Build lora-trainer for three-platform publication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  all       Build all three dist targets (github + skillhub + hf)
  github    Sync root -> dist/github
  skillhub  Sync skills -> skillhub -> dist/skillhub
  hf        Validate dist/huggingface/
  check     Dry-run: report what would change without writing

Examples:
  python build.py check      See what needs updating
  python build.py all        Full build
  python build.py skillhub   Build only SkillHub dist
        """,
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="check",
        choices=["all", "github", "skillhub", "hf", "check"],
        help="Build command (default: check = dry-run)",
    )
    args = parser.parse_args()

    dry_run = args.command == "check"

    if args.command in ("all", "check"):
        build_all(dry_run)
    elif args.command == "github":
        build_github(dry_run)
    elif args.command == "skillhub":
        build_skillhub(dry_run)
    elif args.command == "hf":
        build_hf(dry_run)


if __name__ == "__main__":
    main()
