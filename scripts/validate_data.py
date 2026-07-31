#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JSONL Data Validator for LoRA Fine-tuning
==========================================
Quickly check if your training data is ready for fine-tuning.

Usage:
  python validate_data.py ./data/train.jsonl
  python validate_data.py ./data/train.jsonl --show-issues
  python validate_data.py ./data/train.jsonl --sample 3

No ML dependencies -- pure Python, runs anywhere.
"""

import sys
import json
import re
import argparse
from collections import Counter


def validate(filepath, show_issues=False, show_samples=0):
    """Validate a JSONL file for LoRA fine-tuning.

    Returns: (passed, stats_dict, issues_list)
    """
    # Read file
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            raw_lines = [line for line in f if line.strip()]
    except FileNotFoundError:
        print("[FAIL] File not found: {}".format(filepath))
        return False, {}, ["File not found: {}".format(filepath)]
    except Exception as e:
        print("[FAIL] Cannot read file: {}".format(e))
        return False, {}, ["Cannot read file: {}".format(e)]

    total_lines = len(raw_lines)
    if total_lines == 0:
        print("[FAIL] File is empty.")
        return False, {"total_lines": 0}, ["File is empty."]

    # Parse JSON
    samples = []
    parse_errors = []
    for i, line in enumerate(raw_lines, 1):
        try:
            samples.append(json.loads(line))
        except json.JSONDecodeError as e:
            parse_errors.append((i, str(e)))

    valid_lines = len(samples)
    invalid_lines = len(parse_errors)

    if valid_lines == 0:
        print("[FAIL] No valid JSON lines found ({} total lines).".format(total_lines))
        for line_num, err in parse_errors[:5]:
            print("  Line {}: {}".format(line_num, err))
        return False, {"total_lines": total_lines, "valid_lines": 0}, [
            "No valid JSON lines ({} parse errors)".format(invalid_lines)]

    # Detect format
    first = samples[0]
    detected_format = "unknown"
    if "messages" in first:
        detected_format = "messages"
        key_fields = ["messages"]
    elif "instruction" in first and "output" in first:
        detected_format = "instruction-output"
        key_fields = ["instruction", "output"]
        # Check for optional input field
        has_input = any("input" in s for s in samples)
    elif "conversations" in first:
        detected_format = "conversations"
        key_fields = ["conversations"]
    elif "text" in first:
        detected_format = "text (CPT/pretraining)"
        key_fields = ["text"]
    else:
        key_fields = list(first.keys())

    # Check for empty/invalid content
    empty_count = 0
    empty_details = []
    for i, sample in enumerate(samples):
        for field in key_fields:
            val = sample.get(field)
            if val is None:
                empty_count += 1
                if len(empty_details) < 10:
                    empty_details.append("Line {}: field '{}' is missing/None".format(i + 1, field))
            elif isinstance(val, str) and not val.strip():
                empty_count += 1
                if len(empty_details) < 10:
                    empty_details.append("Line {}: field '{}' is empty string".format(i + 1, field))
            elif isinstance(val, list) and len(val) == 0:
                empty_count += 1
                if len(empty_details) < 10:
                    empty_details.append("Line {}: field '{}' is empty list".format(i + 1, field))

    # Check for near-duplicates
    # Build text fingerprints for each sample
    fingerprints = []
    for sample in samples:
        text = ""
        for field in key_fields:
            val = sample.get(field, "")
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        text += item.get("content", item.get("value", ""))
                    else:
                        text += str(item)
            elif isinstance(val, str):
                text += val
        # Normalize: lowercase, strip whitespace, first 200 chars
        fp = "".join(text.lower().split())[:200]
        fingerprints.append(fp)

    fp_counts = Counter(fingerprints)
    dup_groups = {fp: count for fp, count in fp_counts.items() if count > 1}
    total_duplicates = sum(count - 1 for count in dup_groups.values())
    dup_count = len(dup_groups)

    # Token length estimation
    lengths = []
    for sample in samples:
        text = ""
        for field in key_fields:
            val = sample.get(field, "")
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        text += item.get("content", item.get("value", ""))
                    else:
                        text += str(item)
            elif isinstance(val, str):
                text += val

        # CJK characters ~2 tokens/char, others ~0.28 tokens/char
        cjk_chars = len(re.findall(r'[一-鿿㐀-䶿\U00020000-\U0002a6df]', text))
        other_chars = len(text) - cjk_chars
        est_tokens = int(cjk_chars * 2.0 + other_chars * 0.28)
        lengths.append(est_tokens)

    lengths_sorted = sorted(lengths)
    avg_tokens = sum(lengths) / len(lengths) if lengths else 0
    p50_tokens = lengths_sorted[len(lengths_sorted) // 2] if lengths_sorted else 0
    p95_tokens = lengths_sorted[min(int(len(lengths_sorted) * 0.95), len(lengths_sorted) - 1)] if lengths_sorted else 0
    p99_tokens = lengths_sorted[min(int(len(lengths_sorted) * 0.99), len(lengths_sorted) - 1)] if lengths_sorted else 0

    # Quality grade
    score = 100
    issues_list = []

    if invalid_lines > 0:
        score -= min(30, invalid_lines * 2)
        issues_list.append("{} JSON parse error(s)".format(invalid_lines))

    if empty_count > 0:
        empty_pct = empty_count / valid_lines * 100
        score -= min(25, empty_pct * 5)
        issues_list.append("{} empty/missing field(s) ({:.1f}%)".format(empty_count, empty_pct))

    if dup_count > 0:
        dup_pct = total_duplicates / valid_lines * 100
        score -= min(20, dup_pct * 2)
        issues_list.append("{} near-duplicate group(s) ({} total duplicate samples)".format(
            dup_count, total_duplicates))

    if valid_lines < 100:
        score -= 15
        issues_list.append("Very small dataset ({} samples, recommend  500)".format(valid_lines))
    elif valid_lines < 500:
        score -= 5
        issues_list.append("Small dataset ({} samples, 1000+ recommended)".format(valid_lines))

    if p95_tokens > 4096:
        score -= 10
        issues_list.append("P95 token length ({}) exceeds typical max_seq_length (4096)".format(p95_tokens))

    if avg_tokens < 50:
        score -= 10
        issues_list.append("Average token length too short ({:.0f}), may lack sufficient training signal".format(
            avg_tokens))

    score = max(0, min(100, score))

    # Grade label
    if score >= 90:
        grade = "Excellent"
    elif score >= 75:
        grade = "Good"
    elif score >= 60:
        grade = "Fair"
    elif score >= 40:
        grade = "Poor"
    else:
        grade = "Unusable"

    # Build stats
    stats = {
        "total_lines": total_lines,
        "valid_lines": valid_lines,
        "invalid_lines": invalid_lines,
        "detected_format": detected_format,
        "key_fields": key_fields,
        "empty_count": empty_count,
        "dup_count": dup_count,
        "total_duplicates": total_duplicates,
        "avg_tokens": avg_tokens,
        "p50_tokens": p50_tokens,
        "p95_tokens": p95_tokens,
        "p99_tokens": p99_tokens,
        "score": score,
        "grade": grade,
    }

    # Print report
    print()
    print("=" * 60)
    print("  JSONL Data Validation Report")
    print("=" * 60)
    print("  File:     {}".format(filepath))
    print("  Format:   {}".format(detected_format))
    print("  Fields:   {}".format(", ".join(key_fields)))
    print("-" * 60)
    print("  {:<35s} {:>10s}".format("Metric", "Value"))
    print("-" * 60)
    print("  {:<35s} {:>10d}".format("Total Lines", total_lines))
    print("  {:<35s} {:>10d}".format("Valid JSON Lines", valid_lines))
    if invalid_lines > 0:
        print("  {:<35s} {:>10d} [WARN]".format("JSON Parse Errors", invalid_lines))
    else:
        print("  {:<35s} {:>10d} [OK]".format("JSON Parse Errors", invalid_lines))
    print("  {:<35s} {:>10d}".format("Empty/Missing Fields", empty_count))
    print("  {:<35s} {:>10d}".format("Duplicate Groups", dup_count))
    print("  {:<35s} {:>10d}".format("Total Duplicate Samples", total_duplicates))
    print("-" * 60)
    print("  Token Length Estimation:")
    print("  {:<35s} {:>10.0f}".format("Average", avg_tokens))
    print("  {:<35s} {:>10.0f}".format("Median (P50)", p50_tokens))
    print("  {:<35s} {:>10.0f}".format("P95", p95_tokens))
    print("  {:<35s} {:>10.0f}".format("P99", p99_tokens))
    print("-" * 60)
    print("  {:<35s} {:>9d}/100 ({})".format("Quality Score", int(score), grade))
    print("=" * 60)

    # Issues detail
    if issues_list:
        print()
        print("Issues Found:")
        for issue in issues_list:
            print("  - {}".format(issue))

        if show_issues and empty_details:
            print()
            print("  Empty field details (first 10):")
            for detail in empty_details:
                print("    {}".format(detail))

    # Recommendations
    print()
    print("Recommendations:")
    if score >= 90:
        print("  [OK] Data quality looks great! Ready for fine-tuning.")
        print("  Suggested: /lora:cook {} <model> chat".format(filepath))
    elif score >= 75:
        print("  [OK] Good enough for fine-tuning. Minor issues won't hurt much.")
        if invalid_lines > 0:
            print("  Tip: Fix {} parse errors for cleaner training.".format(invalid_lines))
        if dup_count > 0:
            print("  Tip: {} duplicates found. Some duplication is fine for SFT.".format(total_duplicates))
    elif score >= 60:
        print("  [WARN] Trainable but quality could be better.")
        print("  Fix: Check and fix the issues listed above, then re-run validation.")
    elif score >= 40:
        print("  [WARN] Significant quality issues. Training may produce poor results.")
        print("  Fix: Review your data preparation pipeline.")
        print("  Re-run with --show-issues for detailed diagnostics.")
    else:
        print("  [FAIL] Data quality is too poor for meaningful training.")
        print("  Fix: Re-check your data source. Ensure proper JSONL format:")
        print('    {"instruction": "...", "output": "..."}')
        print('    {"messages": [{"role": "user", "content": "..."}, ...]}')
        print('    {"text": "..."}  (for CPT/pretraining)')

    # Show samples
    if show_samples > 0:
        print()
        print("Sample Records (first {}):".format(min(show_samples, len(samples))))
        print("-" * 40)
        for i, sample in enumerate(samples[:show_samples]):
            print("--- Sample {} ---".format(i + 1))
            # Print a preview of each field
            for field in key_fields:
                val = sample.get(field, "")
                if isinstance(val, list):
                    print("  {}: [list with {} items]".format(field, len(val)))
                    for j, item in enumerate(val[:2]):
                        if isinstance(item, dict):
                            for k, v in item.items():
                                preview = str(v)[:80]
                                print("    [{}].{}: {}...".format(j, k, preview) if len(str(v)) > 80
                                      else "    [{}].{}: {}".format(j, k, preview))
                else:
                    preview = str(val)[:120]
                    print("  {}: {}...".format(field, preview) if len(str(val)) > 120
                          else "  {}: {}".format(field, preview))
            print()

    return score >= 60, stats, issues_list


def main():
    parser = argparse.ArgumentParser(
        description="JSONL Data Validator for LoRA Fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate_data.py ./data/train.jsonl
  python validate_data.py ./data/train.jsonl --show-issues
  python validate_data.py ./data/train.jsonl --sample 3
  python validate_data.py ./data/train.jsonl --json  (machine-readable output)
        """,
    )
    parser.add_argument("filepath", nargs="?", help="Path to JSONL data file")
    parser.add_argument("--show-issues", action="store_true", help="Show detailed issue diagnostics")
    parser.add_argument("--sample", type=int, default=0, help="Show first N sample records")
    parser.add_argument("--json", action="store_true", help="Output as JSON (machine-readable)")
    args = parser.parse_args()

    if not args.filepath:
        parser.print_help()
        sys.exit(1)

    passed, stats, issues = validate(args.filepath, args.show_issues, args.sample)

    if args.json:
        print(json.dumps({
            "passed": passed,
            "stats": {k: (round(v, 1) if isinstance(v, float) else v) for k, v in stats.items()},
            "issues": issues,
        }, indent=2, ensure_ascii=False))

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
