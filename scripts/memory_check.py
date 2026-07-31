#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPU Memory Estimator for LoRA/QLoRA Fine-tuning
================================================
Quickly estimate if your GPU can handle a specific model + LoRA config.

Usage:
  python memory_check.py qwen2-7b
  python memory_check.py llama3-8b --seq-length 4096 --batch-size 2
  python memory_check.py qwen2-7b --list-models

No ML dependencies needed -- pure Python math estimation.
"""

import sys
import argparse

# Model spec database (params, hidden_dim, num_layers)
MODEL_DB = {
    # Qwen2
    "qwen2-0.5b":   {"params": 0.5e9,  "hidden": 896,   "layers": 24},
    "qwen2-1.5b":   {"params": 1.5e9,  "hidden": 1536,  "layers": 28},
    "qwen2-7b":     {"params": 7e9,    "hidden": 3584,  "layers": 28},
    "qwen2-14b":    {"params": 14e9,   "hidden": 5120,  "layers": 40},
    "qwen2-72b":    {"params": 72e9,   "hidden": 8192,  "layers": 80},
    # Qwen2.5
    "qwen2.5-7b":   {"params": 7e9,    "hidden": 3584,  "layers": 28},
    "qwen2.5-14b":  {"params": 14e9,   "hidden": 5120,  "layers": 40},
    "qwen2.5-32b":  {"params": 32e9,   "hidden": 7168,  "layers": 64},
    "qwen2.5-72b":  {"params": 72e9,   "hidden": 8192,  "layers": 80},
    # LLaMA 3
    "llama3-8b":    {"params": 8e9,    "hidden": 4096,  "layers": 32},
    "llama3-70b":   {"params": 70e9,   "hidden": 8192,  "layers": 80},
    "llama3.1-8b":  {"params": 8e9,    "hidden": 4096,  "layers": 32},
    "llama3.1-70b": {"params": 70e9,   "hidden": 8192,  "layers": 80},
    # LLaMA 2
    "llama2-7b":    {"params": 7e9,    "hidden": 4096,  "layers": 32},
    "llama2-13b":   {"params": 13e9,   "hidden": 5120,  "layers": 40},
    "llama2-70b":   {"params": 70e9,   "hidden": 8192,  "layers": 80},
    # Mistral
    "mistral-7b":   {"params": 7e9,    "hidden": 4096,  "layers": 32},
    # DeepSeek
    "deepseek-7b":  {"params": 7e9,    "hidden": 3584,  "layers": 30},
    "deepseek-67b": {"params": 67e9,   "hidden": 8192,  "layers": 95},
    # ChatGLM
    "chatglm3-6b":  {"params": 6e9,    "hidden": 4096,  "layers": 28},
    # Yi
    "yi-6b":        {"params": 6e9,    "hidden": 4096,  "layers": 32},
    "yi-34b":       {"params": 34e9,   "hidden": 7168,  "layers": 60},
    # Baichuan
    "baichuan2-7b":  {"params": 7e9,   "hidden": 4096,  "layers": 32},
    "baichuan2-13b": {"params": 13e9,  "hidden": 5120,  "layers": 40},
    # Phi
    "phi-3-mini":   {"params": 3.8e9,  "hidden": 3072,  "layers": 32},
    "phi-3-small":  {"params": 7e9,    "hidden": 4096,  "layers": 32},
    # MoE models (activation params listed, total in comments)
    "mixtral-8x7b": {"params": 14e9,  "hidden": 4096,  "layers": 32, "is_moe": True, "total_params": 47e9},
    "qwen2-57b":    {"params": 14e9,  "hidden": 4096,  "layers": 48, "is_moe": True, "total_params": 57e9},
}

# Common GPU VRAM capacities (GB)
COMMON_GPUS = {
    "RTX 3060": 12, "RTX 3070": 8, "RTX 3080": 10, "RTX 3090": 24,
    "RTX 4060": 8, "RTX 4070": 12, "RTX 4080": 16, "RTX 4090": 24,
    "A100": 40, "A100-80G": 80, "A6000": 48, "V100": 16, "V100-32G": 32,
    "H100": 80, "T4": 16, "L40S": 48, "L4": 24,
}


def estimate_memory(model_name, seq_length=2048, batch_size=4, lora_rank=8, num_modules=2):
    """Estimate GPU memory for QLoRA fine-tuning.

    Returns dict with breakdown in GB.
    """
    spec = MODEL_DB.get(model_name)
    if spec is None:
        # Fuzzy match
        for key in MODEL_DB:
            if model_name.lower() in key.lower():
                spec = MODEL_DB[key]
                model_name = key
                break
        if spec is None:
            return None

    params = spec["params"]
    hidden = spec["hidden"]
    layers = spec["layers"]
    is_moe = spec.get("is_moe", False)
    total_params = spec.get("total_params", params)

    # 1. Model weights (4-bit QLoRA)
    bytes_per_param = 0.5  # 4-bit
    model_gb = params * bytes_per_param / 1e9

    # 2. LoRA adapter memory
    # LoRA params per module: (hidden * rank + rank * hidden) = 2 * hidden * rank
    lora_params = num_modules * (2 * hidden * lora_rank) * layers
    lora_gb = lora_params * 2 / 1e9  # FP16 = 2 bytes/param

    # 3. Activation memory (rough estimate)
    # ~34 * batch_size * seq_length * hidden bytes per layer, but only a fraction
    # is materialized at once with gradient checkpointing
    act_gb = batch_size * seq_length * hidden * 2 / 1e9 * layers * 0.25

    # 4. Optimizer states (AdamW 8-bit = 1 byte/param for LoRA params)
    opt_gb = lora_params * 1 / 1e9

    # 5. Gradients (FP16 for LoRA params)
    grad_gb = lora_params * 2 / 1e9

    # 6. Overhead (~10%)
    overhead = (model_gb + lora_gb + act_gb + opt_gb + grad_gb) * 0.10

    total_gb = model_gb + lora_gb + act_gb + opt_gb + grad_gb + overhead

    return {
        "model_name": model_name,
        "params": params,
        "total_params": total_params,
        "is_moe": is_moe,
        "hidden": hidden,
        "layers": layers,
        "model_gb": model_gb,
        "lora_gb": lora_gb,
        "activation_gb": act_gb,
        "optimizer_gb": opt_gb,
        "gradient_gb": grad_gb,
        "overhead_gb": overhead,
        "total_gb": total_gb,
        "seq_length": seq_length,
        "batch_size": batch_size,
        "lora_rank": lora_rank,
    }


def format_gb(val):
    """Format GB value with appropriate precision."""
    if val < 0.1:
        return "{:.3f}".format(val)
    elif val < 1:
        return "{:.2f}".format(val)
    else:
        return "{:.1f}".format(val)


def main():
    parser = argparse.ArgumentParser(
        description="GPU Memory Estimator for LoRA/QLoRA Fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python memory_check.py qwen2-7b
  python memory_check.py llama3-8b --seq-length 4096 --batch-size 2 --rank 16
  python memory_check.py --list-models
  python memory_check.py qwen2-7b --compare-gpus
        """,
    )
    parser.add_argument("model", nargs="?", help="Model name (e.g. qwen2-7b, llama3-8b)")
    parser.add_argument("--seq-length", type=int, default=2048, help="Max sequence length (default: 2048)")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-GPU batch size (default: 4)")
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank (default: 8)")
    parser.add_argument("--modules", type=int, default=2, help="Number of target modules (default: 2)")
    parser.add_argument("--list-models", action="store_true", help="List all known models")
    parser.add_argument("--compare-gpus", action="store_true", help="Compare against common GPUs")
    args = parser.parse_args()

    if args.list_models:
        print("\nKnown Models:\n")
        for name, spec in MODEL_DB.items():
            params_str = "{:.1f}B".format(spec["params"] / 1e9)
            if spec.get("is_moe"):
                total = spec.get("total_params", spec["params"]) / 1e9
                params_str = "{:.0f}B (MoE, {:.0f}B active)".format(total, spec["params"] / 1e9)
            print("  {:<20s}  {:>6s}  hidden={}, layers={}".format(
                name, params_str, spec["hidden"], spec["layers"]))
        print("\nUse: python memory_check.py <model_name>\n")
        return

    if not args.model:
        parser.print_help()
        return

    result = estimate_memory(
        args.model,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        lora_rank=args.rank,
        num_modules=args.modules,
    )

    if result is None:
        print("[ERROR] Unknown model: {}".format(args.model))
        print("Use --list-models to see available models.")
        print("Or try fuzzy match with partial name (e.g. 'qwen' matches 'qwen2-7b').")
        sys.exit(1)

    # Print report
    print()
    print("=" * 60)
    print("  GPU Memory Estimation -- QLoRA (4-bit)")
    print("=" * 60)
    print("  Model:       {}".format(result["model_name"]))
    if result["is_moe"]:
        print("  Type:        MoE ({:.0f}B total, {:.1f}B active)".format(
            result["total_params"] / 1e9, result["params"] / 1e9))
    else:
        print("  Parameters:  {:.1f}B".format(result["params"] / 1e9))
    print("  Config:      seq_len={}, batch_size={}, rank={}, target_modules={}".format(
        result["seq_length"], result["batch_size"], result["lora_rank"], args.modules))
    print("-" * 60)
    print("  {:<30s} {:>10s}".format("Component", "VRAM"))
    print("-" * 60)
    print("  {:<30s} {:>9s} GB".format("Model Weights (4-bit)", format_gb(result["model_gb"])))
    print("  {:<30s} {:>9s} GB".format("LoRA Adapters (FP16)", format_gb(result["lora_gb"])))
    print("  {:<30s} {:>9s} GB".format("Activations", format_gb(result["activation_gb"])))
    print("  {:<30s} {:>9s} GB".format("Optimizer States (8-bit)", format_gb(result["optimizer_gb"])))
    print("  {:<30s} {:>9s} GB".format("Gradients (FP16)", format_gb(result["gradient_gb"])))
    print("  {:<30s} {:>9s} GB".format("Overhead (~10%)", format_gb(result["overhead_gb"])))
    print("-" * 60)
    print("  {:<30s} {:>9s} GB".format("TOTAL", format_gb(result["total_gb"])))
    print("=" * 60)

    # Recommendations
    total = result["total_gb"]
    print()
    print("Recommendations:")
    print("-" * 40)

    if total < 8:
        print("  [OK] Fits comfortably on most GPUs ( 8GB).")
        print("  Can increase batch_size or sequence length.")
    elif total < 12:
        print("  [OK] Fits on 12GB+ GPUs (RTX 3060/4070, etc.).")
        print("  Consider reducing batch_size if sharing GPU with other tasks.")
    elif total < 16:
        print("  [OK] Fits on 16GB+ GPUs (RTX 4080, T4, V100, etc.).")
        print("  For 12GB GPUs: reduce batch_size to {} or seq_length to {}.".format(
            max(1, args.batch_size // 2), int(args.seq_length * 0.7)))
    elif total < 24:
        print("  [OK] Fits on 24GB GPUs (RTX 3090/4090).")
        print("  For 16GB GPUs: reduce batch_size to {} or use gradient accumulation.".format(
            max(1, args.batch_size // 3)))
    elif total < 40:
        print("  [WARN] Needs 40GB+ GPU (A100, etc.).")
        print("  Suggestions:")
        print("    - Reduce batch_size to {}".format(max(1, args.batch_size // 4)))
        print("    - Reduce seq_length to {}".format(int(args.seq_length * 0.5)))
        print("    - Use a smaller model variant")
    else:
        print("  [WARN] Needs 80GB GPU (A100-80G, H100).")
        print("  Suggestions:")
        print("    - Use DeepSpeed ZeRO-3 or multi-GPU")
        print("    - Try a smaller model or reduce all parameters significantly")

    # GPU comparison
    if args.compare_gpus:
        print()
        print("GPU Compatibility:")
        print("-" * 60)
        print("  {:<20s} {:>8s} {:>12s} {:>12s}".format("GPU", "VRAM", "Fit?", "Headroom"))
        print("-" * 60)
        for gpu_name, gpu_vram in COMMON_GPUS.items():
            headroom = gpu_vram - total
            if headroom >= 4:
                status = "[OK] Yes"
            elif headroom >= 0:
                status = "[WARN] Tight"
            else:
                status = "[FAIL] No"
            print("  {:<20s} {:>5d} GB {:>12s} {:>+9.1f} GB".format(gpu_name, gpu_vram, status, headroom))
        print("-" * 60)
        print("  Rule of thumb: headroom  4GB for safe training.")

    # Quick tips
    print()
    print("Quick Tips:")
    print("  - Actual usage may vary 10-20% depending on model implementation.")
    print("  - Gradient checkpointing reduces activation memory ~60% (already factored in).")
    print("  - Flash Attention 2 further reduces activation memory ~30-50%.")
    print("  - Use --batch-size 1 and increase gradient_accumulation if VRAM is tight.")
    print("  - Run with --compare-gpus to see which GPUs can handle this config.")
    print()


if __name__ == "__main__":
    main()
