#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QLoRA Fine-tuning Training Script -- Ready to Use
===================================================
Just modify the two paths below. All other parameters are preset with best practices.

Usage:
  python train_qlora.py                         # Train from scratch
  python train_qlora.py --validate-only          # Validate data + env only, no training
  python train_qlora.py --auto-mirror            # Auto-set HF mirror for users in China

Recommended config:
  - Rank (r): 8
  - Alpha: 16
  - Target Modules: [q_proj, v_proj]
  - Dropout: 0.05
  - Learning Rate: 2e-4
  - Epochs: 3
  - Batch Size: 4 x Gradient Accumulation: 4 = Effective Batch: 16
  - Max Seq Length: 2048
"""

import os
import sys
import json
import time
import argparse
import glob as glob_mod
import traceback

# ---------------------------------------------------------------
# Configuration -- Only modify these two lines!
# ---------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2-7B-Instruct"        # TODO: Change to your model name or local path
DATA_PATH = "./data/train.jsonl"              # TODO: Change to your JSONL data path
OUTPUT_DIR = "./output/lora_trained"

# ---------------------------------------------------------------
# LoRA Parameters -- Adjust as needed
# ---------------------------------------------------------------

RANK = 8                # LoRA rank: data<1k->4, 1k-5k->8, 5k-20k->16, >20k->32
ALPHA = 16              # alpha = 2 * rank (standard ratio)
DROPOUT = 0.05          # data<1k->0.1, 1k-10k->0.05, >10k->0
LEARNING_RATE = 2e-4    # LoRA base lr: rank=4->1e-4, rank=16->3e-4
EPOCHS = 3              # <500->5, 500-5k->3, 5k-20k->2, >20k->1
BATCH_SIZE = 4          # By VRAM: <8G->1, 8-16G->2, 16-24G->4, >24G->8
GRAD_ACCUM = 4          # Target effective batch = BATCH_SIZE * GRAD_ACCUM = 16
MAX_SEQ_LENGTH = 2048   # Set based on your data P95 token length
TARGET_MODULES = ["q_proj", "v_proj"]  # chat->[q,v] code->[q,k,v,o] math->[q,v,up,down,gate]

# ---------------------------------------------------------------
# CLI Arguments
# ---------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA Fine-tuning Training Script")
    parser.add_argument("--validate-only", action="store_true", help="Validate data and environment only, skip training")
    parser.add_argument("--auto-mirror", action="store_true", help="Auto-configure HF mirror for users in China")
    parser.add_argument("--debug", action="store_true", help="Verbose logging")
    return parser.parse_args()

# ---------------------------------------------------------------
# Auto Mirror Detection for Users in China
# ---------------------------------------------------------------

def configure_mirror(auto_enable=False):
    """Auto-detect and configure HuggingFace mirror for users in China."""
    if os.environ.get("HF_ENDPOINT", "").endswith("hf-mirror.com"):
        print("[OK] HF mirror already configured: hf-mirror.com")
        return

    # Check if model is a local path -- skip mirror if so
    if os.path.isdir(MODEL_NAME) or MODEL_NAME.startswith("./") or MODEL_NAME.startswith("/"):
        return

    if auto_enable:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("[OK] Auto-configured HF mirror: https://hf-mirror.com")
        print("     (use --auto-mirror flag or set HF_ENDPOINT environment variable)")
        return

    # Network probe: check if huggingface.co is reachable
    try:
        import urllib.request
        req = urllib.request.Request("https://huggingface.co", method="HEAD")
        urllib.request.urlopen(req, timeout=5)
        # Reachable, no mirror needed
        return
    except Exception:
        pass

    # Not reachable -- suggest mirror
    print("=" * 60)
    print("  [INFO] huggingface.co is slow or unreachable from your network.")
    print("  Users in China: configure a mirror to accelerate model downloads.")
    print()
    print("  Option 1 (Recommended): Run with --auto-mirror flag")
    print("    python train_qlora.py --auto-mirror")
    print()
    print("  Option 2: Set environment variable manually")
    print("    export HF_ENDPOINT=https://hf-mirror.com")
    print()
    print("  Option 3: Download model via ModelScope first")
    print("    pip install modelscope")
    print("    python -c \"from modelscope import snapshot_download; snapshot_download('{}', cache_dir='./models')\"".format(MODEL_NAME))
    print("    Then set MODEL_NAME = './models/{}/...' in this script".format(MODEL_NAME.split("/")[-1]))
    print("=" * 60)
    print()

# ---------------------------------------------------------------
# Environment Check
# ---------------------------------------------------------------

def check_environment():
    """Check if training environment is ready. Prints solutions for each issue, then exits if any fail."""
    errors = []
    warnings = []

    # 1. PyTorch + CUDA
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
            cuda_ver = torch.version.cuda or "unknown"
            print("[OK] CUDA available: {} ({:.1f} GB), CUDA {}".format(gpu_name, gpu_mem, cuda_ver))

            # GPU compatibility check
            if gpu_mem < 6:
                errors.append("[FAIL] GPU VRAM < 6 GB: minimum for QLoRA 7B training. Consider using a cloud GPU.")
            elif gpu_mem < 8:
                warnings.append("[WARN] GPU VRAM < 8 GB: may need batch_size=1 and short max_seq_length for 7B models.")
        else:
            errors.append("[FAIL] CUDA not available. Install PyTorch CUDA version:")
            errors.append("       pip install torch --index-url https://download.pytorch.org/whl/cu121")
    except ImportError:
        errors.append("[FAIL] PyTorch not installed. Run: pip install lora-trainer[train]")

    # 2. Core dependencies
    for lib in ["transformers", "peft", "datasets", "bitsandbytes", "accelerate"]:
        try:
            __import__(lib)
        except ImportError:
            errors.append("[FAIL] {} not installed. Run: pip install {}".format(lib, lib))

    # 3. Data file
    if not os.path.exists(DATA_PATH):
        errors.append("[FAIL] Data file not found: {}. Please verify the path.".format(DATA_PATH))

    # 4. HuggingFace network (mirror hint handled by configure_mirror)
    if not os.path.isdir(MODEL_NAME) and not MODEL_NAME.startswith("./"):
        hf_endpoint = os.environ.get("HF_ENDPOINT", "")
        if "hf-mirror.com" in hf_endpoint:
            print("[OK] HF mirror configured: hf-mirror.com")

    # 5. Data validation (quick scan)
    if os.path.exists(DATA_PATH):
        try:
            with open(DATA_PATH, "r", encoding="utf-8") as f:
                lines = [l for l in f if l.strip()]
            if len(lines) == 0:
                errors.append("[FAIL] Data file is empty: {}".format(DATA_PATH))
            else:
                # Validate JSON
                bad_lines = []
                for i, line in enumerate(lines):
                    try:
                        json.loads(line)
                    except json.JSONDecodeError:
                        bad_lines.append(i + 1)
                if bad_lines:
                    errors.append("[FAIL] {} invalid JSON line(s): {}".format(len(bad_lines), bad_lines[:10]))
                else:
                    print("[OK] Data file valid: {} lines, all parseable JSON".format(len(lines)))
        except Exception as e:
            errors.append("[FAIL] Cannot read data file: {}".format(e))

    # Print warnings
    if warnings:
        for w in warnings:
            print(w)

    if errors:
        print("\n[WARN] Environment check failed:\n")
        for e in errors:
            print("  {}".format(e))
        print("\nFix the issues above and re-run: python train_qlora.py")
        sys.exit(1)

    print("[OK] Environment check passed\n")
    return torch

# ---------------------------------------------------------------
# Data Validation (Detailed)
# ---------------------------------------------------------------

def validate_data_detailed():
    """Detailed data quality check before training."""
    print("=" * 60)
    print("  Data Validation")
    print("=" * 60)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    total_lines = len(lines)
    print("  Total lines: {}".format(total_lines))

    samples = []
    for i, line in enumerate(lines):
        try:
            samples.append(json.loads(line))
        except json.JSONDecodeError as e:
            print("[WARN] Line {} JSON parse error: {}".format(i + 1, e))
            continue

    if len(samples) == 0:
        print("[FAIL] No valid JSON lines found.")
        sys.exit(1)

    # Detect format
    first = samples[0]
    detected_format = "unknown"
    if "messages" in first:
        detected_format = "messages"
    elif "instruction" in first and "output" in first:
        detected_format = "instruction-output"
    elif "conversations" in first:
        detected_format = "conversations"
    elif "text" in first:
        detected_format = "cpt (text)"
    print("  Detected format: {}".format(detected_format))

    # Check for empty/invalid content
    empty_count = 0
    key_fields = []
    if detected_format == "instruction-output":
        key_fields = ["instruction", "output"]
    elif detected_format == "messages":
        key_fields = ["messages"]
    elif detected_format == "conversations":
        key_fields = ["conversations"]
    elif detected_format == "cpt (text)":
        key_fields = ["text"]

    for i, sample in enumerate(samples):
        for field in key_fields:
            val = sample.get(field)
            if val is None or (isinstance(val, str) and not val.strip()) or (isinstance(val, list) and len(val) == 0):
                empty_count += 1
                if empty_count <= 10:
                    print("[WARN] Line {}: empty '{}' field".format(i + 1, field))

    if empty_count > 0:
        print("[WARN] {} record(s) have empty key fields".format(empty_count))
    else:
        print("[OK] No empty key fields found")

    # Token length estimation (Chinese: ~2 tokens/char, English: ~0.3 tokens/char)
    import re
    lengths = []
    for sample in samples:
        text = ""
        for field in key_fields:
            val = sample.get(field, "")
            if isinstance(val, list):
                for msg in val:
                    text += msg.get("content", msg.get("value", ""))
            else:
                text += str(val)
        cjk_chars = len(re.findall(r'[一-鿿㐀-䶿]', text))
        other_chars = len(text) - cjk_chars
        est_tokens = int(cjk_chars * 2.0 + other_chars * 0.3)
        lengths.append(est_tokens)

    if lengths:
        lengths.sort()
        avg_tokens = sum(lengths) / len(lengths)
        p50_tokens = lengths[len(lengths) // 2]
        p95_tokens = lengths[int(len(lengths) * 0.95)]

        print("  Estimated tokens: avg={:.0f}, median={:.0f}, p95={:.0f}".format(avg_tokens, p50_tokens, p95_tokens))

        if p95_tokens > MAX_SEQ_LENGTH:
            print("[WARN] P95 token length ({}) exceeds MAX_SEQ_LENGTH ({}).".format(p95_tokens, MAX_SEQ_LENGTH))
            print("       Consider increasing MAX_SEQ_LENGTH or {} records will be truncated.".format(
                sum(1 for l in lengths if l > MAX_SEQ_LENGTH)))

        # Estimate training time
        effective_batch = BATCH_SIZE * GRAD_ACCUM
        est_time_min = (total_lines * avg_tokens * EPOCHS) / (2000 * effective_batch)
        print("  Estimated training time: {:.0f}-{:.0f} minutes (RTX 4090, QLoRA 7B)".format(
            est_time_min * 0.7, est_time_min * 1.3))

    print("=" * 60 + "\n")
    return samples, detected_format

# ---------------------------------------------------------------
# Load Model
# ---------------------------------------------------------------

def load_model_and_tokenizer():
    """Load 4-bit quantized model + inject LoRA."""
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

    print("Loading base model (4-bit QLoRA)...")

    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
    except Exception as e:
        msg = str(e)
        if "HTTPError" in msg or "ConnectionError" in msg or "ConnectTimeout" in msg or "MaxRetryError" in msg:
            print("\n[FAIL] Network error downloading model: {}".format(MODEL_NAME))
            print("  Fix: Run with --auto-mirror or set HF_ENDPOINT=https://hf-mirror.com")
            print("  Or download manually via ModelScope: pip install modelscope")
        elif "out of memory" in msg.lower() or "OOM" in msg:
            print("\n[FAIL] GPU out of memory loading model.")
            print("  Fix: Close other GPU processes, or use a smaller model.")
        elif "not found" in msg.lower() or "does not appear" in msg.lower():
            print("\n[FAIL] Model not found: {}".format(MODEL_NAME))
            print("  Fix: Check model name or local path.")
        else:
            print("\n[FAIL] Model loading error: {}".format(msg))
        sys.exit(1)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            padding_side="right",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print("\n[FAIL] Tokenizer loading error: {}".format(e))
        sys.exit(1)

    lora_config = LoraConfig(
        r=RANK,
        lora_alpha=ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer

# ---------------------------------------------------------------
# Load & Prepare Data
# ---------------------------------------------------------------

def load_and_prepare_data(tokenizer):
    """Load JSONL data, auto-detect format, tokenize."""
    from datasets import load_dataset

    print("Loading data...")

    try:
        dataset = load_dataset("json", data_files=DATA_PATH)
    except Exception as e:
        print("\n[FAIL] Data loading error: {}".format(e))
        print("  Fix: Verify DATA_PATH points to a valid JSONL file.")
        print("  Each line must be a complete JSON object.")
        sys.exit(1)

    # Auto-detect format
    sample = dataset["train"][0] if isinstance(dataset, dict) else dataset[0]

    if "messages" in sample:
        def format_example(example):
            parts = []
            for msg in example["messages"]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                parts.append("<|{}|>\n{}".format(role, content))
            return {"text": "\n".join(parts)}
        print("    Detected format: messages")
    elif "instruction" in sample and "output" in sample:
        def format_example(example):
            text = "### Instruction:\n{}\n\n### Response:\n{}".format(
                example['instruction'], example['output'])
            return {"text": text}
        print("    Detected format: instruction-output")
    elif "conversations" in sample:
        def format_example(example):
            parts = []
            for turn in example["conversations"]:
                role = turn.get("role", turn.get("from", "user"))
                content = turn.get("content", turn.get("value", ""))
                parts.append("<|{}|>\n{}".format(role, content))
            return {"text": "\n".join(parts)}
        print("    Detected format: conversations")
    else:
        print("   [WARN] Unknown format, available fields: {}".format(list(sample.keys())))
        print("   Please modify the format_example function in load_and_prepare_data() to match your data format.")
        sys.exit(1)

    # Train/eval split
    if isinstance(dataset, dict):
        dataset = dataset["train"]
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)

    # Format
    split_dataset = split_dataset.map(format_example)

    # Tokenize
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=MAX_SEQ_LENGTH,
        )

    tokenized = split_dataset.map(
        tokenize,
        batched=True,
        remove_columns=split_dataset["train"].column_names,
    )

    print("   Training set: {} samples | Eval set: {} samples".format(
        len(tokenized["train"]), len(tokenized["test"])))
    return tokenized

# ---------------------------------------------------------------
# Training
# ---------------------------------------------------------------

def get_latest_checkpoint(output_dir):
    """Find the latest checkpoint directory if any exist."""
    pattern = os.path.join(output_dir, "checkpoint-*")
    checkpoints = sorted(glob_mod.glob(pattern))
    if checkpoints:
        return checkpoints[-1]
    return None


def train(model, tokenizer, tokenized_dataset):
    """Execute LoRA training loop with checkpoint resume and OOM recovery."""
    from transformers import (
        TrainingArguments,
        Trainer,
        DataCollatorForSeq2Seq,
    )
    import torch

    num_samples = len(tokenized_dataset["train"])
    current_batch_size = BATCH_SIZE
    max_oom_retries = 2  # halve batch_size at most twice

    for oom_attempt in range(max_oom_retries + 1):
        if oom_attempt > 0:
            current_batch_size = max(1, current_batch_size // 2)
            print("\n[WARN] OOM detected. Retrying with batch_size={} (halved from {}).".format(
                current_batch_size, current_batch_size * 2))
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        total_steps = max(1, (num_samples // (current_batch_size * GRAD_ACCUM)) * EPOCHS)
        warmup = min(max(10, total_steps // 10), max(1, total_steps - 1))
        save_steps = max(50, total_steps // 5)

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=current_batch_size,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LEARNING_RATE,
            warmup_steps=warmup,
            lr_scheduler_type="cosine",
            logging_steps=10,
            save_steps=save_steps,
            eval_steps=save_steps,
            evaluation_strategy="steps",
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            bf16=True,
            gradient_checkpointing=True,
            optim="adamw_8bit",
            neftune_noise_alpha=5,
            max_grad_norm=1.0,
            report_to="none",
        )

        # Check for existing checkpoint to resume from
        resume_path = get_latest_checkpoint(OUTPUT_DIR)
        if resume_path:
            print("\n[INFO] Found checkpoint: {}".format(resume_path))
            print("     Resuming training from checkpoint...")
        else:
            print("\nStarting training (total_steps={}, warmup={})".format(total_steps, warmup))

        print("   Effective batch size: {} x {} = {}".format(
            current_batch_size, GRAD_ACCUM, current_batch_size * GRAD_ACCUM))
        print("   Logs and checkpoints saved to: {}\n".format(OUTPUT_DIR))

        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, return_tensors="pt")

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            data_collator=data_collator,
        )

        try:
            trainer.train(resume_from_checkpoint=resume_path)
            break  # Success, exit retry loop
        except RuntimeError as e:
            msg = str(e)
            if ("out of memory" in msg.lower() or "CUDA out of memory" in msg) and oom_attempt < max_oom_retries:
                continue  # Retry with smaller batch_size
            else:
                print("\n[FAIL] Training error: {}".format(msg))
                if "out of memory" in msg.lower():
                    print("  Suggestion: Reduce MAX_SEQ_LENGTH, or use a smaller model.")
                print("  Check ./output/lora_trained/ for auto-saved checkpoints (if any).")
                print("  Re-run python train_qlora.py to resume from the latest checkpoint.")
                sys.exit(1)
        except KeyboardInterrupt:
            print("\n[INFO] Training interrupted. Checkpoints saved to {}.".format(OUTPUT_DIR))
            print("     Re-run python train_qlora.py to resume from the latest checkpoint.")
            sys.exit(0)

    # Save LoRA adapter
    adapter_path = os.path.join(OUTPUT_DIR, "final_adapter")
    print("\nSaving LoRA weights to {}".format(adapter_path))
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    # Save training config for reproducibility
    config_path = os.path.join(OUTPUT_DIR, "training_config.json")
    with open(config_path, "w") as f:
        json.dump({
            "model_name": MODEL_NAME,
            "data_path": DATA_PATH,
            "rank": RANK, "alpha": ALPHA, "dropout": DROPOUT,
            "learning_rate": LEARNING_RATE, "epochs": EPOCHS,
            "batch_size": current_batch_size, "grad_accum": GRAD_ACCUM,
            "max_seq_length": MAX_SEQ_LENGTH, "target_modules": TARGET_MODULES,
            "num_train_samples": num_samples,
        }, f, indent=2, ensure_ascii=False)
    print("   Config saved to {}".format(config_path))

    print("\n[OK] Training complete!")
    print("   LoRA weights: {}".format(adapter_path))
    print("   Load for inference: model = PeftModel.from_pretrained(base_model, '{}')".format(adapter_path))
    return adapter_path

# ---------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    if args.debug:
        os.environ["TRANSFORMERS_VERBOSITY"] = "debug"
        print("[DEBUG] Verbose mode enabled\n")

    print("=" * 60)
    print("  QLoRA Fine-tuning Training")
    print("=" * 60)
    print("  Model: {}".format(MODEL_NAME))
    print("  Data: {}".format(DATA_PATH))
    print("  Rank: {} | Alpha: {} | LR: {}".format(RANK, ALPHA, LEARNING_RATE))
    print("  Epochs: {} | Batch: {}x{} | Seq: {}".format(EPOCHS, BATCH_SIZE, GRAD_ACCUM, MAX_SEQ_LENGTH))
    print("=" * 60 + "\n")

    # Mirror detection (before any network access)
    configure_mirror(auto_enable=args.auto_mirror)

    # Environment check (always runs)
    torch = check_environment()

    # Detailed data validation (always runs before training)
    samples, format_type = validate_data_detailed()

    if args.validate_only:
        print("[OK] Validation complete. Data and environment are ready for training.")
        print("    Run without --validate-only to start training.")
        sys.exit(0)

    # Load model
    print("=" * 60)
    print("  Loading Model")
    print("=" * 60)
    model, tokenizer = load_model_and_tokenizer()

    # Load data
    print("=" * 60)
    print("  Preparing Data")
    print("=" * 60)
    tokenized_dataset = load_and_prepare_data(tokenizer)

    # Train
    print("=" * 60)
    print("  Training")
    print("=" * 60)
    adapter_path = train(model, tokenizer, tokenized_dataset)
