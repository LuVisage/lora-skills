#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QLoRA 微调训练脚本 — 开箱即用
================================
只需修改下面两个路径，其他参数已根据最佳实践预设好。
运行: python train_qlora.py

推荐配置:
  - Rank (r): 8
  - Alpha: 16
  - Target Modules: [q_proj, v_proj]
  - Dropout: 0.05
  - Learning Rate: 2e-4
  - Epochs: 3
  - Batch Size: 4 × Gradient Accumulation: 4 = Effective Batch: 16
  - Max Seq Length: 2048
"""

import os
import sys

# ═══════════════════════════════════════════════════════════════
# 🔧 配置区域 — 只需修改这两行！
# ═══════════════════════════════════════════════════════════════

MODEL_NAME = "Qwen/Qwen2-7B-Instruct"        # TODO: 改成你的模型名或本地路径
DATA_PATH = "./data/train.jsonl"              # TODO: 改成你的 JSONL 数据路径
OUTPUT_DIR = "./output/lora_trained"

# ═══════════════════════════════════════════════════════════════
# ⚙️ LoRA 参数 — 可根据需要调整
# ═══════════════════════════════════════════════════════════════

RANK = 8                # LoRA rank: 数据<1k→4, 1k-5k→8, 5k-20k→16, >20k→32
ALPHA = 16              # alpha = 2×rank 是标准比例
DROPOUT = 0.05          # 数据<1k→0.1, 1k-10k→0.05, >10k→0
LEARNING_RATE = 2e-4    # LoRA 基准 lr。rank=4→1e-4, rank=16→3e-4
EPOCHS = 3              # <500条→5, 500-5k→3, 5k-20k→2, >20k→1
BATCH_SIZE = 4          # 根据显存: <8G→1, 8-16G→2, 16-24G→4, >24G→8
GRAD_ACCUM = 4          # 目标 effective batch = BATCH_SIZE × GRAD_ACCUM ≈ 16
MAX_SEQ_LENGTH = 2048   # 根据你的数据 P95 token 长度设置
TARGET_MODULES = ["q_proj", "v_proj"]  # chat→[q,v]  code→[q,k,v,o]  math→[q,v,up,down,gate]

# ═══════════════════════════════════════════════════════════════
# 🔍 环境自检
# ═══════════════════════════════════════════════════════════════

def check_environment():
    """检查训练环境是否就绪。遇到问题打印解决方案后退出。"""
    errors = []

    # 1. PyTorch
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
            print(f"✅ CUDA 可用: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            errors.append("❌ CUDA 不可用。请安装 PyTorch CUDA 版本: pip install torch --index-url https://download.pytorch.org/whl/cu121")
    except ImportError:
        errors.append("❌ PyTorch 未安装。运行: pip install lora-trainer[train]")

    # 2. Transformers & PEFT
    for lib in ["transformers", "peft", "datasets", "bitsandbytes", "accelerate"]:
        try:
            __import__(lib)
        except ImportError:
            errors.append(f"❌ {lib} 未安装。运行: pip install {lib}")

    # 3. 数据文件
    if not os.path.exists(DATA_PATH):
        errors.append(f"❌ 数据文件不存在: {DATA_PATH}。请确认路径正确。")

    # 4. HuggingFace 网络（国内用户提示镜像）
    if not os.path.isdir(MODEL_NAME) and not MODEL_NAME.startswith("./"):
        hf_endpoint = os.environ.get("HF_ENDPOINT", "")
        if "hf-mirror.com" not in hf_endpoint:
            print("💡 国内用户提示: 设置 export HF_ENDPOINT=https://hf-mirror.com 可加速下载")
            print("   或使用 ModelScope: pip install modelscope && python -c \"from modelscope import snapshot_download; snapshot_download('{MODEL_NAME}', cache_dir='./models')\"")

    if errors:
        print("\n⚠️ 环境检查未通过:\n")
        for e in errors:
            print(f"  {e}")
        print("\n修复后重新运行: python train_qlora.py")
        sys.exit(1)

    print("✅ 环境检查通过\n")
    return torch

# ═══════════════════════════════════════════════════════════════
# 📦 加载模型
# ═══════════════════════════════════════════════════════════════

def load_model_and_tokenizer():
    """加载 4-bit 量化模型 + LoRA 注入。"""
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

    print("🔧 加载基座模型 (4-bit QLoRA)...")

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

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

# ═══════════════════════════════════════════════════════════════
# 📊 数据加载
# ═══════════════════════════════════════════════════════════════

def load_and_prepare_data(tokenizer):
    """加载 JSONL 数据，自动检测格式并 tokenize。"""
    from datasets import load_dataset

    print("📊 加载数据...")

    dataset = load_dataset("json", data_files=DATA_PATH)

    # 自动检测格式并格式化
    sample = dataset["train"][0] if isinstance(dataset, dict) else dataset[0]

    if "messages" in sample:
        def format_example(example):
            parts = []
            for msg in example["messages"]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                parts.append(f"<|{role}|>\n{content}")
            return {"text": "\n".join(parts)}
        print("   📋 检测到格式: messages")
    elif "instruction" in sample and "output" in sample:
        def format_example(example):
            text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
            return {"text": text}
        print("   📋 检测到格式: instruction-output")
    elif "conversations" in sample:
        def format_example(example):
            parts = []
            for turn in example["conversations"]:
                role = turn.get("role", turn.get("from", "user"))
                content = turn.get("content", turn.get("value", ""))
                parts.append(f"<|{role}|>\n{content}")
            return {"text": "\n".join(parts)}
        print("   📋 检测到格式: conversations")
    else:
        print(f"   ⚠️ 未知格式，可用字段: {list(sample.keys())}")
        print("   请修改 load_and_prepare_data() 中的 format_example 函数适配你的数据格式")
        sys.exit(1)

    # 拆分训练/验证集
    if isinstance(dataset, dict):
        dataset = dataset["train"]
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)

    # 格式化
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

    print(f"   训练集: {len(tokenized['train'])} 条 | 验证集: {len(tokenized['test'])} 条")
    return tokenized

# ═══════════════════════════════════════════════════════════════
# 🚀 训练
# ═══════════════════════════════════════════════════════════════

def train(model, tokenizer, tokenized_dataset):
    """执行 LoRA 训练循环。"""
    from transformers import (
        TrainingArguments,
        Trainer,
        DataCollatorForSeq2Seq,
    )
    import torch

    num_samples = len(tokenized_dataset["train"])
    total_steps = max(1, (num_samples // (BATCH_SIZE * GRAD_ACCUM)) * EPOCHS)
    warmup = min(max(10, total_steps // 10), max(1, total_steps - 1))
    save_steps = max(50, total_steps // 5)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
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

    print(f"\n🚀 开始训练 (total_steps≈{total_steps}, warmup={warmup})")
    print(f"   有效 batch size: {BATCH_SIZE} × {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")
    print(f"   日志和 checkpoint 保存在: {OUTPUT_DIR}\n")

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, return_tensors="pt")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
    )

    trainer.train()

    # 保存 LoRA adapter
    adapter_path = os.path.join(OUTPUT_DIR, "final_adapter")
    print(f"\n💾 保存 LoRA 权重到 {adapter_path}")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    print("\n✅ 训练完成！")
    print(f"   LoRA 权重: {adapter_path}")
    print(f"   推理加载: model = PeftModel.from_pretrained(base_model, '{adapter_path}')")
    return adapter_path

# ═══════════════════════════════════════════════════════════════
# 🏁 入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  QLoRA 微调训练")
    print("=" * 60)
    print(f"  模型: {MODEL_NAME}")
    print(f"  数据: {DATA_PATH}")
    print(f"  Rank: {RANK} | Alpha: {ALPHA} | LR: {LEARNING_RATE}")
    print(f"  Epochs: {EPOCHS} | Batch: {BATCH_SIZE}×{GRAD_ACCUM} | Seq: {MAX_SEQ_LENGTH}")
    print("=" * 60 + "\n")

    torch = check_environment()
    model, tokenizer = load_model_and_tokenizer()
    tokenized_dataset = load_and_prepare_data(tokenizer)
    adapter_path = train(model, tokenizer, tokenized_dataset)
