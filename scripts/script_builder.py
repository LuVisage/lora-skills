"""
训练脚本生成模块 — 根据推荐配置生成可直接运行的训练/推理脚本。
"""

import os
from datetime import datetime
from typing import Dict

import yaml


class ScriptBuilder:
    """根据 LoRA 配置生成训练脚本、推理脚本和配置文件。"""

    def __init__(
        self,
        config: Dict,
        output_dir: str = "./output",
        data_format: str = "instruction-output",
        max_seq_length: int = 2048,
    ):
        self.config = config
        self.output_dir = output_dir
        self.data_format = data_format
        self.max_seq_length = max_seq_length
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── 训练脚本 ────────────────────────────────────────────

    def _build_format_fn(self) -> str:
        """根据数据格式生成对应的格式化函数代码。"""
        if self.data_format in ("messages", "conversations"):
            return '''def format_messages(example):
    """将 messages 格式化为 ChatML 训练文本。"""
    messages = example["messages"]
    # 使用 tokenizer 的 chat template（如果有的话）
    # 否则用通用 ChatML 格式
    text_parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        text_parts.append(f"<|{role}|>\\n{content}")
    text = "\\n".join(text_parts) + "\\n<|assistant|>\\n"
    return {"text": text}'''
        elif self.data_format == "instruction-output":
            return '''def format_instruction(example):
    """将 instruction/output 格式化为训练文本。"""
    text = f"### Instruction:\\n{example['instruction']}\\n\\n### Response:\\n{example['output']}"
    return {"text": text}'''
        else:
            # 未知格式：同时生成三种尝试
            return '''def format_auto(example):
    """自动检测格式并转换。"""
    if "messages" in example and isinstance(example["messages"], list):
        parts = []
        for msg in example["messages"]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|{role}|>\\n{content}")
        text = "\\n".join(parts) + "\\n<|assistant|>\\n"
    elif "instruction" in example and "output" in example:
        text = f"### Instruction:\\n{example['instruction']}\\n\\n### Response:\\n{example['output']}"
    elif "conversations" in example:
        parts = []
        for turn in example["conversations"]:
            role = turn.get("role", turn.get("from", "user"))
            content = turn.get("content", turn.get("value", ""))
            parts.append(f"<|{role}|>\\n{content}")
        text = "\\n".join(parts) + "\\n<|assistant|>\\n"
    else:
        raise ValueError(f"不支持的数据格式: {list(example.keys())}")
    return {"text": text}'''

    def build_training_script(self) -> str:
        """生成完整的 QLoRA 训练脚本。"""
        r = self.config["rank"]["value"]
        alpha = self.config["alpha"]["value"]
        modules = self.config["target_modules"]["value"]
        dropout = self.config["dropout"]["value"]
        lr = self.config["learning_rate"]["value"]
        epochs = self.config["epochs"]["value"]
        bs = self.config["batch_size"]["value"]
        ga = self.config["gradient_accumulation"]["value"]
        max_len = self.max_seq_length

        modules_str = str(modules)
        format_fn_code = self._build_format_fn()

        # 计算 warmup steps（总步数的 10%，但不超总步数-1）
        num_samples = self.config.get("num_samples", 1000)
        total_steps = max(1, (num_samples // (bs * ga)) * epochs if bs * ga > 0 else 100)
        warmup = min(max(10, total_steps // 10), max(1, total_steps - 1))

        script = f'''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LoRA 微调训练脚本
由 LoRA Skill 自动生成于 {self.timestamp}

推荐配置:
  - Rank (r): {r}
  - Alpha: {alpha}
  - Target Modules: {modules_str}
  - Dropout: {dropout}
  - Learning Rate: {lr:.2e}
  - Epochs: {epochs}
  - Batch Size: {bs} × Gradient Accumulation: {ga} = Effective Batch: {bs * ga}
  - Max Seq Length: {max_len}
  - Data Format: {self.data_format}
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from datasets import load_dataset, Dataset
import os

# ═══════════════════════════════════════════════════════════
# 配置区域 — 请根据实际情况修改以下路径
# ═══════════════════════════════════════════════════════════

MODEL_NAME = "{{{{model_path}}}}"   # TODO: 模型路径或 HuggingFace ID
DATA_PATH = "{{{{data_path}}}}"     # TODO: JSONL 数据文件路径
OUTPUT_DIR = "{self.output_dir}/lora_{self.timestamp}"

# ═══════════════════════════════════════════════════════════
# LoRA 配置（由 Skill 智能推荐）
# ═══════════════════════════════════════════════════════════

LORA_CONFIG = LoraConfig(
    r={r},
    lora_alpha={alpha},
    target_modules={modules_str},
    lora_dropout={dropout},
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# ═══════════════════════════════════════════════════════════
# 量化配置（QLoRA 4-bit）
# ═══════════════════════════════════════════════════════════

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,  # 与 bf16 训练一致，避免 dtype cast 损耗
    bnb_4bit_use_double_quant=True,
)

# ═══════════════════════════════════════════════════════════
# 训练参数
# ═══════════════════════════════════════════════════════════

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs={epochs},
    per_device_train_batch_size={bs},
    gradient_accumulation_steps={ga},
    learning_rate={lr:.2e},
    warmup_steps={warmup},
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_steps=min(500, max(50, total_steps // 5)),
    eval_steps=min(500, max(50, total_steps // 5)),
    evaluation_strategy="steps" if os.path.exists(DATA_PATH.replace("train", "eval")) else "no",
    save_total_limit=3,
    load_best_model_at_end=True,
    bf16=True,  # Ampere+ GPU (RTX 30/40, A100) 推荐 bf16，更稳定；老 GPU 改 fp16=True
    gradient_checkpointing=True,
    optim="adamw_8bit",
    neftune_noise_alpha=5,   # NEFTune: embedding 加噪，稳定提升 3-5%
    max_grad_norm=1.0,        # 梯度裁剪，防止 loss spike
    report_to="none",
)

# ═══════════════════════════════════════════════════════════
# 加载模型
# ═══════════════════════════════════════════════════════════

print("🔧 加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="flash_attention_2",  # 2-3× 加速，20-30% 省显存 (Ampere+)
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    padding_side="right",
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 准备 k-bit 训练 + 注入 LoRA
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LORA_CONFIG)
model.print_trainable_parameters()

# ═══════════════════════════════════════════════════════════
# 加载数据
# ═══════════════════════════════════════════════════════════

print("📊 加载数据...")

{format_fn_code}

dataset = load_dataset("json", data_files=DATA_PATH)

# 自动 split：如果没有单独的 eval 文件，从训练集切 10%
if "train" not in dataset:
    dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)

format_fn_name = [k for k in locals().keys() if k.startswith("format_")][0]
format_fn = locals()[format_fn_name]
dataset = dataset.map(format_fn)

def tokenize(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        max_length={max_len},
    )

tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset["train"].column_names)

# ═══════════════════════════════════════════════════════════
# 训练
# ═══════════════════════════════════════════════════════════

print("🚀 开始训练...")
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, return_tensors="pt")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized.get("test"),
    data_collator=data_collator,
)

trainer.train()

# ═══════════════════════════════════════════════════════════
# 保存
# ═══════════════════════════════════════════════════════════

adapter_path = os.path.join(OUTPUT_DIR, "final_adapter")
print(f"💾 保存 LoRA 权重到 {{adapter_path}}...")
model.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path)

print("✅ 训练完成！")
print(f"   LoRA 权重: {{adapter_path}}")
print(f"   使用方式: model = PeftModel.from_pretrained(base_model, '{{adapter_path}}')")
'''
        return script

    # ── 推理脚本 ────────────────────────────────────────────

    def _build_inference_format_fn(self) -> str:
        """根据训练数据格式生成对应的推理格式化代码。与训练时保持一致。"""
        if self.data_format in ("messages", "conversations"):
            return '''def format_prompt(prompt: str) -> str:
    """构造 ChatML 格式 prompt（与训练时一致）。"""
    return f"<|user|>\\n{prompt}\\n<|assistant|>\\n"'''
        else:
            return '''def format_prompt(prompt: str) -> str:
    """构造 Instruction 格式 prompt（与训练时一致）。"""
    return f"### Instruction:\\n{prompt}\\n\\n### Response:\\n"'''

    def build_inference_script(self) -> str:
        """生成 LoRA 模型推理脚本。"""
        format_fn_code = self._build_inference_format_fn()
        script = f'''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LoRA 模型推理脚本
由 LoRA Skill 自动生成 — 训练数据格式: {self.data_format}
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ═══════════════════════════════════════════════
# 配置 — 请修改为你的实际路径
# ═══════════════════════════════════════════════

BASE_MODEL = "{{{{model_path}}}}"     # TODO: 基座模型路径
LORA_PATH = "{{{{lora_path}}}}"       # TODO: LoRA adapter 路径


def load_model():
    """加载基座模型 + LoRA adapter。"""
    print("🔧 加载模型...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, LORA_PATH)
    model = model.merge_and_unload()  # 合并权重，推理更快
    model.eval()
    print("✅ 模型加载完成")
    return model, tokenizer


{format_fn_code}


def chat(model, tokenizer, prompt: str, max_tokens: int = 512) -> str:
    """生成回复。使用与微调时一致的 prompt 格式。"""
    formatted = format_prompt(prompt)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 去掉 prompt 部分，只保留模型输出
    if formatted in response:
        response = response[len(formatted):].strip()
    return response


if __name__ == "__main__":
    model, tokenizer = load_model()

    print("\\n🤖 LoRA 模型对话测试 (输入 'quit' 退出)\\n")
    while True:
        user_input = input("👤 你: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            break
        reply = chat(model, tokenizer, user_input)
        print(f"🤖 助手: {reply}\\n")
'''
        return script

    # ── 配置文件 ────────────────────────────────────────────

    def build_config_yaml(self) -> str:
        """生成 YAML 配置文件。"""
        config_dict = {
            "skill_version": "2.2.1",
            "generated_at": self.timestamp,
            "lora_config": {
                "r": self.config["rank"]["value"],
                "lora_alpha": self.config["alpha"]["value"],
                "target_modules": self.config["target_modules"]["value"],
                "lora_dropout": self.config["dropout"]["value"],
                "learning_rate": self.config["learning_rate"]["value"],
                "num_train_epochs": self.config["epochs"]["value"],
                "per_device_train_batch_size": self.config["batch_size"]["value"],
                "gradient_accumulation_steps": self.config["gradient_accumulation"]["value"],
            },
            "recommendations": {
                "rank_reason": self.config["rank"]["reason"],
                "alpha_reason": self.config["alpha"]["reason"],
                "modules_reason": self.config["target_modules"]["reason"],
                "dropout_reason": self.config["dropout"]["reason"],
                "lr_reason": self.config["learning_rate"]["reason"],
                "epochs_reason": self.config["epochs"]["reason"],
                "batch_size_reason": self.config["batch_size"]["reason"],
            },
            "meta": {
                "task_type": self.config.get("task_type", "chat"),
                "model_size": self.config.get("model_size", "7b"),
                "num_samples": self.config.get("num_samples", 0),
            },
        }
        return yaml.dump(config_dict, allow_unicode=True, default_flow_style=False)

    # ── 批量生成 ────────────────────────────────────────────

    def build_all(self) -> Dict[str, str]:
        """生成所有文件，返回文件路径字典。"""
        os.makedirs(self.output_dir, exist_ok=True)

        train_script = self.build_training_script()
        train_path = os.path.join(
            self.output_dir, f"train_lora_{self.timestamp}.py"
        )
        with open(train_path, "w", encoding="utf-8") as f:
            f.write(train_script)

        inference_script = self.build_inference_script()
        infer_path = os.path.join(self.output_dir, "inference_lora.py")
        with open(infer_path, "w", encoding="utf-8") as f:
            f.write(inference_script)

        config_yaml = self.build_config_yaml()
        config_path = os.path.join(self.output_dir, "lora_config.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(config_yaml)

        return {
            "train_script": train_path,
            "inference_script": infer_path,
            "config": config_path,
        }
