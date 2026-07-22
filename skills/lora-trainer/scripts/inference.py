#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LoRA 模型推理脚本 — 开箱即用
==============================
加载微调后的 LoRA 适配器，进行交互式对话或批量推理。
运行: python inference.py

用法:
  python inference.py                        # 交互式对话
  python inference.py --prompt "你好"         # 单次推理
  python inference.py --batch questions.txt  # 批量推理
"""

import os
import sys
import argparse

# ═══════════════════════════════════════════════════════════════
# 🔧 配置 — 只需修改这两行
# ═══════════════════════════════════════════════════════════════

BASE_MODEL = "Qwen/Qwen2-7B-Instruct"           # TODO: 基座模型名或路径
LORA_PATH = "./output/lora_trained/final_adapter"  # TODO: LoRA adapter 路径

# ═══════════════════════════════════════════════════════════════
# ⚙️ 推理参数
# ═══════════════════════════════════════════════════════════════

MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9
MERGE_WEIGHTS = True  # 合并 LoRA 权重到基座模型，推理更快

# ═══════════════════════════════════════════════════════════════
# 🔍 环境检查
# ═══════════════════════════════════════════════════════════════

def check_environment():
    """检查推理环境。"""
    errors = []
    try:
        import torch
    except ImportError:
        errors.append("❌ PyTorch 未安装。pip install torch")

    for lib in ["transformers", "peft"]:
        try:
            __import__(lib)
        except ImportError:
            errors.append(f"❌ {lib} 未安装。pip install {lib}")

    if not os.path.exists(LORA_PATH):
        errors.append(f"❌ LoRA adapter 不存在: {LORA_PATH}。请确认训练已完成且路径正确。")

    if errors:
        print("\n".join(errors))
        sys.exit(1)

    print("✅ 环境检查通过")

# ═══════════════════════════════════════════════════════════════
# 📦 加载模型
# ═══════════════════════════════════════════════════════════════

def load_model():
    """加载基座模型 + LoRA adapter。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"🔧 加载模型: {BASE_MODEL}")
    print(f"   加载 LoRA: {LORA_PATH}")

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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(base, LORA_PATH)

    if MERGE_WEIGHTS:
        print("   合并 LoRA 权重到基座模型...")
        model = model.merge_and_unload()

    model.eval()
    print("✅ 模型加载完成\n")
    return model, tokenizer

# ═══════════════════════════════════════════════════════════════
# 💬 对话
# ═══════════════════════════════════════════════════════════════

def format_prompt(prompt: str) -> str:
    """构造与训练时一致的 prompt 格式。默认使用 instruction-output 格式。"""
    # 如果你的数据是 messages 格式，改成 f"<|user|>\n{prompt}\n<|assistant|>\n"
    return f"### Instruction:\n{prompt}\n\n### Response:\n"


def generate(model, tokenizer, prompt: str) -> str:
    """生成回复。"""
    import torch

    formatted = format_prompt(prompt)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 去掉 prompt 部分
    if formatted in response:
        response = response[len(formatted):].strip()
    return response


def interactive_chat(model, tokenizer):
    """交互式对话循环。"""
    print("🤖 LoRA 模型对话 (输入 'quit' 退出, 'reset' 清屏)\n")
    while True:
        try:
            user_input = input("👤 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见!")
            break

        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit", "q"]:
            print("👋 再见!")
            break
        if user_input.lower() == "reset":
            import subprocess
            subprocess.run(["cmd", "/c", "cls"] if os.name == "nt" else ["clear"], shell=False)
            continue

        reply = generate(model, tokenizer, user_input)
        print(f"🤖 助手: {reply}\n")


def batch_inference(model, tokenizer, input_file: str):
    """批量推理：每行一个问题，输出到同目录下的 _answers.txt。"""
    output_file = input_file.rsplit(".", 1)[0] + "_answers.txt"

    with open(input_file, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]

    print(f"📝 批量推理: {len(questions)} 个问题")

    with open(output_file, "w", encoding="utf-8") as out:
        for i, q in enumerate(questions, 1):
            answer = generate(model, tokenizer, q)
            out.write(f"Q{i}: {q}\nA{i}: {answer}\n\n")
            print(f"   [{i}/{len(questions)}] {q[:40]}...")

    print(f"✅ 结果保存到: {output_file}")

# ═══════════════════════════════════════════════════════════════
# 🏁 入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA 模型推理")
    parser.add_argument("--prompt", "-p", type=str, help="单次推理的 prompt")
    parser.add_argument("--batch", "-b", type=str, help="批量推理的问题文件（每行一个）")
    args = parser.parse_args()

    check_environment()
    model, tokenizer = load_model()

    if args.prompt:
        print(f"👤 你: {args.prompt}")
        reply = generate(model, tokenizer, args.prompt)
        print(f"🤖 助手: {reply}")
    elif args.batch:
        batch_inference(model, tokenizer, args.batch)
    else:
        interactive_chat(model, tokenizer)
