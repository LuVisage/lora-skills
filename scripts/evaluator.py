"""
评估模块 — 训练前后的模型效果评估。
"""

import json
from typing import Dict, List


class Evaluator:
    """对 LoRA 微调前后的模型做基础质量评估。"""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.test_cases = self._load_test_cases()

    def _load_test_cases(self) -> List[Dict]:
        """加载测试数据。"""
        data = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data[:100]  # 最多评估 100 条

    def build_eval_prompts(self) -> List[str]:
        """构建评估用的 prompt 列表。"""
        prompts = []
        for item in self.test_cases:
            instr = item.get("instruction", "")
            prompts.append(instr)
        return prompts

    def generate_eval_metrics(self) -> Dict:
        """生成评估指标摘要。"""
        if not self.test_cases:
            return {"error": "无测试数据"}

        prompts = self.build_eval_prompts()
        total = len(prompts)

        avg_instr_len = sum(len(p) for p in prompts) / max(total, 1)

        return {
            "total_test_cases": total,
            "avg_instruction_length": round(avg_instr_len, 1),
            "recommended_metrics": [
                "BLEU",
                "ROUGE-L",
                "Perplexity (PPL)",
                "人工评估 (A/B Test)",
            ],
            "eval_script_available": True,
        }

    def build_eval_script(self) -> str:
        """生成评估脚本。"""
        script = '''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型评估脚本 — 对比基座模型与 LoRA 微调后的效果
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

# ═══════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════

BASE_MODEL = "{{model_path}}"
LORA_PATH = "{{lora_path}}"
TEST_DATA = "{{test_data}}"


def load_base_model():
    """加载基座模型。"""
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    return model, tokenizer


def load_lora_model():
    """加载 LoRA 微调后模型。"""
    base, tokenizer = load_base_model()
    model = PeftModel.from_pretrained(base, LORA_PATH)
    model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt: str) -> str:
    """生成回复。"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    # 加载数据
    with open(TEST_DATA, "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f if line.strip()]

    # 加载模型
    print("加载基座模型...")
    base_model, tokenizer = load_base_model()
    print("加载 LoRA 模型...")
    lora_model, _ = load_lora_model()

    # 评估
    results = []
    for item in tqdm(test_data[:50], desc="评估中"):
        prompt = item["instruction"]
        expected = item.get("output", "")

        base_out = generate(base_model, tokenizer, prompt)
        lora_out = generate(lora_model, tokenizer, prompt)

        results.append({
            "instruction": prompt,
            "expected": expected,
            "base_model_output": base_out,
            "lora_model_output": lora_out,
        })

    # 保存结果
    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ 评估完成，结果保存到 eval_results.json")


if __name__ == "__main__":
    main()
'''
        return script
