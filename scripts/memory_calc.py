"""
显存计算模块 — 估算 LoRA 微调所需的 GPU 显存。
"""

from typing import Dict, Tuple

# ── 模型规格数据库 ─────────────────────────────────────────
# 记录常见模型的 hidden_dim、num_layers 等信息，用于精确估算

MODEL_DB = {
    "qwen2-0.5b": {"params": 0.5e9, "hidden": 896, "layers": 24},
    "qwen2-1.5b": {"params": 1.5e9, "hidden": 1536, "layers": 28},
    "qwen2-7b": {"params": 7e9, "hidden": 3584, "layers": 28},
    "qwen2-72b": {"params": 72e9, "hidden": 8192, "layers": 80},
    "llama3-8b": {"params": 8e9, "hidden": 4096, "layers": 32},
    "llama3-70b": {"params": 70e9, "hidden": 8192, "layers": 80},
    "llama2-7b": {"params": 7e9, "hidden": 4096, "layers": 32},
    "llama2-13b": {"params": 13e9, "hidden": 5120, "layers": 40},
    "llama2-70b": {"params": 70e9, "hidden": 8192, "layers": 80},
    "mistral-7b": {"params": 7e9, "hidden": 4096, "layers": 32},
    "yi-6b": {"params": 6e9, "hidden": 4096, "layers": 32},
    "yi-34b": {"params": 34e9, "hidden": 7168, "layers": 60},
    "deepseek-7b": {"params": 7e9, "hidden": 4096, "layers": 30},
    "deepseek-67b": {"params": 67e9, "hidden": 8192, "layers": 95},
    "chatglm3-6b": {"params": 6e9, "hidden": 4096, "layers": 28},
    "baichuan2-7b": {"params": 7e9, "hidden": 4096, "layers": 32},
    "baichuan2-13b": {"params": 13e9, "hidden": 5120, "layers": 40},
    "phi-3-mini": {"params": 3.8e9, "hidden": 3072, "layers": 32},
    "phi-3-small": {"params": 7e9, "hidden": 4096, "layers": 32},
    "phi-3-medium": {"params": 14e9, "hidden": 5120, "layers": 40},
    "gemma-2b": {"params": 2e9, "hidden": 2048, "layers": 18},
    "gemma-7b": {"params": 7e9, "hidden": 3072, "layers": 28},
    "internlm2-7b": {"params": 7e9, "hidden": 4096, "layers": 32},
    "internlm2-20b": {"params": 20e9, "hidden": 6144, "layers": 48},
}


class MemoryCalculator:
    """根据模型、序列长度、batch size 等参数估算 GPU 显存需求。"""

    def __init__(
        self,
        model_name: str = "",
        model_params: int = 0,
        seq_length: int = 2048,
        batch_size: int = 4,
        lora_r: int = 8,
    ):
        # 尝试从数据库匹配模型
        spec = self._lookup_model(model_name)
        if spec:
            self.model_params = spec["params"]
            self.hidden_dim = spec["hidden"]
            self.num_layers = spec["layers"]
        elif model_params:
            self.model_params = model_params
            self.hidden_dim = self._estimate_hidden_dim()
            self.num_layers = self._estimate_num_layers()
        else:
            # 默认 7B
            self.model_params = 7e9
            self.hidden_dim = 4096
            self.num_layers = 32

        self.seq_length = seq_length
        self.batch_size = batch_size
        self.lora_r = lora_r

    def _lookup_model(self, name: str) -> Dict | None:
        """模糊匹配模型名称，返回规格参数。"""
        if not name:
            return None
        name_lower = name.lower().replace("-", "").replace("_", "")
        for key, spec in MODEL_DB.items():
            if key.replace("-", "") in name_lower or name_lower in key.replace("-", ""):
                return spec
        return None

    def _estimate_hidden_dim(self) -> int:
        """根据参数量估算隐藏维度。"""
        if self.model_params <= 1e9:
            return 2048
        elif self.model_params <= 3e9:
            return 2560
        elif self.model_params <= 8e9:
            return 4096
        elif self.model_params <= 14e9:
            return 5120
        elif self.model_params <= 34e9:
            return 7168
        else:
            return 8192

    def _estimate_num_layers(self) -> int:
        """根据参数量估算层数。"""
        if self.model_params <= 1e9:
            return 24
        elif self.model_params <= 3e9:
            return 28
        elif self.model_params <= 8e9:
            return 32
        elif self.model_params <= 14e9:
            return 40
        elif self.model_params <= 34e9:
            return 60
        else:
            return 80

    # ── 各项显存计算 ────────────────────────────────────────

    def calc_model_memory(self, quantized: bool = False) -> float:
        """模型权重显存 (GB)。"""
        if quantized:
            bytes_per_param = 0.5  # 4-bit
        else:
            bytes_per_param = 2.0  # FP16

        # 基座模型
        base = self.model_params * bytes_per_param / 1e9

        # LoRA 参数：每层 2 个 adapter 矩阵 (A 和 B)
        lora_params_per_layer = 2 * self.lora_r * self.hidden_dim
        lora_total = self.num_layers * lora_params_per_layer * 2 / 1e9  # FP16

        return base + lora_total

    def calc_activation_memory(self, gradient_checkpoint: bool = False) -> float:
        """激活值显存 (GB)。"""
        # 简化公式：batch * seq * hidden * 2 bytes * layers
        act = (
            self.batch_size * self.seq_length * self.hidden_dim * 2 / 1e9
        ) * self.num_layers

        if gradient_checkpoint:
            act *= 0.3  # 减少约 70%

        return act

    def calc_optimizer_memory(self) -> float:
        """优化器状态显存 (GB)。AdamW 需要 2 倍模型参数的存储。"""
        return self.model_params * 2 * 2 / 1e9  # FP16 * 2 states

    def get_total(self, quantized: bool = True, gradient_checkpoint: bool = True) -> Dict:
        """汇总所有显存占用。"""
        model_mem = self.calc_model_memory(quantized)
        act_mem = self.calc_activation_memory(gradient_checkpoint)
        optim_mem = self.calc_optimizer_memory()

        # QLoRA 下优化器只需存 LoRA 参数
        if quantized:
            lora_params = self.num_layers * 2 * self.lora_r * self.hidden_dim
            optim_mem = lora_params * 2 * 2 / 1e9

        # 额外开销
        overhead = (model_mem + act_mem + optim_mem) * 0.15

        total = model_mem + act_mem + optim_mem + overhead

        return {
            "model_weight": round(model_mem, 2),
            "activation": round(act_mem, 2),
            "optimizer": round(optim_mem, 2),
            "overhead": round(overhead, 2),
            "total": round(total, 2),
            "quantized": quantized,
            "gradient_checkpoint": gradient_checkpoint,
        }

    def recommend(self, total_gb: float, quantized: bool) -> str:
        """根据总显存给出操作建议。"""
        if total_gb < 6:
            return "✅ 可在 6GB 显卡上运行（如 RTX 3060/4060 笔记本）"
        elif total_gb < 8:
            return "✅ 可在 8GB 显卡上运行（如 RTX 3070/4060Ti）"
        elif total_gb < 12:
            return "✅ 可在 12GB 显卡上运行（如 RTX 3080/4070）"
        elif total_gb < 16:
            return "✅ 可在 16GB 显卡上运行（如 RTX 4080）"
        elif total_gb < 24:
            return "⚠️ 需要 24GB 显存（如 RTX 3090/4090）。建议开启 QLoRA"
        elif total_gb < 48:
            return "⚠️ 需要 48GB 显存（如 A6000）。建议减小 batch size 或序列长度"
        else:
            return "❌ 显存不足！请：1) 降低 seq_length  2) 减小 batch_size  3) 换更小的模型"


# ── 便捷函数 ────────────────────────────────────────────────


def quick_calc(
    model_name: str = "qwen2-7b",
    seq_length: int = 2048,
    batch_size: int = 4,
    lora_r: int = 8,
    quantized: bool = True,
) -> Dict:
    """一行调用，返回显存估算结果。"""
    calc = MemoryCalculator(
        model_name=model_name,
        seq_length=seq_length,
        batch_size=batch_size,
        lora_r=lora_r,
    )
    mem = calc.get_total(quantized=quantized, gradient_checkpoint=True)
    mem["recommendation"] = calc.recommend(mem["total"], quantized)
    mem["model_info"] = {
        "params": f"{calc.model_params / 1e9:.1f}B",
        "hidden_dim": calc.hidden_dim,
        "layers": calc.num_layers,
    }
    return mem
