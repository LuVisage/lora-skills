"""
LoRA 参数推荐模块 — 根据数据量、模型大小、任务类型推荐最优超参数。
"""

from typing import Dict, List, Tuple


class LoRAAdvisor:
    """智能推荐 LoRA 的 r, alpha, target_modules, dropout, learning_rate。"""

    def __init__(
        self,
        num_samples: int,
        model_size: str = "7b",
        task_type: str = "chat",
        avg_seq_length: int = 512,
    ):
        self.num_samples = num_samples
        self.model_size = model_size.lower()
        self.task_type = task_type.lower()
        self.avg_seq_length = avg_seq_length

    # ── 各参数推荐 ──────────────────────────────────────────

    def recommend_rank(self) -> Tuple[int, str]:
        """推荐 LoRA rank (r)。"""
        # 基础推荐
        if self.num_samples < 500:
            r = 4
            reason = "极小数据集（<500），使用 r=4 防止严重过拟合"
        elif self.num_samples < 1000:
            r = 4
            reason = "小数据集（<1k），使用 r=4 防止过拟合"
        elif self.num_samples < 5000:
            r = 8
            reason = "中等数据集（1k-5k），r=8 平衡学习能力与泛化"
        elif self.num_samples < 20000:
            r = 16
            reason = "大数据集（5k-20k），r=16 充分学习新知识"
        else:
            r = 32
            reason = "超大数据集（>20k），r=32 捕获复杂模式"

        # 任务类型调整（受数据量上限约束）
        # 数据量不足时，任务调整不能超越数据量区间的上限
        data_ceiling = 4 if self.num_samples < 500 else (8 if self.num_samples < 1000 else 999)
        if self.task_type == "code":
            target_r = 16
            if target_r > data_ceiling:
                r = data_ceiling
                reason += f"；代码任务建议 r=16，但数据仅 {self.num_samples} 条，降为 r={data_ceiling} 防止过拟合"
            elif r < target_r:
                r = target_r
                reason += "；代码任务需要更高 rank 学习语法结构"
        elif self.task_type == "math":
            target_r = 16
            if target_r > data_ceiling:
                r = data_ceiling
                reason += f"；数学推理建议 r=16，但数据仅 {self.num_samples} 条，降为 r={data_ceiling} 防止过拟合"
            elif r < target_r:
                r = target_r
                reason += "；数学推理需要更高 rank 学习推理链"
        elif self.task_type == "roleplay":
            if r > 8:
                r = 8
            reason += "；角色扮演用较小 rank 保留更多基础语言能力"

        return r, reason

    def recommend_alpha(self, r: int) -> Tuple[int, str]:
        """推荐 alpha 值。通常 alpha = 2*r，但任务类型会影响。"""
        if self.task_type == "roleplay":
            alpha = r * 4
            reason = f"角色扮演：alpha={alpha}（4×rank），增强风格学习强度"
        elif self.task_type == "code":
            alpha = r * 2
            reason = f"代码任务：alpha={alpha}（2×rank），标准比例"
        elif self.task_type == "math":
            alpha = r
            reason = f"数学推理：alpha={alpha}（1×rank），保守更新，保留推理链"
        else:
            alpha = r * 2
            reason = f"通用对话：alpha={alpha}（2×rank），标准比例"

        return alpha, reason

    def recommend_target_modules(self) -> Tuple[List[str], str]:
        """推荐要训练的目标模块。"""
        task = self.task_type
        size = self.model_size

        if task == "code":
            modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            reason = "代码任务：训练 Q/K/V/O 全部注意力层"
        elif task == "math":
            modules = ["q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"]
            reason = "数学推理：注意力层 + 前馈网络层（需要更强的表示能力）"
        elif task == "roleplay":
            modules = ["v_proj"]
            reason = "角色扮演：仅训练 V 投影层，最大限度保留基础语言能力"
        else:
            # chat 通用
            modules = ["q_proj", "v_proj"]
            reason = "通用对话：标准 Q/V 配置"

        # 大模型额外加 o_proj
        if size in ["13b", "70b", "72b", "67b", "34b"] and "o_proj" not in modules:
            modules.append("o_proj")
            reason += "；大模型增加输出投影层"

        return modules, reason

    def recommend_dropout(self) -> Tuple[float, str]:
        """推荐 dropout 值。"""
        if self.num_samples < 500:
            return 0.15, "极小数据集，高 dropout (0.15) 强力防过拟合"
        elif self.num_samples < 1000:
            return 0.1, "小数据集，较高 dropout (0.1) 防过拟合"
        elif self.num_samples < 10000:
            return 0.05, "中等数据集，中等 dropout (0.05)"
        else:
            return 0.0, "大数据集，dropout=0，充分利用数据"

    def recommend_lr(self, r: int) -> Tuple[float, str]:
        """推荐学习率。"""
        # LoRA 典型学习率范围：1e-4 ~ 5e-4
        if r <= 4:
            lr = 1e-4
            reason = f"小 rank ({r})，较低学习率 (1e-4) 防止震荡"
        elif r <= 8:
            lr = 2e-4
            reason = f"标准 rank ({r})，标准学习率 (2e-4)"
        elif r <= 16:
            lr = 3e-4
            reason = f"较大 rank ({r})，较高学习率 (3e-4)"
        else:
            lr = 5e-4
            reason = f"大 rank ({r})，高学习率 (5e-4) 加速收敛"

        # 任务微调
        if self.task_type == "roleplay":
            lr *= 1.5
            reason += "；角色扮演提高学习率增强风格学习"
        elif self.task_type == "math":
            lr *= 0.75
            reason += "；数学推理降低学习率保护推理链"

        # 数据量调整
        if self.num_samples > 20000:
            lr *= 1.2
            reason += "；大数据集适当提高"

        return round(lr, 6), reason

    def recommend_epochs(self) -> Tuple[int, str]:
        """推荐训练轮数。"""
        if self.num_samples < 500:
            return 5, "极小数据集，5 轮充分学习"
        elif self.num_samples < 1000:
            return 3, "小数据集，3 轮"
        elif self.num_samples < 5000:
            return 3, "中等数据集，3 轮"
        elif self.num_samples < 20000:
            return 2, "大数据集，2 轮即可"
        else:
            return 1, "超大数据集，1 轮避免过拟合"

    def recommend_batch_size(self, gpu_memory_gb: float) -> Tuple[int, str]:
        """根据显存推荐 batch size。"""
        if gpu_memory_gb < 8:
            return 1, "显存较小，batch_size=1"
        elif gpu_memory_gb < 16:
            return 2, "中等显存，batch_size=2"
        elif gpu_memory_gb < 24:
            return 4, "显存充裕，batch_size=4"
        elif gpu_memory_gb < 48:
            return 8, "大显存，batch_size=8"
        else:
            return 16, "超大显存，batch_size=16"

    # ── 汇总 ────────────────────────────────────────────────

    def generate_full_config(self, gpu_memory_gb: float = 24) -> Dict:
        """生成完整的 LoRA 配置推荐。"""
        r, r_reason = self.recommend_rank()
        alpha, alpha_reason = self.recommend_alpha(r)
        modules, modules_reason = self.recommend_target_modules()
        dropout, dropout_reason = self.recommend_dropout()
        lr, lr_reason = self.recommend_lr(r)
        epochs, epochs_reason = self.recommend_epochs()
        bs, bs_reason = self.recommend_batch_size(gpu_memory_gb)

        return {
            "rank": {"value": r, "reason": r_reason},
            "alpha": {"value": alpha, "reason": alpha_reason},
            "target_modules": {"value": modules, "reason": modules_reason},
            "dropout": {"value": dropout, "reason": dropout_reason},
            "learning_rate": {"value": lr, "reason": lr_reason},
            "epochs": {"value": epochs, "reason": epochs_reason},
            "batch_size": {"value": bs, "reason": bs_reason},
            "gradient_accumulation": {
                "value": max(1, 16 // bs),
                "reason": f"梯度累积使有效 batch 达到 16",
            },
            "task_type": self.task_type,
            "model_size": self.model_size,
            "num_samples": self.num_samples,
        }


# ── 便捷函数 ────────────────────────────────────────────────


def quick_recommend(
    num_samples: int,
    model_size: str = "7b",
    task_type: str = "chat",
    gpu_memory_gb: float = 24,
) -> Dict:
    """一行调用，返回完整 LoRA 配置推荐。"""
    advisor = LoRAAdvisor(num_samples, model_size, task_type)
    return advisor.generate_full_config(gpu_memory_gb)
