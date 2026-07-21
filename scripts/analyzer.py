"""
数据分析模块 — 读取 JSONL 数据，检查质量，生成统计报告。
"""

import json
from collections import Counter
from typing import Dict, List, Tuple
import numpy as np


class DataAnalyzer:
    """加载并分析 JSONL 格式的微调数据。"""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = self._load_data()

    # ── 数据加载 ───────────────────────────────────────────

    def _load_data(self) -> List[Dict]:
        """加载 JSONL 数据，每行一个 JSON 对象。"""
        data = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    # ── 长度分析 ───────────────────────────────────────────

    def analyze_length_distribution(self) -> Dict:
        """分析 instruction + output 的总长度分布。"""
        lengths = []
        for item in self.data:
            total_len = len(item.get("instruction", "")) + len(
                item.get("output", "")
            )
            lengths.append(total_len)

        if not lengths:
            return {
                "min": 0, "max": 0, "mean": 0, "median": 0,
                "percentile_80": 0, "percentile_95": 0, "total_samples": 0,
            }

        arr = np.array(lengths)
        return {
            "min": int(np.min(arr)),
            "max": int(np.max(arr)),
            "mean": float(round(np.mean(arr), 1)),
            "median": float(round(np.median(arr), 1)),
            "percentile_80": int(np.percentile(arr, 80)),
            "percentile_95": int(np.percentile(arr, 95)),
            "total_samples": len(arr),
        }

    # ── 质量检查 ───────────────────────────────────────────

    def check_data_quality(self) -> Dict:
        """扫描数据中的常见问题：空回复、重复、控制字符。"""
        issues = []

        if not self.data:
            return {
                "has_issues": True,
                "issues": ["数据集为空！"],
                "empty_output_ratio": 0,
                "duplicate_ratio": 0,
            }

        # 1. 空回复
        empty_count = sum(
            1
            for item in self.data
            if len(item.get("output", "").strip()) < 5
        )

        # 2. 重复数据
        texts = [
            item.get("instruction", "") + item.get("output", "")
            for item in self.data
        ]
        duplicate_count = len(texts) - len(set(texts))

        # 3. 控制字符
        control_chars = sum(
            1
            for item in self.data
            if any(
                c in item.get("instruction", "")
                for c in ["\t", "\r", "\x00"]
            )
        )

        # 4. 语言混合度（简单检测：同时包含中英文的样本比例）
        bilingual = 0
        for item in self.data:
            text = item.get("instruction", "") + item.get("output", "")
            has_cn = any("一" <= c <= "鿿" for c in text)
            has_en = any(c.isascii() and c.isalpha() for c in text)
            if has_cn and has_en:
                bilingual += 1

        if empty_count > 0:
            issues.append(
                f"发现 {empty_count} 条回复过短（< 5 字符），可能是生成失败"
            )
        if duplicate_count > 0:
            issues.append(
                f"发现 {duplicate_count} 条完全重复数据，建议去重"
            )
        if control_chars > 0:
            issues.append(
                f"发现 {control_chars} 条包含控制字符（\\t \\r \\x00），可能影响解析"
            )

        return {
            "has_issues": len(issues) > 0,
            "issues": issues,
            "empty_output_ratio": round(empty_count / len(self.data), 4),
            "duplicate_ratio": round(duplicate_count / len(self.data), 4),
            "bilingual_ratio": round(bilingual / len(self.data), 4),
        }

    # ── 字段检测 ───────────────────────────────────────────

    def detect_format(self) -> Dict:
        """自动检测数据格式（instruction/output 还是 messages 格式）。"""
        if not self.data:
            return {"format": "unknown", "fields": []}

        first = self.data[0]
        fields = list(first.keys())

        if "messages" in first and isinstance(first["messages"], list):
            return {"format": "chat", "fields": fields}
        elif "instruction" in first and "output" in first:
            return {"format": "instruction-output", "fields": fields}
        elif "conversations" in first:
            return {"format": "conversations", "fields": fields}
        else:
            return {"format": "unknown", "fields": fields}

    # ── 报告生成 ───────────────────────────────────────────

    def generate_report(self) -> str:
        """生成人类可读的数据分析报告。"""
        length_stats = self.analyze_length_distribution()
        quality_stats = self.check_data_quality()
        fmt = self.detect_format()

        lines = [
            "📊 数据报告",
            "=" * 40,
            f"总样本数: {length_stats['total_samples']}",
            f"数据格式: {fmt['format']}",
            f"字段: {', '.join(fmt['fields'])}",
            "",
            "📏 长度统计 (instruction + output):",
            f"  最短: {length_stats['min']} 字符",
            f"  最长: {length_stats['max']} 字符",
            f"  平均: {length_stats['mean']} 字符",
            f"  中位数: {length_stats['median']} 字符",
            f"  80% 分位: {length_stats['percentile_80']} 字符",
            f"  95% 分位: {length_stats['percentile_95']} 字符",
            "",
            "🔍 质量检查:",
            f"  空回复比例: {quality_stats['empty_output_ratio'] * 100:.1f}%",
            f"  重复率: {quality_stats['duplicate_ratio'] * 100:.1f}%",
            f"  中英混合: {quality_stats['bilingual_ratio'] * 100:.1f}%",
        ]

        if quality_stats["issues"]:
            lines.append("")
            lines.append("⚠️ 发现的问题:")
            for issue in quality_stats["issues"]:
                lines.append(f"  - {issue}")
        else:
            lines.append("")
            lines.append("✅ 数据质量良好！")

        return "\n".join(lines)


# ── 便捷函数（供 Skill 直接调用） ──────────────────────────


def quick_analyze(data_path: str) -> Dict:
    """一行调用，返回所有分析结果。"""
    analyzer = DataAnalyzer(data_path)
    return {
        "length": analyzer.analyze_length_distribution(),
        "quality": analyzer.check_data_quality(),
        "format": analyzer.detect_format(),
        "report": analyzer.generate_report(),
    }
