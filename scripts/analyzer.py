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

    # ── Token 估算 ─────────────────────────────────────────

    def _estimate_tokens(self, text: str) -> float:
        """估算文本的 token 数量（不加载 tokenizer 的近似方法）。

        中文/日/韩字符 ≈ 2.0 tokens/字（大多数 tokenizer 的行为）。
        英文/数字 ≈ 0.28 tokens/字符（~3.5 字符/token）。
        混合文本按字符分类加权估算。
        """
        if not text:
            return 0.0

        cjk_count = 0
        other_count = 0
        for ch in text:
            if '一' <= ch <= '鿿' or '぀' <= ch <= 'ヿ' or '가' <= ch <= '힯':
                cjk_count += 1
            else:
                other_count += 1

        return cjk_count * 2.0 + other_count * 0.28

    # ── 长度分析 ───────────────────────────────────────────

    def analyze_length_distribution(self) -> Dict:
        """分析 instruction + output 的字符长度和估算 token 长度分布。"""
        char_lengths = []
        token_lengths = []
        for item in self.data:
            instr = item.get("instruction", "")
            out = item.get("output", "")
            char_len = len(instr) + len(out)
            char_lengths.append(char_len)
            token_lengths.append(self._estimate_tokens(instr) + self._estimate_tokens(out))

        if not char_lengths:
            return {
                "min": 0, "max": 0, "mean": 0, "median": 0,
                "percentile_80": 0, "percentile_95": 0, "total_samples": 0,
                "estimated_p95_tokens": 0,
            }

        arr_char = np.array(char_lengths)
        arr_token = np.array(token_lengths)
        return {
            "min": int(np.min(arr_char)),
            "max": int(np.max(arr_char)),
            "mean": float(round(np.mean(arr_char), 1)),
            "median": float(round(np.median(arr_char), 1)),
            "percentile_80": int(np.percentile(arr_char, 80)),
            "percentile_95": int(np.percentile(arr_char, 95)),
            "total_samples": len(arr_char),
            # token 估算值（用于 max_seq_length 推荐）
            "estimated_p95_tokens": int(np.percentile(arr_token, 95)),
            "estimated_median_tokens": float(round(np.median(arr_token), 1)),
            "note": "token 数为基于字符类型的近似估算，精确值需用 tokenizer 计算",
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
            "📏 字符长度 (instruction + output):",
            f"  最短: {length_stats['min']} | 最长: {length_stats['max']}",
            f"  平均: {length_stats['mean']} | 中位数: {length_stats['median']}",
            f"  P80: {length_stats['percentile_80']} | P95: {length_stats['percentile_95']}",
            "",
            "📐 估算 Token 长度 (基于字符类型近似):",
            f"  中位数: {length_stats['estimated_median_tokens']} tokens",
            f"  P95: {length_stats['estimated_p95_tokens']} tokens",
            f"  建议 max_seq_length ≥ {length_stats['estimated_p95_tokens']} (P95 × 1.0)",
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
