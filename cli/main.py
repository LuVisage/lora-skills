"""
lora-trainer CLI — LoRA/QLoRA 微调命令行工具。

从数据分析到训练脚本生成，一行命令搞定。
"""

import click

from cli.commands import analyze, recommend, memory, cook, evaluate


@click.group()
@click.version_option(
    package_name="lora-trainer",
    message="lora-trainer v%(version)s 🦾",
)
def main():
    """🦾 LoRA / QLoRA 微调命令行工具

    从数据分析 → 显存评估 → 参数推荐 → 训练脚本生成，全流程覆盖。

    \b
    快速开始:
      lora-trainer analyze data.jsonl           # 分析数据
      lora-trainer memory qwen2-7b              # 估算显存
      lora-trainer recommend --samples 2340     # 推荐参数
      lora-trainer cook --data data.jsonl       # 一键生成训练脚本
      lora-trainer evaluate test.jsonl          # 生成评估脚本
    """
    pass


main.add_command(analyze.analyze)
main.add_command(recommend.recommend)
main.add_command(memory.memory)
main.add_command(cook.cook)
main.add_command(evaluate.evaluate)
