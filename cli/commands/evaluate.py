"""lora-trainer evaluate — 评估脚本生成命令。"""

import json

import click
from rich.console import Console
from rich.panel import Panel

from scripts.evaluator import Evaluator
from scripts.analyzer import DataAnalyzer

console = Console()


@click.command()
@click.argument("data", type=click.Path(exists=True), metavar="TEST_DATA")
@click.option(
    "--format", "-f", "data_format",
    type=click.Choice(["instruction-output", "messages", "conversations", "auto"]),
    default="auto",
    show_default=True,
    help="训练数据格式。auto = 自动检测",
)
@click.option(
    "--output", "-o",
    type=click.Path(writable=True),
    default="eval_script.py",
    show_default=True,
    help="评估脚本输出路径",
)
@click.option(
    "--json", "-j", "json_output",
    is_flag=True,
    default=False,
    help="以 JSON 格式输出元信息（不生成脚本）",
)
def evaluate(
    data: str,
    data_format: str,
    output: str,
    json_output: bool,
) -> None:
    """📏 生成 LoRA 模型评估脚本。

    TEST_DATA 为 JSONL 格式的测试数据文件。

    生成的脚本对比基座模型 vs LoRA 微调后模型的输出效果。

    \b
    示例:
      lora-trainer evaluate ./data/test.jsonl
      lora-trainer evaluate ./data/test.jsonl --format messages -o my_eval.py
      lora-trainer evaluate ./data/test.jsonl --json
    """
    # 自动检测格式
    if data_format == "auto":
        try:
            fmt_result = DataAnalyzer(data).detect_format()
            detected = fmt_result["format"]
            format_map = {
                "chat": "messages",
                "conversations": "conversations",
                "instruction-output": "instruction-output",
            }
            data_format = format_map.get(detected, "instruction-output")
            console.print(f"[dim]🔍 检测到数据格式: {data_format}[/dim]")
        except Exception:
            data_format = "instruction-output"

    try:
        evaluator = Evaluator(data, data_format=data_format)
    except Exception as e:
        console.print(f"[red]❌ 加载测试数据失败: {e}[/red]")
        raise SystemExit(1)

    if json_output:
        metrics = evaluator.generate_eval_metrics()
        click.echo(json.dumps(metrics, ensure_ascii=False, indent=2, default=str))
        return

    # 生成评估脚本
    try:
        script = evaluator.build_eval_script()
        with open(output, "w", encoding="utf-8") as f:
            f.write(script)
    except Exception as e:
        console.print(f"[red]❌ 生成评估脚本失败: {e}[/red]")
        raise SystemExit(1)

    # 输出说明
    console.print(f"\n[bold cyan]📏 评估脚本已生成[/bold cyan]")
    console.print(f"   文件: [green]{output}[/green]")
    console.print(f"   数据格式: [bold]{data_format}[/bold]")

    console.print(Panel(
        f"1. 编辑 [bold]{output}[/bold] 中的 [bold]BASE_MODEL[/bold] 和 [bold]LORA_PATH[/bold]\n"
        f"2. 安装依赖: [dim]pip install torch transformers peft tqdm[/dim]\n"
        f"3. 运行: [dim]python {output}[/dim]\n"
        f"4. 查看结果: [dim]eval_results.json[/dim]",
        title="📋 使用步骤",
        border_style="cyan",
    ))
