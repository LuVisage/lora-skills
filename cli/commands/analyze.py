"""lora-trainer analyze — 数据分析命令。"""

import json

import click
from rich.console import Console
from rich.panel import Panel

from scripts.analyzer import quick_analyze

console = Console()


@click.command()
@click.argument("data", type=click.Path(exists=True), metavar="DATA")
@click.option(
    "--output", "-o",
    type=click.Path(writable=True),
    default=None,
    help="将报告保存到文件",
)
@click.option(
    "--json", "-j", "json_output",
    is_flag=True,
    default=False,
    help="以 JSON 格式输出（供程序调用）",
)
def analyze(data: str, output: str | None, json_output: bool) -> None:
    """📊 分析 JSONL 微调数据质量。

    DATA 为 JSONL 格式的训练数据文件路径。

    \b
    示例:
      lora-trainer analyze ./data/train.jsonl
      lora-trainer analyze ./data/train.jsonl --output report.txt
      lora-trainer analyze ./data/train.jsonl --json | jq .quality
    """
    try:
        result = quick_analyze(data)
    except FileNotFoundError:
        console.print(f"[red]❌ 文件不存在: {data}[/red]")
        raise SystemExit(1)
    except json.JSONDecodeError as e:
        console.print(f"[red]❌ JSON 解析错误: {e}[/red]")
        raise SystemExit(1)
    except Exception as e:
        console.print(f"[red]❌ 分析失败: {e}[/red]")
        raise SystemExit(1)

    if json_output:
        # JSON 模式：输出去掉 report 文本的 dict
        result.pop("report", None)
        click.echo(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return

    # 富文本模式
    console.print(result["report"])

    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(result["report"])
        console.print(f"\n[green]✅ 报告已保存到: {output}[/green]")
