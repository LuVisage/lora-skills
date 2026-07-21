"""lora-trainer memory — 显存估算命令。"""

import json

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from scripts.memory_calc import quick_calc, MODEL_DB

console = Console()


@click.command()
@click.argument("model", metavar="MODEL")
@click.option(
    "--seq-length", "-s",
    type=int,
    default=2048,
    show_default=True,
    help="最大序列长度",
)
@click.option(
    "--batch-size", "-b",
    type=int,
    default=4,
    show_default=True,
    help="每卡 Batch Size",
)
@click.option(
    "--rank", "-r",
    type=int,
    default=8,
    show_default=True,
    help="LoRA Rank",
)
@click.option(
    "--modules", "-m",
    type=int,
    default=2,
    show_default=True,
    help="LoRA 目标模块数量 (chat=2, code=4, math=5, roleplay=1, cpt=7)",
)
@click.option(
    "--no-quantize",
    is_flag=True,
    default=False,
    help="使用 FP16 LoRA（关闭 4-bit 量化）",
)
@click.option(
    "--json", "-j", "json_output",
    is_flag=True,
    default=False,
    help="以 JSON 格式输出（供程序调用）",
)
def memory(
    model: str,
    seq_length: int,
    batch_size: int,
    rank: int,
    modules: int,
    no_quantize: bool,
    json_output: bool,
) -> None:
    """💾 估算 LoRA 微调所需 GPU 显存。

    MODEL 为模型名称，如 qwen2-7b、llama3-8b、mistral-7b 等。

    \b
    示例:
      lora-trainer memory qwen2-7b
      lora-trainer memory llama3-8b --seq-length 4096 --batch-size 8
      lora-trainer memory mistral-7b --rank 16 --modules 4
      lora-trainer memory deepseek-v2 --no-quantize
    """
    quantized = not no_quantize
    mode_label = "QLoRA 4-bit" if quantized else "LoRA FP16"

    # 检查模型是否在数据库里
    model_lower = model.lower().replace("-", "").replace("_", "")
    known = any(
        key.replace("-", "") in model_lower or model_lower in key.replace("-", "")
        for key in MODEL_DB
    )

    try:
        result = quick_calc(
            model_name=model,
            seq_length=seq_length,
            batch_size=batch_size,
            lora_r=rank,
            quantized=quantized,
            num_modules=modules,
        )
    except Exception as e:
        console.print(f"[red]❌ 显存计算失败: {e}[/red]")
        raise SystemExit(1)

    if json_output:
        click.echo(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return

    # 富文本输出
    info = result["model_info"]
    model_desc = f"{info['params']} 参数"
    if info.get("is_moe"):
        model_desc += f" (MoE, 激活 {info['params']} / 总 {info.get('total_params', '?')})"

    console.print(f"\n[bold cyan]💾 显存估算[/bold cyan]")
    console.print(f"   模型: {model} ({model_desc})  |  模式: {mode_label}")
    console.print(f"   序列长度: {seq_length}  |  Batch Size: {batch_size}  |  Rank: {rank}\n")

    # 细分表格
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("项目", style="bold", width=18)
    table.add_column("显存 (GB)", style="green", width=14)

    table.add_row("模型权重 + LoRA", f"{result['model_weight']:.2f}")
    table.add_row("激活值", f"{result['activation']:.2f}")
    table.add_row("优化器状态", f"{result['optimizer']:.2f}")
    table.add_row("系统开销 (15%)", f"{result['overhead']:.2f}")
    table.add_section()
    table.add_row("[bold]总计[/bold]", f"[bold green]{result['total']:.2f}[/bold green]")

    console.print(table)

    # 判定
    total_gb = result["total"]
    if total_gb < 6:
        verdict = "✅ 可在 6GB 显卡运行 (如 RTX 3060/4060 笔记本)"
        style = "green"
    elif total_gb < 8:
        verdict = "✅ 可在 8GB 显卡运行 (如 RTX 3070/4060Ti)"
        style = "green"
    elif total_gb < 12:
        verdict = "✅ 可在 12GB 显卡运行 (如 RTX 3080/4070)"
        style = "green"
    elif total_gb < 16:
        verdict = "✅ 可在 16GB 显卡运行 (如 RTX 4080)"
        style = "green"
    elif total_gb < 24:
        verdict = "✅ 可在 24GB 显卡运行 (如 RTX 3090/4090)"
        style = "green"
    elif total_gb < 48:
        verdict = "⚠️ 需要 48GB 显存 (如 A6000)，建议减小 batch size 或序列长度"
        style = "yellow"
    else:
        verdict = "❌ 显存需求较高，请降低 seq_length、减小 batch_size、或换更小的模型"
        style = "red"

    console.print(Panel(verdict, border_style=style))

    if not known:
        console.print(
            "\n[yellow]⚠️ 模型未在数据库中，使用了启发式估算（基于参数量推算架构参数），结果仅供参考。[/yellow]"
        )

    console.print(f"\n[dim]💡 提示: 使用 --help 查看所有选项，包括 --seq-length / --batch-size / --rank / --modules[/dim]")
