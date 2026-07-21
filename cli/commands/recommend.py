"""lora-trainer recommend — 参数推荐命令。"""

import json

import click
from rich.console import Console
from rich.table import Table

from scripts.lora_advisor import quick_recommend

console = Console()


@click.command()
@click.option(
    "--samples", "-n",
    type=int,
    required=True,
    help="训练样本数量",
)
@click.option(
    "--model", "-m",
    default="7b",
    show_default=True,
    help="模型大小 (如 7b, 13b, 70b)",
)
@click.option(
    "--task", "-t",
    type=click.Choice(["chat", "code", "math", "roleplay", "cpt"]),
    default="chat",
    show_default=True,
    help="任务类型",
)
@click.option(
    "--gpu", "-g",
    type=float,
    default=24,
    show_default=True,
    help="GPU 显存 (GB)",
)
@click.option(
    "--json", "-j", "json_output",
    is_flag=True,
    default=False,
    help="以 JSON 格式输出（供程序调用）",
)
def recommend(
    samples: int,
    model: str,
    task: str,
    gpu: float,
    json_output: bool,
) -> None:
    """⚙️ 根据数据量和任务类型推荐 LoRA 超参数。

    \b
    示例:
      lora-trainer recommend --samples 2340 --model 7b --task chat
      lora-trainer recommend -n 5000 -m 13b -t code -g 16
      lora-trainer recommend -n 10000 --json
    """
    try:
        config = quick_recommend(
            num_samples=samples,
            model_size=model,
            task_type=task,
            gpu_memory_gb=gpu,
        )
    except Exception as e:
        console.print(f"[red]❌ 推荐失败: {e}[/red]")
        raise SystemExit(1)

    if json_output:
        click.echo(json.dumps(config, ensure_ascii=False, indent=2, default=str))
        return

    # 富文本表格输出
    task_labels = {"chat": "💬 通用对话", "code": "💻 代码", "math": "🧮 数学推理",
                   "roleplay": "🎭 角色扮演", "cpt": "📚 继续预训练 (CPT)"}
    task_label = task_labels.get(task, task)

    console.print(f"\n[bold cyan]⚙️ LoRA 参数推荐[/bold cyan]")
    console.print(f"   任务: {task_label}  |  模型: {model}  |  样本: {samples:,}  |  GPU: {gpu} GB\n")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("参数", style="bold", width=22)
    table.add_column("推荐值", style="green", width=12)
    table.add_column("理由", style="dim", width=52)

    rows = [
        ("rank (r)", str(config["rank"]["value"]), config["rank"]["reason"]),
        ("alpha", str(config["alpha"]["value"]), config["alpha"]["reason"]),
        ("target_modules", ", ".join(config["target_modules"]["value"]), config["target_modules"]["reason"]),
        ("dropout", str(config["dropout"]["value"]), config["dropout"]["reason"]),
        ("learning_rate", f"{config['learning_rate']['value']:.2e}", config["learning_rate"]["reason"]),
        ("epochs", str(config["epochs"]["value"]), config["epochs"]["reason"]),
        ("batch_size", str(config["batch_size"]["value"]), config["batch_size"]["reason"]),
        ("gradient_accumulation", str(config["gradient_accumulation"]["value"]), config["gradient_accumulation"]["reason"]),
    ]

    for row in rows:
        table.add_row(*row)

    effective_bs = config["batch_size"]["value"] * config["gradient_accumulation"]["value"]
    table.add_section()
    table.add_row("[bold]有效 Batch Size[/bold]", f"[bold green]{effective_bs}[/bold green]", "per_device_bs × grad_accum")

    console.print(table)
