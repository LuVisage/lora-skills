"""lora-trainer cook — 一键生成训练脚本。"""

import json
import os

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Confirm, Prompt

from scripts.analyzer import quick_analyze
from scripts.lora_advisor import quick_recommend
from scripts.memory_calc import quick_calc
from scripts.script_builder import ScriptBuilder

console = Console()


@click.command()
@click.option(
    "--data", "-d",
    type=click.Path(exists=True),
    default=None,
    help="JSONL 训练数据路径",
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
    default=None,
    help="GPU 显存 (GB)，不指定则尝试自动检测",
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(writable=True),
    default="./output",
    show_default=True,
    help="脚本输出目录",
)
@click.option(
    "--format", "-f", "data_format_opt",
    type=click.Choice(["instruction-output", "messages", "conversations", "auto"]),
    default="auto",
    show_default=True,
    help="训练数据格式。auto = 自动检测",
)
@click.option(
    "--interactive", "-i",
    is_flag=True,
    default=False,
    help="交互模式：逐项确认参数",
)
# 高级覆盖参数
@click.option("--rank", type=int, default=None, help="覆盖推荐的 rank 值")
@click.option("--alpha", type=int, default=None, help="覆盖推荐的 alpha 值")
@click.option("--lr", type=float, default=None, help="覆盖推荐的学习率")
@click.option("--epochs", type=int, default=None, help="覆盖推荐的训练轮数")
@click.option("--batch-size", type=int, default=None, help="覆盖推荐的 batch size")
@click.option(
    "--json", "-j", "json_output",
    is_flag=True,
    default=False,
    help="以 JSON 格式输出配置（不生成脚本）",
)
@click.option(
    "--max-seq-length",
    type=int,
    default=None,
    help="最大序列长度。不指定则根据数据 P95 自动设置",
)
def cook(
    data: str | None,
    model: str,
    task: str,
    gpu: float | None,
    output_dir: str,
    data_format_opt: str,
    interactive: bool,
    rank: int | None,
    alpha: int | None,
    lr: float | None,
    epochs: int | None,
    batch_size: int | None,
    json_output: bool,
    max_seq_length: int | None,
) -> None:
    """🚀 一键生成 LoRA 微调训练脚本。

    自动完成：数据分析 → 参数推荐 → 脚本生成。

    \b
    示例:
      lora-trainer cook --data ./data/train.jsonl
      lora-trainer cook -d ./data/train.jsonl -m qwen2-7b -t chat -g 24
      lora-trainer cook -d ./data/train.jsonl -t code --rank 16
      lora-trainer cook -d ./data/train.jsonl --interactive
      lora-trainer cook -d ./data/train.jsonl --json
    """
    # ── Step 1: 收集信息 ──────────────────────────────
    if not data:
        console.print("[red]❌ 请指定 --data 参数[/red]")
        console.print("[dim]用法: lora-trainer cook --data ./data/train.jsonl[/dim]")
        raise SystemExit(1)

    # ── Step 2: 分析数据 ──────────────────────────────
    console.print("[dim]📊 分析数据...[/dim]")
    try:
        analysis = quick_analyze(data)
    except Exception as e:
        console.print(f"[red]❌ 数据分析失败: {e}[/red]")
        raise SystemExit(1)

    num_samples = analysis["length"]["total_samples"]
    p95_tokens = analysis["length"].get("estimated_p95_tokens", 2048)
    detected_format = analysis["format"]["format"]

    if data_format_opt == "auto":
        format_map = {
            "chat": "messages",
            "conversations": "conversations",
            "instruction-output": "instruction-output",
        }
        data_format = format_map.get(detected_format, "instruction-output")
    else:
        data_format = data_format_opt

    # 序列长度
    if max_seq_length is None:
        # 取 P95 token 数向上取整到 512 的倍数
        seq_len = max(512, ((p95_tokens // 512) + 1) * 512)
        if seq_len > 8192:
            seq_len = min(seq_len, 16384)  # 允许长序列但给出警告
    else:
        seq_len = max_seq_length

    # ── Step 3: GPU 检测 ──────────────────────────────
    if gpu is None:
        import subprocess
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                gpu = float(result.stdout.strip().split()[0]) / 1024  # MiB → GB
                console.print(f"[dim]🔍 检测到 GPU 显存: {gpu:.0f} GB[/dim]")
            else:
                gpu = 24
        except Exception:
            gpu = 24
            console.print("[dim]⚠️ 未检测到 GPU，默认 24GB[/dim]")

    # ── Step 4: 推荐参数 ──────────────────────────────
    console.print("[dim]⚙️ 推荐参数...[/dim]")
    config = quick_recommend(
        num_samples=num_samples,
        model_size=model,
        task_type=task,
        gpu_memory_gb=gpu,
    )

    if rank is not None:
        config["rank"]["value"] = rank
        config["rank"]["reason"] += " [用户覆盖]"
    if alpha is not None:
        config["alpha"]["value"] = alpha
        config["alpha"]["reason"] += " [用户覆盖]"
    if lr is not None:
        config["learning_rate"]["value"] = lr
        config["learning_rate"]["reason"] += " [用户覆盖]"
    if epochs is not None:
        config["epochs"]["value"] = epochs
        config["epochs"]["reason"] += " [用户覆盖]"
    if batch_size is not None:
        config["batch_size"]["value"] = batch_size
        config["batch_size"]["reason"] += " [用户覆盖]"

    # ── Step 5: 显存估算 ──────────────────────────────
    num_modules = len(config["target_modules"]["value"])
    try:
        mem_result = quick_calc(
            model_name=model,
            seq_length=seq_len,
            batch_size=config["batch_size"]["value"],
            lora_r=config["rank"]["value"],
            quantized=True,
            num_modules=num_modules,
        )
    except Exception:
        mem_result = None

    # ── JSON 输出模式 ────────────────────────────────
    if json_output:
        output = {
            "config": config,
            "data_summary": {
                "num_samples": num_samples,
                "p95_tokens": p95_tokens,
                "format": data_format,
                "quality": analysis["quality"],
            },
            "max_seq_length": seq_len,
            "memory": mem_result,
        }
        click.echo(json.dumps(output, ensure_ascii=False, indent=2, default=str))
        return

    # ── 展示配置 ──────────────────────────────────────
    task_labels = {"chat": "💬 通用对话", "code": "💻 代码", "math": "🧮 数学推理",
                   "roleplay": "🎭 角色扮演", "cpt": "📚 继续预训练 (CPT)"}

    console.print(f"\n[bold cyan]📊 数据摘要[/bold cyan]")
    console.print(f"   样本: [bold]{num_samples:,}[/bold] 条  |  P95 Token: {p95_tokens}  |  格式: {data_format}")
    quality = analysis["quality"]
    if quality.get("issues"):
        for issue in quality["issues"]:
            console.print(f"   [yellow]⚠️ {issue}[/yellow]")
    else:
        console.print(f"   [green]✅ 数据质量良好[/green]")

    console.print(f"\n[bold cyan]⚙️ 推荐配置[/bold cyan]")
    console.print(f"   任务: {task_labels.get(task, task)}  |  模型: {model}  |  Max Seq: {seq_len}  |  GPU: {gpu:.0f} GB\n")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("参数", style="bold", width=22)
    table.add_column("推荐值", style="green", width=14)
    table.add_column("理由", style="dim", width=50)

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
    console.print(table)

    effective_bs = config["batch_size"]["value"] * config["gradient_accumulation"]["value"]
    console.print(f"   [dim]有效 Batch Size: {effective_bs}  |  Scheduler: {'constant_with_warmup' if task == 'cpt' else ('linear' if task == 'roleplay' else 'cosine')}[/dim]")

    # 显存
    if mem_result:
        console.print(f"\n[bold cyan]💾 预估显存[/bold cyan]")
        console.print(f"   模型权重: {mem_result['model_weight']:.2f} GB  |  激活值: {mem_result['activation']:.2f} GB  |  优化器: {mem_result['optimizer']:.2f} GB")
        console.print(f"   [bold]总计: {mem_result['total']:.2f} GB[/bold] (GPU: {gpu:.0f} GB, 剩余 {gpu - mem_result['total']:.1f} GB)")

    # ── 交互确认 ──────────────────────────────────────
    if interactive:
        console.print()
        if not Confirm.ask("[bold]确认以上配置，开始生成脚本？[/bold]", default=True):
            console.print("[yellow]已取消[/yellow]")
            return
        # 交互覆盖
        if Confirm.ask("需要修改 rank？", default=False):
            config["rank"]["value"] = int(Prompt.ask("rank", default=str(config["rank"]["value"])))
        if Confirm.ask("需要修改 learning rate？", default=False):
            config["learning_rate"]["value"] = float(Prompt.ask("learning_rate", default=f"{config['learning_rate']['value']:.2e}"))
        if Confirm.ask("需要修改 epochs？", default=False):
            config["epochs"]["value"] = int(Prompt.ask("epochs", default=str(config["epochs"]["value"])))

    # ── Step 6: 生成脚本 ──────────────────────────────
    console.print(f"\n[dim]🔧 生成脚本到 {output_dir}...[/dim]")
    try:
        builder = ScriptBuilder(
            config=config,
            output_dir=output_dir,
            data_format=data_format,
            max_seq_length=seq_len,
            task_type=task,
        )
        paths = builder.build_all()
    except Exception as e:
        console.print(f"[red]❌ 脚本生成失败: {e}[/red]")
        raise SystemExit(1)

    console.print(f"\n[bold green]✅ 脚本生成完成！[/bold green]")
    for key, path in paths.items():
        labels = {"train_script": "训练脚本", "inference_script": "推理脚本", "config": "配置文件"}
        console.print(f"   📄 {labels.get(key, key)}: [green]{path}[/green]")

    console.print(Panel(
        f"1. 编辑训练脚本中的 [bold]MODEL_NAME[/bold] 和 [bold]DATA_PATH[/bold]\n"
        f"2. 安装依赖: [dim]pip install lora-trainer[train][/dim]\n"
        f"3. 运行训练: [dim]python {os.path.basename(paths['train_script'])}[/dim]\n"
        f"4. 训练完成后: [dim]lora-trainer evaluate <test.jsonl>[/dim]",
        title="📋 下一步",
        border_style="green",
    ))
