# LoRA Trainer -- Trigger Catalog

## Quick Start (Copy & Paste These)

| I want to... | Say this |
|-------------|---------|
| Quick training | `/lora:cook ./data/train.jsonl qwen2-7b chat --auto` |
| Full analysis | `/lora:analyze ./data/train.jsonl qwen2-7b chat` |
| Data quality check | `/lora:check-data ./data/train.jsonl` |
| Check GPU memory | `python memory_check.py qwen2-7b --seq-length 2048` |
| Validate data file | `python validate_data.py ./data/train.jsonl` |
| Environment setup | `/lora:setup` |
| Debug training | `/lora:debug ./output/logs` |

## Natural Language (Chinese) -- Just Describe What You Want

### Data Checking
- `帮我看看 ./data/train.jsonl 的数据质量`
- `检查一下这个数据能不能用来微调`
- `扫一下数据有没有空回复和重复`
- `验证一下我的训练数据格式对不对`
- `数据质量打分`

### Parameter Recommendation
- `5000条数据，7b模型，推荐什么参数？`
- `我的数据有1000条，用什么rank合适？`
- `显存只有8GB，能微调llama3-8b吗？`
- `代码任务用什么target_modules比较好？`
- `帮我算一下 qwen2-7b 要多少显存`
- `我的 RTX 4070 12GB 能跑 llama3-8b 吗？`

### Start Fine-tuning
- `帮我微调 ./data/train.jsonl，用 qwen2-7b`
- `一键炼丹 ./data/chat.jsonl，聊天任务`
- `我要继续预训练，数据在 ./data/corpus.jsonl`
- `帮我把这个指令数据集训成 LoRA`
- `帮我准备一个 QLoRA 微调方案`

### Encountering Problems
- `训练OOM了怎么办？`
- `loss不收敛怎么排查？`
- `帮我看看训练日志 ./output/logs`
- `训练完了效果不好，怎么调？`
- `这个报错是什么意思：CUDA out of memory`
- `训练到一半中断了，怎么恢复？`

### Environment
- `帮我检查一下微调环境`
- `国内下载模型太慢怎么办？`
- `我的显卡能不能微调7B模型？`
- `环境配置好了没有，帮我检查一下`

## Natural Language (English)

### Data Checking
- `Analyze my training data at ./data/train.jsonl`
- `Check if my data is good enough for fine-tuning`
- `Scan for empty responses and duplicates in my dataset`

### Parameter Recommendation
- `Recommend parameters for 5000 samples with a 7B model`
- `What LoRA rank should I use for 1000 samples?`
- `Can I fine-tune llama3-8b with 8GB VRAM?`
- `What target modules for code fine-tuning?`

### Start Fine-tuning
- `Help me fine-tune ./data/chat.jsonl with qwen2-7b`
- `Fine-tune my instruction dataset with LoRA`
- `I want to do CPT on ./data/corpus.jsonl`
- `Prepare a QLoRA training setup for me`

### Troubleshooting
- `Training OOM'd, what should I do?`
- `Loss is not converging, how to debug?`
- `My model is overfitting, what hyperparameters should I change?`
- `Training crashed midway, can I resume?`
- `What does this error mean: CUDA out of memory`

## Slash Commands

| Command | Usage | Description |
|---------|-------|-------------|
| `/lora:analyze` | `<data> [model] [task]` | Full analysis -> VRAM -> params -> script |
| `/lora:cook` | `<data> [model] [task]` | Quick training, `--auto` to auto-start |
| `/lora:check-data` | `<data>` | Data quality audit only |
| `/lora:debug` | `<log\|error>` | Training failure diagnosis |
| `/lora:setup` | (no args) | Environment check + dependency install |
