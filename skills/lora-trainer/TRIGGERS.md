# LoRA Trainer — 触发方式目录

不知道怎么说？从这里选一句。

## 斜杠命令（最直接）

```
/lora:analyze <数据路径> [模型] [任务]    → 完整分析
/lora:cook <数据路径> [模型] [任务]       → 快速炼丹
/lora:check-data <数据路径>               → 只看数据质量
/lora:debug <日志路径|报错信息>           → 训练故障诊断
/lora:setup                               → 环境检测 + 安装依赖
```

## 自然语言（中文）

### 数据检查
- "帮我看看 ./data/train.jsonl 的数据质量"
- "检查一下这个数据能不能用来微调"
- "扫一下数据有没有空回复和重复"

### 参数推荐
- "5000 条数据，7b 模型，推荐什么参数？"
- "我的数据有 1000 条，用什么 rank 合适？"
- "显存只有 8GB，能微调 llama3-8b 吗？"

### 开始微调
- "帮我微调 ./data/train.jsonl，用 qwen2-7b"
- "一键炼丹 ./data/chat.jsonl，聊天任务"
- "我要继续预训练，数据在 ./data/corpus.jsonl"

### 遇到问题
- "训练 OOM 了怎么办？"
- "loss 不收敛怎么排查？"
- "帮我看看训练日志 ./output/logs"
- "训练完了效果不好，怎么调？"
- "这个报错是什么意思：CUDA out of memory"

### 环境相关
- "帮我检查一下微调环境"
- "国内下载模型太慢怎么办？"
- "我的显卡能不能微调 7B 模型？"

## Natural Language (English)

- "Analyze my training data at ./data/train.jsonl"
- "Recommend parameters for 5000 samples with a 7B model"
- "Can I fine-tune llama3-8b with 8GB VRAM?"
- "Help me fine-tune ./data/chat.jsonl with qwen2-7b"
- "Training OOM'd, what should I do?"
- "Loss isn't converging, help me debug"
- "Check my environment for LoRA training"
- "How do I download models faster from China?"

## 你不需要记住这些

直接用中文描述你想做什么。我会引导你。
