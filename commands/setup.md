---
name: lora:setup
version: 2.5.0
description: >-
  One-command environment check and dependency installation for LoRA/QLoRA
  fine-tuning. Detects Python, CUDA, GPU, PyTorch, and all required libraries.
  Installs missing deps automatically. Use before starting any training.
  Trigger: /lora:setup, "检查环境", "setup environment", "环境检测".
argument-hint: none
disable-model-invocation: true
allowed-tools: Read, Bash(python *), Bash(pip *), Bash(nvidia-smi *)
---

# /lora:setup — 环境检测 + 依赖安装

一键检查微调环境是否就绪，缺什么装什么。

## 执行

### Step 1: 运行环境检测脚本

```bash
python ${CLAUDE_PLUGIN_ROOT}/skills/lora-trainer/scripts/check_env.py
```

如果脚本不存在（旧版本 skill），手动逐项检查：

```bash
# Python
python --version

# CUDA + GPU
nvidia-smi

# PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 核心依赖
python -c "import transformers, peft, datasets, bitsandbytes, accelerate; print('✅ All deps OK')"
```

### Step 2: 安装缺失依赖

如果检测到缺失，自动运行：

```bash
pip install torch transformers peft datasets bitsandbytes accelerate
```

或一键安装：

```bash
pip install lora-trainer[train]
```

### Step 3: 国内用户网络优化

检测到国内网络时提示：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Step 4: 打印就绪报告

```
✅ 环境就绪！
   Python 3.11 ✅
   CUDA 12.4 + RTX 4090 (24 GB) ✅
   PyTorch 2.5 + CUDA ✅
   所有依赖已安装 ✅
   显存可用: 22.1/24 GB ✅

可以开始微调了！试试:
   /lora:analyze ./你的数据.jsonl qwen2-7b chat
```
