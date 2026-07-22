# LoRA 微调常见问题

## 基础概念

### Q: LoRA 和全量微调效果差多少？
A: LoRA 只训练 0.1%-1% 的参数，但在大多数任务（尤其数据量 < 10k）上效果接近全量微调。数据越少，LoRA 的优势越明显（不易过拟合）。

### Q: 多少数据可以开始微调？
A: 最少 50-100 条就能看到效果。建议至少 500 条获得稳定效果。高质量数据比数量更重要。

### Q: 微调会「忘记」原来的能力吗？
A: 这种现象叫 catastrophic forgetting。LoRA 风险远小于全量微调（只改少量参数）。低 rank + 高 alpha 可进一步保护原始能力。

### Q: QLoRA 和 LoRA 有什么区别？
A: QLoRA = LoRA + 4-bit 量化。显存减半，效果基本持平。目前业界默认使用 QLoRA。

## 参数调优

### Q: rank 设多大合适？
A: 数据 < 1k → r=4；1k-5k → r=8；5k-20k → r=16；> 20k → r=32。代码/数学任务加一档。

### Q: alpha 和 rank 的关系？
A: 通常 alpha = 2×r。alpha 越大，LoRA 对输出的影响越强。角色扮演可以用 4×r，数学推理用 1×r。

### Q: 为什么我的 loss 不下降？
A: 检查顺序：1) 数据格式是否正确 2) 学习率是否太低 3) 是否有 NaN 值 4) max_length 是否截断了重要内容。

### Q: 要不要训练所有层？
A: 不需要。LoRA 默认只训练 Q/V 注意力层就够了。代码/数学任务可以加更多层，角色扮演建议只用 V 层。

## 显存问题

### Q: 我的显卡只有 6GB，能微调 7B 模型吗？
A: 可以，用 QLoRA + batch_size=1 + gradient_accumulation=16 + max_length=1024。训练会很慢但能跑。

### Q: CUDA Out of Memory 怎么办？
A: 依次尝试：1) 减小 max_length 2) 减小 batch_size 3) 开启 gradient_checkpointing 4) 确认用了 4-bit 量化 5) 换更小的模型。

### Q: batch_size 和 gradient_accumulation 怎么配合？
A: 目标有效 batch ≈ 16。显存不够就用小 batch + 大 gradient_accumulation。如 bs=1, ga=16 等价于 bs=16 但省显存。

## 数据问题

### Q: 数据格式应该是什么样的？
A: 推荐 JSONL 格式（每行一个 JSON）。支持三种格式：`{"instruction":"...", "output":"..."}` 或 `{"messages":[...]}` 或 `{"conversations":[...]}`。

### Q: 需要多少数据？
A: 质量 > 数量。1000 条高质量数据胜过 10000 条噪声数据。每条 instruction 应该清晰、有代表性、覆盖目标场景。

### Q: 需要验证集吗？
A: 建议留 5-10% 做验证集。改为 `train.jsonl` 和 `eval.jsonl` 两个文件。

## 训练问题

### Q: 训练多久？
A: 取决于数据量和 epochs。1k 数据 3 epochs 通常在 30 分钟到 2 小时（7B 模型，单卡 24GB）。

### Q: 怎么知道训练好了？
A: 看 eval loss。如果 eval loss 不再下降（甚至上升）= 过拟合，该停了。

### Q: 训练完怎么用？
A: 用 PeftModel 加载 adapter：`model = PeftModel.from_pretrained(base_model, "lora_path")`。推理时可以 merge_and_unload 提速。

### Q: fp16 还是 bf16？
A: RTX 30/40 系列、A100、H100 → bf16（更快更稳，不会炸 NaN）。老 GPU（V100、T4）→ fp16。不确定时用 fp16 更保险。

### Q: loss 突然爆了怎么办？
A: 1) 降低学习率 2) 确认 max_grad_norm=1.0 已设置 3) 检查数据中是否有超长/异常样本 4) 确认 bf16/fp16 没有溢出。

### Q: target_modules 设了但训练效果没变化？
A: 很可能模块名不匹配——`get_peft_model()` 找不到对应层会静默跳过。先 `print(model)` 确认实际的 attention 层名，确保 target_modules 里的名字完全一致。

### Q: 训练到一半停了怎么恢复？
A: 生成的训练脚本默认每 500 steps 保存一个 checkpoint 到 `./output/checkpoint-xxx/`。从中断的 checkpoint 恢复：
```python
# 修改训练脚本中的 model 加载方式
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("./output/checkpoint-500/")
```
然后继续训练即可。注意调整 `num_train_epochs` 减去已训练的轮数。

### Q: 怎么看训练过程是否正常？
A: 三条指标：
1. **loss 曲线**：平滑下降 = 正常；剧烈震荡 = lr 太高；不下降 = 数据或 lr 有问题
2. **显存使用**：稳定在一个水平线 = 正常；逐渐增长 = 可能内存泄漏
3. **GPU 利用率**：`nvidia-smi` 看 GPU-Util，90%+ = 计算瓶颈正常；< 50% = 数据加载太慢
  
用 tensorboard 直观查看：
```bash
tensorboard --logdir ./output/logs
# 浏览器打开 http://localhost:6006
```

### Q: 国内下载模型太慢怎么办？
A: 三种方式：
1. **hf-mirror.com**（最快）：`export HF_ENDPOINT=https://hf-mirror.com` 然后正常下载
2. **ModelScope**（阿里云）：`pip install modelscope` → `snapshot_download('qwen/Qwen2-7B-Instruct')`
3. **手动下载**：浏览器访问 hf-mirror.com 下载文件到本地

国内用户建议优先用 ModelScope 下载 Qwen 系列（原生支持，速度最快）。生成的训练脚本中把 `MODEL_NAME` 改成本地路径即可。
