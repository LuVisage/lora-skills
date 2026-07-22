---
name: lora:debug
version: 2.5.0
description: >-
  Diagnose training failures and runtime errors. Analyzes error messages, training
  logs, loss curves, and GPU memory issues. Use when training fails, loss behaves
  unexpectedly, or you need to understand what went wrong. Trigger: /lora:debug,
  "训练报错", "loss 不收敛", "OOM 怎么办", "帮我看看日志", "debug training".
argument-hint: <log-path | error-message>
disable-model-invocation: true
allowed-tools: Read, Bash(python *), Bash(nvidia-smi *), Glob, Grep
---

# /lora:debug — 训练故障诊断

训练出问题了？把日志或报错发给我，帮你定位原因。

## 参数

- `$ARGUMENTS[0]` — 训练日志路径、报错信息、或问题描述，必填

## 示例

```
/lora:debug ./output/logs
/lora:debug "CUDA out of memory"
/lora:debug loss 不收敛
/lora:debug 训练完了效果不好
```

## 执行

加载 lora-trainer skill 的故障排查能力。

### 诊断流程

1. **判断输入类型**
   - 文件路径 → 读取日志，提取关键指标（loss 曲线、显存峰值、训练时长）
   - 报错信息 → 匹配故障排查表中的已知问题
   - 自然语言描述 → 根据描述的 symptoms 反推可能原因

2. **逐项排查**
   - 环境问题：CUDA 可用？依赖装全了？显存够？
   - 数据问题：格式对？有空值？长度分布合理？
   - 参数问题：lr 合理？rank 匹配数据量？batch size 匹配显存？
   - 脚本问题：target_modules 匹配模型？量化配置开启？

3. **给出解决方案**
   - 每个可能原因附带具体操作步骤
   - 优先推荐最简单的修复（如降 seq_length 而非换模型）
   - 如果单次调整可能不够，给出递进方案（先试 A，不行再 B）

### 诊断清单

逐项检查，不跳步：

- [ ] CUDA 驱动正常？`nvidia-smi`
- [ ] PyTorch 能识别 GPU？`python -c "import torch; print(torch.cuda.is_available())"`
- [ ] 显存够？对比训练脚本的 batch_size × seq_length 和实际可用显存
- [ ] 数据格式对？每行是合法 JSON？
- [ ] 数据无空值？instruction 和 output 都有内容？
- [ ] 量化配置开启？`load_in_4bit=True` 确认
- [ ] target_modules 名称匹配模型架构？
- [ ] 学习率与 rank 匹配？
- [ ] gradient_accumulation 补齐了有效 batch？

### 输出格式

```
🔍 诊断：<问题一句话总结>

可能原因（按概率排序）：

1. ✅ <最可能原因>（概率 ~XX%）
   → 解决方法：<具体操作步骤>

2. ⚠️ <次要原因>（概率 ~XX%）
   → 解决方法：<具体操作步骤>

建议：先试方案 1，不行再试方案 2。
```

如果无法确定原因，列出需要用户提供的额外信息（完整报错栈、训练配置、GPU 型号等）。
