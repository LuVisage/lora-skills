"""
训练脚本模板 — 用于 script_builder.py 生成最终训练脚本时参考。

这是一个 Jinja2 风格的模板文件，{{变量}} 会被实际值替换。
实际生成由 core/script_builder.py 的 build_training_script() 方法完成。
"""

# 模板变量说明：
# {{model_path}}      — 基座模型路径
# {{data_path}}        — 训练数据路径
# {{output_dir}}       — 输出目录
# {{rank}}             — LoRA rank
# {{alpha}}            — LoRA alpha
# {{target_modules}}   — 目标模块列表
# {{dropout}}          — LoRA dropout
# {{learning_rate}}    — 学习率
# {{epochs}}           — 训练轮数
# {{batch_size}}       — 批次大小
# {{grad_accum}}       — 梯度累积步数
# {{max_length}}       — 最大序列长度
