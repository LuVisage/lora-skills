#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
环境检测脚本 — 一键诊断微调环境是否就绪
==========================================
运行: python check_env.py

检测项目:
  1. Python 版本 (≥ 3.10)
  2. CUDA 驱动 + GPU 信息
  3. PyTorch (CUDA 版本)
  4. 核心依赖 (transformers, peft, datasets, bitsandbytes, accelerate)
  5. GPU 显存状态
  6. 可选: 国内网络镜像提示
"""

import sys
import os
import subprocess


def check(label: str, ok: bool, detail: str = "", fix: str = "") -> bool:
    """打印检查结果，返回是否通过。"""
    if ok:
        print(f"  ✅ {label}: {detail}" if detail else f"  ✅ {label}")
    else:
        print(f"  ❌ {label}")
        if detail:
            print(f"     → {detail}")
        if fix:
            print(f"     🔧 修复: {fix}")
    return ok


def main():
    print("=" * 55)
    print("  LoRA 微调环境检测")
    print("=" * 55)
    all_ok = True

    # ── 1. Python 版本 ──
    py_ver = sys.version_info
    all_ok &= check(
        "Python 版本",
        py_ver >= (3, 10),
        f"Python {py_ver.major}.{py_ver.minor}.{py_ver.micro}",
        "需要 Python ≥ 3.10: https://python.org/downloads",
    )

    # ── 2. CUDA 驱动 ──
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,cuda_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            info = result.stdout.strip()
            print(f"  ✅ CUDA 驱动 + GPU:")
            for line in info.split("\n"):
                if line.strip():
                    print(f"     {line.strip()}")
        else:
            all_ok &= check("nvidia-smi", False, "", "安装 NVIDIA 驱动: https://nvidia.com/drivers")
    except FileNotFoundError:
        all_ok &= check("nvidia-smi", False, "未找到 nvidia-smi", "安装 NVIDIA 驱动: https://nvidia.com/drivers")
    except Exception:
        print("  ⚠️ nvidia-smi 检测失败（可能无 GPU）")

    # ── 3. PyTorch ──
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
            print(f"  ✅ PyTorch {torch.__version__} + CUDA {torch.version.cuda}")
            print(f"     GPU: {gpu_name} ({gpu_mem:.1f} GB) × {gpu_count}")
        else:
            print(f"  ⚠️ PyTorch {torch.__version__} — CUDA 不可用")
            print(f"     如使用 GPU 训练，请装 CUDA 版: pip install torch --index-url https://download.pytorch.org/whl/cu121")
    except ImportError:
        all_ok &= check("PyTorch", False, "", "pip install torch")

    # ── 4. 核心依赖 ──
    deps = [
        ("transformers", "pip install transformers"),
        ("peft", "pip install peft"),
        ("datasets", "pip install datasets"),
        ("bitsandbytes", "pip install bitsandbytes"),
        ("accelerate", "pip install accelerate"),
    ]
    for lib, fix in deps:
        try:
            mod = __import__(lib)
            ver = getattr(mod, "__version__", "?")
            print(f"  ✅ {lib} {ver}")
        except ImportError:
            all_ok &= check(lib, False, "", fix)

    # ── 5. 显存状态 ──
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for i, line in enumerate(result.stdout.strip().split("\n")):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    free_mb, used_mb, total_mb = int(parts[0]), int(parts[1]), int(parts[2])
                    free_gb = free_mb / 1024
                    total_gb = total_mb / 1024
                    used_pct = used_mb / total_mb * 100 if total_mb > 0 else 0

                    if free_gb < 6:
                        status = "⚠️ 紧张 (QLoRA 7B 需 ≥ 6GB)"
                    elif free_gb < 10:
                        status = "⚡ 可训练小模型"
                    else:
                        status = "✅ 充裕"

                    print(f"  📊 GPU {i}: {free_gb:.1f}/{total_gb:.1f} GB 可用 ({used_pct:.0f}% 占用) — {status}")
    except Exception:
        pass

    # ── 6. 国内用户网络提示 ──
    hf_endpoint = os.environ.get("HF_ENDPOINT", "")
    if "hf-mirror.com" in hf_endpoint:
        print("  🌐 HF 镜像已配置: hf-mirror.com ✅")
    else:
        print("  💡 国内用户可加速下载:")
        print("     export HF_ENDPOINT=https://hf-mirror.com")
        print("     或用 ModelScope: pip install modelscope && python -c \"from modelscope import snapshot_download; snapshot_download('模型名', cache_dir='./models')\"")

    # ── 结果 ──
    print("\n" + "=" * 55)
    if all_ok:
        print("  ✅ 环境就绪，可以开始微调！")
        print("=" * 55)
        return 0
    else:
        print("  ⚠️ 存在问题，修复后重新运行: python check_env.py")
        print("=" * 55)
        return 1


if __name__ == "__main__":
    sys.exit(main())
