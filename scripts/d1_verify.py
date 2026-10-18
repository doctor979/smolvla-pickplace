"""环境自检：校验运行期依赖版本与 CUDA 可用性。

逐项 import 依赖并打印版本；单项失败只标 FAIL 不中断，便于一次性看清缺了哪些。
最后校验 torch 能否识别 GPU 及计算能力（驱动正常不代表 torch 可用 GPU）。

Usage:
    python scripts/d1_verify.py

Exit code:
    0  全部依赖可用且 CUDA 可用
    1  存在 FAIL 项，或 CUDA 不可用
"""
import importlib
import sys

DEPENDENCIES = [
    "torch", "torchvision", "lerobot", "transformers", "numpy", "torchcodec",
    "pandas", "pyarrow", "datasets", "accelerate", "safetensors", "PIL", "cv2",
]

failed = []

print("python:", sys.version.split()[0])
for name in DEPENDENCIES:
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", getattr(mod, "version", "?"))
        print("  OK   %-14s %s" % (name, ver))
    except Exception as exc:                      # noqa: BLE001 - 自检脚本需容错
        failed.append(name)
        print("  FAIL %-14s %s: %s" % (name, type(exc).__name__, exc))

import torch                                       # noqa: E402 - 需先确认 import 成功

cuda_ok = torch.cuda.is_available()
print("cuda_available:", cuda_ok)
if cuda_ok:
    # NOTE: capability 用于判断是否装错 CUDA 版本（30/40 系为 (8,6)/(8,9)）
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))

if failed or not cuda_ok:
    print("RESULT: FAILED  (%d 个依赖缺失)" % len(failed))
    sys.exit(1)
print("RESULT: OK")
