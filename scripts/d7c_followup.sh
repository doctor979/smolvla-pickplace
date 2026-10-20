#!/bin/bash
# 实验 C 训练结束后的自动接力：等训练退出 -> 复查 GPU -> 记录逐帧数据 -> 跑鲁棒性测试
#
# Usage:
#     setsid nohup bash scripts/d7c_followup.sh > logs/d7c_followup.log 2>&1 &
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PWD/env.sh"
PY="${PY:-python3}"

echo "=== d7c watcher 启动 $(date) ==="
while pgrep -f "lerobot[-]train" > /dev/null; do
  sleep 60
done
echo "=== 训练进程已退出 $(date) ==="
sleep 30

USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
echo "评测前复查 GPU0: ${USED} MiB"
if [ "$USED" -gt 20000 ]; then
  echo "!!!! GPU0 已被大量占用，按规矩不硬跑，退出"
  exit 3
fi

CK=outputs/expC_aug40/checkpoints/040000/pretrained_model
if [ ! -d "$CK" ]; then
  echo "!!!! 找不到 checkpoint: $CK"
  echo "实际存档:"
  ls outputs/expC_aug40/checkpoints/ 2>/dev/null
  exit 4
fi

echo "=== 跑鲁棒性测试 C40k ==="
CUDA_VISIBLE_DEVICES=0 $PY scripts/d7_robustness.py \
  --ckpt "$CK" --tag C40k --stride 10 --device cuda \
  --out results/robust_C40k.json

RC=$?
echo "=== 鲁棒性测试退出码=$RC  $(date) ==="
