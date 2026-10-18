#!/bin/bash
# 过夜串行任务：先跑 A，成功后再跑 B；A 失败则跳过 B，避免重复同一错误
#
# Usage:
#     bash scripts/d3_overnight.sh
#
# 前置检查:
#     GPU0 已用显存超过 1000 MiB 时直接退出（3），不使用他人正在占用的卡
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PWD/env.sh"

echo "===== 过夜任务开始 $(date) ====="

# 用户规矩：占卡前必须查空闲，被占就停下不硬跑
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
echo "GPU0 当前已用: ${USED} MiB"
if [ "$USED" -gt 1000 ]; then
  echo "!!!! GPU0 已被占用（${USED} MiB），按规矩不硬跑，任务中止"
  exit 3
fi
echo "GPU0 空闲，开始。"

echo "----- 实验 A：40 条训练集 + 留出 10 条，40k 步 -----"
bash scripts/d3_train.sh A > logs/d3_expA.log 2>&1
RC_A=$?
echo "实验 A 退出码=$RC_A  结束时间=$(date)"

if [ "$RC_A" -eq 0 ]; then
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
  echo "跑 B 前复查 GPU0: ${USED} MiB"
  echo "----- 实验 B：30 条训练集（数据量消融），40k 步 -----"
  bash scripts/d3_train.sh B > logs/d3_expB.log 2>&1
  echo "实验 B 退出码=$?  结束时间=$(date)"
else
  echo "实验 A 失败，按设计跳过实验 B（避免重复同样错误）"
fi

echo "===== 过夜任务结束 $(date) ====="
