#!/bin/bash
# 种子方差实验：串行训练 A 配置的两个额外种子（1001/1002），并逐帧记录。
#
# Usage:
#     bash scripts/d10_seeds.sh
#
# Output:
#     outputs/expA_train40_seed<S>/            训练存档（8 档，每档约 1GB）
#     results/perframe/A40k_seed<S>.npz        逐帧记录，供种子方差分析
#     logs/d3_seed<S>.log                      每个种子的完整训练日志
#
# Notes:
#     - 串行而非并行：GPU 0 上并行会让单次训练变慢，净耗时接近但不便于观察
#     - 种子 1000 的模型即主线 A，无需重跑
#     - **不要把训练输出接进管道**（如 `| tr | grep | tail`）：
#       lerobot 的进度条会高频写 stdout，管道消费端跟不上时训练进程会阻塞在
#       pipe_write 上假死（2026-10-19 实际踩到，卡 25 分钟且 GPU 空转）。
#       一律重定向到独立日志文件，需要看进度时再离线读日志。
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
source env.sh

# 开跑前守卫：GPU 0 已被别人占用超过 15GB 就中止，不与他人抢卡
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
if [ "${USED:-99999}" -ge 15000 ]; then
  echo "!!!! GPU 0 已占用 ${USED} MiB，停止（不与他人抢卡）"
  exit 3
fi
echo "GPU 0 可用：已用 ${USED} MiB / 24564 MiB"

SNAP=$PROJECT_ROOT/data/lerobot/hub/datasets--lerobot--svla_so101_pickplace/snapshots/f641879e22172be7e8161d5e6c1503c2d2feb657
mkdir -p results/perframe

for S in 1001 1002; do
  LOG="$PROJECT_ROOT/logs/d3_seed$S.log"
  echo "########## seed $S 训练开始 $(date)  日志=$LOG"

  bash scripts/d3_train.sh A "$S" > "$LOG" 2>&1
  RC=$?
  echo "########## seed $S 训练结束 rc=$RC $(date)"

  CKPT="$PROJECT_ROOT/outputs/expA_train40_seed$S/checkpoints/040000/pretrained_model"
  if [ -d "$CKPT" ]; then
    echo "########## seed $S 逐帧记录 $(date)"
    $PY -u scripts/d9_record.py \
        --ckpt "$CKPT" --tag "A40k_s$S" --stride 1 --batch-size 8 \
        --ds-root "$SNAP" --out "results/perframe/A40k_seed$S.npz" \
        > "$PROJECT_ROOT/logs/d9_seed$S.log" 2>&1
    echo "########## seed $S 记录完成 rc=$? $(date)"
  else
    echo "!!!! seed $S 缺少存档: $CKPT（见 $LOG 末尾）"
  fi
done

echo "ALL SEEDS DONE $(date)"
