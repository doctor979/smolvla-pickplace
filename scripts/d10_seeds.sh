#!/bin/bash
# 种子方差实验：串行训练 A 配置的两个额外种子（1001/1002），并逐帧记录。
#
# Usage:
#     bash scripts/_d10_seeds.sh
#
# Output:
#     outputs/expA_train40_seed<S>/            训练存档（8 档，每档约 1GB）
#     results/perframe/A40k_seed<S>.npz        逐帧记录，供种子方差分析
#
# Notes:
#     - 串行而非并行：GPU 0 上并行会让单次训练变慢，净耗时接近但不便于观察
#     - 种子 1000 的模型即主线 A，无需重跑
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
source env.sh

# 开跑前守卫：GPU 0 已被别人占用超过 4GB 就中止，不与他人抢卡
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
if [ "${USED:-99999}" -ge 4000 ]; then
  echo "!!!! GPU 0 已占用 ${USED} MiB，停止（不与他人抢卡）"
  exit 3
fi
echo "GPU 0 可用：已用 ${USED} MiB / 24564 MiB"

SNAP=$PROJECT_ROOT/data/lerobot/hub/datasets--lerobot--svla_so101_pickplace/snapshots/f641879e22172be7e8161d5e6c1503c2d2feb657
mkdir -p results/perframe

for S in 1001 1002; do
  echo "########## seed $S 训练开始 $(date)"
  bash scripts/d3_train.sh A "$S" 2>&1 | tr '\r' '\n' | grep -aE "任务=|step:(10000|20000|30000|40000) |End of training|退出码" | tail -6

  CKPT="$PROJECT_ROOT/outputs/expA_train40_seed$S/checkpoints/040000/pretrained_model"
  if [ -d "$CKPT" ]; then
    echo "########## seed $S 逐帧记录 $(date)"
    $PY -u scripts/d9_record.py \
        --ckpt "$CKPT" --tag "A40k_s$S" --stride 1 --batch-size 8 \
        --ds-root "$SNAP" --out "results/perframe/A40k_seed$S.npz" 2>&1 \
      | tr '\r' '\n' | grep -aE "已保存|平均" | tail -3
  else
    echo "!!!! seed $S 缺少存档: $CKPT"
  fi
done

echo "ALL SEEDS DONE $(date)"
