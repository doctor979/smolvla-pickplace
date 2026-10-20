#!/bin/bash
# 批量评测：遍历 base 与三个实验的全部存档，逐档评测后汇总
#
# Usage:
#     bash scripts/d5_eval_all.sh [stride] [batch_size]      默认 5 8
#
# Notes:
#     - 存档目录名是 6 位补零（005000），且模型位于 pretrained_model/ 子目录
#     - 自动探测本地数据集快照以跳过 Hub 查询；网络慢时不加会阻塞在 list_repo_refs
#     - GPU0 占用超过 20 GB 时中止，不与他人抢占
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
source "$PWD/env.sh"
PY="${PY:-python3}"
STRIDE=${1:-5}
BATCH=${2:-8}
mkdir -p results

# 本地数据集快照：跳过 Hub 的网络查询（网络慢时不加会卡在 list_repo_refs）
SNAP=$(ls -d data/lerobot/hub/datasets--lerobot--svla_so101_pickplace/snapshots/*/ 2>/dev/null | head -1)
if [ -n "$SNAP" ]; then DSROOT="--ds-root $SNAP"; else DSROOT=""; fi
echo "stride=$STRIDE  batch=$BATCH  ds_root=${SNAP:-<Hub>}"

# 用户规矩：占卡前先查空闲
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
echo "开跑前 GPU0 已用: ${USED} MiB"
if [ "$USED" -gt 20000 ]; then
  echo "!!!! GPU0 已被他人大量占用，按规矩不硬跑，评测中止"
  exit 3
fi

run() {
  if [ ! -e "$1" ]; then echo "跳过 $2（路径不存在: $1）"; return; fi
  echo "----- 评测 $2 -----"
  $PY scripts/d5_eval_holdout.py --ckpt "$1" --tag "$2" --stride $STRIDE \
      --batch-size $BATCH $DSROOT \
      --device cuda --out results/$2.json 2>&1 | tr "\r" "\n" | tail -8
}

run data/models/smolvla_base base
for exp in expA_train40 expB_train30 expC_aug40; do
  for s in 5000 10000 15000 20000 25000 30000 35000 40000; do
    d=$(printf "%06d" $s)
    run outputs/$exp/checkpoints/$d ${exp:3:1}_${s}
  done
done

echo
echo "################ 汇总 ################"
$PY scripts/d5_summary.py
