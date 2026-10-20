#!/bin/bash
# 实验 D：动作块长度消融。chunk_size=10，其余参数与实验 A 一致
#
# Usage:
#     bash scripts/d9_chunk_train.sh ignite|full       ignite=100 步验证参数
#
# Notes:
#     - n_action_steps 必须 <= chunk_size，故两者同时设为 10
#     - 与 A(20k) 比较时须用同一训练步数，否则块长与步数两个变量混杂
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
source "$PWD/env.sh"
export CUDA_VISIBLE_DEVICES=0
BIN="${BIN:-$(dirname "$(command -v python3)")}"
MODEL="$PWD/data/models/smolvla_base"
DS=lerobot/svla_so101_pickplace

MODE=$1
if [ "$MODE" = "ignite" ]; then
  STEPS=100;   SAVE=100;   OUT="$PWD/outputs/d9_ignite"; JOB=chunk10_ignite
elif [ "$MODE" = "full" ]; then
  STEPS=20000; SAVE=5000;  OUT="$PWD/outputs/expD_chunk10"; JOB=chunk10
else
  echo "用法: $0 ignite|full"; exit 2
fi

echo "=== mode=$MODE  steps=$STEPS  chunk_size=10 ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | head -1

$BIN/lerobot-train \
  --dataset.repo_id=$DS \
  --dataset.eval_split=0.2 \
  --policy.path=$MODEL \
  --policy.chunk_size=10 \
  --policy.n_action_steps=10 \
  --rename_map='{"observation.images.up": "observation.images.camera1", "observation.images.side": "observation.images.camera2"}' \
  --output_dir=$OUT \
  --job_name=$JOB \
  --batch_size=8 \
  --steps=$STEPS \
  --save_freq=$SAVE \
  --eval_steps=0 \
  --log_freq=20 \
  --num_workers=4 \
  --seed=1000 \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --wandb.enable=false
echo "=== 退出码=$? ==="
