#!/bin/bash
# 微调训练入口：单次运行按 save_freq 产出多个存档
#
# Usage:
#     bash scripts/d3_train.sh ignition   100 步点火，验证参数与显存
#     bash scripts/d3_train.sh A          40 条训练集，40k 步（主线）
#     bash scripts/d3_train.sh B          30 条训练集（数据量消融）
#     bash scripts/d3_train.sh A 1001     同上，但换随机种子（输出目录自动加 _seed1001）
#
# 必填参数说明:
#     --rename_map            模型声明 camera1/2/3，数据集只有 up/side，不映射会报错
#     --policy.push_to_hub=false  push_to_hub 默认为 true，否则报 repo_id missing
#     --dataset.eval_split    按任务末尾比例留出评测集，训练中同步算验证损失
#
# Exit code: 0 训练完成；非 0 见日志 logs/ 下同名文件
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
ROOT="${PROJECT_ROOT}"
BIN="${BIN:?未设置 BIN：请先 source env.sh（或激活 conda 环境）}"
MODEL=$ROOT/data/models/smolvla_base
DS=lerobot/svla_so101_pickplace

MODE=$1
SEED=${2:-1000}
case "$MODE" in
  ignition) STEPS=100;   SAVE=100;   EVAL=0;    OUT=$ROOT/outputs/ignition;   EXTRA="--dataset.eval_split=0.2" ;;
  A)        STEPS=40000; SAVE=5000; EVAL=2000; OUT=$ROOT/outputs/expA_train40; EXTRA="--dataset.eval_split=0.2" ;;
  B)        STEPS=40000; SAVE=5000; EVAL=0;    OUT=$ROOT/outputs/expB_train30; EXTRA="--dataset.episodes=[$(seq -s, 0 29)]" ;;
  *) echo "用法: $0 ignition|A|B [seed]"; exit 2 ;;
esac

# 非默认种子另开目录，避免覆盖主线存档
[ "$SEED" != "1000" ] && OUT="${OUT}_seed${SEED}"

echo "=== 任务=$MODE  seed=$SEED  步数=$STEPS  输出=$OUT ==="
date
echo "--- 开跑前 GPU ---"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | head -1

$BIN/lerobot-train \
  --dataset.repo_id=$DS \
  $EXTRA \
  --policy.path=$MODEL \
  --rename_map='{"observation.images.up": "observation.images.camera1", "observation.images.side": "observation.images.camera2"}' \
  --output_dir=$OUT \
  --job_name=smolvla_$MODE \
  --batch_size=8 \
  --steps=$STEPS \
  --save_freq=$SAVE \
  --eval_steps=$EVAL \
  --log_freq=20 \
  --num_workers=4 \
  --seed=$SEED \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --wandb.enable=false

RC=$?
echo "=== 退出码=$RC ==="
echo "--- 结束后 GPU ---"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | head -1
exit $RC
