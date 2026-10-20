#!/bin/bash
# 实验 C：数据增强训练入口。除 image_transforms 外，参数与实验 A 完全一致
#
# Usage:
#     bash scripts/d7_aug_train.sh ignite|full        增强训练，默认 seed 1000
#     bash scripts/d7_aug_train.sh full 1001          换随机种子，输出目录加 _seed1001
#
# 增强组合: 亮度 0.4~1.6 / 对比度 0.5~1.5 / 高斯模糊 k=9 / 随机遮挡 4 洞 / 仿射 ±5°
# 代价: 每步约慢 35%（变换在 CPU 侧完成），显存不变
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
ROOT="${PROJECT_ROOT}"
BIN="${BIN:?未设置 BIN：请先 source env.sh（或激活 conda 环境）}"
MODEL=${MODEL_PATH:-$ROOT/data/models/smolvla_base}
DS=lerobot/svla_so101_pickplace

MODE=${1:-ignite}
SEED=${2:-1000}
case "$MODE" in
  ignite) STEPS=100;   SAVE=100;   EVAL=0;    OUT=$ROOT/outputs/expC_ignite ;;
  full)   STEPS=40000; SAVE=5000;  EVAL=2000; OUT=$ROOT/outputs/expC_aug40  ;;
  *) echo "用法: $0 ignite|full [seed]"; exit 2 ;;
esac

# 非默认种子另开目录，避免覆盖主线存档
[ "$SEED" != "1000" ] && OUT="${OUT}_seed${SEED}"

# 增强组合：覆盖鲁棒性测试里最脆弱的几类
#   brightness/contrast -> 亮度、对比度
#   blur                -> 失焦/镜头脏（测试里 1.26x）
#   erasing             -> 遮挡（测试里最差，1.31x）
#   affine              -> 轻微位移，模拟相机安装误差
TFS='{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.4,1.6]}},"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.5,1.5]}},"blur":{"weight":1.0,"type":"GaussianBlur","kwargs":{"kernel_size":9,"sigma":[0.3,2.5]}},"erasing":{"weight":1.0,"type":"CoarseDropout","kwargs":{"max_holes":4,"max_height_frac":0.25,"max_width_frac":0.25}},"affine":{"weight":1.0,"type":"RandomAffine","kwargs":{"degrees":[-5.0,5.0],"translate":[0.04,0.04]}}}'

echo "=== 实验 C（数据增强） mode=$MODE seed=$SEED steps=$STEPS out=$OUT ==="
date
echo "--- 开跑前 GPU ---"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES:-0}"

$BIN/lerobot-train \
  --dataset.repo_id=$DS \
  --dataset.eval_split=0.2 \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=2 \
  --dataset.image_transforms.random_order=true \
  --dataset.image_transforms.tfs="$TFS" \
  --policy.path=$MODEL \
  --rename_map='{"observation.images.up": "observation.images.camera1", "observation.images.side": "observation.images.camera2"}' \
  --output_dir=$OUT \
  --job_name=smolvla_aug_${MODE}_s${SEED} \
  --batch_size=8 \
  --steps=$STEPS \
  --save_freq=$SAVE \
  --eval_steps=$EVAL \
  --log_freq=20 \
  --num_workers=${NUM_WORKERS:-4} \
  --seed=$SEED \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --wandb.enable=false

RC=$?
echo "=== 退出码=$RC ==="
echo "--- 结束后 GPU ---"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES:-0}"
exit $RC
