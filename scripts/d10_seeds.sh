#!/bin/bash
# 种子方差实验：给 A（常规微调）和 C（图像增强）各补若干随机种子，跑完自动逐帧记录。
#
# Usage:
#     bash scripts/d10_seeds.sh              # 默认补 seed 1001 1002
#     bash scripts/d10_seeds.sh 1001 1002 1003
#     GPU_A=6 GPU_C=7 bash scripts/d10_seeds.sh
#
# Output:
#     outputs/expA_train40_seed<S>/             A 的训练存档
#     outputs/expC_aug40_seed<S>/               C 的训练存档
#     results/perframe/A40k_seed<S>.npz         逐帧记录（A）
#     results/perframe/C40k_seed<S>.npz         逐帧记录（C）
#     logs/seeds_A.log  logs/seeds_C.log        两条流水线的总日志
#     logs/d3_seed<S>.log  logs/d7_seed<S>.log  单次训练日志
#
# 为什么要 A 和 C 都补：只给 A 加种子，无法判断「C 比 A 好约 3%」是真实效应还是训练随机性。
# 判定方式：看各 seed 下 (C − A) 的差值符号与幅度是否一致。只补 A 的种子证明不了这一点。
#
# 为什么流水线内串行、流水线间并行：同一条流水线的多个种子共用一张卡，并行只会互相拖慢；
# A 与 C 用两张不同的空卡，并行能把墙钟时间减半。前提是 NUM_WORKERS 调小（见下方 pipeline）。
#
# 注意：**不要把训练输出接进管道**（如 `| tr | grep | tail`）。lerobot 的进度条会高频写
# stdout，管道消费端跟不上时训练进程会阻塞在 pipe_write 上假死（2026-10-19 实际踩到，
# GPU 空转 25 分钟）。一律重定向到独立日志文件，需要看进度时离线读日志。
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
source env.sh

# 不要设 HF_HUB_OFFLINE=1：lerobot 启动时会查
# https://hf-mirror.com/api/datasets/.../refs 解析 revision，离线模式下直接抛
# OfflineModeIsEnabled 并退出（实测 rc=1），即使数据与权重都已缓存在本地。
# 两条流水线同时发起该请求会触发限流（"Rate limited. Waiting 44.0s before retry"），
# 因此改为错开启动：A 先跑 STAGGER 秒，等它完成元数据校验后再起 C。

SEEDS="${*:-1001 1002}"
GPU_A="${GPU_A:-6}"
GPU_C="${GPU_C:-7}"
LIMIT_MIB="${LIMIT_MIB:-15000}"

# 开跑前守卫：目标卡被别人占用过多就中止，不与他人抢卡
for G in "$GPU_A" "$GPU_C"; do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$G" 2>/dev/null)
  if [ "${USED:-99999}" -ge "$LIMIT_MIB" ]; then
    echo "!!!! GPU $G 已占用 ${USED} MiB（阈值 ${LIMIT_MIB}），停止"
    exit 3
  fi
  echo "GPU $G 可用：已用 ${USED} MiB"
done

# 数据与模型放到本地盘：/home 是 NFS，训练时 dataloader 的小块随机读会挂死在
# rpc_wait_bit_killable（主进程随之卡 pipe_write，GPU 0% 空转）。
# 数据集仅 83MB、模型 695MB，整份拷到 /tmp（本地 ext4）即可完全绕开。
# 拷贝成本约 8 秒，换来的是读速从 1~2 MB/s 变成内存带宽级别。
LOCAL=${LOCAL:-/tmp/smolvla}
if [ -d "$LOCAL/lerobot/hub" ]; then
  export HF_LEROBOT_HOME="$LOCAL/lerobot"
  export MODEL_PATH="$LOCAL/models/smolvla_base"
  SNAP=$LOCAL/lerobot/hub/datasets--lerobot--svla_so101_pickplace/snapshots/f641879e22172be7e8161d5e6c1503c2d2feb657
  echo "使用本地盘副本：$LOCAL"
else
  SNAP=$PROJECT_ROOT/data/lerobot/hub/datasets--lerobot--svla_so101_pickplace/snapshots/f641879e22172be7e8161d5e6c1503c2d2feb657
  echo "本地盘副本不存在，回退到项目目录（NFS，可能很慢）"
fi
mkdir -p results/perframe logs

# $1 = A|C, $2 = GPU 编号
pipeline() {
  local KIND=$1 GPU=$2
  export CUDA_VISIBLE_DEVICES=$GPU
  # 必须用 0：spawn 模式下每个 dataloader worker 会**重新 import 整个库栈**，
  # 而 Python 环境在 NFS 上，worker 会卡死在 rpc_wait_bit_killable（主进程随之卡
  # pipe_write，GPU 0% 空转、日志停在 0/40000）。主进程已经 import 完毕，用 0 就
  # 不再 fork/import，实测 ignite 100 步 35 秒跑通（3.58 step/s，data_s 约 0.10s/步）。
  # 代价是数据加载在 CPU 串行，比 4 worker 的 6.4 step/s 慢一倍。
  export NUM_WORKERS="${NUM_WORKERS:-0}"
  local S OUT TAG TRLOG CKPT RC

  for S in $SEEDS; do
    if [ "$KIND" = "A" ]; then
      OUT=$PROJECT_ROOT/outputs/expA_train40_seed$S
      TAG=A40k_s$S
      TRLOG=$PROJECT_ROOT/logs/d3_seed$S.log
      bash scripts/d3_train.sh A "$S" > "$TRLOG" 2>&1
    else
      OUT=$PROJECT_ROOT/outputs/expC_aug40_seed$S
      TAG=C40k_s$S
      TRLOG=$PROJECT_ROOT/logs/d7_seed$S.log
      bash scripts/d7_aug_train.sh full "$S" > "$TRLOG" 2>&1
    fi
    RC=$?
    echo "[$KIND s$S] 训练结束 rc=$RC $(date)"

    CKPT=$OUT/checkpoints/040000/pretrained_model
    if [ -d "$CKPT" ]; then
      echo "[$KIND s$S] 逐帧记录开始 $(date)"
      $PY -u scripts/d9_record.py \
          --ckpt "$CKPT" --tag "$TAG" --stride 1 --batch-size 8 \
          --ds-root "$SNAP" --out "$PROJECT_ROOT/results/perframe/$TAG.npz" \
          > "$PROJECT_ROOT/logs/d9_${TAG}.log" 2>&1
      echo "[$KIND s$S] 记录完成 rc=$? $(date)"
    else
      echo "[$KIND s$S] !!!! 缺少存档 $CKPT（见 $TRLOG 末尾）"
    fi
  done
}

echo "=== 种子: $SEEDS | A -> GPU$GPU_A, C -> GPU$GPU_C  $(date) ==="

pipeline A "$GPU_A" > "$PROJECT_ROOT/logs/seeds_A.log" 2>&1 &
PA=$!

# 错开启动：两条流水线同时请求 hf-mirror 的 dataset refs 会被限流，两条都卡住
STAGGER="${STAGGER:-90}"
echo "等待 ${STAGGER}s 后启动 C（避开元数据请求限流）"
sleep "$STAGGER"

pipeline C "$GPU_C" > "$PROJECT_ROOT/logs/seeds_C.log" 2>&1 &
PC=$!

wait $PA; echo "=== 流水线 A 结束 $(date) ==="
wait $PC; echo "=== 流水线 C 结束 $(date) ==="
echo "ALL SEEDS DONE $(date)"
