# SmolVLA 项目环境变量 —— 跑任何命令前先 source 本文件

# 项目根目录：按本文件所在位置推断，换机器/改目录名都不用改这里
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}"
export PROJECT_ROOT

# 小文件走镜像（huggingface.co 不可直连时）；大模型权重见 README 的 ModelScope 说明
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# 缓存与数据全部落在项目目录内：不碰 /home 配额，也不写 /tmp
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/data/hf}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$PROJECT_ROOT/data/lerobot}"

# 关键：禁用 Xet 后端 —— hf-mirror 不代理 xethub 的 CAS 服务，启用会 401 Unauthorized
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

# 只占一张卡；多卡机器上按需改
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# 训练日志只写本地，不上传 wandb
export WANDB_MODE=offline
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false

# 解释器：脚本统一通过 $PY / $BIN 调用；激活了 conda 环境时默认就是对的
export PY="${PY:-python3}"
export BIN="${BIN:-$(dirname "$(command -v "${PY}" 2>/dev/null || command -v python3)")}"

# 机器私有覆盖（不入库）：例如 PY=/path/to/envs/lerobot/bin/python
[ -f "$PROJECT_ROOT/env.local.sh" ] && . "$PROJECT_ROOT/env.local.sh"
