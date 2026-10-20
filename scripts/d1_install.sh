#!/bin/bash
# 环境安装：conda 环境 (py3.12) + ffmpeg + lerobot[dataset,training,smolvla]
#
# Usage:
#     bash scripts/d1_install.sh
#
# 约束:
#     - lerobot 0.6.1 要求 Python >= 3.12、torch >= 2.7
#     - smolvla extra 不含数据集依赖（torchcodec/pandas/pyarrow/datasets），
#       必须带 dataset extra，否则读取数据集时报 ModuleNotFoundError
#     - conda-forge 与 PyPI 均走清华镜像；直连境外源实测约 1 MB/min
#     - 幂等：环境已存在则跳过创建；ffmpeg 失败不阻断（torchcodec 自带 FFmpeg 库）
#
# Exit code: 0 成功；1 环境创建或 lerobot 安装失败

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
# 非交互式 shell 不加载 .bashrc，conda 可能不在 PATH；取不到时按常见安装位置回退
CONDA="${CONDA:-$(command -v conda)}"
if [ ! -x "$CONDA" ]; then
  for c in "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" "$HOME/software/miniconda3/bin/conda"; do
    [ -x "$c" ] && { CONDA="$c"; break; }
  done
fi
[ -x "$CONDA" ] || { echo "!!!! 找不到 conda，请用 CONDA=/path/to/conda 指定"; exit 1; }
PIPM=https://pypi.tuna.tsinghua.edu.cn/simple
CF=https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge

# 1) conda 环境（幂等）
$CONDA env list 2>/dev/null | grep -q "^lerobot " || $CONDA create -n lerobot python=3.12 -y
[ $? -ne 0 ] && { echo "!!!! conda 创建环境失败"; exit 1; }

# 2) ffmpeg：conda-forge 优先，失败不阻断
echo "===== 2b. ffmpeg（清华 conda-forge 镜像，失败不阻塞）====="
$CONDA install -n lerobot -c $CF ffmpeg -y
RC=$?
echo "conda ffmpeg 退出码: $RC"
if [ $RC -ne 0 ]; then
  echo "!!!! conda 装 ffmpeg 失败，改用 pip imageio-ffmpeg 兜底"
  $CONDA run -n lerobot --no-capture-output python -m pip install imageio-ffmpeg -i $PIPM
fi

# 3) lerobot 本体：extras 必须包含 dataset
echo "===== 3. pip 装 lerobot[smolvla] ====="
$CONDA run -n lerobot --no-capture-output python -m pip install --upgrade pip -i $PIPM
$CONDA run -n lerobot --no-capture-output python -m pip install "lerobot[smolvla]" -i $PIPM
RC=$?
echo "pip 退出码: $RC"
if [ $RC -ne 0 ]; then echo "!!!! lerobot 安装失败，终止"; exit 1; fi

# 4) ffmpeg 兜底：conda 失败时软链 pip 版 imageio-ffmpeg
echo "===== 3b. ffmpeg 软链兜底 ====="
FF=$($CONDA run -n lerobot --no-capture-output python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())" 2>/dev/null)
# 目标目录必须取 lerobot 环境自己的 bin：此时 env.sh 里的 BIN 指向的还是系统 python
LRBIN="$($CONDA info --base)/envs/lerobot/bin"
if [ -n "$FF" ]; then ln -sf "$FF" "$LRBIN/ffmpeg"; echo "软链 ffmpeg -> $FF"; else echo "imageio_ffmpeg 未装（conda ffmpeg 成功则不需要）"; fi

# 5) 核对版本与 CUDA
# 用独立脚本核对：内联 python 的引号在 bash 中难以配平，曾导致 syntax error
echo "===== 4. 核对 ====="
$CONDA run -n lerobot --no-capture-output python scripts/d1_verify.py || echo "!!!! d1_verify 未通过"
$CONDA run -n lerobot --no-capture-output python -c "import torchcodec" && echo "torchcodec OK" || echo "!!!! torchcodec import 失败"
which ffmpeg || echo "!!!! ffmpeg 不在 PATH"
echo "===== DONE ====="
