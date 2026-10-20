#!/bin/bash
# 训练过程汇总：过夜任务状态 + 训练曲线采样 + 磁盘占用
#
# Usage:
#     bash scripts/d3_summary.sh

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# ---- 1. 谁启动了、跑到哪、退出码多少 ----
echo "################ 过夜任务状态 ################"
cat logs/d3_overnight.log
echo
# ---- 2. 曲线与存档清单（由 d3_summary.py 生成）----
echo "################ 训练曲线汇总 ################"
"${PY:-python3}" scripts/d3_summary.py
echo
# ---- 3. 磁盘：每个实验 8 个存档，别把配额写爆 ----
echo "################ 磁盘占用 ################"
du -sh . 2>/dev/null
