"""评测结果汇总：扫描 results/*.json 生成对照表。

列：标签 / 单步 L1 / 动作块 L1 / 单步 MSE / 帧数。

Usage:
    python scripts/d5_summary.py

Notes:
    只扫顶层目录；batched/（批量推理版）与 perframe/（逐帧分析）不计入本表。
"""
import glob, json, os
from pathlib import Path
# 结果目录（仓库内 results/）
ROOT = str(Path(__file__).resolve().parents[1] / "results")
# 每个 json 是一份评测结果，字段缺失时用 None 占位，排序保证表稳定
rows = []
for f in sorted(glob.glob(ROOT + "/*.json")):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    rows.append((d.get("tag") or os.path.basename(f),
                 (d.get("l1_step0") or {}).get("mean"),
                 (d.get("l1_chunk") or {}).get("mean"),
                 (d.get("mse_step0") or {}).get("mean"),
                 d.get("n_frames_evaluated")))
if not rows:
    print("results/ 里还没有结果文件")
else:
    print("%-12s %14s %14s %14s %8s" % ("标签", "单步L1", "动作块L1", "单步MSE", "帧数"))
    for r in rows:
        print("%-12s %14.4f %14.4f %14.4f %8d" % (r[0], r[1] or 0.0, r[2] or 0.0, r[3] or 0.0, r[4] or 0))
