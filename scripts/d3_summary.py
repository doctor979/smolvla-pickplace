"""训练过程汇总：解析日志得到起点、最新状态与每 5k 步采样，并列出存档目录。

Usage:
    python scripts/d3_summary.py
    bash scripts/d3_summary.sh     # 连磁盘占用一起看
"""
import re, os
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1])

# 解析一行日志：抽 step / loss / lr / 梯度范数 / 显存，缺哪个就跳过哪个
def parse(path):
    pts = []
    if not os.path.exists(path):
        return pts
    with open(path, "rb") as f:
        txt = f.read().decode("utf-8", "ignore").replace("\r", "\n")
    for line in txt.split("\n"):
        if "loss:" not in line:
            continue
        d = {}
        m = re.search(r"\bstep:(\d+)", line)
        if m:
            d["step"] = int(m.group(1))
        for k in ["loss", "lr", "grdn", "mem_gb"]:
            mm = re.search(k + r":([0-9.eE+-]+)", line)
            if mm:
                d[k] = float(mm.group(1))
        if "step" in d:
            pts.append(d)
    return pts

# 统一格式化一行输出，保证多次运行对齐好看
def show(p):
    return "step=%6d  loss=%.4f  lr=%.2e  mem=%.2fGB" % (
        p["step"], p.get("loss", 0.0), p.get("lr", 0.0), p.get("mem_gb", 0.0))

def looks_like_eval(line):
    low = line.lower()
    return ("eval" in low) or ("val_" in low)

# 实验 A 与 B 的日志路径（A=40 条训练集，B=30 条，用于数据量消融）
for name, path in [("实验A（40条训练集 + 留出10条）", ROOT + "/logs/d3_expA.log"),
                   ("实验B（30条训练集，数据量消融）", ROOT + "/logs/d3_expB.log")]:
    print("=" * 62)
    print(name)
    pts = parse(path)
    if not pts:
        print("  暂无训练记录（还没开始，或日志为空）")
        continue
    print("  记录点数: %d" % len(pts))
    print("  起点: " + show(pts[0]))
    print("  最新: " + show(pts[-1]))
    drops = pts[0].get("loss", 0) - pts[-1].get("loss", 0)
    print("  loss 变化: %+.4f" % (-drops))
    seen = {}
    for p in pts:
        seen[int(p["step"] // 5000)] = p
    print("  曲线采样（每 5k 步一个点）:")
    for b in sorted(seen):
        print("    " + show(seen[b]))

print("=" * 62)
# 存档清单：expA/expB 每 5000 步一个，直接看有没有按预期产出
print("checkpoint 存档情况")
for exp in ["ignition", "expA_train40", "expB_train30"]:
    ck = ROOT + "/outputs/" + exp + "/checkpoints"
    if os.path.isdir(ck):
        subs = sorted(os.listdir(ck))
        print("  %-14s -> %s" % (exp, subs if subs else "（空）"))
    else:
        print("  %-14s -> 还没有" % exp)

print("=" * 62)
print("GPU0 当前占用")
os.system("nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | head -1")
