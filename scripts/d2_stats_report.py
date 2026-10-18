"""数据统计报告：episode 时长分布、动作与状态统计量，输出 Markdown。

Output:
    docs/05_数据统计报告.md

Usage:
    python scripts/d2_stats_report.py
"""
import json, os
import torch

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from lerobot.datasets.lerobot_dataset import LeRobotDataset

DS = "lerobot/svla_so101_pickplace"
ds = LeRobotDataset(DS)
st = ds.meta.stats
L = []
def w(s=""):
    print(s)
    L.append(s)

w("=" * 66)
w("# 数据统计报告：%s" % DS)
w()
w("| 项目 | 值 |")
w("|---|---|")
w("| episodes | %d |" % ds.num_episodes)
w("| frames | %d |" % ds.num_frames)
w("| fps | %d |" % ds.fps)
w("| 总时长 | %.1f 秒 |" % (ds.num_frames / ds.fps))
w("| 平均每条 | %.1f 秒 |" % (ds.num_frames / ds.fps / ds.num_episodes))

lens = None
try:
    ep = ds.meta.episodes
    if hasattr(ep, "to_pandas"):
        df = ep.to_pandas()
        w("| episodes 表列 | %s |" % ", ".join(list(df.columns)[:8]))
        if "dataset_from_index" in df.columns and "dataset_to_index" in df.columns:
            lens = (df["dataset_to_index"] - df["dataset_from_index"]).tolist()
except Exception as e:
    w("| episode 长度读取 | 失败 %s |" % type(e).__name__)

w()
if lens:
    t = torch.tensor([float(x) for x in lens])
    w("## episode 帧数分布（共 %d 条）" % len(lens))
    w()
    w("| 统计量 | 帧数 | 折算秒数 |")
    w("|---|---|---|")
    for name, val in [("最小", t.min()), ("25%", t.quantile(0.25)), ("中位", t.median()),
                      ("均值", t.mean()), ("75%", t.quantile(0.75)), ("最大", t.max())]:
        w("| %s | %.0f | %.1f |" % (name, float(val), float(val) / ds.fps))
    w()

w("## 动作 / 状态 统计量")
w()
for key in ["action", "observation.state"]:
    s = st.get(key)
    if not isinstance(s, dict):
        continue
    w("### %s" % key)
    w()
    w("| 维 | mean | std | min | max |")
    w("|---|---|---|---|---|")
    mn = s["mean"].flatten()
    sd = s["std"].flatten()
    lo = s["min"].flatten()
    hi = s["max"].flatten()
    for j in range(mn.size):
        w("| %d | %.3f | %.3f | %.3f | %.3f |" % (j, float(mn[j]), float(sd[j]), float(lo[j]), float(hi[j])))
    w()

w("## 与训练相关的关键数字")
w()
act_sd = st["action"]["std"].flatten()
w("- 动作各维标准差: %s" % [round(float(x), 3) for x in act_sd])
w("- 动作第 5 维（夹爪）范围: %.2f ~ %.2f" % (float(st["action"]["min"].flatten()[5]), float(st["action"]["max"].flatten()[5])))
w("- 每 epoch 步数(batch=8, 训练 40 条): 约 %d" % (int(ds.num_frames * 0.8 / 8) + 1))
w("- 40k 步折合 epoch: 约 %.1f" % (40000 / (ds.num_frames * 0.8 / 8)))
w()

os.makedirs("docs", exist_ok=True)
with open("docs/05_数据统计报告.md", "w") as f:
    f.write("\n".join(L) + "\n")
print("已保存: docs/05_数据统计报告.md")
