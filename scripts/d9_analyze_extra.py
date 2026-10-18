"""补充分析：亮度与运动速度的联合效应、数值分布、分阶段差异。

在 d9_analyze.py 的单变量相关性之外，检查两个特征是否互相混杂：
按「亮度三分位 x 速度三分位」交叉分组，观察误差是否只在特定组合下升高。

Usage:
    python scripts/d9_analyze_extra.py
"""
import json
import numpy as np
from scipy import stats


def load(p):
    """读取 npz 为 dict（与 d9_analyze.load 一致）。"""
    d = np.load(p, allow_pickle=True)
    return {k: d[k] for k in d.files}


def main():
    A = load("results/perframe/A40k.npz")
    C = load("results/perframe/C40k.npz")
    B = load("results/perframe/base.npz")

    meta, gt, img = A["meta"], A["gt"], A["img"]
    l1 = meta[:, 3]
    bright, contrast = img[:, 0], img[:, 1]

    vel = np.zeros_like(l1)
    pos = {(int(m[1]), int(m[2])): i for i, m in enumerate(meta)}
    for i in range(len(meta)):
        j = pos.get((int(meta[i, 1]), int(meta[i, 2]) - 1))
        if j is not None:
            vel[i] = np.linalg.norm(gt[i, 0] - gt[j, 0])
    # 帧间速度 = 0 的（每段首帧）剔除
    keep = vel > 0
    l1k, velk, brk, ctk = l1[keep], vel[keep], bright[keep], contrast[keep]

    def q(x):
        return [round(float(v), 3) for v in np.quantile(x, [0, .1, .25, .5, .75, .9, 1.0])]

    print("=" * 74)
    print("【分布】%d 帧（剔除每段首帧）" % len(l1k))
    print("  亮度 up  分位(0/10/25/50/75/90/100):", q(brk))
    print("  对比度 up 分位:", q(ctk))
    print("  帧间速度  分位:", q(velk))
    print("  亮度 vs 对比度 Spearman: %.3f" % stats.spearmanr(brk, ctk)[0])
    print("  亮度 vs 帧间速度 Spearman: %.3f" % stats.spearmanr(brk, velk)[0])

    # 3x3 联合表：亮度三分位 x 速度三分位
    bt = np.quantile(brk, [1 / 3, 2 / 3])
    vt = np.quantile(velk, [1 / 3, 2 / 3])
    bi = np.digitize(brk, bt)
    vi = np.digitize(velk, vt)
    print("-" * 74)
    print("【联合效应】单步 L1（行=亮度三分位，列=速度三分位）")
    print("%14s %12s %12s %12s" % ("", "慢速", "中速", "快速"))
    for b in range(3):
        row = []
        for v in range(3):
            m = (bi == b) & (vi == v)
            row.append(l1k[m].mean() if m.sum() else float("nan"))
        lab = ["暗（低1/3）", "中", "亮（高1/3）"][b]
        print("%14s %12.3f %12.3f %12.3f" % (lab, row[0], row[1], row[2]))
    print("  → 最差组合 / 最好组合 = %.2fx"
          % (max(l1k[(bi == b) & (vi == v)].mean() for b in range(3) for v in range(3)
                 if ((bi == b) & (vi == v)).sum()) /
             min(l1k[(bi == b) & (vi == v)].mean() for b in range(3) for v in range(3)
                 if ((bi == b) & (vi == v)).sum())))

    print("-" * 74)
    print("【亮度分档】")
    edges = [0, 0.55, 0.60, 0.65, 0.70, 1.0]
    for i in range(len(edges) - 1):
        m = (brk >= edges[i]) & (brk < edges[i + 1])
        if m.sum():
            print("  亮度 %.2f-%.2f  n=%4d   L1=%.3f" % (edges[i], edges[i + 1], m.sum(), l1k[m].mean()))

    # A vs C 配对
    print("=" * 74)
    la, lc, lb = A["meta"][:, 3], C["meta"][:, 3], B["meta"][:, 3]
    n = min(len(la), len(lc))
    rng = np.random.default_rng(0)
    for name, x, y in [("C40k - A40k", la[:n], lc[:n])]:
        d = y - x
        boots = np.array([rng.choice(d, n, replace=True).mean() for _ in range(5000)])
        ci = np.percentile(boots, [2.5, 97.5])
        p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
        print("【配对检验】%s   n=%d" % (name, n))
        print("  A40k=%.4f  C40k=%.4f  diff=%+.4f  95%%CI[%+.4f,%+.4f]  p=%.3g  dz=%.2f"
              % (x.mean(), y.mean(), d.mean(), ci[0], ci[1], min(1.0, p),
                 d.mean() / (d.std(ddof=1) + 1e-12)))
        # 分阶段看差异
        eps = A["meta"][:, 1].astype(int)
        t = A["meta"][:, 2].astype(int)
        lens = {}
        for e, tt in zip(eps, t):
            lens[e] = max(lens.get(e, 0), tt + 1)
        tn = np.array([t[i] / max(1, lens[eps[i]] - 1) for i in range(len(eps))])
        print("  分阶段差异（C40k - A40k）：")
        for d10 in range(5):
            m = (tn >= d10 * 0.2) & (tn < (d10 + 1) * 0.2)
            if m.sum():
                print("    %3.0f-%3.0f%%  A=%.3f  C=%.3f  diff=%+.3f  (n=%d)"
                      % (d10 * 20, (d10 + 1) * 20, la[:n][m[:n]].mean(),
                         lc[:n][m[:n]].mean(), (lc[:n] - la[:n])[m[:n]].mean(), m[:n].sum()))

    # 汇总写 json
    out = {
        "n_frames": int(len(l1k)),
        "brightness_quantiles": q(brk),
        "contrast_quantiles": q(ctk),
        "velocity_quantiles": q(velk),
        "corr_bright_contrast": round(float(stats.spearmanr(brk, ctk)[0]), 3),
        "corr_bright_velocity": round(float(stats.spearmanr(brk, velk)[0]), 3),
        "joint_table": [[round(float(l1k[(bi == b) & (vi == v)].mean()), 3) for v in range(3)]
                        for b in range(3)],
        "by_brightness_band": [],
    }
    for i in range(len(edges) - 1):
        m = (brk >= edges[i]) & (brk < edges[i + 1])
        if m.sum():
            out["by_brightness_band"].append(
                {"range": [edges[i], edges[i + 1]], "n": int(m.sum()),
                 "l1": round(float(l1k[m].mean()), 3)})
    with open("results/perframe/analysis_extra.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n已保存: results/perframe/analysis_extra.json")


if __name__ == "__main__":
    main()
