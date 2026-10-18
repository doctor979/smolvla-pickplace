"""配对检验的两种重采样单位，以及种子方差。

问题：
    d9_analyze.py 按「帧」重采样。相邻帧高度相关（误差成段出现），
    等于把有效样本量高估数十倍，区间偏窄。

做法：
    同一批数据给两套区间——
      frame     : 逐帧有放回重采样（旧口径，保留以便对照）
      episode   : 按整条 episode 重采样（新口径，作为主结果）
    另外若提供多个种子，报告种子间的均值与标准差。

Usage:
    python scripts/d10_stats.py \
        --runs base=results/perframe/base.npz,A40k=results/perframe/A40k.npz \
        --seed-runs A40k=results/perframe/A40k.npz,A40k_s1001=results/perframe/A40k_seed1001.npz \
        --seed-ref base=results/perframe/base.npz \
        --out results/stats_episode.json

Output:
    JSON：每对比较的 diff 与两套 95% CI；种子组内各模型的均值、标准差与相对基线的差值

Notes:
    - 只读 npz，CPU 数十秒；n_boot 默认 5000
    - 配对前提：两个 npz 的帧顺序完全一致，脚本会断言
"""
import argparse
import json
import os
import sys

import numpy as np


def load(path):
    d = np.load(path)
    l1 = np.abs(d["chunk"][:, 0] - d["gt"][:, 0]).mean(axis=1)   # 每帧单步 L1
    ep = d["meta"][:, 1].astype(int)
    return l1, ep


def boot_frame(a, b, n_boot, rng):
    n = len(a)
    out = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.integers(0, n, n)
        out[i] = b[s].mean() - a[s].mean()
    return out


def boot_episode(a, b, ep, n_boot, rng):
    """按 episode 整段重采样：抽中的整条（含其全部帧）一起进来。"""
    eps = np.unique(ep)
    groups = [np.where(ep == e)[0] for e in eps]
    out = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(eps), len(eps))
        idx = np.concatenate([groups[j] for j in pick])
        out[i] = b[idx].mean() - a[idx].mean()
    return out


def summarise(samples, point):
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return {"diff": float(point), "ci95": [float(lo), float(hi)],
            "width": float(hi - lo),
            "p_two_sided": float(2 * min((samples >= 0).mean(), (samples <= 0).mean()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="tag=npz，第一个作为参照")
    ap.add_argument("--seed-runs", default="", help="种子组：tag=npz")
    ap.add_argument("--seed-ref", default="", help="种子组的参照 run：tag=npz")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/stats_episode.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    runs = {}
    for item in args.runs.split(","):
        tag, path = item.split("=", 1)
        if not os.path.exists(path):
            print("跳过（不存在）:", path)
            continue
        runs[tag] = load(path)
        print("载入 %-12s %d 帧 / %d 段" % (tag, len(runs[tag][0]), len(np.unique(runs[tag][1]))))

    if len(runs) < 2:
        sys.exit("至少需要两个 run")

    ref_tag = list(runs)[0]
    a, ep = runs[ref_tag]
    out = {"n_frames": int(len(a)), "n_episodes": int(len(np.unique(ep))), "comparisons": {}}

    print("\n%-22s %9s %-20s %-20s %6s" % ("比较", "差值", "帧级 95% CI", "episode 级 95% CI", "变宽"))
    for tag, (b, ep_b) in runs.items():
        if tag == ref_tag:
            continue
        if not np.array_equal(ep, ep_b):
            print("  !! %s 的 episode 序列与参照不一致，跳过" % tag)
            continue
        s_f = boot_frame(a, b, args.n_boot, rng)
        s_e = boot_episode(a, b, ep, args.n_boot, rng)
        r_f = summarise(s_f, b.mean() - a.mean())
        r_e = summarise(s_e, b.mean() - a.mean())
        out["comparisons"]["%s>%s" % (ref_tag, tag)] = {"frame": r_f, "episode": r_e}
        print("%-22s %9.4f [%7.3f,%7.3f]      [%7.3f,%7.3f]   %5.2fx"
              % ("%s → %s" % (ref_tag, tag), b.mean() - a.mean(),
                 r_f["ci95"][0], r_f["ci95"][1], r_e["ci95"][0], r_e["ci95"][1],
                 r_e["width"] / r_f["width"]))

    # 种子方差
    if args.seed_runs and args.seed_ref:
        ref_path = args.seed_ref.split("=", 1)[1]
        base, _ = load(ref_path)
        seeds = {}
        for item in args.seed_runs.split(","):
            tag, path = item.split("=", 1)
            if not os.path.exists(path):
                print("  种子缺失，跳过:", path)
                continue
            l1, _ = load(path)
            seeds[tag] = {"mean_l1": float(l1.mean()), "diff_vs_base": float(l1.mean() - base.mean())}
        if len(seeds) >= 1:
            vals = [v["mean_l1"] for v in seeds.values()]
            diffs = [v["diff_vs_base"] for v in seeds.values()]
            out["seeds"] = {
                "per_run": seeds,
                "n": len(vals),
                "mean_l1": float(np.mean(vals)),
                "sd_l1": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                "diff_mean": float(np.mean(diffs)),
                "diff_sd": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else None,
            }
            print("\n种子组（n=%d）" % len(vals))
            for tag, v in seeds.items():
                print("  %-16s L1=%.4f  Δvs base=%+.4f" % (tag, v["mean_l1"], v["diff_vs_base"]))
            if len(vals) > 1:
                print("  %-16s L1=%.4f ± %.4f   Δ=%+.4f ± %.4f"
                      % ("均值±标准差", out["seeds"]["mean_l1"], out["seeds"]["sd_l1"],
                         out["seeds"]["diff_mean"], out["seeds"]["diff_sd"]))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n已写入", args.out)


if __name__ == "__main__":
    main()
