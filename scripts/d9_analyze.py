"""留出集逐帧分析：失败归因、重规划间隔权衡、模型间配对检验。

Input:
    d9_record.py 产出的 npz。配对比较要求各文件来自同一批帧、同一顺序、同一 seed。

Output:
    --out 指定 JSON，含三段：
      failure  误差与候选特征的相关性、分位对比、按任务阶段的分布、最差帧列表
      replan   各重规划间隔 k 下的控制误差与动作抖动
      paired   同帧差值的 bootstrap 95% CI、p 值、配对效应量

Usage:
    python scripts/d9_analyze.py --run A40k=a.npz --run base=b.npz --main A40k --out r.json

Notes:
    - replan 假设动作块内部开环执行，与 LeRobot 部署行为一致。
    - k=1 的控制误差应等于逐帧 l1_step0，可用于自检实现是否偏离口径。
"""
import argparse, json, os
import numpy as np
from scipy import stats


def load(path):
    """读取 npz 为 dict。

    Returns:
        dict：键与 npz 内数组名一致（chunk / gt / meta / img / tag）。
    """
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def ep_key_index(meta):
    """(episode, t_in_ep) -> 行号。

    用于 O(1) 定位同段的前一帧，避免每帧都做一次全表扫描。
    """
    """(episode, t_in_ep) -> 行号"""
    return {(int(m[1]), int(m[2])): i for i, m in enumerate(meta)}


def ep_lengths(meta):
    """各 episode 的帧数。

    Returns:
        dict: episode -> 帧数（取该段内最大 t_in_ep + 1）。
    """
    out = {}
    for m in meta:
        out[int(m[1])] = max(out.get(int(m[1]), 0), int(m[2]) + 1)
    return out


def failure_analysis(run, tag, fps=30.0, topn=20):
    """误差归因：相关性、分位对比、任务阶段分布、最差帧。

    Args:
        run: load() 的返回值。
        tag: 结果标签，仅用于输出。
        fps: 视频帧率，用于把帧序号换算成秒。
        topn: 输出最差帧的条数。

    Returns:
        dict，对应 JSON 的 failure 段。

    Notes:
        Spearman 只反映单调相关，不能解释为因果；阶段与速度可能互相混杂，
        故同时给出分位对比与阶段分布两种视角。
    """
    meta, gt, img = run["meta"], run["gt"], run["img"]
    l1 = meta[:, 3]
    ep = meta[:, 1].astype(int)
    t = meta[:, 2].astype(int)
    lens = ep_lengths(meta)

    # 特征1：真值动作的「速度」= 同一帧内前两步动作差（模型要预判的即时变化量）
    vel = np.linalg.norm(gt[:, 1] - gt[:, 0], axis=1)
    # 特征2：帧间速度 = 与上一帧真值动作的差（真实机械臂运动快慢）
    pos = ep_key_index(meta)
    frame_vel = np.zeros_like(l1)
    for i in range(len(meta)):
        j = pos.get((int(ep[i]), int(t[i]) - 1))
        if j is not None:
            frame_vel[i] = np.linalg.norm(gt[i, 0] - gt[j, 0])
    # 特征3：动作幅值（离零多远，反映姿态是否极端）
    mag = np.linalg.norm(gt[:, 0], axis=1)
    # 特征4：任务阶段
    tnorm = np.array([t[i] / max(1, lens[ep[i]] - 1) for i in range(len(meta))])
    # 特征5：夹爪变化（夹爪在动 = 抓取/放置时刻）
    grip = np.abs(gt[:, 1, 5] - gt[:, 0, 5])
    # 图像特征
    up_mean, up_std, sd_mean, sd_std = img[:, 0], img[:, 1], img[:, 2], img[:, 3]

    feats = [
        ("动作速度 |a(t+1)-a(t)|", vel),
        ("帧间速度 |a(t)-a(t-1)|", frame_vel),
        ("动作幅值 |a(t)|", mag),
        ("任务阶段 t/T", tnorm),
        ("夹爪变化 |Δgripper|", grip),
        ("图像亮度 up", up_mean),
        ("图像对比度 up", up_std),
        ("图像亮度 side", sd_mean),
    ]

    rows = []
    for name, f in feats:
        r, p = stats.spearmanr(f, l1)
        rows.append({"feature": name, "spearman_r": round(float(r), 3),
                     "p_value": float(p), "n": int(len(l1))})

    # 分位对比：最高/最低 20%
    detal = []
    for name, f in feats:
        q80, q20 = np.quantile(f, 0.8), np.quantile(f, 0.2)
        hi = l1[f >= q80]
        lo = l1[f <= q20]
        detal.append({"feature": name,
                      "l1_top20": round(float(hi.mean()), 4),
                      "l1_bottom20": round(float(lo.mean()), 4),
                      "ratio": round(float(hi.mean() / max(1e-9, lo.mean())), 3)})

    # 按任务阶段分 10 段
    dec = np.clip((tnorm * 10).astype(int), 0, 9)
    by_phase = []
    for d in range(10):
        m = dec == d
        if m.sum():
            by_phase.append({"decile": d, "range": "%.0f-%.0f%%" % (d * 10, (d + 1) * 10),
                             "mean_l1": round(float(l1[m].mean()), 4), "n": int(m.sum())})

    # 最差的帧
    order = np.argsort(-l1)[:topn]
    worst = []
    for i in order:
        worst.append({"global_idx": int(meta[i, 0]), "episode": int(ep[i]), "t": int(t[i]),
                      "t_norm": round(float(tnorm[i]), 3), "l1_step0": round(float(l1[i]), 3),
                      "frame_vel": round(float(frame_vel[i]), 3),
                      "up_mean": round(float(up_mean[i]), 3), "up_std": round(float(up_std[i]), 3)})

    print("=" * 74)
    print("【失败归因】%s   帧数=%d   单步L1 均值=%.4f" % (tag, len(l1), l1.mean()))
    print("-" * 74)
    print("%-26s %10s %12s   %s" % ("特征", "Spearman", "p值", "说明"))
    for r in rows:
        sig = "显著" if r["p_value"] < 0.001 else ("弱" if r["p_value"] < 0.05 else "不显著")
        print("%-26s %10.3f %12.2e   %s" % (r["feature"], r["spearman_r"], r["p_value"], sig))
    print("-" * 74)
    print("%-26s %11s %11s %8s" % ("特征（分位对比）", "最高20%L1", "最低20%L1", "倍率"))
    for r in detal:
        print("%-26s %11.4f %11.4f %7.2fx" % (r["feature"], r["l1_top20"], r["l1_bottom20"], r["ratio"]))
    print("-" * 74)
    print("按任务阶段（t/T 十分位）：")
    for r in by_phase:
        bar = "#" * int(r["mean_l1"] / max(0.1, max(x["mean_l1"] for x in by_phase)) * 40)
        print("  %-8s %s  L1=%.3f" % (r["range"], bar, r["mean_l1"]))
    print("-" * 74)
    print("误差最大的 %d 帧：" % topn)
    for w in worst[:10]:
        print("  ep%-3d t=%-4d (%.0f%%)  L1=%6.3f  帧间速度=%5.2f  亮度=%.3f"
              % (w["episode"], w["t"], w["t_norm"] * 100, w["l1_step0"], w["frame_vel"], w["up_mean"]))

    return {"tag": tag, "n_frames": int(len(l1)), "mean_l1": round(float(l1.mean()), 4),
            "correlations": rows, "quantile_compare": detal,
            "by_phase": by_phase, "worst_frames": worst}


def replan_sweep(run, tag, ks, fps=30.0, infer_ms=225.0):
    """模拟「每 k 帧重新规划一次」，统计控制误差与动作抖动。

    Args:
        run: load() 的返回值。
        ks: 待测的重规划间隔（单位：帧）。
        infer_ms: 单次推理耗时，用于判断 k 是否满足实时约束（k * 帧间隔 >= infer_ms）。

    Returns:
        list[dict]：每项含 control_l1 / jerk / step_jump / replan_hz / realtime_ok。

    Notes:
        k 超过动作块长度时该窗口无可用动作，直接跳过。
    """
    """同一模型，执行前 k 步再重规划：控制误差 + 抖动"""
    meta, chunk, gt = run["meta"], run["chunk"], run["gt"]
    idx_of = ep_key_index(meta)
    lens = ep_lengths(meta)
    out = []

    for k in ks:
        errs, jerks, jumps = [], [], []
        for ep, n in lens.items():
            ex = []
            for t in range(n):
                j0 = (t // k) * k
                row = idx_of.get((ep, j0))
                if row is None:       # stride>1 时缺少该决策帧
                    row = idx_of.get((ep, t))
                    if row is None:
                        continue
                    off = 0
                else:
                    off = t - j0
                if off >= chunk.shape[1]:
                    continue
                ex.append(chunk[row][off])
                g = idx_of.get((ep, t))
                if g is not None:
                    errs.append(np.abs(chunk[row][off] - gt[g][0]).mean())
            ex = np.asarray(ex)
            if len(ex) >= 5:
                # jerk：三阶差分（单位：动作单位/帧^3）
                j = ex[3:] - 3 * ex[2:-1] + 3 * ex[1:-2] - ex[:-3]
                jerks.append(np.abs(j).mean())
                # 相邻步跳变（一阶差分，反映不平滑）
                jumps.append(np.abs(np.diff(ex, axis=0)).mean())

        if not errs:
            continue
        feas = k * (1000.0 / fps) >= infer_ms
        out.append({"k": int(k), "control_l1": round(float(np.mean(errs)), 4),
                    "jerk": round(float(np.mean(jerks)), 4),
                    "step_jump": round(float(np.mean(jumps)), 4),
                    "replan_hz": round(fps / k, 2),
                    "budget_ms": round(k * 1000.0 / fps, 1),
                    "realtime_ok": bool(feas)})

    print("=" * 74)
    print("【重规划间隔权衡】%s  （推理耗时 %.0fms，帧间隔 %.1fms）" % (tag, infer_ms, 1000.0 / fps))
    print("-" * 74)
    print("%4s %10s %10s %10s %9s %8s" % ("k", "控制L1", "jerk", "步间跳变", "重规划Hz", "实时?"))
    best_k, best_l1 = None, 1e9
    for r in out:
        print("%4d %10.4f %10.4f %10.4f %9.2f %8s"
              % (r["k"], r["control_l1"], r["jerk"], r["step_jump"], r["replan_hz"],
                 "OK" if r["realtime_ok"] else "NO"))
        if r["control_l1"] < best_l1:
            best_l1, best_k = r["control_l1"], r["k"]
    print("  → 控制误差最低的 k = %d（L1=%.4f）；理论下限 k≈%.1f（推理 %.0fms / 帧间隔 %.1fms）"
          % (best_k, best_l1, infer_ms / (1000.0 / fps), infer_ms, 1000.0 / fps))
    return out


def paired_test(a, b, tag_a, tag_b, n_boot=5000, seed=0):
    """同帧配对检验：比较两个模型的单步误差差（b - a，负值表示 b 更好）。

    Args:
        a, b: 两个 run，npz 顺序必须一致。
        n_boot: bootstrap 重采样次数。
        seed: 重采样随机种子，保证结果可复现。

    Returns:
        dict：diff、ci95、p_bootstrap、p_ttest、cohens_dz。

    Notes:
        逐帧误差方差很大，均值差的显著性必须用配对设计来消除帧难度差异。
    """
    """同帧配对比较：b - a（负值=b 更好）"""
    la, lb = a["meta"][:, 3], b["meta"][:, 3]
    n = min(len(la), len(lb))
    d = lb[:n] - la[:n]
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(d, n, replace=True).mean() for _ in range(n_boot)])
    ci = np.percentile(boots, [2.5, 97.5])
    # 双侧 p：bootstrap 分布里 0 以外的比例
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    p = float(min(1.0, p))
    tstat, tp = stats.ttest_rel(lb[:n], la[:n])
    dz = d.mean() / (d.std(ddof=1) + 1e-12)
    print("=" * 74)
    print("【配对检验】%s vs %s   同一批 %d 帧" % (tag_b, tag_a, n))
    print("  %s 单步L1 = %.4f ± %.3f" % (tag_a, la[:n].mean(), la[:n].std() / np.sqrt(n)))
    print("  %s 单步L1 = %.4f ± %.3f" % (tag_b, lb[:n].mean(), lb[:n].std() / np.sqrt(n)))
    print("  差值(%s-%s) = %+.4f  bootstrap 95%%CI [%+.4f, %+.4f]  p=%.3g  配对Cohen's d=%.2f"
          % (tag_b, tag_a, d.mean(), ci[0], ci[1], p, dz))
    print("  → %s" % ("差异显著" if p < 0.05 else "差异不显著"))
    return {"a": tag_a, "b": tag_b, "n": int(n),
            "mean_a": round(float(la[:n].mean()), 4), "mean_b": round(float(lb[:n].mean()), 4),
            "diff": round(float(d.mean()), 4),
            "ci95": [round(float(ci[0]), 4), round(float(ci[1]), 4)],
            "p_bootstrap": p, "p_ttest": float(tp), "cohens_dz": round(float(dz), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[], help="tag=path，可多次")
    ap.add_argument("--main", default="", help="做失败归因与重规划分析的 tag")
    ap.add_argument("--ks", default="1,2,3,5,7,10,15,25,50")
    ap.add_argument("--infer-ms", type=float, default=225.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    runs = {}
    for spec in args.run:
        tag, path = spec.split("=", 1)
        runs[tag] = load(path)
        print("载入 %-6s %s  帧数=%d" % (tag, path, len(runs[tag]["meta"])))

    ks = [int(x) for x in args.ks.split(",")]
    main_tag = args.main or list(runs)[0]

    res = {"runs": list(runs), "main": main_tag}
    res["failure"] = failure_analysis(runs[main_tag], main_tag)
    res["replan"] = {t: replan_sweep(runs[t], t, ks, infer_ms=args.infer_ms) for t in runs}

    tags = list(runs)
    if len(tags) >= 2:
        res["paired"] = []
        base = tags[0]
        for t in tags[1:]:
            res["paired"].append(paired_test(runs[base], runs[t], base, t))

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print("\n已保存:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
