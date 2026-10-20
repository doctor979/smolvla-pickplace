"""种子方差分析：同一配置换随机种子重训后，指标波动有多大；(C − A) 的差值是否稳定。

问题：
    主线只有一个种子（1000）。「C 比 A 好约 3%」这个差异可能只是训练随机性。
    补充种子后要回答两件事：
      1. 同一配置内，种子之间差多少（组内离散度）
      2. (C − A) 的差值在各 seed 下符号和幅度是否一致（增强效应是否稳定）

口径（与主表一致）：
    单步 L1  = k=0 的平均 |pred − gt|，不涉及越界，无需掩码
    动作块 L1 = k=1..49，只统计 t+k 仍在同一 episode 内的步，在 6 个关节上取均值后
                **按预测步长等权**（每步先平均、再对步平均）——与 d12 的
                "full_2759|pad=excluded|agg=equal_k" 同口径。
    因此本脚本算出的 seed 1000 数值即为 README 主表中的 4.729 与 9.014。

Usage:
    python scripts/d13_seed_variance.py \
        --runs A40k=results/perframe/A40k.npz,A40k_s1001=results/perframe/A40k_seed1001.npz,C40k=...,... \
        --out results/seed_variance.json

Output:
    JSON：逐 run 指标、按配置分组的均值/极差、同 seed 配对差值 (C−A)

Notes:
    - 纯 CPU，只读 npz
    - run 命名约定：`A40k` 为 seed 1000，`A40k_s1001` 为 seed 1001；
      脚本按 `A`/`C` 前缀归类，按 `_s<数字>` 解析种子
"""
import argparse
import json
import os
import re

import numpy as np


def metrics(path):
    """返回 (单步 L1, 动作块 L1 k=1..49 池化, 帧数)。"""
    d = np.load(path)
    chunk, gt, meta = d["chunk"], d["gt"], d["meta"]
    ep, t = meta[:, 1].astype(int), meta[:, 2].astype(int)

    lengths = {e: int(t[ep == e].max()) + 1 for e in np.unique(ep)}
    n = np.array([lengths[e] for e in ep])
    C = chunk.shape[1]

    err = np.abs(chunk - gt).mean(axis=2)                       # (N, C)
    step0 = float(err[:, 0].mean())                             # k=0 不越界

    ks = list(range(1, min(C, n.max())))                        # k=1..49
    mask = np.stack([(t + k) < n for k in ks], axis=1)          # (N, K)
    blk = err[:, 1:1 + mask.shape[1]]
    # 按预测步长等权：每一步单独求均值，再对步平均。与 d10_baseline 的 equal_k 口径一致，
    # 因此 seed 1000 应复现主表的 9.014。不按 (帧,步) 对池化——那样等于给有效步多的
    # 长 episode 更大权重，与主表不可比。
    per_k = [float(blk[mask[:, j], j].mean()) for j in range(mask.shape[1])]
    chunk_l1 = float(np.mean(per_k))

    return step0, chunk_l1, int(len(ep))


def kind_of(tag):
    m = re.match(r"^([AC])", tag, re.I)
    return m.group(1).upper() if m else "?"


def seed_of(tag):
    m = re.search(r"_s(\d+)$", tag)
    return int(m.group(1)) if m else 1000


def summarize(vals):
    a = np.array(vals, dtype=float)
    return {
        "n": int(len(a)),
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if len(a) > 1 else None,
        "min": float(a.min()),
        "max": float(a.max()),
        "range": float(a.max() - a.min()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True,
                    help="逗号分隔，格式 tag=path，例如 A40k=results/perframe/A40k.npz")
    ap.add_argument("--out", default="results/seed_variance.json")
    args = ap.parse_args()

    runs = {}
    for item in args.runs.split(","):
        tag, path = item.split("=", 1)
        if not os.path.exists(path):
            print("跳过缺失文件", path)
            continue
        s0, ch, nf = metrics(path)
        runs[tag] = {"kind": kind_of(tag), "seed": seed_of(tag),
                     "step0": s0, "chunk": ch, "frames": nf}

    print("%-14s %-5s %-6s %10s %10s" % ("run", "配置", "种子", "单步 L1", "动作块 L1"))
    for tag in sorted(runs, key=lambda x: (runs[x]["kind"], runs[x]["seed"])):
        r = runs[tag]
        print("%-14s %-5s %-6d %10.4f %10.4f"
              % (tag, r["kind"], r["seed"], r["step0"], r["chunk"]))

    groups = {}
    for kind in ("A", "C"):
        sel = [r for r in runs.values() if r["kind"] == kind]
        if not sel:
            continue
        groups[kind] = {
            "step0": summarize([r["step0"] for r in sel]),
            "chunk": summarize([r["chunk"] for r in sel]),
            "seeds": sorted(r["seed"] for r in sel),
        }
        g = groups[kind]
        print("\n配置 %s：%d 个种子 %s" % (kind, g["step0"]["n"], g["seeds"]))
        print("  单步 L1    %.4f ± %.4f  [%.4f, %.4f]  极差 %.4f"
              % (g["step0"]["mean"], g["step0"]["std"] or 0,
                 g["step0"]["min"], g["step0"]["max"], g["step0"]["range"]))
        print("  动作块 L1  %.4f ± %.4f  [%.4f, %.4f]  极差 %.4f"
              % (g["chunk"]["mean"], g["chunk"]["std"] or 0,
                 g["chunk"]["min"], g["chunk"]["max"], g["chunk"]["range"]))

    # 同 seed 配对：只看 A 与 C 都有种子的情况
    paired = []
    for s in sorted({r["seed"] for r in runs.values()}):
        a = next((r for r in runs.values() if r["kind"] == "A" and r["seed"] == s), None)
        c = next((r for r in runs.values() if r["kind"] == "C" and r["seed"] == s), None)
        if a and c:
            paired.append({
                "seed": s,
                "A_chunk": a["chunk"], "C_chunk": c["chunk"],
                "delta_chunk": c["chunk"] - a["chunk"],
                "A_step0": a["step0"], "C_step0": c["step0"],
                "delta_step0": c["step0"] - a["step0"],
            })

    delta = {}
    if paired:
        print("\n同 seed 配对 (C − A)，负值表示增强更好：")
        print("%-6s %10s %10s %12s" % ("seed", "A", "C", "Δ(C−A)"))
        for p in paired:
            print("%-6d %10.4f %10.4f %+12.4f"
                  % (p["seed"], p["A_chunk"], p["C_chunk"], p["delta_chunk"]))
        ds = [p["delta_chunk"] for p in paired]
        delta = {
            "chunk": summarize(ds),
            "sign_consistent": bool(all(d < 0 for d in ds) or all(d > 0 for d in ds)),
            "seeds": [p["seed"] for p in paired],
        }
        print("  Δ 均值 %+.4f，范围 [%+.4f, %+.4f]，符号一致：%s"
              % (delta["chunk"]["mean"], delta["chunk"]["min"],
                 delta["chunk"]["max"], "是" if delta["sign_consistent"] else "否"))
        if len(ds) == 1:
            print("  注意：只有一个配对种子，无法判断稳定性，需要至少 2 个")

    out = {"runs": runs, "groups": groups, "paired": paired, "delta": delta}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n已写入", args.out)


if __name__ == "__main__":
    main()
