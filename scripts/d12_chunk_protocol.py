"""动作块 L1 的口径对照：把「帧数」「padding」「聚合方式」三个因素拆开。

问题：
    d11_vision_ablation 与 d10_baseline 算出的动作块 L1 不能直接比较，
    因为前者不排除 action_is_pad、按 (帧, 步) 全部有效对求平均；
    后者排除越界项、按预测步长等权后跨帧平均。差异曾被错误地全部归因于「抽样帧数」。

做法：
    用同一份 npz（同一模型、同一批 2759 帧），穷举
        帧集   : 全量 2759 / stride 5 抽样 552
        padding : 不排除 / 排除 t+k >= 段长
        聚合   : 所有有效 (帧,步) 对等权 / 每一步单独求均值再等权跨步
    得到 2x2x2 的表，各因素的分量一目了然。

Usage:
    python scripts/d12_chunk_protocol.py --run A40k=results/perframe/A40k.npz \
        --out results/chunk_protocol.json

Output:
    JSON + 屏幕表格：每种口径下的动作块 L1

Notes:
    - 纯 CPU，只读 npz，秒级
    - 「段长」由 meta 的 (episode, t) 推出，不依赖外部 meta 文件
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="A40k=results/perframe/A40k.npz")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--out", default="results/chunk_protocol.json")
    args = ap.parse_args()

    tag, path = args.run.split("=", 1)
    d = np.load(path)
    chunk, gt, meta = d["chunk"], d["gt"], d["meta"]
    ep, t = meta[:, 1].astype(int), meta[:, 2].astype(int)

    lengths = {e: int(t[ep == e].max()) + 1 for e in np.unique(ep)}
    n = np.array([lengths[e] for e in ep])
    C = chunk.shape[1]

    err = np.abs(chunk - gt).mean(axis=2)                      # (N, C) 已在 6 个关节上取均值
    valid = np.stack([(t + k) < n for k in range(C)], axis=1)    # (N, C) 布尔

    rows_full = np.arange(len(ep))
    rows_sub = np.arange(0, len(ep), args.stride)

    def agg(rows, use_mask, equal_k):
        e, v = err[rows], valid[rows]
        if use_mask:
            e = np.where(v, e, np.nan)
            if equal_k:
                per_k = np.nanmean(e, axis=0)
                return float(np.nanmean(per_k))
            return float(np.nanmean(e))
        if equal_k:
            return float(np.mean(e.mean(axis=0)))
        return float(np.mean(e))

    table = {}
    for name, rows in (("full_2759", rows_full), ("stride5_552", rows_sub)):
        for use_mask in (False, True):
            for eq in (False, True):
                key = "%s|pad=%s|agg=%s" % (name, "excluded" if use_mask else "included",
                                            "equal_k" if eq else "pair_weighted")
                table[key] = agg(rows, use_mask, eq)

    print("模型 %s，共 %d 帧" % (tag, len(ep)))
    print("%-14s %-14s %-14s %10s" % ("帧集", "padding", "聚合", "动作块 L1"))
    for name in ("full_2759", "stride5_552"):
        for use_mask in (False, True):
            for eq in (False, True):
                key = "%s|pad=%s|agg=%s" % (name, "excluded" if use_mask else "included",
                                            "equal_k" if eq else "pair_weighted")
                print("%-14s %-14s %-14s %10.3f"
                      % (name, "排除" if use_mask else "包含",
                         "按步等权" if eq else "按对等权", table[key]))

    base = table["stride5_552|pad=included|agg=pair_weighted"]     # 旧 d11 口径
    tgt = table["full_2759|pad=excluded|agg=equal_k"]              # 旧 d10 口径
    print("\n旧 d11 口径 = %.3f ；旧 d10 口径 = %.3f ；差 %.3f" % (base, tgt, tgt - base))
    print("其中：只换帧集 %+.3f ；只换 padding/聚合 %+.3f"
          % (table["full_2759|pad=included|agg=pair_weighted"] - base,
             tgt - table["full_2759|pad=included|agg=pair_weighted"]))

    out = {"run": tag, "frames": int(len(ep)), "table": table,
           "unified": tgt,
           "unified_definition": "全量帧 + 排除 t+k>=段长 + 按预测步长等权"}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n已写入", args.out)


if __name__ == "__main__":
    main()
