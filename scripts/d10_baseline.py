"""平凡基线：不训练，直接用「当前状态」或「上一条动作」作为预测，与模型对比。

动机：
    动作是绝对关节角，30 fps 下相邻帧位移很小，因此「预测下一帧 = 当前姿态」
    在单步 L1 上天然占优。若不与之对照，单步 L1 会系统性高估策略质量。

两种平凡基线：
    state_hold  : predict(t+k) = observation.state(t)      —— 把当前观测当动作
    action_hold : predict(t+k) = action(t)                 —— 保持上一条指令不变

口径（与 d9_record.py 一致）：
    - 帧集：留出集 10 条 episode 的全部帧（2759 帧，stride=1）
    - 多步指标排除 episode 末尾越界项（t+k >= 该段长度）

Usage:
    python scripts/d10_baseline.py --ds-root <快照> \
        --runs base=results/perframe/base.npz,A40k=results/perframe/A40k.npz \
        --out results/baseline_static.json

Output:
    JSON：逐 k 的基线与各模型 L1、两种聚合口径（按 k 等权 / 按有效帧对加权）

Notes:
    - 只读 parquet 的状态与动作，不解码视频，CPU 数十秒
    - 某些 run 的 chunk_size 小于 50，记录时用 0 补齐（gt 同样被补齐）；
      这类 k 上 L1 恒为 0，必须排除，否则会得到「短块模型多步误差极小」的荒谬结论
    - 脚本会自动探测每个 run 的有效 k 上限并写入 JSON
"""
import argparse
import glob
import json
import os
import sys

import numpy as np


def load_holdout(ds_root, eps):
    """按 (episode, frame) 顺序读出留出集的 state 与 action。"""
    import pyarrow.parquet as pq

    files = sorted(glob.glob(os.path.join(ds_root, "data", "*", "*.parquet")))
    if not files:
        sys.exit("找不到 parquet：%s" % ds_root)

    st, ac, epi, fr = [], [], [], []
    for f in files:
        t = pq.read_table(f, columns=["observation.state", "action",
                                      "episode_index", "frame_index"])
        e = np.asarray(t.column("episode_index"))
        keep = np.isin(e, eps)
        if not keep.any():
            continue
        st.append(np.asarray(t.column("observation.state").to_pylist(), dtype=np.float32)[keep])
        ac.append(np.asarray(t.column("action").to_pylist(), dtype=np.float32)[keep])
        epi.append(e[keep])
        fr.append(np.asarray(t.column("frame_index"))[keep])

    state, action = np.concatenate(st), np.concatenate(ac)
    ep, t = np.concatenate(epi), np.concatenate(fr)
    order = np.lexsort((t, ep))          # 先按 episode，再按 frame
    return state[order], action[order], ep[order], t[order]


def load_run(path):
    """载入 npz，并探测有效 k 上限（排除补零段）。"""
    d = np.load(path)
    chunk, gt = d["chunk"], d["gt"]
    kmax = -1
    for k in range(chunk.shape[1]):
        if np.all(chunk[:, k] == 0) and np.all(gt[:, k] == 0):
            break
        kmax = k
    return {"chunk": chunk, "gt": gt, "kmax": kmax, "meta": d["meta"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds-root", required=True)
    ap.add_argument("--runs", default="base=results/perframe/base.npz,A40k=results/perframe/A40k.npz")
    ap.add_argument("--episodes", default="40-49")
    ap.add_argument("--out", default="results/baseline_static.json")
    args = ap.parse_args()

    lo, hi = args.episodes.split("-")
    eps = list(range(int(lo), int(hi) + 1))

    state, action, ep, t = load_holdout(args.ds_root, eps)
    lengths = {e: int((ep == e).sum()) for e in eps}
    print("留出集 %d 帧 / %d 段，长度 %s" % (len(ep), len(eps), [lengths[e] for e in eps]))

    runs = {}
    for item in args.runs.split(","):
        tag, path = item.split("=", 1)
        if not os.path.exists(path):
            print("  跳过（不存在）:", path)
            continue
        r = load_run(path)
        if not (np.array_equal(r["meta"][:, 1].astype(int), ep.astype(int)) and
                np.array_equal(r["meta"][:, 2].astype(int), t.astype(int))):
            print("  !! %s 帧顺序与 parquet 不一致，跳过" % tag)
            continue
        runs[tag] = r
        print("  载入 %-12s 有效 k=0..%d" % (tag, r["kmax"]))

    n = np.array([lengths[e] for e in ep])
    idx = np.arange(len(ep))

    state_hold, action_hold = {}, {}
    models = {tag: {} for tag in runs}
    for k in range(50):
        m = (t + k) < n                       # 仍在同一 episode 内
        gt_k = action[idx[m] + k]
        state_hold[k] = float(np.abs(state[m] - gt_k).mean())
        action_hold[k] = float(np.abs(action[m] - gt_k).mean())
        for tag, r in runs.items():
            if k <= r["kmax"]:
                models[tag][k] = float(np.abs(r["chunk"][m, k] - r["gt"][m, k]).mean())

    def mean_over(d, k1, k2):
        v = [d[k] for k in range(k1, k2 + 1) if k in d]
        return float(np.mean(v)) if v else None

    out = {
        "protocol": {
            "episodes": eps,
            "frames": int(len(ep)),
            "state_hold": "predict(t+k) = observation.state(t)",
            "action_hold": "predict(t+k) = action(t)",
            "mask": "排除 t+k >= 该 episode 长度",
            "valid_k": {tag: r["kmax"] for tag, r in runs.items()},
        },
        "per_k": {"state_hold": state_hold, "action_hold": action_hold,
                  **{tag: {str(k): v for k, v in d.items()} for tag, d in models.items()}},
        "aggregate": {
            "state_hold": {"k0": state_hold[0], "k1_9": mean_over(state_hold, 1, 9),
                           "k1_49": mean_over(state_hold, 1, 49)},
            "action_hold": {"k0": action_hold[0], "k1_9": mean_over(action_hold, 1, 9),
                            "k1_49": mean_over(action_hold, 1, 49)},
            **{tag: {"k0": d.get(0), "k1_9": mean_over(d, 1, 9), "k1_49": mean_over(d, 1, 49)}
               for tag, d in models.items()},
        },
    }

    print("\n逐 k 单步/多步误差（同一批帧）")
    head = ["static_state", "static_action"] + list(runs)
    print("%-6s" % "k" + "".join("%14s" % h for h in head))
    for k in [0, 1, 5, 9, 10, 25, 49]:
        row = "%-6d" % k
        vals = [state_hold[k], action_hold[k]] + [models[tg].get(k) for tg in runs]
        for v in vals:
            row += "%14s" % ("-" if v is None else "%.3f" % v)
        print(row)

    print("\n聚合")
    print("%-14s %8s %10s %10s" % ("", "k=0", "k=1..9", "k=1..49"))
    for tag, v in out["aggregate"].items():
        print("%-14s %8.3f %10s %10s" % (
            tag, v["k0"],
            "%.3f" % v["k1_9"] if v["k1_9"] is not None else "-",
            "%.3f" % v["k1_49"] if v["k1_49"] is not None else "-"))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n已写入", args.out)


if __name__ == "__main__":
    main()
