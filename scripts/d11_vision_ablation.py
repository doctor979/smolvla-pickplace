"""输入消融：破坏图像或本体状态，看动作误差涨多少 —— 判断模型依赖哪一路输入。

八种输入条件（其余一切不变）：
    clean        原图 + 原始状态
    black        图像全零（像素空间）
    noise        图像为均匀随机噪声
    swap         两路相机互换（画面真实但视角错配）
    shuffle_near 同批次内图像后移一格（stride=5 时约 0.17 s，局部错帧）
    shuffle_far  使用数据集另一端的画面（跨 episode，秒级以上错帧）
    state_zero   本体状态置零
    state_far    本体状态换成另一段的（时序完全错配）

为什么需要这一组：
    d7_robustness 只做了遮挡/模糊这类**保留大部分信息**的扰动，误差仅涨 10–30%。
    要判断模型靠视觉还是靠本体感知，必须把每一路**分别**破坏，再看哪边掉得更狠——
    单向破坏无法换算成贡献比例，双向对照可以排出相对重要性。

口径（与 d10_baseline / d12_chunk_protocol 统一）：
    动作块 L1 = 排除 action_is_pad 标记的补零步，在 6 个关节与剩余步上取均值。
    同时保留「包含 padding」的数值（l1_chunk_all），便于与旧结果对照。

Usage:
    python scripts/d11_vision_ablation.py --ckpt <ckpt> --ds-root <快照> \
        --out results/vision_ablation.json [--stride 5] [--batch-size 8]

Notes:
    - stride 默认 5（552 帧）；结论是条件之间的相对比较，抽样不影响方向
    - black/noise 作用在像素空间（预处理之前）；state 类作用在原始观测上
    - shuffle_far / state_far 需要额外载入远端样本，解码开销约翻倍
    - 每个 batch 前固定随机种子，使 flow matching 的采样噪声在各条件间一致
"""
import argparse
import json
import os
import time

import numpy as np
import torch

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

RENAME = {"observation.images.up": "observation.images.camera1",
          "observation.images.side": "observation.images.camera2"}
CAMS = ["observation.images.up", "observation.images.side"]
STATE = "observation.state"
VARIANTS = ["clean", "black", "noise", "swap", "shuffle_near", "shuffle_far",
            "state_zero", "state_far"]


def to_tensor(x):
    if torch.is_tensor(x):
        return x
    if hasattr(x, "action"):
        return x.action
    if isinstance(x, dict) and "action" in x:
        return x["action"]
    raise TypeError(type(x))


def stack_batch(ps):
    out = {}
    for k in ps[0]:
        vals = [d[k] for d in ps]
        if all(torch.is_tensor(v) for v in vals):
            out[k] = torch.cat([v if v.dim() > 0 else v.unsqueeze(0) for v in vals], 0)
    return out


def corrupt(sample, variant, far, gen):
    """按条件改写输入；far 为远端样本（shuffle_far / state_far 时提供）。"""
    if variant == "clean":
        return sample
    s = dict(sample)
    if variant == "black":
        for c in CAMS:
            if c in s:
                s[c] = torch.zeros_like(s[c])
    elif variant == "noise":
        for c in CAMS:
            if c in s:
                s[c] = torch.rand(s[c].shape, generator=gen, dtype=s[c].dtype)
    elif variant == "swap":
        up, side = s.get(CAMS[0]), s.get(CAMS[1])
        if up is not None and side is not None:
            s[CAMS[0]], s[CAMS[1]] = side, up
    elif variant == "shuffle_far" and far is not None:
        for c in CAMS:
            if c in s and c in far:
                s[c] = far[c]
    elif variant == "state_zero":
        if STATE in s:
            s[STATE] = torch.zeros_like(s[STATE])
    elif variant == "state_far" and far is not None and STATE in far:
        s[STATE] = far[STATE]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="lerobot/svla_so101_pickplace")
    ap.add_argument("--ds-root", default="")
    ap.add_argument("--episodes", default="40-49")
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--out", default="results/vision_ablation.json")
    args = ap.parse_args()

    a, b = args.episodes.split("-")
    eps = list(range(int(a), int(b) + 1))
    bs = max(1, args.batch_size)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    print("=" * 62, flush=True)
    print("D11 输入消融  episodes=%s stride=%d bs=%d" % (args.episodes, args.stride, bs), flush=True)

    dts = {"action": [i / float(args.fps) for i in range(args.chunk_size)]}
    if args.ds_root:
        ds = LeRobotDataset(args.dataset, root=args.ds_root, episodes=eps, delta_timestamps=dts)
    else:
        ds = LeRobotDataset(args.dataset, episodes=eps, delta_timestamps=dts)

    ck = args.ckpt
    if not os.path.exists(os.path.join(ck, "config.json")):
        alt = os.path.join(ck, "pretrained_model")
        if os.path.exists(os.path.join(alt, "config.json")):
            ck = alt
    cfg = SmolVLAConfig.from_pretrained(ck)
    cfg.pretrained_path = ck
    cfg.device = args.device
    policy = make_policy(cfg, ds_meta=ds.meta, rename_map=RENAME)
    policy.to(args.device)
    policy.eval()

    # 每帧的段长与段内序号：用于排除「t+k 越界」的补零步。
    # 注意 action_is_pad 在本数据集里不足以标出全部越界步（实测掩码几乎无效），
    # 因此按 episode 长度自行计算，与 d12_chunk_protocol.py 保持同一口径。
    lens, ep_of, t_of, len_arr = None, None, None, None
    try:
        df = ds.meta.episodes.to_pandas()
        if "episode_index" in df.columns:
            df = df[df["episode_index"].isin(eps)].sort_values("episode_index")
        if "dataset_from_index" in df.columns and "dataset_to_index" in df.columns:
            lens = [(int(r["dataset_to_index"]) - int(r["dataset_from_index"]))
                    for _, r in df.iterrows()]
        elif "length" in df.columns:
            lens = [int(x) for x in df["length"].tolist()]
    except Exception as e:
        print("  meta.episodes 读取失败(%s)" % type(e).__name__, flush=True)

    if lens and len(lens) == len(eps) and sum(lens) == len(ds):
        ep_of, t_of = [], []
        for e, n_ep in zip(eps, lens):
            ep_of += [e] * n_ep
            t_of += list(range(n_ep))
        ep_of, t_of = np.array(ep_of), np.array(t_of)
        len_arr = np.array([lens[eps.index(e)] for e in ep_of])
        print("  段长映射: %d 帧 / %d 段" % (len(ep_of), len(lens)), flush=True)
    else:
        print("  !! 段长映射失败，动作块指标退化为「含 padding」", flush=True)

    pp_ov = {
        "device_processor": {"device": args.device},
        "normalizer_processor": {"features": {**cfg.input_features, **cfg.output_features},
                                 "norm_map": cfg.normalization_mapping, "stats": ds.meta.stats},
        "rename_observations_processor": {"rename_map": RENAME},
    }
    po_ov = {"unnormalizer_processor": {"features": cfg.output_features,
                                        "norm_map": cfg.normalization_mapping,
                                        "stats": ds.meta.stats}}
    pre, post = make_pre_post_processors(cfg, pretrained_path=ck,
                                         preprocessor_overrides=pp_ov,
                                         postprocessor_overrides=po_ov,
                                         dataset_stats=ds.meta.stats)

    idxs = list(range(0, len(ds), args.stride))
    N = len(idxs)
    print("  评估帧数: %d" % N, flush=True)

    results = {}
    for variant in variants:
        t0 = time.time()
        s0, scm, sca, nm = 0.0, 0.0, 0.0, 0
        gen = torch.Generator(device="cpu").manual_seed(args.seed)

        for start in range(0, N, bs):
            sel = idxs[start:start + bs]
            need_far = variant in ("shuffle_far", "state_far")
            far = [ds[idxs[(start + j + N // 2) % N]] for j in range(len(sel))] if need_far else None

            prepared, raws = [], []
            for j, i in enumerate(sel):
                raw = ds[i]                      # 只解码一次，后面算真值时复用
                raws.append(raw)
                sample = corrupt(raw, variant, far[j] if need_far else None, gen)
                obs = {RENAME.get(k, k): v for k, v in sample.items()
                       if k not in ("action", "action_is_pad")}
                bb = pre(obs)
                prepared.append({k: (v.to(args.device) if torch.is_tensor(v) else v)
                                 for k, v in bb.items()})
            batch = stack_batch(prepared)

            # 局部错帧：批次内滚动一格（stride=5 时约 0.17 s）
            if variant == "shuffle_near" and len(sel) > 1:
                for c in ("observation.images.camera1", "observation.images.camera2"):
                    if c in batch:
                        batch[c] = torch.roll(batch[c], 1, dims=0)

            torch.manual_seed(args.seed + sel[0])
            policy.reset()
            with torch.no_grad():
                ch = to_tensor(policy.predict_action_chunk(batch))
                ch = to_tensor(post(ch))
            ch = ch.float().cpu().numpy()

            for j, i in enumerate(sel):
                gt = raws[j]["action"].float().reshape(-1, 6).numpy()
                m = min(ch[j].shape[0], gt.shape[0])
                d = np.abs(ch[j][:m] - gt[:m])
                s0 += float(d[0].mean())
                sca += float(d.mean())          # 含 padding（旧口径）
                if len_arr is not None:
                    valid_len = int(len_arr[i] - t_of[i])      # 本帧还能看到的真实步数
                    keep = np.arange(m) < valid_len
                    scm += float(d[keep].mean()) if keep.any() else 0.0
                else:
                    scm += float(d.mean())
                nm += 1

        results[variant] = {
            "n": nm,
            "l1_step0": s0 / nm,
            "l1_chunk_pad_excluded": scm / nm,
            "l1_chunk_all": sca / nm,
            "elapsed_s": round(time.time() - t0, 1),
        }
        print("  %-13s 单步 %.3f  动作块(排pad) %.3f  动作块(含pad) %.3f  (%.0fs)"
              % (variant, results[variant]["l1_step0"],
                 results[variant]["l1_chunk_pad_excluded"],
                 results[variant]["l1_chunk_all"], results[variant]["elapsed_s"]), flush=True)

    c = results.get("clean")
    if c:
        for v in results.values():
            v["ratio_step0"] = v["l1_step0"] / c["l1_step0"]
            v["ratio_chunk"] = v["l1_chunk_pad_excluded"] / c["l1_chunk_pad_excluded"]
        print("\n相对 clean 的倍率（动作块口径：已排除 padding）")
        print("%-13s %10s %10s" % ("条件", "单步", "动作块"))
        for k, v in results.items():
            print("%-13s %9.2fx %9.2fx" % (k, v["ratio_step0"], v["ratio_chunk"]))

    out = {"ckpt": args.ckpt, "stride": args.stride, "frames": N,
           "episodes": args.episodes,
           "protocol": "动作块 L1 排除 action_is_pad；同时给含 padding 的旧口径",
           "results": results}
    d = os.path.dirname(os.path.abspath(args.out))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n已写入", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
