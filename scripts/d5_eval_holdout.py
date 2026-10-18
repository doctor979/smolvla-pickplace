"""留出集离线评测：计算单步与动作块的 L1 / MSE。

Usage:
    python scripts/d5_eval_holdout.py --ckpt <ckpt|repo_id> [--stride 5] [--batch-size 8]

Input:
    --ckpt    checkpoint 目录（自动探测 pretrained_model/ 子目录）或 HF repo id
    --ds-root 本地数据集快照路径；不传则走 Hub API（网络慢时会阻塞）

Output:
    JSON（--out）：l1_step0 / l1_chunk / mse_step0 / mse_chunk、per_joint_l1_mean、
    batch_size、n_forwards、ms_per_frame，以及前 3 个样本的预测与真值对照。

Invariants（改动会使与历史结果的比较失效）:
    1. 动作取自 predict_action_chunk。select_action 带动作队列，队列非空时返回缓存值，
       逐帧调用会用旧动作对齐新真值。
    2. 每个 batch 前调用 policy.reset()，清空观测队列。
    3. 预/后处理器复用训练脚本的 overrides，并把 dataset.meta.stats 传入
       normalizer/unnormalizer；缺失时归一化会静默退化为恒等变换。

Notes:
    - 批处理只合并前向，预处理与误差统计仍逐帧（单帧预处理约 2.7 ms）。
    - 批量模式下 flow-matching 噪声按 batch 采样，与逐帧版存在 <1% 数值差异。
"""
import argparse, json, os, time
import torch

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

RENAME = {"observation.images.up": "observation.images.camera1",
          "observation.images.side": "observation.images.camera2"}


def parse_episodes(s):
    """解析 episode 区间。

    Args:
        s: "40-49" 或 "40,42,45"。

    Returns:
        list[int]，升序。
    """
    s = s.strip()
    if "-" in s and "," not in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",") if x.strip()]


def to_tensor(x):
    """从 policy 输出中取出动作张量。

    兼容三种返回形态：Tensor / 具 .action 属性的对象 / 含 "action" 键的 dict。

    Returns:
        torch.Tensor，最后一维为动作维度。
    """
    if torch.is_tensor(x):
        return x
    if hasattr(x, "action"):
        return x.action
    if isinstance(x, dict) and "action" in x:
        return x["action"]
    raise TypeError("无法从 %s 取出张量" % type(x))


def stat(v):
    """汇总一组标量的统计量。

    Args:
        v: list[float]，长度 >= 1。

    Returns:
        dict: mean / std / min / max / n。n == 1 时 std 取 0.0，避免 nan 传播到结果文件。
    """
    t = torch.tensor(v)
    sd = float(t.std()) if t.numel() > 1 else 0.0
    return {"mean": float(t.mean()), "std": sd,
            "min": float(t.min()), "max": float(t.max()), "n": len(v)}


def stack_batch(per_sample):
    """把 B 个单帧预处理结果拼成一个 batch。

    Args:
        per_sample: list[dict]，每个元素是 pre() 对单帧的输出。

    Returns:
        dict：张量沿 dim0 拼接；无 batch 维的先 unsqueeze；非张量且各样本一致的取第一个。
    """
    """把 B 个「单帧预处理结果」拼成一个 batch dict。

    张量沿 dim0 拼接；非张量（如任务字符串）取第一个（同任务下内容相同）。
    若某个张量没有 batch 维，先 unsqueeze 再拼。
    """
    out = {}
    keys = per_sample[0].keys()
    for k in keys:
        vals = [d[k] for d in per_sample]
        if all(torch.is_tensor(v) for v in vals):
            try:
                out[k] = torch.cat([v if v.dim() > 0 else v.unsqueeze(0) for v in vals], dim=0)
            except Exception:
                out[k] = vals
        else:
            out[k] = vals[0] if len(set(map(str, vals))) == 1 else vals
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint 目录或 HF repo id")
    ap.add_argument("--tag", default="", help="结果标签, 如 base / A20k")
    ap.add_argument("--dataset", default="lerobot/svla_so101_pickplace")
    ap.add_argument("--ds-root", default="", help="本地数据集快照路径（跳过 Hub 查询，网络慢时用）")
    ap.add_argument("--episodes", default="40-49")
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = 不限制")
    ap.add_argument("--batch-size", type=int, default=8, help="推理批大小；1 = 逐帧模式")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out", default="")

    args = ap.parse_args()
    eps = parse_episodes(args.episodes)
    tag = args.tag or os.path.basename(str(args.ckpt).rstrip("/"))
    bs = max(1, args.batch_size)

    print("=" * 66)
    print("D5 离线评测（批量推理）")
    print("  标签        :", tag)
    print("  checkpoint  :", args.ckpt)
    print("  留出集      :", args.episodes, "| stride:", args.stride)
    print("  批大小      :", bs)
    print("  设备        :", args.device)
    print("=" * 66, flush=True)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    dts = {"action": [i / float(args.fps) for i in range(args.chunk_size)]}
    if args.ds_root:
        try:
            ds = LeRobotDataset(args.dataset, root=args.ds_root, episodes=eps,
                                delta_timestamps=dts)
            print("  数据集      : 本地快照（跳过 Hub 查询）", flush=True)
        except Exception as e:
            print("  本地快照加载失败(%s)，回退 Hub" % type(e).__name__, flush=True)
            ds = LeRobotDataset(args.dataset, episodes=eps, delta_timestamps=dts)
    else:
        ds = LeRobotDataset(args.dataset, episodes=eps, delta_timestamps=dts)
    print("  留出集帧数  :", len(ds), flush=True)

    ck = args.ckpt
    if not os.path.exists(os.path.join(ck, "config.json")):
        alt = os.path.join(ck, "pretrained_model")
        if os.path.exists(os.path.join(alt, "config.json")):
            print("  路径自动切换到: " + alt, flush=True)
            ck = alt
    args.ckpt = ck

    cfg = SmolVLAConfig.from_pretrained(args.ckpt)
    cfg.pretrained_path = args.ckpt
    cfg.device = args.device
    policy = make_policy(cfg, ds_meta=ds.meta, rename_map=RENAME)
    policy.to(args.device)
    policy.eval()
    n_par = sum(p.numel() for p in policy.parameters())
    print("  参数量      : %.1f M" % (n_par / 1e6), flush=True)

    pp_ov = {
        "device_processor": {"device": args.device},
        "normalizer_processor": {
            "features": {**cfg.input_features, **cfg.output_features},
            "norm_map": cfg.normalization_mapping,
            "stats": ds.meta.stats,
        },
        "rename_observations_processor": {"rename_map": RENAME},
    }
    po_ov = {
        "unnormalizer_processor": {
            "features": cfg.output_features,
            "norm_map": cfg.normalization_mapping,
            "stats": ds.meta.stats,
        },
    }
    pre, post = make_pre_post_processors(cfg, pretrained_path=args.ckpt,
                                         preprocessor_overrides=pp_ov,
                                         postprocessor_overrides=po_ov,
                                         dataset_stats=ds.meta.stats)
    print("  预处理器就绪", flush=True)

    idxs = list(range(0, len(ds), args.stride))
    if args.max_frames > 0:
        idxs = idxs[: args.max_frames]

    l1s, mses, l1_0, mse_0, per_joint = [], [], [], [], []
    preview = []
    n_forwards = 0
    t0 = time.time()

    for start in range(0, len(idxs), bs):
        chunk_idx = idxs[start: start + bs]
        samples = []
        prepared = []
        for i in chunk_idx:
            s = ds[i]
            samples.append(s)
            obs = {RENAME.get(k, k): v for k, v in s.items()
                   if k not in ("action", "action_is_pad")}
            b = pre(obs)
            prepared.append({k: (v.to(args.device) if torch.is_tensor(v) else v)
                             for k, v in b.items()})

        batched = stack_batch(prepared)
        torch.manual_seed(args.seed + chunk_idx[0])
        policy.reset()
        with torch.no_grad():
            ch = to_tensor(policy.predict_action_chunk(batched))
            ch = to_tensor(post(ch))
        n_forwards += 1
        ch = ch.float().cpu()

        for j, i in enumerate(chunk_idx):
            s = samples[j]
            pred = ch[j].reshape(-1, ch.shape[-1])
            gt = s["action"].float().reshape(-1, s["action"].shape[-1])
            pad = s.get("action_is_pad")
            if pad is not None and pad.numel() == pred.shape[0]:
                keep = ~pad.bool()
                pred = pred[keep]
                gt = gt[keep]
            m = min(pred.shape[0], gt.shape[0])
            if m == 0:
                continue
            d = pred[:m] - gt[:m]
            l1s.append(d.abs().mean().item())
            mses.append((d ** 2).mean().item())
            l1_0.append(d[0].abs().mean().item())
            mse_0.append((d[0] ** 2).mean().item())
            per_joint.append([round(float(x), 4) for x in d.abs().mean(dim=0)])
            if len(preview) < 3:
                preview.append({"idx": i, "steps": int(m),
                                "gt": [round(float(x), 3) for x in gt[0]],
                                "pred": [round(float(x), 3) for x in pred[0]]})

        done = min(start + bs, len(idxs))
        if (start // bs + 1) % 5 == 0 or done == len(idxs):
            print("  进度 %d/%d  前向 %d 次  用时 %.1fs"
                  % (done, len(idxs), n_forwards, time.time() - t0), flush=True)

    if not l1s:
        print("!! 没有有效样本，评测失败")
        return 1

    elapsed = time.time() - t0
    pj = torch.tensor(per_joint)
    res = {
        "tag": tag, "ckpt": args.ckpt, "episodes": eps, "stride": args.stride,
        "batch_size": bs, "n_forwards": n_forwards,
        "n_frames_evaluated": len(l1s), "holdout_frames_total": len(ds),
        "device": args.device, "params_M": n_par / 1e6,
        "elapsed_s": round(elapsed, 1),
        "ms_per_frame": round(elapsed * 1000.0 / max(1, len(l1s)), 2),
        "l1_step0": stat(l1_0), "mse_step0": stat(mse_0),
        "l1_chunk": stat(l1s), "mse_chunk": stat(mses),
        "per_joint_l1_mean": [round(float(x), 4) for x in pj.mean(dim=0)],
        "preview": preview,
    }

    print("=" * 66)
    print("结果 [%s]" % tag)
    print("  评测帧数        : %d  （前向 %d 次，批大小 %d）"
          % (res["n_frames_evaluated"], n_forwards, bs))
    print("  单步 L1 (均值±std): %.4f  ± %.4f" % (res["l1_step0"]["mean"], res["l1_step0"]["std"]))
    print("  单步 MSE        : %.4f" % res["mse_step0"]["mean"])
    print("  动作块 L1       : %.4f  ± %.4f" % (res["l1_chunk"]["mean"], res["l1_chunk"]["std"]))
    print("  动作块 MSE      : %.4f" % res["mse_chunk"]["mean"])
    print("  各关节 L1 均值  :", res["per_joint_l1_mean"])
    print("  用时            : %.1f s  (%.2f ms/帧)" % (elapsed, res["ms_per_frame"]))
    for s in preview:
        print("  样本 idx=%d (%d 步) 真值=%s" % (s["idx"], s["steps"], s["gt"]))
        print("                        预测=%s" % s["pred"])
    print("=" * 66)

    if args.out:
        d = os.path.dirname(os.path.abspath(args.out))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print("已保存:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
