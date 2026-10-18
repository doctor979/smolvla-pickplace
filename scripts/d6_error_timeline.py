"""逐帧误差时序：按 episode 归一化进度统计误差，用于观察误差在任务内的分布。

Output:
    JSON：每个 episode 的逐帧误差序列与分段均值。

Usage:
    python scripts/d6_error_timeline.py --ckpt <ckpt> --episodes 40,45

Notes:
    末端误差普遍下降是任务特性（机械臂趋于静止），不是模型变准；
    比较不同模型时应看同一位置的相对差异，而非曲线的绝对形状。
"""
import argparse, json, os, time
import torch

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

RENAME = {"observation.images.up": "observation.images.camera1",
          "observation.images.side": "observation.images.camera2"}


def to_tensor(x):
    if torch.is_tensor(x):
        return x
    if hasattr(x, "action"):
        return x.action
    if isinstance(x, dict) and "action" in x:
        return x["action"]
    raise TypeError(str(type(x)))


def seg_stats(vals, k=3):
    """把序列等分成若干段，返回各段均值。

    Args:
        arr: 1D 数值序列。
        n: 段数。

    Returns:
        list[float]，长度 n；末段不足时按实际长度取均值。
    """
    n = len(vals)
    if n == 0:
        return []
    out = []
    for i in range(k):
        a = int(i * n / k)
        b = int((i + 1) * n / k)
        seg = vals[a:b]
        out.append(round(sum(seg) / max(1, len(seg)), 3))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dataset", default="lerobot/svla_so101_pickplace")
    ap.add_argument("--episodes", default="40,45")
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    eps = [int(x) for x in args.episodes.split(",")]

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    ck = args.ckpt
    if not os.path.exists(os.path.join(ck, "config.json")):
        alt = os.path.join(ck, "pretrained_model")
        if os.path.exists(os.path.join(alt, "config.json")):
            ck = alt
    print("ckpt =", ck, flush=True)

    ds_full = LeRobotDataset(args.dataset)
    dts = {"action": [i / float(args.fps) for i in range(args.chunk_size)]}

    cfg = SmolVLAConfig.from_pretrained(ck)
    cfg.pretrained_path = ck
    cfg.device = args.device
    policy = make_policy(cfg, ds_meta=ds_full.meta, rename_map=RENAME)
    policy.to(args.device)
    policy.eval()

    pp_ov = {
        "device_processor": {"device": args.device},
        "normalizer_processor": {
            "features": {**cfg.input_features, **cfg.output_features},
            "norm_map": cfg.normalization_mapping,
            "stats": ds_full.meta.stats,
        },
        "rename_observations_processor": {"rename_map": RENAME},
    }
    po_ov = {
        "unnormalizer_processor": {
            "features": cfg.output_features,
            "norm_map": cfg.normalization_mapping,
            "stats": ds_full.meta.stats,
        },
    }
    pre, post = make_pre_post_processors(cfg, pretrained_path=ck,
                                         preprocessor_overrides=pp_ov,
                                         postprocessor_overrides=po_ov,
                                         dataset_stats=ds_full.meta.stats)
    print("policy + processors ready", flush=True)

    out = {"tag": args.tag, "ckpt": ck, "chunk_size": args.chunk_size,
           "fps": args.fps, "episodes": {}}

    for ep in eps:
        ds_ep = LeRobotDataset(args.dataset, episodes=[ep], delta_timestamps=dts)
        n = len(ds_ep)
        s0, sch = [], []
        t0 = time.time()
        for i in range(n):
            s = ds_ep[i]
            obs = {k: v for k, v in s.items() if k not in ("action", "action_is_pad")}
            obs = {RENAME.get(k, k): v for k, v in obs.items()}
            b = pre(obs)
            b = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in b.items()}
            policy.reset()
            with torch.no_grad():
                ch = to_tensor(policy.predict_action_chunk(b))
                ch = to_tensor(post(ch))
            ch = ch.reshape(-1, ch.shape[-1]).float().cpu()
            gt = s["action"].float().reshape(-1, s["action"].shape[-1])
            pad = s.get("action_is_pad")
            if pad is not None and pad.numel() == ch.shape[0]:
                keep = ~pad.bool()
                ch = ch[keep]
                gt = gt[keep]
            m = min(ch.shape[0], gt.shape[0])
            if m == 0:
                s0.append(0.0)
                sch.append(0.0)
                continue
            d = (ch[:m] - gt[:m]).abs()
            s0.append(float(d[0].mean()))
            sch.append(float(d.mean()))
            if (i + 1) % 100 == 0:
                print("   ep%d %d/%d  %.0fs" % (ep, i + 1, n, time.time() - t0), flush=True)

        k = max(1, n // 23)
        out["episodes"][str(ep)] = {
            "n_frames": n,
            "duration_s": round(n / float(args.fps), 2),
            "l1_step0_mean": round(sum(s0) / max(1, len(s0)), 4),
            "l1_chunk_mean": round(sum(sch) / max(1, len(sch)), 4),
            "seg3_step0": seg_stats(s0),
            "seg3_chunk": seg_stats(sch),
            "curve_step0": [round(x, 3) for x in s0[::k]],
            "curve_chunk": [round(x, 3) for x in sch[::k]],
            "series_step0": [round(x, 4) for x in s0],
        }
        r = out["episodes"][str(ep)]
        print("EP%d  n=%d  dur=%.1fs" % (ep, n, r["duration_s"]))
        print("   step0  mean=%.3f  seg3=%s" % (r["l1_step0_mean"], r["seg3_step0"]))
        print("   chunk  mean=%.3f  seg3=%s" % (r["l1_chunk_mean"], r["seg3_chunk"]))
        print("   curve_step0 = %s" % json.dumps(r["curve_step0"]))
        print("   curve_chunk = %s" % json.dumps(r["curve_chunk"]))

    os.makedirs(args.outdir, exist_ok=True)
    p = os.path.join(args.outdir, "timeline_%s.json" % args.tag)
    json.dump(out, open(p, "w"), ensure_ascii=False)
    print("saved:", p)


if __name__ == "__main__":
    main()
