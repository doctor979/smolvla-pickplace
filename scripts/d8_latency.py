"""推理延迟与吞吐测量：bs=1..N，分别测纯推理与端到端（含预处理）。

Output:
    JSON：每个批大小、每种口径下的单次耗时、每帧摊薄耗时、吞吐、显存。

Usage:
    python scripts/d8_latency.py --ckpt <ckpt> --out results/latency.json

Notes:
    - 必须先预热：GPU 在空闲后会降频，未预热时首个测量点偏慢约 15%。
    - e2e 与纯推理的差值即预处理开销；二者接近说明预处理不是瓶颈。
"""
import argparse, json, os, statistics, time
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
    raise TypeError("取不出张量: %s" % type(x))


def build_batch(obs_single, bs, device):
    """把单帧观测复制成批大小为 bs 的 batch。

    Args:
        obs: 单帧观测 dict（已预处理或原始张量均可）。
        bs: 目标批大小。
        device: 目标设备。

    Returns:
        dict：张量沿 dim0 复制 bs 份。
    """
    out = {}
    for k, v in obs_single.items():
        if torch.is_tensor(v):
            if v.dim() == 0:
                out[k] = v.repeat(bs).to(device)
            else:
                out[k] = v.unsqueeze(0).repeat(bs, *([1] * v.dim())).to(device)
        else:
            out[k] = [v] * bs if isinstance(v, str) else v
    return out


def bench(policy, pre, post, obs, bs, device, iters, warmup, mode):
    """测量单次调用的耗时。

    Args:
        fn: 无参可调用对象，内部完成一次完整推理。
        iters: 计时次数，不含预热。
        warmup: 预热次数。GPU 升频需要预热，否则第一个点系统性偏慢。

    Returns:
        list[float]，每次调用的毫秒数。
    """
    """mode: 'policy' = 只测推理（预处理在计时外）; 'e2e' = 预处理+推理+后处理（仅 bs=1）

    注意：predict_action_chunk 需要预处理产物（含 observation.language.tokens），
    不能直接喂原始观测，否则 KeyError: 'observation.language.tokens'。
    所以 policy 模式下先把 batch 预处理一次（计时外），计时只包住推理。
    """
    obs_b = build_batch(obs, bs, device) if bs > 1 else obs
    prepared = pre(obs_b)
    prepared = {k: (v.to(device) if torch.is_tensor(v) else v)
                for k, v in prepared.items()}

    def run():
        if mode == "e2e":
            b = pre(obs)
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            ch = to_tensor(policy.predict_action_chunk(b))
            to_tensor(post(ch))
        else:
            ch = to_tensor(policy.predict_action_chunk(prepared))

    for _ in range(warmup):
        policy.reset()
        with torch.no_grad():
            run()
    torch.cuda.synchronize()

    ts = []
    for _ in range(iters):
        policy.reset()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            run()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)

    ts.sort()
    return {
        "batch": bs, "mode": mode, "n": len(ts),
        "median_ms": round(statistics.median(ts), 2),
        "p90_ms": round(ts[int(len(ts) * 0.9)], 2),
        "min_ms": round(ts[0], 2),
        "per_frame_ms": round(statistics.median(ts) / bs, 2),
        "throughput_fps": round(1000.0 * bs / statistics.median(ts), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--dataset", default="lerobot/svla_so101_pickplace")
    ap.add_argument("--ds-root", default="", help="本地数据集快照路径，传了就完全不走 Hub")
    ap.add_argument("--episodes", default="40-49")
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--batches", default="1,2,4,8,16")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--test-fp32", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    tag = args.tag or os.path.basename(str(args.ckpt).rstrip("/"))
    a, b = args.episodes.split("-")
    eps = list(range(int(a), int(b) + 1))

    print("[0] 准备 import lerobot ...", flush=True)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    print("[1] import 完成，准备加载数据集 ...", flush=True)

    dts = {"action": [i / float(args.fps) for i in range(args.chunk_size)]}
    if args.ds_root:
        print("[1b] 使用本地数据集路径:", args.ds_root, flush=True)
        ds = LeRobotDataset(args.dataset, root=args.ds_root,
                            episodes=eps, delta_timestamps=dts)
    else:
        ds = LeRobotDataset(args.dataset, episodes=eps, delta_timestamps=dts)
    print("[2] 数据集加载完成，帧数 =", len(ds), flush=True)

    ck = args.ckpt
    if not os.path.exists(os.path.join(ck, "config.json")):
        alt = os.path.join(ck, "pretrained_model")
        if os.path.exists(os.path.join(alt, "config.json")):
            ck = alt
    args.ckpt = ck

    cfg = SmolVLAConfig.from_pretrained(ck)
    cfg.pretrained_path = ck
    cfg.device = args.device

    dtype_fields = [f for f in cfg.__dataclass_fields__ if "dtype" in f.lower()]
    print("=" * 72)
    print("D8 推理延迟测量   tag =", tag)
    print("  精度相关字段:", dtype_fields or "（无）")
    print("  chunk_size =", getattr(cfg, "chunk_size", "?"),
          "| n_action_steps =", getattr(cfg, "n_action_steps", "?"))
    print("=" * 72, flush=True)

    policy = make_policy(cfg, ds_meta=ds.meta, rename_map=RENAME)
    policy.to(args.device)
    policy.eval()

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
    pre, post = make_pre_post_processors(cfg, pretrained_path=ck,
                                         preprocessor_overrides=pp_ov,
                                         postprocessor_overrides=po_ov,
                                         dataset_stats=ds.meta.stats)

    sample = ds[0]
    obs_raw = {k: v for k, v in sample.items() if k not in ("action", "action_is_pad")}
    obs = {RENAME.get(k, k): v for k, v in obs_raw.items()}

    rows = []
    batches = [int(x) for x in args.batches.split(",")]

    print("[3] 全局热身（20 次 bs=1 推理，让 GPU 升频）...", flush=True)
    b0 = pre(obs)
    b0 = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in b0.items()}
    for _ in range(20):
        policy.reset()
        with torch.no_grad():
            policy.predict_action_chunk(b0)
    torch.cuda.synchronize()
    print("[4] 热身完成，显存 %.2f GB，开始测量" %
          (torch.cuda.memory_allocated() / 1e9), flush=True)

    for mode in ["policy", "e2e"]:
        for bs in (batches if mode == "policy" else [1]):
            torch.cuda.reset_peak_memory_stats()
            r = bench(policy, pre, post, obs, bs, args.device,
                      args.iters, args.warmup, mode)
            r["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            rows.append(r)
            print("  %-7s bs=%-3d 中位 %8.2f ms  每帧 %7.2f ms  吞吐 %7.1f fps  显存 %5.2f GB"
                  % (mode, bs, r["median_ms"], r["per_frame_ms"],
                     r["throughput_fps"], r["peak_mem_gb"]), flush=True)

    chunk_dur = args.chunk_size / float(args.fps)
    print("-" * 72)
    print("  动作块时长: %d 步 / %d fps = %.3f 秒" % (args.chunk_size, args.fps, chunk_dur))
    print("  执行整块后再推理，所需推理频率 = %.2f Hz（每 %.0f ms 一次）"
          % (1.0 / chunk_dur, chunk_dur * 1000))
    print("  每 10 步重规划，所需推理频率 = %.2f Hz（每 %.0f ms 一次）"
          % (args.fps / 10.0, 1000.0 * 10 / args.fps))
    print("-" * 72)

    out = {
        "tag": tag, "ckpt": ck, "device": args.device,
        "chunk_size": getattr(cfg, "chunk_size", None),
        "n_action_steps": getattr(cfg, "n_action_steps", None),
        "fps": args.fps, "chunk_duration_s": round(chunk_dur, 3),
        "iters": args.iters, "rows": rows,
    }
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print("已保存:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
