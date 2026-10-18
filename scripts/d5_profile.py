"""评测耗时归因：把单帧成本拆成「视频解码 / 预处理 / 前向」三段。

用途是解释「批量化理论上限与实际加速比之间的差距」：
批处理后前向占比下降，瓶颈转移到逐帧解码（阿姆达尔定律）。

Usage:
    python scripts/d5_profile.py --ckpt <ckpt> --frames 32 --batch 8

Output:
    终端表格：各段 ms/帧、占比、理论上限加速比。
"""
import argparse, os, time
import torch

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

RENAME = {"observation.images.up": "observation.images.camera1",
          "observation.images.side": "observation.images.camera2"}


def to_tensor(x):
    """从 policy 输出中取出动作张量（兼容 Tensor / .action / dict 三种形态）。"""
    if torch.is_tensor(x):
        return x
    if hasattr(x, "action"):
        return x.action
    if isinstance(x, dict) and "action" in x:
        return x["action"]
    raise TypeError(type(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="lerobot/svla_so101_pickplace")
    ap.add_argument("--ds-root", default="")
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    dts = {"action": [i / 30.0 for i in range(50)]}
    eps = list(range(40, 50))
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

    N = args.frames
    idxs = list(range(0, len(ds), max(1, len(ds) // N)))[:N]

    # 每个阶段单独计时（GPU 前先 synchronize）
    t_dec, t_pre = 0.0, 0.0
    prepared = []
    for i in idxs:
        t = time.perf_counter()
        s = ds[i]
        t_dec += time.perf_counter() - t

        obs = {RENAME.get(k, k): v for k, v in s.items() if k not in ("action", "action_is_pad")}
        t = time.perf_counter()
        b = pre(obs)
        t_pre += time.perf_counter() - t
        prepared.append({k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in b.items()})

    def stack(ps):
        out = {}
        for k in ps[0]:
            vals = [d[k] for d in ps]
            if all(torch.is_tensor(v) for v in vals):
                out[k] = torch.cat([v if v.dim() > 0 else v.unsqueeze(0) for v in vals], 0)
        return out

    def bench(fn, iters=5, warmup=2):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        return (time.perf_counter() - t) / iters * 1000.0

    bs = min(args.batch, len(prepared))
    one = prepared[0]
    many = stack(prepared[:bs])

    ms_b1 = bench(lambda: policy.predict_action_chunk(one))
    ms_bn = bench(lambda: policy.predict_action_chunk(many))

    print("=" * 60)
    print("评测耗时归因（%d 帧采样，GPU=%s）" % (len(idxs), args.device))
    print("=" * 60)
    print("  1. 视频解码 + 读盘   : %7.2f ms/帧" % (t_dec / len(idxs) * 1000))
    print("  2. 预处理 pre()      : %7.2f ms/帧" % (t_pre / len(idxs) * 1000))
    print("  3. 前向 bs=1         : %7.2f ms/帧" % ms_b1)
    print("  4. 前向 bs=%d        : %7.2f ms/次 = %.2f ms/帧" % (bs, ms_bn, ms_bn / bs))
    print("-" * 60)
    per_frame_old = t_dec / len(idxs) * 1000 + t_pre / len(idxs) * 1000 + ms_b1
    per_frame_new = t_dec / len(idxs) * 1000 + t_pre / len(idxs) * 1000 + ms_bn / bs
    print("  理论每帧（逐帧）    : %.1f ms" % per_frame_old)
    print("  理论每帧（bs=%d）    : %.1f ms" % (bs, per_frame_new))
    print("  理论上限加速        : %.2fx" % (per_frame_old / per_frame_new))
    print("-" * 60)
    print("  前向占比（逐帧）    : %.0f%%" % (ms_b1 / per_frame_old * 100))
    print("  前向占比（批量）    : %.0f%%" % ((ms_bn / bs) / per_frame_new * 100))
    print("  解码占比（批量后）  : %.0f%%" % ((t_dec / len(idxs) * 1000) / per_frame_new * 100))
    print("=" * 60)


if __name__ == "__main__":
    raise SystemExit(main())
