"""轨迹可视化：六关节的预测与真值曲线，另存某一帧的动作块。

Output:
    results/traj_<tag>.png        整段轨迹对比（子图 = 关节）
    results/chunk_<tag>.png       单帧的 50 步动作块对比
    results/traj_<tag>.json       原始数组，便于复算或重绘

Usage:
    python scripts/d6_plot_traj.py --ckpt <ckpt> --episode 40 --stride 12

Notes:
    - 与 d5_eval_holdout.py 使用同一套预/后处理器 overrides，否则反归一化不会生效。
    - 依赖 matplotlib（Agg 后端，无需显示器）。
"""
import argparse, json, os, time
import torch

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RENAME = {"observation.images.up": "observation.images.camera1",
          "observation.images.side": "observation.images.camera2"}
JN = ["joint 0", "joint 1", "joint 2", "joint 3", "joint 4", "gripper"]


def to_tensor(x):
    """从 policy 输出中取出动作张量（兼容 Tensor / .action / dict 三种形态）。"""
    if torch.is_tensor(x):
        return x
    if hasattr(x, "action"):
        return x.action
    if isinstance(x, dict) and "action" in x:
        return x["action"]
    raise TypeError("无法从 %s 取出张量" % type(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--dataset", default="lerobot/svla_so101_pickplace")
    ap.add_argument("--episode", type=int, default=40)
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    tag = args.tag or ("ep%d" % args.episode)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    dts = {"action": [i / float(args.fps) for i in range(args.chunk_size)]}
    ds = LeRobotDataset(args.dataset, episodes=[args.episode], delta_timestamps=dts)
    n = len(ds)
    print("episode %d 共 %d 帧" % (args.episode, n), flush=True)

    cfg = SmolVLAConfig.from_pretrained(args.ckpt)
    cfg.pretrained_path = args.ckpt
    cfg.device = args.device
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
    pre, post = make_pre_post_processors(cfg, pretrained_path=args.ckpt,
                                         preprocessor_overrides=pp_ov,
                                         postprocessor_overrides=po_ov,
                                         dataset_stats=ds.meta.stats)

    idxs = list(range(0, n, args.stride))
    if args.max_frames > 0:
        idxs = idxs[: args.max_frames]

    gt_seq, pred_seq, ts = [], [], []
    chunk_pred, chunk_gt, chunk_t = None, None, 0
    mid = idxs[len(idxs) // 2]
    t0 = time.time()
    for k, i in enumerate(idxs):
        s = ds[i]
        torch.manual_seed(args.seed + i)
        obs = {kk: v for kk, v in s.items() if kk not in ("action", "action_is_pad")}
        obs = {RENAME.get(kk, kk): v for kk, v in obs.items()}
        b = pre(obs)
        b = {kk: (v.to(args.device) if torch.is_tensor(v) else v) for kk, v in b.items()}
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
        if ch.shape[0] == 0:
            continue
        gt_seq.append(gt[0].tolist())
        pred_seq.append(ch[0].tolist())
        ts.append(i / float(args.fps))
        if i == mid:
            chunk_pred = ch.tolist()
            chunk_gt = gt.tolist()
            chunk_t = i
        if (k + 1) % 20 == 0:
            print("  进度 %d/%d  %.1fs" % (k + 1, len(idxs), time.time() - t0), flush=True)

    if not gt_seq:
        print("!! 没有采到数据")
        return 1

    os.makedirs(args.outdir, exist_ok=True)
    A = torch.tensor(gt_seq)
    B = torch.tensor(pred_seq)
    err = (B - A).abs()

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for j in range(6):
        ax = axes[j // 3][j % 3]
        ax.plot(ts, A[:, j].tolist(), label="ground truth", linewidth=1.6)
        ax.plot(ts, B[:, j].tolist(), label="prediction", linewidth=1.4, linestyle="--")
        ax.set_title(JN[j], fontsize=11)
        ax.set_xlabel("time (s)", fontsize=9)
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(fontsize=8)
    fig.suptitle("Episode %d  prediction vs ground truth  (tag=%s, stride=%d)"
                 % (args.episode, tag, args.stride), fontsize=12)
    fig.tight_layout()
    f1 = os.path.join(args.outdir, "traj_%s.png" % tag)
    fig.savefig(f1, dpi=120)
    plt.close(fig)
    print("已保存:", f1)

    if chunk_pred is not None:
        fig2, axes2 = plt.subplots(2, 3, figsize=(15, 7))
        cp = torch.tensor(chunk_pred)
        cg = torch.tensor(chunk_gt)
        cx = [i / float(args.fps) for i in range(cp.shape[0])]
        for j in range(6):
            ax = axes2[j // 3][j % 3]
            ax.plot(cx, cg[:, j].tolist(), label="ground truth", linewidth=1.8)
            ax.plot(cx, cp[:, j].tolist(), label="predicted chunk", linewidth=1.5, linestyle="--")
            ax.set_title(JN[j], fontsize=11)
            ax.set_xlabel("seconds ahead", fontsize=9)
            ax.grid(alpha=0.3)
            if j == 0:
                ax.legend(fontsize=8)
        fig2.suptitle("Action chunk at frame %d of episode %d  (tag=%s)" % (chunk_t, args.episode, tag),
                      fontsize=12)
        fig2.tight_layout()
        f2 = os.path.join(args.outdir, "chunk_%s.png" % tag)
        fig2.savefig(f2, dpi=120)
        plt.close(fig2)
        print("已保存:", f2)

    out = {
        "tag": tag, "ckpt": args.ckpt, "episode": args.episode, "stride": args.stride,
        "n_points": len(gt_seq), "fps": args.fps,
        "l1_step0_mean": float(err.mean()), "l1_step0_std": float(err.std()),
        "per_joint_l1": [round(float(x), 4) for x in err.mean(dim=0)],
        "time_s": ts, "gt": A.tolist(), "pred": B.tolist(),
        "chunk_frame": chunk_t, "chunk_gt": chunk_gt, "chunk_pred": chunk_pred,
    }
    fj = os.path.join(args.outdir, "traj_%s.json" % tag)
    with open(fj, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print("已保存:", fj)
    print("该集单步 L1: %.4f   各关节 L1: %s" % (float(err.mean()), out["per_joint_l1"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
