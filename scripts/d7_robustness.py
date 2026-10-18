"""图像扰动鲁棒性评测：在同一留出集上比较各扰动条件下的动作误差。

Conditions（确定性，同 seed 同结果）:
    brightness  x0.3 / x0.6 / x1.4
    noise       高斯噪声 sigma 0.03 / 0.08 / 0.15
    blur        高斯模糊 kernel 3 / 7 / 15
    occlusion   随机方形遮挡，面积占比 10% / 25% / 40%

Usage:
    python scripts/d7_robustness.py --ckpt <ckpt> [--stride 10] [--batch-size 8]

Output:
    JSON：baseline_l1_step0 与各条件的 l1_step0 / l1_chunk / ratio_vs_clean。

Notes:
    - 预处理与误差统计逐帧，仅前向按 batch 合并；指标口径与 d5_eval_holdout.py 一致。
    - 同一批帧的 13 个条件复用同一份解码结果，避免重复读盘。
"""
import argparse, json, os, time
import torch
import torchvision.transforms.functional as TF

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

RENAME = {"observation.images.up": "observation.images.camera1",
          "observation.images.side": "observation.images.camera2"}

KIND_IDX = {"none": 0, "brightness": 1, "noise": 2, "blur": 3, "occlusion": 4}

CONDITIONS = [
    ("none", 0.0),
    ("brightness", 0.3), ("brightness", 0.6), ("brightness", 1.4),
    ("noise", 0.03), ("noise", 0.08), ("noise", 0.15),
    ("blur", 3.0), ("blur", 7.0), ("blur", 15.0),
    ("occlusion", 0.10), ("occlusion", 0.25), ("occlusion", 0.40),
]


def to_tensor(x):
    if torch.is_tensor(x):
        return x
    if hasattr(x, "action"):
        return x.action
    if isinstance(x, dict) and "action" in x:
        return x["action"]
    raise TypeError("无法从 %s 取出张量" % type(x))


def perturb(img, kind, level, seed):
    """对单张图像施加指定扰动。

    Args:
        img: (3, H, W) float，取值 [0, 1]。
        kind: brightness | noise | blur | occlusion。
        level: 强度，语义随 kind 变化（见模块头）。
        seed: 随机种子，保证同输入同输出。

    Returns:
        同形状张量，已截断至 [0, 1]。
    """
    """对单张图 (3,H,W) float [0,1] 施加扰动。确定性：同 seed 同结果。"""
    if kind == "none":
        return img
    if kind == "brightness":
        return (img * level).clamp(0.0, 1.0)
    if kind == "noise":
        g = torch.Generator().manual_seed(seed)
        n = torch.randn(img.shape, generator=g) * level
        return (img + n).clamp(0.0, 1.0)
    if kind == "blur":
        k = int(level)
        return TF.gaussian_blur(img, [k, k], [k / 6.0, k / 6.0])
    if kind == "occlusion":
        h, w = img.shape[-2:]
        side = int((level * h * w) ** 0.5)
        side = max(4, min(side, min(h, w) - 1))
        g = torch.Generator().manual_seed(seed)
        y0 = int(torch.randint(0, h - side + 1, (1,), generator=g).item())
        x0 = int(torch.randint(0, w - side + 1, (1,), generator=g).item())
        out = img.clone()
        out[..., y0:y0 + side, x0:x0 + side] = 0.0
        return out
    raise ValueError(kind)


def stack_batch(per_sample):
    """把 B 个单帧预处理结果拼成一个 batch（与 d5_eval_holdout.py 实现一致）。

    Returns:
        dict：张量沿 dim0 拼接，非张量取第一个样本的值。
    """
    out = {}
    for k in per_sample[0].keys():
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
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--dataset", default="lerobot/svla_so101_pickplace")
    ap.add_argument("--ds-root", default="")
    ap.add_argument("--episodes", default="40-49")
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8, help="每个条件的前向批大小；1 = 逐帧")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    tag = args.tag or os.path.basename(str(args.ckpt).rstrip("/"))
    bs = max(1, args.batch_size)
    a, b = args.episodes.split("-")
    eps = list(range(int(a), int(b) + 1))

    print("=" * 70, flush=True)
    print("D7 鲁棒性测试（批量推理）")
    print("  tag     :", tag)
    print("  ckpt    :", args.ckpt)
    print("  留出集  :", args.episodes, "| stride:", args.stride, "| 批大小:", bs)
    print("  条件数  :", len(CONDITIONS))
    print("=" * 70, flush=True)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    dts = {"action": [i / float(args.fps) for i in range(args.chunk_size)]}
    if args.ds_root:
        try:
            ds = LeRobotDataset(args.dataset, root=args.ds_root, episodes=eps,
                                delta_timestamps=dts)
            print("  数据集    : 本地快照（跳过 Hub 查询）", flush=True)
        except Exception as e:
            print("  本地快照加载失败(%s)，回退 Hub" % type(e).__name__, flush=True)
            ds = LeRobotDataset(args.dataset, episodes=eps, delta_timestamps=dts)
    else:
        ds = LeRobotDataset(args.dataset, episodes=eps, delta_timestamps=dts)

    ck = args.ckpt
    if not os.path.exists(os.path.join(ck, "config.json")):
        alt = os.path.join(ck, "pretrained_model")
        if os.path.exists(os.path.join(alt, "config.json")):
            print("  路径自动切换:", alt, flush=True)
            ck = alt
    args.ckpt = ck

    cfg = SmolVLAConfig.from_pretrained(ck)
    cfg.pretrained_path = ck
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
    pre, post = make_pre_post_processors(cfg, pretrained_path=ck,
                                         preprocessor_overrides=pp_ov,
                                         postprocessor_overrides=po_ov,
                                         dataset_stats=ds.meta.stats)

    IMG_KEYS = [k for k in ds.features if k.startswith("observation.images.")]
    print("  图像 key:", IMG_KEYS, flush=True)
    print("  总帧数  :", len(ds), flush=True)

    idxs = list(range(0, len(ds), args.stride))
    acc = {c: {"l1_step0": [], "mse_step0": [], "l1_chunk": []} for c in CONDITIONS}
    n_forwards = 0

    t0 = time.time()
    for start in range(0, len(idxs), bs):
        chunk_idx = idxs[start: start + bs]
        # 先一次性把这一批帧读进内存（读盘一次，13 个条件复用）
        frame_cache = []
        for i in chunk_idx:
            s = ds[i]
            base_obs = {k: v for k, v in s.items()
                        if k not in ("action", "action_is_pad")}
            gt = s["action"].float().reshape(-1, s["action"].shape[-1])
            frame_cache.append((i, base_obs, gt, s.get("action_is_pad")))

        for kind, level in CONDITIONS:
            prepared = []
            for (i, base_obs, gt, pad) in frame_cache:
                seed = KIND_IDX[kind] * 1000000 + int(level * 1000) * 1000 + i
                obs_raw = dict(base_obs)
                if kind != "none":
                    for ik in IMG_KEYS:
                        if ik in obs_raw and torch.is_tensor(obs_raw[ik]):
                            obs_raw[ik] = perturb(obs_raw[ik], kind, level, seed)
                obs = {RENAME.get(k, k): v for k, v in obs_raw.items()}
                bb = pre(obs)
                prepared.append({k: (v.to(args.device) if torch.is_tensor(v) else v)
                                 for k, v in bb.items()})

            batched = stack_batch(prepared)
            torch.manual_seed(args.seed + chunk_idx[0])
            policy.reset()
            with torch.no_grad():
                ch = to_tensor(policy.predict_action_chunk(batched))
                ch = to_tensor(post(ch))
            n_forwards += 1
            ch = ch.float().cpu()

            for j, (i, base_obs, gt, pad) in enumerate(frame_cache):
                pred = ch[j].reshape(-1, ch.shape[-1])
                g = gt
                if pad is not None and pad.numel() == pred.shape[0]:
                    keep = ~pad.bool()
                    pred = pred[keep]
                    g = g[keep]
                m = min(pred.shape[0], g.shape[0])
                if m == 0:
                    continue
                d = pred[:m] - g[:m]
                acc[(kind, level)]["l1_step0"].append(d[0].abs().mean().item())
                acc[(kind, level)]["mse_step0"].append((d[0] ** 2).mean().item())
                acc[(kind, level)]["l1_chunk"].append(d.abs().mean().item())

        done = min(start + bs, len(idxs))
        if (start // bs + 1) % 3 == 0 or done == len(idxs):
            print("  进度 %d/%d 帧  前向 %d 次  用时 %.0fs"
                  % (done, len(idxs), n_forwards, time.time() - t0), flush=True)

    base_l1 = float(torch.tensor(acc[("none", 0.0)]["l1_step0"]).mean())

    rows = []
    for kind, level in CONDITIONS:
        d = acc[(kind, level)]
        if not d["l1_step0"]:
            continue
        l1 = float(torch.tensor(d["l1_step0"]).mean())
        rows.append({
            "kind": kind, "level": level,
            "l1_step0": round(l1, 4),
            "l1_chunk": round(float(torch.tensor(d["l1_chunk"]).mean()), 4),
            "mse_step0": round(float(torch.tensor(d["mse_step0"]).mean()), 4),
            "ratio_vs_clean": round(l1 / base_l1, 3) if base_l1 > 0 else None,
            "n": len(d["l1_step0"]),
        })

    elapsed = time.time() - t0
    print("=" * 70)
    print("结果 [%s]   基线(无扰动) 单步L1 = %.4f   批大小 %d" % (tag, base_l1, bs))
    print("%-12s %8s %10s %10s %8s" % ("扰动", "强度", "单步L1", "相对基线", "帧数"))
    for r in rows:
        print("%-12s %8.2f %10.4f %9.2fx %8d"
              % (r["kind"], r["level"], r["l1_step0"], r["ratio_vs_clean"], r["n"]))
    print("  总用时: %.1f s （%d 次前向，%.1f ms/帧·条件）"
          % (elapsed, n_forwards, elapsed * 1000.0 / max(1, len(idxs) * len(CONDITIONS))))

    out = {
        "tag": tag, "ckpt": args.ckpt, "episodes": eps, "stride": args.stride,
        "batch_size": bs, "n_forwards": n_forwards,
        "device": args.device, "params_M": sum(p.numel() for p in policy.parameters()) / 1e6,
        "n_frames": len(idxs), "elapsed_s": round(elapsed, 1),
        "baseline_l1_step0": round(base_l1, 4),
        "conditions": rows,
    }
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print("已保存:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
