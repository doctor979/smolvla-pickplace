"""逐帧记录：保存留出集每帧的整块预测动作、真值与图像统计量。

Output（npz）:
    chunk (N, C, 6) float32  预测动作块，已反归一化
    gt    (N, C, 6) float32  真值动作块
    meta  (N, 6)    float32  [global_idx, episode, t_in_ep, l1_step0, l1_chunk, pad_count]
    img   (N, 4)    float32  [up_mean, up_std, side_mean, side_std]

Why full chunk:
    保存全部 C 步才能事后模拟不同重规划间隔 k，无需重新推理（见 d9_analyze.py）。

Usage:
    python scripts/d9_record.py --ckpt <ckpt> --out results/perframe/<tag>.npz [--stride 1]

Notes:
    - meta.episodes 的 dataset_from_index 是相对完整数据集的坐标，而 episodes 子集的
      下标从 0 开始，故按子集顺序重新累加各段长度；不匹配时退化为单段并告警。
    - stride > 1 时无法构造连续决策帧，重规划分析会退化，建议保持 1。
"""
import argparse, os, time
import numpy as np
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
    raise TypeError(type(x))


def stack_batch(ps):
    out = {}
    for k in ps[0]:
        vals = [d[k] for d in ps]
        if all(torch.is_tensor(v) for v in vals):
            out[k] = torch.cat([v if v.dim() > 0 else v.unsqueeze(0) for v in vals], 0)
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
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tag = args.tag or os.path.basename(str(args.ckpt).rstrip("/"))
    a, b = args.episodes.split("-")
    eps = list(range(int(a), int(b) + 1))
    bs = max(1, args.batch_size)

    print("=" * 62, flush=True)
    print("D9 逐帧记录 [%s]  episodes=%s stride=%d bs=%d" % (tag, args.episodes, args.stride, bs))

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

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

    # 建立「帧 -> (episode, 帧内序号)」映射
    # 注意：meta.episodes 里的 dataset_from_index 是相对**整个数据集**的，
    # 而 ds(episodes=[40..49]) 的下标是相对**子集**的（从 0 开始），所以必须按子集重新累加。
    lens = None
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

    ep_of, t_of = [], []
    if lens and len(lens) == len(eps) and sum(lens) == len(ds):
        start = 0
        for ep, n in zip(eps, lens):
            for t in range(n):
                ep_of.append(ep)
                t_of.append(t)
            start += n
        print("  帧映射建立: %d 帧 / %d 段（各段长度 %s）"
              % (len(ep_of), len(lens), lens), flush=True)
    else:
        print("  !! 映射失败(len=%s, sum=%s, frames=%d)，退化为单段"
              % (len(lens) if lens else None, sum(lens) if lens else None, len(ds)), flush=True)
        ep_of = [eps[0]] * len(ds)
        t_of = list(range(len(ds)))
    ep_of = np.array(ep_of)
    t_of = np.array(t_of)

    idxs = list(range(0, len(ds), args.stride))
    N = len(idxs)
    C = args.chunk_size
    chunk_arr = np.zeros((N, C, 6), dtype=np.float32)
    gt_arr = np.zeros((N, C, 6), dtype=np.float32)
    meta_arr = np.zeros((N, 6), dtype=np.float32)
    img_arr = np.zeros((N, 4), dtype=np.float32)

    t0 = time.time()
    w = 0
    for start in range(0, N, bs):
        sel = idxs[start:start + bs]
        prepared, samples = [], []
        for i in sel:
            s = ds[i]
            samples.append(s)
            obs = {RENAME.get(k, k): v for k, v in s.items() if k not in ("action", "action_is_pad")}
            bb = pre(obs)
            prepared.append({k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in bb.items()})

        torch.manual_seed(args.seed + sel[0])
        policy.reset()
        with torch.no_grad():
            ch = to_tensor(policy.predict_action_chunk(stack_batch(prepared)))
            ch = to_tensor(post(ch))
        ch = ch.float().cpu().numpy()

        for j, i in enumerate(sel):
            s = samples[j]
            gt = s["action"].float().reshape(-1, s["action"].shape[-1]).numpy()
            pad = s.get("action_is_pad")
            npad = int(pad.sum().item()) if pad is not None else 0
            m = min(ch[j].shape[0], gt.shape[0])
            d = ch[j][:m] - gt[:m]
            l1_0 = float(np.abs(d[0]).mean())
            l1_c = float(np.abs(d).mean())

            chunk_arr[w, :m] = ch[j][:m]
            gt_arr[w, :m] = gt[:m]
            meta_arr[w] = [i, ep_of[i], t_of[i], l1_0, l1_c, npad]

            iu = s.get("observation.images.up")
            isd = s.get("observation.images.side")
            img_arr[w] = [
                float(iu.mean()) if iu is not None else 0.0,
                float(iu.std()) if iu is not None else 0.0,
                float(isd.mean()) if isd is not None else 0.0,
                float(isd.std()) if isd is not None else 0.0,
            ]
            w += 1

        done = min(start + bs, N)
        if (start // bs + 1) % 10 == 0 or done == N:
            print("  进度 %d/%d  用时 %.0fs" % (done, N, time.time() - t0), flush=True)

    d = os.path.dirname(os.path.abspath(args.out))
    if d:
        os.makedirs(d, exist_ok=True)
    np.savez_compressed(args.out, chunk=chunk_arr, gt=gt_arr, meta=meta_arr,
                        img=img_arr, tag=np.array([tag]))
    print("已保存: %s  (%.1f MB, %.1f s)"
          % (args.out, os.path.getsize(args.out) / 1048576.0, time.time() - t0))
    print("平均: 单步L1=%.4f  动作块L1=%.4f"
          % (meta_arr[:, 3].mean(), meta_arr[:, 4].mean()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
