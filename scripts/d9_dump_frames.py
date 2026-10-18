"""抽取指定帧拼成对照图，输出单张 JPEG（便于跨机传输）。

用途：定性确认「误差最大的帧长什么样」，与 d9_analyze.py 的定量结论互相印证。
帧号、相机、尺寸与质量均为硬编码，改动时同步调整。

Usage:
    python scripts/d9_dump_frames.py
"""
import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw

RENAME = {"observation.images.up": "observation.images.camera1",
          "observation.images.side": "observation.images.camera2"}

DS = "lerobot/svla_so101_pickplace"
SNAP = str(Path(__file__).resolve().parents[1] /
           "data/lerobot/hub/datasets--lerobot--svla_so101_pickplace"
           "/snapshots/f641879e22172be7e8161d5e6c1503c2d2feb657")
OUT = str(Path(__file__).resolve().parents[1] / "results/perframe")


def main():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(DS, root=SNAP, episodes=[42])
    npz = np.load(OUT + "/A40k.npz", allow_pickle=True)
    meta, gt, chunk = npz["meta"], npz["gt"], npz["chunk"]

    # 取 ep42 的逐帧误差
    m = meta[meta[:, 1] == 42]
    err = {int(r[2]): float(r[3]) for r in m}
    picks = [(10, "good"), (55, "worst"), (57, "worst"), (59, "worst"), (61, "worst")]

    W, H = 200, 150
    rows, cols = ["up", "side"], picks
    PAD, TOP, LEFT = 4, 26, 44
    gw = LEFT + len(cols) * (W + PAD) + PAD
    gh = TOP + len(rows) * (H + PAD) + PAD
    cv = Image.new("RGB", (gw, gh), (255, 255, 255))
    dr = ImageDraw.Draw(cv)
    dr.text((6, 8), "episode 42  |  A40k (finetuned)  |  L1 on each frame", fill=(20, 50, 100))

    for r, cam in enumerate(rows):
        y = TOP + r * (H + PAD)
        dr.text((4, y + H // 2), cam.upper(), fill=(20, 20, 20))
        for c, (t, label) in enumerate(picks):
            s = ds[t]
            img = s["observation.images." + cam]
            arr = (img.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
            im = Image.fromarray(arr).resize((W, H), Image.LANCZOS)
            x = LEFT + c * (W + PAD)
            cv.paste(im, (x, y))
            dr.rectangle([x, y, x + W - 1, y + H - 1], outline=(160, 185, 210))
            if r == 0:
                dr.text((x + 3, y - 13), "t=%d  L1=%.1f" % (t, err.get(t, -1)), fill=(120, 30, 30))

    p = OUT + "/worst_frames_grid.jpg"
    cv.save(p, quality=52, optimize=True)
    print("拼图:", p, os.path.getsize(p), "bytes", cv.size)

    # 这些帧上模型预测 vs 真值（第 0 步，6 维）
    pos = {(int(r[1]), int(r[2])): i for i, r in enumerate(meta)}
    for t, _ in picks:
        i = pos.get((42, t))
        if i is None:
            continue
        print("t=%-3d  gt=%s" % (t, [round(float(v), 1) for v in gt[i, 0]]))
        print("        pr=%s" % [round(float(v), 1) for v in chunk[i, 0]])


if __name__ == "__main__":
    raise SystemExit(main())
