"""数据预览：从数据集视频抽帧拼图，确认模型实际看到的画面。

Output:
    results/data_preview/grid.jpg（小尺寸，便于跨机传输）

Usage:
    python scripts/d2_preview_frames.py

Notes:
    纯 CPU，不占 GPU；依赖 ffmpeg 可执行文件（conda 环境内）。
"""
import os, subprocess
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = str(Path(__file__).resolve().parents[1])
VID = (ROOT + "/data/lerobot/hub/datasets--lerobot--svla_so101_pickplace/"
       "snapshots/f641879e22172be7e8161d5e6c1503c2d2feb657/videos")
FF = os.environ.get("FFMPEG", "ffmpeg")   # 指向 conda 环境内的 ffmpeg 可设 FFMPEG=/path/to/ffmpeg
OUT = ROOT + "/results/data_preview"
os.makedirs(OUT, exist_ok=True)

CAMS = [("UP", VID + "/observation.images.up/chunk-000/file-000.mp4"),
        ("SIDE", VID + "/observation.images.side/chunk-000/file-000.mp4")]
TIMES = [0.0, 2.0, 4.0, 6.0]
W, H = 168, 126

tiles = []
for cam, path in CAMS:
    row = []
    for t in TIMES:
        f = "%s/%s_%.1f.jpg" % (OUT, cam, t)
        subprocess.run([FF, "-y", "-ss", str(t), "-i", path, "-frames:v", "1",
                        "-vf", "scale=%d:%d" % (W, H), "-q:v", "6", f],
                       capture_output=True)
        row.append(Image.open(f).convert("RGB") if os.path.exists(f)
                   else Image.new("RGB", (W, H), (210, 210, 210)))
    tiles.append(row)

PAD, TOP, LEFT = 5, 30, 52
gw = LEFT + len(TIMES) * (W + PAD) + PAD
gh = TOP + len(CAMS) * (H + PAD) + PAD
cv = Image.new("RGB", (gw, gh), (255, 255, 255))
dr = ImageDraw.Draw(cv)
dr.text((6, 9), "svla_so101_pickplace  ep.0   pink lego -> transparent box", fill=(20, 60, 110))
for r, (cam, _) in enumerate(CAMS):
    y = TOP + r * (H + PAD)
    dr.text((5, y + H // 2 - 4), cam, fill=(20, 20, 20))
    for c, t in enumerate(TIMES):
        x = LEFT + c * (W + PAD)
        cv.paste(tiles[r][c], (x, y))
        dr.rectangle([x, y, x + W - 1, y + H - 1], outline=(160, 185, 210))
        if r == 0:
            dr.text((x + 3, y - 11), "%.0fs" % t, fill=(70, 70, 70))

big = OUT + "/grid_big.jpg"
cv.resize((gw * 3, gh * 3), Image.LANCZOS).save(big, quality=55, optimize=True)
small = OUT + "/grid.jpg"
cv.save(small, quality=55, optimize=True)
print("小图:", small, os.path.getsize(small), "bytes", cv.size)
print("大图:", big, os.path.getsize(big), "bytes (gw*3, gh*3)")
