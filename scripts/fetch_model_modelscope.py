"""从 ModelScope 下载 smolvla_base 权重到本地目录。

选用理由：
    hf-mirror 对该仓库实测约 175 KB/s（872 MB 需约 66 分钟），
    ModelScope 约 17 MB/s（约 2 分钟）。

Usage:
    python scripts/fetch_model_modelscope.py

Notes:
    断点续传按文件大小判断，只有大小一致才跳过；中断后重跑即可。
"""
import json, os, urllib.request
from pathlib import Path

# ModelScope 的文件列表接口（Revision=master 对应 HF 的 main 分支）
API = "https://modelscope.cn/api/v1/models/lerobot/smolvla_base/repo/files?Revision=master&Root="
OUT = str(Path(__file__).resolve().parents[1] / "data/models/smolvla_base")

d = json.loads(urllib.request.urlopen(urllib.request.Request(API, headers={"User-Agent": "Mozilla/5.0"}), timeout=30).read())
files = d["Data"]["Files"]
print("仓库文件数:", len(files))
os.makedirs(OUT, exist_ok=True)

# 逐个文件下载；Type=blob 才是文件，dir 要跳过
for f in files:
    if f.get("Type") != "blob":
        continue
    path = f["Path"]
    size = f.get("Size", 0) or 0
    out = os.path.join(OUT, path)
    os.makedirs(os.path.dirname(out), exist_ok=True)
# 断点续传的简化版：大小一致就认为已完整，直接跳过
    if os.path.exists(out) and os.path.getsize(out) == size:
        print("  已有 %-70s" % path)
        continue
    url = "https://modelscope.cn/models/lerobot/smolvla_base/resolve/master/" + path
    print("  下载 %-50s %8.2f MB" % (path, size / 1048576.0))
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as r, open(out, "wb") as w:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            w.write(chunk)
    got = os.path.getsize(out)
    print("       完成 %8.2f MB %s" % (got / 1048576.0, "OK" if got == size else "!! 大小不符"))
print("全部完成")
