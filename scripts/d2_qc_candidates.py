"""数据集体检：按体积 / episodes / 字段 / 版本四项筛除不合格候选。

Usage:
    python scripts/d2_qc_candidates.py

Notes:
    HF 的 tree API 单页最多返回 1000 条，靠响应头 Link 取下一页。
"""
import json, urllib.request

# 候选清单（含官方 SO-101、社区 SO-101、以及两个 SO-100 备选）
CANDS = ["lerobot/svla_so101_pickplace",
         "SO-101-robot-learning/so101_pick_place",
         "youliangtan/so101-table-cleanup",
         "lerobot/svla_so100_pickplace",
         "lerobot/svla_so100_sorting"]

BASE = "https://hf-mirror.com"

# 字节数 → 人类可读
def human(n):
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return "%.2f %s" % (n, u)
        n = n / 1024.0
    return "%.2f PB" % n

# 递归遍历仓库全部文件累加体积（自动翻页）
def size_of(repo):
    url = BASE + "/api/datasets/" + repo + "/tree/main?recursive=true&limit=1000"
    total = 0
    nf = 0
    pages = 0
    while url and pages < 40:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=60)
        items = json.loads(resp.read())
        for it in items:
            if it.get("type") == "file":
                total += it.get("size", 0) or 0
                nf += 1
        link = resp.headers.get("Link", "")
        url = None
        if "rel=\"next\"" in link:
            nxt = link.split(";")[0].strip().strip("<>")
            if nxt.startswith("/"):
                nxt = BASE + nxt
            url = nxt
        pages += 1
    return total, nf

# 读 meta/info.json：episodes / frames / fps / robot_type / 特征表都在这里
def info(repo):
    url = BASE + "/datasets/" + repo + "/resolve/main/meta/info.json"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return json.loads(urllib.request.urlopen(req, timeout=40).read())

# 逐个候选打印体检结果；读不到的直接跳过，不中断整轮
for r in CANDS:
    print("=" * 60)
    print("REPO:", r)
    try:
        s, nf = size_of(r)
        print("  体积:", human(s), " 文件数:", nf)
    except Exception as e:
        print("  体积: 读取失败", repr(e))
    try:
        d = info(r)
    except Exception as e:
        print("  info.json 读取失败:", repr(e))
        continue
    print("  episodes:", d.get("total_episodes"), "| frames:", d.get("total_frames"),
          "| fps:", d.get("fps"), "| robot:", d.get("robot_type"),
          "| ver:", d.get("codebase_version"))
    for k, v in (d.get("features") or {}).items():
        print("    -", k, "|", v.get("dtype"), "| shape", v.get("shape"), "| video", v.get("video", False))
