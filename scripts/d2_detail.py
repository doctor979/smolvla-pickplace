"""数据集细节核查：文件清单、任务字符串、基座模型体积。

三段输出：
    1. 候选仓库的文件清单与体积（v3.0 会把视频合并为大 mp4，文件数很少）
    2. meta/tasks.json(l) 内容，用于判断是否为单任务数据集
    3. smolvla_base 各文件体积与合计，用于估算下载时间

Usage:
    python scripts/d2_detail.py
"""
import json, urllib.request

# 一律走 hf-mirror：直连 huggingface.co 在这台机器上不通
BASE = "https://hf-mirror.com"

def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=40).read()

# 把字节数换成人类可读（MB/GB）
def human(n):
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return "%.2f %s" % (n, u)
        n = n / 1024.0
    return "%.2f PB" % n

# ---- 第 1 部分：文件清单与各自体积 ----
print("############ 1. 候选数据集的文件清单 ############")
for repo in ["lerobot/svla_so101_pickplace", "youliangtan/so101-table-cleanup"]:
    print("=" * 55)
    print("REPO:", repo)
    try:
        items = json.loads(get(BASE + "/api/datasets/" + repo + "/tree/main?recursive=true&limit=1000"))
        for it in items[:25]:
            print("  %-12s %12s  %s" % (it.get("type"), human(it.get("size", 0)), it.get("path")))
        if len(items) > 25:
            print("  ... 共 %d 项" % len(items))
    except Exception as e:
        print("  失败:", repr(e))

print()
# ---- 第 2 部分：任务串（判断是不是单任务数据集）----
print("############ 2. 任务字符串（决定任务串是否统一）############")
for repo in ["lerobot/svla_so101_pickplace", "youliangtan/so101-table-cleanup"]:
    print("=" * 55)
    print("REPO:", repo)
    got = False
    for fn in ["meta/tasks.jsonl", "meta/tasks.json"]:
        try:
            raw = get(BASE + "/datasets/" + repo + "/resolve/main/" + fn).decode("utf-8", "ignore")
            print("  文件:", fn)
            txt = raw.strip()
            print("  内容:", txt[:1200])
            got = True
            break
        except Exception as e:
            print("  %s 不可用 (%s)" % (fn, type(e).__name__))
    if not got:
        print("  未找到 JSON 格式任务文件（v3.0 可能是 parquet，需装好环境后再读）")

print()
# ---- 第 3 部分：基座模型体积（决定下载耗时）----
print("############ 3. 基座模型 lerobot/smolvla_base 体积 ############")
try:
    items = json.loads(get(BASE + "/api/models/lerobot/smolvla_base/tree/main?recursive=true&limit=1000"))
    tot = 0
    for it in items:
        if it.get("type") == "file":
            tot += it.get("size", 0) or 0
            print("  %12s  %s" % (human(it.get("size", 0)), it.get("path")))
    print("  >>> 合计:", human(tot))
except Exception as e:
    print("  失败:", repr(e))
