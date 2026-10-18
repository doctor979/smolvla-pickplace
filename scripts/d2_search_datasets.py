"""数据集候选检索：按关键词查询 hf-mirror，按下载量降序输出前 20 条。

只做粗筛，不判定可用性；字段完整性与任务串唯一性由 d2_qc_candidates.py 判定。

Usage:
    python scripts/d2_search_datasets.py
"""
import json, urllib.request
# 按下载量降序取前 100 条，search 会匹配仓库名与描述
def q(search, limit=100):
    url = ("https://hf-mirror.com/api/datasets?search=" + search +
           "&limit=" + str(limit) + "&sort=downloads&direction=-1")
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8.0"})
    return json.load(urllib.request.urlopen(req, timeout=30))
# 5 组关键词：SO-101 的两种写法、SO-100（老版本）、以及 LeRobot 官方数据集系列
for s in ["so101", "so-101", "so100", "svla", "smolvla"]:
    print("=" * 12, s)
    try:
        data = q(s)
    except Exception as e:
        print("ERR", repr(e))
        continue
    for d in data[:20]:
        print("%8d %5d  %s" % (d.get("downloads", 0), d.get("likes", 0), d["id"]))
