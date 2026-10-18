#!/usr/bin/env python
"""训练结束后的自动评测守卫：轮询训练进程，退出后复查 GPU 再执行批量评测。

流程：
    等待 lerobot-train 与过夜脚本退出 -> 复查 GPU0 占用 -> 调用 d5_eval_all.sh

Usage:
    python scripts/d5_watcher.py &

Notes:
    GPU0 占用超过阈值时按约定中止，不与他人抢占；日志写入 logs/d5_watcher.log。
"""
import datetime, os, subprocess, sys, time
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
LOG = ROOT + "/logs/d5_watcher.log"
EVAL_LOG = ROOT + "/logs/d5_eval_all.log"


def log(msg):
    """同时写 stdout 与日志文件，带时间戳前缀。"""
    line = "[%s] %s" % (datetime.datetime.now().strftime("%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def running(pat):
    """判断是否存在匹配该正则的进程。

    Args:
        pat: 传给 pgrep -f 的正则（建议用 [] 包裹首字母，避免匹配到自身命令行）。
    """
    r = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True)
    return bool(r.stdout.strip())


def gpu_used():
    """读取 GPU0 已用显存（MiB）。

    Returns:
        int，读取失败时返回 -1。
    """
    r = subprocess.run(
        "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0",
        shell=True, capture_output=True, text=True)
    try:
        return int(r.stdout.strip())
    except Exception:
        return -1


log("=" * 50)
log("watcher 启动, 等待训练结束")

while running("d3_overnight[.]sh") or running("lerobot[-]train"):
    time.sleep(60)

log("训练已结束(或已退出)")
time.sleep(30)

u = gpu_used()
log("评测前复查 GPU0 已用: %d MiB" % u)
if u < 0:
    log("无法读取 GPU 状态, 仍尝试评测")
elif u > 20000:
    log("!!!! GPU0 已被大量占用, 按规矩不硬跑, 退出")
    sys.exit(3)

log("开始评测(全部 checkpoint)")
with open(EVAL_LOG, "w") as f:
    p = subprocess.run(["bash", ROOT + "/scripts/d5_eval_all.sh", "5"],
                       cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
log("评测退出码: %d" % p.returncode)
log("完成. 结果见 results/ 与 logs/d5_eval_all.log")
