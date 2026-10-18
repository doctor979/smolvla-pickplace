"""从训练日志提取 loss 曲线，输出单行 JSON。

Output:
    {"A": [[step, loss, epoch], ...], "B": [...]}，抽样到约 55 点便于粘贴。

Usage:
    python scripts/d3_loss_curve.py > loss.json

Notes:
    日志中 step 存在 5000 / 9.6K / 40K 三种写法，需解析 K/M 后缀后再转数值。
"""
import json, re

# 匹配「数字 + 可选 K/M 后缀」，日志里 step/loss/epch 都用这个格式
NUM = r"(\d+(?:\.\d+)?)([KM]?)"

def val(num, suf):
    x = float(num)
    if suf == "K":
        x *= 1e3
    elif suf == "M":
        x *= 1e6
    return x

# 逐行扫日志：只保留带 loss 的 INFO 行，抽 step/loss/epoch 三个数
def parse(p):
    pts = []
    try:
        txt = open(p, "rb").read().decode("utf-8", "ignore").replace("\r", "\n")
    except Exception:
        return pts
    for ln in txt.split("\n"):
        if "loss:" not in ln:
            continue
        ms = re.search(r"step:" + NUM, ln)
        ml = re.search(r"loss:" + NUM, ln)
        me = re.search(r"epch:" + NUM, ln)
        if not (ms and ml):
            continue
        st = val(ms.group(1), ms.group(2))
        lo = float(ml.group(1))
        ep = val(me.group(1), me.group(2)) if me else -1
        pts.append([st, lo, ep])
    return pts

# 实验 A（40 条数据）与 B（30 条数据）各提取一条曲线
out = {}
for name, p in [("A", "logs/d3_expA.log"), ("B", "logs/d3_expB.log")]:
    d = parse(p)
    print("EXP %s: %d 点" % (name, len(d)))
    if d:
        print("   首: step=%.0f loss=%.4f | 末: step=%.0f loss=%.4f"
              % (d[0][0], d[0][1], d[-1][0], d[-1][1]))
        k = max(1, len(d) // 55)
        s = d[::k]
        if s[-1] is not d[-1]:
            s.append(d[-1])
        out[name] = [[round(x[0]), round(x[1], 4), round(x[2], 1)] for x in s]
        print("   采样 %d 点" % len(s))
print("JSON=" + json.dumps(out))
