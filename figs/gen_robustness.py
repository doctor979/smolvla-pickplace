"""生成扰动对比条形图（SVG）：各条件下的误差相对干净基线的倍率。

Input:
    results/robust_*.json（d7_robustness.py 产出）；自动扫描目录，有几组画几组。

Output:
    figs/robustness*.svg（路径由命令行第 2 个参数决定）

Usage:
    python figs/gen_robustness.py <results_dir> <out.svg>
"""
import json
import os
import sys

KIND_CN = {"none": "无扰动", "brightness": "亮度", "noise": "噪声",
           "blur": "模糊", "occlusion": "遮挡"}


def level_label(kind, level):
    if kind == "none":
        return "—"
    if kind == "brightness":
        return "×%.1f" % level
    if kind == "noise":
        return "σ=%.2f" % level
    if kind == "blur":
        return "k=%d" % int(level)
    if kind == "occlusion":
        return "%d%%" % int(level * 100)
    return str(level)


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    res_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    out = sys.argv[2] if len(sys.argv) > 2 else "figs/robustness.svg"

    LABELS = {
        "base": ("未微调基座", "#85B7EB"),
        "A40k": ("40k · 无增强", "#1D9E75"),
        "C40k": ("40k · 有增强", "#EF9F27"),
    }
    DEFAULT_COLORS = ["#7F77DD", "#D4537E", "#888780", "#0F6E56"]

    import glob
    files = sorted(glob.glob(os.path.join(res_dir, "robust_*.json")))
    groups = []
    for i, f in enumerate(files):
        d = load(f)
        if not d:
            continue
        tag = d.get("tag") or os.path.basename(f)
        if tag in LABELS:
            name, color = LABELS[tag]
        else:
            name, color = tag, DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
        groups.append((name, color, d))
    groups.sort(key=lambda g: 0 if g[2].get("tag") == "base" else 1)
    if not groups:
        print("没有找到结果文件")
        return 1

    primary = groups[-1][2]
    conds = [c for c in primary["conditions"] if c["kind"] != "none"]
    conds.sort(key=lambda c: -c["ratio_vs_clean"])

    def find(d, kind, level):
        if d is None:
            return None
        for c in d["conditions"]:
            if c["kind"] == kind and abs(c["level"] - level) < 1e-9:
                return c
        return None

    n = len(conds)
    row_h = 26
    top = 76
    lx = 150
    x0 = 160
    per_unit = 880.0
    max_ratio = max([c["ratio_vs_clean"] for c in conds] +
                    [(find(g[2], c["kind"], c["level"]) or {}).get("ratio_vs_clean") or 0
                     for g in groups for c in conds] + [1.3])
    xmax = x0 + (max_ratio - 1.0 + 0.05) * per_unit
    if xmax > 620:
        per_unit = (620 - x0) / (max_ratio - 1.0 + 0.05)
        xmax = 620

    h = top + n * row_h + 62
    s = []
    s.append('<svg viewBox="0 0 680 %d" width="100%%" role="img" xmlns="http://www.w3.org/2000/svg">' % h)
    s.append('<title>图像扰动对动作误差的影响</title>')
    s.append('<desc>横向条形图，对比不同图像扰动条件下动作误差相对干净数据的倍率。倍率越高说明模型越脆弱。</desc>')
    s.append('<text x="40" y="30" fill="#2C2C2A" font-size="14" font-weight="500">'
             '图像扰动下动作误差的变化（倍率，1.0 = 干净图像）</text>')

    lgd_x = 40
    for name, color, _ in groups:
        s.append('<rect x="%d" y="42" width="10" height="10" rx="2" fill="%s"/>' % (lgd_x, color))
        s.append('<text x="%d" y="51" fill="#5F5E5A" font-size="12">%s</text>' % (lgd_x + 15, name))
        lgd_x += 15 + len(name) * 13 + 28

    for xv in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
        px = x0 + (xv - 1.0) * per_unit
        if px > 626:
            continue
        dash = ' stroke-dasharray="2 3"' if xv > 1.0 else ''
        s.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="#D3D1C7" stroke-width="0.5"%s/>'
                 % (px, top - 8, px, top + n * row_h, dash))
        s.append('<text x="%.1f" y="%d" fill="#888780" font-size="11" text-anchor="middle">%.1fx</text>'
                 % (px, top + n * row_h + 14, xv))

    bar_h = max(6, 12 - 3 * (len(groups) - 1))
    for i, c in enumerate(conds):
        y = top + i * row_h
        lab = "%s %s" % (KIND_CN.get(c["kind"], c["kind"]), level_label(c["kind"], c["level"]))
        s.append('<text x="%d" y="%d" fill="#444441" font-size="12" text-anchor="end">%s</text>'
                 % (lx, y + 12, lab))
        for gi, (gname, gcolor, gdata) in enumerate(groups):
            cc = find(gdata, c["kind"], c["level"])
            if not cc:
                continue
            ratio = cc["ratio_vs_clean"]
            w = max(2.0, (ratio - 1.0) * per_unit)
            by = y + 2 + gi * (bar_h + 2)
            s.append('<rect x="%d" y="%d" width="%.1f" height="%d" rx="2" fill="%s"/>'
                     % (x0, by, w, bar_h, gcolor))
            s.append('<text x="%.1f" y="%d" fill="#5F5E5A" font-size="11">%.2fx</text>'
                     % (x0 + w + 5, by + bar_h - 1, ratio))

    bl = primary.get("baseline_l1_step0")
    eps = primary.get("episodes") or []
    ep_str = "ep%d-%d" % (min(eps), max(eps)) if eps else "留出集"
    s.append('<text x="40" y="%d" fill="#5F5E5A" font-size="12">'
             '每个条件 %d 帧（%s，stride=%d）；微调后的干净图像基线 单步L1 = %.2f</text>'
             % (h - 30, primary.get("n_frames", 0), ep_str,
                primary.get("stride", 1), bl if bl else 0))
    s.append('</svg>')

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(s))
    print("已生成:", out, os.path.getsize(out), "bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
