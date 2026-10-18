"""生成 figs/error_timeline.svg：逐帧误差随任务进度的四联图。

Input:
    文件顶部的硬编码数组，由服务器脚本 d6_error_timeline.py 产出后抄回；
    图形文件不便跨机回传，故采用「取数 + 本地重绘」。

Output:
    figs/error_timeline.svg

Notes:
    绿色虚线为最后三分之一分界；该处误差下降源于机械臂趋于静止，而非模型变好。
"""
import os
# 逐帧单步误差：base（未微调）与 A40k（微调 40k 步），episode 40 与 45 各一组
BASE_40_S0 = [3.194,2.519,4.131,3.009,6.105,5.55,5.251,8.461,4.921,3.568,3.728,3.689,2.91,6.155,2.974,6.735,9.027,7.502,6.359,8.788,4.606,7.338,3.701,3.14,2.735]
A40K_40_S0 = [1.035,0.461,0.328,0.414,0.704,3.774,5.747,8.996,5.267,5.625,5.593,5.372,6.205,7.716,3.329,3.013,3.954,1.737,1.47,3.263,2.213,3.92,2.156,0.8,0.581]
BASE_40_CH = [11.275,14.632,20.218,23.512,24.064,20.493,22.134,23.553,14.838,13.6,11.763,14.057,20.227,23.28,27.982,25.845,26.805,20.51,19.754,20.393,16.839,17.259,8.712,4.892,2.932]
A40K_40_CH = [5.491,3.5,4.841,11.822,14.673,13.338,13.714,15.108,17.005,14.038,18.495,11.742,7.831,8.855,7.919,8.255,5.987,4.908,5.505,5.76,3.15,4.142,1.579,0.996,0.683]

BASE_45_S0 = [2.065,3.73,2.587,2.351,5.409,8.775,3.682,7.736,4.574,3.582,3.279,3.154,5.945,7.007,8.001,11.31,5.437,4.628,6.918,6.736,4.478,3.711,1.989,2.341,2.903]
A40K_45_S0 = [1.859,2.265,1.121,1.359,0.47,3.881,6.851,7.356,4.853,4.902,5.251,4.92,2.386,3.681,3.815,2.361,1.978,1.77,1.732,1.98,0.502,1.438,0.603,0.437,0.549]
BASE_45_CH = [11.439,18.237,18.271,19.284,27.253,27.576,27.196,15.108,14.727,17.159,24.213,30.967,29.316,23.391,18.742,22.556,14.853,25.047,20.745,15.974,15.732,9.553,7.652,5.759,3.94]
A40K_45_CH = [11.795,9.248,11.545,10.269,9.897,17.434,14.23,10.238,8.067,7.203,10.781,16.777,9.357,6.692,7.318,5.475,10.206,6.818,3.675,2.48,2.176,1.679,0.958,0.82,0.576]

# 每条曲线 25 个采样点；PANELS 里是 (x0, x1, 标题, 基线数组, 微调数组, y 轴上限)
N = 25
PANELS = [
    (40, 320, "episode 40 - single step", BASE_40_S0, A40K_40_S0, 12),
    (360, 640, "episode 40 - chunk (50 steps)", BASE_40_CH, A40K_40_CH, 32),
    (40, 320, "episode 45 - single step", BASE_45_S0, A40K_45_S0, 12),
    (360, 640, "episode 45 - chunk (50 steps)", BASE_45_CH, A40K_45_CH, 32),
]
ROWS = [(42, 166), (214, 338)]

S = []
a = S.append

a('<svg viewBox="0 0 680 372" width="100%" role="img" xmlns="http://www.w3.org/2000/svg">')
a('<title>Per-frame prediction error over the course of two held-out episodes</title>')
a('<desc>Four line charts showing absolute action error as a function of task progress, comparing the unfinetuned base model with the model finetuned for 40000 steps. Error falls sharply in the final third of every episode because the robot is coming to rest.</desc>')
a('<line x1="40" y1="14" x2="66" y2="14" stroke="#D85A30" stroke-width="2" stroke-dasharray="4 3"/>')
a('<text x="72" y="18" fill="#993C1D" font-size="11">base: not finetuned</text>')
a('<line x1="200" y1="14" x2="226" y2="14" stroke="#185FA5" stroke-width="2"/>')
a('<text x="232" y="18" fill="#0C447C" font-size="11">finetuned A_40000</text>')
a('<text x="470" y="18" fill="#888780" font-size="11">x axis = progress through the episode (0 to 100 per cent)</text>')


# 把数值数组折成 SVG polyline 的坐标串（x 均匀分布，y 按 vmax 归一）
def poly(vals, x0, x1, y0, y1, vmax):
    pts = []
    for i, v in enumerate(vals):
        x = x0 + (i / float(N - 1)) * (x1 - x0)
        y = y1 - (v / float(vmax)) * (y1 - y0)
        pts.append("%.1f,%.1f" % (x, y))
    return " ".join(pts)


# 逐个面板画：底板 → 网格线 → 标题 → 三分线 → 两条曲线
for idx, (x0, x1, title, bvals, avals, vmax) in enumerate(PANELS):
    y0, y1 = ROWS[idx // 2]
    a('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="6" fill="#FBFBF9" stroke="#E8E6DF" stroke-width="0.6"/>' % (x0, y0, x1 - x0, y1 - y0))
    for f in (0.25, 0.5, 0.75):
        yy = y0 + f * (y1 - y0)
        a('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#E8E6DF" stroke-width="0.5" stroke-dasharray="2 3"/>' % (x0, yy, x1, yy))
    a('<text x="%.1f" y="%.1f" fill="#2C2C2A" font-size="11">%s</text>' % (x0, y0 - 7, title))
    a('<text x="%.1f" y="%.1f" fill="#888780" font-size="11">%d</text>' % (x0 - 14, y0 + 4, vmax))
    a('<text x="%.1f" y="%.1f" fill="#888780" font-size="11">0</text>' % (x0 - 14, y1))
    third = x0 + (x1 - x0) * 2.0 / 3.0
    a('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#9FE1CB" stroke-width="1" stroke-dasharray="3 3"/>' % (third, y0, third, y1))
    a('<polyline points="%s" fill="none" stroke="#D85A30" stroke-width="1.8" stroke-dasharray="4 3" stroke-linejoin="round"/>' % poly(bvals, x0, x1, y0, y1, vmax))
    a('<polyline points="%s" fill="none" stroke="#185FA5" stroke-width="1.8" stroke-linejoin="round"/>' % poly(avals, x0, x1, y0, y1, vmax))

a('<text x="40" y="358" fill="#5F5E5A" font-size="11">Green dashed line = start of the final third. Error drops there in every panel because the arm is settling, not because we got better.</text>')
a('</svg>')

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "error_timeline.svg")

svg = "".join(S)
with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    f.write(svg)
print("written:", OUT, len(svg), "bytes")
