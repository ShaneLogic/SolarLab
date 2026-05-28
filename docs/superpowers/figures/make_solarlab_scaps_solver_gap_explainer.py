from __future__ import annotations

from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont


OUT = Path(__file__).with_name("solarlab_scaps_solver_gap_explainer.png")

W, H = 1920, 1080

FONT_CN = "/System/Library/Fonts/STHeiti Medium.ttc"
FONT_CN_LIGHT = "/System/Library/Fonts/STHeiti Light.ttc"
FONT_MONO = "/System/Library/Fonts/SFNSMono.ttf"


def font(size: int, *, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    if mono:
        try:
            return ImageFont.truetype(FONT_MONO, size)
        except OSError:
            return ImageFont.truetype(FONT_CN_LIGHT, size)
    return ImageFont.truetype(FONT_CN if bold else FONT_CN_LIGHT, size)


F_TITLE = font(43, bold=True)
F_SUB = font(25)
F_SEC = font(31, bold=True)
F_H = font(24, bold=True)
F_BODY = font(20)
F_SMALL = font(17)
F_TINY = font(15)
F_MONO = font(17, mono=True)
F_MONO_SMALL = font(14, mono=True)


def rounded(draw: ImageDraw.ImageDraw, xy, fill, outline=None, width=1, radius=18):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def text(draw, xy, s, f, fill="#0f172a", anchor=None):
    draw.text(xy, s, font=f, fill=fill, anchor=anchor)


def wrap_text(draw, s, f, max_w):
    out = []
    for para in s.split("\n"):
        line = ""
        for ch in para:
            cand = line + ch
            if draw.textlength(cand, font=f) <= max_w:
                line = cand
            else:
                if line:
                    out.append(line)
                line = ch
        if line:
            out.append(line)
    return out


def multiline(draw, x, y, s, f, fill="#1e293b", max_w=600, leading=8):
    yy = y
    for line in wrap_text(draw, s, f, max_w):
        draw.text((x, yy), line, font=f, fill=fill)
        yy += f.size + leading
    return yy


def arrow(draw, start, end, color, width=4):
    draw.line([start, end], fill=color, width=width)
    x1, y1 = start
    x2, y2 = end
    if abs(y2 - y1) >= abs(x2 - x1):
        if y2 >= y1:
            pts = [(x2, y2), (x2 - 14, y2 - 30), (x2 + 14, y2 - 30)]
        else:
            pts = [(x2, y2), (x2 - 14, y2 + 30), (x2 + 14, y2 + 30)]
    else:
        if x2 >= x1:
            pts = [(x2, y2), (x2 - 30, y2 - 14), (x2 - 30, y2 + 14)]
        else:
            pts = [(x2, y2), (x2 + 30, y2 - 14), (x2 + 30, y2 + 14)]
    draw.polygon(pts, fill=color)


def box(draw, xy, title, body, fill, outline, accent=None, mono=None):
    x1, y1, x2, y2 = xy
    rounded(draw, xy, fill, outline, 2, 14)
    text(draw, (x1 + 22, y1 + 20), title, F_H, "#0f172a")
    yy = y1 + 58
    if mono:
        text(draw, (x1 + 22, yy), mono, F_MONO, "#0f172a")
        yy += 32
    yy = multiline(draw, x1 + 22, yy, body, F_SMALL, "#334155", x2 - x1 - 44, 7)
    if accent:
        text(draw, (x1 + 22, y2 - 24), accent, F_TINY, "#991b1b")


def compact_badge(draw, xy, title, line1, line2, fill, outline):
    x1, y1, x2, y2 = xy
    rounded(draw, xy, fill, outline, 2, 12)
    text(draw, (x1 + 18, y1 + 16), title, font(21, bold=True), "#0f172a")
    text(draw, (x1 + 18, y1 + 48), line1, F_TINY, "#334155")
    text(draw, (x1 + 18, y1 + 70), line2, F_TINY, "#334155")


def draw_node_schematic_current(draw, x, y, scale=1.0):
    w = int(620 * scale)
    h = int(94 * scale)
    rounded(draw, (x, y, x + w, y + h), "#ffffff", "#cbd5e1", 2, 14)
    text(draw, (x + 22, y + 18), "当前离散图像：界面 face 上只有一个 SG flux", F_SMALL, "#334155")
    yy = y + 63
    draw.line([(x + 80, yy), (x + w - 80, yy)], fill="#475569", width=4)
    for cx, col in [(x + 150, "#3b82f6"), (x + 260, "#3b82f6"), (x + 365, "#ef4444"), (x + 470, "#ef4444")]:
        draw.ellipse((cx - 13, yy - 13, cx + 13, yy + 13), fill=col)
    draw.line([(x + 310, y + 40), (x + 310, y + h - 10)], fill="#94a3b8", width=2)
    draw.arc((x + 250, y + 32, x + 385, y + 102), 190, 350, fill="#dc2626", width=5)
    arrow(draw, (x + 330, y + 84), (x + 370, y + 84), "#dc2626", 4)
    text(draw, (x + 130, y + 76), "PVK", F_MONO_SMALL, "#334155")
    text(draw, (x + 450, y + 76), "ETL", F_MONO_SMALL, "#334155")


def draw_node_schematic_target(draw, x, y, scale=1.0):
    w = int(620 * scale)
    h = int(94 * scale)
    rounded(draw, (x, y, x + w, y + h), "#ffffff", "#cbd5e1", 2, 14)
    text(draw, (x + 22, y + 18), "目标离散图像：bulk 与 interface plane 分开", F_SMALL, "#334155")
    yy = y + 63
    draw.line([(x + 70, yy), (x + 235, yy)], fill="#475569", width=4)
    draw.line([(x + 385, yy), (x + 550, yy)], fill="#475569", width=4)
    for cx, col in [(x + 125, "#3b82f6"), (x + 235, "#3b82f6"), (x + 385, "#ef4444"), (x + 495, "#ef4444")]:
        draw.ellipse((cx - 13, yy - 13, cx + 13, yy + 13), fill=col)
    rounded(draw, (x + 278, y + 44, x + 342, y + 96), "#fef3c7", "#f59e0b", 2, 10)
    text(draw, (x + 310, y + 60), "n1s,p1s", F_MONO_SMALL, "#0f172a", anchor="mm")
    text(draw, (x + 310, y + 82), "n2s,p2s", F_MONO_SMALL, "#0f172a", anchor="mm")
    arrow(draw, (x + 235, yy), (x + 278, yy), "#059669", 4)
    arrow(draw, (x + 385, yy), (x + 342, yy), "#059669", 4)
    text(draw, (x + 100, y + 76), "PVK bulk", F_MONO_SMALL, "#334155")
    text(draw, (x + 460, y + 76), "ETL bulk", F_MONO_SMALL, "#334155")


def main():
    im = Image.new("RGB", (W, H), "#f8fbff")
    d = ImageDraw.Draw(im)

    # soft background bands
    d.rectangle((0, 0, W, H), fill="#f8fbff")
    d.polygon([(0, 620), (W, 440), (W, H), (0, H)], fill="#eff8f1")
    d.polygon([(1220, 0), (W, 0), (W, H), (1550, H)], fill="#fff7ed")

    text(d, (70, 52), "SolarLab vs SCAPS：差异来自界面求解结构", F_TITLE)
    text(d, (70, 104), "核心不是单个参数没调好，而是 heterointerface carrier density 的定义不同", F_SUB, "#334155")

    # metric badges: below the subtitle, safely separated from the title.
    compact_badge(d, (70, 138, 340, 225), "ETL doping", "SolarLab ≈ 1095 mV", "SCAPS ≈ 137 mV, 约 8x over", "#fffbeb", "#f59e0b")
    compact_badge(d, (365, 138, 635, 225), "CBO response", "约 83-85% closure", "band offset 基本对", "#ecfdf5", "#34d399")
    compact_badge(d, (660, 138, 990, 225), "核心 gap", "interface-plane density", "不是 bulk density", "#fef2f2", "#f87171")

    # panels
    rounded(d, (70, 250, 870, 892), "#ffffff", "#cbd5e1", 2, 22)
    rounded(d, (1050, 250, 1850, 892), "#ffffff", "#cbd5e1", 2, 22)
    rounded(d, (900, 325, 1020, 775), "#ffffff", "#cbd5e1", 2, 22)

    text(d, (112, 296), "当前 SolarLab 路径", F_SEC)
    text(d, (112, 330), "MoL + Scharfetter-Gummel bulk flux + Radau transient", F_SMALL, "#334155")
    text(d, (1092, 296), "目标 SCAPS-like 路径", F_SEC)
    text(d, (1092, 330), "steady-state interface-plane equations + TE/QSS constraints", F_SMALL, "#334155")

    # middle
    for i, s in enumerate(["差异", "来自", "界面", "定义"]):
        text(d, (960, 370 + i * 42), s, F_H, "#0f172a", anchor="mm")
    arrow(d, (960, 548), (960, 642), "#dc2626", 6)
    for i, s in enumerate(["不是", "简单", "调参"]):
        text(d, (960, 680 + i * 31), s, F_H, "#991b1b", anchor="mm")

    box(d, (112, 365, 825, 455), "1. 统一网格上的连续 bulk 状态",
        "每个节点只有一组 n, p, φ；界面只是左右 layer 的交界 face/node。",
        "#eff6ff", "#93c5fd")
    arrow(d, (470, 455), (470, 492), "#2563eb", 5)
    box(d, (112, 492, 825, 602), "2. SG flux 跨界面直接耦合",
        "χ step 被塞进同一个 face flux；默认倾向维持连续的 bulk quasi-Fermi 结构。",
        "#f8fafc", "#dbe3ec", mono="J_n[f] = SG(φ + χ, n_L, n_R, D_face)")
    arrow(d, (470, 602), (470, 632), "#dc2626", 5)
    box(d, (112, 632, 825, 780), "3. interface SRH 只能采样 bulk / 近邻 density",
        "低 ETL doping 时 local Δφ 与 N_D 强耦合。",
        "#fef2f2", "#fca5a5",
        accent="结果：ETL 被放大；修正后 CBO/N_t 又容易崩。",
        mono="R_s ≈ SRH(n_bulk-side, p_bulk-side, N_t, σ)")
    draw_node_schematic_current(d, 160, 792)

    box(d, (1092, 365, 1805, 455), "1. 界面有独立 interface-plane unknowns",
        "n1s, p1s, n2s, p2s 与 bulk 节点分开，不强迫跨界面连续。",
        "#ecfdf5", "#86efac")
    arrow(d, (1450, 455), (1450, 492), "#059669", 5)
    box(d, (1092, 492, 1805, 602), "2. TE 边界 + χ-step / Q-Fermi step",
        "SCAPS 通过 analytical V1/V2 与 per-side quasi-Fermi level 定义界面 density。",
        "#f5f3ff", "#c4b5fd", mono="J_TE = v_th * (n_bulk*exp(-V_i/V_T) - n_is)")
    arrow(d, (1450, 602), (1450, 632), "#059669", 5)
    box(d, (1092, 632, 1805, 780), "3. QSS / Newton-Krylov 解 steady-state 界面方程",
        "界面 recombination 使用 interface-plane density，不再用 bulk proxy。",
        "#ecfdf5", "#86efac",
        accent="目标：同时保留 CBO、N_t 与 ETL doping 的正确尺度。",
        mono="0 = TE refill - SRH(n1s,p2s) - cross-interface transport")
    draw_node_schematic_target(d, 1140, 792)

    # bottom summary
    rounded(d, (70, 925, 1850, 1045), "#0f172a", None, 0, 24)
    text(d, (112, 956), "实验证据：7 个 prototype 都在同一结构矛盾上失败", F_H, "#ffffff")
    text(d, (112, 990), "BBD 修 CBO 但 ETL 更差；thin-shell 修 ETL 但 CBO 崩；iface-state/χ-step 让 N_t 崩或刚性太强；split-flux raw drain ≈ 1e36 m^-3/s，Newton >24 min 挂起。", F_TINY, "#cbd5e1")
    text(d, (112, 1022), "结论：YAML/loader 修正是必要的；若要复现 SCAPS，需要新的 steady-state interface-plane 求解路径，而不是继续在当前 Radau/MoL 路径上加局部 correction factor。", F_TINY, "#fef3c7")

    im.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
