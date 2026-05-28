from __future__ import annotations

from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont


OUT = Path(__file__).with_name("pauwels-vanhoutte-scaps-interface-srh-simple.png")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/STHeiti Medium.ttc" if bold else "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


F_TITLE = font(46, True)
F_H = font(30, True)
F_BODY = font(24)
F_SMALL = font(20)
F_MATH = font(23)


def wrapped(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], width: int, fill: str, fnt: ImageFont.FreeTypeFont, line_gap: int = 8) -> int:
    # Use conservative character wrapping; mixed Chinese/English scientific
    # labels remain readable without needing browser text layout.
    lines: list[str] = []
    for para in text.split("\n"):
        if not para:
            lines.append("")
            continue
        current = ""
        for ch in para:
            trial = current + ch
            if draw.textbbox((0, 0), trial, font=fnt)[2] <= width:
                current = trial
            else:
                if current:
                    lines.append(current)
                current = ch
        if current:
            lines.append(current)
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=fnt, fill=fill)
        y += fnt.size + line_gap
    return y


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], fill: str = "#263238", width: int = 5) -> None:
    draw.line([start, end], fill=fill, width=width)
    x1, y1 = start
    x2, y2 = end
    if abs(x2 - x1) >= abs(y2 - y1):
        sign = 1 if x2 >= x1 else -1
        pts = [(x2, y2), (x2 - sign * 18, y2 - 10), (x2 - sign * 18, y2 + 10)]
    else:
        sign = 1 if y2 >= y1 else -1
        pts = [(x2, y2), (x2 - 10, y2 - sign * 18), (x2 + 10, y2 - sign * 18)]
    draw.polygon(pts, fill=fill)


def rounded(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: str, outline: str, radius: int = 18, width: int = 3) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def main() -> None:
    W, H = 1800, 1280
    img = Image.new("RGB", (W, H), "#F6F8FA")
    draw = ImageDraw.Draw(img)

    draw.text((70, 46), "SCAPS 界面缺陷处理：Pauwels-Vanhoutte SRH 的物理意义", font=F_TITLE, fill="#1D2733")
    draw.text((72, 103), "简化理解图：加入论文提取的界面密度/界面 SRH 结构；V1/V2 是 band-bending variables，不是 local grid Δφ", font=F_SMALL, fill="#596675")

    # Panel 1
    rounded(draw, (70, 150, 1730, 535), "#FFFFFF", "#D4DDE7")
    draw.text((100, 182), "1. 物理图像：界面态在 heterojunction plane 复合电子和空穴", font=F_H, fill="#1D2733")

    draw.rectangle((130, 245, 820, 460), fill="#E8F4FF", outline="#B7CBE0", width=3)
    draw.rectangle((820, 245, 1510, 460), fill="#FFF0CC", outline="#DFBF68", width=3)
    draw.line((820, 225, 820, 485), fill="#111820", width=5)
    draw.text((350, 260), "PVK / absorber side", font=F_BODY, fill="#263238", anchor="mm")
    draw.text((1180, 260), "ETL / transport side", font=F_BODY, fill="#263238", anchor="mm")
    draw.text((835, 222), "interface", font=F_SMALL, fill="#263238")

    # bands
    draw.line([(170, 340), (430, 350), (700, 385), (820, 392), (950, 345), (1240, 325), (1470, 318)], fill="#0B5CAD", width=8, joint="curve")
    draw.line([(170, 430), (430, 420), (700, 435), (820, 442), (950, 405), (1240, 390), (1470, 384)], fill="#C43D3D", width=8, joint="curve")
    draw.line([(170, 375), (430, 374), (700, 378), (805, 380)], fill="#3D8B3D", width=4)
    draw.line([(835, 365), (1020, 357), (1240, 352), (1470, 350)], fill="#3D8B3D", width=4)

    draw.text((155, 318), "E_C", font=F_SMALL, fill="#0B5CAD")
    draw.text((155, 440), "E_V", font=F_SMALL, fill="#C43D3D")
    draw.text((420, 400), "E_Fn / E_Fp", font=F_SMALL, fill="#3D8B3D")

    rounded(draw, (930, 270, 1635, 440), "#F8FBFF", "#B9C7D5", radius=14)
    wrapped(
        draw,
        "Pauwels-Vanhoutte 的关键不是再拿 local Δφ 手动投影，"
        "而是用界面势垒/带弯曲变量 V1,V2 定义 interface-plane carrier densities。"
        "这些 V1,V2 在 heavy-doping 极限由 band offsets/结构决定，而不是随 ETL bulk N_D 强烈变化。",
        (955, 295),
        640,
        "#263238",
        F_BODY,
        7,
    )

    # Panel 2
    rounded(draw, (70, 570, 1730, 930), "#FFFFFF", "#D4DDE7")
    draw.text((100, 602), "2. 公式位置：先求 interface-plane density，再算 per-side interface SRH", font=F_H, fill="#1D2733")

    box_y = 662
    rounded(draw, (110, box_y, 470, box_y + 110), "#EEF6FF", "#7AADDD")
    draw.text((135, box_y + 28), "Band bending", font=F_H, fill="#1D2733")
    draw.text((135, box_y + 70), "求 V1, V2", font=F_BODY, fill="#263238")
    arrow(draw, (470, box_y + 55), (555, box_y + 55), "#263238")

    rounded(draw, (555, box_y, 970, box_y + 110), "#FFF8E1", "#E0A722")
    draw.text((580, box_y + 28), "Interface densities", font=F_H, fill="#1D2733")
    draw.text((580, box_y + 70), "n_2s, p_1s, p_2s", font=F_MATH, fill="#263238")
    arrow(draw, (970, box_y + 55), (1055, box_y + 55), "#263238")

    rounded(draw, (1055, box_y, 1440, box_y + 110), "#F3FFF2", "#7CBF74")
    draw.text((1080, box_y + 28), "PV interface SRH", font=F_H, fill="#1D2733")
    draw.text((1080, box_y + 70), "j_s1, j_s2", font=F_BODY, fill="#263238")
    arrow(draw, (1440, box_y + 55), (1530, box_y + 55), "#263238")

    rounded(draw, (1530, box_y, 1685, box_y + 110), "#F8FBFD", "#B8C3CF")
    draw.text((1550, box_y + 32), "RHS", font=F_H, fill="#1D2733")
    draw.text((1550, box_y + 72), "dn,dp sink", font=F_SMALL, fill="#263238")

    rounded(draw, (110, 790, 820, 895), "#F8FBFF", "#B9C7D5", radius=14)
    draw.text((135, 818), "Interface-plane densities (Eq. 8, 9, 11)", font=F_BODY, fill="#1D2733")
    draw.text((150, 850), "n_2s = n_2 exp(-V_2)       p_1s = p_1 exp(-V_1)", font=F_MATH, fill="#263238")
    draw.text((150, 878), "p_2s = p_2 exp(V_2 + V)", font=F_MATH, fill="#263238")

    rounded(draw, (860, 790, 1685, 895), "#FFF8E1", "#D6A400", radius=14)
    draw.text((885, 818), "Per-side interface SRH rates (Eq. 12, 13)", font=F_BODY, fill="#1D2733")
    draw.text((900, 850), "j_s1 = s_1 [ n_1s - n_1 exp(V_1) ]", font=F_MATH, fill="#263238")
    draw.text((900, 878), "j_s2 = s_2 [ n_2s - n_2 exp(-V_2 - V) ]", font=F_MATH, fill="#263238")

    # Panel 3
    rounded(draw, (70, 965, 1730, 1190), "#FFFFFF", "#D4DDE7")
    draw.text((100, 997), "3. 数值算法：把界面面复合率放进离散 continuity equation", font=F_H, fill="#1D2733")
    wrapped(
        draw,
        "在 SolarLab/SCAPS 这类 drift-diffusion 求解中，先解 φ(x), n(x), p(x)。"
        "对界面格点 idx，使用 Pauwels-Vanhoutte 的 V1/V2 定义界面平面密度并计算 j_s。"
        "若在 finite-volume / method-of-lines 中实现，可把 surface rate 转成局部体源项：R_vol = j_s / Δx_cell，"
        "再在 RHS 中执行 dn[idx]/dt -= R_vol, dp[idx]/dt -= R_vol。",
        (115, 1045),
        1120,
        "#263238",
        F_BODY,
        7,
    )

    rounded(draw, (1280, 1030, 1685, 1138), "#FFF3CD", "#D6A400", radius=14)
    wrapped(
        draw,
        "关键修正：BBD 失败是因为用了 local Δφ；PV heavy-doping limit 的 V1/V2 不应随 N_D_ETL 强烈放大。",
        (1305, 1055),
        350,
        "#263238",
        F_SMALL,
        6,
    )

    img.save(OUT, "PNG")
    print(OUT)


if __name__ == "__main__":
    main()
