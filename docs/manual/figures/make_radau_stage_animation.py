from __future__ import annotations

from math import atan2, cos, sin
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "docs" / "manual" / "figures"
GIF_PATH = OUT_DIR / "radau_stage_process.gif"
PNG_PATH = OUT_DIR / "radau_stage_process_preview.png"

W, H = 1280, 720

C = {
    "bg": "#F7F9FC",
    "white": "#FFFFFF",
    "navy": "#162033",
    "slate": "#334155",
    "muted": "#64748B",
    "line": "#CBD5E1",
    "grid": "#E2E8F0",
    "blue": "#2563EB",
    "blue_light": "#DBEAFE",
    "green": "#16803C",
    "green_light": "#DCFCE7",
    "amber": "#A16207",
    "amber_light": "#FEF3C7",
    "purple": "#6D28D9",
    "purple_light": "#EDE9FE",
    "red": "#B42318",
    "red_light": "#FEE4E2",
}


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


F = {
    "h1": _font(30, bold=True),
    "h2": _font(24, bold=True),
    "label": _font(20, bold=True),
    "body": _font(19),
    "small": _font(16),
    "tiny": _font(14),
    "math": _font(22),
    "math_bold": _font(22, bold=True),
    "math_small": _font(18),
    "sub": _font(12),
}


def rounded(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], fill: str, outline: str, width: int = 2, radius: int = 18) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def text_center(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], text: str, font: ImageFont.FreeTypeFont, fill: str, spacing: int = 5) -> None:
    box = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing, align="center")
    tw = box[2] - box[0]
    th = box[3] - box[1]
    x = xy[0] + (xy[2] - xy[0] - tw) / 2
    y = xy[1] + (xy[3] - xy[1] - th) / 2
    draw.multiline_text((x, y), text, font=font, fill=fill, spacing=spacing, align="center")


def sub_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    base: str,
    sub: str,
    fill: str,
    *,
    font: ImageFont.FreeTypeFont | None = None,
    sub_font: ImageFont.FreeTypeFont | None = None,
) -> int:
    font = font or F["math"]
    sub_font = sub_font or F["sub"]
    x, y = xy
    draw.text((x, y), base, font=font, fill=fill)
    bbox = draw.textbbox((x, y), base, font=font)
    bx = bbox[2] - bbox[0]
    draw.text((x + bx + 1, y + int(font.size * 0.52)), sub, font=sub_font, fill=fill)
    sb = draw.textbbox((x + bx + 1, y + int(font.size * 0.52)), sub, font=sub_font)
    return sb[2] - x


def arrow(draw: ImageDraw.ImageDraw, start: tuple[float, float], end: tuple[float, float], color: str, width: int = 3) -> None:
    draw.line([start, end], fill=color, width=width)
    angle = atan2(end[1] - start[1], end[0] - start[0])
    length = 14
    spread = 0.42
    p1 = (end[0] - length * cos(angle - spread), end[1] - length * sin(angle - spread))
    p2 = (end[0] - length * cos(angle + spread), end[1] - length * sin(angle + spread))
    draw.polygon([end, p1, p2], fill=color)


def draw_sub_expr(draw: ImageDraw.ImageDraw, xy: tuple[int, int], parts: list[tuple[str, str | None]], fill: str, font: ImageFont.FreeTypeFont | None = None) -> int:
    font = font or F["math"]
    x, y = xy
    for base, sub in parts:
        if sub is None:
            draw.text((x, y), base, font=font, fill=fill)
            bb = draw.textbbox((x, y), base, font=font)
            x = bb[2]
        else:
            width = sub_label(draw, (x, y), base, sub, fill, font=font)
            x += width
    return x - xy[0]


def stage_positions() -> list[tuple[str, float, float, str]]:
    return [
        ("K", 0.155, 1.0, "1"),
        ("K", 0.645, 2.0, "2"),
        ("K", 1.000, 3.0, "3"),
    ]


def draw_timeline(draw: ImageDraw.ImageDraw, active: int) -> None:
    panel = (50, 38, 1230, 255)
    rounded(draw, panel, C["white"], C["line"], width=2, radius=18)
    draw.text((78, 63), "One internal Radau step", font=F["h1"], fill=C["slate"])
    draw.text((78, 101), "Radau advances the state from the beginning to the end of one adaptive solver step.", font=F["body"], fill=C["muted"])

    x0, x1, y = 180, 1110, 174
    draw.line([(x0, y), (x1, y)], fill=C["slate"], width=3)
    arrow(draw, (x1 - 12, y), (x1 + 10, y), C["slate"], width=3)
    draw.line([(x0, y - 28), (x0, y + 28)], fill=C["slate"], width=3)
    draw.line([(x1, y - 28), (x1, y + 28)], fill=C["slate"], width=3)
    draw_sub_expr(draw, (x0 - 24, y + 58), [("t", "n")], C["slate"], font=F["small"])
    draw_sub_expr(draw, (x1 - 65, y + 58), [("t", "n"), (" + h", None)], C["slate"], font=F["small"])
    draw_sub_expr(draw, (x0 - 41, y - 52), [("Y", "n")], C["blue"], font=F["math_bold"])
    draw_sub_expr(draw, (x1 - 54, y - 52), [("Y", "n+1")], C["green"], font=F["math_bold"])

    for idx, (_, c, _, sub) in enumerate(stage_positions(), start=1):
        sx = x0 + c * (x1 - x0)
        fill = C["purple_light"] if idx != active else C["amber_light"]
        edge = C["purple"] if idx != active else C["amber"]
        draw.line([(sx, y - 18), (sx, y + 18)], fill=edge, width=2)
        draw.ellipse((sx - 29, y - 29, sx + 29, y + 29), fill=fill, outline=edge, width=4 if idx == active else 2)
        text_center(draw, (int(sx - 24), y - 24, int(sx + 24), y + 24), f"K{sub}", F["label"], edge)
        c_label_y = y - 60 if idx == 3 else y + 38
        draw.text((sx - 38, c_label_y), f"c{sub}={c:.3g}", font=F["tiny"], fill=C["muted"])


def draw_equation_panel(draw: ImageDraw.ImageDraw, active: int) -> None:
    panel = (50, 285, 705, 615)
    rounded(draw, panel, C["white"], C["line"], width=2, radius=18)
    draw.text((78, 310), "Implicit stage equations", font=F["h2"], fill=C["slate"])
    draw.text((78, 343), "The stage slopes are solved together because each stage uses the others.", font=F["body"], fill=C["muted"])

    rounded(draw, (78, 388, 677, 492), C["purple_light"], C["purple"], width=2, radius=16)
    draw.text((104, 411), "For each stage:", font=F["math_bold"], fill=C["purple"])
    draw.text((104, 448), "build a trial state  ->  call F(Y)  ->  obtain a slope K", font=F["math_small"], fill=C["navy"])

    labels = [
        ("1", "stage location", "inside [t_n, t_n+h]"),
        ("2", "trial state", "uses all K stages"),
        ("3", "RHS call", "F gives a slope"),
    ]
    for i, (num, title, body) in enumerate(labels):
        x = 88 + 195 * i
        fill = C["amber_light"] if i + 1 == active else C["bg"]
        edge = C["amber"] if i + 1 == active else C["line"]
        rounded(draw, (x, 522, x + 178, 584), fill, edge, width=3 if i + 1 == active else 1, radius=12)
        draw.ellipse((x + 13, 539, x + 43, 569), fill=edge, outline=edge)
        text_center(draw, (x + 13, 539, x + 43, 569), num, F["small"], C["white"])
        draw.text((x + 54, 531), title, font=F["tiny"], fill=C["slate"])
        draw.text((x + 54, 553), body, font=F["tiny"], fill=C["muted"])


def draw_coupling_panel(draw: ImageDraw.ImageDraw, active: int) -> None:
    panel = (735, 285, 1230, 615)
    rounded(draw, panel, C["white"], C["line"], width=2, radius=18)
    draw.text((763, 310), "What K1-K3 mean", font=F["h2"], fill=C["slate"])
    draw.text((763, 343), "They are internal stage slopes, not J-V data points.", font=F["body"], fill=C["muted"])

    centers = [(825, 450), (982, 450), (1139, 450)]
    for i, (cx, cy) in enumerate(centers, start=1):
        fill = C["purple_light"] if i != active else C["amber_light"]
        edge = C["purple"] if i != active else C["amber"]
        draw.ellipse((cx - 46, cy - 46, cx + 46, cy + 46), fill=fill, outline=edge, width=4 if i == active else 2)
        text_center(draw, (cx - 35, cy - 33, cx + 35, cy + 10), f"K{i}", F["label"], edge)
        draw.text((cx - 37, cy + 12), f"stage {i}", font=F["tiny"], fill=C["muted"])

    for a, b in [(centers[0], centers[1]), (centers[1], centers[2]), (centers[2], centers[0])]:
        arrow(draw, (a[0] + 48 if a[0] < b[0] else a[0] - 40, a[1] - 8), (b[0] - 48 if a[0] < b[0] else b[0] + 40, b[1] - 8), C["muted"], width=2)
    draw.arc((820, 385, 1145, 540), start=200, end=340, fill=C["muted"], width=2)

    rounded(draw, (780, 540, 1188, 585), C["green_light"], C["green"], width=2, radius=13)
    text_center(draw, (790, 546, 1178, 579), "Newton iteration finds a self-consistent set of K1, K2, K3", F["small"], C["green"], spacing=2)


def draw_footer(draw: ImageDraw.ImageDraw, phase: int) -> None:
    messages = [
        "Start with Y_n and a solver-chosen internal step size h.",
        "Evaluate three implicit stages inside the same time step.",
        "The stages are coupled; Radau solves K1-K3 together.",
        "Combine weighted stages to obtain Y_{n+1}.",
    ]
    draw.rectangle((0, 650, W, H), fill="#EAF2FF")
    draw.text((52, 671), messages[phase], font=F["label"], fill=C["navy"])


def draw_frame(phase: int, active: int) -> Image.Image:
    image = Image.new("RGB", (W, H), C["bg"])
    draw = ImageDraw.Draw(image)
    draw_timeline(draw, active)
    draw_equation_panel(draw, active)
    draw_coupling_panel(draw, active)
    draw_footer(draw, phase)
    return image


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sequence = [
        (0, 1, 900),
        (1, 1, 900),
        (1, 2, 900),
        (1, 3, 900),
        (2, 1, 900),
        (2, 2, 900),
        (2, 3, 900),
        (3, 3, 1300),
    ]
    frames = [draw_frame(phase, active) for phase, active, _ in sequence]
    durations = [duration for _, _, duration in sequence]
    frames[3].save(PNG_PATH)
    frames[0].save(
        GIF_PATH,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(GIF_PATH)
    print(PNG_PATH)


if __name__ == "__main__":
    main()
