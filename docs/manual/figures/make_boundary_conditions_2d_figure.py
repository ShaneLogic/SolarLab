from __future__ import annotations

from html import escape
from pathlib import Path
import math
import textwrap

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "manual" / "figures"
PNG = OUT_DIR / "boundary_conditions_2d_overview.png"
SVG = OUT_DIR / "boundary_conditions_2d_overview.svg"

W, H = 1920, 1080
SCALE = 2

FONT_DIR = Path("/System/Library/Fonts/Supplemental")
ARIAL = FONT_DIR / "Arial Unicode.ttf"
ARIAL_BOLD = FONT_DIR / "Arial Bold.ttf"

C = {
    "bg": "#F7F9FC",
    "panel": "#FFFFFF",
    "off": "#F8FAFC",
    "line": "#CBD5E1",
    "navy": "#162033",
    "slate": "#334155",
    "muted": "#64748B",
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
    "strip": "#EAF2FF",
    "htl": "#E6F7EE",
    "abs": "#EEF5FF",
    "etl": "#FFF2D8",
}


def rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def sc(v: float) -> int:
    return int(round(v * SCALE))


def font(size: float, bold: bool = False) -> ImageFont.FreeTypeFont:
    face = ARIAL_BOLD if bold and ARIAL_BOLD.exists() else ARIAL
    return ImageFont.truetype(str(face), sc(size))


img = Image.new("RGB", (W * SCALE, H * SCALE), rgb(C["bg"]))
draw = ImageDraw.Draw(img)


def rounded_rect(
    x: float,
    y: float,
    w: float,
    h: float,
    fill: str,
    outline: str = "line",
    radius: float = 12,
    width: float = 2,
) -> None:
    draw.rounded_rectangle(
        [sc(x), sc(y), sc(x + w), sc(y + h)],
        radius=sc(radius),
        fill=rgb(C[fill] if fill in C else fill),
        outline=rgb(C[outline] if outline in C else outline),
        width=sc(width),
    )


def rect(x: float, y: float, w: float, h: float, fill: str, outline: str | None = None, width: float = 1) -> None:
    draw.rectangle(
        [sc(x), sc(y), sc(x + w), sc(y + h)],
        fill=rgb(C[fill] if fill in C else fill),
        outline=rgb(C[outline] if outline and outline in C else outline) if outline else None,
        width=sc(width),
    )


def text_size(txt: str, fnt: ImageFont.FreeTypeFont) -> tuple[int, int]:
    if not txt:
        return 0, 0
    box = draw.textbbox((0, 0), txt, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def wrap_line(line: str, fnt: ImageFont.FreeTypeFont, max_width: float) -> list[str]:
    if not line:
        return [""]
    words = line.split(" ")
    out: list[str] = []
    current = ""
    max_px = sc(max_width)
    for word in words:
        trial = word if not current else f"{current} {word}"
        if text_size(trial, fnt)[0] <= max_px:
            current = trial
        else:
            if current:
                out.append(current)
            current = word
    if current:
        out.append(current)
    return out


def draw_text(
    txt: str,
    x: float,
    y: float,
    w: float,
    size: float,
    color: str = "navy",
    bold: bool = False,
    align: str = "left",
    line_gap: float = 1.18,
) -> None:
    fnt = font(size, bold)
    lines: list[str] = []
    for part in txt.split("\n"):
        lines.extend(wrap_line(part, fnt, w))
    line_h = size * line_gap
    yy = y
    for line in lines:
        line_px = text_size(line, fnt)[0] / SCALE
        if align == "center":
            xx = x + (w - line_px) / 2
        elif align == "right":
            xx = x + w - line_px
        else:
            xx = x
        draw.text((sc(xx), sc(yy)), line, font=fnt, fill=rgb(C[color] if color in C else color))
        yy += line_h


def rich_line_size(parts: list[tuple[str, str]], base_size: float, bold: bool = False) -> tuple[float, float]:
    width = 0.0
    height = base_size
    for txt, kind in parts:
        size = base_size * (0.68 if kind in {"sub", "sup"} else 1.0)
        fnt = font(size, bold)
        w_px, h_px = text_size(txt, fnt)
        width += w_px / SCALE
        height = max(height, h_px / SCALE)
    return width, height


def draw_rich_line(
    parts: list[tuple[str, str]],
    x: float,
    y: float,
    w: float,
    base_size: float,
    color: str = "navy",
    bold: bool = False,
    align: str = "left",
) -> None:
    total_w, _ = rich_line_size(parts, base_size, bold)
    if align == "center":
        xx = x + (w - total_w) / 2
    elif align == "right":
        xx = x + w - total_w
    else:
        xx = x
    fill = rgb(C[color] if color in C else color)
    for txt, kind in parts:
        size = base_size * (0.68 if kind in {"sub", "sup"} else 1.0)
        fnt = font(size, bold)
        yy = y
        if kind == "sub":
            yy += base_size * 0.34
        elif kind == "sup":
            yy -= base_size * 0.22
        draw.text((sc(xx), sc(yy)), txt, font=fnt, fill=fill)
        xx += text_size(txt, fnt)[0] / SCALE


def arrow(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: str = "muted",
    width: float = 3,
    head: float = 12,
    dash: bool = False,
) -> None:
    col = rgb(C[color] if color in C else color)
    if dash:
        length = math.hypot(x2 - x1, y2 - y1)
        if length == 0:
            return
        ux, uy = (x2 - x1) / length, (y2 - y1) / length
        step, gap = 18, 10
        dist = 0.0
        while dist < length - head:
            a = dist
            b = min(dist + step, length - head)
            draw.line(
                [sc(x1 + ux * a), sc(y1 + uy * a), sc(x1 + ux * b), sc(y1 + uy * b)],
                fill=col,
                width=sc(width),
            )
            dist += step + gap
    else:
        draw.line([sc(x1), sc(y1), sc(x2), sc(y2)], fill=col, width=sc(width))
    ang = math.atan2(y2 - y1, x2 - x1)
    pts = [
        (x2, y2),
        (x2 - head * math.cos(ang - math.pi / 7), y2 - head * math.sin(ang - math.pi / 7)),
        (x2 - head * math.cos(ang + math.pi / 7), y2 - head * math.sin(ang + math.pi / 7)),
    ]
    draw.polygon([(sc(px), sc(py)) for px, py in pts], fill=col)


def double_arrow(x1: float, y1: float, x2: float, y2: float, color: str = "muted", width: float = 3) -> None:
    arrow(x1, y1, x2, y2, color=color, width=width)
    arrow(x2, y2, x1, y1, color=color, width=width)


def card(x: float, y: float, w: float, h: float, title: str, body: str, accent: str, fill: str) -> None:
    rounded_rect(x, y, w, h, fill, accent, radius=10, width=2)
    rect(x, y, 10, h, accent)
    draw_text(title, x + 24, y + 18, w - 42, 18, accent, bold=True)
    draw_text(body, x + 24, y + 58, w - 42, 15.2, "navy", bold=False)


def pill(x: float, y: float, w: float, label: str, fill: str, accent: str) -> None:
    rounded_rect(x, y, w, 42, fill, accent, radius=18, width=1.5)
    draw_text(label, x + 14, y + 10, w - 28, 14.5, accent, bold=True, align="center")


# Page shell
draw_text("2D Boundary Conditions in SolarLab", 68, 42, 1120, 36, "navy", bold=True)
draw_text(
    "Separate the lateral x-boundary from the transport/contact y-boundary: only some parts are user-configurable.",
    70,
    96,
    1320,
    20,
    "slate",
)
rounded_rect(60, 145, 1800, 760, "panel", "line", radius=14, width=2)


# Device diagram
dx, dy, dw, dh = 610, 280, 700, 500
rounded_rect(dx - 16, dy - 44, dw + 32, dh + 92, "off", "line", radius=12, width=2)
draw_text("2D device domain", dx + 230, dy - 30, 240, 18, "slate", bold=True, align="center")

layer_h = [115, 270, 115]
layers = [
    ("HTL / y=0 side", "htl", "green"),
    ("Absorber", "abs", "blue"),
    ("ETL / y=Ly side", "etl", "amber"),
]
yy = dy
for (label, fill, accent), hh in zip(layers, layer_h):
    rect(dx, yy, dw, hh, fill, "line", width=1)
    draw_text(label, dx + 20, yy + hh / 2 - 11, 190, 15, accent, bold=True)
    yy += hh

# subtle grid and grain-boundary cue
for i in range(1, 6):
    x = dx + i * dw / 6
    draw.line([sc(x), sc(dy), sc(x), sc(dy + dh)], fill=rgb("#D8E0EC"), width=sc(1))
for j in range(1, 5):
    y = dy + j * dh / 5
    draw.line([sc(dx), sc(y), sc(dx + dw), sc(y)], fill=rgb("#D8E0EC"), width=sc(1))
rect(dx + dw / 2 - 7, dy + 128, 14, 244, "red_light")
draw_text("example grain boundary", dx + dw / 2 + 16, dy + 366, 185, 12.5, "red")

rounded_rect(dx, dy - 42, dw, 36, "green_light", "green", radius=8, width=1.5)
draw_text("Poisson Dirichlet:  φ(y=0,x) = 0", dx + 18, dy - 32, dw - 36, 14.5, "green", bold=True, align="center")
rounded_rect(dx, dy + dh + 8, dw, 38, "amber_light", "amber", radius=8, width=1.5)
draw_rich_line(
    [
        ("Poisson Dirichlet:  φ(y=L", "normal"),
        ("y", "sub"),
        (",x) = V", "normal"),
        ("bi", "sub"),
        (" - V", "normal"),
        ("app", "sub"),
    ],
    dx + 18,
    dy + dh + 18,
    dw - 36,
    14.5,
    "amber",
    bold=True,
    align="center",
)

# Axes and boundary labels
arrow(dx + 20, dy + dh + 75, dx + 180, dy + dh + 75, "slate", width=2.5, head=11)
draw_text("x lateral", dx + 190, dy + dh + 65, 90, 12.5, "slate")
arrow(dx - 44, dy + 12, dx - 44, dy + 172, "slate", width=2.5, head=11)
draw_text("y transport", dx - 92, dy + 184, 105, 12.5, "slate")
draw_text("x=0", dx - 40, dy + dh / 2 - 10, 35, 12.5, "muted", align="right")
draw_text("x=Lx", dx + dw + 8, dy + dh / 2 - 10, 50, 12.5, "muted")

# Lateral cue on device
double_arrow(dx - 2, dy + 72, dx + dw + 2, dy + 72, "blue", width=2.2)
draw_text("periodic wrap option", dx + 245, dy + 47, 210, 12.5, "blue", bold=True, align="center")
draw.line([sc(dx), sc(dy + 380), sc(dx), sc(dy + 452)], fill=rgb(C["red"]), width=sc(7))
draw.line([sc(dx + dw), sc(dy + 380), sc(dx + dw), sc(dy + 452)], fill=rgb(C["red"]), width=sc(7))
draw_text("zero lateral flux option", dx + 238, dy + 420, 230, 12.5, "red", bold=True, align="center")

# Left panels
card(
    92,
    230,
    430,
    245,
    'x-boundary option A: periodic',
    'Selected by lateral_bc = "periodic".\nLeft and right lateral edges are coupled as a repeating unit cell.\nConceptually: u(0,y) = u(Lx,y) for phi, n, p.',
    "blue",
    "blue_light",
)
arrow(522, 342, dx - 22, dy + 72, "blue", width=2.2)

card(
    92,
    525,
    430,
    250,
    "x-boundary option B: Neumann",
    'Selected by lateral_bc = "neumann".\nNo lateral current leaves the simulated window.\nPoisson: dphi/dx = 0\nTransport: normal carrier current = 0 at x=0,Lx.',
    "red",
    "red_light",
)
arrow(522, 646, dx - 20, dy + 414, "red", width=2.2)

# Right panels
card(
    1396,
    230,
    420,
    242,
    "y-boundary: electrode potential",
    "Fixed in the current 2D solver.\nThese are Dirichlet conditions for Poisson:\ny=0:   phi = 0\ny=Ly:  phi = Vbi - Vapp\nThis is how applied voltage enters electrostatics.",
    "green",
    "green_light",
)
arrow(1396, 324, dx + dw + 30, dy - 24, "green", width=2.2)
arrow(1396, 392, dx + dw + 30, dy + dh + 28, "amber", width=2.2)

card(
    1396,
    525,
    420,
    296,
    "y-boundary: carrier contacts",
    "Configurable through S fields in mode = full.\nMissing or null S: ohmic Dirichlet\nS = 0: blocking / Neumann limit\nFinite S: Robin selective contact\nLarge S: approaches ohmic limit\nRobin form: Jcontact = ±qS(c − ceq).",
    "purple",
    "purple_light",
)
arrow(1396, 652, dx + dw + 26, dy + 26, "purple", width=2.0)
arrow(1396, 710, dx + dw + 26, dy + dh - 28, "purple", width=2.0)

# Bottom switching strip
rounded_rect(92, 845, 1724, 42, "off", "line", radius=8, width=1.4)
draw_text(
    "Code mapping:  S_*_left maps to the y=0 / HTL side;  S_*_right maps to the y=Ly / ETL side.",
    118,
    857,
    1672,
    13.2,
    "muted",
    align="center",
)

rect(0, 930, W, 150, "strip")
draw_text("Switching locations in the repo", 94, 952, 330, 17, "navy", bold=True)
pill(94, 995, 480, 'run_jv_sweep_2d(..., lateral_bc="periodic"|"neumann")', "blue_light", "blue")
pill(610, 995, 520, "YAML/UI: mode=full + Sn/Sp contact fields", "purple_light", "purple")
pill(1166, 995, 560, "Poisson electrodes: fixed Dirichlet phi = 0 and Vbi-Vapp", "amber_light", "amber")


def svg_rect(x, y, w, h, fill, stroke="line", rx=10, sw=1.5) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{C.get(fill, fill)}" stroke="{C.get(stroke, stroke)}" stroke-width="{sw}"/>'
    )


def svg_text(txt, x, y, w, size=14, color="navy", bold=False, anchor="start") -> str:
    weight = "700" if bold else "400"
    lines = txt.split("\n")
    if anchor == "middle":
        tx = x + w / 2
    elif anchor == "end":
        tx = x + w
    else:
        tx = x
    parts = [
        f'<text x="{tx:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{C.get(color, color)}" text-anchor="{anchor}">'
    ]
    for idx, line in enumerate(lines):
        dy_attr = "0" if idx == 0 else f"{size * 1.22:.1f}"
        parts.append(f'<tspan x="{tx:.1f}" dy="{dy_attr}">{escape(line)}</tspan>')
    parts.append("</text>")
    return "".join(parts)


def svg_arrow(x1, y1, x2, y2, color="muted", sw=2.2, dash=False) -> str:
    dash_attr = ' stroke-dasharray="10 7"' if dash else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{C.get(color, color)}" stroke-width="{sw}" marker-end="url(#{color}-arrow)"{dash_attr}/>'
    )


svg: list[str] = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
    f'<rect width="{W}" height="{H}" fill="{C["bg"]}"/>',
    "<defs>",
]
for name in ["muted", "blue", "green", "amber", "purple", "red", "slate"]:
    svg.append(
        f'<marker id="{name}-arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">'
        f'<path d="M0,0 L0,6 L9,3 z" fill="{C[name]}"/></marker>'
    )
svg.append("</defs>")
svg.append(svg_text("2D Boundary Conditions in SolarLab", 68, 64, 1100, 32, "navy", True))
svg.append(svg_text("Separate the lateral x-boundary from the transport/contact y-boundary: only some parts are user-configurable.", 70, 116, 1320, 19, "slate"))
svg.append(svg_rect(60, 145, 1800, 760, "panel", "line", 14, 2))
svg.append(svg_rect(dx - 16, dy - 44, dw + 32, dh + 92, "off", "line", 12, 2))
svg.append(svg_text("2D device domain", dx + 230, dy - 14, 240, 18, "slate", True, "middle"))
yy = dy
for (label, fill, accent), hh in zip(layers, layer_h):
    svg.append(f'<rect x="{dx}" y="{yy}" width="{dw}" height="{hh}" fill="{C[fill]}" stroke="{C["line"]}"/>')
    svg.append(svg_text(label, dx + 20, yy + hh / 2 + 5, 190, 15, accent, True))
    yy += hh
svg.append(svg_rect(dx, dy - 42, dw, 36, "green_light", "green", 8, 1.5))
svg.append(svg_text("Poisson Dirichlet: φ(y=0,x) = 0", dx, dy - 18, dw, 14.5, "green", True, "middle"))
svg.append(svg_rect(dx, dy + dh + 8, dw, 38, "amber_light", "amber", 8, 1.5))
svg.append(svg_text("Poisson Dirichlet: phi(y=Ly,x) = Vbi - Vapp", dx, dy + dh + 32, dw, 14.5, "amber", True, "middle"))
svg.append(svg_arrow(dx + 20, dy + dh + 75, dx + 180, dy + dh + 75, "slate", 2.5))
svg.append(svg_text("x lateral", dx + 190, dy + dh + 82, 90, 12.5, "slate"))
svg.append(svg_arrow(dx - 44, dy + 12, dx - 44, dy + 172, "slate", 2.5))
svg.append(svg_text("y transport", dx - 92, dy + 198, 105, 12.5, "slate"))
svg.append(svg_arrow(dx - 2, dy + 72, dx + dw + 2, dy + 72, "blue", 2.2))
svg.append(svg_arrow(dx + dw + 2, dy + 72, dx - 2, dy + 72, "blue", 2.2))
svg.append(svg_text("periodic wrap option", dx + 245, dy + 62, 210, 12.5, "blue", True, "middle"))
svg.append(f'<line x1="{dx}" y1="{dy+380}" x2="{dx}" y2="{dy+452}" stroke="{C["red"]}" stroke-width="7"/>')
svg.append(f'<line x1="{dx+dw}" y1="{dy+380}" x2="{dx+dw}" y2="{dy+452}" stroke="{C["red"]}" stroke-width="7"/>')
svg.append(svg_text("zero lateral flux option", dx + 238, dy + 435, 230, 12.5, "red", True, "middle"))

svg.append(svg_rect(92, 230, 430, 245, "blue_light", "blue", 10, 2))
svg.append(svg_text('x-boundary option A: periodic', 116, 266, 382, 18, "blue", True))
svg.append(svg_text('lateral_bc = "periodic"\nLeft/right edges are coupled.\nu(0,y) = u(Lx,y) for phi, n, p.', 116, 310, 382, 15.2, "navy"))
svg.append(svg_arrow(522, 342, dx - 22, dy + 72, "blue", 2.2))
svg.append(svg_rect(92, 525, 430, 250, "red_light", "red", 10, 2))
svg.append(svg_text("x-boundary option B: Neumann", 116, 561, 382, 18, "red", True))
svg.append(svg_text('lateral_bc = "neumann"\nNo lateral current leaves.\ndphi/dx = 0\nnormal carrier current = 0 at x=0,Lx.', 116, 604, 382, 15.2, "navy"))
svg.append(svg_arrow(522, 646, dx - 20, dy + 414, "red", 2.2))

svg.append(svg_rect(1396, 230, 420, 242, "green_light", "green", 10, 2))
svg.append(svg_text("y-boundary: electrode potential", 1420, 266, 372, 18, "green", True))
svg.append(svg_text("Fixed in the current 2D solver.\nDirichlet for Poisson:\ny=0: phi = 0\ny=Ly: phi = Vbi - Vapp", 1420, 310, 372, 15.2, "navy"))
svg.append(svg_arrow(1396, 324, dx + dw + 30, dy - 24, "green", 2.2))
svg.append(svg_arrow(1396, 392, dx + dw + 30, dy + dh + 28, "amber", 2.2))
svg.append(svg_rect(1396, 525, 420, 296, "purple_light", "purple", 10, 2))
svg.append(svg_text("y-boundary: carrier contacts", 1420, 561, 372, 18, "purple", True))
svg.append(svg_text("Configurable through S fields.\nMissing S: ohmic Dirichlet\nS = 0: blocking / Neumann\nFinite S: Robin selective contact\nLarge S: ohmic limit", 1420, 604, 372, 15.2, "navy"))
svg.append(svg_arrow(1396, 652, dx + dw + 26, dy + 26, "purple", 2.0))
svg.append(svg_arrow(1396, 710, dx + dw + 26, dy + dh - 28, "purple", 2.0))
svg.append(svg_rect(92, 845, 1724, 42, "off", "line", 8, 1.4))
svg.append(svg_text("Code mapping: S_*_left maps to y=0 / HTL; S_*_right maps to y=Ly / ETL.", 118, 871, 1672, 13.2, "muted", False, "middle"))
svg.append(f'<rect x="0" y="930" width="{W}" height="150" fill="{C["strip"]}"/>')
svg.append(svg_text("Switching locations in the repo", 94, 975, 330, 17, "navy", True))
for x, label, fill, accent, width in [
    (94, 'run_jv_sweep_2d(..., lateral_bc="periodic"|"neumann")', "blue_light", "blue", 480),
    (610, "YAML/UI: mode=full + Sn/Sp contact fields", "purple_light", "purple", 520),
    (1166, "Poisson electrodes: fixed Dirichlet phi = 0 and Vbi-Vapp", "amber_light", "amber", 560),
]:
    svg.append(svg_rect(x, 995, width, 42, fill, accent, 18, 1.5))
    svg.append(svg_text(label, x + 14, 1021, width - 28, 14.5, accent, True, "middle"))
svg.append("</svg>")

OUT_DIR.mkdir(parents=True, exist_ok=True)
img = img.resize((W, H), Image.Resampling.LANCZOS)
img.save(PNG)
SVG.write_text("\n".join(svg), encoding="utf-8")
print(PNG)
print(SVG)
