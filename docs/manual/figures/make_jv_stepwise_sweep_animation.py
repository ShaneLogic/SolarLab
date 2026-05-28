from __future__ import annotations

from math import atan2, cos, sin
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "docs" / "manual" / "figures"
GIF_PATH = OUT_DIR / "jv_stepwise_transient_sweep.gif"
PNG_PATH = OUT_DIR / "jv_stepwise_transient_sweep_preview.png"

BASE_W, BASE_H = 1280, 720
RENDER_SCALE = 1.5
W, H = int(BASE_W * RENDER_SCALE), int(BASE_H * RENDER_SCALE)
V_MAX = 1.4
FORWARD_V = [0.00, 0.35, 0.70, 1.05, 1.40]
REVERSE_V = [1.40, 1.05, 0.70, 0.35, 0.00]
SWEEP = [("forward", v) for v in FORWARD_V] + [("reverse", v) for v in REVERSE_V]

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
    "amber": "#A16207",
    "amber_light": "#FEF3C7",
    "green": "#16803C",
    "green_light": "#DCFCE7",
    "purple": "#6D28D9",
    "purple_light": "#EDE9FE",
    "red": "#B42318",
    "red_light": "#FEE4E2",
}


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    size = max(1, int(round(size * RENDER_SCALE)))
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
    "h1": _font(32, bold=True),
    "h2": _font(26, bold=True),
    "label": _font(22, bold=True),
    "body": _font(20),
    "small": _font(18),
    "tiny": _font(15),
    "sub": _font(11),
}


class ScaledDraw:
    """Draw in logical 1280x720 coordinates onto a native high-res canvas."""

    def __init__(self, draw: ImageDraw.ImageDraw, scale: float) -> None:
        self.draw = draw
        self.scale = scale

    def _p(self, p: tuple[float, float]) -> tuple[float, float]:
        return (p[0] * self.scale, p[1] * self.scale)

    def _xy(self, xy):
        if isinstance(xy, tuple) and len(xy) == 4:
            return tuple(v * self.scale for v in xy)
        if isinstance(xy, tuple) and len(xy) == 2:
            return self._p(xy)
        return [self._p(p) for p in xy]

    def rounded_rectangle(self, xy, radius=0, fill=None, outline=None, width=1):
        return self.draw.rounded_rectangle(
            self._xy(xy),
            radius=int(round(radius * self.scale)),
            fill=fill,
            outline=outline,
            width=max(1, int(round(width * self.scale))),
        )

    def rectangle(self, xy, fill=None, outline=None, width=1):
        return self.draw.rectangle(
            self._xy(xy),
            fill=fill,
            outline=outline,
            width=max(1, int(round(width * self.scale))),
        )

    def line(self, xy, fill=None, width=1):
        return self.draw.line(
            self._xy(xy),
            fill=fill,
            width=max(1, int(round(width * self.scale))),
        )

    def ellipse(self, xy, fill=None, outline=None, width=1):
        return self.draw.ellipse(
            self._xy(xy),
            fill=fill,
            outline=outline,
            width=max(1, int(round(width * self.scale))),
        )

    def polygon(self, xy, fill=None, outline=None):
        return self.draw.polygon(self._xy(xy), fill=fill, outline=outline)

    def text(self, xy, text, font=None, fill=None, **kwargs):
        return self.draw.text(self._p(xy), text, font=font, fill=fill, **kwargs)

    def multiline_text(self, xy, text, font=None, fill=None, **kwargs):
        return self.draw.multiline_text(self._p(xy), text, font=font, fill=fill, **kwargs)

    def textbbox(self, xy, text, font=None, **kwargs):
        box = self.draw.textbbox(self._p(xy), text, font=font, **kwargs)
        return tuple(v / self.scale for v in box)

    def multiline_textbbox(self, xy, text, font=None, **kwargs):
        box = self.draw.multiline_textbbox(self._p(xy), text, font=font, **kwargs)
        return tuple(v / self.scale for v in box)


def rounded(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], fill: str, outline: str, width: int = 2, radius: int = 18) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def text_center(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], text: str, font: ImageFont.FreeTypeFont, fill: str) -> None:
    box = draw.multiline_textbbox((0, 0), text, font=font, spacing=4, align="center")
    tw = box[2] - box[0]
    th = box[3] - box[1]
    x = xy[0] + (xy[2] - xy[0] - tw) / 2
    y = xy[1] + (xy[3] - xy[1] - th) / 2
    draw.multiline_text((x, y), text, font=font, fill=fill, spacing=4, align="center")


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
    """Draw simple presentation math like V with subscript max."""
    font = font or F["tiny"]
    sub_font = sub_font or F["sub"]
    x, y = xy
    draw.text((x, y), base, font=font, fill=fill)
    bbox = draw.textbbox((x, y), base, font=font)
    bx = bbox[2] - bbox[0]
    scale = getattr(draw, "scale", 1.0)
    sub_y = y + int((font.size / scale) * 0.52)
    draw.text((x + bx + 1, sub_y), sub, font=sub_font, fill=fill)
    sb = draw.textbbox((x + bx + 1, sub_y), sub, font=sub_font)
    return sb[2] - x


def arrow(draw: ImageDraw.ImageDraw, start: tuple[float, float], end: tuple[float, float], color: str, width: int = 3) -> None:
    draw.line([start, end], fill=color, width=width)
    angle = atan2(end[1] - start[1], end[0] - start[0])
    length = 14
    spread = 0.42
    p1 = (end[0] - length * cos(angle - spread), end[1] - length * sin(angle - spread))
    p2 = (end[0] - length * cos(angle + spread), end[1] - length * sin(angle + spread))
    draw.polygon([end, p1, p2], fill=color)


def dashed_line(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], fill: str, width: int = 2, dash: int = 13) -> None:
    for a, b in zip(points[:-1], points[1:]):
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        length = (dx * dx + dy * dy) ** 0.5
        if length == 0:
            continue
        n = max(1, int(length // dash))
        for i in range(0, n, 2):
            t0 = i / n
            t1 = min(1.0, (i + 1) / n)
            draw.line(
                [(a[0] + dx * t0, a[1] + dy * t0), (a[0] + dx * t1, a[1] + dy * t1)],
                fill=fill,
                width=width,
            )


def current_density(v: float, direction: str) -> float:
    base = 22.0 * (1.0 - (v / 1.12) ** 5)
    hysteresis = 2.0 * (1.0 - v / V_MAX) if direction == "reverse" else 0.0
    return max(-22.0, min(24.0, base + hysteresis))


def map_voltage(i: float, v: float, plot: tuple[int, int, int, int]) -> tuple[float, float]:
    x0, y0, x1, y1 = plot
    return x0 + i / len(SWEEP) * (x1 - x0), y1 - v / V_MAX * (y1 - y0)


def draw_voltage_panel(draw: ImageDraw.ImageDraw, step: int, phase: int, phase_frac: float) -> None:
    panel = (50, 38, 1230, 295)
    rounded(draw, panel, C["white"], C["line"], width=2, radius=18)
    draw.text((78, 62), "Voltage schedule: fixed plateaus", font=F["h1"], fill=C["slate"])
    draw.text((78, 99), "Each highlighted segment is one Radau transient solve at constant applied voltage.", font=F["body"], fill=C["muted"])

    plot = (128, 143, 1165, 252)
    x0, y0, x1, y1 = plot
    draw.line([(x0, y1), (x1, y1)], fill=C["line"], width=2)
    draw.line([(x0, y0), (x0, y1)], fill=C["line"], width=2)
    sub_label(draw, (70, y0 - 11), "V", "max", C["muted"], font=F["tiny"])
    draw.text((84, y1 - 8), "0 V", font=F["tiny"], fill=C["muted"])
    draw.text((610, y1 + 20), "time", font=F["tiny"], fill=C["muted"])

    ramp = [
        map_voltage(0, 0.0, plot),
        map_voltage(len(FORWARD_V), V_MAX, plot),
        map_voltage(len(SWEEP), 0.0, plot),
    ]
    dashed_line(draw, ramp, "#94A3B8", width=2)
    draw.text((78, 123), "dashed line: ideal continuous ramp", font=F["tiny"], fill=C["muted"])

    last_y = None
    for i, (direction, v) in enumerate(SWEEP):
        color = C["blue"] if direction == "forward" else C["amber"]
        x_start, y = map_voltage(i, v, plot)
        x_end, _ = map_voltage(i + 1, v, plot)
        if last_y is not None:
            draw.line([(x_start, last_y), (x_start, y)], fill=C["line"], width=2)
        segment_color = color if i <= step else C["line"]
        segment_width = 8 if i == step else 4 if i < step else 2
        draw.line([(x_start, y), (x_end, y)], fill=segment_color, width=segment_width)
        if i == step:
            dot_x = x_start + phase_frac * (x_end - x_start)
            draw.ellipse((dot_x - 11, y - 11, dot_x + 11, y + 11), fill=C["amber_light"], outline=C["amber"], width=3)
            label_w = 232
            label_h = 34
            badge = (962, 68, 1204, 132)
            label_x = min(max(x_start + 18, 165), x1 - label_w - 16)
            label_y = y - 44 if y > y0 + 50 else y + 20
            candidate = (label_x - 8, label_y - 6, label_x + label_w, label_y + label_h)
            overlaps_badge = not (
                candidate[2] <= badge[0]
                or candidate[0] >= badge[2]
                or candidate[3] <= badge[1]
                or candidate[1] >= badge[3]
            )
            if overlaps_badge:
                label_x = max(165, min(x_start - label_w - 18, badge[0] - label_w - 18))
                candidate = (label_x - 8, label_y - 6, label_x + label_w, label_y + label_h)
                if candidate[1] < y0 - 8:
                    label_y = y + 22
            rounded(draw, (int(label_x) - 8, int(label_y) - 6, int(label_x) + label_w, int(label_y) + label_h), C["white"], C["line"], width=1, radius=8)
            draw.text((label_x, label_y), f"hold V = {v:.2f} V", font=F["small"], fill=C["navy"])
        last_y = y

    direction, _ = SWEEP[step]
    fill = C["blue_light"] if direction == "forward" else C["amber_light"]
    edge = C["blue"] if direction == "forward" else C["amber"]
    rounded(draw, (962, 68, 1204, 132), fill, edge, width=3, radius=16)
    text_center(draw, (962, 68, 1204, 132), f"{direction.title()} scan\nstep {step + 1} / {len(SWEEP)}", F["label"], edge)


def draw_jv_panel(draw: ImageDraw.ImageDraw, solved: int, current_step: int) -> None:
    panel = (50, 323, 615, 660)
    rounded(draw, panel, C["white"], C["line"], width=2, radius=18)
    draw.text((78, 349), "J-V curve builds point by point", font=F["h2"], fill=C["slate"])
    draw.text((78, 381), "A point appears after each voltage plateau settles.", font=F["body"], fill=C["muted"])

    plot = (128, 452, 560, 600)
    x0, y0, x1, y1 = plot
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x = x0 + frac * (x1 - x0)
        draw.line([(x, y0), (x, y1)], fill=C["grid"], width=1)
    for frac in [0.0, 0.5, 1.0]:
        y = y0 + frac * (y1 - y0)
        draw.line([(x0, y), (x1, y)], fill=C["grid"], width=1)
    draw.line([(x0, y1), (x1, y1)], fill=C["slate"], width=2)
    draw.line([(x0, y0), (x0, y1)], fill=C["slate"], width=2)
    draw.text((112, y0 - 18), "J", font=F["tiny"], fill=C["muted"])
    draw.text((292, y1 + 21), "applied bias V", font=F["tiny"], fill=C["muted"])
    draw.text((x0 - 34, y0 + 7), "+Jsc", font=F["tiny"], fill=C["muted"])
    draw.text((x0 - 20, y1 + 6), "0", font=F["tiny"], fill=C["muted"])
    sub_label(draw, (x1 - 13, y1 + 6), "V", "max", C["muted"], font=F["tiny"])

    def map_j(v: float, j: float) -> tuple[float, float]:
        return x0 + v / V_MAX * (x1 - x0), y1 - (j + 22.0) / 46.0 * (y1 - y0)

    fwd: list[tuple[float, float]] = []
    rev: list[tuple[float, float]] = []
    for i in range(min(solved, len(SWEEP))):
        direction, v = SWEEP[i]
        xy = map_j(v, current_density(v, direction))
        if direction == "forward":
            fwd.append(xy)
        else:
            rev.append(xy)
    if len(fwd) > 1:
        draw.line(fwd, fill=C["blue"], width=4)
    if len(rev) > 1:
        draw.line(rev, fill=C["amber"], width=4)

    for i in range(min(solved, len(SWEEP))):
        direction, v = SWEEP[i]
        xy = map_j(v, current_density(v, direction))
        color = C["blue"] if direction == "forward" else C["amber"]
        r = 10 if i == current_step else 6
        draw.ellipse((xy[0] - r, xy[1] - r, xy[0] + r, xy[1] + r), fill=color, outline=C["white"], width=2)

    draw.rectangle((404, 405, 584, 442), fill=C["white"], outline=C["line"], width=1)
    draw.line([(421, 418), (453, 418)], fill=C["blue"], width=4)
    draw.text((462, 409), "forward", font=F["tiny"], fill=C["slate"])
    draw.line([(421, 434), (453, 434)], fill=C["amber"], width=4)
    draw.text((462, 425), "reverse", font=F["tiny"], fill=C["slate"])


def draw_process_panel(draw: ImageDraw.ImageDraw, step: int, phase: int) -> None:
    panel = (665, 323, 1230, 660)
    rounded(draw, panel, C["white"], C["line"], width=2, radius=18)
    draw.text((693, 349), "One sample: hold, integrate, record", font=F["h2"], fill=C["slate"])
    draw.text((693, 381), "The final state warm-starts the next voltage plateau.", font=F["body"], fill=C["muted"])

    cards = [
        ("HOLD", "V fixed", C["blue_light"], C["blue"]),
        ("RADAU", "state\nadvances", C["purple_light"], C["purple"]),
        ("RECORD", "J(V)", C["green_light"], C["green"]),
    ]
    x0 = 698
    card_w = 148
    card_gap = 186
    for i, (label, body, fill, edge) in enumerate(cards):
        x = x0 + card_gap * i
        y = 438
        active = i == phase
        rounded(draw, (x, y, x + card_w, y + 94), fill, edge if active else C["line"], width=4 if active else 2, radius=16)
        text_center(draw, (x + 6, y + 12, x + card_w - 6, y + 44), label, F["label"], edge)
        text_center(draw, (x + 10, y + 47, x + card_w - 10, y + 86), body, F["body"], C["navy"])
        if i < len(cards) - 1:
            arrow(draw, (x + card_w + 12, y + 47), (x + card_gap - 14, y + 47), C["muted"], width=2)

    if phase == 1:
        stage_y = 558
        stage_x = [914, 962, 1010]
        draw.line([(stage_x[0], stage_y), (stage_x[-1], stage_y)], fill=C["purple"], width=2)
        for idx, cx in enumerate(stage_x, start=1):
            r = 17
            draw.ellipse((cx - r, stage_y - r, cx + r, stage_y + r), fill=C["purple_light"], outline=C["purple"], width=2)
            text_center(draw, (cx - r, stage_y - r, cx + r, stage_y + r), f"K{idx}", F["tiny"], C["purple"])

    direction, v = SWEEP[step]
    fill = C["blue_light"] if direction == "forward" else C["amber_light"]
    edge = C["blue"] if direction == "forward" else C["amber"]
    rounded(draw, (705, 592, 1188, 640), fill, edge, width=2, radius=13)
    draw.text((730, 607), f"V = {v:.2f} V", font=F["small"], fill=C["navy"])
    draw.text((875, 607), f"scan: {direction}", font=F["small"], fill=C["navy"])
    draw.text((1034, 607), "dt set by scan rate", font=F["small"], fill=C["navy"])


def draw_frame(step: int, phase: int) -> Image.Image:
    phase_frac = [0.12, 0.55, 0.95][phase]
    solved = step + (1 if phase == 2 else 0)
    image = Image.new("RGB", (W, H), C["bg"])
    draw = ScaledDraw(ImageDraw.Draw(image), RENDER_SCALE)
    draw_voltage_panel(draw, step, phase, phase_frac)
    draw_jv_panel(draw, solved, step)
    draw_process_panel(draw, step, phase)
    draw.text((52, 681), "Scan rate controls dwell time per voltage sample; state carry-over produces history dependence.", font=F["label"], fill=C["navy"])
    return image


def quantize_for_clean_gif(frames: list[Image.Image]) -> list[Image.Image]:
    """Use one no-dither adaptive palette to avoid noisy borders in GIF output."""
    cols, rows = 5, 6
    contact = Image.new("RGB", (W * cols, H * rows), C["bg"])
    for idx, frame in enumerate(frames):
        x = (idx % cols) * W
        y = (idx // cols) * H
        contact.paste(frame, (x, y))
    palette = contact.quantize(colors=192, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    return [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames: list[Image.Image] = []
    durations: list[int] = []
    for step in range(len(SWEEP)):
        for phase in range(3):
            frames.append(draw_frame(step, phase))
            durations.append(800 if phase != 2 else 1200)
    frames[25].save(PNG_PATH)
    qframes = quantize_for_clean_gif(frames)
    qframes[0].save(
        GIF_PATH,
        save_all=True,
        append_images=qframes[1:],
        duration=durations,
        loop=0,
        optimize=False,
        disposal=2,
    )
    print(GIF_PATH)
    print(PNG_PATH)


if __name__ == "__main__":
    main()
