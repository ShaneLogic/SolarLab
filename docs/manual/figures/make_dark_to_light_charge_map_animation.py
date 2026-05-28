from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "docs" / "manual" / "figures"
GIF_PATH = OUT_DIR / "initial_condition_dark_to_light_charge_map.gif"
PNG_PATH = OUT_DIR / "initial_condition_dark_to_light_charge_map_preview.png"


def _setup_font() -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })


def _smoothstep(value: float, edge0: float, edge1: float) -> float:
    if edge0 == edge1:
        return float(value >= edge1)
    t = np.clip((value - edge0) / (edge1 - edge0), 0.0, 1.0)
    return float(t * t * (3.0 - 2.0 * t))


def _stage(progress: float) -> tuple[str, str]:
    if progress < 0.10:
        return "dark state", "near-neutral carrier distribution before illumination"
    if progress < 0.28:
        return "light turns on", "optical generation creates electron-hole pairs"
    if progress < 0.82:
        return "Radau settling", "time integration repeatedly updates charge and flux"
    return "light quasi-steady state", "generation, recombination, transport, and extraction nearly balance"


def _charge_map(progress: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, 180)
    y = np.linspace(0.0, 1.0, 280)
    xx, yy = np.meshgrid(x, y)

    dark = (
        0.035 * np.exp(-((yy - 0.83) / 0.06) ** 2)
        - 0.035 * np.exp(-((yy - 0.17) / 0.06) ** 2)
    )

    settle = _smoothstep(progress, 0.12, 0.85)
    top_holes = np.exp(-((yy - 0.76) / 0.13) ** 2)
    bottom_electrons = np.exp(-((yy - 0.24) / 0.13) ** 2)
    lateral_shape = 1.0 + 0.07 * np.cos(2.0 * np.pi * (xx - 0.5))
    light = 0.46 * lateral_shape * (top_holes - bottom_electrons)

    transient_front = np.exp(-((yy - (0.82 - 0.44 * settle)) / 0.10) ** 2)
    transient = 0.11 * np.sin(2.0 * np.pi * xx) * transient_front * (1.0 - settle)
    rho = (1.0 - settle) * dark + settle * light + transient

    generation = _smoothstep(progress, 0.08, 0.18) * np.exp(-3.0 * (1.0 - yy))
    return xx, yy, rho, generation


def _pair_positions(progress: float) -> list[tuple[float, float, float, float, float]]:
    rng = np.random.default_rng(7)
    pairs: list[tuple[float, float, float, float, float]] = []
    for birth in np.linspace(0.12, 0.72, 10):
        local = np.clip((progress - birth) / 0.34, 0.0, 1.0)
        if local <= 0.0:
            continue
        x0 = 0.18 + 0.64 * rng.random()
        y0 = 0.34 + 0.34 * rng.random()
        alpha = 1.0 - 0.45 * local
        hole_y = y0 + 0.23 * local
        electron_y = y0 - 0.23 * local
        pairs.append((x0, hole_y, electron_y, local, alpha))
    return pairs


def _draw_layers(ax: plt.Axes) -> None:
    bands = [
        (0.84, 1.00, "HTL / top contact", "#B42318"),
        (0.20, 0.84, "absorber", "#6D28D9"),
        (0.00, 0.20, "ETL / bottom contact", "#2563EB"),
    ]
    for y0, y1, label, color in bands:
        ax.hlines([y0, y1], 0.0, 1.0, colors=color, lw=1.8)
        ax.text(0.50, (y0 + y1) / 2, label, ha="center", va="center", color=color, weight="bold")


def _draw_frame(progress: float) -> Image.Image:
    _setup_font()
    _, _, rho, generation = _charge_map(progress)
    stage, caption = _stage(progress)

    cmap = LinearSegmentedColormap.from_list(
        "charge_balance",
        ["#1E63B6", "#F8FAFC", "#C21F1A"],
        N=256,
    )
    norm = TwoSlopeNorm(vmin=-0.55, vcenter=0.0, vmax=0.55)

    fig = plt.figure(figsize=(7.6, 7.8), dpi=130, facecolor="#F7F9FC")
    ax = fig.add_axes([0.12, 0.18, 0.60, 0.70])
    cax = fig.add_axes([0.77, 0.29, 0.045, 0.48])
    time_ax = fig.add_axes([0.12, 0.08, 0.70, 0.05])

    image = ax.imshow(rho, origin="lower", extent=(0, 1, 0, 1), cmap=cmap, norm=norm, interpolation="bilinear")
    ax.contour(
        generation,
        levels=[0.20, 0.45, 0.70],
        origin="lower",
        extent=(0, 1, 0, 1),
        colors=["#A16207"],
        linewidths=[0.7, 0.9, 1.1],
        alpha=0.35,
    )
    _draw_layers(ax)

    if progress >= 0.08:
        for x0 in np.linspace(0.18, 0.82, 5):
            ax.annotate(
                "",
                xy=(x0, 0.86),
                xytext=(x0, 1.10),
                arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#D97706"),
                annotation_clip=False,
            )
        ax.text(0.86, 1.08, "light", ha="left", va="center", color="#D97706", weight="bold", clip_on=False)

    for x0, hole_y, electron_y, local, alpha in _pair_positions(progress):
        ax.plot(x0 - 0.018 * local, hole_y, "o", ms=5.4, color="#C21F1A", alpha=alpha)
        ax.plot(x0 + 0.018 * local, electron_y, "o", ms=5.4, color="#1E63B6", alpha=alpha)
        ax.plot([x0 - 0.018 * local, x0 + 0.018 * local], [hole_y, electron_y], color="#64748B", lw=0.5, alpha=0.25)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.8)
        spine.set_color("#162033")

    cb = fig.colorbar(image, cax=cax)
    cb.set_ticks([-0.5, 0.0, 0.5])
    cb.set_ticklabels(["electron-rich", "neutral", "hole-rich"])
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_edgecolor("#CBD5E1")

    fig.text(0.12, 0.94, "Dark-to-light charge redistribution", fontsize=20, weight="bold", color="#162033")
    fig.text(0.12, 0.905, f"{stage}: {caption}", fontsize=11.5, color="#334155")
    fig.text(0.77, 0.81, "net charge\nmap", fontsize=11, weight="bold", color="#162033", ha="left")
    fig.text(0.77, 0.20, "schematic\nnot solver data", fontsize=8.5, color="#64748B", ha="left")

    time_ax.set_xlim(0, 1)
    time_ax.set_ylim(0, 1)
    time_ax.axis("off")
    time_ax.plot([0.0, 1.0], [0.48, 0.48], color="#CBD5E1", lw=7, solid_capstyle="round")
    time_ax.plot([0.0, progress], [0.48, 0.48], color="#16803C", lw=7, solid_capstyle="round")
    time_ax.plot(progress, 0.48, "o", ms=8, color="#16803C")
    time_ax.text(0.00, 0.02, "0 s", ha="left", va="bottom", fontsize=9, color="#64748B")
    time_ax.text(1.00, 0.02, "1e-3 s", ha="right", va="bottom", fontsize=9, color="#64748B")
    time_ax.text(0.50, 0.88, "short Radau pre-settle", ha="center", va="top", fontsize=10, color="#334155", weight="bold")

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    frame = Image.fromarray(rgba).convert("RGB")
    plt.close(fig)
    return frame


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    progress_values = [0.0, 0.0] + list(np.linspace(0.10, 0.90, 9)) + [1.0, 1.0]
    frames = [_draw_frame(float(progress)) for progress in progress_values]
    frames[0].save(
        GIF_PATH,
        save_all=True,
        append_images=frames[1:],
        duration=1900,
        loop=0,
        optimize=True,
    )
    frames[-1].save(PNG_PATH)
    print(GIF_PATH)
    print(PNG_PATH)


if __name__ == "__main__":
    main()
