from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "docs" / "manual" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
GIF_PATH = OUT_DIR / "initial_condition_dark_to_light.gif"
PNG_PATH = OUT_DIR / "initial_condition_dark_to_light_preview.png"


def _font():
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })


def _profiles(progress: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return schematic n, p, G, R profiles for a teaching animation.

    The profiles are intentionally simplified.  They preserve the qualitative
    SolarLab initialization logic: a dark quasi-neutral state is perturbed by
    optical generation, carriers redistribute under transport/contact extraction,
    and recombination grows until a light-soaked quasi-steady state is reached.
    """
    x = np.linspace(0.0, 1.0, 320)

    # Dark carrier profiles: weak doping/contact asymmetry, no optical source.
    n_dark = 0.28 + 0.07 * x
    p_dark = 0.35 - 0.06 * x

    # Light-soaked carrier profiles: photocarriers build up mostly in the
    # absorber, with electron/hole asymmetry near selective contacts.
    absorber = np.exp(-((x - 0.50) / 0.28) ** 2)
    n_light = n_dark + 0.46 * absorber * (0.70 + 0.65 * x)
    p_light = p_dark + 0.46 * absorber * (1.25 - 0.55 * x)

    # Smooth time approach; fast early carrier relaxation, slow final settling.
    s = 1.0 - np.exp(-3.2 * progress)
    if progress <= 0.08:
        s = 0.0
    s = min(s, 1.0)

    n = (1.0 - s) * n_dark + s * n_light
    p = (1.0 - s) * p_dark + s * p_light

    # Optical generation is turned on after the first frame and then stays on.
    g_amp = 0.0 if progress < 0.10 else min(1.0, (progress - 0.10) / 0.12)
    G = g_amp * (0.18 + 0.82 * np.exp(-2.7 * x))

    # Recombination follows n*p and approaches G/extraction balance.
    R_raw = n * p
    R = 0.18 * s * R_raw / R_raw.max()
    return x, n, p, G, R


def _stage(progress: float) -> tuple[str, str]:
    if progress < 0.10:
        return "1  Dark initial state", "Ydark: quasi-neutral n, p; ions remain at the initial background"
    if progress < 0.25:
        return "2  Turn on illumination", "G(x) > 0 creates electron-hole pairs in the absorber"
    if progress < 0.78:
        return "3  Radau transient integration", "Poisson + SG flux + recombination update dY/dt repeatedly"
    return "4  Illuminated quasi-steady state", "generation, recombination, transport, and contact extraction nearly balance"


def _draw_frame(progress: float) -> Image.Image:
    _font()
    x, n, p, G, R = _profiles(progress)
    title, subtitle = _stage(progress)

    fig = plt.figure(figsize=(12.8, 7.2), dpi=120, facecolor="#F7F9FC")
    gs = fig.add_gridspec(
        3, 3,
        height_ratios=[0.42, 1.62, 0.82],
        width_ratios=[1.18, 1.0, 1.0],
        left=0.055,
        right=0.965,
        top=0.94,
        bottom=0.08,
        hspace=0.38,
        wspace=0.30,
    )

    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.text(
        0.0, 0.70,
        "Initial Condition Pre-Settle: Dark State → Illuminated Quasi-Steady State",
        fontsize=22,
        weight="bold",
        color="#162033",
        ha="left",
        va="center",
    )
    ax_title.text(0.0, 0.18, subtitle, fontsize=12, color="#334155", ha="left", va="center")

    ax_device = fig.add_subplot(gs[1, 0])
    ax_carrier = fig.add_subplot(gs[1, 1:])
    ax_balance = fig.add_subplot(gs[2, :])

    # Device strip.
    ax_device.set_xlim(0, 1)
    ax_device.set_ylim(0, 1)
    ax_device.axis("off")
    ax_device.text(0.02, 1.02, title, fontsize=13, weight="bold", color="#334155", va="bottom")
    layers = [
        (0.76, 0.94, "#FEE4E2", "#B42318", "HTL / contact"),
        (0.30, 0.76, "#EDE9FE", "#6D28D9", "absorber"),
        (0.12, 0.30, "#E0F2FE", "#2563EB", "ETL / contact"),
    ]
    for y0, y1, fill, edge, label in layers:
        ax_device.add_patch(plt.Rectangle((0.17, y0), 0.66, y1 - y0, fc=fill, ec=edge, lw=1.6))
        ax_device.text(0.50, (y0 + y1) / 2, label, ha="center", va="center", color=edge, weight="bold")
    ax_device.add_patch(plt.Rectangle((0.17, 0.94), 0.66, 0.055, fc="#E2E8F0", ec="#334155", lw=1.2))
    ax_device.add_patch(plt.Rectangle((0.17, 0.065), 0.66, 0.055, fc="#E2E8F0", ec="#334155", lw=1.2))
    ax_device.text(0.50, 0.97, "top electrode", ha="center", va="center", fontsize=8.5, color="#334155", weight="bold")
    ax_device.text(0.50, 0.09, "bottom electrode", ha="center", va="center", fontsize=8.5, color="#334155", weight="bold")

    if progress >= 0.10:
        for xx in np.linspace(0.25, 0.75, 5):
            ax_device.annotate(
                "",
                xy=(xx, 0.62),
                xytext=(xx, 0.91),
                arrowprops=dict(arrowstyle="-|>", color="#E11D48", lw=1.2),
            )
        ax_device.text(0.91, 0.86, "G(y)", ha="center", va="center", color="#E11D48", weight="bold")
    if progress >= 0.28:
        ax_device.annotate("", xy=(0.30, 0.91), xytext=(0.30, 0.75), arrowprops=dict(arrowstyle="<->", color="#B42318", lw=1.4))
        ax_device.annotate("", xy=(0.70, 0.13), xytext=(0.70, 0.30), arrowprops=dict(arrowstyle="<->", color="#2563EB", lw=1.4))
        ax_device.text(0.91, 0.52, "contact\nexchange", ha="center", va="center", fontsize=8.2, color="#334155")

    # Carrier profiles.
    ax_carrier.set_title("Carrier redistribution during short transient settle", loc="left", color="#334155", weight="bold")
    ax_carrier.plot(x, n, color="#2563EB", lw=3.0, label="electrons n(y)")
    ax_carrier.plot(x, p, color="#B42318", lw=3.0, label="holes p(y)")
    ax_carrier.plot(x, G, color="#A16207", lw=2.2, ls="--", label="generation G(y)")
    ax_carrier.plot(x, R, color="#16803C", lw=2.2, ls="--", label="recombination R(y)")
    ax_carrier.set_xlim(0, 1)
    ax_carrier.set_ylim(0, 1.05)
    ax_carrier.set_xlabel("vertical coordinate y / device thickness")
    ax_carrier.set_ylabel("normalized magnitude")
    ax_carrier.grid(True, color="#E2E8F0", lw=0.8)
    ax_carrier.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=9)
    ax_carrier.text(0.02, 0.93, f"t = {progress * 1e-3:.1e} s", transform=ax_carrier.transAxes, color="#64748B", fontsize=10)

    # Balance strip.
    ax_balance.set_xlim(0, 1)
    ax_balance.set_ylim(0, 1)
    ax_balance.axis("off")
    ax_balance.add_patch(plt.Rectangle((0.00, 0.10), 1.00, 0.72, fc="#FFFFFF", ec="#CBD5E1", lw=1.2))

    labels = [
        ("Ydark", "#DBEAFE", "#2563EB"),
        ("G turns on", "#FEF3C7", "#A16207"),
        ("solve Poisson\npotential", "#F8FAFC", "#334155"),
        ("SG carrier\nflux", "#EDE9FE", "#6D28D9"),
        ("RHS update\nfor n,p", "#DCFCE7", "#16803C"),
        ("Ylight", "#DCFCE7", "#16803C"),
    ]
    xs = np.linspace(0.05, 0.85, len(labels))
    active_idx = min(len(labels) - 1, int(progress * len(labels)))
    for i, ((label, fill, edge), x0) in enumerate(zip(labels, xs)):
        alpha = 1.0 if i <= active_idx or progress > 0.82 else 0.45
        ax_balance.add_patch(plt.Rectangle((x0, 0.31), 0.11, 0.30, fc=fill, ec=edge, lw=1.5, alpha=alpha))
        ax_balance.text(x0 + 0.055, 0.46, label, ha="center", va="center", fontsize=9.4, weight="bold", color="#162033", alpha=alpha)
        if i < len(labels) - 1:
            ax_balance.annotate("", xy=(x0 + 0.15, 0.46), xytext=(x0 + 0.115, 0.46), arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#64748B", alpha=alpha))
    ax_balance.text(
        0.02, 0.88,
        "Radau integrates dY/dt = F(Y) until the fast carrier transient settles",
        fontsize=11,
        weight="bold",
        color="#162033",
        ha="left",
    )

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    image = Image.fromarray(rgba).convert("RGB")
    plt.close(fig)
    return image


def main() -> None:
    # Hold the first and final states a little longer by repeating progress values.
    progress_values = (
        [0.0, 0.0]
        + list(np.linspace(0.10, 0.90, 9))
        + [1.0, 1.0]
    )
    frames = [_draw_frame(float(p)) for p in progress_values]
    frames[0].save(
        GIF_PATH,
        save_all=True,
        append_images=frames[1:],
        duration=1800,
        loop=0,
        optimize=True,
    )
    frames[-1].save(PNG_PATH)
    print(GIF_PATH)
    print(PNG_PATH)


if __name__ == "__main__":
    main()
