from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "docs" / "manual" / "figures"
PNG_PATH = OUT_DIR / "scaps_1d_input_parameters_16x9.png"
SVG_PATH = OUT_DIR / "scaps_1d_input_parameters_16x9.svg"


ROWS = [
    (
        "Cell / stack",
        r"layer order, thickness $d$, area $A$, temperature $T$",
        "1D geometry and operating temperature",
        "#2563EB",
    ),
    (
        "Layer material",
        r"$E_g$, $\chi$, $\varepsilon_r$, $N_C$, $N_V$, $v_{\mathrm{th},n}$, $v_{\mathrm{th},p}$",
        "band alignment, electrostatics, carrier statistics",
        "#7C3AED",
    ),
    (
        "Transport",
        r"$\mu_n$, $\mu_p$, $N_A$, $N_D$; grading $P(x)$ or $P(y)$",
        "drift-diffusion mobility and fixed doping",
        "#0F766E",
    ),
    (
        "Recombination",
        r"$B_{\mathrm{rad}}$, $C_n/C_p$, bulk defects $N_t$, $E_t$, $\sigma_n/\sigma_p$",
        "SRH / radiative / Auger loss channels",
        "#15803D",
    ),
    (
        "Interface",
        r"interface defects, tunneling, $m_e$, $m_h$",
        "heterojunction recombination and tunneling",
        "#A16207",
    ),
    (
        "Contacts",
        r"work function $\Phi_m$ or flat-band; $S_n$, $S_p$; optical filter",
        "boundary carrier extraction and optical loss",
        "#B45309",
    ),
    (
        "Optical input",
        r"spectrum, illumination side, $R/T$, absorption $\alpha(\lambda)$",
        "generation profile G(x)",
        "#DC2626",
    ),
    (
        "Simulation setup",
        "working point V, f, T; JV / CV / Cf / QE ranges; mesh settings",
        "defines which measurement is solved",
        "#334155",
    ),
]


def _font() -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 13,
        "svg.fonttype": "none",
    })


def _draw() -> None:
    _font()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(16, 9), dpi=180, facecolor="#F7F9FC")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(
        0.78,
        8.38,
        "SCAPS-1D Input Parameters",
        fontsize=28,
        weight="bold",
        color="#162033",
        ha="left",
        va="center",
    )
    ax.text(
        0.80,
        7.92,
        "Slide-ready overview of the main inputs used to define a 1D thin-film solar-cell simulation",
        fontsize=14,
        color="#475569",
        ha="left",
        va="center",
    )

    x0, y0 = 0.72, 0.92
    width, height = 14.55, 6.62
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            fc="#FFFFFF",
            ec="#CBD5E1",
            lw=1.4,
        )
    )

    col = [x0, x0 + 2.28, x0 + 8.62, x0 + width]
    headers = ["Input block", "Typical parameters", "Physical role"]
    header_y = y0 + height - 0.62
    ax.add_patch(Rectangle((x0, header_y), width, 0.62, fc="#E2E8F0", ec="none"))
    for i, text in enumerate(headers):
        ax.text(col[i] + 0.18, header_y + 0.31, text, ha="left", va="center", fontsize=13.5, weight="bold", color="#162033")

    n = len(ROWS)
    row_h = (height - 0.62) / n
    body_top = header_y
    for r, (block, params, role, color) in enumerate(ROWS):
        y_top = body_top - r * row_h
        y_bot = y_top - row_h
        if r % 2 == 0:
            ax.add_patch(Rectangle((x0, y_bot), width, row_h, fc="#F8FAFC", ec="none"))
        ax.plot([x0, x0 + width], [y_bot, y_bot], color="#E2E8F0", lw=0.9)
        ax.add_patch(Rectangle((x0, y_bot), 0.08, row_h, fc=color, ec="none"))

        ax.text(col[0] + 0.18, y_bot + row_h / 2, block, ha="left", va="center", fontsize=13.2, weight="bold", color=color)
        ax.text(col[1] + 0.18, y_bot + row_h / 2, params, ha="left", va="center", fontsize=12.0, color="#162033")
        ax.text(col[2] + 0.18, y_bot + row_h / 2, role, ha="left", va="center", fontsize=11.7, color="#334155")

    for x in col[1:-1]:
        ax.plot([x, x], [y0, y0 + height], color="#E2E8F0", lw=0.9)

    ax.text(
        0.80,
        0.45,
        "Compact table for 16:9 slides; parameter symbols follow SCAPS script/manual naming where possible.",
        fontsize=10.5,
        color="#64748B",
        ha="left",
        va="center",
    )

    fig.savefig(PNG_PATH, dpi=180, bbox_inches=None)
    fig.savefig(SVG_PATH, bbox_inches=None)
    plt.close(fig)


if __name__ == "__main__":
    _draw()
    print(PNG_PATH)
    print(SVG_PATH)
