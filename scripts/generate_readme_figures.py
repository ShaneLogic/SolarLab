"""Generate the SolarLab README figures in strict native Arial typography.

Every figure renders at 300 DPI using genuine Arial (Regular, Bold, Italic, and
custom Arial mathtext) with zero DejaVu/Computer-Modern fallback. Validation
series come from the verified benchmark baselines:
  - scaps_pvk_xy_spe_full_262.json (sha256: a77bec53bbfc51d3..., 262/262 converged)
  - hysteresis_full_45.json (sha256: 9cbc22b7f955a2bd..., 45/45 completed)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from matplotlib.ticker import NullFormatter, ScalarFormatter  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "figures"
DPI = 300

# Unified publication color palette (colorblind-safe, high contrast on white)
C_NAVY = "#1E3A5F"
C_BLUE = "#2563EB"
C_TEAL = "#0D9488"
C_AMBER = "#D97706"
C_CORAL = "#DC2626"
C_PURPLE = "#7C3AED"
C_INK = "#0F172A"
C_TEXT = "#1E293B"
C_MUTED = "#475569"
C_SUBTLE = "#64748B"
C_LINE = "#CBD5E1"
C_GRID = "#E2E8F0"
C_BG_BLUE = "#EFF6FF"
C_BG_TEAL = "#F0FDFA"
C_BG_AMBER = "#FFFBEB"
C_BG_SLATE = "#F8FAFC"
C_BG_PURPLE = "#F5F3FF"
C_BG_ROSE = "#FEF2F2"


def configure_strict_arial() -> None:
    """Bind all Matplotlib text and mathtext families strictly to native Arial."""
    resolved = fm.findfont("Arial", fallback_to_default=False)
    if "arial" not in Path(resolved).name.lower():
        raise RuntimeError(f"Expected native Arial font, got {resolved}")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "mathtext.sf": "Arial",
            "mathtext.default": "regular",
            "axes.unicode_minus": True,
            "axes.edgecolor": C_MUTED,
            "axes.linewidth": 0.9,
            "xtick.color": C_TEXT,
            "ytick.color": C_TEXT,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


# Verified benchmark arrays (scaps_pvk_xy_spe_full_262: a77bec53..., hysteresis_full_45: 9cbc22b7...)
BENCHMARK_DATA: dict[str, Any] = {
    "scaps_cbo": {
        "x": [-1.0, -0.75, -0.5, -0.25, -0.2, -0.16, -0.1, -0.05, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "sl_voc": [
            0.3327, 0.58123, 0.83122, 1.08077, 1.12974, 1.16805, 1.21776,
            1.24049, 1.2487, 1.25138, 1.25112, 1.25068, 1.24956, 1.23269,
        ],
        "ref_voc": [
            0.33235, 0.58095, 0.83085, 1.08049, 1.12931, 1.16755, 1.2172,
            1.24004, 1.2477, 1.25039, 1.25016, 1.24994, 1.24884, 1.22832,
        ],
        "sl_pce": [
            5.8563, 11.8335, 18.0781, 24.4183, 25.6913, 26.7016, 28.1567,
            29.1276, 29.6316, 29.8142, 29.8, 29.6948, 22.5243, 1.5126,
        ],
        "ref_pce": [
            5.8685, 11.8314, 18.0802, 24.4188, 25.6782, 26.6926, 28.1546,
            29.1364, 29.6512, 29.8159, 29.8035, 29.7103, 21.3846, 0.7158,
        ],
    },
    "scaps_nt_pvk_etl": {
        "x": [1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15],
        "sl_voc": [1.25044, 1.24922, 1.23632, 1.16805, 1.08912, 1.0234, 0.96852],
        "ref_voc": [1.24947, 1.24825, 1.23572, 1.16755, 1.08851, 1.02281, 0.96776],
    },
    "scaps_nt_pvk_cb": {
        "x": [1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15],
        "sl_voc": [1.16815, 1.16814, 1.16814, 1.16805, 1.16715, 1.15996, 1.12982],
        "ref_voc": [1.16765, 1.16765, 1.16764, 1.16755, 1.16668, 1.15956, 1.12908],
    },
    "scaps_dvoc_mv": [
        0.018, 0.0845, 0.133, 0.1414, 0.2235, 0.2786, 0.2788, 0.2796, 0.2798, 0.3204,
        0.3214, 0.3487, 0.3497, 0.3555, 0.3659, 0.3669, 0.3765, 0.387, 0.3888, 0.3982,
        0.4019, 0.4072, 0.4075, 0.4149, 0.4238, 0.4302, 0.4312, 0.4482, 0.4482, 0.4512,
        0.4537, 0.4578, 0.4679, 0.4711, 0.4783, 0.4826, 0.4826, 0.4827, 0.4828, 0.4857,
        0.4857, 0.486, 0.486, 0.486, 0.4872, 0.4892, 0.4893, 0.4958, 0.4966, 0.4968,
        0.4968, 0.4973, 0.4976, 0.4981, 0.4981, 0.4981, 0.4981, 0.4982, 0.4986, 0.4999,
        0.5002, 0.5002, 0.5002, 0.5003, 0.5007, 0.5008, 0.5008, 0.5009, 0.5009, 0.5011,
        0.5011, 0.5012, 0.5012, 0.5012, 0.5012, 0.5012, 0.5012, 0.5012, 0.5012, 0.5012,
        0.5012, 0.5012, 0.5012, 0.5012, 0.5012, 0.5012, 0.5013, 0.5013, 0.5015, 0.5017,
        0.5017, 0.5021, 0.5045, 0.5075, 0.5083, 0.5139, 0.5146, 0.5172, 0.5172, 0.521,
        0.5361, 0.5444, 0.5574, 0.5584, 0.5658, 0.5707, 0.571, 0.5825, 0.5853, 0.5915,
        0.5915, 0.5916, 0.5917, 0.5926, 0.5926, 0.5926, 0.5939, 0.5969, 0.6016, 0.6016,
        0.6016, 0.602, 0.6026, 0.6026, 0.6028, 0.604, 0.604, 0.604, 0.6043, 0.6072,
        0.6073, 0.6079, 0.6082, 0.609, 0.6145, 0.631, 0.6391, 0.6567, 0.6576, 0.6584,
        0.6636, 0.6736, 0.678, 0.6784, 0.679, 0.6881, 0.6887, 0.707, 0.7193, 0.7193,
        0.7223, 0.741, 0.746, 0.746, 0.7463, 0.7463, 0.7467, 0.7469, 0.7488, 0.7533,
        0.7535, 0.7622, 0.7622, 0.7622, 0.7637, 0.765, 0.7651, 0.7658, 0.7744, 0.7768,
        0.7774, 0.7935, 0.7937, 0.8171, 0.8171, 0.8458, 0.854, 0.8547, 0.8905, 0.9059,
        0.9282, 0.9408, 0.9434, 0.9557, 0.9604, 0.9608, 0.9613, 0.9625, 0.9625, 0.9647,
        0.9661, 0.9662, 0.9664, 0.9703, 0.9703, 0.9703, 0.9706, 0.9706, 0.9706, 0.9707,
        0.9709, 0.971, 0.9711, 0.9713, 0.9713, 0.9714, 0.9714, 0.9714, 0.9714, 0.9726,
        0.9737, 0.9738, 0.9808, 0.9812, 0.9856, 0.9856, 0.9943, 1.0024, 1.0024, 1.0041,
        1.005, 1.0057, 1.0065, 1.0092, 1.0136, 1.0207, 1.0277, 1.028, 1.03, 1.0316,
        1.1276, 1.3703, 1.3759, 1.3839, 1.4258, 1.4595, 4.3605, 4.3605, 5.105, 5.141,
        5.1466, 9.3974, 12.1782, 12.8419, 14.2088, 16.2183, 16.2691, 48.7746,
    ],
    "hyst_diffusivity": {
        "x": [round(0.1 * i, 1) for i in range(21)],
        "sl_hi": [
            0.0, 0.978876, 0.98988, 0.969533, 0.903311, 0.77651, 0.632702,
            0.514736, 0.429763, 0.369764, 0.324657, 0.291329, 0.263569, 0.243272,
            0.225912, 0.210926, 0.198883, 0.188616, 0.179741, 0.171793, 0.164086,
        ],
        "ref_hi": [
            0.0, 0.980969, 0.989712, 0.969085, 0.900569, 0.770581, 0.627715,
            0.51184, 0.428362, 0.36924, 0.324093, 0.291422, 0.263461, 0.243117,
            0.226595, 0.211598, 0.199422, 0.189036, 0.180006, 0.173009, 0.165746,
        ],
    },
    "hyst_density": {
        "x": [round(0.1 * i, 1) for i in range(21)],
        "sl_hi": [
            0.0, 0.786028, 0.906444, 0.893521, 0.822121, 0.705749, 0.585937,
            0.489497, 0.418228, 0.365744, 0.324657, 0.29339, 0.266515, 0.246507,
            0.229714, 0.21488, 0.202537, 0.191924, 0.183051, 0.175614, 0.1683,
        ],
        "ref_hi": [
            0.0, 0.790229, 0.90905, 0.894722, 0.820991, 0.702693, 0.583489,
            0.488178, 0.417547, 0.36549, 0.324093, 0.293308, 0.26605, 0.246012,
            0.229764, 0.215064, 0.202597, 0.19187, 0.183376, 0.176122, 0.169634,
        ],
    },
    "hyst_d_hi_norm": [
        0.0, 1e-08, 5.057e-05, 5.434e-05, 5.945e-05, 8.198e-05, 9.35e-05,
        0.0001081, 0.00015549, 0.00016725, 0.00018479, 0.00025384, 0.00026566,
        0.00032471, 0.0004201, 0.00044813, 0.00046529, 0.0004952, 0.0005083,
        0.00052425, 0.00053934, 0.000564, 0.00056401, 0.00056401, 0.00067152,
        0.00068134, 0.00068287, 0.00113006, 0.00120031, 0.0012159, 0.0013193,
        0.00133377, 0.0014009, 0.00165992, 0.00170012, 0.00209318, 0.00244765,
        0.00260665, 0.00274146, 0.00289617, 0.00305575, 0.00420085, 0.00498709,
        0.00592866, 0.00607415,
    ],
    "rep_curves": {
        "hi_density_00": {
            "v_fwd": [
                -0.2, -0.12, -0.04, 0.04, 0.12, 0.2, 0.28, 0.36, 0.44, 0.52,
                0.6, 0.68, 0.72, 0.76, 0.8, 0.84, 0.88, 0.9,
            ],
            "j_fwd": [
                15.47, 15.439, 15.404, 15.366, 15.323, 15.274, 15.218, 15.154,
                15.079, 14.989, 14.88, 14.74, 14.642, 14.482, 14.083, 12.688,
                6.974, -0.995,
            ],
            "v_rev": [
                0.9, 0.86, 0.82, 0.78, 0.74, 0.7, 0.62, 0.54, 0.46, 0.38,
                0.3, 0.22, 0.14, 0.06, -0.02, -0.1, -0.2,
            ],
            "j_rev": [
                -0.995, 10.819, 13.613, 14.336, 14.575, 14.695, 14.849, 14.964,
                15.058, 15.136, 15.203, 15.261, 15.311, 15.355, 15.395, 15.431,
                15.47,
            ],
        },
        "hi_rate_20": {
            "v_fwd": [
                -0.2, -0.12, -0.04, 0.04, 0.12, 0.2, 0.28, 0.36, 0.4, 0.44,
                0.48, 0.52, 0.56, 0.6, 0.64, 0.68,
            ],
            "j_fwd": [
                15.955, 15.945, 15.933, 15.918, 15.897, 15.865, 15.794, 15.577,
                15.308, 14.789, 13.807, 12.045, 9.172, 5.094, 0.077, -5.821,
            ],
            "v_rev": [
                0.76, 0.72, 0.68, 0.64, 0.6, 0.56, 0.52, 0.48, 0.4, 0.32,
                0.24, 0.16, 0.08, 0.0, -0.1, -0.2,
            ],
            "j_rev": [
                -2.493, 5.306, 9.213, 11.476, 13.024, 14.153, 14.933, 15.422,
                15.856, 15.976, 16.007, 16.014, 16.017, 16.017, 16.018, 16.018,
            ],
        },
        "hi_rate_40": {
            "v_fwd": [
                -0.2, -0.12, -0.04, 0.04, 0.12, 0.2, 0.28, 0.36, 0.4, 0.44,
                0.48, 0.52, 0.56, 0.6, 0.64,
            ],
            "j_fwd": [
                15.709, 15.65, 15.577, 15.485, 15.364, 15.196, 14.917, 14.291,
                13.629, 12.458, 10.435, 7.218, 2.847, -1.941, -6.239,
            ],
            "v_rev": [
                0.78, 0.74, 0.7, 0.66, 0.62, 0.58, 0.54, 0.5, 0.42, 0.34,
                0.26, 0.18, 0.1, 0.02, -0.1, -0.2,
            ],
            "j_rev": [
                -0.513, 7.448, 10.704, 12.24, 13.166, 13.891, 14.526, 15.056,
                15.705, 15.936, 16.0, 16.018, 16.02, 16.02, 16.02, 16.021,
            ],
        },
        "hi_rate_80": {
            "v_fwd": [
                -0.2, -0.12, -0.04, 0.04, 0.12, 0.2, 0.28, 0.36, 0.4, 0.44,
                0.48, 0.52,
            ],
            "j_fwd": [
                12.56, 11.754, 10.87, 9.912, 8.879, 7.758, 6.467, 4.629, 3.167,
                1.007, -2.036, -5.635,
            ],
            "v_rev": [
                0.78, 0.74, 0.7, 0.66, 0.62, 0.58, 0.54, 0.5, 0.42, 0.34,
                0.26, 0.18, 0.1, 0.02, -0.1, -0.2,
            ],
            "j_rev": [
                -3.81, 7.069, 10.977, 12.57, 13.389, 13.945, 14.404, 14.809,
                15.447, 15.806, 15.954, 16.012, 16.02, 16.021, 16.021, 16.021,
            ],
        },
    },
}


def _rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fc: str,
    ec: str,
    lw: float = 1.3,
    radius: float = 0.016,
    accent: str | None = None,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.0,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)
    if accent is not None:
        bar = Rectangle((x, y + h - 0.022), w, 0.022, facecolor=accent, edgecolor="none")
        bar.set_clip_path(patch)
        ax.add_patch(bar)


def _arrow(
    ax: plt.Axes,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    color: str = C_NAVY,
    lw: float = 1.6,
    style: str = "-|>",
    scale: float = 13.0,
) -> None:
    arr = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle=style,
        mutation_scale=scale,
        linewidth=lw,
        color=color,
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(arr)


def _clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color=C_GRID, linewidth=0.75, alpha=0.9)
    ax.set_axisbelow(True)


def generate_architecture_flow() -> Path:
    """1. ArchitectureFlow.png: End-to-end dataflow and package boundaries."""
    fig, ax = plt.subplots(figsize=(13.6, 4.4), dpi=DPI)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(
        0.02,
        0.93,
        "SolarLab End-to-End Simulation & Provenance Pipeline",
        fontsize=13.5,
        fontweight="bold",
        color=C_INK,
        va="center",
    )
    ax.text(
        0.98,
        0.93,
        "Strict pre-solve capability gates · Zero implicit solver fallback",
        fontsize=9.5,
        color=C_MUTED,
        ha="right",
        va="center",
    )

    stages = [
        (
            "1. Declarations",
            "YAML / SCAPS / JSON\nWeb Stack Editor\nPackaged Material Data",
            C_BG_SLATE,
            C_NAVY,
        ),
        (
            "2. Preparation",
            "Typed DeviceInput\nImmutable PreparedDevice\nBeer-Lambert / TMM Optics",
            C_BG_BLUE,
            C_BLUE,
        ),
        (
            "3. Sparse Assembly",
            "Unified SystemLayout\nExact Analytical Jacobian\n1D & 2D Control Volumes",
            C_BG_TEAL,
            C_TEAL,
        ),
        (
            "4. Verified Solvers",
            "Damped Newton DC\nConservative BE / BDF\nExact Complex AC",
            C_BG_AMBER,
            C_AMBER,
        ),
        (
            "5. RunStore & UI",
            "SHA-256 ExecutionManifest\nLossless NPY / JSON Read\nSSE Stream & Plotly UI",
            C_BG_PURPLE,
            C_PURPLE,
        ),
    ]

    w = 0.172
    gap = 0.025
    x0 = 0.02
    y_top = 0.39
    h_top = 0.44

    for idx, (title, body, bg, accent) in enumerate(stages):
        x = x0 + idx * (w + gap)
        _rounded_box(ax, x, y_top, w, h_top, fc=bg, ec=accent, lw=1.35, accent=accent)
        ax.text(
            x + w / 2,
            y_top + h_top - 0.085,
            title,
            fontsize=10.8,
            fontweight="bold",
            color=C_INK,
            ha="center",
            va="center",
        )
        ax.text(
            x + w / 2,
            y_top + 0.175,
            body,
            fontsize=9.3,
            color=C_TEXT,
            ha="center",
            va="center",
            linespacing=1.48,
        )
        if idx < len(stages) - 1:
            _arrow(
                ax,
                x + w + 0.003,
                y_top + h_top / 2,
                x + w + gap - 0.003,
                y_top + h_top / 2,
                color=C_NAVY,
                lw=1.7,
            )

    # Bottom package boundary bar
    _rounded_box(
        ax, 0.02, 0.06, 0.96, 0.24, fc="#F8FAFC", ec=C_LINE, lw=1.1, radius=0.014
    )
    ax.text(
        0.04,
        0.23,
        "Enforced First-Party Package Boundaries (12 / 12 import-linter Contracts)",
        fontsize=10.0,
        fontweight="bold",
        color=C_INK,
        va="center",
    )

    pkg_specs = [
        (0.035, 0.335, "src/solarlab · Core Physics, Mesh, Assembler & Solvers", C_BLUE),
        (0.385, 0.315, "src/solarlab_server · HTTP/SSE API & SPA", C_TEAL),
        (0.715, 0.250, "src/solarlab_research · R1 & CBO CLIs", C_PURPLE),
    ]
    for px, pw, label, col in pkg_specs:
        _rounded_box(ax, px, 0.085, pw, 0.095, fc="white", ec=col, lw=1.15, radius=0.01)
        ax.text(
            px + pw / 2,
            0.132,
            label,
            fontsize=8.3,
            fontweight="bold",
            color=col,
            ha="center",
            va="center",
        )

    out = OUT_DIR / "ArchitectureFlow.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close(fig)
    return out


def generate_device_physics_overview() -> Path:
    """2. DevicePhysicsOverview.png: 1D stack/contacts, band diagram, and 2D grain mesh."""
    fig = plt.figure(figsize=(13.8, 4.7), dpi=DPI, facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.08, 1.15, 0.97], wspace=0.20)

    # Panel (a): 1D Multilayer Stack & Contact Boundary
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_xlim(0.0, 1.0)
    ax_a.set_ylim(0.0, 1.0)
    ax_a.axis("off")
    ax_a.text(0.0, 0.96, "(a)  Multilayer Stack & Contact Laws", fontsize=11.5, fontweight="bold", color=C_INK)

    layers = [
        (0.03, 0.16, "Left\nContact", "#E2E8F0", C_TEXT),
        (0.19, 0.16, "HTL", "#DBEAFE", C_BLUE),
        (0.35, 0.31, "Perovskite\nAbsorber\n($n, p, P^+, P^-$)", "#FEF3C7", C_AMBER),
        (0.66, 0.16, "ETL", "#CCFBF1", C_TEAL),
        (0.82, 0.16, "Right\nContact", "#E2E8F0", C_TEXT),
    ]
    for lx, lw, lbl, fc, tcol in layers:
        rect = Rectangle((lx, 0.44), lw, 0.33, facecolor=fc, edgecolor=C_INK, linewidth=1.1)
        ax_a.add_patch(rect)
        ax_a.text(
            lx + lw / 2,
            0.605,
            lbl,
            fontsize=8.6 if lw <= 0.16 else 9.0,
            fontweight="bold",
            color=tcol,
            ha="center",
            va="center",
            linespacing=1.25,
        )

    # Illumination arrow
    _arrow(ax_a, 0.11, 0.89, 0.11, 0.79, color=C_AMBER, lw=2.0)
    ax_a.text(0.145, 0.845, "AM1.5G Spectrum (BL / Coherent TMM)", fontsize=8.8, fontweight="bold", color=C_AMBER, va="center")

    # Coordinate axis
    _arrow(ax_a, 0.19, 0.36, 0.82, 0.36, color=C_INK, lw=1.3)
    ax_a.text(0.19, 0.305, "$x = 0$", fontsize=9.2, color=C_INK, ha="center")
    ax_a.text(0.505, 0.305, "Electrical stack axis $x$", fontsize=9.0, color=C_MUTED, ha="center")
    ax_a.text(0.82, 0.305, "$x = L$", fontsize=9.2, color=C_INK, ha="center")

    _rounded_box(ax_a, 0.02, 0.04, 0.96, 0.21, fc=C_BG_SLATE, ec=C_LINE, lw=1.0, radius=0.015)
    ax_a.text(
        0.50,
        0.145,
        "Electrostatics:  φ(0) = 0,  φ($L$) = $V_{\\mathrm{bi}}$ − $s \\cdot V_{\\mathrm{app}}$\n"
        "Carriers: Ohmic or Robin $S$  ·  Ions: $J_{\\mathrm{ion}}(0) = J_{\\mathrm{ion}}(L) = 0$",
        fontsize=8.4,
        color=C_TEXT,
        ha="center",
        va="center",
        linespacing=1.48,
    )

    # Panel (b): Illuminated Band Bending & Reciprocal Two-Sided Interface
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_xlim(0.0, 1.0)
    ax_b.set_ylim(-2.45, 2.85)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.set_xticks([0.125, 0.50, 0.875])
    ax_b.set_xticklabels(["HTL", "Absorber", "ETL"], fontsize=9.5, fontweight="bold")
    ax_b.set_yticks([])
    ax_b.set_ylabel("Energy (eV, schematic)", fontsize=10.0, color=C_TEXT)
    ax_b.set_title("(b)  Band Bending & Two-Sided Heterointerface", fontsize=11.5, fontweight="bold", color=C_INK, loc="left", pad=10)

    ax_b.axvspan(0.0, 0.25, color="#EFF6FF", alpha=0.75)
    ax_b.axvspan(0.25, 0.75, color="#FFFBEB", alpha=0.75)
    ax_b.axvspan(0.75, 1.0, color="#F0FDFA", alpha=0.75)
    ax_b.axvline(0.25, color=C_LINE, lw=1.1, ls="-")
    ax_b.axvline(0.75, color=C_LINE, lw=1.1, ls="-")

    x1 = np.linspace(0.0, 0.25, 40)
    x2 = np.linspace(0.25, 0.75, 80)
    x3 = np.linspace(0.75, 1.0, 40)

    ec1 = 1.90 - 0.12 * (x1 / 0.25) ** 1.5
    ev1 = ec1 - 2.25
    ec2 = 1.12 - 0.32 * (x2 - 0.25) / 0.50
    ev2 = ec2 - 1.60
    ec3 = 0.70 - 0.10 * ((x3 - 0.75) / 0.25) ** 0.7
    ev3 = ec3 - 2.65

    ax_b.plot(np.r_[x1, x2, x3], np.r_[ec1, ec2, ec3], color=C_BLUE, lw=2.3, label="Conduction band $E_{\\mathrm{c}}$")
    ax_b.plot(np.r_[x1, x2, x3], np.r_[ev1, ev2, ev3], color=C_CORAL, lw=2.3, label="Valence band $E_{\\mathrm{v}}$")

    efn_x = np.linspace(0.10, 0.95, 60)
    efp_x = np.linspace(0.05, 0.90, 60)
    ax_b.plot(efn_x, 0.50 - 0.10 * efn_x, color=C_TEAL, lw=1.8, ls="--", label="Electron QFL $E_{\\mathrm{Fn}}$")
    ax_b.plot(efp_x, -0.22 - 0.08 * efp_x, color=C_PURPLE, lw=1.8, ls="--", label="Hole QFL $E_{\\mathrm{Fp}}$")

    ax_b.text(0.27, 1.32, "Δ$E_{\\mathrm{c}}$ (reciprocal TE)", fontsize=8.6, fontweight="bold", color=C_BLUE)
    ax_b.text(0.31, -1.72, "Two-sided / 7-level\nGaussian interface SRH", fontsize=8.5, color=C_MUTED, linespacing=1.25)
    ax_b.legend(loc="upper right", frameon=True, facecolor="white", edgecolor=C_LINE, fontsize=8.2)

    # Panel (c): 2D Grain-Boundary Overlap & Mobile-Ion Extrusion
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.set_xlim(0.0, 1.0)
    ax_c.set_ylim(0.0, 1.0)
    ax_c.axis("off")
    ax_c.text(0.0, 0.96, "(c)  2D Grain & Ion Mesh ($N_y$ × $N_x$)", fontsize=11.5, fontweight="bold", color=C_INK)

    mx0, my0, mw, mh = 0.10, 0.35, 0.80, 0.49
    ax_c.add_patch(Rectangle((mx0, my0 + 0.37), mw, 0.12, facecolor="#DBEAFE", edgecolor="none"))
    ax_c.add_patch(Rectangle((mx0, my0 + 0.12), mw, 0.25, facecolor="#FEF3C7", edgecolor="none"))
    ax_c.add_patch(Rectangle((mx0, my0), mw, 0.12, facecolor="#CCFBF1", edgecolor="none"))
    ax_c.add_patch(
        Rectangle((mx0 + 0.35, my0 + 0.12), 0.10, 0.25, facecolor="#FCA5A5", alpha=0.85, edgecolor=C_CORAL, lw=1.1)
    )
    for gx in np.linspace(mx0, mx0 + mw, 7):
        ax_c.plot([gx, gx], [my0, my0 + mh], color="#94A3B8", lw=0.75)
    for gy in np.linspace(my0, my0 + mh, 7):
        ax_c.plot([mx0, mx0 + mw], [gy, gy], color="#94A3B8", lw=0.75)
    ax_c.add_patch(Rectangle((mx0, my0), mw, mh, facecolor="none", edgecolor=C_INK, lw=1.2))

    ax_c.text(
        mx0 + mw / 2,
        my0 + 0.245,
        "GB overlap $f_{\\mathrm{GB}}$",
        fontsize=8.4,
        fontweight="bold",
        color=C_CORAL,
        ha="center",
        va="center",
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": C_CORAL, "lw": 0.9, "alpha": 0.95},
    )
    ax_c.text(mx0 + mw / 2, my0 - 0.055, "Lateral coordinate $x$ (periodic / Neumann)", fontsize=8.7, color=C_MUTED, ha="center")

    _rounded_box(ax_c, 0.04, 0.04, 0.92, 0.21, fc=C_BG_SLATE, ec=C_LINE, lw=1.0, radius=0.015)
    ax_c.text(
        0.50,
        0.145,
        "Exact CV area-fraction GB recombination\n"
        "Frozen-ion DC or coupled 2D mobile-ion BE\n"
        "Sparse scaling β ≈ 0.99 (49× faster at $N_x = 16$)",
        fontsize=8.5,
        color=C_TEXT,
        ha="center",
        va="center",
        linespacing=1.38,
    )

    out = OUT_DIR / "DevicePhysicsOverview.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close(fig)
    return out


def generate_solver_topology() -> Path:
    """3. SolverTopology.png: Unified solver matrix, unknowns, and certificates."""
    fig, ax = plt.subplots(figsize=(13.6, 5.2), dpi=DPI)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(
        0.02,
        0.945,
        "SolarLab Unified Equation Assembly & Solver Matrix",
        fontsize=13.5,
        fontweight="bold",
        color=C_INK,
        va="center",
    )
    ax.text(
        0.98,
        0.945,
        "Positive log-activity carrier variables (n > 0, p > 0 by construction)",
        fontsize=9.5,
        color=C_MUTED,
        ha="right",
        va="center",
    )

    cols = [
        (0.02, 0.19, "Execution Lane"),
        (0.225, 0.21, "Primary Unknowns"),
        (0.45, 0.27, "Assembled Discrete Physics"),
        (0.735, 0.245, "Solver & Gate Certificate"),
    ]
    for cx, cw, header in cols:
        _rounded_box(ax, cx, 0.81, cw, 0.085, fc="#E2E8F0", ec="#CBD5E1", lw=1.0, radius=0.01)
        ax.text(cx + cw / 2, 0.852, header, fontsize=10.0, fontweight="bold", color=C_INK, ha="center", va="center")

    rows = [
        (
            "1D Electronic DC",
            "[φ, $z_n$, $z_p$]",
            "SG carrier continuity + series Poisson\nTwo-sided TE & Gaussian interface SRH",
            "Damped sparse Newton\nDirect bracketed $V_{\\mathrm{oc}}$ refinement",
            C_BLUE,
            C_BG_BLUE,
        ),
        (
            "1D Mobile-Ion &\nMulti-Domain",
            "[φ, $z_n$, $z_p$, $P^+$, ($P^-$)]",
            "Finite-site steric PNP + multi-domain\nCancellation-free expm1 inventories",
            "Backward Euler / adaptive BDF1-2\nIon drift ≤ 1.8 × $10^{-13}$",
            C_TEAL,
            C_BG_TEAL,
        ),
        (
            "Open-Circuit DAE\n(TPV / Suns-$V_{\\mathrm{oc}}$)",
            "[φ, $z_n$, $z_p$, ($P$), $V_{\\mathrm{term}}$]",
            "Floating-terminal Maxwell closure\n$J_{\\mathrm{cond}}$ + d$D$/d$t$ = 0 on every face",
            "Coupled port-charge integrator\nPaired unpulsed drift subtraction",
            C_AMBER,
            C_BG_AMBER,
        ),
        (
            "Small-Signal AC\n(Impedance / C-V)",
            "δ$u$(ω)  (complex phasor)",
            "Exact analytical tangent linearization\n($G$ + iω$C$) δ$u$ = $b$  with ion/defect AC",
            "Direct sparse complex LU\nAll-face Maxwell admittance check",
            C_PURPLE,
            C_BG_PURPLE,
        ),
        (
            "2D Microstructure\n& Dynamic Ion J-V",
            "[φ, $z_n$, $z_p$, ($P^+$)]($y$, $x$)",
            "2D control-volume GB overlap\n2D SG carrier & mobile-ion fluxes",
            "Sparse 2D Newton & BE\nCross-section current continuity",
            C_CORAL,
            C_BG_ROSE,
        ),
    ]

    row_h = 0.128
    row_gap = 0.022
    top_y = 0.655
    for idx, (lane, unk, disc, cert, accent, bg) in enumerate(rows):
        ry = top_y - idx * (row_h + row_gap)
        cells = [
            (cols[0][0], cols[0][1], lane, bg, accent, "bold", C_INK),
            (cols[1][0], cols[1][1], unk, "#F8FAFC", C_LINE, "bold", C_NAVY),
            (cols[2][0], cols[2][1], disc, "white", C_LINE, "normal", C_TEXT),
            (cols[3][0], cols[3][1], cert, "white", C_LINE, "normal", C_TEXT),
        ]
        for j, (cx, cw, txt, fc, ec, weight, tcol) in enumerate(cells):
            _rounded_box(ax, cx, ry, cw, row_h, fc=fc, ec=ec, lw=1.2 if j == 0 else 1.0, radius=0.012)
            if j == 0:
                strip = Rectangle((cx, ry), 0.008, row_h, facecolor=accent, edgecolor="none")
                ax.add_patch(strip)
            ax.text(
                cx + cw / 2,
                ry + row_h / 2,
                txt,
                fontsize=9.3 if j > 0 else 9.6,
                fontweight=weight,
                color=tcol,
                ha="center",
                va="center",
                linespacing=1.35,
            )

    out = OUT_DIR / "SolverTopology.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close(fig)
    return out


def generate_validation_scaps() -> Path:
    """4. ValidationScapsOverview.png: Native zero-fit 262-case SCAPS-1D alignment summary."""
    fig = plt.figure(figsize=(13.8, 4.5), dpi=DPI, facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 1.02], wspace=0.25)

    # (a) CBO sweep (Voc)
    ax_a = fig.add_subplot(gs[0, 0])
    _clean_axes(ax_a)
    cbo = BENCHMARK_DATA["scaps_cbo"]
    ax_a.plot(cbo["x"], cbo["ref_voc"], "s--", color=C_INK, ms=5.6, lw=1.6, label="SCAPS-1D $V_{\\mathrm{oc}}$")
    ax_a.plot(cbo["x"], cbo["sl_voc"], "o-", color=C_BLUE, ms=4.2, lw=1.9, label="SolarLab $V_{\\mathrm{oc}}$")
    ax_a.set_xlabel("Conduction-Band Offset Δ$E_{\\mathrm{c}}$ (eV)")
    ax_a.set_ylabel("Open-Circuit Voltage $V_{\\mathrm{oc}}$ (V)", color=C_INK)
    ax_a.set_xlim(-1.05, 0.55)
    ax_a.set_ylim(0.25, 1.35)
    ax_a.set_title("(a)  Heterointerface Δ$E_{\\mathrm{c}}$ Sweep", fontweight="bold", color=C_INK, loc="left", pad=8)
    ax_a.legend(loc="lower right", frameon=True, facecolor="white", edgecolor=C_LINE, fontsize=8.8)
    ax_a.text(
        -0.98,
        1.20,
        "Native zero-fit parity\nMed |Δ$V_{\\mathrm{oc}}$| = 0.53 mV",
        fontsize=8.6,
        color=C_MUTED,
        linespacing=1.35,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": C_BG_SLATE, "edgecolor": C_LINE},
    )

    # (b) Interface & Bulk Defect Density Sweeps
    ax_b = fig.add_subplot(gs[0, 1])
    _clean_axes(ax_b)
    nt_if = BENCHMARK_DATA["scaps_nt_pvk_etl"]
    nt_bk = BENCHMARK_DATA["scaps_nt_pvk_cb"]
    ax_b.semilogx(nt_if["x"], nt_if["ref_voc"], "s--", color=C_INK, ms=5.6, lw=1.5, label="SCAPS PVK/ETL $N_{\\mathrm{t}}$")
    ax_b.semilogx(nt_if["x"], nt_if["sl_voc"], "o-", color=C_TEAL, ms=4.2, lw=1.9, label="SolarLab PVK/ETL $N_{\\mathrm{t}}$")
    ax_b.semilogx(nt_bk["x"], nt_bk["ref_voc"], "^--", color=C_MUTED, ms=5.4, lw=1.4, label="SCAPS Bulk $N_{\\mathrm{t}}$")
    ax_b.semilogx(nt_bk["x"], nt_bk["sl_voc"], "d-", color=C_AMBER, ms=4.2, lw=1.8, label="SolarLab Bulk $N_{\\mathrm{t}}$")
    ax_b.set_xlabel("Defect Parameter Scale ($10^9$ to $10^{15}$)")
    ax_b.set_ylabel("Open-Circuit Voltage $V_{\\mathrm{oc}}$ (V)")
    ax_b.set_ylim(0.93, 1.29)
    ax_b.set_title("(b)  7-Level Gaussian Defect Sweeps", fontweight="bold", color=C_INK, loc="left", pad=8)
    ax_b.legend(loc="lower left", frameon=True, facecolor="white", edgecolor=C_LINE, fontsize=8.3)

    # (c) 248 Numeric Cases |ΔVoc| ECDF (log scale to resolve sub-mV clustering)
    ax_c = fig.add_subplot(gs[0, 2])
    _clean_axes(ax_c)
    dvoc = np.asarray(BENCHMARK_DATA["scaps_dvoc_mv"], dtype=float)
    ecdf_y = np.arange(1, dvoc.size + 1) / dvoc.size * 100.0
    ax_c.semilogx(dvoc, ecdf_y, color=C_BLUE, lw=2.3, label="|Δ$V_{\\mathrm{oc}}$| (248 cases)")
    ax_c.axvline(2.0, color=C_TEAL, lw=1.5, ls="--", label="2 mV gate (95.2%)")
    ax_c.scatter([np.median(dvoc), np.percentile(dvoc, 90)], [50.0, 90.0], color=C_CORAL, zorder=5, s=34)
    ax_c.set_xlim(0.012, 220.0)
    ax_c.set_ylim(0.0, 103.5)
    ax_c.set_xticks([0.02, 0.1, 0.5, 2.0, 10.0, 100.0])
    fmt = ScalarFormatter()
    fmt.set_scientific(False)
    ax_c.xaxis.set_major_formatter(fmt)
    ax_c.set_xticklabels(["0.02", "0.1", "0.5", "2", "10", "100"])
    ax_c.xaxis.set_minor_formatter(NullFormatter())
    ax_c.set_xlabel("Absolute Voltage Deviation |Δ$V_{\\mathrm{oc}}$| (mV)")
    ax_c.set_ylabel("Cumulative Cases (%)")
    ax_c.set_title("(c)  Full 262-Case Zero-Fit Scorecard", fontweight="bold", color=C_INK, loc="left", pad=8)
    ax_c.legend(
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor=C_LINE,
        fontsize=7.9,
        handlelength=1.6,
        borderpad=0.35,
    )
    ax_c.text(
        0.975,
        0.055,
        "262 / 262 converged (0 failed)\n"
        "221 / 248 (89.1%) pass all 4 gates\n"
        "Median |Δ$V_{\\mathrm{oc}}$| = 0.60 mV\n"
        "Median |Δ$J_{\\mathrm{sc}}$| = 0.009 mA $\\mathrm{cm}^{-2}$\n"
        "Median |ΔFF| = 0.031%\n"
        "Median |ΔPCE| = 0.009%",
        transform=ax_c.transAxes,
        ha="right",
        va="bottom",
        multialignment="left",
        zorder=4,
        fontsize=7.8,
        color=C_TEXT,
        linespacing=1.36,
        bbox={"boxstyle": "round,pad=0.32", "facecolor": C_BG_BLUE, "edgecolor": C_BLUE, "lw": 0.9},
    )

    out = OUT_DIR / "ValidationScapsOverview.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close(fig)
    return out


def generate_validation_hysteresis() -> Path:
    """5. ValidationHysteresisOverview.png: Verified 45-case dynamic ion hysteresis."""
    fig = plt.figure(figsize=(13.8, 4.5), dpi=DPI, facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.08, 1.05, 0.97], wspace=0.25)

    # (a) Representative Forward/Reverse J-V Loops
    ax_a = fig.add_subplot(gs[0, 0])
    _clean_axes(ax_a)
    rc = BENCHMARK_DATA["rep_curves"]
    loop_specs = [
        ("hi_density_00", "Ion-free control", C_INK),
        ("hi_rate_20", "20 mV/s (HI = 0.16)", C_TEAL),
        ("hi_rate_40", "40 mV/s (HI = 0.32)", C_BLUE),
        ("hi_rate_80", "80 mV/s (HI = 0.78)", C_CORAL),
    ]
    for key, label, col in loop_specs:
        c = rc[key]
        ax_a.plot(c["v_fwd"], c["j_fwd"], "-", color=col, lw=1.9, label=label)
        if key != "hi_density_00":
            ax_a.plot(c["v_rev"], c["j_rev"], "--", color=col, lw=1.8)
    ax_a.axhline(0.0, color=C_SUBTLE, lw=0.8)
    ax_a.set_xlim(-0.2, 0.94)
    ax_a.set_ylim(-5.5, 17.5)
    ax_a.set_xlabel("Applied Voltage $V$ (V)")
    ax_a.set_ylabel("Current Density $J$ (mA $\\mathrm{cm}^{-2}$)")
    ax_a.set_title("(a)  Dynamic Forward (—) & Reverse (--) J-V", fontweight="bold", color=C_INK, loc="left", pad=8)
    ax_a.legend(loc="lower left", frameon=True, facecolor="white", edgecolor=C_LINE, fontsize=8.3)

    # (b) 45-Case Diffusivity & Density Bell Curves
    ax_b = fig.add_subplot(gs[0, 1])
    _clean_axes(ax_b)
    hd = BENCHMARK_DATA["hyst_diffusivity"]
    hn = BENCHMARK_DATA["hyst_density"]
    ax_b.plot(hd["x"], hd["ref_hi"], "s--", color=C_INK, ms=4.5, lw=1.3, label="Reference ($D_{\\mathrm{ion}}$ axis)")
    ax_b.plot(hd["x"], hd["sl_hi"], "o-", color=C_BLUE, ms=4.6, lw=1.9, label="SolarLab ($D_{\\mathrm{ion}}$ axis)")
    ax_b.plot(hn["x"], hn["ref_hi"], "^--", color=C_MUTED, ms=4.5, lw=1.3, label="Reference ($P_0$ axis)")
    ax_b.plot(hn["x"], hn["sl_hi"], "d-", color=C_TEAL, ms=4.6, lw=1.9, label="SolarLab ($P_0$ axis)")
    ax_b.set_xlim(-0.05, 2.05)
    ax_b.set_ylim(-0.03, 1.06)
    ax_b.set_xlabel("Normalized Sweep Coordinate (0.0 to 2.0)")
    ax_b.set_ylabel("Normalized Hysteresis Index $\\mathrm{HI}_{\\mathrm{norm}}$")
    ax_b.set_title("(b)  45-Case Hysteresis Matrix Overlay", fontweight="bold", color=C_INK, loc="left", pad=8)
    ax_b.legend(loc="upper right", frameon=True, facecolor="white", edgecolor=C_LINE, fontsize=8.2)

    # (c) |ΔHI_norm| Distribution & Conservation Summary
    ax_c = fig.add_subplot(gs[0, 2])
    _clean_axes(ax_c)
    dhi = np.asarray(BENCHMARK_DATA["hyst_d_hi_norm"], dtype=float) * 1e3
    ecdf_y = np.arange(1, dhi.size + 1) / dhi.size * 100.0
    ax_c.plot(dhi, ecdf_y, color=C_TEAL, lw=2.3, label="|Δ$\\mathrm{HI}_{\\mathrm{norm}}$| (45 cases)")
    ax_c.axvline(5.0, color=C_AMBER, lw=1.5, ls="--", label="5 × $10^{-3}$ band (95.6%)")
    ax_c.scatter([np.median(dhi)], [50.0], color=C_CORAL, zorder=5, s=36)
    ax_c.set_xlim(0.0, 6.6)
    ax_c.set_ylim(0.0, 103.0)
    ax_c.set_xlabel("Deviation |Δ$\\mathrm{HI}_{\\mathrm{norm}}$| (× $10^{-3}$)")
    ax_c.set_ylabel("Cumulative Cases (%)")
    ax_c.set_title("(c)  Accuracy & Ion Conservation", fontweight="bold", color=C_INK, loc="left", pad=8)
    ax_c.legend(loc="lower right", frameon=True, facecolor="white", edgecolor=C_LINE, fontsize=8.4)
    ax_c.text(
        2.05,
        36.0,
        "45 / 45 completed (0 stalled)\n"
        "Median |Δ$\\mathrm{HI}_{\\mathrm{norm}}$| = 5.6 × $10^{-4}$\n"
        "43 / 45 (95.6%) ≤ 5 × $10^{-3}$\n"
        "Max |Δ$\\mathrm{HI}_{\\mathrm{norm}}$| = 6.1 × $10^{-3}$\n"
        "Max ion drift = 1.8 × $10^{-13}$",
        fontsize=8.6,
        color=C_TEXT,
        linespacing=1.42,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": C_BG_TEAL, "edgecolor": C_TEAL, "lw": 0.9},
    )

    out = OUT_DIR / "ValidationHysteresisOverview.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    configure_strict_arial()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    generated = (
        generate_architecture_flow(),
        generate_device_physics_overview(),
        generate_solver_topology(),
        generate_validation_scaps(),
        generate_validation_hysteresis(),
    )
    for path in generated:
        print(f"Generated {path.relative_to(ROOT)} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
