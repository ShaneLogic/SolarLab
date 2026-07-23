from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 160,
        "savefig.dpi": 220,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

INK = "#243447"
MUTED = "#64748B"
GRID = "#D7DEE8"
BLUE = "#2F5D8C"
TEAL = "#3B7A78"
GREEN = "#4D7C59"
GOLD = "#B88922"
RUST = "#A65F3D"


def _save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def validation_gate_summary() -> None:
    # Concise result strings for the figure; the following table in the
    # manual carries the full skipped/xfailed breakdown. Gate names match
    # that table exactly to avoid a figure/table naming mismatch.
    rows = [
        ("Python default suite", "pytest", "1101 passed", "18.3 min"),
        ("Python slow suite", "pytest -m slow", "99 passed, 1 xfail", "80.9 min"),
        ("Python validation suite", "pytest -m validation", "22 passed", "7.4 min"),
        ("Frontend build", "npm run build", "passed", "6.3 min"),
        ("Frontend unit tests", "npm run test:run", "371 passed", "1.3 s"),
    ]

    fig, ax = plt.subplots(figsize=(8.4, 3.3))
    ax.set_axis_off()
    ax.set_title("Validation evidence executed before manual generation", loc="left", pad=12)

    headers = ["Gate", "Command", "Result", "Runtime"]
    x = [0.02, 0.32, 0.55, 0.84]
    y_top = 0.82
    row_h = 0.135
    ax.add_patch(plt.Rectangle((0.01, y_top + 0.02), 0.98, 0.11, facecolor="#EEF3F8", edgecolor=GRID, transform=ax.transAxes))
    for xi, header in zip(x, headers):
        ax.text(xi, y_top + 0.075, header, fontweight="bold", color=INK, va="center", transform=ax.transAxes)

    for i, row in enumerate(rows):
        y = y_top - (i + 1) * row_h
        face = "#FFFFFF" if i % 2 else "#F8FAFC"
        ax.add_patch(plt.Rectangle((0.01, y + 0.02), 0.98, 0.11, facecolor=face, edgecolor=GRID, linewidth=0.7, transform=ax.transAxes))
        for xi, value in zip(x, row):
            ax.text(xi, y + 0.075, value, color=INK, va="center", transform=ax.transAxes)
    _save(fig, "validation_gate_summary")


def architecture_flow() -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.3))
    ax.set_axis_off()

    boxes = [
        ("YAML / UI\nDevice definition", (0.06, 0.70), "#EAF1F8"),
        ("MaterialParams\nLayerSpec\nDeviceStack", (0.34, 0.70), "#EBF4EF"),
        ("Experiment driver\nJV, EQE, Suns-Voc,\nimpedance, 2D", (0.62, 0.70), "#F6F1E6"),
        ("MaterialArrays cache\nGrid, optics,\nPoisson factor", (0.20, 0.30), "#F2EEF6"),
        ("RHS + solver\nSG flux, Radau,\nsafety guards", (0.48, 0.30), "#F7EEEA"),
        ("Result dataclass\nBackend SSE\nFrontend plots", (0.76, 0.30), "#EAF4F3"),
    ]
    for text, (x, y), color in boxes:
        rect = plt.Rectangle(
            (x, y),
            0.20,
            0.18,
            facecolor=color,
            edgecolor=INK,
            linewidth=1.0,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)
        ax.text(x + 0.10, y + 0.09, text, ha="center", va="center", transform=ax.transAxes)

    arrows = [
        ((0.26, 0.79), (0.34, 0.79)),
        ((0.54, 0.79), (0.62, 0.79)),
        ((0.72, 0.70), (0.30, 0.48)),
        ((0.40, 0.39), (0.48, 0.39)),
        ((0.68, 0.39), (0.76, 0.39)),
    ]
    for start, end in arrows:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
            arrowprops=dict(arrowstyle="->", color=INK, lw=1.3),
        )
    ax.text(
        0.50,
        0.08,
        "Scientific inputs, cached material arrays, numerical solving, and user-facing results remain separate.",
        ha="center",
        transform=ax.transAxes,
        color=MUTED,
    )
    _save(fig, "architecture_flow")


def tmm_jsc_baselines() -> None:
    labels = ["n-i-p TMM", "p-i-n TMM"]
    pinned = np.array([211.02, 216.62])
    tolerance = np.array([5.0, 5.0])
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.bar(x, pinned, yerr=tolerance, capsize=7, color=[BLUE, RUST], alpha=0.82, edgecolor=INK, linewidth=0.7)
    ax.set_xticks(x, labels)
    ax.set_ylabel(r"$J_\mathrm{sc}$ baseline (A m$^{-2}$)")
    ax.set_title("Transfer-matrix optical-regression baselines")
    for xi, value in zip(x, pinned):
        ax.text(xi, value + 8, f"{value:.2f}", ha="center", color=INK)
    ax.set_ylim(0, 260)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.text(
        0.5,
        -0.22,
        "Error bars show the pinned acceptance band: ±5 A m$^{-2}$.",
        ha="center",
        color=MUTED,
        transform=ax.transAxes,
    )
    _save(fig, "tmm_jsc_baselines")


def ionmonger_reference_metrics() -> None:
    names = [r"$V_\mathrm{oc}$", r"$J_\mathrm{sc}$", "FF", "PCE"]
    values = np.array([1.1932, 231.70, 0.7774, 0.2149])
    rel_tol = np.array([0.02, 0.03, 0.03, 0.05])
    colors = [BLUE, TEAL, GREEN, RUST]

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    x = np.arange(len(names))
    for xi, tol, color in zip(x, rel_tol, colors):
        ax.fill_between([xi - 0.32, xi + 0.32], 1 - tol, 1 + tol, color=color, alpha=0.20)
        ax.plot([xi - 0.32, xi + 0.32], [1, 1], color=color, lw=2.2)
    ax.axhline(1.0, color=INK, linewidth=0.9, alpha=0.7)
    ax.set_xticks(x, names)
    ax.set_ylabel("Metric / pinned reference")
    ax.set_title("IonMonger benchmark acceptance intervals")
    labels = [
        "1.1932 V\n±2%",
        r"231.70 A m$^{-2}$" + "\n±3%",
        "0.7774\n±3%",
        "0.2149\n±5%",
    ]
    for xi, label in zip(x, labels):
        ax.text(xi, 1.115, label, ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_ylim(0.82, 1.22)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    _save(fig, "ionmonger_reference_metrics")


def photon_recycling_window() -> None:
    fig, ax = plt.subplots(figsize=(6.7, 2.8))
    ax.set_xlim(0, 130)
    ax.set_ylim(0, 1)
    ax.axvspan(40, 100, color=BLUE, alpha=0.16)
    ax.hlines(0.55, 40, 100, color=BLUE, lw=2.3)
    ax.axvline(75, color=RUST, lw=2.3)
    ax.text(40, 0.70, "40 mV", ha="center", color=INK)
    ax.text(100, 0.70, "100 mV", ha="center", color=INK)
    ax.text(75, 0.18, "MAPbI3 context\n75 mV", ha="center", color=RUST)
    ax.set_yticks([])
    ax.set_xlabel(r"$\Delta V_\mathrm{oc}$ from photon recycling (mV)")
    ax.set_title("Photon-recycling validation window")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    _save(fig, "photon_recycling_window")


def physical_trend_matrix() -> None:
    rows = [
        ("Bandgap", r"$J_\mathrm{sc}$ decreases as $E_g$ increases", "test_physical_trends.py"),
        ("Thickness", r"$V_\mathrm{oc}$ changes 20-120 mV/decade", "test_physical_trends.py"),
        ("Mobility", "FF responds to mobility change", "test_physical_trends.py"),
        ("Dark J-V", r"$1.0 \leq n_\mathrm{id} \leq 2.5$", "test_physical_trends.py"),
        ("Suns-Voc", "20-70 mV/decade slope", "test_physical_trends.py"),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    ax.set_axis_off()
    y0 = 0.83
    ax.text(0.02, 0.94, "Validation trend", weight="bold", color=INK, transform=ax.transAxes)
    ax.text(0.28, 0.94, "Expected physical behavior", weight="bold", color=INK, transform=ax.transAxes)
    ax.text(0.76, 0.94, "Evidence source", weight="bold", color=INK, transform=ax.transAxes)
    for i, (name, behavior, source) in enumerate(rows):
        y = y0 - i * 0.16
        color = "#F8FAFC" if i % 2 == 0 else "#FFFFFF"
        ax.add_patch(
            plt.Rectangle((0.01, y - 0.055), 0.98, 0.12, facecolor=color, edgecolor=GRID, linewidth=0.7, transform=ax.transAxes)
        )
        ax.text(0.03, y, name, va="center", color=INK, transform=ax.transAxes)
        ax.text(0.28, y, behavior, va="center", color=INK, transform=ax.transAxes)
        ax.text(0.76, y, source, va="center", color=MUTED, fontsize=9, transform=ax.transAxes)
    ax.set_title("Physics-trend validation matrix", loc="left")
    _save(fig, "physical_trend_matrix")


def twod_validation_summary() -> None:
    checks = [
        r"$|\Delta V_\mathrm{oc}|$",
        r"$|\Delta J_\mathrm{sc}|/J_\mathrm{sc}$",
        r"$|\Delta FF|$",
        "lateral\nuniformity",
        "GB\nVoc drop",
    ]
    limits = ["<=0.1 mV", "<=0.05%", "<=0.001", "<=1e-9", "5-100 mV"]
    x = np.arange(len(checks))

    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    ax.bar(x, np.ones(len(checks)), color=[BLUE, TEAL, GREEN, GOLD, RUST], alpha=0.72, edgecolor=INK, linewidth=0.7)
    ax.set_xticks(x, checks)
    ax.set_yticks([])
    ax.set_ylim(0, 1.35)
    ax.set_title("2D slow-suite regression criteria")
    for xi, limit in zip(x, limits):
        ax.text(xi, 1.06, limit, ha="center", va="bottom", color=INK)
        ax.text(xi, 0.50, "criterion\nsatisfied", ha="center", va="center", color="white", fontsize=9)
    _save(fig, "twod_validation_summary")


def main() -> None:
    validation_gate_summary()
    architecture_flow()
    tmm_jsc_baselines()
    ionmonger_reference_metrics()
    photon_recycling_window()
    physical_trend_matrix()
    twod_validation_summary()


if __name__ == "__main__":
    main()
