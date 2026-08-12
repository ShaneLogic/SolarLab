"""Generate the figures used by the SolarLab manual and GitHub READMEs.

Diagrams are implementation summaries. Numerical figures are read from the
reproducibility registry or a machine-readable result file; they are not
transcribed into this script. Missing or inconsistent evidence stops the build.

Each figure is written as a vector PDF for the manual and a high-resolution PNG
for GitHub, which does not render PDF assets inline reliably.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import numpy as np
import yaml


MANUAL_DIR = Path(__file__).resolve().parent
ROOT = MANUAL_DIR.parents[1]
OUT = MANUAL_DIR / "figures"
REGISTRY_PATH = ROOT / "perovskite-sim/reproducibility/config_benchmark_matrix.yaml"
CBO_PATH = (
    ROOT
    / "perovskite-sim/outputs/interface-cbo/scan-fermi-edge-qf-grid-40-50-60.json"
)

OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 9.2,
        "axes.titlesize": 10.2,
        "axes.labelsize": 9.2,
        "legend.fontsize": 8.2,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
    }
)

# Color-blind-safe, print-friendly semantic palette.
INK = "#202A35"
MUTED = "#66727F"
GRID = "#D9DEE5"
PALE = "#F5F7F9"
BLUE = "#2C6EAA"
TEAL = "#16827C"
GREEN = "#4D7D55"
GOLD = "#B07A18"
RUST = "#A95632"
VIOLET = "#765A9E"
RED = "#A33D3D"


def _save(fig: Figure, name: str) -> None:
    """Write one publication-quality PDF and its web-renderable PNG peer."""
    common = {
        "bbox_inches": "tight",
        "pad_inches": 0.04,
    }
    fig.savefig(
        OUT / f"{name}.pdf",
        format="pdf",
        **common,
        metadata={
            "Title": name.replace("_", " ").title(),
            "Author": "SolarLab Project",
            "Subject": "SolarLab technical manual figure",
        },
    )
    fig.savefig(
        OUT / f"{name}.png",
        format="png",
        dpi=180,
        facecolor="white",
        **common,
    )
    plt.close(fig)


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected a mapping in {path}")
    return data


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected an object in {path}")
    return data


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _panel_label(
    ax: Axes,
    label: str,
    *,
    x: float = -0.08,
    y: float = 1.04,
) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=INK,
        fontweight="bold",
        fontsize=10.2,
    )


def _box(
    ax: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
    *,
    face: str = "white",
    edge: str = INK,
    color: str = INK,
    fontsize: float = 8.8,
    linewidth: float = 0.9,
) -> None:
    ax.add_patch(
        Rectangle(
            (x, y),
            width,
            height,
            transform=ax.transAxes,
            facecolor=face,
            edgecolor=edge,
            linewidth=linewidth,
        )
    )
    ax.text(
        x + width / 2,
        y + height / 2,
        label,
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=color,
        fontsize=fontsize,
        linespacing=1.15,
    )


def _arrow(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = INK,
    style: str = "-",
    linewidth: float = 1.1,
    head_size: float = 9.0,
    arrowstyle: str = "->",
) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        arrowprops={
            "arrowstyle": arrowstyle,
            "color": color,
            "lw": linewidth,
            "linestyle": style,
            "mutation_scale": head_size,
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )


def architecture_flow() -> None:
    fig, ax = plt.subplots(figsize=(7.8, 3.0), layout="constrained")
    ax.set_axis_off()

    _box(
        ax,
        0.02,
        0.66,
        0.16,
        0.18,
        "YAML | Python API\nbrowser editor",
        face="#EAF2F8",
        fontsize=9.0,
    )
    _box(
        ax,
        0.24,
        0.66,
        0.18,
        0.18,
        "Validated schema\nMaterialParams\nDeviceStack",
        face="#EAF3EE",
        fontsize=8.6,
    )
    _box(
        ax,
        0.48,
        0.66,
        0.18,
        0.18,
        "Experiment driver\nsettings +\ncapability check",
        face="#F6F1E6",
        fontsize=8.6,
    )
    _box(
        ax,
        0.24,
        0.27,
        0.18,
        0.18,
        "Shared construction\ngrid | optics\nMaterialArrays",
        face="#F0EDF5",
        fontsize=8.6,
    )
    _box(
        ax,
        0.48,
        0.27,
        0.18,
        0.18,
        "Explicit solver path\ntransient | SS\nQF | frequency | 2D",
        face="#F8ECE8",
        fontsize=8.4,
    )
    _box(
        ax,
        0.72,
        0.27,
        0.20,
        0.18,
        "Result dataclass\nfiles | API/SSE | plots",
        face="#E8F3F2",
        fontsize=8.8,
    )

    _arrow(ax, (0.18, 0.75), (0.24, 0.75))
    _arrow(ax, (0.42, 0.75), (0.48, 0.75))
    _arrow(ax, (0.57, 0.66), (0.39, 0.45))
    _arrow(ax, (0.42, 0.36), (0.48, 0.36))
    _arrow(ax, (0.57, 0.66), (0.57, 0.45))
    _arrow(ax, (0.66, 0.36), (0.72, 0.36))

    ax.text(
        0.50,
        0.08,
        "No implicit solver substitution. Unsupported physics stops before the numerical solve.",
        transform=ax.transAxes,
        ha="center",
        color=INK,
        fontsize=8.8,
    )
    _save(fig, "architecture_flow")


def device_contact_boundary() -> None:
    fig = plt.figure(figsize=(8.0, 3.45), layout="constrained")
    grid = fig.add_gridspec(1, 2, width_ratios=(1.35, 1.0))
    ax_stack = fig.add_subplot(grid[0, 0])
    ax_bc = fig.add_subplot(grid[0, 1])
    for ax in (ax_stack, ax_bc):
        ax.set_axis_off()

    _panel_label(ax_stack, "(a)")
    ax_stack.set_title("Electrical coordinate and shipped layer order", loc="left", pad=8)

    x0, y0, h = 0.04, 0.31, 0.34
    widths = [0.12, 0.17, 0.34, 0.17, 0.12]
    labels = ["left\ncontact", "HTL", "absorber", "ETL", "right\ncontact"]
    colors = ["#D4D8DC", "#DDEAF5", "#F7E8C9", "#DDEEE3", "#D4D8DC"]
    xpos = x0
    for width, label, face in zip(widths, labels, colors):
        ax_stack.add_patch(
            Rectangle(
                (xpos, y0),
                width,
                h,
                transform=ax_stack.transAxes,
                facecolor=face,
                edgecolor=INK,
                linewidth=0.9,
            )
        )
        ax_stack.text(
            xpos + width / 2,
            y0 + h / 2,
            label,
            transform=ax_stack.transAxes,
            ha="center",
            va="center",
            fontsize=8.4,
            color=INK,
        )
        xpos += width

    ax_stack.annotate(
        "AM1.5G",
        xy=(0.18, y0 + h + 0.01),
        xytext=(0.18, 0.86),
        xycoords=ax_stack.transAxes,
        textcoords=ax_stack.transAxes,
        ha="center",
        color=GOLD,
        fontsize=9.0,
        arrowprops={"arrowstyle": "-|>", "color": GOLD, "lw": 1.5},
    )
    ax_stack.annotate(
        "",
        xy=(0.96, 0.17),
        xytext=(0.04, 0.17),
        xycoords=ax_stack.transAxes,
        textcoords=ax_stack.transAxes,
        arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.1},
    )
    ax_stack.text(0.04, 0.08, r"$x=0$", transform=ax_stack.transAxes, ha="center")
    ax_stack.text(0.96, 0.08, r"$x=L$", transform=ax_stack.transAxes, ha="center")
    ax_stack.text(0.50, 0.08, "electrical coordinate", transform=ax_stack.transAxes, ha="center", color=MUTED)
    ax_stack.text(
        0.50,
        0.72,
        "preset identifier: nip_MAPbI3",
        transform=ax_stack.transAxes,
        ha="center",
        color=MUTED,
        fontsize=8.2,
    )
    _panel_label(ax_bc, "(b)")
    ax_bc.set_title("Electrostatic source and carrier exchange", loc="left", pad=8)
    _box(ax_bc, 0.02, 0.70, 0.28, 0.16, "semiconductor\nwork functions", face="#EAF3EE", fontsize=8.3)
    _box(ax_bc, 0.36, 0.70, 0.28, 0.16, "metal\nwork functions", face="#EAF2F8", fontsize=8.3)
    _box(ax_bc, 0.70, 0.70, 0.28, 0.16, "legacy manual\nmagnitude", face="#F0F1F2", color=MUTED, fontsize=8.3)
    _arrow(ax_bc, (0.16, 0.70), (0.38, 0.55), color=GREEN)
    _arrow(ax_bc, (0.50, 0.70), (0.50, 0.55), color=BLUE)
    _arrow(ax_bc, (0.84, 0.70), (0.62, 0.55), color=MUTED, style="--")
    _box(
        ax_bc,
        0.21,
        0.41,
        0.58,
        0.14,
        r"signed $V_{bi}^{bc}$ and $s=\mathrm{sign}(V_{bi}^{bc})$",
        face="#F8F2E7",
        edge=GOLD,
        fontsize=8.7,
    )
    _box(
        ax_bc,
        0.13,
        0.20,
        0.74,
        0.13,
        r"$\phi(0)=0$     $\phi(L)=V_{bi}^{bc}-sV_{app}$",
        face="white",
        edge=INK,
        fontsize=9.4,
    )
    ax_bc.text(
        0.50,
        0.07,
        "Carrier boundary: Dirichlet pin or finite-rate Robin S",
        transform=ax_bc.transAxes,
        ha="center",
        color=TEAL,
        fontweight="bold",
        fontsize=8.8,
    )
    _save(fig, "device_contact_boundary")


def _shade_layers(ax: Axes) -> None:
    spans = [(0.0, 1.0, "HTL", "#EAF2F8"), (1.0, 2.0, "absorber", "#FBF0D8"), (2.0, 3.0, "ETL", "#E8F2EA")]
    for left, right, label, face in spans:
        ax.axvspan(left, right, color=face, zorder=0)
        ax.text((left + right) / 2, -7.12, label, ha="center", va="bottom", fontsize=8.0, color=MUTED)
    ax.axvline(1.0, color=GRID, lw=0.8)
    ax.axvline(2.0, color=GRID, lw=0.8)


def _schematic_potential(x: np.ndarray, total_drop: float) -> np.ndarray:
    """Smooth built-in potential used only to show band bending."""
    u = np.clip(np.asarray(x, dtype=float) / 3.0, 0.0, 1.0)
    return total_drop * 0.5 * (1.0 - np.cos(np.pi * u))


def _plot_bent_bands(
    ax: Axes,
    ec_reference: tuple[float, float, float],
    ev_reference: tuple[float, float, float],
    *,
    potential_drop: float,
) -> None:
    """Plot continuous electrostatic bending plus abrupt material offsets."""
    for index, (ec_i, ev_i) in enumerate(zip(ec_reference, ev_reference)):
        xx = np.linspace(float(index), float(index + 1), 81)
        shift = _schematic_potential(xx, potential_drop)
        ax.plot(
            xx,
            ec_i - shift,
            color=BLUE,
            lw=1.8,
            label=r"$E_c/q=-(\phi+\chi)$" if index == 0 else None,
        )
        ax.plot(
            xx,
            ev_i - shift,
            color=RUST,
            lw=1.8,
            label=r"$E_v/q=E_c/q-E_g$" if index == 0 else None,
        )
        if index < 2:
            boundary = float(index + 1)
            bend = float(_schematic_potential(np.array([boundary]), potential_drop)[0])
            ax.plot(
                [boundary, boundary],
                [ec_i - bend, ec_reference[index + 1] - bend],
                color=BLUE,
                lw=1.2,
            )
            ax.plot(
                [boundary, boundary],
                [ev_i - bend, ev_reference[index + 1] - bend],
                color=RUST,
                lw=1.2,
            )


def band_interface_convention() -> None:
    fig = plt.figure(figsize=(8.0, 4.85), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(2.35, 1.35))
    ax_eq = fig.add_subplot(grid[0, 0])
    ax_op = fig.add_subplot(grid[0, 1])
    ax_default = fig.add_subplot(grid[1, 0])
    ax_qf = fig.add_subplot(grid[1, 1])

    ec = (-2.25, -3.90, -4.05)
    ev = (-5.25, -5.50, -7.05)
    for ax in (ax_eq, ax_op):
        _shade_layers(ax)
        ax.set_xlim(0.0, 3.0)
        ax.set_ylim(-7.90, -1.85)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel("Energy (eV, schematic)")

    _panel_label(ax_eq, "(a)")
    ax_eq.set_title("Dark equilibrium with band bending", loc="left")
    _plot_bent_bands(ax_eq, ec, ev, potential_drop=0.62)
    ax_eq.axhline(-4.90, color=INK, lw=1.3, label=r"single $E_F$")
    ax_eq.text(
        1.10,
        -3.30,
        r"$\Delta E_c=E_{c,R}-E_{c,L}$",
        color=BLUE,
        fontsize=8.2,
    )
    ax_eq.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 0.98),
        frameon=False,
        ncol=1,
        handlelength=1.6,
    )

    _panel_label(ax_op, "(b)")
    ax_op.set_title("Illuminated operation with band bending", loc="left")
    _plot_bent_bands(ax_op, ec, ev, potential_drop=0.42)
    x_qf = np.array([0.15, 2.85])
    ax_op.plot(x_qf, [-4.46, -4.61], color=TEAL, lw=1.4, ls="--", label=r"$E_{Fn}$")
    ax_op.plot(x_qf, [-5.01, -5.18], color=VIOLET, lw=1.4, ls="--", label=r"$E_{Fp}$")
    ax_op.text(1.45, -3.65, r"$e^{-}$ flow", ha="center", va="center", color=TEAL, fontsize=8.2)
    ax_op.annotate(
        "",
        xy=(1.93, -3.65),
        xytext=(1.65, -3.65),
        arrowprops={"arrowstyle": "->", "color": TEAL, "lw": 1.05, "mutation_scale": 9.0},
    )
    ax_op.text(0.78, -4.79, r"$h^{+}$ flow", ha="center", va="center", color=VIOLET, fontsize=8.2)
    ax_op.annotate(
        "",
        xy=(0.28, -4.79),
        xytext=(0.62, -4.79),
        arrowprops={"arrowstyle": "->", "color": VIOLET, "lw": 1.05, "mutation_scale": 9.0},
    )
    ax_op.legend(loc="upper right", frameon=False, ncol=1)

    for ax in (ax_default, ax_qf):
        ax.set_axis_off()

    _panel_label(ax_default, "(c)")
    ax_default.set_title("Abrupt-interface transport choices", loc="left", pad=5)
    ax_default.text(
        0.50,
        0.82,
        "Default SG face",
        transform=ax_default.transAxes,
        ha="center",
        va="center",
        color=MUTED,
        fontsize=8.2,
        fontweight="bold",
    )
    _box(ax_default, 0.01, 0.33, 0.20, 0.36, "bulk node L", face="#F2F4F6", fontsize=8.2)
    _box(
        ax_default,
        0.30,
        0.33,
        0.40,
        0.36,
        "SG face\nnet $J_n, J_p$\nTE cap (optional)",
        face="#EAF2F8",
        edge=BLUE,
        fontsize=8.0,
    )
    _box(ax_default, 0.79, 0.33, 0.20, 0.36, "bulk node R", face="#F2F4F6", fontsize=8.2)
    _arrow(ax_default, (0.23, 0.51), (0.28, 0.51), color=BLUE, linewidth=1.0, head_size=7.0, arrowstyle="<->")
    _arrow(ax_default, (0.72, 0.51), (0.77, 0.51), color=BLUE, linewidth=1.0, head_size=7.0, arrowstyle="<->")
    ax_default.text(
        0.50,
        0.07,
        "bidirectional drift-diffusion face flux",
        transform=ax_default.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.7,
        color=INK,
    )

    ax_qf.text(
        0.50,
        0.82,
        "Explicit QF interface (opt-in)",
        transform=ax_qf.transAxes,
        ha="center",
        va="center",
        color=MUTED,
        fontsize=8.2,
        fontweight="bold",
    )
    _box(ax_qf, 0.005, 0.35, 0.14, 0.34, "bulk\nL", face="#F2F4F6", fontsize=7.4)
    _box(
        ax_qf,
        0.22,
        0.35,
        0.22,
        0.34,
        "$n_{I,L}$\n$p_{I,L}$",
        face="#EAF3EE",
        edge=TEAL,
        fontsize=8.6,
    )
    _box(
        ax_qf,
        0.56,
        0.35,
        0.22,
        0.34,
        "$n_{I,R}$\n$p_{I,R}$",
        face="#F7ECE8",
        edge=RUST,
        fontsize=8.6,
    )
    _box(ax_qf, 0.855, 0.35, 0.14, 0.34, "bulk\nR", face="#F2F4F6", fontsize=7.4)
    _arrow(ax_qf, (0.16, 0.52), (0.205, 0.52), color=INK, linewidth=0.9, head_size=6.5, arrowstyle="<->")
    _arrow(ax_qf, (0.455, 0.52), (0.545, 0.52), color=INK, linewidth=1.0, head_size=7.0, arrowstyle="<->")
    _arrow(ax_qf, (0.795, 0.52), (0.84, 0.52), color=INK, linewidth=0.9, head_size=6.5, arrowstyle="<->")
    ax_qf.text(
        0.50,
        0.05,
        r"$F_{\mathrm{bulk}}+F_{\mathrm{cross}}+F_{\mathrm{SRH}}=0$" "\nzero thickness; shared trap occupancy",
        transform=ax_qf.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.4,
        color=INK,
        linespacing=1.15,
    )
    _save(fig, "band_interface_convention")


def solver_topology() -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.65), layout="constrained")
    ax.set_axis_off()

    columns = [0.02, 0.20, 0.43, 0.70]
    widths = [0.16, 0.21, 0.25, 0.27]
    headers = ["driver", "unknowns", "discrete model", "solve and certificate"]
    for x, width, header in zip(columns, widths, headers):
        ax.add_patch(Rectangle((x, 0.88), width, 0.08, transform=ax.transAxes, facecolor="#E9EDF1", edgecolor=GRID, lw=0.8))
        ax.text(x + width / 2, 0.92, header, transform=ax.transAxes, ha="center", va="center", fontweight="bold", color=INK)

    rows = [
        (
            "transient",
            r"$[n,p,P^+,(P^-)]$",
            "SG carrier/ion continuity\nPoisson at each RHS call",
            "Radau IIA\nhistory-preserving state",
            BLUE,
        ),
        (
            "steady_state",
            r"$[\ln n,\ln p]$",
            "1D density RHS\nfixed ion profile",
            "direct Newton\nDC residual checks",
            GREEN,
        ),
        (
            "quasi_fermi",
            r"$[\Delta\varphi_n,\Delta\varphi_p]$",
            "eliminated Poisson response\noptional local interface solve",
            "Newton continuation\ncell, current, Poisson certificates",
            RUST,
        ),
        (
            "QF frequency",
            "complex small signal",
            "linearization about a\ncertified QF operating point",
            "refined complex solve\nall-face admittance continuity",
            VIOLET,
        ),
        (
            "2D",
            r"$[n(y,x),p(y,x)]$",
            "2D SG + sparse Poisson\nstatic ion background",
            "Radau-LU\nlateral/vertical diagnostics",
            TEAL,
        ),
    ]
    y_top = 0.83
    row_h = 0.145
    for index, (driver, unknowns, model, solve, color) in enumerate(rows):
        y = y_top - index * row_h
        face = "#FBFCFD" if index % 2 == 0 else "#F5F7F9"
        for x, width in zip(columns, widths):
            ax.add_patch(Rectangle((x, y - 0.105), width, 0.115, transform=ax.transAxes, facecolor=face, edgecolor=GRID, lw=0.7))
        ax.add_patch(Rectangle((columns[0], y - 0.105), 0.008, 0.115, transform=ax.transAxes, facecolor=color, edgecolor=color))
        values = [driver, unknowns, model, solve]
        for x, width, value in zip(columns, widths, values):
            ax.text(
                x + width / 2,
                y - 0.047,
                value,
                transform=ax.transAxes,
                ha="center",
                va="center",
                color=INK,
                fontsize=8.7,
                linespacing=1.17,
            )

    ax.text(
        0.50,
        0.045,
        "Driver choice fixes the unknowns, supported physics,\n"
        "and required certification checks.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=INK,
        fontsize=8.8,
        linespacing=1.2,
    )
    _save(fig, "solver_topology")


def csi_qf_convergence(registry: dict[str, Any]) -> None:
    benchmarks = registry.get("benchmarks")
    _require(isinstance(benchmarks, dict), "registry is missing benchmarks")
    jv = benchmarks.get("csi-qf-internal-validation")
    cv = benchmarks.get("csi-qf-frequency-domain-cv")
    _require(isinstance(jv, dict) and jv.get("status") == "pass", "c-Si QF J-V evidence is missing or not passed")
    _require(isinstance(cv, dict) and cv.get("status") == "pass", "c-Si QF C-V evidence is missing or not passed")

    jv_grids = jv["observed"]["grids"]
    cv_grids = cv["observed"]["grids"]
    requested = np.asarray([row["N_grid"] for row in jv_grids], dtype=float)
    _require(np.array_equal(requested, np.array([200.0, 300.0, 400.0])), "unexpected c-Si J-V grid ladder")
    _require([row["N_grid"] for row in cv_grids] == [200, 300, 400], "unexpected c-Si C-V grid ladder")

    metric_fields = [
        ("Jsc_A_m2", r"$J_{sc}$", BLUE),
        ("Voc_V", r"$V_{oc}$", TEAL),
        ("FF", "FF", GOLD),
        ("PCE_percent", "PCE", RUST),
    ]

    fig = plt.figure(figsize=(8.1, 4.10), layout="constrained")
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.0, 1.05),
        height_ratios=(1.0, 0.14),
    )
    ax_metrics = fig.add_subplot(grid[0, 0])
    ax_cv = fig.add_subplot(grid[0, 1])
    ax_metrics_note = fig.add_subplot(grid[1, 0])
    ax_cv_note = fig.add_subplot(grid[1, 1])
    ax_metrics_note.set_axis_off()
    ax_cv_note.set_axis_off()

    _panel_label(ax_metrics, "(a)")
    for field, label, color in metric_fields:
        values = np.asarray([row[field] for row in jv_grids], dtype=float)
        relative = 100.0 * (values / values[-1] - 1.0)
        ax_metrics.plot(requested, relative, marker="o", ms=4.5, color=color, label=label)
    ax_metrics.axhline(0.0, color=INK, lw=0.8)
    ax_metrics.set_xlabel(r"Requested grid intervals $N$")
    ax_metrics.set_ylabel("Difference from N=400 (%)")
    ax_metrics.set_title("QF J-V metric contraction", loc="left")
    ax_metrics.set_xticks(requested)
    ax_metrics.grid(True, color=GRID, lw=0.7)

    adjacent = jv["observed"]["adjacent_curve_changes"]
    fine_jsc = float(jv_grids[-1]["Jsc_A_m2"])
    normalized_changes = [100.0 * float(row["max_abs_current_A_m2"]) / fine_jsc for row in adjacent]
    handles, labels = ax_metrics.get_legend_handles_labels()
    ax_metrics_note.legend(
        handles,
        labels,
        frameon=False,
        ncol=4,
        loc="upper center",
        fontsize=7.8,
        handlelength=1.6,
        columnspacing=1.1,
    )
    ax_metrics_note.text(
        0.5,
        0.08,
        r"Adjacent max $|\Delta J|/J_{sc}$: "
        f"{normalized_changes[0]:.3f}% (200 to 300); "
        f"{normalized_changes[1]:.3f}% (300 to 400)",
        transform=ax_metrics_note.transAxes,
        ha="center",
        va="bottom",
        color=MUTED,
        fontsize=7.6,
    )

    _panel_label(ax_cv, "(b)")
    biases = np.asarray(cv["protocol"]["biases_V"], dtype=float)
    colors = [BLUE, TEAL, RUST]
    for row, color in zip(cv_grids, colors):
        capacitance_mf = 1.0e3 * np.asarray(row["capacitance_100k_F_m2"], dtype=float)
        ax_cv.plot(biases, capacitance_mf, marker="o", ms=4.0, color=color, label=f"N={row['N_grid']}")
    ax_cv.set_xlabel(r"DC bias $V_{app}$ (V)")
    ax_cv.set_ylabel(r"$C$ at 100 kHz (mF m$^{-2}$)")
    ax_cv.set_title("QF frequency-domain C-V", loc="left")
    ax_cv.grid(True, color=GRID, lw=0.7)
    ax_cv.legend(frameon=False, loc="upper left")
    finest_change = 100.0 * float(cv["observed"]["adjacent_max_capacitance_changes"][-1])
    ax_cv_note.text(
        0.5,
        0.70,
        f"N=300 to 400: max |dC|/C = {finest_change:.3f}%",
        transform=ax_cv_note.transAxes,
        ha="center",
        va="center",
        color=MUTED,
        fontsize=7.8,
    )
    ax_cv_note.text(
        0.5,
        0.18,
        "Internal numerical evidence; not an external device fit.",
        transform=ax_cv_note.transAxes,
        ha="center",
        va="center",
        color=MUTED,
        fontsize=7.6,
    )
    _save(fig, "csi_qf_convergence")


def cbo_interface_validation(data: dict[str, Any]) -> None:
    _require(data.get("schema") == "solarlab.interface_cbo_scan", "unexpected CBO evidence schema")
    _require(data.get("complete") is True, "CBO evidence is incomplete")
    _require(data.get("numerical_certified") is True, "CBO numerical certificate is not passed")
    _require(data.get("certified") is False, "CBO top-level status changed; review the manual claim")

    runs = data.get("grid_runs")
    _require(isinstance(runs, list) and len(runs) == 3, "CBO evidence must contain three grid runs")
    runs = sorted(runs, key=lambda row: int(row["settings"]["N_grid"]))
    grid_counts = [int(row["settings"]["N_grid"]) for row in runs]
    _require(grid_counts == [40, 50, 60], "unexpected CBO grid ladder")

    external = data["external_validation"]["certificates"][0]
    convergence = data["grid_convergence"]
    envelope_mev = 1.0e3 * float(convergence["envelope_width_eV"])
    envelope_limit_mev = 1.0e3 * float(convergence["maximum_envelope_width_eV"])
    external_error = float(external["max_normalized_error"])
    external_limit = float(external["maximum_normalized_error"])

    fig = plt.figure(figsize=(7.9, 5.55), layout="constrained")
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=(1.08, 0.92),
        width_ratios=(1.18, 1.0),
    )
    ax_trace = fig.add_subplot(grid[0, :])
    ax_interval = fig.add_subplot(grid[1, 0])
    ax_status = fig.add_subplot(grid[1, 1])

    _panel_label(ax_trace, "(a)", x=-0.10, y=1.06)
    colors = [BLUE, TEAL, RUST]
    for run, color in zip(runs, colors):
        trace = run["short_circuit_trace"]
        delta = np.asarray([point["delta_ec_eV"] for point in trace], dtype=float)
        current = np.asarray([point["current_A_m2"] for point in trace], dtype=float)
        ax_trace.plot(delta, current / current[0], marker="o", ms=2.8, color=color, label=f"SolarLab N={run['settings']['N_grid']}")
    ax_trace.plot(
        np.asarray(external["matched_delta_ec_eV"], dtype=float),
        np.asarray(external["reference_normalized"], dtype=float),
        marker="s",
        ms=4.0,
        color=INK,
        ls="--",
        label="SCAPS reference",
    )
    ax_trace.axhline(0.99, color=GOLD, lw=1.0, ls=":", label="1% Jsc drop")
    ax_trace.set_xlim(-0.01, 0.51)
    ax_trace.set_ylim(-0.02, 1.05)
    ax_trace.set_xlabel(r"Conduction-band offset $\Delta E_c$ (eV)")
    ax_trace.set_ylabel(r"$J_{sc}(\Delta E_c)/J_{sc}(0)$")
    ax_trace.set_title("Physical-interface CBO response", loc="left")
    ax_trace.grid(True, color=GRID, lw=0.7)
    ax_trace.legend(
        frameon=False,
        loc="lower left",
        ncol=3,
        fontsize=7.6,
        handlelength=1.8,
        columnspacing=1.2,
    )

    _panel_label(ax_interval, "(b)", x=-0.17, y=1.06)
    intervals = np.asarray(convergence["critical_intervals_eV"], dtype=float)
    midpoints = intervals.mean(axis=1)
    half_widths = np.diff(intervals, axis=1).ravel() / 2.0
    y = np.arange(len(grid_counts))
    for yi, midpoint, half_width, color in zip(y, midpoints, half_widths, colors):
        ax_interval.errorbar(midpoint, yi, xerr=half_width, fmt="o", ms=5.0, capsize=4, color=color)
    ax_interval.axvspan(
        float(convergence["envelope_lower_eV"]),
        float(convergence["envelope_upper_eV"]),
        color=GOLD,
        alpha=0.18,
        label="three-grid union",
    )
    ax_interval.set_yticks(y, [f"N={value}" for value in grid_counts])
    ax_interval.invert_yaxis()
    ax_interval.set_xlim(0.378, 0.390)
    ax_interval.set_xlabel(r"1% $J_{sc}$ critical interval (eV)")
    ax_interval.set_title("Grid-ladder onset", loc="left")
    ax_interval.grid(axis="x", color=GRID, lw=0.7)
    ax_interval.text(
        0.97,
        0.08,
        f"union = {envelope_mev:.3f} meV\nlimit = {envelope_limit_mev:.1f} meV",
        transform=ax_interval.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.0,
        color=INK,
    )

    _panel_label(ax_status, "(c)", x=-0.16, y=1.06)
    ratios = [envelope_mev / envelope_limit_mev, external_error / external_limit]
    labels = ["grid\nenvelope", "SCAPS\nshape error"]
    status_colors = [GREEN, RED]
    y_status = np.arange(2)
    for yi, ratio, color in zip(y_status, ratios, status_colors):
        ax_status.hlines(yi, 0.38, ratio, color=color, lw=4.0, alpha=0.72)
    ax_status.scatter(ratios, y_status, c=status_colors, s=50, zorder=3)
    ax_status.axvline(1.0, color=INK, lw=1.0, ls="--")
    ax_status.set_xscale("log")
    ax_status.set_xlim(0.35, 15.0)
    ax_status.set_yticks(y_status, labels)
    ax_status.invert_yaxis()
    ax_status.set_xlabel("observed / limit")
    ax_status.set_title("Certification gates", loc="left")
    ax_status.grid(axis="x", color=GRID, lw=0.7)
    for yi, ratio, status in zip(y_status, ratios, ("pass", "fail")):
        offset_y = -15 if yi == 0 else 15
        ax_status.annotate(
            f"{ratio:.2f}\n{status}",
            xy=(ratio, yi),
            xytext=(0, offset_y),
            textcoords="offset points",
            va="top" if yi == 0 else "bottom",
            ha="center" if ratio < 2.0 else "right",
            color=INK,
            fontsize=8.0,
            linespacing=1.0,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8},
        )
    _save(fig, "cbo_interface_validation")


def twod_scope() -> None:
    fig = plt.figure(figsize=(8.0, 4.0), layout="constrained")
    grid = fig.add_gridspec(1, 2, width_ratios=(1.0, 1.18))
    ax_domain = fig.add_subplot(grid[0, 0])
    ax_scope = fig.add_subplot(grid[0, 1])
    for ax in (ax_domain, ax_scope):
        ax.set_axis_off()

    _panel_label(ax_domain, "(a)")
    ax_domain.set_title("Registered lateral-uniform parity domain", loc="left", pad=8)
    content_top = 0.88
    content_bottom = 0.26
    stack_x = 0.08
    stack_w = 0.18
    content_height = content_top - content_bottom
    heights = [content_height * ratio / 70.0 for ratio in (18.0, 34.0, 18.0)]
    labels = ["HTL", "absorber", "ETL"]
    faces = ["#DDEAF5", "#F7E8C9", "#DDEEE3"]
    ypos = content_top
    for height, label, face in zip(heights, labels, faces):
        ypos -= height
        ax_domain.add_patch(Rectangle((stack_x, ypos), stack_w, height, transform=ax_domain.transAxes, facecolor=face, edgecolor=INK, lw=0.8))
        ax_domain.text(stack_x + stack_w / 2, ypos + height / 2, label, transform=ax_domain.transAxes, ha="center", va="center", fontsize=8.0)
    ax_domain.text(stack_x + stack_w / 2, 0.16, "1D y stack", transform=ax_domain.transAxes, ha="center", color=MUTED)

    transfer_y = (content_top + content_bottom) / 2.0
    _arrow(ax_domain, (0.28, transfer_y), (0.44, transfer_y), color=TEAL)
    ax_domain.text(
        0.36,
        transfer_y + 0.055,
        "extrude in x",
        transform=ax_domain.transAxes,
        ha="center",
        va="bottom",
        color=TEAL,
        fontsize=7.8,
    )

    mesh_left, mesh_right = 0.46, 0.94
    mesh_bottom, mesh_top = content_bottom, content_top
    for value in np.linspace(mesh_left, mesh_right, 7):
        ax_domain.plot([value, value], [mesh_bottom, mesh_top], transform=ax_domain.transAxes, color=GRID, lw=0.7)
    for value in np.linspace(mesh_bottom, mesh_top, 10):
        ax_domain.plot([mesh_left, mesh_right], [value, value], transform=ax_domain.transAxes, color=GRID, lw=0.7)
    ax_domain.add_patch(Rectangle((mesh_left, mesh_bottom), mesh_right - mesh_left, mesh_top - mesh_bottom, transform=ax_domain.transAxes, fill=False, edgecolor=INK, lw=1.0))
    ax_domain.text((mesh_left + mesh_right) / 2, 0.16, r"2D $(y,x)$ mesh", transform=ax_domain.transAxes, ha="center", color=MUTED)
    ax_domain.text(
        0.50,
        0.04,
        "nip_MAPbI3_uniform | frozen ions | matched vertical grid\n"
        "periodic lateral boundary condition",
        transform=ax_domain.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.0,
        color=INK,
    )

    _panel_label(ax_scope, "(b)")
    ax_scope.set_title("Physics included in the comparison", loc="left", pad=8)
    rows = [
        ("electrons and holes", "active", GREEN),
        ("2D sparse Poisson + band offsets", "active", GREEN),
        ("configured ion charge", "static background", TEAL),
        ("lateral grain-boundary lifetime field", "available", GREEN),
        ("mobile-ion dynamics during 2D J-V", "not included", RUST),
        ("interface SRH / defects /\nphysical QF boundary", "not connected", RED),
    ]
    row_box_h = 0.09
    row_pitch = (content_top - content_bottom - row_box_h) / (len(rows) - 1)
    for index, (model, status, color) in enumerate(rows):
        row_top = content_top - index * row_pitch
        row_bottom = row_top - row_box_h
        row_center = (row_top + row_bottom) / 2.0
        face = "#FBFCFD" if index % 2 == 0 else "#F4F6F8"
        ax_scope.add_patch(Rectangle((0.02, row_bottom), 0.96, row_box_h, transform=ax_scope.transAxes, facecolor=face, edgecolor=GRID, lw=0.6))
        ax_scope.add_patch(Rectangle((0.02, row_bottom), 0.012, row_box_h, transform=ax_scope.transAxes, facecolor=color, edgecolor=color))
        ax_scope.text(
            0.06,
            row_center,
            model,
            transform=ax_scope.transAxes,
            ha="left",
            va="center",
            fontsize=8.1,
            color=INK,
            linespacing=1.05,
        )
        ax_scope.text(
            0.95,
            row_center,
            status,
            transform=ax_scope.transAxes,
            ha="right",
            va="center",
            fontsize=8.1,
            color=color,
            fontweight="bold",
        )
    ax_scope.text(
        0.50,
        0.045,
        "Quantitative 1D/2D parity is meaningful only when both drivers\n"
        "solve the same configured physics.",
        transform=ax_scope.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.2,
        color=INK,
    )
    _save(fig, "twod_scope")


def main() -> None:
    registry = _load_yaml(REGISTRY_PATH)
    cbo = _load_json(CBO_PATH)

    architecture_flow()
    device_contact_boundary()
    band_interface_convention()
    solver_topology()
    csi_qf_convergence(registry)
    cbo_interface_validation(cbo)
    twod_scope()


if __name__ == "__main__":
    main()
