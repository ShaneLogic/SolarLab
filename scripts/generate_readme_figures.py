"""Generate README diagram PNGs.

These figures are intentionally simple, text-heavy documentation assets.  Keep
them synchronized with README physics/API descriptions when solver features
change.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "perovskite-sim" / "docs" / "images"
FONT = "DejaVu Sans"


def setup(width: float = 16, height: float = 10) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(width, height), dpi=120)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def rounded(ax: plt.Axes, xy, w, h, fc="#ffffff", ec="#d5d5d5", lw=1.5, r=0.02):
    patch = FancyBboxPatch(
        xy, w, h,
        boxstyle=f"round,pad=0.012,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw,
    )
    ax.add_patch(patch)
    return patch


def label(ax: plt.Axes, x, y, text, size=14, weight="normal", color="#222", ha="left", va="center"):
    ax.text(
        x, y, text,
        fontsize=size, fontweight=weight, color=color, ha=ha, va=va,
        family=FONT,
    )


def pill(ax: plt.Axes, x, y, text, color):
    rounded(ax, (x, y - 0.017), 0.09, 0.034, fc=color, ec=color, lw=0, r=0.008)
    label(ax, x + 0.045, y, text, size=9, weight="bold", color="white", ha="center")


def generate_device_structure() -> None:
    fig, ax = setup(16, 11)
    label(ax, 0.03, 0.96, "Device Structure and Solver Views", size=24, weight="bold")
    label(ax, 0.50, 0.89, "AM1.5G illumination", size=15, weight="bold", color="#d99a00", ha="center")
    ax.arrow(0.50, 0.865, 0, -0.035, width=0.004, head_width=0.025, head_length=0.025, color="#e0a000")

    x0, y0, w, h = 0.23, 0.14, 0.54, 0.70
    layers = [
        ("Left / top contact", 0.12, "#b8b8b8", "phi=0 | ohmic or Robin S_left"),
        ("HTL - hole transport", 0.18, "#e7f0ff", "high N_A, blocks electrons"),
        ("Absorber - perovskite", 0.34, "#fff2d8", "G(x), SRH/rad/Auger, mobile ions; GBs reduce tau"),
        ("ETL - electron transport", 0.20, "#e6f4e8", "high N_D, blocks holes"),
        ("Right / bottom contact", 0.16, "#b8b8b8", "phi=V_bi - V_app | ohmic or Robin S_right"),
    ]
    yy = y0 + h
    for title, frac, color, subtitle in layers:
        lh = h * frac
        yy -= lh
        ax.add_patch(Rectangle((x0, yy), w, lh, facecolor=color, edgecolor="#666", linewidth=1.4))
        label(ax, x0 + 0.025, yy + lh * 0.62, title, size=15, weight="bold")
        label(ax, x0 + 0.025, yy + lh * 0.34, subtitle, size=11, color="#4f4f4f")

    ax.plot([x0, x0 + w], [y0 - 0.025, y0 - 0.025], color="#555", lw=2)
    ax.arrow(x0 + w - 0.01, y0 - 0.025, 0.001, 0, head_width=0.018, head_length=0.018, color="#555")
    label(ax, x0, y0 - 0.055, "1D axis: x=0", size=11, color="#555")
    label(ax, x0 + w, y0 - 0.055, "x=L", size=11, color="#555", ha="right")

    rounded(ax, (0.06, 0.42), 0.12, 0.20, fc="#f8f8f8")
    label(ax, 0.12, 0.59, "1D default", size=13, weight="bold", ha="center")
    label(ax, 0.12, 0.54, "tanh grid", size=10, ha="center", color="#555")
    label(ax, 0.12, 0.50, "fast sweeps", size=10, ha="center", color="#555")
    label(ax, 0.12, 0.46, "most experiments", size=10, ha="center", color="#555")

    rounded(ax, (0.82, 0.40), 0.13, 0.24, fc="#f8f8ff")
    label(ax, 0.885, 0.61, "2D extension", size=13, weight="bold", ha="center")
    label(ax, 0.885, 0.56, "lateral x", size=10, ha="center", color="#555")
    label(ax, 0.885, 0.52, "vertical y stack", size=10, ha="center", color="#555")
    label(ax, 0.885, 0.48, "grain boundaries", size=10, ha="center", color="#555")
    label(ax, 0.885, 0.44, "frozen ions in JV", size=10, ha="center", color="#555")

    save(fig, "device_structure.png")


def generate_transport_equations() -> None:
    fig, ax = setup(16, 12)
    label(ax, 0.03, 0.96, "Transport Processes and Boundary Conditions", size=24, weight="bold")
    boxes = [
        (0.04, 0.55, 0.44, 0.33, "Carrier transport", "#3b73e8"),
        (0.52, 0.55, 0.44, 0.33, "Ion migration", "#f28c18"),
        (0.04, 0.14, 0.44, 0.33, "Recombination channels", "#d9443f"),
        (0.52, 0.14, 0.44, 0.33, "Optics, Poisson, and 2D", "#2f9a4a"),
    ]
    for x, y, w, h, title, color in boxes:
        rounded(ax, (x, y), w, h, fc="#fbfbfb")
        label(ax, x + 0.02, y + h - 0.04, title, size=15, weight="bold")
        ax.plot([x + 0.02, x + w - 0.02], [y + h - 0.095, y + h - 0.095], color=color, lw=2)

    label(ax, 0.06, 0.755, "J_n = q mu_n n E + q D_n dn/dx", size=13, weight="bold")
    label(ax, 0.06, 0.705, "J_p = q mu_p p E - q D_p dp/dx", size=13, weight="bold")
    label(ax, 0.06, 0.655, "Default contacts: n,p pinned to equilibrium", size=11)
    pill(ax, 0.37, 0.655, "DIRICHLET", "#3b73e8")
    label(ax, 0.06, 0.610, "FULL selective contacts: J = +/- q S (u - u_eq)", size=11)
    pill(ax, 0.37, 0.610, "ROBIN", "#6b55c7")
    label(ax, 0.06, 0.570, "S=null -> ohmic pin | S=0 -> blocking | large S -> ohmic limit", size=10, color="#666")

    label(ax, 0.54, 0.755, "F_P = -D_ion [dP/dx + (q/kBT) P dphi/dx] steric(P)", size=12, weight="bold")
    label(ax, 0.54, 0.705, "dP/dt = -dF_P/dx; dual mode also tracks P-", size=11)
    label(ax, 0.54, 0.645, "BCs: F(0)=F(L)=0", size=12, weight="bold")
    pill(ax, 0.74, 0.645, "ZERO-FLUX", "#f28c18")
    label(ax, 0.54, 0.585, "2D J-V freezes ions as static Poisson background", size=10, color="#666")

    label(ax, 0.06, 0.345, "R_SRH = (np - ni^2) / [ tau_p(n+n1) + tau_n(p+p1) ]", size=11)
    label(ax, 0.06, 0.305, "R_rad = B(np - ni^2), with photon recycling / reabsorption hooks", size=11)
    label(ax, 0.06, 0.260, "R_Auger = (C_n n + C_p p)(np - ni^2)", size=11)
    label(ax, 0.06, 0.215, "R_iface and 2D grain boundaries modify local lifetimes", size=11)

    label(ax, 0.54, 0.345, "G_BL(x) or transfer-matrix G_TMM(x, lambda)", size=12, weight="bold")
    label(ax, 0.54, 0.305, "Poisson: -d/dx(eps0 eps_r dphi/dx) = q(p-n+ND-NA+P-P0)", size=10)
    label(ax, 0.54, 0.260, "2D: sparse Poisson + SG fluxes on (Ny x Nx) mesh", size=11)
    label(ax, 0.54, 0.215, "BCs: vertical contacts, periodic lateral boundaries", size=11)
    save(fig, "transport_equations.png")


def generate_solver_pipeline() -> None:
    fig, ax = setup(16, 11)
    label(ax, 0.03, 0.96, "Solver Pipeline: 1D Core and 2D Extension", size=24, weight="bold")

    steps = [
        ("DeviceStack / YAML", "layers, mode, contacts, optics, microstructure", "#e8f4ea"),
        ("Build caches", "MaterialArrays or MaterialArrays2D; Poisson factors; optical G", "#e8f0ff"),
        ("Assemble RHS", "charge density, Poisson solve, SG fluxes, recombination", "#fff4d8"),
        ("Boundary hooks", "ohmic pins or FULL Robin contacts; zero-flux ions", "#f2eaff"),
        ("Integrate / sweep", "Radau time stepping; JV, EQE, IS, TPV, degradation, 2D JV", "#fbe7e7"),
        ("Results", "metrics, snapshots, current decomposition, V_oc(L_g)", "#f4f4f4"),
    ]
    x, w, h = 0.25, 0.50, 0.095
    y = 0.82
    for title, desc, color in steps:
        rounded(ax, (x, y), w, h, fc=color, ec="#777", lw=1.5)
        label(ax, x + w / 2, y + h * 0.62, title, size=14, weight="bold", ha="center")
        label(ax, x + w / 2, y + h * 0.30, desc, size=10, color="#555", ha="center")
        if y > 0.24:
            ax.arrow(x + w / 2, y - 0.005, 0, -0.032, head_width=0.018, head_length=0.018, color="#666")
        y -= 0.13

    rounded(ax, (0.05, 0.45), 0.15, 0.20, fc="#ffffff")
    label(ax, 0.125, 0.61, "1D default", size=13, weight="bold", ha="center")
    label(ax, 0.125, 0.56, "fast screen", size=10, ha="center", color="#555")
    label(ax, 0.125, 0.52, "full hooks", size=10, ha="center", color="#555")
    label(ax, 0.125, 0.48, "all main experiments", size=10, ha="center", color="#555")

    rounded(ax, (0.80, 0.43), 0.15, 0.24, fc="#ffffff")
    label(ax, 0.875, 0.63, "2D path", size=13, weight="bold", ha="center")
    label(ax, 0.875, 0.58, "jv_2d", size=10, ha="center", color="#555")
    label(ax, 0.875, 0.54, "voc_grain_sweep", size=10, ha="center", color="#555")
    label(ax, 0.875, 0.50, "microstructure", size=10, ha="center", color="#555")
    label(ax, 0.875, 0.46, "frozen ions", size=10, ha="center", color="#555")
    save(fig, "solver_pipeline.png")


def generate_ui_layout() -> None:
    fig, ax = setup(15, 8)
    label(ax, 0.03, 0.94, "Web UI Layout", size=24, weight="bold")
    outer = (0.04, 0.18, 0.92, 0.62)
    rounded(ax, (outer[0], outer[1]), outer[2], outer[3], fc="#ffffff", ec="#444", lw=2)
    x0, y0, w, h = outer
    header_h = 0.09
    ax.plot([x0, x0 + w], [y0 + h - header_h, y0 + h - header_h], color="#444", lw=2)
    left_w, center_w = 0.18, 0.35
    ax.plot([x0 + left_w, x0 + left_w], [y0, y0 + h], color="#444", lw=2)
    ax.plot([x0 + left_w + center_w, x0 + left_w + center_w], [y0, y0 + h], color="#444", lw=2)

    label(ax, x0 + left_w / 2, y0 + h - header_h / 2, "Devices", size=13, weight="bold", ha="center")
    label(ax, x0 + left_w + center_w / 2, y0 + h - header_h / 2, "Device | Help", size=13, weight="bold", ha="center")
    label(ax, x0 + left_w + center_w + (w - left_w - center_w) / 2, y0 + h - header_h / 2,
          "Experiments", size=13, weight="bold", ha="center")

    label(ax, x0 + 0.03, y0 + 0.41, "Results / Compare", size=12)
    label(ax, x0 + 0.03, y0 + 0.34, "Select runs to overlay", size=10, color="#777")
    label(ax, x0 + 0.03, y0 + 0.29, "1D, 2D, tandem outputs", size=10, color="#777")

    cx = x0 + left_w + 0.03
    label(ax, cx, y0 + 0.40, "Device Configuration", size=12, weight="bold")
    label(ax, cx, y0 + 0.34, "Preset / Reset / Mode / Save As", size=11)
    label(ax, cx, y0 + 0.28, "Layer Builder + Stack Visualizer", size=11)
    label(ax, cx, y0 + 0.22, "Geometry / Transport / Recombination", size=10, color="#777")
    label(ax, cx, y0 + 0.17, "Ions & Optics / Contacts / Advanced", size=10, color="#777")

    rx = x0 + left_w + center_w + 0.03
    label(ax, rx, y0 + 0.43, "1D: J-V, Dark J-V, EQE, IS, TPV", size=11)
    label(ax, rx, y0 + 0.37, "C-V, Suns-Voc, Degradation", size=11)
    label(ax, rx, y0 + 0.31, "2D: J-V Sweep, V_oc(L_g)", size=11)
    label(ax, rx, y0 + 0.25, "Tandem: 2T J-V", size=11)
    rounded(ax, (rx, y0 + 0.08), 0.31, 0.10, fc="#f7f7f7", ec="#dddddd", lw=1)
    label(ax, rx + 0.155, y0 + 0.13, "Plotly panes + metric cards", size=11, color="#888", ha="center")
    save(fig, "ui_layout.png")


def main() -> None:
    generate_device_structure()
    generate_transport_equations()
    generate_solver_pipeline()
    generate_ui_layout()


if __name__ == "__main__":
    main()
