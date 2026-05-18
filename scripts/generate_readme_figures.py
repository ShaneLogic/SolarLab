"""Generate README diagram PNGs.

The figures intentionally follow the original README visual language: airy
white space, light panels, and concise physics labels.  They are generated so
future README figure updates do not require manual image editing.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "perovskite-sim" / "docs" / "images"
FONT = "Arial"

plt.rcParams.update({
    "font.family": FONT,
    "mathtext.fontset": "custom",
    "mathtext.rm": FONT,
    "mathtext.it": f"{FONT}:italic",
    "mathtext.bf": f"{FONT}:bold",
})


def setup(width: float, height: float) -> tuple[plt.Figure, plt.Axes]:
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


def text(ax: plt.Axes, x: float, y: float, s: str, *, size=16, weight="normal",
         color="#222222", ha="left", va="center", style="normal",
         rotation: float = 0.0) -> None:
    ax.text(
        x, y, s,
        fontsize=size, fontweight=weight, color=color, ha=ha, va=va,
        fontstyle=style, family=FONT, rotation=rotation,
    )


def panel(ax: plt.Axes, x: float, y: float, w: float, h: float, title: str,
          accent: str, *, title_size=16) -> None:
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.008,rounding_size=0.008",
        linewidth=1.5, edgecolor="#d4d4d4", facecolor="#fbfbfb",
    )
    ax.add_patch(box)
    text(ax, x + 0.02, y + h - 0.045, title, size=title_size, weight="bold")
    ax.plot([x + 0.02, x + w - 0.02], [y + h - 0.075, y + h - 0.075],
            color=accent, lw=2)


def pill(
    ax: plt.Axes,
    x: float,
    y: float,
    label: str,
    *,
    fc: str,
    color: str,
    width: float = 0.078,
    size: float = 9.5,
) -> None:
    box = FancyBboxPatch(
        (x, y - 0.014), width, 0.028,
        boxstyle="round,pad=0.004,rounding_size=0.004",
        linewidth=0, facecolor=fc,
    )
    ax.add_patch(box)
    text(ax, x + width / 2, y, label, size=size, weight="bold", color=color, ha="center")


def generate_transport_equations() -> None:
    fig, ax = setup(15.8, 15.3)
    text(ax, 0.03, 0.965, "Transport Processes and Boundary Conditions", size=25, weight="bold")

    panel(ax, 0.04, 0.57, 0.44, 0.34, "Carrier Transport (Scharfetter-Gummel)", "#4b7fff")
    panel(ax, 0.52, 0.57, 0.44, 0.34, "Ion Migration (Steric SG)", "#ff901a")
    panel(ax, 0.04, 0.18, 0.44, 0.33, "Recombination Channels", "#f04444")
    panel(ax, 0.52, 0.18, 0.44, 0.33, "Optical Generation, Poisson, and 2D", "#32a050")

    x, y = 0.06, 0.795
    text(ax, x, y, r"$\mathbf{J_n}=q\,\mu_n nE+qD_n\,\partial n/\partial x$", size=15.4)
    text(ax, x + 0.25, y, "electron current", size=11.0, color="#888888")
    text(ax, x, y - 0.048, r"$\mathbf{J_p}=q\,\mu_p pE-qD_p\,\partial p/\partial x$", size=15.4)
    text(ax, x + 0.25, y - 0.048, "hole current", size=11.0, color="#888888")
    ax.plot([0.06, 0.46], [0.710, 0.710], color="#dddddd", lw=1)
    text(ax, x, 0.678, r"$\mathbf{BCs:}$ default $n(0)=n_L,\ n(L)=n_R$", size=13.0)
    pill(ax, 0.370, 0.678, "DIRICHLET", fc="#eef0ff", color="#3150a8", width=0.080, size=8.6)
    text(ax, x, 0.644, r"default $p(0)=p_L,\ p(L)=p_R$", size=13.0)
    pill(ax, 0.370, 0.644, "DIRICHLET", fc="#eef0ff", color="#3150a8", width=0.080, size=8.6)
    text(ax, x, 0.612, r"FULL Robin: $J_{c,s}=\pm qS_{c,s}(u_c-u_{c,\mathrm{eq}})$", size=12.1)
    pill(ax, 0.370, 0.612, "ROBIN", fc="#efeaff", color="#5d45bd", width=0.080, size=8.8)
    text(ax, x, 0.586, r"$S=\mathrm{None}$: ohmic; $S=0$: blocking; large $S$: near-ohmic", size=10.5, color="#666666")

    x, y = 0.54, 0.795
    text(ax, x, y, r"$\mathbf{F_{ion}}=-D_{ion}\,[\,\partial_xP+(q/k_BT)P(1-P/P_{lim})\,\partial_x\varphi\,]$", size=12.5)
    text(ax, x, y - 0.050, r"$\partial P^{+}/\partial t=-\partial F_{ion,+}/\partial x$", size=14.0)
    text(ax, x + 0.235, y - 0.050, "positive vacancies", size=10.7, color="#888888")
    text(ax, x, y - 0.100, r"$\partial P^{-}/\partial t=-\partial F_{ion,-}/\partial x$", size=14.0)
    text(ax, x + 0.235, y - 0.100, "negative species", size=10.7, color="#888888")
    ax.plot([0.54, 0.94], [0.682, 0.682], color="#dddddd", lw=1)
    text(ax, x, 0.650, r"$\mathbf{BCs:}\ F(0)=F(L)=0$", size=14.0)
    pill(ax, 0.830, 0.650, "ZERO-FLUX", fc="#fff2df", color="#d94800", width=0.092, size=8.4)
    text(ax, x, 0.612, r"2D J-V freezes ions as static Poisson background", size=11.4, color="#666666")

    x, y = 0.06, 0.395
    text(ax, x, y, r"$\mathbf{R_{SRH}}=\dfrac{np-n_i^2}{\tau_p(n+n_1)+\tau_n(p+p_1)}$", size=14.4)
    text(ax, x + 0.300, y, "Shockley-Read-Hall", size=10.8, color="#888888")
    text(ax, x, y - 0.055, r"$\mathbf{R_{rad}}=B(np-n_i^2)$", size=14.8)
    text(ax, x + 0.130, y - 0.055, "radiative + recycling", size=10.8, color="#888888")
    text(ax, x, y - 0.110, r"$\mathbf{R_{Auger}}=(C_n n+C_p p)(np-n_i^2)$", size=14.8)
    text(ax, x + 0.238, y - 0.110, "Auger", size=10.8, color="#888888")
    text(ax, x, y - 0.165, r"$\mathbf{R_{iface}}$ and 2D grain boundaries modify local lifetimes", size=13.2)

    x, y = 0.54, 0.395
    text(ax, x, y, "Beer-Lambert", size=14.5, weight="bold")
    text(ax, x + 0.130, y, "legacy / fast fallback", size=10.8, color="#888888")
    text(
        ax,
        x,
        y - 0.034,
        r"$G_{\mathrm{BL}}(x)=\int_{\lambda}\alpha(\lambda,x)\,\Phi_0(\lambda)$",
        size=11.6,
    )
    text(
        ax,
        x + 0.010,
        y - 0.068,
        r"$\times\,\exp\!\left[-\int_0^x\alpha(\lambda,x')\,dx'\right]\,d\lambda$",
        size=11.6,
    )
    ax.plot([0.54, 0.94], [0.304, 0.304], color="#dddddd", lw=1)
    text(ax, x, y - 0.110, "Transfer Matrix (TMM)", size=14.2, weight="bold")
    text(ax, x + 0.205, y - 0.110, "FULL tier", size=10.8, color="#888888")
    text(ax, x, y - 0.150, r"$G_{TMM}(x,\lambda)=\hbar^{-1}\omega^{-1}(n/n_{amb})\,\alpha |E|^2$", size=11.8)
    ax.plot([0.54, 0.94], [0.222, 0.222], color="#dddddd", lw=1)
    text(ax, x, 0.206, r"$\mathbf{Poisson:}\ -\partial_x(\varepsilon_0\varepsilon_r\partial_x\varphi)=q(p-n+N_D-N_A+P-P_0)$", size=10.8)
    text(ax, x, 0.185, r"$\mathbf{2D:}$ sparse Poisson + SG fluxes on $(N_y\times N_x)$ mesh", size=10.8)

    save(fig, "transport_equations.png")


def generate_solver_pipeline() -> None:
    fig, ax = setup(15.0, 15.8)
    text(ax, 0.03, 0.965, "Solver Pipeline: How Each Timestep Works", size=25, weight="bold")

    def box(x, y, w, h, title, subtitle, fc, ec, title_color="#222222"):
        rounded = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.010,rounding_size=0.009",
            facecolor=fc, edgecolor=ec, linewidth=2,
        )
        ax.add_patch(rounded)
        text(ax, x + w / 2, y + h * 0.62, title, size=15, weight="bold", color=title_color, ha="center")
        text(ax, x + w / 2, y + h * 0.32, subtitle, size=12, color=title_color, ha="center")

    box(0.31, 0.84, 0.38, 0.060, "Initial Condition", "dark equilibrium or illuminated steady state", "#eaf6ec", "#2c7b34", "#1f7a2b")
    ax.arrow(0.50, 0.835, 0, -0.035, head_width=0.012, head_length=0.015, color="#666666")
    box(0.34, 0.745, 0.32, 0.045, "State vector y(t)", r"$y=[\,n,\ p,\ P^+,\ (P^-)\,]$ per grid node", "#e4f1ff", "#1764bd", "#0a55b2")
    ax.arrow(0.50, 0.740, 0, -0.035, head_width=0.012, head_length=0.015, color="#666666")

    outer = FancyBboxPatch((0.15, 0.325), 0.70, 0.35, boxstyle="round,pad=0.012,rounding_size=0.010",
                           facecolor="#fff7df", edgecolor="#f27e16", linewidth=2)
    ax.add_patch(outer)
    text(ax, 0.50, 0.650, r"$\mathbf{assemble\_rhs(t,y)\ \rightarrow\ dy/dt}$", size=15.5,
         color="#e64c00", ha="center")
    rows = [
        ("1", "Apply contact BCs to n, p", r"ohmic pins or FULL Robin $S_{c,s}$"),
        ("2", "Compute charge density rho", r"$\rho=q(p-n+P-P_0-N_A+N_D)$"),
        ("3", "Solve Poisson equation", r"$\varphi(0)=0,\ \varphi(L)=V_{bi}-V_{app}$"),
        ("4", "Compute generation G(x)", "TMM or Beer-Lambert fallback"),
        ("5", "Carrier continuity: dn/dt, dp/dt", "SG fluxes + G - R"),
        ("6", "Interface recombination + TE cap", "surface SRH + Richardson-Dushman"),
        ("7", r"Ion continuity: $dP^{+}/dt$, $dP^{-}/dt$", "zero-flux contacts; 2D freezes ions"),
    ]
    y0 = 0.605
    for i, (num, title, note) in enumerate(rows):
        yy = y0 - i * 0.043
        item = FancyBboxPatch((0.185, yy - 0.017), 0.31, 0.032,
                              boxstyle="round,pad=0.004,rounding_size=0.005",
                              facecolor="white", edgecolor="#bbbbbb", linewidth=1)
        ax.add_patch(item)
        circle_color = "#4285f4" if i < 3 else "#34a853" if i == 3 else "#ea4335" if i < 6 else "#f59e0b"
        ax.scatter([0.205], [yy], s=260, color=circle_color, zorder=3)
        text(ax, 0.205, yy, num, size=10.5, weight="bold", color="white", ha="center")
        text(ax, 0.225, yy, title, size=12.5)
        text(ax, 0.52, yy, note, size=10.5, color="#7d7d7d")

    ax.arrow(0.50, 0.315, 0, -0.035, head_width=0.012, head_length=0.015, color="#666666")
    box(0.26, 0.225, 0.48, 0.055, "Radau IIA (implicit Runge-Kutta)",
        r"stiff ODE; max_step capped near $V_{bi}$", "#eee7fb", "#6137b6", "#4f2cab")
    ax.arrow(0.50, 0.220, 0, -0.035, head_width=0.012, head_length=0.015, color="#666666")
    box(0.28, 0.125, 0.44, 0.060, "Outputs and diagnostics",
        r"$J(V)$, $Z(f)$, profiles, 2D J-V, $V_{oc}(L_g)$", "#fde8ee", "#d3222a", "#b51c24")

    ax.plot([0.74, 0.89, 0.89, 0.66], [0.252, 0.252, 0.770, 0.770], color="#6038bf", lw=1.5)
    text(ax, 0.90, 0.50, "next timestep", size=11.5, weight="bold", color="#6038bf",
         rotation=-90, ha="center")

    save(fig, "solver_pipeline.png")


def generate_device_structure() -> None:
    fig, ax = setup(16.0, 14.6)
    text(ax, 0.03, 0.965, "Device Structure (n-i-p Perovskite Solar Cell)", size=25, weight="bold")
    text(ax, 0.50, 0.905, "AM1.5G illumination", size=15, weight="bold", color="#e0a000", ha="center")
    ax.add_patch(Polygon([[0.49, 0.888], [0.51, 0.888], [0.50, 0.868]], color="#e0a000"))

    x, y, w, h = 0.22, 0.16, 0.58, 0.68
    parts = [
        ("layer", 0.095, "#b5b5b5", "Contact at x = 0 (Anode)", r"$\varphi(0)=0$ | ohmic or Robin $S_{\mathrm{left}}$"),
        ("layer", 0.170, "#e8f1ff", "HTL - Hole Transport Layer", r"high $N_A$ doping; blocks electrons ($\mu_n \ll \mu_p$)"),
        ("interface", 0.035, "#fffdf7", r"Interface SRH recombination $(v_n,v_p)$", ""),
        ("absorber", 0.315, "#fff2d8", "Absorber - Perovskite", r"generation $G(x)$; mobile ions $P^+(x,t)$ and $P^-(x,t)$; GBs reduce $\tau$"),
        ("interface", 0.035, "#fffdf7", r"Interface SRH recombination $(v_n,v_p)$", ""),
        ("layer", 0.205, "#e6f4e8", "ETL - Electron Transport Layer", r"high $N_D$ doping; blocks holes ($\mu_p \ll \mu_n$)"),
        ("layer", 0.145, "#b5b5b5", "Contact at x = L (Cathode)", r"$\varphi(L)=V_{bi}-V_{app}$ | ohmic or Robin $S_{\mathrm{right}}$"),
    ]
    yy = y + h
    for kind, frac, color, title, subtitle in parts:
        hh = h * frac
        yy -= hh
        ax.add_patch(Rectangle((x, yy), w, hh, facecolor=color, edgecolor="#666666", linewidth=1.5))
        if kind == "interface":
            text(ax, x + w / 2, yy + hh * 0.52, title, size=10.4, color="#777777", ha="center")
        elif kind == "absorber":
            text(ax, x + 0.025, yy + hh * 0.72, title, size=15.0, weight="bold")
            text(ax, x + w / 2, yy + hh * 0.51, r"$n(x,t)\leftarrow drift \rightarrow p(x,t)$", size=12.4, ha="center")
            text(ax, x + w / 2, yy + hh * 0.40, r"$\uparrow\ diffusion\ \downarrow$", size=11.8, style="italic", ha="center")
            text(ax, x + 0.025, yy + hh * 0.23, subtitle, size=11.4, color="#444444")
        else:
            text(ax, x + 0.025, yy + hh * 0.65, title, size=14.7, weight="bold")
            text(ax, x + 0.025, yy + hh * 0.34, subtitle, size=11.4, color="#444444")

    axis_x = x + w + 0.055
    ax.annotate(
        "",
        xy=(axis_x, y),
        xytext=(axis_x, y + h),
        arrowprops={"arrowstyle": "-|>", "color": "#555555", "lw": 2, "mutation_scale": 18},
    )
    text(ax, axis_x + 0.018, y + h, "x = 0", size=12, color="#666666", va="center")
    text(ax, axis_x + 0.018, y, "x = L", size=12, color="#666666", va="center")
    text(ax, axis_x + 0.040, y + h / 2, "1D coordinate x", size=12, color="#555555", rotation=-90, ha="center")
    text(
        ax,
        x + w / 2,
        0.095,
        "x increases through the stack; 2D extension adds a lateral direction",
        size=13,
        color="#444444",
        ha="center",
    )

    save(fig, "device_structure.png")


def generate_ui_layout() -> None:
    fig, ax = setup(14.8, 8.1)
    text(ax, 0.03, 0.90, "Web UI Layout", size=25, weight="bold")
    x, y, w, h = 0.04, 0.18, 0.92, 0.60
    frame = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.000,rounding_size=0.008",
                           facecolor="white", edgecolor="#444444", linewidth=2)
    ax.add_patch(frame)
    header_y = y + h - 0.085
    ax.plot([x, x + w], [header_y, header_y], color="#444444", lw=2)
    c1, c2 = x + 0.18, x + 0.53
    ax.plot([c1, c1], [y, y + h], color="#444444", lw=2)
    ax.plot([c2, c2], [y, y + h], color="#444444", lw=2)
    text(ax, x + 0.09, y + h - 0.043, "Devices", size=14, weight="bold", ha="center")
    text(ax, (c1 + c2) / 2, y + h - 0.043, "Device  |  Help", size=14, weight="bold", ha="center")
    text(ax, (c2 + x + w) / 2, y + h - 0.043,
         "J-V / Dark / EQE / IS / TPV / 2D", size=13.5, weight="bold", ha="center")
    text(ax, x + 0.02, y + 0.31, "Results / Compare", size=13.5)
    text(ax, x + 0.02, y + 0.24, "Select runs to overlay", size=11.0, color="#888888")
    text(ax, x + 0.02, y + 0.19, "1D, 2D, tandem plots", size=11.0, color="#888888")
    text(ax, c1 + 0.02, y + 0.37, "Device Configuration", size=14, weight="bold", color="#555555")
    text(ax, c1 + 0.02, y + 0.31, "preset  |  Reset  |  Mode  |  Save As", size=12.0)
    text(ax, c1 + 0.02, y + 0.24, "Layer Builder + Detail Editor", size=12.0)
    text(ax, c1 + 0.02, y + 0.18, "Geometry / Transport / Recombination", size=10.8, color="#888888")
    text(ax, c1 + 0.02, y + 0.14, "Ions & Optics / Contacts / Advanced", size=10.8, color="#888888")
    text(ax, c2 + 0.02, y + 0.40, "Experiment Parameters", size=13.5, weight="bold", color="#555555")
    text(ax, c2 + 0.02, y + 0.34, "Run     + progress bar", size=12.0)
    text(ax, c2 + 0.02, y + 0.28, r"2D J-V and $V_{oc}(L_g)$ use the same job stream", size=10.8, color="#777777")
    inner = FancyBboxPatch((c2 + 0.02, y + 0.10), 0.34, 0.11,
                           boxstyle="round,pad=0.010,rounding_size=0.008",
                           facecolor="#fbfbfb", edgecolor="#dddddd", linewidth=1)
    ax.add_patch(inner)
    text(ax, c2 + 0.19, y + 0.155, "Main Plot (Plotly) + metric cards", size=11.8, color="#888888", ha="center")
    save(fig, "ui_layout.png")


def main() -> None:
    generate_device_structure()
    generate_transport_equations()
    generate_solver_pipeline()
    generate_ui_layout()


if __name__ == "__main__":
    main()
