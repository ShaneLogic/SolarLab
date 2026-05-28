from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
from PIL import Image


OUT_DIR = Path(__file__).resolve().parent
GIF_PATH = OUT_DIR / "rhs_assembly_2d.gif"
PNG_PATH = OUT_DIR / "rhs_assembly_2d_preview.png"
STEPS_DIR = OUT_DIR / "rhs_assembly_2d_steps"
MILLISECONDS_PER_STEP = 5000

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]


STEPS = [
    ("1", "Unpack state Y", "Y -> n(y,x), p(y,x)"),
    ("2", "Charge density", "rho = q(p - n + N_D - N_A + P_static - P0)"),
    ("3", "Poisson solve", "div(eps grad phi) = -rho"),
    ("4", "Recombination", "R = R_SRH + R_rad + R_Auger"),
    ("5", "Generation", "G from optics or Beer-Lambert profile"),
    ("6", "SG currents", "n,p,phi -> J_n, J_p on cell faces"),
    ("7", "Current divergence", "div J from face currents"),
    ("8", "Carrier rates", "dn/dt, dp/dt from continuity equations"),
    ("9", "Contact BCs", "pin ohmic contacts or apply Robin flux"),
    ("10", "Return RHS", "flatten [dn/dt, dp/dt] -> dY/dt"),
]


def normalized(a: np.ndarray) -> np.ndarray:
    amin = float(np.nanmin(a))
    amax = float(np.nanmax(a))
    if abs(amax - amin) < 1e-30:
        return np.zeros_like(a)
    return (a - amin) / (amax - amin)


def synthetic_fields() -> dict[str, np.ndarray]:
    nx, ny = 42, 28
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x, y)

    absorber = np.exp(-((Y - 0.52) / 0.24) ** 8)
    gb = np.exp(-((X - 0.52) / 0.035) ** 2) * absorber

    n = 0.20 + 0.65 * X + 0.10 * np.exp(-((Y - 0.78) / 0.18) ** 2)
    p = 0.80 - 0.55 * X + 0.12 * np.exp(-((Y - 0.22) / 0.18) ** 2)
    rho = p - n + 0.22 * np.tanh((Y - 0.5) / 0.08)
    phi = 1.0 - Y + 0.10 * np.sin(np.pi * X) * np.sin(2.0 * np.pi * Y)
    R = 0.18 + 0.78 * normalized(n * p) + 0.60 * gb
    G = 0.15 + 0.85 * np.exp(-4.0 * Y) * absorber

    dphi_dx = np.gradient(phi, x, axis=1)
    dphi_dy = np.gradient(phi, y, axis=0)
    dn_dx = np.gradient(n, x, axis=1)
    dn_dy = np.gradient(n, y, axis=0)
    dp_dx = np.gradient(p, x, axis=1)
    dp_dy = np.gradient(p, y, axis=0)
    jnx = -(dn_dx + n * dphi_dx)
    jny = -(dn_dy + n * dphi_dy)
    jpx = -(dp_dx - p * dphi_dx)
    jpy = -(dp_dy - p * dphi_dy)
    div_jn = np.gradient(jnx, x, axis=1) + np.gradient(jny, y, axis=0)
    div_jp = np.gradient(jpx, x, axis=1) + np.gradient(jpy, y, axis=0)
    dn_dt = div_jn + G - R
    dp_dt = -div_jp + G - R
    dn_dt[0, :] = 0.0
    dn_dt[-1, :] = 0.0
    dp_dt[0, :] = 0.0
    dp_dt[-1, :] = 0.0

    return {
        "n": n,
        "p": p,
        "rho": rho,
        "phi": phi,
        "R": R,
        "G": G,
        "J": np.hypot(jnx + jpx, jny + jpy),
        "divJ": div_jn - div_jp,
        "dn_dt": dn_dt,
        "dp_dt": dp_dt,
    }


def draw_pipeline(ax, active: int) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    y_positions = np.linspace(0.93, 0.07, len(STEPS))
    for idx, ((num, title, _), y) in enumerate(zip(STEPS, y_positions)):
        is_active = idx == active
        is_done = idx < active
        face = "#fee8a8" if is_active else ("#d7f0df" if is_done else "#edf1f7")
        edge = "#b7791f" if is_active else ("#2f855a" if is_done else "#9aa6b2")
        ax.add_patch(
            Rectangle(
                (0.08, y - 0.032),
                0.84,
                0.055,
                facecolor=face,
                edgecolor=edge,
                linewidth=1.6 if is_active else 1.0,
            )
        )
        ax.text(0.115, y - 0.005, num, ha="center", va="center", fontsize=9, weight="bold")
        ax.text(0.17, y - 0.005, title, ha="left", va="center", fontsize=8.5)
        if idx < len(STEPS) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (0.50, y - 0.041),
                    (0.50, y_positions[idx + 1] + 0.023),
                    arrowstyle="-|>",
                    mutation_scale=8,
                    linewidth=0.8,
                    color="#7b8794",
                )
            )


def add_grid_overlay(ax, step: int) -> None:
    for x in np.linspace(-0.5, 41.5, 7):
        ax.plot([x, x], [-0.5, 27.5], color="white", lw=0.4, alpha=0.22)
    for y in np.linspace(-0.5, 27.5, 5):
        ax.plot([-0.5, 41.5], [y, y], color="white", lw=0.4, alpha=0.22)
    if step == 8:
        ax.plot([-0.5, 41.5], [-0.5, -0.5], color="#ffe08a", lw=3)
        ax.plot([-0.5, 41.5], [27.5, 27.5], color="#ffe08a", lw=3)


def frame_for(step_index: int, fields: dict[str, np.ndarray]) -> np.ndarray:
    num, title, formula = STEPS[step_index]
    fig = plt.figure(figsize=(12.8, 7.2), dpi=120)
    gs = fig.add_gridspec(
        3,
        4,
        width_ratios=[1.15, 1.15, 1.15, 0.95],
        height_ratios=[0.30, 1.0, 0.18],
        wspace=0.28,
        hspace=0.25,
    )
    fig.patch.set_facecolor("#f7f8fb")

    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.text(
        0.01,
        0.70,
        "2D RHS Assembly: Method-of-Lines operator F(Y)",
        fontsize=20,
        weight="bold",
        color="#172033",
        ha="left",
    )
    ax_title.text(
        0.01,
        0.22,
        f"Step {num}: {title}     {formula}",
        fontsize=13,
        color="#334155",
        ha="left",
    )

    if step_index == 0:
        panels = [("n(y,x)", fields["n"], "viridis"), ("p(y,x)", fields["p"], "magma"), ("packed Y", fields["n"] * 0 + 0.45, "Greys")]
    elif step_index == 1:
        panels = [("n", fields["n"], "viridis"), ("p", fields["p"], "magma"), ("rho", fields["rho"], "coolwarm")]
    elif step_index == 2:
        panels = [("rho", fields["rho"], "coolwarm"), ("Poisson solve", fields["rho"] * 0 + 0.5, "Greys"), ("phi", fields["phi"], "cividis")]
    elif step_index == 3:
        panels = [("n", fields["n"], "viridis"), ("p", fields["p"], "magma"), ("R(n,p)", fields["R"], "inferno")]
    elif step_index == 4:
        panels = [("optical stack", fields["G"], "YlGnBu"), ("G(y,x)", fields["G"], "YlGnBu"), ("source term", fields["G"], "YlGnBu")]
    elif step_index == 5:
        panels = [("phi", fields["phi"], "cividis"), ("n,p", 0.5 * normalized(fields["n"]) + 0.5 * normalized(fields["p"]), "viridis"), ("|J_n + J_p|", fields["J"], "plasma")]
    elif step_index == 6:
        panels = [("face currents", fields["J"], "plasma"), ("div J", fields["divJ"], "coolwarm"), ("control volumes", fields["divJ"] * 0 + 0.4, "Greys")]
    elif step_index == 7:
        panels = [("G - R", fields["G"] - fields["R"], "coolwarm"), ("dn/dt", fields["dn_dt"], "coolwarm"), ("dp/dt", fields["dp_dt"], "coolwarm")]
    elif step_index == 8:
        panels = [("before BC", fields["dn_dt"] + 0.15, "coolwarm"), ("contact rows", fields["dn_dt"] * 0 + 0.5, "Greys"), ("after BC", fields["dn_dt"], "coolwarm")]
    else:
        panels = [("dn/dt", fields["dn_dt"], "coolwarm"), ("dp/dt", fields["dp_dt"], "coolwarm"), ("flatten dY/dt", fields["dp_dt"] * 0 + 0.65, "Greys")]

    for i, (label, data, cmap) in enumerate(panels):
        ax = fig.add_subplot(gs[1, i])
        ax.set_title(label, fontsize=12, color="#1f2937", pad=8)
        im = ax.imshow(data, origin="lower", cmap=cmap, aspect="auto")
        add_grid_overlay(ax, step_index)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("x: lateral", fontsize=9)
        if i == 0:
            ax.set_ylabel("y: stack direction", fontsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#cbd5e1")
        fig.colorbar(im, ax=ax, shrink=0.78, pad=0.02)

    ax_pipe = fig.add_subplot(gs[1, 3])
    draw_pipeline(ax_pipe, step_index)

    ax_note = fig.add_subplot(gs[2, :])
    ax_note.axis("off")
    notes = [
        "Y is the packed solver state. The code reshapes it into 2D electron and hole density fields.",
        "Charge density couples carriers, doping, and frozen ionic background into Poisson's equation.",
        "Poisson is elliptic: one solve updates the electrostatic potential across the whole device.",
        "Recombination removes carriers; a grain boundary increases R by reducing local lifetimes.",
        "Generation adds carriers where light is absorbed, usually strongest near the illuminated side.",
        "Scharfetter-Gummel computes stable drift-diffusion currents on grid faces.",
        "Finite-volume divergence converts face currents into net inflow or outflow per cell.",
        "Continuity equations combine transport, generation, and recombination into time derivatives.",
        "Contact boundary conditions overwrite or correct boundary-row derivatives.",
        "The final RHS is packed back into the vector shape expected by Radau.",
    ]
    ax_note.text(0.01, 0.58, notes[step_index], fontsize=12, color="#334155", ha="left", va="center")

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    frame = buf[:, :, :3].copy()
    plt.close(fig)
    return frame


def main() -> None:
    fields = synthetic_fields()
    frames = []
    STEPS_DIR.mkdir(exist_ok=True)
    for idx in range(len(STEPS)):
        frame = frame_for(idx, fields)
        frames.append(frame)
        imageio.imwrite(STEPS_DIR / f"step_{idx + 1:02d}.png", frame)
        if idx == 0:
            imageio.imwrite(PNG_PATH, frame)

    pil_frames = [Image.fromarray(frame).convert("P", palette=Image.Palette.ADAPTIVE) for frame in frames]
    pil_frames[0].save(
        GIF_PATH,
        save_all=True,
        append_images=pil_frames[1:],
        duration=MILLISECONDS_PER_STEP,
        loop=0,
        optimize=False,
        disposal=2,
    )
    print(GIF_PATH)
    print(PNG_PATH)
    print(STEPS_DIR)


if __name__ == "__main__":
    main()
