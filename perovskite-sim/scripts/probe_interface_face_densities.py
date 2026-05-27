#!/usr/bin/env python
"""Phase E1.6a / E2a investigation — instrument scaps_mirror's PVK/ETL interface.

Dumps:
- Solver state ``(n, p, φ)`` at the PVK-interior (idx-1), interface (idx),
  and ETL-interior (idx+1) nodes at V_app ∈ {V_oc, cliff -0.5, spike +0.3}
- Four face-density candidates side-by-side:
    1. E1.5 cross-carrier (CURRENT SOLVER)
    2. Boltzmann-from-Fermi (E1.6 v2, REVERTED — photo-injection blow-up)
    3. SG-Selberherr (Phase A1, REJECTED — p ETL collapse)
    4. E2 band-bending depletion (NEW — same-layer φ-only projection)

Output goes to ``outputs/<name>.txt`` as plain text for the Phase A3 /
Phase E2 design RFC memos.

Run: ``cd perovskite-sim && python scripts/probe_interface_face_densities.py``
"""
from __future__ import annotations

import argparse
import dataclasses
import math
from pathlib import Path

import numpy as np

from perovskite_sim.constants import V_T
from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.experiments.jv_sweep import run_jv_sweep
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.scaps_compat import load_scaps_yaml
from perovskite_sim.experiments.jv_sweep import extract_spatial_snapshot
from perovskite_sim.solver.mol import build_material_arrays
from perovskite_sim.solver.illuminated_ss import solve_illuminated_ss
from perovskite_sim.sweeps.device_parameter_sweep import SweepPoint, apply_sweep_point


def sg_face_density_n(n_L: float, n_R: float, phi_n_L: float, phi_n_R: float) -> float:
    """SG-flux-implied face concentration for electrons.

    Selberherr formula: n_face = (n_L - n_R · exp(-ξ_n)) / (1 - exp(-ξ_n))
    where ξ_n = (Φ_n_R - Φ_n_L) / V_T and Φ_n = φ + χ is the electron-band
    Slotboom potential.

    Degenerate case ξ_n → 0 falls back to the arithmetic mean.
    """
    xi = (phi_n_R - phi_n_L) / V_T
    if abs(xi) < 1e-8:
        return 0.5 * (n_L + n_R)
    denom = 1.0 - math.exp(-xi)
    return (n_L - n_R * math.exp(-xi)) / denom


def sg_face_density_p(p_L: float, p_R: float, phi_p_L: float, phi_p_R: float) -> float:
    """SG-flux-implied face concentration for holes.

    Selberherr formula sign-flipped for hole transport:
    p_face = (p_L - p_R · exp(+ξ_p)) / (1 - exp(+ξ_p))
    where ξ_p = (Φ_p_R - Φ_p_L) / V_T and Φ_p = φ + χ + Eg.
    """
    xi = (phi_p_R - phi_p_L) / V_T
    if abs(xi) < 1e-8:
        return 0.5 * (p_L + p_R)
    denom = 1.0 - math.exp(xi)
    return (p_L - p_R * math.exp(xi)) / denom


def probe_one_point(stack, V_app: float, label: str, out_lines: list[str]) -> None:
    """Run an illuminated steady-state at V_app, dump interface neighbourhood."""
    elec = electrical_layers(stack)
    layers_grid = [Layer(thickness=L.thickness, N=30 // len(elec)) for L in elec]
    x = multilayer_grid(layers_grid)
    mat = build_material_arrays(x, stack)
    y = solve_illuminated_ss(x, stack, V_app=V_app)
    N = len(x)

    # Use the experiment-side spatial-snapshot helper so n/p/φ/ρ are
    # extracted with the same routine the JV sweeps use — guarantees
    # state vector unpacking + Poisson reconstruction matches the
    # solver's own conventions.
    snap = extract_spatial_snapshot(x, y, stack, V_app=V_app, mat=mat)
    n = np.asarray(snap.n)
    p = np.asarray(snap.p)
    phi = np.asarray(snap.phi)

    # PVK/ETL interface is the last interior interface (k = -1 in 3-layer stack).
    k = len(mat.interface_nodes) - 1
    idx = mat.interface_nodes[k]
    idx_L = max(idx - 1, 0)
    idx_R = min(idx + 1, N - 1)

    # Slotboom-potential face inputs (Φ_n = φ + χ; Φ_p = φ + χ + Eg).
    Phi_n_L = phi[idx_L] + mat.chi[idx_L]
    Phi_n_R = phi[idx_R] + mat.chi[idx_R]
    Phi_p_L = phi[idx_L] + mat.chi[idx_L] + mat.Eg[idx_L]
    Phi_p_R = phi[idx_R] + mat.chi[idx_R] + mat.Eg[idx_R]

    # SG face densities — Selberherr formula.
    n_face_sg = sg_face_density_n(n[idx_L], n[idx_R], Phi_n_L, Phi_n_R)
    p_face_sg = sg_face_density_p(p[idx_L], p[idx_R], Phi_p_L, Phi_p_R)

    # Boltzmann-from-Fermi (E1.6 attempt-2 formula, for cross-reference).
    delta_E_C = mat.chi[idx_L] - mat.chi[idx_R]
    delta_E_V = (mat.chi[idx_R] + mat.Eg[idx_R]) - (mat.chi[idx_L] + mat.Eg[idx_L])
    # N_C estimate under N_C = N_V symmetry: ni · exp(Eg / 2 V_T).
    NC_L = float(mat.ni_sq[idx_L]) ** 0.5 * math.exp(mat.Eg[idx_L] / (2.0 * V_T))
    NC_R = float(mat.ni_sq[idx_R]) ** 0.5 * math.exp(mat.Eg[idx_R] / (2.0 * V_T))
    n_face_boltzmann = min(NC_R, n[idx_L] * (NC_R / NC_L) * math.exp(-delta_E_C / V_T))
    p_face_boltzmann = min(NC_L, p[idx_R] * (NC_L / NC_R) * math.exp(+delta_E_V / V_T))

    # E1.5 cross-carrier sampling (CURRENT solver path).
    n_e15 = float(n[idx_R])
    p_e15 = float(p[idx_L])

    # Phase E2 candidate (d) — band-bending depletion within source layer.
    # Project majority carrier from its bulk to the interface plane using
    # ONLY same-layer φ band-bending (no χ step crossing). This avoids the
    # photo-injection breakdown of candidate (b), which assumed quasi-Fermi
    # continuity ACROSS the χ step and amplified Q-Fermi splitting through
    # exp(ΔE_V/V_T) ≈ 8e8.
    #
    # In ETL (right side), under SS the quasi-Fermi level for electrons is
    # flat across the bulk (drift gradient ≪ V_T), so:
    #   n(idx) = n(idx+1) · exp((φ(idx) − φ(idx+1))/V_T)
    # where the exponential captures the BAND-BENDING DEPLETION as φ drops
    # entering the heterojunction depletion zone. Similarly for holes
    # entering from the PVK side:
    #   p(idx) = p(idx−1) · exp(−(φ(idx) − φ(idx−1))/V_T)
    # (negative exponent because hole density scales with −φ.)
    n_face_bbd = float(n[idx_R]) * math.exp((phi[idx] - phi[idx_R]) / V_T)
    p_face_bbd = float(p[idx_L]) * math.exp(-(phi[idx] - phi[idx_L]) / V_T)

    out_lines.append(f"=== {label}  (V_app = {V_app:+.3f} V) ===")
    out_lines.append(
        f"  Interface idx={idx}, neighbours [idx-1={idx_L}, idx+1={idx_R}]"
    )
    out_lines.append(f"  φ:    L={phi[idx_L]:+.4f}  M={phi[idx]:+.4f}  R={phi[idx_R]:+.4f}")
    out_lines.append(f"  n:    L={n[idx_L]:.3e}  M={n[idx]:.3e}  R={n[idx_R]:.3e}")
    out_lines.append(f"  p:    L={p[idx_L]:.3e}  M={p[idx]:.3e}  R={p[idx_R]:.3e}")
    out_lines.append(
        f"  ΔE_C(L−R)={delta_E_C:+.3f} eV   ΔE_V(R−L)={delta_E_V:+.3f} eV"
    )
    out_lines.append(
        f"  Δφ(M−R)={phi[idx]-phi[idx_R]:+.4f} V   Δφ(M−L)={phi[idx]-phi[idx_L]:+.4f} V"
    )
    out_lines.append(
        f"  E1.5 cross sample:        n_eval={n_e15:.3e}  p_eval={p_e15:.3e}"
    )
    out_lines.append(
        f"  Boltzmann (E1.6 v2):      n_face={n_face_boltzmann:.3e}  p_face={p_face_boltzmann:.3e}"
    )
    out_lines.append(
        f"  SG-Selberherr (proposed): n_face={n_face_sg:.3e}  p_face={p_face_sg:.3e}"
    )
    out_lines.append(
        f"  E2 band-bending depl:     n_face={n_face_bbd:.3e}  p_face={p_face_bbd:.3e}"
    )
    np_e15 = n_e15 * p_e15
    np_boltzmann = n_face_boltzmann * p_face_boltzmann
    np_sg = n_face_sg * p_face_sg
    np_bbd = n_face_bbd * p_face_bbd
    out_lines.append(
        f"  np product: E1.5={np_e15:.3e}  Boltz={np_boltzmann:.3e}  SG={np_sg:.3e}  BBD={np_bbd:.3e}"
    )
    out_lines.append("")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path,
        default=Path("outputs/scaps_validation_e1_6a_probe.txt"),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/scaps_mirror.yaml"))
    parser.add_argument(
        "--robin", action="store_true",
        help=(
            "Force Robin contacts active via dataclasses.replace (mode=full, "
            "S_n_right=1e5, S_p_left=1e5, S_n_left=1e-4, S_p_right=1e-4). "
            "Phase A2 hypothesis: does Robin shift p_ETL out of the machine-"
            "epsilon regime so SG-Selberherr face density becomes viable?"
        ),
    )
    args = parser.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    base = load_scaps_yaml(args.config)

    if args.robin:
        # Phase A2 — Robin contacts to test whether p_ETL exits the
        # machine-epsilon regime. Same S pattern as Sprint 1a probe.
        base = dataclasses.replace(
            base, mode="full",
            S_n_right=1.0e5, S_p_right=1.0e-4,
            S_p_left=1.0e5, S_n_left=1.0e-4,
        )
        print("Phase A2 — Robin contacts ACTIVE (mode=full)")

    # First measure V_oc on the current scaps_mirror baseline (E1.5 active).
    r = run_jv_sweep(base, N_grid=30, n_points=20, v_rate=5.0, V_max=1.6)
    voc = r.metrics_fwd.V_oc
    print(f"Baseline V_oc = {voc:.4f} V (bracketed={r.metrics_fwd.voc_bracketed})")

    out_lines: list[str] = [
        f"Phase E2a probe — face-density formulations at PVK/ETL interface",
        f"Config: {args.config}",
        f"Baseline V_oc = {voc:.4f} V",
        "",
        "Four face-density candidates compared:",
        "  • E1.5 cross-carrier (CURRENT SOLVER) — n at idx+1 (ETL interior),",
        "    p at idx-1 (PVK interior). 5-order over-count vs SCAPS interface plane.",
        "  • Boltzmann-from-Fermi (E1.6 attempt-2, REVERTED) — derived from",
        "    Fermi continuity ACROSS the χ step. Amplified Q-Fermi splitting via",
        "    exp(ΔE_V/V_T) ≈ 8e8 — non-physical under photo-injection.",
        "  • SG-Selberherr (Phase A1, REJECTED) — extracted from Bernoulli flux",
        "    discretization. Collapses on ETL side (p machine epsilon ⇒ cliff",
        "    direction LOST). Rejected by Phase A1 probe; Phase A2 Robin",
        "    confirmed dead.",
        "  • E2 Band-bending depletion (NEW) — project majority carrier from",
        "    its bulk to interface plane via SAME-LAYER φ Boltzmann only. No",
        "    χ step crossing ⇒ photo-injection safe. Hypothesis: this is the",
        "    depletion factor SCAPS' thermionic-emission boundary applies",
        "    implicitly via Q-Fermi step at the heterointerface.",
        "",
    ]

    # Three probe points: V_oc baseline, deep cliff (-0.5), spike (+0.3)
    probe_one_point(base, V_app=voc, label="V_oc baseline", out_lines=out_lines)

    for d in [-0.5, +0.3]:
        pt = SweepPoint("p", "c", f"{d:+.2f}", {"etl_delta_ec_eV": d})
        swept = apply_sweep_point(base, pt)
        r_swept = run_jv_sweep(swept, N_grid=30, n_points=20, v_rate=5.0, V_max=1.6)
        voc_swept = r_swept.metrics_fwd.V_oc
        out_lines.append(f"--- ΔE_C={d:+.2f} V_oc={voc_swept:.4f} ---")
        probe_one_point(
            swept, V_app=voc_swept,
            label=f"ΔE_C={d:+.2f} V_oc point",
            out_lines=out_lines,
        )

    args.out.write_text("\n".join(out_lines))
    print(f"Probe output written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
