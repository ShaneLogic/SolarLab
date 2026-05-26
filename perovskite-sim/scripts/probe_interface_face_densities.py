#!/usr/bin/env python
"""Phase E1.6a investigation — instrument scaps_mirror's PVK/ETL interface.

Dumps:
- Solver state ``(n, p, φ)`` at the PVK-interior (idx-1), interface (idx),
  and ETL-interior (idx+1) nodes at V_app ∈ {V_oc, cliff -0.5, spike +0.3}
- SG flux-implied face densities (Selberherr face concentration formula
  derived from the SG flux discretization in ``physics/continuity.py``)
- Boltzmann-from-Fermi-continuity predictions (Phase E1.6 attempt-2
  formula, for cross-reference)
- Computed E1.5 cross-carrier sampling values (``n[idx+1]·face_factor·NC_ratio``
  with current MaterialArrays values)

Generates a table comparing all three candidates side-by-side. Output
goes to ``outputs/scaps_validation_e1_6a_probe.txt`` as plain text for
inclusion in the Phase A3 RFC memo.

Goal: identify which face-density formulation matches SCAPS' interface-
plane carrier sampling within ~1 order of magnitude (current E1.5
cross-carrier over-counts by ~5 orders). Numbers feed the RFC decision
between Anderson abrupt-junction face density vs SG-flux-consistent
face density vs thin-shell volumetric SRH.

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
        f"  E1.5 cross sample:        n_eval={n_e15:.3e}  p_eval={p_e15:.3e}"
    )
    out_lines.append(
        f"  Boltzmann (E1.6 v2):      n_face={n_face_boltzmann:.3e}  p_face={p_face_boltzmann:.3e}"
    )
    out_lines.append(
        f"  SG-Selberherr (proposed): n_face={n_face_sg:.3e}  p_face={p_face_sg:.3e}"
    )
    np_e15 = n_e15 * p_e15
    np_boltzmann = n_face_boltzmann * p_face_boltzmann
    np_sg = n_face_sg * p_face_sg
    out_lines.append(
        f"  np product: E1.5={np_e15:.3e}  Boltzmann={np_boltzmann:.3e}  SG={np_sg:.3e}"
    )
    out_lines.append("")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path,
        default=Path("outputs/scaps_validation_e1_6a_probe.txt"),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/scaps_mirror.yaml"))
    args = parser.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    base = load_scaps_yaml(args.config)

    # First measure V_oc on the current scaps_mirror baseline (E1.5 active).
    r = run_jv_sweep(base, N_grid=30, n_points=20, v_rate=5.0, V_max=1.6)
    voc = r.metrics_fwd.V_oc
    print(f"Baseline V_oc = {voc:.4f} V (bracketed={r.metrics_fwd.voc_bracketed})")

    out_lines: list[str] = [
        f"Phase E1.6a probe — face-density formulations at PVK/ETL interface",
        f"Config: {args.config}",
        f"Baseline V_oc = {voc:.4f} V",
        "",
        "Three face-density candidates compared:",
        "  • E1.5 cross-carrier (CURRENT SOLVER) — n at idx+1 (ETL interior),",
        "    p at idx-1 (PVK interior). 5-order over-count vs SCAPS interface plane.",
        "  • Boltzmann-from-Fermi (E1.6 attempt-2, REVERTED) — derived from",
        "    Fermi continuity assumption; non-physical under photo-injection.",
        "  • SG-Selberherr (PROPOSED Phase E1.6) — extracted from the same",
        "    Bernoulli flux discretization the solver already uses; consistent",
        "    with the existing SG flux machinery in physics/continuity.py.",
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
