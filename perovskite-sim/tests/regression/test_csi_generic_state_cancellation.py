"""Root-cause certificate for the c-Si general-driver capability boundary.

The local c-Si homojunction has a residual-certified illuminated solution in
split quasi-Fermi reference/increment variables.  Collapsing that same state
to absolute carrier densities removes Newton-scale quasi-Fermi differences at
the highly conducting emitter.  The ordinary density-form SG assembly then
reports an O(1) continuity defect even though the underlying physical state is
certified.  This test distinguishes that representation loss from a bad seed
or an insufficient transient settle.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from perovskite_sim.constants import Q
from perovskite_sim.experiments.jv_sweep import build_electrical_grid
from perovskite_sim.experiments.quasi_fermi_steady_state import (
    solve_quasi_fermi_steady_state,
)
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.solver.mol import assemble_rhs, build_material_arrays


pytestmark = [pytest.mark.slow, pytest.mark.regression]
ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/cSi_homojunction.yaml"


def test_certified_qf_state_exposes_density_sg_cancellation_boundary():
    stack = load_device_from_yaml(CONFIG)
    x = build_electrical_grid(stack, 200)
    mat = build_material_arrays(x, stack)
    qf = solve_quasi_fermi_steady_state(
        x,
        stack,
        V_app=0.0,
        illuminated=True,
        mat=mat,
    )

    assert qf.max_normalized_cell_residual < 1.0e-10
    assert qf.electron_continuity_bound_A_m2 < 1.0e-6
    assert qf.hole_continuity_bound_A_m2 < 1.0e-6
    assert qf.face_current_spread_A_m2 < 1.0e-6

    # This is the exact state the general transient/algebraic drivers receive:
    # absolute n and p, with Poisson recomputed by the ordinary RHS.  No state
    # perturbation, solver iteration, or alternate physical model is involved.
    generic_rate = assemble_rhs(
        0.0,
        qf.y,
        x,
        stack,
        mat,
        illuminated=True,
        V_app=0.0,
    )
    node_count = len(x)
    carrier_peak = max(float(np.max(qf.y[: 2 * node_count])), 1.0)
    max_peak_scaled_residual = float(
        np.max(np.abs(generic_rate[: 2 * node_count] / carrier_peak))
    )
    interior = slice(1, -1)
    electron_bound = float(
        Q
        * np.sum(
            np.abs(generic_rate[:node_count][interior])
            * mat.dx_cell[interior]
        )
    )
    hole_bound = float(
        Q
        * np.sum(
            np.abs(generic_rate[node_count : 2 * node_count][interior])
            * mat.dx_cell[interior]
        )
    )

    # Broad lower bounds make this a capability discriminator, not a pinned
    # floating-point snapshot.  The measured values are about 33.1, 3.18 A/m2,
    # and 7.10e-4 A/m2, respectively, versus sub-nA/m2 QF defects above.
    assert max_peak_scaled_residual > 10.0
    assert electron_bound > 1.0
    assert hole_bound > 1.0e-4
    assert electron_bound > 1.0e8 * qf.electron_continuity_bound_A_m2
    assert hole_bound > 1.0e4 * qf.hole_continuity_bound_A_m2
