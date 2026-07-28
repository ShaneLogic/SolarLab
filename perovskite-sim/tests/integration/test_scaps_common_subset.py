"""Review section 7 case 10 — the SCAPS common-subset benchmark.

The review's point is that SCAPS is not an equivalent solver reached by
substituting a parameter set, so a comparison is only meaningful once the
common subset of physics is established: no tunnelling, no grading, matched
contacts and matched generation, base drift-diffusion verified FIRST, then
each additional mechanism added one at a time.

This file asserts the two things that make such a comparison honest:

1. The alignment PRECONDITIONS actually hold on the comparison configs, so
   a run cannot silently compare two different models. These are cheap
   structural checks and they are the ones most likely to rot -- a default
   flip elsewhere in the codebase would otherwise change what "the SCAPS
   comparison" means without any test noticing.

2. The J_sc disagreement is OPTICAL, and provably so rather than by
   attribution. The absorbed-photon budget bounds the collectable current
   from above, so if the budget is already below the SCAPS J_sc then no
   amount of recombination, transport or interface physics can close the
   gap. That converts an open-ended "our J_sc is low" into a closed
   statement about which subsystem owns the difference.

Measured 2026-07-28, mesh-independent at N_grid = 30 / 60 / 120:

    config              absorbed budget   SCAPS J_sc    deficit
    scaps_mirror         219.01 A/m^2     262.82        -16.7 %
    scaps_mirror_v2      235.06 A/m^2     262.82        -10.6 %

Both budgets are the telescoped transfer-matrix Poynting-flux drop over the
electrical layers, so they are exact on any mesh (see
tests/regression/test_tmm_photon_conservation.py). The cause is the one
named in the manual's alignment table: SCAPS integrates a scalar absorption
coefficient with no Fresnel reflection at the transport-layer boundary,
while SolarLab runs a coherent transfer matrix over the full stack.

NOT claimed here: that SolarLab's optics are more correct than SCAPS's. The
claim is only that the difference lives in the optical model and cannot be
repaired downstream of it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.experiments.jv_sweep import _layer_node_counts, run_jv_sweep
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.physics.generation import dual_cell_widths
from perovskite_sim.scaps_compat import load_scaps_yaml
from perovskite_sim.solver.mol import build_material_arrays

Q = 1.602176634e-19

_CONFIGS = ("configs/scaps_mirror.yaml", "configs/scaps_mirror_v2.yaml")
_REFERENCE = Path("tests/integration/scaps_reference.json")


@pytest.fixture(scope="module")
def scaps_base():
    """SCAPS base-model figures of merit, from the extracted reference."""
    return json.loads(_REFERENCE.read_text())["base_model"]


def _absorbed_budget(stack, N_grid: int) -> float:
    """q * sum(G * dx_cell) -- the photons the ELECTRICAL layers absorb.

    This is exactly the integral the RHS performs, so it is the true upper
    bound on the terminal photocurrent, not an estimate of it.
    """
    x = multilayer_grid([
        Layer(l.thickness, n) for l, n in
        zip(electrical_layers(stack), _layer_node_counts(stack, N_grid))
    ])
    mat = build_material_arrays(x, stack)
    return Q * float(np.sum(mat.G_optical * dual_cell_widths(x)))


# ---------------------------------------------------------------------------
# 1. the comparison is between the models we think it is
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfg", _CONFIGS)
def test_tunnelling_and_grading_are_off(cfg):
    """The review requires both sides to disable tunnelling and grading, or
    to validate them separately. SolarLab satisfies this by default -- pin
    it, so a future default flip cannot silently change what the SCAPS
    comparison means."""
    stack = load_scaps_yaml(cfg)
    assert stack.interface_tunneling is False, (
        f"{cfg}: interface_tunneling is on; SCAPS uses WKB tunnelling and "
        "the two forms are not comparable (manual alignment table, row 7)"
    )
    assert stack.band_grading is False, (
        f"{cfg}: band_grading is on; SolarLab does not grade the optics, so "
        "only the common subset is comparable (row 8)"
    )


@pytest.mark.parametrize("cfg", _CONFIGS)
def test_optics_are_transfer_matrix_on_every_electrical_layer(cfg):
    """The J_sc claim below is a statement about the OPTICAL model, so the
    optical model has to be the one under test: every electrical layer must
    carry n,k data rather than falling back to a scalar alpha."""
    stack = load_scaps_yaml(cfg)
    missing = [
        l.params.optical_material for l in electrical_layers(stack)
        if l.params.optical_material is None
    ]
    assert not missing, (
        f"{cfg}: {len(missing)} electrical layer(s) have no optical_material, "
        "so this config is not exercising the transfer-matrix path"
    )


# ---------------------------------------------------------------------------
# 2. the J_sc gap is optical, and provably not closable downstream
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfg", _CONFIGS)
def test_absorbed_budget_is_mesh_independent(cfg):
    """The bound is only a bound if it does not move with the mesh."""
    stack = load_scaps_yaml(cfg)
    budgets = {n: _absorbed_budget(stack, n) for n in (30, 60, 120)}
    ref = budgets[60]
    worst = max(abs(v - ref) / ref for v in budgets.values())
    assert worst < 1e-12, f"{cfg}: budget is mesh-dependent: {budgets}"


@pytest.mark.parametrize("cfg", _CONFIGS)
def test_jsc_gap_to_scaps_is_optical_not_electrical(cfg, scaps_base):
    """The decisive statement: the photon budget is already below the SCAPS
    J_sc, so the gap cannot be closed by any downstream physics.

    J_sc <= (photons absorbed by the electrical layers) is an identity, not
    a modelling assumption -- every collected carrier needs an absorbed
    photon. So a budget below the SCAPS value means recombination,
    transport, contacts and interface formulation are all irrelevant to this
    particular disagreement.
    """
    stack = load_scaps_yaml(cfg)
    budget = _absorbed_budget(stack, 60)
    scaps_jsc = scaps_base["Jsc_mA_cm2"] * 10.0        # mA/cm^2 -> A/m^2

    assert budget < scaps_jsc, (
        f"{cfg}: absorbed-photon budget {budget:.2f} A/m^2 is NOT below the "
        f"SCAPS J_sc {scaps_jsc:.2f} A/m^2 -- the optical attribution in the "
        "manual's alignment table no longer holds and must be re-derived"
    )


@pytest.mark.parametrize("cfg", _CONFIGS)
def test_simulated_jsc_respects_its_own_photon_budget(cfg):
    """Closes the argument at the other end: the current the solver actually
    reports must sit under the budget used above.

    Without this the previous test would bound a quantity the sweep does not
    respect, which is precisely the defect the photon-conservation work
    fixed (the superseded J_sc exceeded its own budget).
    """
    stack = load_scaps_yaml(cfg)
    budget = _absorbed_budget(stack, 30)
    m = run_jv_sweep(stack, N_grid=30, n_points=20, v_rate=5.0,
                     V_max=1.6).metrics_fwd
    assert 0.0 < m.J_sc <= budget * (1.0 + 1e-9), (
        f"{cfg}: J_sc = {m.J_sc:.4f} A/m^2 exceeds its own absorbed-photon "
        f"budget {budget:.4f} A/m^2"
    )
