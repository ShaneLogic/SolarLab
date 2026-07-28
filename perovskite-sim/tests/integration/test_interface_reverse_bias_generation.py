"""Review section 7 case 3 — interface generation under reverse bias.

The review's F-04 says the cross-carrier sampling plus the R_s >= 0 clamp
destroys physical depletion-region generation at reverse bias, so the
default is not a general interface-SRH model, and asks (first-priority
item 4) that the interface-plane / supply-conserving model become the
high-fidelity default with cross-sampling kept only as a compatibility
mode.

THE DIAGNOSIS IS CORRECT. THE PRESCRIBED REMEDY DOES NOT DELIVER IT.

Measured on configs/scaps_mirror_v2.yaml, dark, settled at V = -0.5 V,
N_grid = 30, as the areal interface current at the two electrical
interface nodes (negative = net generation):

    formulation                                   J_if [A/m^2]
    default (clamp on, defect-scoped)             0            0
    clamp disabled (SOLARLAB_IFACE_ALLOW_GEN=1)   -8.3644      ~0
    QSS plane rate (SOLARLAB_IFACE_QSS=1)         0            0
    interface_plane_closure = True                0            0

So all three formulations that are candidates for "the physical default"
return EXACTLY zero, and only removing the clamp produces generation. The
plane models are not being clamped -- they are non-negative BY
CONSTRUCTION: ``_qss_interface_R`` solves ``v_th*delta = SRH(proj - delta)``
for a depletion ``delta >= 0`` and returns ``R = v_th*delta``, so
``np < ni^2`` yields ``delta = 0`` rather than a negative rate. Promoting
either to the default would therefore LOCK IN the absence of depletion
generation rather than restore it.

What the clamp is for is also real, which is why this is not simply a bug
to delete: at a declared-defect interface the rate is referenced to a
cross-carrier bulk-asymptotic ni_eff^2, and part of the negative excursion
that reference produces is an artifact of the sampling, not physics. The
E9.3 work scoped the clamp to exactly those interfaces so defect-free ones
keep their real generation (pinned below).

Net position, recorded so it is not re-litigated from intuition: a correct
fix needs a rate model whose REFERENCE is physically right at the
interface plane and which is therefore *allowed* to go negative. Neither
shipped formulation qualifies. Until one exists, reverse-bias and dark-J
studies on defect-bearing interfaces under-report generation by about
8.4 A/m^2 on this stack -- roughly 3.6 % of its 235.06 A/m^2 absorbed
photon budget, which is not negligible for a dark-current study and is
negligible for the illuminated figures of merit.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from perovskite_sim.constants import Q
from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.experiments.jv_sweep import _layer_node_counts, _state_fields
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.scaps_compat import load_scaps_yaml
from perovskite_sim.solver.mol import (
    _apply_interface_recombination, build_material_arrays, run_transient,
)
from perovskite_sim.solver.newton import solve_equilibrium

_CONFIG = "configs/scaps_mirror_v2.yaml"
_V_REVERSE = -0.5
_N_GRID = 30
_SETTLE = 1.0e-4

# Measured 2026-07-28 with the clamp disabled; the physical generation the
# default suppresses. Asserted with generous margin -- the claim is "of this
# order and unambiguously negative", not a pinned value.
_EXPECTED_GEN = 8.3644


def _reverse_bias_state(stack):
    x = multilayer_grid([
        Layer(l.thickness, n) for l, n in
        zip(electrical_layers(stack), _layer_node_counts(stack, _N_GRID))
    ])
    mat = build_material_arrays(x, stack)
    y0 = solve_equilibrium(x, stack)
    sol = run_transient(
        x, y0, (0.0, _SETTLE), np.array([_SETTLE]), stack,
        illuminated=False, V_app=_V_REVERSE, rtol=1e-4, atol=1e-6, mat=mat,
    )
    assert sol.success, f"dark reverse-bias settle failed: {sol.message}"
    n, p, phi, _ = _state_fields(x, sol.y[:, -1], stack, _V_REVERSE, mat)
    return x, mat, n, p, phi


def _interface_currents(stack, x, mat, n, p, phi):
    """Areal interface current per interface node [A/m^2].

    Negative means the interface term is a net carrier SOURCE, i.e. physical
    depletion-region generation. Calls the solver's OWN sink on zeroed
    derivative buffers, so whichever formulation is active is the one
    measured.
    """
    dn = np.zeros_like(n)
    dp = np.zeros_like(p)
    _apply_interface_recombination(dn, dp, n, p, stack, mat, phi)
    return [float(-dn[i] * mat.dx_cell[i] * Q) for i in mat.interface_nodes]


@pytest.fixture(scope="module")
def base_stack():
    return load_scaps_yaml(_CONFIG)


@pytest.fixture(scope="module")
def base_state(base_stack):
    return _reverse_bias_state(base_stack)


# ---------------------------------------------------------------------------
# the physics is present, and the default suppresses it
# ---------------------------------------------------------------------------

def test_generation_appears_when_the_clamp_is_lifted(
    base_stack, base_state, monkeypatch,
):
    """With the clamp off, the defect interface is a net carrier SOURCE.

    This is the positive control: it establishes that the reverse-bias
    generation the review asks about genuinely exists in the state, so a
    zero elsewhere in this file is a suppression rather than an absence.
    """
    monkeypatch.setenv("SOLARLAB_IFACE_ALLOW_GEN", "1")
    x, mat, n, p, phi = base_state
    J = _interface_currents(base_stack, x, mat, n, p, phi)
    assert min(J) < -1.0, (
        f"no interface generation with the clamp lifted: {J}"
    )
    assert min(J) == pytest.approx(-_EXPECTED_GEN, rel=0.5), (
        f"generation magnitude moved far from the measured "
        f"{_EXPECTED_GEN} A/m^2: {J}"
    )


def test_default_suppresses_it_entirely(base_stack, base_state):
    """The shipped default returns exactly zero at both defect interfaces.

    Pinned as a KNOWN LIMITATION, not as desired behaviour -- see the module
    docstring and the manual's formulation-limitations chapter.
    """
    x, mat, n, p, phi = base_state
    J = _interface_currents(base_stack, x, mat, n, p, phi)
    assert all(j >= 0.0 for j in J), (
        f"default produced interface generation: {J}. If this is now "
        "intended, the F-04 limitation text in the manual is stale."
    )


# ---------------------------------------------------------------------------
# the review's prescribed remedy does not restore it
# ---------------------------------------------------------------------------

def test_qss_plane_rate_is_also_non_negative(
    base_stack, base_state, monkeypatch,
):
    """The QSS interface-plane rate cannot represent generation either.

    Not because it is clamped -- because it solves for a depletion
    delta >= 0. Promoting it to the default would lock the absence in.
    """
    monkeypatch.setenv("SOLARLAB_IFACE_QSS", "1")
    x, mat, n, p, phi = base_state
    J = _interface_currents(base_stack, x, mat, n, p, phi)
    assert all(j >= 0.0 for j in J), f"QSS produced generation: {J}"


def test_plane_closure_is_also_non_negative(base_state):
    """Same for the interface-plane closure formulation."""
    stack = dataclasses.replace(
        load_scaps_yaml(_CONFIG), interface_plane_closure=True,
    )
    x, mat, n, p, phi = base_state
    J = _interface_currents(stack, x, mat, n, p, phi)
    assert all(j >= 0.0 for j in J), f"plane closure produced generation: {J}"


# ---------------------------------------------------------------------------
# the clamp really is scoped to declared-defect interfaces
# ---------------------------------------------------------------------------

def test_defect_free_interfaces_keep_their_generation(base_state):
    """Strip the InterfaceDefects and the same stack regains generation.

    This is what the E9.3 defect-scoping bought: the clamp targets the
    cross-carrier ni_eff^2 reference, which only exists where a defect
    declared cross-carrier evaluation nodes. Same geometry, same bias, same
    state -- only the defect declaration differs.
    """
    stack = load_scaps_yaml(_CONFIG)
    stripped = dataclasses.replace(
        stack, interface_defects=tuple(None for _ in stack.interface_defects),
    )
    x, mat, n, p, phi = base_state
    mat_stripped = build_material_arrays(x, stripped)
    J = _interface_currents(stripped, x, mat_stripped, n, p, phi)
    assert min(J) < 0.0, (
        "defect-free interfaces no longer produce depletion generation -- "
        f"the E9.3 clamp scoping has regressed: {J}"
    )
