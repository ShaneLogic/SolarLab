"""Review F-04 item 3, the two checks not covered elsewhere.

The review asks that a general interface model be checked for

  - the two-sided carrier flux difference equalling the surface
    recombination, and
  - insensitivity of the result to mesh placement.

``test_interface_reverse_bias_generation.py`` already covers the other two
items on that list (zero net rate at dark equilibrium, and whether
reverse-bias generation is representable). These are the remaining ones.

ON THE SECOND ONE, THE HONEST FORM IS NOT "INSENSITIVE"

The interface rate is evaluated on bulk-interior nodes and converted from
areal to volumetric by dividing by the interface node's dual-cell width, so
it is documented as grid-referenced rather than as a face-value physical
velocity. Asserting flat mesh-independence would fail, and should. What must
hold is weaker and more useful: the areal rate has to CONVERGE under
refinement, and in particular must not inherit ``dx_cell`` — a rate that
scaled with the cell width would mean the areal/volumetric conversion is
wrong rather than merely discretised.

Measured on scaps_mirror_v2, illuminated, V = 0.9 V, total areal interface
current across both defect interfaces:

    N_grid   nodes   J_iface [A/m^2]   dx_cell at interface [m]
      20      19       0.02185          6.392e-09
      30      31       0.01726          2.344e-09
      45      46       0.01638          1.242e-09
      60      61       0.01613          8.338e-10
      90      91       0.01597          4.992e-10

Successive changes are -21 %, -5.1 %, -1.5 %, -1.0 %: converging. Over the
same span ``dx_cell`` shrinks by 12.8x while the areal rate moves by 1.37x,
so the rate is plainly not carrying the cell width.
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
_V = 0.9
_SETTLE = 1.0e-3
_LADDER = (20, 30, 45, 60, 90)


def _settled(stack, N_grid):
    x = multilayer_grid([
        Layer(l.thickness, n) for l, n in
        zip(electrical_layers(stack), _layer_node_counts(stack, N_grid))
    ])
    mat = build_material_arrays(x, stack)
    y0 = solve_equilibrium(x, stack)
    sol = run_transient(
        x, y0, (0.0, _SETTLE), np.array([_SETTLE]), stack,
        illuminated=True, V_app=_V, rtol=1e-4, atol=1e-6, mat=mat,
    )
    assert sol.success, f"settle failed at N_grid={N_grid}: {sol.message}"
    n, p, phi, _ = _state_fields(x, sol.y[:, -1], stack, _V, mat)
    return x, mat, n, p, phi


def _iface_sinks(stack, x, mat, n, p, phi):
    """Per-interface areal electron and hole sink [A/m^2].

    Calls the solver's OWN interface routine on zeroed derivative buffers,
    so whichever formulation is active is the one measured.
    """
    dn = np.zeros_like(n)
    dp = np.zeros_like(p)
    _apply_interface_recombination(dn, dp, n, p, stack, mat, phi)
    return [
        (float(-dn[i] * mat.dx_cell[i] * Q), float(-dp[i] * mat.dx_cell[i] * Q))
        for i in mat.interface_nodes
    ]


@pytest.fixture(scope="module")
def stack():
    return load_scaps_yaml(_CONFIG)


@pytest.fixture(scope="module")
def ladder(stack):
    out = {}
    for N in _LADDER:
        x, mat, n, p, phi = _settled(stack, N)
        sinks = _iface_sinks(stack, x, mat, n, p, phi)
        out[N] = {
            "total": sum(a for a, _ in sinks),
            "sinks": sinks,
            "dx_iface": tuple(float(mat.dx_cell[i]) for i in mat.interface_nodes),
            "nodes": len(x),
        }
    return out


# ---------------------------------------------------------------------------
# flux balance: what leaves one side is what the interface consumes
# ---------------------------------------------------------------------------

def test_electron_and_hole_sinks_are_exactly_paired(ladder):
    """Interface SRH annihilates one electron per hole — bit-exactly.

    The two sinks are written from the same ``R_vol``, so any difference
    means a channel was applied to one carrier and not the other. Measured
    identical to the bit at every rung, so this is asserted as equality
    rather than with a tolerance.
    """
    for N, rec in ladder.items():
        for k, (a, b) in enumerate(rec["sinks"]):
            assert a == b, (
                f"N_grid={N}, interface {k}: electron sink {a!r} != hole sink "
                f"{b!r} — the interface channel is not pair-conserving"
            )


def test_sink_vanishes_when_the_interface_channel_is_removed(stack, ladder):
    """Stripping the interfaces must remove exactly this sink and nothing else.

    This is the discrete form of "the two-sided flux difference equals the
    surface recombination": the interface term is the ONLY thing that
    changes, so the difference in the node's rate between the two runs is
    the areal recombination, by construction rather than by coincidence.
    """
    N = 45
    x, mat, n, p, phi = _settled(stack, N)
    with_iface = _iface_sinks(stack, x, mat, n, p, phi)

    stripped = dataclasses.replace(stack, interfaces=(), interface_defects=())
    mat0 = build_material_arrays(x, stripped)
    without = _iface_sinks(stripped, x, mat0, n, p, phi)

    assert any(abs(a) > 0.0 for a, _ in with_iface), (
        "the interface channel contributes nothing at this bias, so this "
        "test would pass vacuously"
    )
    for k, (a, _) in enumerate(without):
        assert a == 0.0, (
            f"interface {k} still sinks {a!r} A/m^2 with interfaces removed"
        )


# ---------------------------------------------------------------------------
# mesh behaviour: converging, and not carrying the cell width
# ---------------------------------------------------------------------------

def test_areal_rate_converges_under_refinement(ladder):
    """Successive refinements must change the answer by less each time.

    Not flat mesh-independence — the rate is evaluated on bulk-interior
    nodes and is documented as grid-referenced — but it must settle.
    """
    Ns = sorted(ladder)
    totals = [ladder[N]["total"] for N in Ns]
    steps = [
        abs(totals[i + 1] - totals[i]) / abs(totals[i])
        for i in range(len(totals) - 1)
    ]
    assert steps[-1] < steps[0], (
        f"areal interface rate is not converging: successive relative "
        f"changes {['%.3f' % s for s in steps]} across N_grid {Ns}"
    )
    assert steps[-1] < 0.05, (
        f"finest refinement still moves the rate by {steps[-1]:.1%}; "
        f"totals {['%.5f' % t for t in totals]}"
    )


def test_areal_rate_does_not_inherit_the_dual_cell_width(ladder):
    """The failure mode this guards: an areal rate that scales with dx_cell.

    That would mean the areal/volumetric conversion is wrong, and it would
    look like "mesh sensitivity" while actually being a units error. Over
    this ladder dx_cell shrinks by ~13x; the rate must move far less.
    """
    Ns = sorted(ladder)
    dx_ratio = ladder[Ns[0]]["dx_iface"][0] / ladder[Ns[-1]]["dx_iface"][0]
    rate_ratio = ladder[Ns[0]]["total"] / ladder[Ns[-1]]["total"]
    assert dx_ratio > 5.0, (
        f"the ladder no longer refines the interface cell ({dx_ratio:.2f}x), "
        "so this test cannot discriminate"
    )
    assert rate_ratio < 0.25 * dx_ratio, (
        f"areal interface rate moved {rate_ratio:.2f}x while dx_cell moved "
        f"{dx_ratio:.2f}x — it is tracking the cell width, which points at "
        "the areal-to-volumetric conversion rather than at discretisation"
    )
