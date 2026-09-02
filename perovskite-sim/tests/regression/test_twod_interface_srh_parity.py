"""TWOD-E1: the missing 1D<->2D parity gate for interface SRH.

Five 1D/2D parity gates exist (`test_twod_validation.py`) and every one runs
on an interface-FREE config. The 2D interface-SRH channel has its own
registered lane (`twod-mobile-ion-interface-srh-v1`) and four test files, but
nothing had ever compared it against the 1D channel it shares a rate primitive
with. This file is that comparison.

What it found, and what it did not
----------------------------------
The expectation going in was divergence, on two formulation differences that
are real in the source: 2D floors both SRH pairs unconditionally
(`interface_recombination_2d.py`, `np.maximum(raw, 0.0)`), while 1D clamps
pair A only when the escape hatch is unset AND the interface carries a
declared defect AND the rate is negative (`mol.py`, `if nogen and eval_n_idx
!= idx and R_s < 0.0`).

Measured, the two agree **exactly** — relative difference 0.0 across six
decades of carrier density, including deep depletion where the raw rate is
negative and both floor. Every configuration in which they *could* differ is
refused by the 2D builder rather than approximated:

* a defect-free interface raises, and that is the only case where 1D's clamp
  condition would be false while 2D's floor still fired;
* the `SOLARLAB_IFACE_ALLOW_GEN` escape hatch raises, and that is the only
  other way to switch 1D's clamp off.

So the guard set is what makes the agreement safe, and the refusals are
pinned here alongside the agreement for that reason.

The real hazard is the default
------------------------------
`interface_srh` defaults to `"off"`. On a config that declares
`device.interfaces`, a 2D run therefore drops the channel silently — measured
below as 24 orders of magnitude at the interface node. That is the practical
half of the warning in `perovskite-sim/CLAUDE.md`, and it survives even though
the absolute claim there ("2D CANNOT USE THIS CHANNEL AT ALL") does not.
"""

from __future__ import annotations

import numpy as np
import pytest

from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.models.device import DeviceStack, InterfaceDefect, LayerSpec
from perovskite_sim.models.parameters import MaterialParams
from perovskite_sim.solver.mol import assemble_rhs, build_material_arrays
from perovskite_sim.twod.grid_2d import build_grid_2d
from perovskite_sim.twod.microstructure import Microstructure
from perovskite_sim.twod.solver_2d import assemble_rhs_2d, build_material_arrays_2d


_INTERVALS_PER_LAYER = 8
_LATERAL_NODES = 3
_INTERFACE_NODE = 8  # one interface, at the shared face of the two layers

_MATERIAL = MaterialParams(
    eps_r=10.0,
    mu_n=1.0e-3,
    mu_p=1.0e-3,
    D_ion=0.0,
    P_lim=1.0e24,
    P0=0.0,
    ni=1.0e12,
    tau_n=1.0e30,
    tau_p=1.0e30,
    n1=1.0e12,
    p1=1.0e12,
    B_rad=0.0,
    C_n=0.0,
    C_p=0.0,
    alpha=0.0,
    N_A=0.0,
    N_D=0.0,
    chi=4.0,
    Eg=1.5,
    Nc300=1.0e25,
    Nv300=1.0e25,
)


def _stack(*, with_defect: bool = True) -> DeviceStack:
    """Bulk SRH is switched off (tau = 1e30) so only the interface can act."""

    return DeviceStack(
        layers=(
            LayerSpec("left", 1.0e-7, _MATERIAL, role="absorber"),
            LayerSpec("right", 1.0e-7, _MATERIAL, role="ETL"),
        ),
        interfaces=((0.03, 0.05),),
        interface_defects=(InterfaceDefect(E_t_eV=0.5),) if with_defect else (),
        interface_two_sided=True,
        V_bi=0.0,
        Phi=0.0,
        mode="full",
    )


def _grids(stack: DeviceStack):
    layers = [Layer(layer.thickness, _INTERVALS_PER_LAYER) for layer in stack.layers]
    grid_1d = multilayer_grid(layers, alpha=1.0)
    grid_2d = build_grid_2d(
        layers,
        lateral_length=1.0e-7,
        Nx=_LATERAL_NODES,
        alpha_y=1.0,
        lateral_uniform=True,
    )
    return grid_1d, grid_2d


def _interface_rate_pair(density: float, *, interface_srh: str):
    """Electron RHS at the interface, from each solver's production path.

    Lateral-uniform state, so the Stage-A parity already established that
    every term other than the interface treatment agrees; the difference here
    isolates that treatment.
    """

    stack = _stack()
    grid_1d, grid_2d = _grids(stack)
    count = grid_1d.size

    material_1d = build_material_arrays(grid_1d, stack)
    state_1d = np.concatenate(
        [np.full(count, density), np.full(count, density), np.zeros(count)]
    )
    rate_1d = assemble_rhs(
        0.0, state_1d, grid_1d, stack, material_1d, illuminated=False, V_app=0.0
    )[:count][_INTERFACE_NODE]

    material_2d = build_material_arrays_2d(
        grid_2d,
        stack,
        Microstructure(),
        lateral_bc="neumann",
        interface_srh=interface_srh,
    )
    lateral = _LATERAL_NODES + 1
    state_2d = np.concatenate(
        [np.full(count * lateral, density), np.full(count * lateral, density)]
    )
    rate_2d = assemble_rhs_2d(0.0, state_2d, material_2d, 0.0)[
        : count * lateral
    ].reshape(count, lateral)[_INTERFACE_NODE, 0]

    return float(rate_1d), float(rate_2d)


# --------------------------------------------------------------------------
# The gate that was missing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "density",
    [1.0e18, 1.0e15, 1.0e13, 1.0e12, 3.0e11, 1.0e11],
    ids=["injection", "moderate", "low", "at_ni", "depleted", "deep_depletion"],
)
def test_the_two_solvers_agree_exactly_on_the_interface_rate(density):
    """Exact, not approximate — the rate primitive is literally shared.

    `interface_recombination_2d.py` imports `interface_recombination` from the
    1D `physics/recombination.py`, so on the reachable domain there is one
    rate law and two assemblies of it. The assemblies agree bit for bit; a
    tolerance here would hide an assembly difference behind rounding.

    The last three cases sit at and below `n*p = ni^2`, where the raw rate is
    negative and BOTH sides floor it — the regime the two clamp rules were
    expected to differ in.
    """
    rate_1d, rate_2d = _interface_rate_pair(
        density, interface_srh="two_sided_cross_node"
    )

    assert rate_2d == rate_1d


def test_the_channel_actually_acts_so_the_agreement_is_not_vacuous():
    """Two solvers agreeing on zero would prove nothing."""
    injection, _ = _interface_rate_pair(1.0e18, interface_srh="two_sided_cross_node")
    depleted, _ = _interface_rate_pair(1.0e11, interface_srh="two_sided_cross_node")

    # Strong recombination under injection...
    assert injection < -1.0e20
    # ...and the floored, essentially inert depletion limit.
    assert abs(depleted) < 1.0e-15


# --------------------------------------------------------------------------
# The hazard that survives: the channel is off by default
# --------------------------------------------------------------------------


def test_the_2d_default_silently_drops_a_declared_interface_channel():
    """`interface_srh` defaults to "off", and nothing warns.

    On a stack that declares `device.interfaces` this is a 24-order-of-
    magnitude omission at the interface node, reported by a solver that
    otherwise matches 1D. It is silent because the 2D builder accepts the
    stack without complaint — the interface block is simply not consumed.
    """
    rate_1d, rate_2d_off = _interface_rate_pair(1.0e18, interface_srh="off")

    assert rate_1d < -1.0e20
    assert abs(rate_2d_off) < 1.0e-9
    assert abs(rate_1d) / max(abs(rate_2d_off), 1.0e-300) > 1.0e20


# --------------------------------------------------------------------------
# The refusals that make the agreement safe
# --------------------------------------------------------------------------


def test_a_defect_free_interface_is_refused_rather_than_approximated():
    """The one configuration where the clamp rules would differ.

    1D clamps pair A only at interfaces with a declared defect, because on a
    defect-free interface a negative rate is real depletion generation
    (review F06) and suppressing it both loses physics and rides a
    non-differentiable corner. 2D floors unconditionally — so on a defect-free
    interface the two WOULD diverge. 2D refuses to build there instead.
    """
    stack = _stack(with_defect=False)
    _, grid_2d = _grids(stack)

    with pytest.raises(ValueError, match="requires an InterfaceDefect"):
        build_material_arrays_2d(
            grid_2d,
            stack,
            Microstructure(),
            lateral_bc="neumann",
            interface_srh="two_sided_cross_node",
        )


def test_the_interface_generation_escape_hatch_is_refused(monkeypatch):
    """The only other way to switch the 1D clamp off.

    `SOLARLAB_IFACE_ALLOW_GEN=1` turns off 1D's clamp entirely, which would
    leave 1D reporting negative rates while 2D floors them. 2D refuses rather
    than silently disagreeing.
    """
    monkeypatch.setenv("SOLARLAB_IFACE_ALLOW_GEN", "1")
    stack = _stack()
    _, grid_2d = _grids(stack)

    with pytest.raises(ValueError, match="escape hatch"):
        build_material_arrays_2d(
            grid_2d,
            stack,
            Microstructure(),
            lateral_bc="neumann",
            interface_srh="two_sided_cross_node",
        )


def test_a_periodic_lateral_boundary_is_refused():
    """The channel is declared only for Neumann lateral boundaries.

    Recorded here because it is why no existing periodic parity baseline can
    be reused for this comparison, which is the practical cost of adding it.
    """
    stack = _stack()
    _, grid_2d = _grids(stack)

    with pytest.raises(ValueError):
        build_material_arrays_2d(
            grid_2d,
            stack,
            Microstructure(),
            lateral_bc="periodic",
            interface_srh="two_sided_cross_node",
        )
