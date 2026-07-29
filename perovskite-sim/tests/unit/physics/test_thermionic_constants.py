"""Richardson constants must be consistent with the effective DOS.

``A*`` and ``N`` are both fixed by one effective mass, so the emission
velocity the thermionic bound is built from collapses to a thermal
velocity::

    A*  = 4 pi q m* k_B^2 / h^3
    N   = 2 (2 pi m* k_B T / h^2)^{3/2}
    =>  v_R = A* T^2 / (q N) = sqrt(k_B T / 2 pi m*)

Shipped configs left ``A_star_n``/``A_star_p`` at the free-electron default
while ``Nc300``/``Nv300`` came from the material parameter set, so ``v_R``
was a ratio of unrelated constants — measured 4.63x / 0.54x / 2.17x adrift
on the HTL / absorber / ETL of the SCAPS mirror stack, in different
directions per layer.

``build_material_arrays`` now derives ``A*`` from each layer's own DOS when
the layer declares one and the Richardson constant is still the untouched
default. That is what makes ``v_R`` a real emission velocity, and it is what
unblocked the physically normalized thermionic bound: before the fix,
enabling ``te_physical_norm`` on a near-insulating contact drove the solve
to non-finite densities; after it the same sweep completes.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from perovskite_sim.constants import A_STAR_FREE_ELECTRON, Q, T as T_REF
from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.experiments.jv_sweep import _layer_node_counts
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.physics.thermionic_constants import (
    effective_mass_from_dos, emission_velocity, is_free_electron_default,
    richardson_from_dos,
)
from perovskite_sim.scaps_compat import load_scaps_yaml
from perovskite_sim.solver.mol import build_material_arrays


def _mat(stack, N_grid=30):
    x = multilayer_grid([
        Layer(l.thickness, n) for l, n in
        zip(electrical_layers(stack), _layer_node_counts(stack, N_grid))
    ])
    return x, build_material_arrays(x, stack)


# ---------------------------------------------------------------------------
# the identity the fix rests on
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N_dos", [1e24, 1e25, 8e25, 2.5e26, 1e27])
def test_derived_richardson_makes_v_r_the_thermal_velocity(N_dos):
    """v_R = A*T^2/(qN) must equal sqrt(kT/2 pi m*) to round-off."""
    a = richardson_from_dos(N_dos)
    v_r = a * T_REF ** 2 / (Q * N_dos)
    assert v_r == pytest.approx(emission_velocity(N_dos), rel=1e-12)


@pytest.mark.parametrize("N_dos", [1e24, 8e25, 1e27])
def test_effective_mass_round_trips(N_dos):
    """m* -> N -> m* is the identity, so the inversion is not a fit."""
    m = effective_mass_from_dos(N_dos)
    h = 6.62607015e-34
    k = 1.380649e-23
    N_back = 2.0 * (2.0 * math.pi * m * k * T_REF / (h * h)) ** 1.5
    assert N_back == pytest.approx(N_dos, rel=1e-12)


def test_free_electron_dos_recovers_the_free_electron_constant():
    """Feeding the free-electron m* back must return the classic value."""
    h, k = 6.62607015e-34, 1.380649e-23
    m_e = 9.1093837015e-31
    N_free = 2.0 * (2.0 * math.pi * m_e * k * T_REF / (h * h)) ** 1.5
    assert richardson_from_dos(N_free) == pytest.approx(
        A_STAR_FREE_ELECTRON, rel=1e-12
    )


def test_dos_must_be_positive():
    for bad in (0.0, -1e25):
        with pytest.raises(ValueError):
            effective_mass_from_dos(bad)


# ---------------------------------------------------------------------------
# the default sentinel
# ---------------------------------------------------------------------------

def test_recognises_the_rounded_literal_default():
    """MaterialParams stores 1.2017e6; the exact value is 1.201732e6.

    The sentinel has to absorb that rounding, or the fix silently never
    fires.
    """
    assert is_free_electron_default(1.2017e6)
    assert is_free_electron_default(A_STAR_FREE_ELECTRON)


def test_does_not_mistake_a_configured_constant_for_the_default():
    """A user-set A* must survive. The smallest real departure measured on
    the SCAPS stack is 0.54x, three orders outside the tolerance."""
    for configured in (0.54 * 1.2017e6, 2.17 * 1.2017e6, 4.63 * 1.2017e6):
        assert not is_free_electron_default(configured)


# ---------------------------------------------------------------------------
# what the solver actually builds
# ---------------------------------------------------------------------------

def test_dos_bearing_layers_get_their_own_richardson_constant():
    """Each layer's A* must match what its own DOS implies."""
    stack = load_scaps_yaml("configs/scaps_mirror_v2.yaml")
    x, mat = _mat(stack)
    checked = 0
    offset = 0.0
    for lay in electrical_layers(stack):
        p = lay.params
        span = (x >= offset - 1e-15) & (x <= offset + lay.thickness + 1e-15)
        offset += lay.thickness
        if not p.Nc300:
            continue
        want = richardson_from_dos(float(p.Nc300))
        got = float(np.median(mat.A_star_n[span]))
        assert got == pytest.approx(want, rel=1e-9), (
            f"{lay.name}: A*_n = {got:.4e}, expected {want:.4e}"
        )
        checked += 1
    assert checked >= 3, f"only {checked} layers carried a DOS"


def test_configs_without_dos_keep_the_free_electron_value():
    """No DOS declared, nothing to derive from — must not change."""
    stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    _, mat = _mat(stack)
    assert np.allclose(mat.A_star_n, A_STAR_FREE_ELECTRON, rtol=1e-3)
    assert np.allclose(mat.A_star_p, A_STAR_FREE_ELECTRON, rtol=1e-3)


def test_an_explicit_richardson_constant_is_not_overridden():
    """A configured A* is a statement about the interface; it wins.

    Set deliberately to a value no DOS would imply, on a stack that HAS a
    DOS, and confirm the derivation stands down.
    """
    stack = load_scaps_yaml("configs/scaps_mirror_v2.yaml")
    sentinel = 3.3e5
    layers = tuple(
        dataclasses.replace(l, params=dataclasses.replace(
            l.params, A_star_n=sentinel, A_star_p=sentinel))
        for l in stack.layers
    )
    _, mat = _mat(dataclasses.replace(stack, layers=layers))
    assert np.allclose(mat.A_star_n, sentinel, rtol=1e-9), (
        "an explicitly configured Richardson constant was overwritten by the "
        "DOS derivation"
    )


def test_the_correction_is_material_not_cosmetic():
    """The derived constants must actually differ from the default.

    Guards against the fix quietly never firing — if every layer came back
    at the free-electron value, every other test here would still pass.
    """
    stack = load_scaps_yaml("configs/scaps_mirror_v2.yaml")
    _, mat = _mat(stack)
    ratios = np.asarray(mat.A_star_n, dtype=float) / A_STAR_FREE_ELECTRON
    assert ratios.max() > 2.0 and ratios.min() < 0.8, (
        f"derived Richardson constants span {ratios.min():.2f}x to "
        f"{ratios.max():.2f}x of the free-electron value; expected the "
        "measured 0.54x-4.63x spread"
    )
