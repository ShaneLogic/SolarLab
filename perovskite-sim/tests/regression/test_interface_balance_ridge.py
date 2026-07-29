"""The band-alignment optimum is set by the BALANCE between the two
hetero-interfaces, not by their absolute strength.

Stage 2c's result is that the absorber acceptance region is a tilted band
tracking a constant work function chi + Eg/2. Both hetero-interfaces in that
scan carried the same surface-recombination velocity, which forces the two
competing loss arms to share a prefactor -- so the tilt could in principle have
been an artifact of that assumed symmetry rather than a property of the
absorber. Scaling both interfaces together cannot settle it: that leaves the
balance point alone by construction. Breaking the symmetry can.

Two things are pinned here, and they fail for different reasons:

* ``test_ridge_survives_a_ten_fold_interface_imbalance`` is the claim Stage 2c
  rests on. Making one interface 10x more recombination-active than the other
  moves the optimum by at most 0.048 eV, against a ridge drift of 0.164 eV
  across the accepted band-gap span -- under a third of the effect. If a change
  to the interface formulation ever makes the optimum strongly SRV-dependent,
  the tilted-band result and the screening criterion derived from it stop being
  properties of the absorber, and this test says so.
* ``test_imbalance_moves_the_ridge_in_the_expected_direction`` checks the sign,
  which a magnitude bound alone would miss. Loading the HTL side pushes the
  optimum away from it (chi down); loading the ETL side pushes it the other way
  (chi up). A formulation that moved the ridge the wrong way would pass a
  bound and fail here.

Absolute efficiency is a different matter and is deliberately not pinned: it
does depend on the SRV (19.03 % symmetric, ~18.6 % imbalanced). The separation
between "geometry is robust" and "absolutes are conditioned" is the point.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from perovskite_sim.constants import V_T
from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.experiments.jv_sweep import (
    _grid_node_count, _layer_node_counts, run_jv_sweep)
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import InterfaceDefect, electrical_layers

pytestmark = pytest.mark.slow

CFG = "configs/solarscale_nip_band_aligned.yaml"
#: Stage 2c operating point and numerics.
EG, TAU, N_GRID = 1.40, 1e-6, 60
A0 = 0.839              # absorptance anchor (optics-only)
E_T_FRAC = 0.5          # mid-gap trap
DV, V_MAX = 0.025, 1.2  # fixed voltage resolution; 1.2 V brackets V_oc here
#: chi grid across the ridge, predicted at 4.50 - Eg/2 = 3.80.
CHIS = tuple(round(3.725 + 0.025 * i, 4) for i in range(7))


def _above_gap_flux(Eg: float) -> float:
    lam, flux = [], []
    with open("perovskite_sim/data/am15g.csv") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                a, b = line.split(",")
                lam.append(float(a)); flux.append(float(b))
    lam, flux = np.array(lam), np.array(flux)
    order = np.argsort(lam)
    lam, flux = lam[order], flux[order]
    m = lam <= 1239.841984 / Eg
    return float(np.trapezoid(flux[m], lam[m] * 1e-9))


def _pce(base, chi: float, v_htl: float, v_etl: float) -> float:
    """PCE (%) for one alignment and one pair of interface velocities."""
    ni = base.layers[_ABS].params.ni * math.exp(
        -(EG - base.layers[_ABS].params.Eg) / (2.0 * V_T))
    params = dataclasses.replace(
        base.layers[_ABS].params, Eg=EG, chi=chi, tau_n=TAU, tau_p=TAU,
        ni=ni, n1=ni, p1=ni)
    layers = list(base.layers)
    layers[_ABS] = dataclasses.replace(layers[_ABS], params=params)
    lead = len(base.layers) - 1 - 2          # substrate-side slots carry no defect
    defect = InterfaceDefect(E_t_eV=E_T_FRAC * EG)
    stack = dataclasses.replace(
        base, layers=tuple(layers),
        interfaces=tuple([(0.0, 0.0)] * lead
                         + [(v_htl, v_htl), (v_etl, v_etl)]),
        interface_defects=tuple([None] * lead + [defect, defect]))

    elec = electrical_layers(stack)
    x = multilayer_grid([Layer(l.thickness, n) for l, n
                         in zip(elec, _layer_node_counts(stack, N_GRID))])
    ai = next(i for i, l in enumerate(elec) if l.role == "absorber")
    x0 = sum(l.thickness for l in elec[:ai])
    d_abs = elec[ai].thickness
    G = np.zeros(_grid_node_count(stack, N_GRID))
    G[(x >= x0 - 1e-12) & (x <= x0 + d_abs + 1e-12)] = _FLUX / d_abs

    r = run_jv_sweep(stack, N_grid=N_GRID, n_points=int(round(V_MAX / DV)) + 1,
                     v_rate=0.5, fixed_generation=G, V_max=V_MAX,
                     v_max_max_attempts=2)
    m = r.metrics_fwd
    assert m.voc_bracketed, f"chi={chi} v=({v_htl},{v_etl}) did not bracket V_oc"
    return m.V_oc * (A0 * m.J_sc) * m.FF / 1000.0 * 100.0


def _ridge(base, v_htl: float, v_etl: float) -> float:
    """chi at the PCE maximum, refined off the 0.025 eV grid.

    The peak is flat enough that plain argmax quantises the answer to the grid
    step; parabolic interpolation through the maximum and its two neighbours
    recovers it, and the shifts under test are of order one grid step.
    """
    pce = [_pce(base, chi, v_htl, v_etl) for chi in CHIS]
    k = int(np.argmax(pce))
    assert 0 < k < len(pce) - 1, (
        f"peak at the edge of the chi scan for v=({v_htl},{v_etl}); "
        "the scan window no longer contains the ridge")
    den = pce[k - 1] - 2 * pce[k] + pce[k + 1]
    if den == 0:
        return CHIS[k]
    return CHIS[k] + 0.5 * (pce[k - 1] - pce[k + 1]) / den * (CHIS[1] - CHIS[0])


_BASE_STACK = load_device_from_yaml(CFG)
_ABS = next(i for i, l in enumerate(_BASE_STACK.layers) if l.role == "absorber")
_FLUX = _above_gap_flux(EG)


@pytest.fixture(scope="module")
def ridges():
    """Ridge position under symmetric and 10x-imbalanced interfaces."""
    base = dataclasses.replace(_BASE_STACK, layers=tuple(
        dataclasses.replace(l, params=(dataclasses.replace(l.params, D_ion=0.0)
                                       if l.params else None))
        for l in _BASE_STACK.layers))
    return {
        "symmetric": _ridge(base, 0.1, 0.1),
        "htl_loaded": _ridge(base, 1.0, 0.1),
        "etl_loaded": _ridge(base, 0.1, 1.0),
    }


def test_ridge_survives_a_ten_fold_interface_imbalance(ridges):
    """A 10x imbalance moves the optimum far less than the effect it underpins.

    Measured 2026-07-29: symmetric chi* = 3.8075; HTL-loaded 3.7780
    (-0.029 eV); ETL-loaded 3.8559 (+0.048 eV). The ridge drifts 0.164 eV in
    chi across the accepted Eg span, so the worst imbalance costs under a third
    of that. The bound is 0.08 eV -- clear of the measurement (0.048) but well
    inside the 0.164 eV effect it protects.
    """
    base = ridges["symmetric"]
    worst = max(abs(ridges[k] - base) for k in ("htl_loaded", "etl_loaded"))
    assert worst < 0.08, (
        f"a 10x interface imbalance moved the ridge by {worst:.4f} eV "
        f"(was 0.048). Approaching the 0.164 eV ridge drift across the Eg "
        "span would mean the tilted band is a property of the assumed "
        "interface symmetry, not of the absorber -- which is what Stage 2c "
        "and the screening criterion built on it assume."
    )


def test_imbalance_moves_the_ridge_in_the_expected_direction(ridges):
    """Loading one interface pushes the optimum away from it.

    Making the HTL side more recombination-active favours an absorber sitting
    lower in chi (less carrier density at that interface); loading the ETL side
    does the reverse. A magnitude bound alone would accept a formulation that
    moved the ridge the wrong way, so the sign is pinned separately -- and it
    is what identifies the mechanism as interface balance rather than a generic
    sensitivity to the SRV value.
    """
    base = ridges["symmetric"]
    assert ridges["htl_loaded"] < base, (
        f"loading the HTL side moved the ridge UP ({ridges['htl_loaded']:.4f} "
        f"vs {base:.4f}); it should move down, away from the loaded interface"
    )
    assert ridges["etl_loaded"] > base, (
        f"loading the ETL side moved the ridge DOWN ({ridges['etl_loaded']:.4f} "
        f"vs {base:.4f}); it should move up, away from the loaded interface"
    )
