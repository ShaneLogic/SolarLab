"""Flat-band METAL contact reservoir floor (DeviceStack.flat_band_metal_contacts).

The contact carrier reservoir is max(doping-equilibrium, metal-work-function
density N_C/N_V·exp(-phi_B/V_T)). Dormant (bit-identical) at heavily-doped
contacts; doping-independent metal supply at weakly-doped contacts. Fixes the
low-N_D,ETL contact starvation that leaves V_oc unbracketed. LEGACY forces off.
"""
import dataclasses as dc
import math
from pathlib import Path

import numpy as np
import pytest

from perovskite_sim.scaps_compat import load_scaps_yaml
from perovskite_sim.solver.mol import build_material_arrays
from perovskite_sim.sweeps.device_parameter_sweep import SweepPoint, apply_sweep_point

CFG = Path(__file__).resolve().parents[3] / "configs" / "scaps_mirror_v2.yaml"
V_T_300 = 1.380649e-23 * 300.0 / 1.602176634e-19  # ~0.025852 V
PHI_B = 0.42


def _grid(stack):
    L = sum(l.thickness for l in stack.layers if l.role != "substrate")
    return np.linspace(0.0, L, 31)


def _stack(nd_etl_cm3, *, metal=False, phi_B=PHI_B, mode="full"):
    base = load_scaps_yaml(CFG)
    base = dc.replace(base, flat_band_metal_contacts=metal,
                      contact_phi_B_eV=phi_B, mode=mode)
    sp = SweepPoint("p", "Nd_ETL", f"{nd_etl_cm3:.3e}", {"etl_doping_cm3": nd_etl_cm3})
    return apply_sweep_point(base, sp)


def _n_R(stack):
    return float(build_material_arrays(_grid(stack), stack).n_R)


def test_default_off_reservoir_is_doping_derived():
    """Flag off (default): the weakly-doped ETL contact reservoir tracks N_D."""
    n_R = _n_R(_stack(1e11, metal=False))
    # N_D,ETL = 1e11 cm^-3 = 1e17 m^-3 -> ohmic reservoir ~1e17, NOT the ~7e18 floor.
    assert n_R == pytest.approx(1e17, rel=0.05)


def test_flag_on_low_doping_is_floored_to_work_function():
    """Flag on: weakly-doped ETL contact reservoir floored at the metal density."""
    last = load_scaps_yaml(CFG).layers[-1].params
    expected = float(last.Nc300) * math.exp(-PHI_B / V_T_300)   # ~7.0e18
    n_R = _n_R(_stack(1e11, metal=True))
    assert n_R == pytest.approx(expected, rel=0.02)
    assert n_R > 10 * 1e17            # clearly lifted above the 1e17 doping value


def test_flag_on_high_doping_is_dormant_bit_identical():
    """At base ETL doping the floor is dormant -> reservoir == flag-off value."""
    off = _n_R(_stack(1e18, metal=False))
    on = _n_R(_stack(1e18, metal=True))
    assert on == off                  # bit-identical (max picks the doping term)
    assert on == pytest.approx(1e24, rel=0.05)


def test_legacy_mode_forces_floor_off():
    """LEGACY tier disables the floor even with the flag + low doping."""
    n_R = _n_R(_stack(1e11, metal=True, mode="legacy"))
    assert n_R == pytest.approx(1e17, rel=0.05)   # doping-derived, floor off


def test_phi_B_zero_gives_full_dos_floor():
    """phi_B = 0 -> the floor is the full effective DOS N_C."""
    last = load_scaps_yaml(CFG).layers[-1].params
    n_R = _n_R(_stack(1e10, metal=True, phi_B=0.0))
    assert n_R == pytest.approx(float(last.Nc300), rel=0.02)


def test_left_contact_floors_hole_reservoir_doping_sign_pin_safe():
    """Doping-sign rule: a weakly-doped p-type contact floors the HOLE
    reservoir (not electrons) — the pin-safe majority-carrier selection."""
    base = load_scaps_yaml(CFG)
    htl_i = next(i for i, l in enumerate(base.layers) if l.role == "HTL")
    htl = base.layers[htl_i].params
    assert htl.Nv300 and htl.N_A > htl.N_D          # p-type contact with DOS
    base = dc.replace(base, flat_band_metal_contacts=True, contact_phi_B_eV=PHI_B)
    layers = list(base.layers)
    layers[htl_i] = dc.replace(layers[htl_i], params=dc.replace(htl, N_A=1e17))  # weakly doped
    mat = build_material_arrays(_grid(base), dc.replace(base, layers=tuple(layers)))
    expected_p = float(htl.Nv300) * math.exp(-PHI_B / V_T_300)
    assert float(mat.p_L) == pytest.approx(expected_p, rel=0.02)          # hole reservoir floored
    assert float(mat.n_L) == pytest.approx(htl.ni ** 2 / expected_p, rel=0.05)
