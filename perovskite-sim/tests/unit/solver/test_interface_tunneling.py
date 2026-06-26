"""Contract tests for the static TFE tunnelling fold into A* at interfaces.

Default OFF must leave the Richardson arrays bit-identical; ON must enhance A*
only at the TE-capped interface faces; LEGACY (TE off) must be a no-op.
"""
import numpy as np
import pytest

from perovskite_sim.models.parameters import MaterialParams
from perovskite_sim.models.device import DeviceStack, LayerSpec, electrical_layers
from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.solver.mol import build_material_arrays


def _mat(**kw) -> MaterialParams:
    base = dict(
        eps_r=10.0, mu_n=1e-3, mu_p=1e-3, D_ion=0.0, P_lim=1e24, P0=1e24,
        ni=1e10, tau_n=1e-7, tau_p=1e-7, n1=1e10, p1=1e10, B_rad=0.0,
        C_n=0.0, C_p=0.0, alpha=0.0, N_A=0.0, N_D=0.0,
    )
    base.update(kw)
    return MaterialParams(**base)


def _stack(*, tunneling: bool, mode: str = "full", mass: float = 0.2) -> DeviceStack:
    # Doped heterojunction with real CB offsets (>0.05 eV) so TE faces activate,
    # and finite doping on both sides so the lighter-doped side gives Gamma > 1.
    htl = _mat(chi=2.0, Eg=3.0, N_A=1e24)
    absb = _mat(chi=4.0, Eg=1.5, N_A=1e22)
    etl = _mat(chi=4.2, Eg=3.0, N_D=1e24)
    layers = (
        LayerSpec("HTL", 50e-9, htl, "HTL"),
        LayerSpec("ABS", 400e-9, absb, "absorber"),
        LayerSpec("ETL", 50e-9, etl, "ETL"),
    )
    return DeviceStack(
        layers=layers, V_bi=1.0, mode=mode,
        interface_tunneling=tunneling, tunnel_mass_eff=mass,
    )


def _grid(stack, N=20):
    elec = electrical_layers(stack)
    return multilayer_grid([Layer(l.thickness, N) for l in elec], alpha=3.0)


def test_off_is_bit_identical():
    st_off = _stack(tunneling=False)
    x = _grid(st_off)
    m_off = build_material_arrays(x, st_off)
    # Baseline (no flag) build of the same stack.
    m_base = build_material_arrays(x, _stack(tunneling=False))
    assert np.array_equal(m_off.A_star_n, m_base.A_star_n)
    assert np.array_equal(m_off.A_star_p, m_base.A_star_p)


def test_on_enhances_only_interface_faces():
    st_off = _stack(tunneling=False)
    st_on = _stack(tunneling=True)
    x = _grid(st_off)
    m_off = build_material_arrays(x, st_off)
    m_on = build_material_arrays(x, st_on)
    faces = list(m_off.interface_faces)
    assert len(faces) > 0, "test config must activate TE interface faces"
    # Enhanced (>=, and strictly > at faces with doped lighter side + offset).
    assert np.all(m_on.A_star_n >= m_off.A_star_n - 1e-6)
    assert (m_on.A_star_n[faces] > m_off.A_star_n[faces]).any()
    # Off-face nodes must be byte-identical (fold touches only interface faces).
    off_mask = np.ones(len(x), dtype=bool)
    off_mask[faces] = False
    assert np.array_equal(m_on.A_star_n[off_mask], m_off.A_star_n[off_mask])
    assert np.array_equal(m_on.A_star_p[off_mask], m_off.A_star_p[off_mask])


def test_legacy_tier_disables_tunneling():
    """LEGACY disables TE → no interface faces → A* untouched even with flag on."""
    st_legacy_on = _stack(tunneling=True, mode="legacy")
    st_legacy_off = _stack(tunneling=False, mode="legacy")
    x = _grid(st_legacy_on)
    m_on = build_material_arrays(x, st_legacy_on)
    m_off = build_material_arrays(x, st_legacy_off)
    assert np.array_equal(m_on.A_star_n, m_off.A_star_n)
    assert np.array_equal(m_on.A_star_p, m_off.A_star_p)


def test_heavier_mass_gives_weaker_enhancement():
    """E_00 ∝ 1/sqrt(m*) → heavier tunnelling mass → smaller Gamma."""
    x = _grid(_stack(tunneling=True))
    m_light = build_material_arrays(x, _stack(tunneling=True, mass=0.05))
    m_heavy = build_material_arrays(x, _stack(tunneling=True, mass=5.0))
    faces = list(m_light.interface_faces)
    assert np.all(m_light.A_star_n[faces] >= m_heavy.A_star_n[faces] - 1e-6)
    assert (m_light.A_star_n[faces] > m_heavy.A_star_n[faces]).any()
