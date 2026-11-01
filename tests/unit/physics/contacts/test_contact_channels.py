"""Contact exchange channels and reservoir sources (M class).

Each carrier/side channel independently chooses a fixed reservoir, a
blocking (S = 0) or a finite Robin exchange; the reservoir source (doping,
statistics closure, Schottky barrier, metal floor) is declared separately and
never selects the Poisson potential. The unified boundary particle flux (N
class) is tested in ``test_contact_boundary_flux.py``.
"""

from __future__ import annotations

import math

import pytest

from solarlab.physics.contacts import (
    BLOCKING, FIXED_RESERVOIR, ROBIN, CarrierExchangeChannel, doping_equilibrium_densities,
    exchange_kind, metal_reservoir_floor, schottky_equilibrium_n, schottky_equilibrium_p,
)


def test_each_channel_declares_its_own_exchange_kind():
    assert exchange_kind(None) == FIXED_RESERVOIR
    assert exchange_kind(0.0) == BLOCKING
    assert exchange_kind(1.0e-12) == ROBIN and exchange_kind(1.0e9) == ROBIN
    for bad in (-1.0, math.nan, math.inf):
        with pytest.raises(ValueError, match="finite and non-negative"):
            exchange_kind(bad)
    channel = CarrierExchangeChannel("n", "right", 1.0e3, 1.0e24, "schottky_barrier")
    assert channel.kind == ROBIN and channel.outward_normal == 1.0
    assert CarrierExchangeChannel("p", "left", None, 1.0e24).outward_normal == -1.0
    for bad in (dict(carrier="e"), dict(side="top"), dict(reservoir_source="guess"),
                dict(reservoir_density_m3=-1.0), dict(velocity_m_s=-5.0)):
        args = dict(carrier="n", side="left", velocity_m_s=1.0, reservoir_density_m3=1.0,
                    reservoir_source="doping_equilibrium")
        args.update(bad)
        with pytest.raises(ValueError):
            CarrierExchangeChannel(**args)


def test_schottky_and_metal_reservoirs():
    N_c, V_T = 2.2e24, 0.025852
    assert schottky_equilibrium_n(N_c, 0.0, V_T) == N_c
    assert schottky_equilibrium_n(N_c, 0.5, V_T) == pytest.approx(N_c * math.exp(-0.5 / V_T))
    assert schottky_equilibrium_p(1.8e25, 0.3, V_T) == pytest.approx(
        1.8e25 * math.exp(-0.3 / V_T))
    ni_sq = 1.0e20
    # Heavily doped n contact: the metal floor is dormant (bit-identical pass-through).
    n_heavy, p_heavy = doping_equilibrium_densities(1.0e24, 0.0, ni_sq)
    assert metal_reservoir_floor(n_heavy, p_heavy, intrinsic_density_squared_m6=ni_sq,
                                 donor_density_m3=1.0e24, acceptor_density_m3=0.0,
                                 conduction_dos_m3=N_c, valence_dos_m3=1.8e25,
                                 barrier_eV=0.3, thermal_voltage_V=V_T) == (n_heavy, p_heavy)
    # Lightly doped n contact: the floor supplies N_c exp(-phi_B/V_T) and keeps n p = ni^2.
    n_light, p_light = doping_equilibrium_densities(1.0e12, 0.0, ni_sq)
    n_floor, p_floor = metal_reservoir_floor(
        n_light, p_light, intrinsic_density_squared_m6=ni_sq, donor_density_m3=1.0e12,
        acceptor_density_m3=0.0, conduction_dos_m3=N_c, valence_dos_m3=1.8e25,
        barrier_eV=0.3, thermal_voltage_V=V_T)
    assert n_floor == N_c * math.exp(-0.3 / V_T) and n_floor > n_light
    assert n_floor * p_floor == pytest.approx(ni_sq, rel=1e-15)
    # p-type contact floors the hole reservoir; a missing DOS leaves it untouched.
    n_p, p_p = doping_equilibrium_densities(0.0, 1.0e12, ni_sq)
    floored = metal_reservoir_floor(n_p, p_p, intrinsic_density_squared_m6=ni_sq,
                                    donor_density_m3=0.0, acceptor_density_m3=1.0e12,
                                    conduction_dos_m3=N_c, valence_dos_m3=1.8e25,
                                    barrier_eV=0.3, thermal_voltage_V=V_T)
    assert floored[1] == 1.8e25 * math.exp(-0.3 / V_T)
    assert metal_reservoir_floor(n_p, p_p, intrinsic_density_squared_m6=ni_sq,
                                 donor_density_m3=0.0, acceptor_density_m3=1.0e12,
                                 conduction_dos_m3=N_c, valence_dos_m3=None, barrier_eV=0.3,
                                 thermal_voltage_V=V_T) == (n_p, p_p)
