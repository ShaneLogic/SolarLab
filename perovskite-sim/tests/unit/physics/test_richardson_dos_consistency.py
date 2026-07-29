"""``A*`` and ``N_C`` do not share an effective mass — the F-01 blocker.

The emission velocity the physical thermionic bound is built from,

    v_R = A* T^2 / (q N_C),

reduces to ``sqrt(k_B T / 2 pi m*)`` EXACTLY when both constants derive from
one effective mass, because

    A*  = 4 pi q m* k_B^2 / h^3
    N_C = 2 (2 pi m* k_B T / h^2)^{3/2}.

They do not here. ``A_star_n``/``A_star_p`` sit at the free-electron default
while ``Nc300``/``Nv300`` come from the parameter set, so v_R is a ratio of
two unrelated constants.

WHY THIS MATTERS, measured 2026-07-29. It is the concrete blocker on making
``te_physical_norm`` the default. A v_R that is too small makes the bound
throttle interface transport harder than the physics warrants, and on a
near-insulating contact that is not a shifted number but a diverging solve:

    scaps_mirror + Robin contacts + ETL N_D = 1e12 cm^-3, V_max = 1.2
      te_physical_norm = False                 V_oc = 1.44815 V
      te_physical_norm = True                  ValueError: infs or NaNs
      te_physical_norm = True, A* from N_C     V_oc = 1.38519 V

Deriving A* from each layer's own N_C removes the divergence, which is what
identifies the inconsistency — rather than the normalization — as the
blocker. Making them self-consistent is a behavioural change to every
config carrying effective-DOS data, so it is recorded rather than applied.

WHAT THIS FILE IS FOR. It pins the inconsistency as a known state. If a
future change makes the two consistent, these tests fail and say so, which
is the signal to re-examine whether ``te_physical_norm`` can become the
default. It deliberately does NOT assert that the flip diverges — that
would pin broken behaviour in place.
"""
from __future__ import annotations

import math

import pytest

from perovskite_sim.models.device import electrical_layers
from perovskite_sim.scaps_compat import load_scaps_yaml

K_B = 1.380649e-23
Q = 1.602176634e-19
H = 6.62607015e-34
T_REF = 300.0
_FREE_ELECTRON_A_STAR = 1.2017e6      # A m^-2 K^-2


def _a_star_from_dos(N_dos: float) -> float:
    """The Richardson constant implied by an effective density of states."""
    m_eff = (N_dos / 2.0) ** (2.0 / 3.0) * H * H / (2.0 * math.pi * K_B * T_REF)
    return 4.0 * math.pi * Q * m_eff * K_B * K_B / (H ** 3)


def _v_r(a_star: float, N_dos: float) -> float:
    return a_star * T_REF ** 2 / (Q * N_dos)


def _v_r_analytic(N_dos: float) -> float:
    """sqrt(k T / 2 pi m*), with m* read back out of N_dos."""
    m_eff = (N_dos / 2.0) ** (2.0 / 3.0) * H * H / (2.0 * math.pi * K_B * T_REF)
    return math.sqrt(K_B * T_REF / (2.0 * math.pi * m_eff))


@pytest.fixture(scope="module")
def dos_layers():
    stack = load_scaps_yaml("configs/scaps_mirror_v2.yaml")
    out = [
        l for l in electrical_layers(stack)
        if getattr(l.params, "Nc300", None) and getattr(l.params, "A_star_n", None)
    ]
    assert out, "no layer carries both Nc300 and A_star_n"
    return out


def test_the_identity_holds_when_both_come_from_one_effective_mass():
    """Sanity on the algebra this file rests on, before asserting a defect."""
    for N_dos in (1e25, 8e25, 2.5e26):
        a = _a_star_from_dos(N_dos)
        assert _v_r(a, N_dos) == pytest.approx(_v_r_analytic(N_dos), rel=1e-12)


def test_shipped_richardson_constants_are_not_dos_consistent(dos_layers):
    """The defect: A* is the free-electron value regardless of N_C.

    Pinned as a KNOWN INCONSISTENCY. When it is fixed this fails, and the
    F-01 default question should be reopened — see the module docstring.
    """
    offenders = []
    for l in dos_layers:
        implied = _a_star_from_dos(float(l.params.Nc300))
        ratio = float(l.params.A_star_n) / implied
        if abs(math.log10(ratio)) > math.log10(1.2):     # >20 % adrift
            offenders.append((l.name, ratio))
    assert offenders, (
        "every layer's Richardson constant now agrees with its effective "
        "density of states — the F-01 blocker may be gone; re-measure "
        "whether te_physical_norm can default to True"
    )


def test_the_emission_velocity_is_materially_off_at_a_transport_layer(dos_layers):
    """Quantify it, so the record is a number rather than an adjective.

    Measured 4.6x at the hole-transport layer of scaps_mirror_v2.
    """
    worst_name, worst = None, 1.0
    for l in dos_layers:
        got = _v_r(float(l.params.A_star_n), float(l.params.Nc300))
        want = _v_r_analytic(float(l.params.Nc300))
        r = got / want
        if abs(math.log10(r)) > abs(math.log10(worst)):
            worst_name, worst = l.name, r
    assert worst < 0.5 or worst > 2.0, (
        f"worst emission-velocity error is only {worst:.2f}x (at "
        f"{worst_name}); the inconsistency this file records has shrunk and "
        "the F-01 blocker should be re-measured"
    )


def test_a_star_defaults_to_the_free_electron_value(dos_layers):
    """Name the mechanism: nothing derives A* from the layer at all."""
    for l in dos_layers:
        assert float(l.params.A_star_n) == pytest.approx(
            _FREE_ELECTRON_A_STAR, rel=1e-6
        ), (
            f"{l.name} no longer carries the free-electron Richardson "
            "constant — if it is now layer-derived, reopen the F-01 default"
        )
