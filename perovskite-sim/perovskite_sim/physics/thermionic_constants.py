"""Richardson constants derived from the effective density of states.

The Richardson constant and the band-edge effective density of states are
not independent inputs — both are set by the same effective mass::

    A*  = 4 pi q m* k_B^2 / h^3
    N   = 2 (2 pi m* k_B T / h^2)^{3/2}

so the emission velocity that the interface-limited thermionic bound is
built from collapses to a pure thermal velocity::

    v_R = A* T^2 / (q N) = sqrt(k_B T / 2 pi m*)

That identity only holds when both come from one m*. SolarLab's per-layer
``A_star_n`` / ``A_star_p`` default to the free-electron value while
``Nc300`` / ``Nv300`` are read from the material parameter set, so a layer
that declares a DOS but leaves the Richardson constant alone ends up with a
``v_R`` that is a ratio of two unrelated constants.

Measured on the SCAPS mirror stack, the Richardson constant each layer's own
DOS implies, against the free-electron default it actually carries:

    layer        A*(N_C) / A*_default
    HTL                4.63x
    absorber           0.54x
    ETL                2.17x

i.e. the emission velocity is wrong by those factors, in different
directions per layer. This is not academic: it is what blocked making the
dimensionally correct thermionic normalization the default, because an
under-sized ``v_R`` throttles interface transport hard enough to drive a
near-insulating-contact stack to a non-finite solve.

An explicitly configured ``A_star_*`` always wins — a user who sets it is
making a statement about their interface that this module must not override.
Only the untouched free-electron default is replaced.
"""
from __future__ import annotations

import math

from perovskite_sim.constants import (
    A_STAR_FREE_ELECTRON, H_PLANCK, K_B, Q, T as T_REF,
)

__all__ = [
    "effective_mass_from_dos",
    "richardson_from_dos",
    "emission_velocity",
    "is_free_electron_default",
]

# Relative tolerance for recognising the untouched default. ``MaterialParams``
# stores the ROUNDED literal 1.2017e6 while the exact expression evaluates to
# 1.201732e6 — 2.7e-5 apart — so the tolerance has to absorb that rounding.
# 1e-3 does, and is still three orders tighter than any real material's
# departure from the free-electron value (the SCAPS stack's own layers imply
# 0.54x to 4.63x), so a configured A* can never be mistaken for the default.
_DEFAULT_RTOL = 1e-3


def effective_mass_from_dos(N_dos: float, T: float = T_REF) -> float:
    """Effective mass [kg] implied by a band-edge DOS ``N_dos`` [m^-3].

    Inverts ``N = 2 (2 pi m* k_B T / h^2)^{3/2}``.
    """
    if N_dos <= 0.0:
        raise ValueError(f"N_dos must be positive, got {N_dos!r}")
    return (N_dos / 2.0) ** (2.0 / 3.0) * H_PLANCK * H_PLANCK / (
        2.0 * math.pi * K_B * T
    )


def richardson_from_dos(N_dos: float, T: float = T_REF) -> float:
    """Richardson constant [A m^-2 K^-2] consistent with ``N_dos``.

    ``A* = 4 pi q m* k_B^2 / h^3`` with ``m*`` taken from the DOS, so that
    ``A* T^2 / (q N_dos)`` is exactly the thermal emission velocity.
    """
    m_eff = effective_mass_from_dos(N_dos, T)
    return 4.0 * math.pi * Q * m_eff * K_B * K_B / (H_PLANCK ** 3)


def emission_velocity(N_dos: float, T: float = T_REF) -> float:
    """``sqrt(k_B T / 2 pi m*)`` [m/s] — what ``v_R`` should equal."""
    m_eff = effective_mass_from_dos(N_dos, T)
    return math.sqrt(K_B * T / (2.0 * math.pi * m_eff))


def is_free_electron_default(a_star: float) -> bool:
    """True when ``a_star`` is the untouched free-electron default.

    Used to distinguish "the user configured this" from "nobody set it",
    since only the latter may be replaced.
    """
    return math.isclose(
        float(a_star), A_STAR_FREE_ELECTRON, rel_tol=_DEFAULT_RTOL,
    )
