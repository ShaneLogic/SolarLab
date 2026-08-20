from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np


SRHDenominatorObserver = Callable[[str, np.ndarray], None]
_SRH_DENOMINATOR_OBSERVER: ContextVar[SRHDenominatorObserver | None] = (
    ContextVar("srh_denominator_observer", default=None)
)


@contextmanager
def _observe_srh_denominators(
    observer: SRHDenominatorObserver,
) -> Iterator[None]:
    """Route production SRH denominators to one per-solve observer."""

    token = _SRH_DENOMINATOR_OBSERVER.set(observer)
    try:
        yield
    finally:
        _SRH_DENOMINATOR_OBSERVER.reset(token)


def _record_srh_denominator(kind: str, denominator: np.ndarray) -> None:
    observer = _SRH_DENOMINATOR_OBSERVER.get()
    if observer is not None:
        observer(kind, np.asarray(denominator))


def bulk_srh_denominator(
    n: np.ndarray,
    p: np.ndarray,
    tau_n: float,
    tau_p: float,
    n1: float,
    p1: float,
) -> np.ndarray:
    """Bulk SRH denominator [s m^-3], without altering invalid inputs."""

    return tau_p * (n + n1) + tau_n * (p + p1)


def interface_srh_denominator(
    n: float,
    p: float,
    n1: float,
    p1: float,
    v_n: float,
    v_p: float,
) -> float:
    """Surface SRH denominator [s m^-4] for positive capture velocities."""

    return (n + n1) / v_p + (p + p1) / v_n


def srh_recombination(
    n: np.ndarray, p: np.ndarray, ni_sq: float,
    tau_n: float, tau_p: float, n1: float, p1: float,
) -> np.ndarray:
    """Shockley-Read-Hall recombination rate [m⁻³ s⁻¹]."""
    denominator = bulk_srh_denominator(n, p, tau_n, tau_p, n1, p1)
    _record_srh_denominator("bulk", denominator)
    return (n * p - ni_sq) / denominator


def radiative_recombination(
    n: np.ndarray, p: np.ndarray, ni_sq: float, B_rad: float,
) -> np.ndarray:
    """Bimolecular radiative recombination rate [m⁻³ s⁻¹]."""
    return B_rad * (n * p - ni_sq)


def auger_recombination(
    n: np.ndarray, p: np.ndarray, ni_sq: float,
    C_n: float, C_p: float,
) -> np.ndarray:
    """Auger recombination rate [m⁻³ s⁻¹]."""
    return (C_n * n + C_p * p) * (n * p - ni_sq)


def interface_recombination(
    n: float, p: float, ni_sq: float,
    n1: float, p1: float,
    v_n: float, v_p: float,
) -> float:
    """Interface (surface) SRH recombination rate [m⁻² s⁻¹].

    Parameters
    ----------
    n, p : carrier densities at the interface node [m⁻³]
    ni_sq : intrinsic carrier density squared [m⁻⁶]
    n1, p1 : SRH trap-level carrier densities [m⁻³]
    v_n, v_p : surface recombination velocities [m/s]
    """
    if v_n <= 0.0 or v_p <= 0.0:
        # A single blocked capture channel blocks the full SRH cycle: the
        # denominator diverges as v -> 0, so the physical limit is R -> 0.
        # Guarding both (not just the both-zero case) also prevents a
        # ZeroDivisionError for configs with one-sided passivation.
        return 0.0
    denominator = interface_srh_denominator(n, p, n1, p1, v_n, v_p)
    _record_srh_denominator("interface", np.asarray(denominator))
    return (n * p - ni_sq) / denominator


def total_recombination(
    n: np.ndarray, p: np.ndarray, ni_sq: float,
    tau_n: float, tau_p: float, n1: float, p1: float,
    B_rad: float, C_n: float, C_p: float,
) -> np.ndarray:
    """Sum of SRH + radiative + Auger [m⁻³ s⁻¹]."""
    return (
        srh_recombination(n, p, ni_sq, tau_n, tau_p, n1, p1)
        + radiative_recombination(n, p, ni_sq, B_rad)
        + auger_recombination(n, p, ni_sq, C_n, C_p)
    )
