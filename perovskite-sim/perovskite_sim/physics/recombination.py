from __future__ import annotations
import numpy as np


def srh_recombination(
    n: np.ndarray, p: np.ndarray, ni_sq: float,
    tau_n: float, tau_p: float, n1: float, p1: float,
) -> np.ndarray:
    """Shockley-Read-Hall recombination rate [m⁻³ s⁻¹]."""
    return (n * p - ni_sq) / (tau_p * (n + n1) + tau_n * (p + p1))


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
    if v_n == 0.0 and v_p == 0.0:
        return 0.0
    return (n * p - ni_sq) / ((n + n1) / v_p + (p + p1) / v_n)


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
