"""Absolute-tolerance policies for density-state ODE integrations.

SciPy controls local error componentwise with ``atol_i + rtol * abs(y_i)``.
The transient state blocks in SolarLab are all densities in m^-3, but their
physically relevant zero scales differ by species and position.  This module
builds opt-in vector tolerances from material reference states while keeping
the historical scalar tolerance path available unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np


@dataclass(frozen=True)
class ComponentwiseAtol:
    """Reference-scaled absolute-tolerance policy for density states.

    The generated tolerance for component ``i`` of species ``s`` is

    ``refinement_factor * max(minimum_atol, fraction_s * reference_i)``.

    The default fractions are deliberately much smaller than the solver's
    usual relative tolerances.  They define the near-zero error floor; once a
    state is appreciable, ``rtol * abs(y_i)`` remains the dominant term.
    Constructing this object is explicit opt-in: existing scalar defaults do
    not instantiate or apply this policy.
    """

    carrier_fraction: float = 1.0e-12
    ion_fraction: float = 1.0e-12
    interface_fraction: float = 1.0e-12
    minimum_atol: float = 1.0e-6
    refinement_factor: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "carrier_fraction",
            "ion_fraction",
            "interface_fraction",
            "minimum_atol",
            "refinement_factor",
        ):
            value = getattr(self, name)
            if isinstance(value, bool):
                raise ValueError(f"{name} must be a finite positive number")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{name} must be a finite positive number"
                ) from exc
            if not math.isfinite(number) or number <= 0.0:
                raise ValueError(f"{name} must be a finite positive number")
            object.__setattr__(self, name, number)

    def refined(self, factor: float = 0.1) -> "ComponentwiseAtol":
        """Return a uniformly tighter/looser policy for refinement studies.

        ``factor < 1`` tightens every generated absolute tolerance by the
        same ratio; ``factor > 1`` loosens it.  Repeated calls compose.
        """
        if isinstance(factor, bool):
            raise ValueError("refinement factor must be finite and positive")
        try:
            factor_value = float(factor)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "refinement factor must be finite and positive"
            ) from exc
        if not math.isfinite(factor_value) or factor_value <= 0.0:
            raise ValueError("refinement factor must be finite and positive")
        return replace(
            self,
            refinement_factor=self.refinement_factor * factor_value,
        )


AbsoluteTolerance = float | ComponentwiseAtol
ResolvedAbsoluteTolerance = float | np.ndarray


def _finite_vector(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _matching_vector(
    value: np.ndarray,
    name: str,
    reference: np.ndarray,
) -> np.ndarray:
    array = _finite_vector(value, name)
    if array.shape != reference.shape:
        raise ValueError(
            f"{name} shape {array.shape} does not match {reference.shape}"
        )
    return array


def _carrier_reference_scales(
    ni_sq: np.ndarray,
    N_A: np.ndarray,
    N_D: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return stable local dark-equilibrium reference scales for n and p."""
    ni_sq_arr = _finite_vector(ni_sq, "ni_sq")
    if np.any(ni_sq_arr < 0.0):
        raise ValueError("ni_sq must be non-negative")
    N_A_arr = _matching_vector(N_A, "N_A", ni_sq_arr)
    N_D_arr = _matching_vector(N_D, "N_D", ni_sq_arr)
    if np.any(N_A_arr < 0.0) or np.any(N_D_arr < 0.0):
        raise ValueError("N_A and N_D must be non-negative")

    ni = np.sqrt(ni_sq_arr)
    net = N_D_arr - N_A_arr
    disc = np.hypot(net, 2.0 * ni)
    majority_n = 0.5 * (net + disc)
    majority_p = 0.5 * (-net + disc)
    with np.errstate(divide="ignore", invalid="ignore"):
        n_ref = np.where(
            net >= 0.0,
            majority_n,
            np.divide(
                ni_sq_arr,
                majority_p,
                out=np.zeros_like(ni_sq_arr),
                where=majority_p > 0.0,
            ),
        )
        p_ref = np.where(
            net >= 0.0,
            np.divide(
                ni_sq_arr,
                majority_n,
                out=np.zeros_like(ni_sq_arr),
                where=majority_n > 0.0,
            ),
            majority_p,
        )
    return n_ref, p_ref


def _scaled_atol(
    reference: np.ndarray,
    fraction: float,
    policy: ComponentwiseAtol,
) -> np.ndarray:
    return policy.refinement_factor * np.maximum(
        policy.minimum_atol,
        float(fraction) * np.abs(reference),
    )


def build_componentwise_atol_1d(
    policy: ComponentwiseAtol,
    *,
    y0: np.ndarray,
    ni_sq: np.ndarray,
    N_A: np.ndarray,
    N_D: np.ndarray,
    P_ion0: np.ndarray,
    has_dual_ions: bool = False,
    P_ion0_neg: np.ndarray | None = None,
    n_interface_states: int = 0,
) -> np.ndarray:
    """Build ``(n, p, P[, P_neg][, interface])`` absolute tolerances."""
    if isinstance(n_interface_states, bool):
        raise ValueError("n_interface_states must be a non-negative integer")
    try:
        n_iface = int(n_interface_states)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "n_interface_states must be a non-negative integer"
        ) from exc
    if n_iface != n_interface_states or n_iface < 0:
        raise ValueError("n_interface_states must be a non-negative integer")

    n_ref, p_ref = _carrier_reference_scales(ni_sq, N_A, N_D)
    P_ref = _matching_vector(P_ion0, "P_ion0", n_ref)
    if np.any(P_ref < 0.0):
        raise ValueError("P_ion0 must be non-negative")

    expected_size = 3 * n_ref.size + 4 * n_iface
    P_neg_ref = None
    if has_dual_ions:
        if P_ion0_neg is None:
            raise ValueError("P_ion0_neg is required for a dual-ion state")
        P_neg_ref = _matching_vector(P_ion0_neg, "P_ion0_neg", n_ref)
        if np.any(P_neg_ref < 0.0):
            raise ValueError("P_ion0_neg must be non-negative")
        expected_size += n_ref.size

    state = _finite_vector(y0, "y0")
    if state.size != expected_size:
        raise ValueError(
            f"y0 size {state.size} does not match the 1D state layout "
            f"size {expected_size}"
        )

    blocks = [
        _scaled_atol(n_ref, policy.carrier_fraction, policy),
        _scaled_atol(p_ref, policy.carrier_fraction, policy),
        _scaled_atol(P_ref, policy.ion_fraction, policy),
    ]
    if P_neg_ref is not None:
        blocks.append(_scaled_atol(P_neg_ref, policy.ion_fraction, policy))
    if n_iface:
        interface_size = 4 * n_iface
        interface_ref = np.abs(state[-interface_size:])
        blocks.append(
            _scaled_atol(
                interface_ref,
                policy.interface_fraction,
                policy,
            )
        )
    return np.concatenate(blocks)


def build_componentwise_atol_ions(
    policy: ComponentwiseAtol,
    *,
    P_ion0: np.ndarray,
    P_ion0_neg: np.ndarray | None = None,
) -> np.ndarray:
    """Build the ion-only tolerance used by ``split_step``."""
    P_ref = _finite_vector(P_ion0, "P_ion0")
    if np.any(P_ref < 0.0):
        raise ValueError("P_ion0 must be non-negative")
    blocks = [_scaled_atol(P_ref, policy.ion_fraction, policy)]
    if P_ion0_neg is not None:
        P_neg_ref = _matching_vector(P_ion0_neg, "P_ion0_neg", P_ref)
        if np.any(P_neg_ref < 0.0):
            raise ValueError("P_ion0_neg must be non-negative")
        blocks.append(_scaled_atol(P_neg_ref, policy.ion_fraction, policy))
    return np.concatenate(blocks)


def build_componentwise_atol_2d(
    policy: ComponentwiseAtol,
    *,
    ni: np.ndarray,
    N_A: np.ndarray,
    N_D: np.ndarray,
    P_ion0: np.ndarray | None = None,
) -> np.ndarray:
    """Build flattened ``(n, p[, P])`` tolerances for the 2D solver."""
    ni_arr = np.asarray(ni, dtype=float)
    N_A_arr = np.asarray(N_A, dtype=float)
    N_D_arr = np.asarray(N_D, dtype=float)
    if ni_arr.ndim != 2 or ni_arr.size == 0:
        raise ValueError("ni must be a non-empty two-dimensional array")
    if N_A_arr.shape != ni_arr.shape or N_D_arr.shape != ni_arr.shape:
        raise ValueError("N_A and N_D must match the 2D ni shape")
    if (
        not np.all(np.isfinite(ni_arr))
        or not np.all(np.isfinite(N_A_arr))
        or not np.all(np.isfinite(N_D_arr))
    ):
        raise ValueError("2D carrier reference arrays must be finite")
    if np.any(ni_arr < 0.0):
        raise ValueError("ni must be non-negative")
    n_ref, p_ref = _carrier_reference_scales(
        np.square(ni_arr).ravel(),
        N_A_arr.ravel(),
        N_D_arr.ravel(),
    )
    blocks = [
        _scaled_atol(n_ref, policy.carrier_fraction, policy),
        _scaled_atol(p_ref, policy.carrier_fraction, policy),
    ]
    if P_ion0 is not None:
        P_ref = np.asarray(P_ion0, dtype=float)
        if P_ref.shape != ni_arr.shape or not np.all(np.isfinite(P_ref)):
            raise ValueError("P_ion0 must be finite and match the 2D ni shape")
        if np.any(P_ref < 0.0):
            raise ValueError("P_ion0 must be non-negative")
        blocks.append(
            _scaled_atol(P_ref.ravel(), policy.ion_fraction, policy)
        )
    return np.concatenate(blocks)
