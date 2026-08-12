"""Static one-dimensional dopant profiles within an electrical layer.

``MaterialParams.N_A`` and ``N_D`` remain uniform unless a layer explicitly
declares a profile.  For a profiled species the scalar density is the value at
the selected layer edge and ``N_*_bulk`` is the deep-layer asymptote.  The
Gaussian convention is

    N(d) = N_bulk + (N_edge - N_bulk) * exp(-(d / L)^2),

where ``d`` is distance from the selected edge.  This convention makes the
decay length an unambiguous 1/e distance and directly represents diffused
silicon emitters without splitting one material into many artificial layers.
"""
from __future__ import annotations

import math

import numpy as np


_VALID_SHAPES = ("gaussian",)
_VALID_EDGES = ("front", "back")


def has_doping_profile_params(params) -> bool:
    """Return whether either dopant species has a bulk asymptote."""
    if params is None:
        return False
    return (
        getattr(params, "N_A_bulk", None) is not None
        or getattr(params, "N_D_bulk", None) is not None
    )


def validate_doping_profile_params(params) -> None:
    """Fail closed on incomplete or non-physical profile parameters."""
    if params is None:
        return
    active = has_doping_profile_params(params)
    shape = getattr(params, "doping_profile_shape", None)
    decay_length = getattr(params, "doping_decay_length", None)
    edge = str(getattr(params, "doping_edge", "front"))

    if not active:
        if shape is not None or decay_length is not None:
            raise ValueError(
                "doping profile shape/decay length requires N_A_bulk or "
                "N_D_bulk"
            )
        return
    if shape not in _VALID_SHAPES:
        raise ValueError(
            f"doping_profile_shape must be one of {_VALID_SHAPES}, got "
            f"{shape!r}"
        )
    if edge not in _VALID_EDGES:
        raise ValueError(
            f"doping_edge must be one of {_VALID_EDGES}, got {edge!r}"
        )
    try:
        length = float(decay_length)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "doping_decay_length must be finite and positive"
        ) from exc
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("doping_decay_length must be finite and positive")

    for name in ("N_A", "N_D", "N_A_bulk", "N_D_bulk"):
        value = getattr(params, name, None)
        if value is None:
            continue
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")


def gaussian_doping_profile(
    x_local: np.ndarray,
    thickness: float,
    edge_density: float,
    bulk_density: float,
    decay_length: float,
    *,
    edge: str,
) -> np.ndarray:
    """Evaluate one Gaussian dopant species on local layer coordinates."""
    positions = np.asarray(x_local, dtype=float)
    if edge == "front":
        distance = positions
    elif edge == "back":
        distance = float(thickness) - positions
    else:
        raise ValueError(
            f"doping edge must be one of {_VALID_EDGES}, got {edge!r}"
        )
    kernel = np.exp(-((distance / float(decay_length)) ** 2))
    return float(bulk_density) + (
        float(edge_density) - float(bulk_density)
    ) * kernel


def layer_doping_profiles(
    x_local: np.ndarray,
    thickness: float,
    params,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-node ``(N_A, N_D)`` for one layer."""
    validate_doping_profile_params(params)
    positions = np.asarray(x_local, dtype=float)
    acceptors = np.full_like(positions, float(params.N_A))
    donors = np.full_like(positions, float(params.N_D))
    if not has_doping_profile_params(params):
        return acceptors, donors

    kwargs = {
        "thickness": float(thickness),
        "decay_length": float(params.doping_decay_length),
        "edge": str(params.doping_edge),
    }
    if params.N_A_bulk is not None:
        acceptors = gaussian_doping_profile(
            positions,
            edge_density=float(params.N_A),
            bulk_density=float(params.N_A_bulk),
            **kwargs,
        )
    if params.N_D_bulk is not None:
        donors = gaussian_doping_profile(
            positions,
            edge_density=float(params.N_D),
            bulk_density=float(params.N_D_bulk),
            **kwargs,
        )
    return acceptors, donors


def doping_at_position(
    params,
    position: float,
    thickness: float,
) -> tuple[float, float]:
    """Return scalar ``(N_A, N_D)`` at a layer position."""
    acceptors, donors = layer_doping_profiles(
        np.asarray([float(position)]), float(thickness), params
    )
    return float(acceptors[0]), float(donors[0])
