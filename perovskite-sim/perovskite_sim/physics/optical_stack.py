"""Single source of truth for DeviceStack -> TMM layer construction."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from perovskite_sim.data import load_nk
from perovskite_sim.models.device import DeviceStack
from perovskite_sim.physics.cigs_optics import (
    ggi_profile_from_coordinate,
    minoura_nk,
)
from perovskite_sim.physics.grading import grading_coordinate, has_grading_params
from perovskite_sim.physics.optics import TMMLayer


@dataclass(frozen=True, slots=True)
class DeviceOpticalStack:
    """Expanded TMM stack plus its mapping to physical device layers."""

    layers: tuple[TMMLayer, ...]
    boundaries_m: np.ndarray
    physical_layer_slices: tuple[slice, ...]
    graded_physical_layer_indices: tuple[int, ...]


def has_wavelength_resolved_optics(stack: DeviceStack) -> bool:
    """Whether this stack can activate the wavelength-resolved TMM path."""

    has_table = any(
        layer.params is not None and layer.params.optical_material is not None
        for layer in stack.layers
    )
    has_cigs = bool(getattr(stack, "graded_optics", False)) and any(
        layer.params is not None
        and layer.params.cigs_graded_optics is not None
        for layer in stack.layers
    )
    return has_table or has_cigs


def _validate_graded_optics_activation(stack: DeviceStack) -> tuple[int, ...]:
    if not bool(getattr(stack, "graded_optics", False)):
        return ()
    if str(stack.mode).strip().lower() == "legacy":
        raise ValueError("graded_optics is unavailable in legacy simulation mode")
    if not bool(getattr(stack, "band_grading", False)):
        raise ValueError(
            "graded_optics requires band_grading so optics and Eg/chi share "
            "the same composition coordinate"
        )

    active: list[int] = []
    for index, layer in enumerate(stack.layers):
        params = layer.params
        if params is None or params.cigs_graded_optics is None:
            continue
        if layer.role != "absorber":
            raise ValueError(
                f"cigs_graded_optics is restricted to absorber layers; "
                f"layer {layer.name!r} has role {layer.role!r}"
            )
        if not has_grading_params(params):
            raise ValueError(
                f"graded CIGS optical layer {layer.name!r} must declare an "
                "Eg_back or chi_back electrical endpoint"
            )
        active.append(index)
    if not active:
        raise ValueError(
            "graded_optics is enabled but no layer declares cigs_graded_optics"
        )
    return tuple(active)


def build_device_optical_stack(
    stack: DeviceStack,
    wavelengths_nm: np.ndarray,
) -> DeviceOpticalStack:
    """Build a full optical stack, expanding active CIGS layers into slices.

    Slice-centre GGI values consume ``grading_coordinate`` with the exact same
    profile, direction, and characteristic length used by electrical Eg/chi.
    No graded block is read while the device master gate is off.
    """

    wavelengths = np.asarray(wavelengths_nm, dtype=float)
    if wavelengths.ndim != 1 or wavelengths.size == 0:
        raise ValueError(
            "wavelengths_nm must be a non-empty one-dimensional array"
        )
    if not np.all(np.isfinite(wavelengths)) or np.any(wavelengths <= 0.0):
        raise ValueError("wavelengths_nm must contain finite positive values")

    graded_indices = _validate_graded_optics_activation(stack)
    graded_set = set(graded_indices)
    wavelengths_m = wavelengths * 1e-9
    n_wavelengths = len(wavelengths)
    layers: list[TMMLayer] = []
    physical_slices: list[slice] = []

    for physical_index, layer in enumerate(stack.layers):
        params = layer.params
        if params is None:
            raise ValueError(
                f"Layer {layer.name!r} has no MaterialParams; cannot build TMM stack"
            )
        start = len(layers)
        if physical_index in graded_set:
            model = params.cigs_graded_optics
            assert model is not None
            centres = (
                np.arange(model.slices, dtype=float) + 0.5
            ) * (layer.thickness / model.slices)
            coordinate = grading_coordinate(
                centres,
                layer.thickness,
                params.grading_profile,
                params.grading_char_length,
                params.grading_direction,
            )
            ggi_values = ggi_profile_from_coordinate(coordinate, model)
            for ggi in ggi_values:
                n_arr, k_arr = minoura_nk(
                    wavelengths,
                    float(ggi),
                    model.cgi,
                    quadrature_order=model.kk_quadrature_order,
                )
                layers.append(
                    TMMLayer(
                        d=layer.thickness / model.slices,
                        n=n_arr,
                        k=k_arr,
                        incoherent=False,
                    )
                )
        else:
            if params.optical_material is not None:
                _, n_arr, k_arr = load_nk(params.optical_material, wavelengths)
            elif params.n_optical is not None:
                n_arr = np.full(n_wavelengths, params.n_optical)
                k_arr = params.alpha * wavelengths_m / (4.0 * np.pi)
            else:
                n_arr = np.full(n_wavelengths, np.sqrt(params.eps_r))
                k_arr = params.alpha * wavelengths_m / (4.0 * np.pi)
            layers.append(
                TMMLayer(
                    d=layer.thickness,
                    n=n_arr,
                    k=k_arr,
                    incoherent=bool(params.incoherent),
                )
            )
        physical_slices.append(slice(start, len(layers)))

    boundaries = np.empty(len(layers) + 1, dtype=float)
    boundaries[0] = 0.0
    boundaries[1:] = np.cumsum([layer.d for layer in layers])
    if not np.isclose(
        boundaries[-1],
        sum(float(layer.thickness) for layer in stack.layers),
        rtol=2e-14,
        atol=1e-18,
    ):
        raise RuntimeError("expanded optical stack does not preserve total thickness")
    return DeviceOpticalStack(
        layers=tuple(layers),
        boundaries_m=boundaries,
        physical_layer_slices=tuple(physical_slices),
        graded_physical_layer_indices=graded_indices,
    )


def cigs_nk_at_electrical_gap_edge(
    layer,
    wavelength_m: float,
) -> tuple[float, float]:
    """CIGS n,k at the narrower electrical-gap endpoint for recycling."""

    params = layer.params
    model = params.cigs_graded_optics
    if model is None:
        raise ValueError("layer does not declare cigs_graded_optics")
    gap_back = params.Eg_back if params.Eg_back is not None else params.Eg
    if params.Eg <= gap_back:
        coordinate = 0.0
    else:
        coordinate = 1.0
    ggi = float(ggi_profile_from_coordinate(np.array([coordinate]), model)[0])
    n, k = minoura_nk(
        np.array([float(wavelength_m) * 1e9]),
        ggi,
        model.cgi,
        quadrature_order=model.kk_quadrature_order,
    )
    return float(n[0]), float(k[0])
