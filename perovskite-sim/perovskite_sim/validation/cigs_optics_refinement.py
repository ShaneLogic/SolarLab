"""Registered P4.4 composition-graded CIGS optical refinement lane."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from perovskite_sim._compat.numpy_compat import trapezoid
from perovskite_sim.data import load_am15g
from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.physics.cigs_optics import (
    HC_EV_NM,
    carron_absorption_coefficient,
    carron_band_gap_eV,
    ggi_profile_from_coordinate,
    minoura_nk,
)
from perovskite_sim.physics.generation import dual_cell_widths
from perovskite_sim.physics.grading import grading_coordinate, has_grading_params
from perovskite_sim.physics.optical_stack import (
    build_device_optical_stack,
    has_wavelength_resolved_optics,
)
from perovskite_sim.physics.optics import tmm_reflectance
from perovskite_sim.solver.mol import _compute_tmm_generation

from .dae_refinement import (
    _finite_option,
    _integer_option,
    _protocol_metadata,
    _string_option,
)
from .numerical_certificate import LaneDefinition, MatrixPoint
from .refinement_runner import CellMeasurement


def _kk_quadrature_order(base_order: int, tolerance_factor: float) -> int:
    exact = base_order / float(tolerance_factor)
    rounded = round(exact)
    if (
        not math.isfinite(exact)
        or rounded < 48
        or rounded > 2048
        or not math.isclose(exact, rounded, rel_tol=0.0, abs_tol=1.0e-12)
    ):
        raise ValueError(
            "base_kk_quadrature_order / tolerance_factor must be an integer "
            "in [48, 2048]"
        )
    return int(rounded)


def _replace_cigs_resolution(
    stack,
    *,
    absorber_layer_name: str,
    slices: int,
    quadrature_order: int,
):
    matches = [
        (index, layer)
        for index, layer in enumerate(stack.layers)
        if layer.params is not None
        and layer.params.cigs_graded_optics is not None
    ]
    if len(matches) != 1:
        raise ValueError("CIGS optical refinement requires exactly one model block")
    index, absorber = matches[0]
    if absorber.name != absorber_layer_name or absorber.role != "absorber":
        raise ValueError("CIGS optical refinement absorber identity mismatch")
    model = replace(
        absorber.params.cigs_graded_optics,
        slices=slices,
        kk_quadrature_order=quadrature_order,
    )
    layers = list(stack.layers)
    layers[index] = replace(
        absorber,
        params=replace(absorber.params, cigs_graded_optics=model),
    )
    return replace(stack, layers=tuple(layers)), index, model


def _replace_cigs_model(stack, absorber_index: int, model):
    layers = list(stack.layers)
    absorber = layers[absorber_index]
    layers[absorber_index] = replace(
        absorber,
        params=replace(absorber.params, cigs_graded_optics=model),
    )
    return replace(stack, layers=tuple(layers))


def _execution_protocol(
    lane: LaneDefinition,
    *,
    absorber_layer_name: str,
    fixed_electrical_intervals_per_layer: int,
    wavelength_points: int,
    wavelength_min_nm: float,
    wavelength_max_nm: float,
    base_kk_quadrature_order: int,
    carron_energy_points_per_composition: int,
    carron_minimum_excess_above_gap_eV: float,
) -> dict[str, object]:
    return {
        "constitutive_closure": {
            "absorption_benchmark": "carron_2018_equations_2_to_6",
            "complex_index": "minoura_2015_shifted_tauc_lorentz",
            "composition_coordinate": "shared_with_electrical_Eg_chi_grade",
            "dielectric_causality": "epsilon2_then_kramers_kronig_epsilon1",
            "optical_discretization": "piecewise_constant_slice_centres",
        },
        "independent_benchmark": {
            "composition_points": "ggi_front_midpoint_back",
            "energy_max_eV": 2.5,
            "energy_min_formula": (
                "max(1.35, carron_gap_eV + minimum_excess_eV)"
            ),
            "energy_points_per_composition": carron_energy_points_per_composition,
            "minimum_excess_eV": carron_minimum_excess_above_gap_eV,
        },
        "matrix": {
            "grid_parameter": lane.grid_parameter,
            "grid_values": list(lane.grid_values),
            "tolerance_factors": list(lane.tolerance_factors),
            "tolerance_parameter": lane.tolerance_parameter,
        },
        "numerical_resolution": {
            "base_kk_quadrature_order": base_kk_quadrature_order,
            "effective_kk_order_formula": (
                "base_kk_quadrature_order / matrix.tolerance_factor"
            ),
            "fixed_electrical_intervals_per_layer": (
                fixed_electrical_intervals_per_layer
            ),
            "wavelength_max_nm": wavelength_max_nm,
            "wavelength_min_nm": wavelength_min_nm,
            "wavelength_points": wavelength_points,
        },
        "operating_point": {
            "illumination": "AM1.5G",
            "quantity": "build_once_photon_conserving_generation",
            "transport_solve": "excluded",
        },
        "schema_version": "cigs-graded-optics-refinement-protocol-v1",
        "source": {
            "absorber_layer_name": absorber_layer_name,
            "carron_doi": "10.1080/14686996.2018.1458579",
            "config_path": lane.config_path,
            "config_sha256": lane.config_sha256,
            "minoura_doi": "10.1063/1.4921300",
        },
        "topology": {
            "graded_optics_gate": "explicit_true",
            "non_cigs_layers": "historical_scalar_optical_fallback",
            "production_adapter": "solver.mol._compute_tmm_generation",
            "uniform_composition_limit": "one_slice_vs_matrix_slice_count",
        },
    }


def _carron_benchmark(
    model,
    *,
    quadrature_order: int,
    energy_points: int,
    minimum_excess_eV: float,
) -> tuple[float, float, float, int]:
    compositions = (
        model.ggi_front,
        0.5 * (model.ggi_front + model.ggi_back),
        model.ggi_back,
    )
    medians: list[float] = []
    minimum_ratio = math.inf
    maximum_ratio = -math.inf
    for ggi in compositions:
        energy_min = max(1.35, carron_band_gap_eV(ggi) + minimum_excess_eV)
        if energy_min >= 2.5:
            raise ValueError("Carron benchmark energy interval is empty")
        energy = np.linspace(energy_min, 2.5, energy_points)
        wavelengths_nm = HC_EV_NM / energy
        _, k = minoura_nk(
            wavelengths_nm,
            ggi,
            model.cgi,
            quadrature_order=quadrature_order,
        )
        alpha_minoura = 4.0 * np.pi * k / (wavelengths_nm * 1.0e-9)
        alpha_carron = carron_absorption_coefficient(
            energy,
            ggi,
            model.cgi,
        )
        ratio = alpha_minoura / alpha_carron
        if not np.all(np.isfinite(ratio)) or np.any(alpha_carron <= 0.0):
            raise ValueError("independent Carron comparison is non-finite")
        medians.append(float(np.median(np.abs(ratio - 1.0))))
        minimum_ratio = min(minimum_ratio, float(np.min(ratio)))
        maximum_ratio = max(maximum_ratio, float(np.max(ratio)))
    return (
        max(medians),
        minimum_ratio,
        maximum_ratio,
        len(compositions) * energy_points,
    )


def _uniform_composition_reflectance_error(
    stack,
    *,
    absorber_index: int,
    model,
    wavelengths_nm: np.ndarray,
) -> float:
    one_model = replace(
        model,
        ggi_back=model.ggi_front,
        slices=1,
    )
    many_model = replace(one_model, slices=model.slices)
    one = build_device_optical_stack(
        _replace_cigs_model(stack, absorber_index, one_model),
        wavelengths_nm,
    )
    many = build_device_optical_stack(
        _replace_cigs_model(stack, absorber_index, many_model),
        wavelengths_nm,
    )
    one_reflectance = tmm_reflectance(one.layers, wavelengths_nm * 1.0e-9)
    many_reflectance = tmm_reflectance(many.layers, wavelengths_nm * 1.0e-9)
    return float(np.max(np.abs(one_reflectance - many_reflectance)))


def run_cigs_graded_optics_refinement(
    lane: LaneDefinition,
    point: MatrixPoint,
    project_root: Path,
) -> CellMeasurement:
    """Execute one CIGS optical-slice / KK-quadrature matrix cell."""

    options: dict[str, Any] = lane.options
    if _string_option(options, "config_loader", "standard") != "standard":
        raise ValueError("CIGS optical refinement requires config_loader='standard'")
    absorber_name = _string_option(
        options,
        "absorber_layer_name",
        "CIGS_absorber",
    )
    electrical_intervals = _integer_option(
        options,
        "fixed_electrical_intervals_per_layer",
        32,
        minimum=2,
    )
    wavelength_points = _integer_option(
        options,
        "wavelength_points",
        100,
        minimum=8,
    )
    wavelength_min_nm = _finite_option(options, "wavelength_min_nm", 300.0)
    wavelength_max_nm = _finite_option(options, "wavelength_max_nm", 1000.0)
    if wavelength_max_nm <= wavelength_min_nm:
        raise ValueError("wavelength_max_nm must exceed wavelength_min_nm")
    base_order = _integer_option(
        options,
        "base_kk_quadrature_order",
        96,
        minimum=48,
    )
    quadrature_order = _kk_quadrature_order(
        base_order,
        point.tolerance_factor,
    )
    carron_points = _integer_option(
        options,
        "carron_energy_points_per_composition",
        151,
        minimum=16,
    )
    carron_margin = _finite_option(
        options,
        "carron_minimum_excess_above_gap_eV",
        0.15,
    )

    base_stack = load_device_from_yaml(project_root / lane.config_path)
    stack, absorber_index, model = _replace_cigs_resolution(
        base_stack,
        absorber_layer_name=absorber_name,
        slices=point.grid,
        quadrature_order=quadrature_order,
    )
    electrical = electrical_layers(stack)
    grid = multilayer_grid(
        tuple(
            Layer(layer.thickness, electrical_intervals)
            for layer in electrical
        )
    )
    weights = dual_cell_widths(grid)
    generation = _compute_tmm_generation(
        grid,
        stack,
        n_wavelengths=wavelength_points,
        lam_min=wavelength_min_nm,
        lam_max=wavelength_max_nm,
    )
    if generation is None:
        raise RuntimeError("active CIGS optical model produced no TMM generation")
    generation = np.asarray(generation, dtype=float)
    absorbed_flux = float(np.sum(generation * weights))
    peak_generation = float(np.max(generation))
    if not math.isfinite(absorbed_flux) or absorbed_flux <= 0.0:
        raise RuntimeError("CIGS absorbed photon flux must be finite and positive")
    if not math.isfinite(peak_generation) or peak_generation <= 0.0:
        raise RuntimeError("CIGS generation peak must be finite and positive")
    normalized_generation = generation / peak_generation
    device_thickness = float(grid[-1] - grid[0])
    generation_centroid = float(
        np.sum(generation * weights * (grid - grid[0]))
        / (absorbed_flux * device_thickness)
    )

    wavelengths_nm = np.linspace(
        wavelength_min_nm,
        wavelength_max_nm,
        wavelength_points,
    )
    wavelengths_m = wavelengths_nm * 1.0e-9
    _, spectral_flux = load_am15g(wavelengths_nm)
    incident_flux = float(trapezoid(spectral_flux, wavelengths_m))
    optical = build_device_optical_stack(stack, wavelengths_nm)
    reflectance = tmm_reflectance(optical.layers, wavelengths_m)
    all_n = np.concatenate([layer.n for layer in optical.layers])
    all_k = np.concatenate([layer.k for layer in optical.layers])
    nk_causal = bool(
        np.all(np.isfinite(all_n))
        and np.all(np.isfinite(all_k))
        and np.all(all_n > 0.0)
        and np.all(all_k >= 0.0)
    )
    reflectance_violation = max(
        0.0,
        -float(np.min(reflectance)),
        float(np.max(reflectance)) - 1.0,
    )
    photon_budget_excess = max(0.0, absorbed_flux / incident_flux - 1.0)

    carron_median, carron_min, carron_max, carron_count = _carron_benchmark(
        model,
        quadrature_order=quadrature_order,
        energy_points=carron_points,
        minimum_excess_eV=carron_margin,
    )
    uniform_error = _uniform_composition_reflectance_error(
        stack,
        absorber_index=absorber_index,
        model=model,
        wavelengths_nm=wavelengths_nm,
    )
    absorber = stack.layers[absorber_index]
    electrical_gaps = (absorber.params.Eg, absorber.params.Eg_back)
    if electrical_gaps[1] is None:
        raise ValueError("CIGS optical absorber must declare Eg_back")
    optical_gaps = (
        carron_band_gap_eV(model.ggi_front),
        carron_band_gap_eV(model.ggi_back),
    )
    gap_mismatch = max(
        abs(float(electrical) - optical)
        for electrical, optical in zip(
            electrical_gaps,
            optical_gaps,
            strict=True,
        )
    )

    centres = (
        np.arange(model.slices, dtype=float) + 0.5
    ) * (absorber.thickness / model.slices)
    coordinate = grading_coordinate(
        centres,
        absorber.thickness,
        absorber.params.grading_profile,
        absorber.params.grading_char_length,
        absorber.params.grading_direction,
    )
    ggi = ggi_profile_from_coordinate(coordinate, model)
    coordinate_verified = bool(
        coordinate.shape == (model.slices,)
        and ggi.shape == coordinate.shape
        and np.all(np.isfinite(coordinate))
        and np.all((coordinate >= 0.0) & (coordinate <= 1.0))
        and np.all((ggi >= 0.0) & (ggi <= 1.0))
    )
    gate_off = replace(stack, graded_optics=False)
    default_gate_off_inert = bool(
        not has_wavelength_resolved_optics(gate_off)
        and build_device_optical_stack(
            gate_off,
            wavelengths_nm,
        ).graded_physical_layer_indices
        == ()
    )
    topology_verified = bool(
        stack.graded_optics
        and stack.band_grading
        and has_grading_params(absorber.params)
        and optical.graded_physical_layer_indices == (absorber_index,)
        and len(optical.layers) == len(stack.layers) + model.slices - 1
    )
    protocol = _execution_protocol(
        lane,
        absorber_layer_name=absorber_name,
        fixed_electrical_intervals_per_layer=electrical_intervals,
        wavelength_points=wavelength_points,
        wavelength_min_nm=wavelength_min_nm,
        wavelength_max_nm=wavelength_max_nm,
        base_kk_quadrature_order=base_order,
        carron_energy_points_per_composition=carron_points,
        carron_minimum_excess_above_gap_eV=carron_margin,
    )

    return CellMeasurement.from_mapping(
        {
            "observables": {
                "absorbed_photon_flux_m2_s": absorbed_flux,
                "generation_centroid_fraction": generation_centroid,
                "mean_spectral_reflectance": float(np.mean(reflectance)),
                "normalized_generation_profile": normalized_generation,
            },
            "quality": {
                "causal_nk_verified": float(nk_causal),
                "cigs_optical_topology_verified": float(topology_verified),
                "default_gate_off_inert": float(default_gate_off_inert),
                "independent_carron_energy_points_completed": carron_count,
                "max_carron_composition_median_relative_error": carron_median,
                "max_minoura_to_carron_ratio": carron_max,
                "max_electrical_optical_gap_mismatch_eV": gap_mismatch,
                "max_photon_budget_excess_fraction": photon_budget_excess,
                "max_reflectance_bound_violation": reflectance_violation,
                "max_uniform_composition_reflectance_difference": uniform_error,
                "min_minoura_to_carron_ratio": carron_min,
                "positive_absorbed_photon_flux": float(absorbed_flux > 0.0),
                "shared_composition_coordinate_verified": float(
                    coordinate_verified
                ),
            },
            "units": {
                "absorbed_photon_flux_m2_s": "m-2 s-1",
                "causal_nk_verified": "1",
                "cigs_optical_topology_verified": "1",
                "default_gate_off_inert": "1",
                "generation_centroid_fraction": "1",
                "independent_carron_energy_points_completed": "1",
                "max_carron_composition_median_relative_error": "1",
                "max_minoura_to_carron_ratio": "1",
                "max_electrical_optical_gap_mismatch_eV": "eV",
                "max_photon_budget_excess_fraction": "1",
                "max_reflectance_bound_violation": "1",
                "max_uniform_composition_reflectance_difference": "1",
                "mean_spectral_reflectance": "1",
                "min_minoura_to_carron_ratio": "1",
                "normalized_generation_profile": "1",
                "positive_absorbed_photon_flux": "1",
                "shared_composition_coordinate_verified": "1",
            },
            "metadata": {
                **_protocol_metadata(protocol),
                "actual": {
                    "absorber_index": absorber_index,
                    "carron_composition_points": [
                        model.ggi_front,
                        0.5 * (model.ggi_front + model.ggi_back),
                        model.ggi_back,
                    ],
                    "electrical_grid_nodes": int(grid.size),
                    "generation_sha256": hashlib.sha256(
                        np.ascontiguousarray(generation).tobytes()
                    ).hexdigest(),
                    "incident_photon_flux_m2_s": incident_flux,
                    "kk_quadrature_order": quadrature_order,
                    "optical_layer_count": len(optical.layers),
                    "optical_slices": model.slices,
                },
            },
        }
    )


__all__ = ["run_cigs_graded_optics_refinement"]
