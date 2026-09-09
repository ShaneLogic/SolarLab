"""Physical p-i-n reference across space, solver accuracy and voltage sampling."""

from dataclasses import asdict
import math
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from perovskite_sim.constants import Q
from perovskite_sim.discretization.fe_operators import bernoulli
from perovskite_sim.discretization.grid import Layer, multilayer_grid, tanh_grid
from perovskite_sim.experiments.jv_sweep import compute_metrics
from perovskite_sim.experiments.quasi_fermi_steady_state import (
    solve_quasi_fermi_jv_sweep,
    solve_quasi_fermi_steady_state,
)
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.physics.generation import (
    beer_lambert_generation,
    dual_cell_integral,
)
from perovskite_sim.physics.recombination import total_recombination
from perovskite_sim.physics.tunneling_channel_device import (
    physical_quasi_fermi_levels_eV,
)
from perovskite_sim.solver.mol import build_material_arrays
from .dae_refinement import _finite_option, _protocol_metadata
from .numerical_certificate import LaneDefinition, MatrixPoint
from .refinement_runner import CellMeasurement


def foundation_reference_protocol(lane: LaneDefinition) -> dict:
    protocol = {
        "schema_version": "foundation-physical-reference-v1",
        "lane_id": lane.lane_id,
        "executor_version": lane.executor_version,
        "temperature_K": 300.0,
        "grid_values": list(lane.grid_values),
        "interval_ratio": [3, 16, 3],
        "grid_alpha": 0.0,
        "tolerance_factors": list(lane.tolerance_factors),
        "voltage_steps_V": [0.02, 0.01, 0.005],
        "maximum_voltage_V": 1.6,
        "photon_energy_eV": _finite_option(lane.options, "photon_energy_eV", 2.0),
        "profile_points_per_layer": 17,
        "curve_points": 321,
        "open_circuit_root_tolerance_V": 1e-10,
        "maximum_power_voltage_tolerance_V": 1e-7,
        "initialization": "density_predictor_then_certified_qf",
        "current_normalization": "incident_photon_charge_flux",
        "recombination_normalization": "incident_photon_flux_over_total_thickness",
        "physical_contract": "docs/FoundationReferenceValidationContractV1.md",
    }
    if lane.executor_version == "v2":
        protocol.update(
            schema_version="foundation-physical-reference-v2",
            grid_kind="interface_balanced_tanh",
            grid_alpha=_finite_option(lane.options, "absorber_alpha", 3.0),
            profile_points_per_layer=33,
            profile_interpolation="scharfetter_gummel_constant_face_flux",
            profile_boundary_distances_m=[
                5e-11,
                1e-10,
                2e-10,
                5e-10,
                1e-9,
                2e-9,
                5e-9,
                1e-8,
            ],
            physical_contract="docs/FoundationReferenceValidationContractV2.md",
            initialization="certified_coarse_qf_prolongation",
            initialization_grid_intervals=88,
        )
    elif lane.executor_version != "v1":
        raise ValueError("unknown foundation steady-state executor version")
    return protocol


def _balanced_reference_grid(layers, intervals, absorber_alpha=3.0):
    """Resolve both sides of each interface with the same physical spacing."""
    target = tanh_grid(int(intervals[1]), layers[1].thickness, absorber_alpha)[1]
    alphas = tuple(
        brentq(
            lambda alpha: tanh_grid(int(count), layer.thickness, alpha)[1] - target,
            1e-8,
            12.0,
            xtol=1e-13,
        )
        for layer, count in zip(layers, intervals)
    )
    grid = multilayer_grid(
        [Layer(layer.thickness, int(count)) for layer, count in zip(layers, intervals)],
        alpha=alphas,
    )
    return grid, alphas


def _sample_sg_density(
    grid, density, potential, positions, thermal_voltage, *, drift_sign=1.0
):
    """Local constant-flux SG reconstruction, including the zero-field limit."""
    grid, density, potential, positions = (
        np.asarray(value, dtype=float)
        for value in (grid, density, potential, positions)
    )
    if (
        not np.isfinite(thermal_voltage)
        or thermal_voltage <= 0.0
        or drift_sign not in (-1.0, 1.0)
    ):
        raise ValueError(
            "SG sampling requires positive thermal voltage and a carrier charge sign"
        )
    if (
        grid.ndim != 1
        or grid.size < 2
        or np.any(np.diff(grid) <= 0.0)
        or density.shape != grid.shape
        or potential.shape != grid.shape
        or np.any(density <= 0.0)
        or np.any(positions < grid[0])
        or np.any(positions > grid[-1])
        or not all(
            np.all(np.isfinite(value))
            for value in (grid, density, potential, positions)
        )
    ):
        raise ValueError(
            "SG sampling requires positive density on a finite increasing grid"
        )
    faces = np.clip(
        np.searchsorted(grid, positions, side="right") - 1, 0, grid.size - 2
    )
    fraction = (positions - grid[faces]) / (grid[faces + 1] - grid[faces])
    xi = drift_sign * (potential[faces + 1] - potential[faces]) / thermal_voltage
    weight = fraction * bernoulli(xi) / bernoulli(fraction * xi)
    sampled = (1.0 - weight) * density[faces] + weight * density[faces + 1]
    if np.any(~np.isfinite(sampled)) or np.any(sampled <= 0.0):
        raise ValueError("SG reconstruction produced an invalid physical density")
    return sampled


def _finest_relative_change(values):
    values = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values == 0):
        raise ValueError(
            "the reference requires resolved, finite nonzero photovoltaic metrics"
        )
    return float(abs(values[-1] - values[-2]) / abs(values[-1]))


def run_foundation_reference_steady_state(
    lane: LaneDefinition,
    point: MatrixPoint,
    project_root: Path,
) -> CellMeasurement:
    protocol = foundation_reference_protocol(lane)
    stack = load_device_from_yaml(project_root / lane.config_path)
    layers = electrical_layers(stack)
    if len(layers) != 3 or stack.T != 300.0 or point.grid % 22:
        raise ValueError(
            "the foundation reference requires three 300 K layers and 22*k intervals"
        )
    length = math.fsum(layer.thickness for layer in layers)
    if not np.allclose(
        [layer.thickness / length for layer in layers],
        np.array([3, 16, 3]) / 22,
        rtol=0.0,
        atol=1e-14,
    ):
        raise ValueError(
            "the foundation reference has changed its physical layer proportions"
        )
    intervals = np.array([3, 16, 3]) * (point.grid // 22)
    resolved_profiles = lane.executor_version == "v2"
    if resolved_profiles:
        grid, grid_alphas = _balanced_reference_grid(
            layers, intervals, protocol["grid_alpha"]
        )
    else:
        grid = np.linspace(0.0, length, point.grid + 1)
        grid_alphas = (0.0, 0.0, 0.0)
    material = build_material_arrays(grid, stack)
    if np.any(material.P_ion0) or (
        material.P_ion0_neg is not None and np.any(material.P_ion0_neg)
    ):
        raise ValueError("the steady foundation reference is ion-free")
    photon_energy = float(protocol["photon_energy_eV"])
    if photon_energy < float(np.max(material.Eg_phys)) or stack.Phi <= 0.0:
        raise ValueError(
            "the declared monochromatic source must exceed the physical gap"
        )
    current_reference = Q * stack.Phi
    power_reference = photon_energy * current_reference
    recombination_reference = stack.Phi / length
    controls = {
        "newton_residual_tolerance": _finite_option(
            lane.options, "base_newton_residual_tolerance", 1e-8
        )
        * point.tolerance_factor,
        "poisson_tolerance_V": _finite_option(
            lane.options, "base_poisson_tolerance_V", 1e-10
        )
        * point.tolerance_factor,
        "finite_difference_step": _finite_option(
            lane.options, "base_finite_difference_step", 1e-5
        )
        * math.sqrt(point.tolerance_factor),
        "continuity_tolerance_A_m2": 1e-4,
        "current_spread_tolerance_A_m2": 1e-4,
        "max_newton_iterations": 50,
    }
    voltages = np.arange(321) * 0.005
    if resolved_profiles and point.grid > protocol["initialization_grid_intervals"]:
        seed_intervals = np.array([3, 16, 3]) * (
            protocol["initialization_grid_intervals"] // 22
        )
        seed_grid, _ = _balanced_reference_grid(
            layers, seed_intervals, protocol["grid_alpha"]
        )
        seed = solve_quasi_fermi_steady_state(
            seed_grid,
            stack,
            0.0,
            use_density_predictor=True,
            require_contact_certificate=True,
            **controls,
        )
        short_circuit = solve_quasi_fermi_steady_state(
            grid,
            stack,
            0.0,
            mat=material,
            initial_state=seed,
            initial_state_grid=seed_grid,
            illumination_steps=(1.0,),
            require_contact_certificate=True,
            **controls,
        )
    else:
        short_circuit = solve_quasi_fermi_steady_state(
            grid,
            stack,
            0.0,
            mat=material,
            use_density_predictor=True,
            require_contact_certificate=True,
            **controls,
        )
    sweep = solve_quasi_fermi_jv_sweep(
        grid,
        stack,
        voltages,
        mat=material,
        P_in_W_m2=power_reference,
        initial_short_circuit_state=short_circuit,
        require_contact_certificate=True,
        stop_after_voc=True,
        voc_stop_grid_V=voltages[::4],
        **controls,
    )
    if not sweep.certified or not sweep.metrics.voc_bracketed:
        raise ValueError(
            "the common reference must have a certified photovoltaic zero-current bracket"
        )
    metrics = [
        compute_metrics(
            sweep.voltages_V[::stride],
            sweep.currents_A_m2[::stride],
            P_in=power_reference,
            V_oc_max=float(np.min(material.Eg_phys)),
        )
        for stride in (4, 2, 1)
    ]
    if any(not item.voc_bracketed for item in metrics):
        raise ValueError(
            "every nested voltage sampling must retain its own open-circuit bracket"
        )
    cache = {float(state.V_app): state for state in sweep.points}

    def illuminated_state(voltage):
        voltage = float(voltage)
        if voltage not in cache:
            nearest = min(cache, key=lambda value: abs(value - voltage))
            cache[voltage] = solve_quasi_fermi_steady_state(
                grid,
                stack,
                voltage,
                mat=material,
                initial_state=cache[nearest],
                require_contact_certificate=True,
                illumination_steps=(1.0,),
                **controls,
            )
        return cache[voltage]

    crossing = int(np.flatnonzero(sweep.currents_A_m2 <= 0.0)[0])
    if crossing == 0:
        raise ValueError(
            "the reference short-circuit point must deliver positive photocurrent"
        )
    root = brentq(
        lambda voltage: illuminated_state(voltage).current_A_m2,
        float(sweep.voltages_V[crossing - 1]),
        float(sweep.voltages_V[crossing]),
        xtol=float(protocol["open_circuit_root_tolerance_V"]),
        rtol=1e-12,
    )
    optimum = minimize_scalar(
        lambda voltage: -voltage * illuminated_state(voltage).current_A_m2,
        bounds=(0.0, root),
        method="bounded",
        options={"xatol": protocol["maximum_power_voltage_tolerance_V"]},
    )
    if not optimum.success:
        raise ValueError("the independent maximum-power reference did not converge")
    equilibrium = solve_quasi_fermi_steady_state(
        grid,
        stack,
        0.0,
        mat=material,
        illuminated=False,
        require_contact_certificate=True,
        **controls,
    )
    states = {
        "equilibrium": equilibrium,
        "short_circuit": sweep.points[0],
        "open_circuit": illuminated_state(root),
        "maximum_power": illuminated_state(optimum.x),
    }
    positions = []
    offset = 0.0
    for layer in layers:
        local_positions = (np.arange(17) + 0.5) * layer.thickness / 17
        if resolved_profiles:
            edge_distances = np.asarray(protocol["profile_boundary_distances_m"])
            local_positions = np.unique(
                np.r_[local_positions, edge_distances, layer.thickness - edge_distances]
            )
        positions.extend(offset + local_positions)
        offset += layer.thickness
    positions = np.asarray(positions)
    observed = {
        "voc_V": metrics[-1].V_oc,
        "jsc_A_m2": metrics[-1].J_sc,
        "fill_factor": metrics[-1].FF,
        "power_conversion_fraction": metrics[-1].PCE,
        "open_circuit_root_V": root,
        "maximum_power_voltage_V": float(optimum.x),
        "photovoltaic_current_normalized_trace": np.maximum(
            np.interp(voltages, sweep.voltages_V, sweep.currents_A_m2, right=0.0),
            0.0,
        )
        / current_reference,
    }
    raw = {}
    equilibrium_levels = None
    for name, state in states.items():
        n, p = state.y[: grid.size], state.y[grid.size : 2 * grid.size]
        efn, efp = physical_quasi_fermi_levels_eV(
            potential_V=state.phi,
            affinity_eV=material.chi_phys,
            band_gap_eV=material.Eg_phys,
            electron_density_m3=n,
            hole_density_m3=p,
            conduction_dos_m3=material.N_C_physical,
            valence_dos_m3=material.N_V_physical,
            thermal_voltage_V=material.V_T_device,
        )
        recombination = total_recombination(
            n,
            p,
            material.ni_sq,
            material.tau_n,
            material.tau_p,
            material.n1,
            material.p1,
            material.B_rad,
            material.C_n,
            material.C_p,
        )
        fields = {
            "potential_V": state.phi,
            "electron_log_density": np.log(n),
            "hole_log_density": np.log(p),
            "electron_fermi_eV": efn,
            "hole_fermi_eV": efp,
            "recombination_normalized": recombination / recombination_reference,
        }
        for field, values in fields.items():
            observed[f"{name}_{field}"] = np.interp(positions, grid, values)
        if resolved_profiles:
            sampled_n = _sample_sg_density(
                grid, n, state.phi + material.chi, positions, material.V_T_device
            )
            sampled_p = _sample_sg_density(
                grid,
                p,
                state.phi + material.chi + material.Eg,
                positions,
                material.V_T_device,
                drift_sign=-1.0,
            )
            sampled_phi = np.interp(positions, grid, state.phi)

            def sample(values):
                return np.interp(positions, grid, values)

            sampled_efn, sampled_efp = physical_quasi_fermi_levels_eV(
                potential_V=sampled_phi,
                affinity_eV=sample(material.chi_phys),
                band_gap_eV=sample(material.Eg_phys),
                electron_density_m3=sampled_n,
                hole_density_m3=sampled_p,
                conduction_dos_m3=sample(material.N_C_physical),
                valence_dos_m3=sample(material.N_V_physical),
                thermal_voltage_V=material.V_T_device,
            )
            sampled_recombination = total_recombination(
                sampled_n,
                sampled_p,
                sample(material.ni_sq),
                sample(material.tau_n),
                sample(material.tau_p),
                sample(material.n1),
                sample(material.p1),
                sample(material.B_rad),
                sample(material.C_n),
                sample(material.C_p),
            )
            observed.update(
                {
                    f"{name}_electron_log_density": np.log(sampled_n),
                    f"{name}_hole_log_density": np.log(sampled_p),
                    f"{name}_electron_fermi_eV": sampled_efn,
                    f"{name}_hole_fermi_eV": sampled_efp,
                    f"{name}_recombination_normalized": sampled_recombination
                    / recombination_reference,
                }
            )
        raw[name] = {
            "voltage_V": state.V_app,
            "density_state_m3": state.y.tolist(),
            "potential_V": state.phi.tolist(),
            "electron_current_A_m2": state.electron_face_current_A_m2.tolist(),
            "hole_current_A_m2": state.hole_face_current_A_m2.tolist(),
            "net_recombination_m3_s": recombination.tolist(),
            "electron_fermi_eV": efn.tolist(),
            "hole_fermi_eV": efp.tolist(),
        }
        if name == "equilibrium":
            equilibrium_levels = np.r_[efn, efp]
    all_states = [equilibrium, *cache.values()]
    generation = beer_lambert_generation(grid, material.alpha, stack.Phi)
    maximum_power = -float(optimum.fun)
    quality = {
        "all_states_certified": float(all(state.certified for state in all_states)),
        "contacts_thermodynamically_consistent": float(
            all(
                state.contact_thermodynamic_status == "certified"
                for state in all_states
            )
        ),
        "maximum_continuity_bound_A_m2": max(
            max(state.electron_continuity_bound_A_m2, state.hole_continuity_bound_A_m2)
            for state in all_states
        ),
        "maximum_face_current_spread_A_m2": max(
            state.face_current_spread_A_m2 for state in all_states
        ),
        "maximum_poisson_residual_C_m2": max(
            state.poisson_residual_C_m2 for state in all_states
        ),
        "maximum_residual_over_limit": max(
            state.max_normalized_cell_residual / state.numerical_residual_limit
            for state in all_states
        ),
        "equilibrium_current_A_m2": float(
            np.max(np.abs(equilibrium.total_face_current_A_m2))
        ),
        "equilibrium_fermi_span_eV": float(np.ptp(equilibrium_levels)),
        "open_circuit_current_A_m2": abs(states["open_circuit"].current_A_m2),
        "maximum_density_over_dos": max(
            max(
                float(np.max(state.y[: grid.size] / material.N_C_physical)),
                float(
                    np.max(state.y[grid.size : 2 * grid.size] / material.N_V_physical)
                ),
            )
            for state in all_states
        ),
        "carrier_densities_positive": float(
            all(
                np.all(np.isfinite(state.y[: 2 * grid.size]))
                and np.all(state.y[: 2 * grid.size] > 0.0)
                for state in all_states
            )
        ),
        "absorbed_photon_fraction": dual_cell_integral(grid, generation) / stack.Phi,
        "maximum_power_conversion_fraction": maximum_power / power_reference,
        "voltage_voc_absolute_change_V": abs(metrics[-1].V_oc - metrics[-2].V_oc),
        "voltage_jsc_relative_change": _finest_relative_change(
            [item.J_sc for item in metrics]
        ),
        "voltage_ff_relative_change": _finest_relative_change(
            [item.FF for item in metrics]
        ),
        "voltage_pce_relative_change": _finest_relative_change(
            [item.PCE for item in metrics]
        ),
        "sampled_voc_root_error_V": abs(metrics[-1].V_oc - root),
        "sampled_power_relative_error": abs(
            metrics[-1].PCE * power_reference / maximum_power - 1.0
        ),
    }
    units = {name: "1" for name in [*observed, *quality]}
    for name in units:
        for suffix, unit in (
            ("A_m2", "A m-2"),
            ("C_m2", "C m-2"),
            ("_eV", "eV"),
            ("_V", "V"),
            ("_m3", "m-3"),
        ):
            if name.endswith(suffix):
                units[name] = unit
                break
    return CellMeasurement.from_mapping(
        {
            "observables": observed,
            "quality": quality,
            "units": units,
            "metadata": {
                **_protocol_metadata(protocol),
                "actual": {
                    "solve_controls": controls,
                    "intervals": int(grid.size - 1),
                    "density_basin_initializations": short_circuit.density_basin_initializations,
                    "minimum_carrier_density_m3": min(
                        float(np.min(state.y[: 2 * grid.size])) for state in all_states
                    ),
                    "intervals_per_layer": intervals.tolist(),
                    "grid_alphas": list(grid_alphas),
                    "incident_power_W_m2": power_reference,
                    "fixed_current_reference_A_m2": current_reference,
                    "fixed_recombination_reference_m3_s": recombination_reference,
                    "voltage_sampling_absolute_changes": {
                        name: np.abs(
                            np.diff([getattr(item, name) for item in metrics])
                        ).tolist()
                        for name in ("V_oc", "J_sc", "FF", "PCE")
                    },
                    "raw": {
                        "positions_m": grid.tolist(),
                        "profile_positions_m": positions.tolist(),
                        "voltage_V": sweep.voltages_V.tolist(),
                        "current_A_m2": sweep.currents_A_m2.tolist(),
                        "voltage_metrics": [asdict(item) for item in metrics],
                        "operating_states": raw,
                        "generation_m3_s": generation.tolist(),
                    },
                },
            },
        }
    )
