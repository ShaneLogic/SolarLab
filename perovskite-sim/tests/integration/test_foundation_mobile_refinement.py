"""Independent space, time and tolerance refinements of prepared mobile ions."""

from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import platform

import numpy as np
import pytest
import scipy

from perovskite_sim.constants import EPS_0, Q
from perovskite_sim.discretization.grid import tanh_grid
from perovskite_sim.experiments.jv_sweep import (
    compute_current_components,
    extract_spatial_snapshot,
)
from perovskite_sim.experiments.quasi_fermi_steady_state import (
    solve_quasi_fermi_steady_state,
)
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.physics.poisson import solve_poisson_prefactored
from perovskite_sim.solver.mol import (
    StateVec,
    assemble_rhs,
    build_material_arrays,
    run_transient,
)
from perovskite_sim.solver.numerical_diagnostics import NumericalDiagnosticsPolicy
from perovskite_sim.solver.tolerances import ComponentwiseAtol
from perovskite_sim.solver import mol
from perovskite_sim.validation.foundation_reference_refinement import (
    _sample_sg_density,
)
from tests.accepted_trajectory import AcceptedTrajectoryMonitor, observe_accepted_radau


pytestmark = pytest.mark.slow
ROOT = Path(__file__).resolve().parents[2]


def _voltage(t):
    return 0.6 * min(float(t) / 1e-4, 1.0)


def _mobile_grid(layers, total):
    """Resolve the early carrier fronts while preserving ionic cell volumes."""
    counts = np.array([3, 5, 3]) * (total // 11)
    if total % 11 or np.min(counts) < 3:
        raise ValueError(
            "mobile reference grid needs at least three intervals per layer"
        )
    middle = tanh_grid(int(counts[1]), layers[1].thickness, 3.0)
    spacing = middle[1]
    pieces, offset = [], 0.0
    for index, (layer, count) in enumerate(zip(layers, counts)):
        local = (
            middle
            if index == 1
            else np.r_[
                0.0,
                np.linspace(spacing, layer.thickness - spacing, int(count) - 1),
                layer.thickness,
            ]
        )
        pieces.append(offset + local if index == 0 else (offset + local)[1:])
        offset += layer.thickness
    grid = np.concatenate(pieces)
    return grid, {
        "kind": "matched_uniform_leads_tanh_absorber",
        "intervals_per_layer": counts.tolist(),
        "absorber_alpha": 3.0,
        "interface_spacing_m": float(spacing),
    }


def _run(base, dual, count, factor, max_step):
    grid, mesh = _mobile_grid(base.layers, count)
    dark = solve_quasi_fermi_steady_state(
        grid, base, illuminated=False, require_contact_certificate=True
    )
    layers = list(base.layers)
    layers[1] = replace(
        layers[1],
        params=replace(
            layers[1].params,
            P0=1e22,
            D_ion=1e-12,
            P0_neg=0.7e22 if dual else 0.0,
            D_ion_neg=0.5e-12 if dual else 0.0,
            P_lim_neg=1e26,
        ),
    )
    stack = replace(base, layers=tuple(layers), ion_steric_diffusion_only=True)
    material = build_material_arrays(grid, stack)
    initial = StateVec.pack(
        dark.y[: grid.size],
        dark.y[grid.size : 2 * grid.size],
        material.P_ion0.copy(),
        material.P_ion0_neg.copy() if dual else None,
    )
    times = np.unique(
        np.r_[0.0, np.geomspace(1e-10, 1e-4, 33), np.geomspace(1e-4, 2e-3, 17)]
    )
    parts = []
    diagnostics = []
    trajectories = []
    previous = initial
    for index, (start, stop) in enumerate(((0.0, 1e-4), (1e-4, 2e-3))):
        selected = times[(times >= start) & (times <= stop)]
        monitor = AcceptedTrajectoryMonitor(
            grid.size,
            initial,
            weights=material.dx_cell,
            ion_capacities=(material.P_lim_node,)
            + ((material.P_lim_neg_node,) if dual else ()),
        )
        with observe_accepted_radau(mol, monitor):
            solved = run_transient(
                grid,
                previous,
                (start, stop),
                selected,
                stack,
                mat=material,
                V_app=_voltage,
                rtol=1e-4 * factor,
                atol=ComponentwiseAtol(refinement_factor=factor),
                max_step=max_step,
                numerical_diagnostics=NumericalDiagnosticsPolicy.research_strict(
                    bulk_srh_denominator_floor_s_m3=0.0
                ),
                max_nfev=200000,
            )
        assert solved.success, solved.message
        assert solved.numerical_diagnostics.would_pass_strict
        previous = solved.y[:, -1]
        parts.append(solved.y if index == 0 else solved.y[:, 1:])
        diagnostics.append(asdict(solved.numerical_diagnostics))
        trajectories.append(monitor.report())
    states = np.concatenate(parts, axis=1)
    assert states.shape[1] == times.size
    assert np.all(np.isfinite(states)) and np.all(states[: 2 * grid.size] > 0.0)
    positions = []
    offset = 0.0
    for layer in layers:
        positions.extend(offset + (np.arange(17) + 0.5) * layer.thickness / 17)
        offset += layer.thickness
    positions = np.asarray(positions)
    eps_face = (
        EPS_0
        * 2
        * material.eps_r[:-1]
        * material.eps_r[1:]
        / (material.eps_r[:-1] + material.eps_r[1:])
    )
    maximum_spread = 0.0
    current, potential, electrons, holes = [], [], [], []
    for t, state in zip(times, states.T):
        bias = _voltage(t)
        rate = StateVec.unpack(
            assemble_rhs(t, state, grid, stack, material, V_app=bias), grid.size
        )
        rho_rate = Q * (rate.p - rate.n + rate.P - (rate.P_neg if dual else 0.0))
        phi_rate = solve_poisson_prefactored(
            material.poisson_factor,
            rho_rate,
            0.0,
            -material.junction_polarity * (6000.0 if t < 1e-4 else 0.0),
        )
        displacement = (
            material.junction_polarity * eps_face * np.diff(phi_rate) / np.diff(grid)
        )
        total = (
            compute_current_components(grid, state, stack, bias, mat=material).J_total
            + displacement
        )
        maximum_spread = max(maximum_spread, float(np.ptp(total)))
        current.append(float(np.average(total, weights=np.diff(grid))))
        phi = extract_spatial_snapshot(grid, state, stack, bias, mat=material).phi
        potential.append(np.interp(positions, grid, phi))
        electrons.append(
            np.log(
                _sample_sg_density(
                    grid,
                    state[: grid.size],
                    phi + material.chi,
                    positions,
                    material.V_T_device,
                )
            )
        )
        holes.append(
            np.log(
                _sample_sg_density(
                    grid,
                    state[grid.size : 2 * grid.size],
                    phi + material.chi + material.Eg,
                    positions,
                    material.V_T_device,
                    drift_sign=-1.0,
                )
            )
        )
    assert maximum_spread < 1e-4
    ions = {}
    for name, block, background, capacity in (
        ("positive", 2, material.P_ion0, material.P_lim_node),
        ("negative", 3, material.P_ion0_neg, material.P_lim_neg_node),
    ):
        if name == "negative" and not dual:
            continue
        values = states[block * grid.size : (block + 1) * grid.size]
        inventory = material.dx_cell @ values
        drift = float(np.max(np.abs(inventory / inventory[0] - 1)))
        assert drift < 1e-10
        active = background > 0.0
        assert np.all(values[active] > 0.0)
        assert np.max(values[active] / capacity[active, None]) < 1.0
        np.testing.assert_array_equal(values[~active], 0.0)
        reference = float(np.max(background))
        assert np.max(np.abs(values[:, -1] - values[:, 0])) > 1e-3 * reference
        expected_inventory = reference * layers[1].thickness
        assert inventory[0] == pytest.approx(expected_inventory, rel=1e-12)
        ions[name] = {
            "inventory_m2": inventory,
            "maximum_relative_drift": drift,
            "profile_normalized": np.array(
                [np.interp(positions[17:34], grid, row) / reference for row in values.T]
            ),
        }
    return {
        "grid": count,
        "factor": factor,
        "max_step_s": max_step,
        "mesh": mesh,
        "times_s": times,
        "positions_m": grid,
        "profile_positions_m": positions,
        "states_m3": states,
        "potential_profiles_V": np.array(potential),
        "electron_log_profiles": np.array(electrons),
        "hole_log_profiles": np.array(holes),
        "current_A_m2": np.array(current),
        "maximum_current_spread_A_m2": maximum_spread,
        "ions": ions,
        "diagnostics": diagnostics,
        "accepted_trajectories": trajectories,
    }


@pytest.fixture(scope="module", params=[False, True], ids=["single", "dual"])
def cases(request):
    base = load_device_from_yaml(
        ROOT / "tests/fixtures/configs/tpv_physical_reference.yaml"
    )
    cache = {}
    destination = os.environ.get("SOLARLAB_F7_MOBILE_EVIDENCE_DIRECTORY")
    if destination:

        def save_evidence():
            sources = (
                "tests/accepted_trajectory.py",
                "perovskite_sim/solver/mol.py",
                "perovskite_sim/experiments/jv_sweep.py",
                "perovskite_sim/experiments/quasi_fermi_steady_state.py",
                "perovskite_sim/validation/foundation_reference_refinement.py",
                "tests/fixtures/configs/tpv_physical_reference.yaml",
                "tests/integration/test_foundation_mobile_refinement.py",
            )
            payload = {
                "kind": "foundation_prepared_mobile_ion_refinement",
                "dual": request.param,
                "acceptance_record": "paired pytest/JUnit result",
                "cases": list(cache.values()),
                "environment": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "scipy": scipy.__version__,
                },
                "source_sha256": {
                    name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                    for name in sources
                },
                "history": "uniform compensated ions at dark carrier equilibrium; light on at zero; 0-to-0.6 V ramp in 1e-4 s, hold to 2e-3 s",
            }

            def serialize(value):
                if isinstance(value, np.ndarray):
                    return value.tolist()
                if isinstance(value, np.generic):
                    return value.item()
                raise TypeError(type(value).__name__)

            path = Path(destination) / (
                "FoundationMobileDualV1.json"
                if request.param
                else "FoundationMobileSingleV1.json"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(payload, default=serialize, allow_nan=False) + "\n"
            )

        request.addfinalizer(save_evidence)

    def solve(grid=352, factor=0.01, max_step=5e-6):
        key = grid, factor, max_step
        if key not in cache:
            cache[key] = _run(base, request.param, grid, factor, max_step)
        for trajectory in cache[key]["accepted_trajectories"]:
            assert trajectory["passed"], trajectory
        return cache[key]

    return solve


def _compare(coarse, fine):
    np.testing.assert_array_equal(coarse["times_s"], fine["times_s"])
    assert (
        np.max(np.abs(coarse["potential_profiles_V"] - fine["potential_profiles_V"]))
        < 1e-3
    )
    for species in ("electron", "hole"):
        assert np.max(
            np.abs(coarse[f"{species}_log_profiles"] - fine[f"{species}_log_profiles"])
        ) < np.log(1.01)
    for species in fine["ions"]:
        assert (
            np.max(
                np.abs(
                    coarse["ions"][species]["profile_normalized"]
                    - fine["ions"][species]["profile_normalized"]
                )
            )
            < 0.01
        )
    assert (
        np.max(np.abs(coarse["current_A_m2"] - fine["current_A_m2"])) < 0.005 * Q * 1e21
    )


def test_space_refinement_preserves_each_ion_inventory_and_profiles(cases):
    values = [cases(grid=count) for count in (176, 352, 704)]
    _compare(values[-2], values[-1])


def test_time_refinement_preserves_the_same_ramp_and_hold(cases):
    values = [cases(max_step=step) for step in (2e-5, 1e-5, 5e-6)]
    _compare(values[-2], values[-1])


def test_tolerance_refinement_preserves_mobile_and_electronic_states(cases):
    values = [cases(factor=factor) for factor in (1.0, 0.1, 0.01)]
    _compare(values[-2], values[-1])
