"""Independently refine two-dimensional transport with active TE and contacts."""

from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import platform

import numpy as np
import pytest
import scipy

from perovskite_sim.constants import V_T
from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.experiments.jv_sweep import compute_current_components
from perovskite_sim.models.device import DeviceStack, LayerSpec
from perovskite_sim.models.parameters import MaterialParams
from perovskite_sim.solver.mol import StateVec, build_material_arrays, run_transient
from perovskite_sim.solver import mol
from perovskite_sim.twod import solver_2d
from perovskite_sim.twod.grid_2d import Grid2D
from perovskite_sim.twod.microstructure import Microstructure
from perovskite_sim.twod.solver_2d import (
    build_material_arrays_2d,
    extract_snapshot_2d,
    run_transient_2d,
)
from tests.accepted_trajectory import AcceptedTrajectoryMonitor, observe_accepted_radau


pytestmark = pytest.mark.slow


def _stack():
    ni = 1e24 * math.exp(-1.2 / (2 * V_T))
    params = MaterialParams(
        eps_r=10.0,
        mu_n=0.02,
        mu_p=0.02,
        chi=4.0,
        Eg=1.2,
        Nc300=1e24,
        Nv300=1e24,
        ni=ni,
        n1=ni,
        p1=ni,
        N_A=0.0,
        N_D=1e20,
        tau_n=1e-6,
        tau_p=1e-6,
        B_rad=0.0,
        C_n=0.0,
        C_p=0.0,
        D_ion=0.0,
        P0=0.0,
        P_lim=1e26,
        alpha=0.0,
    )
    return DeviceStack(
        layers=(
            LayerSpec("left", 100e-9, params, role="absorber"),
            LayerSpec("right", 100e-9, replace(params, chi=3.8), role="ETL"),
        ),
        Phi=0.0,
        built_in_potential_mode="semiconductor_work_function",
        te_physical_norm=True,
        mode="full",
        S_n_left=1e3,
        S_p_right=0.0,
    )


@pytest.fixture(scope="module")
def axes(request):
    stack = _stack()
    density_scales = np.array([1e20, stack.layers[0].params.ni ** 2 / 1e20])
    cache = {}
    destination = os.environ.get("SOLARLAB_F7_TWOD_AXES_EVIDENCE")
    if destination:

        def save_evidence():
            root = Path(__file__).resolve().parents[2]
            sources = (
                "tests/accepted_trajectory.py",
                "perovskite_sim/solver/mol.py",
                "perovskite_sim/twod/solver_2d.py",
                "perovskite_sim/twod/continuity_2d.py",
                "perovskite_sim/twod/poisson_2d.py",
                "perovskite_sim/physics/thermionic_transport.py",
                "tests/integration/test_foundation_twod_axis_refinement.py",
            )
            payload = {
                "kind": "foundation_twod_independent_axis_refinement",
                "cases": list(cache.values()),
                "acceptance_record": "paired pytest/JUnit result",
                "environment": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "scipy": scipy.__version__,
                },
                "source_sha256": {
                    name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                    for name in sources
                },
                "carrier_reference_m3": density_scales.tolist(),
                "voltage_V": 0.02,
                "duration_s": 1e-10,
                "maximum_step_s": 1e-11,
                "rtol": 1e-8,
                "atol_m3": 1e-4,
            }

            def serialize(value):
                if isinstance(value, np.ndarray):
                    return value.tolist()
                if isinstance(value, np.generic):
                    return value.item()
                raise TypeError(type(value).__name__)

            path = Path(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(payload, default=serialize, allow_nan=False) + "\n"
            )

        request.addfinalizer(save_evidence)

    def solve(vertical, horizontal):
        if (vertical, horizontal) in cache:
            return cache[(vertical, horizontal)]
        y = multilayer_grid(
            [Layer(layer.thickness, vertical) for layer in stack.layers], alpha=2.0
        )
        grid = Grid2D(x=np.linspace(0, 300e-9, horizontal + 1), y=y)
        one = build_material_arrays(y, stack)
        two = build_material_arrays_2d(
            grid, stack, Microstructure(), lateral_bc="neumann"
        )
        assert one.te_physical_norm and two.te_physical_norm
        assert one.S_n_L == two.S_n_top == 1e3
        assert one.S_p_R == two.S_p_bot == 0.0
        n = density_scales[0] * (1 + 0.05 * np.sin(np.pi * y / y[-1]))
        p = np.full(y.size, density_scales[1])
        n[-1], p[0] = one.n_R, one.p_L
        initial = StateVec.pack(n, p, one.P_ion0)
        extruded = np.concatenate(
            [
                np.broadcast_to(density[:, None], (grid.Ny, grid.Nx)).ravel()
                for density in (n, p)
            ]
        )
        capped = compute_current_components(y, initial, stack, 0.02, mat=one)
        unlimited = compute_current_components(
            y, initial, stack, 0.02, mat=replace(one, interface_faces=())
        )
        cap_effect = np.max(np.abs(capped.J_n - unlimited.J_n)) / np.max(
            np.abs(unlimited.J_n)
        )
        assert cap_effect > 0.5
        accepted1 = AcceptedTrajectoryMonitor(
            y.size,
            initial,
            weights=one.dx_cell,
            ion_capacities=(one.P_lim_node,),
        )
        with observe_accepted_radau(mol, accepted1):
            result1 = run_transient(
                y,
                initial,
                (0.0, 1e-10),
                np.array([1e-10]),
                stack,
                mat=one,
                illuminated=False,
                V_app=0.02,
                rtol=1e-8,
                atol=1e-4,
                max_step=1e-11,
                max_nfev=50000,
            )
        assert result1.success, result1.message
        accepted1.assert_valid()
        accepted2 = AcceptedTrajectoryMonitor(grid.Nx * grid.Ny, extruded)
        with observe_accepted_radau(solver_2d, accepted2):
            result2 = run_transient_2d(
                extruded,
                two,
                V_app=0.02,
                t_end=1e-10,
                rtol=1e-8,
                atol=1e-4,
                max_step=1e-11,
                max_nfev=50000,
            )
        accepted2.assert_valid()
        final2 = result2.reshape(2, grid.Ny, grid.Nx)
        final1 = result1.y[: 2 * grid.Ny, -1].reshape(2, grid.Ny)
        assert np.all(final2 > 0.0) and np.all(np.isfinite(final2))
        np.testing.assert_allclose(
            final2, np.broadcast_to(final2[:, :, :1], final2.shape), rtol=1e-9, atol=0.0
        )
        snapshot = extract_snapshot_2d(result2, two, V_app=0.02)
        current1 = compute_current_components(y, result1.y[:, -1], stack, 0.02, mat=one)
        errors = np.max(np.abs(final2[:, :, 0] - final1), axis=1) / density_scales
        currents1 = np.array([current1.J_n, current1.J_p])
        currents2 = -one.junction_polarity * np.array(
            [snapshot.Jy_n[:, 0], snapshot.Jy_p[:, 0]]
        )
        current_errors = np.max(np.abs(currents2 - currents1), axis=1) / np.max(
            np.abs(currents1), axis=1
        )
        cache[(vertical, horizontal)] = {
            "vertical_intervals_per_layer": vertical,
            "lateral_intervals": horizontal,
            "positions_x_m": grid.x,
            "positions_y_m": y,
            "state_1d_m3": final1,
            "state_2d_m3": final2,
            "current_1d_A_m2": currents1,
            "current_2d_A_m2": currents2,
            "density_reference_m3": density_scales,
            "carrier_errors": errors,
            "current_errors": current_errors,
            "active_cap_relative_change": float(cap_effect),
            "accepted_trajectory_1d": accepted1.report(),
            "accepted_trajectory_2d": accepted2.report(),
        }
        return cache[(vertical, horizontal)]

    return solve


def test_vertical_refinement_approaches_the_same_active_transport_limit(axes):
    levels = [axes(count, 2) for count in (16, 32, 64)]
    for key in ("carrier_errors", "current_errors"):
        values = np.array([level[key] for level in levels])
        assert np.all(values[-1] < values[-2]) and np.all(values[-2] < values[-3])
        assert np.max(values[-1]) < 0.005


def test_lateral_refinement_preserves_the_uniform_state_and_currents(axes):
    levels = [axes(32, count) for count in (2, 4, 8)]
    for level in levels[:-1]:
        np.testing.assert_allclose(
            level["state_2d_m3"][:, :, 0],
            levels[-1]["state_2d_m3"][:, :, 0],
            rtol=1e-8,
            atol=0.0,
        )
        np.testing.assert_allclose(
            level["current_2d_A_m2"],
            levels[-1]["current_2d_A_m2"],
            rtol=1e-7,
            atol=1e-8,
        )
