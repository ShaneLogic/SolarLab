from __future__ import annotations

import json
import hashlib
import operator
import os
from pathlib import Path
import platform

import numpy as np
import pytest
import scipy

from perovskite_sim.solver import illuminated_ss, mol
from perovskite_sim.twod import solver_2d
from perovskite_sim.twod.experiments import jv_sweep_2d
from perovskite_sim.twod.microstructure import lateral_dual_cell_widths
from perovskite_sim.validation.numerical_certificate import (
    MatrixPoint,
    load_refinement_registry,
)
from perovskite_sim.validation.refinement_executors import (
    run_twod_mobile_ion_interface_srh,
)
from perovskite_sim.validation.refinement_runner import source_provenance
from tests.accepted_trajectory import AcceptedTrajectoryMonitor, observe_accepted_radau


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def accepted_trajectories(request):
    records = {}
    destination = os.environ.get("SOLARLAB_F7_ACCEPTED_TWOD_EVIDENCE")
    if destination:

        def save():
            sources = (
                "tests/accepted_trajectory.py",
                "tests/integration/test_twod_combined_refinement.py",
            )
            payload = {
                "kind": "foundation_registered_twod_accepted_trajectories",
                "source": source_provenance(ROOT),
                "test_source_sha256": {
                    name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                    for name in sources
                },
                "environment": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "scipy": scipy.__version__,
                },
                "cases": records,
            }
            Path(destination).write_text(
                json.dumps(payload, indent=2, allow_nan=False) + "\n"
            )

        request.addfinalizer(save)
    return records


def test_real_registered_combined_twod_cell_returns_all_certificate_axes():
    lane = load_refinement_registry(
        ROOT / "reproducibility/NumericalRefinementRegistry.yaml",
        project_root=ROOT,
    ).lane("twod-mobile-ion-interface-srh-v1")

    measurement = run_twod_mobile_ion_interface_srh(
        lane,
        MatrixPoint(4, 0.1),
        ROOT,
    )

    observables = {item.name: item for item in measurement.observables}
    quality = {item.name: item.values[0] for item in measurement.quality}
    assert set(observables) == {gate.metric for gate in lane.observables}
    assert set(quality) == {gate.metric for gate in lane.quality_gates}
    assert quality["combined_gb_ion_interface_topology_verified"] == 1.0
    assert quality["clamp_inactive_slice_verified"] == 1.0
    assert quality["minimum_mobile_ion_relative_redistribution"] > 1.0e-4
    assert quality["minimum_lateral_carrier_variation_relative"] > 1.0e-4
    metadata = json.loads(measurement.metadata_json)
    assert metadata["execution_protocol"]["implicit_legacy_protocol"] is False
    assert metadata["execution_protocol"]["current_composition"] == (
        "electron_hole_positive_ion_displacement"
    )


@pytest.mark.slow
@pytest.mark.parametrize("grid", [4, 6, 8])
@pytest.mark.parametrize("factor", [1.0, 0.1, 0.01])
def test_registered_twod_matrix_keeps_every_accepted_state_physical(
    grid,
    factor,
    monkeypatch,
    accepted_trajectories,
):
    lane = load_refinement_registry(
        ROOT / "reproducibility/NumericalRefinementRegistry.yaml",
        project_root=ROOT,
    ).lane("twod-mobile-ion-interface-srh-v1")
    point = MatrixPoint(grid, factor)
    record = {
        "preparation": [],
        "runs": [],
        "lane_definition_sha256": lane.definition_sha256,
    }
    accepted_trajectories[point.key] = record
    original = solver_2d.run_transient_2d
    original_preparation = illuminated_ss.run_transient
    initial_ions = None
    initial_ions_1d = None

    def prepare_observed(x, y0, t_span, t_eval, stack, **kwargs):
        nonlocal initial_ions_1d
        material = kwargs["mat"]
        assert not material.has_dual_ions and material.N_iface_state == 0
        initial_ions_1d = np.asarray(y0)[2 * x.size :].copy()
        monitor = AcceptedTrajectoryMonitor(
            x.size,
            y0,
            weights=material.dx_cell,
            ion_capacities=(material.P_lim_node,),
            inventory_rtol=lane.options["ion_inventory_rtol"],
        )
        try:
            with observe_accepted_radau(mol, monitor):
                result = original_preparation(x, y0, t_span, t_eval, stack, **kwargs)
            assert result.success, result.message
            monitor.assert_valid()
            return result
        finally:
            record["preparation"].append(
                {"time_span_s": list(t_span), "trajectory": monitor.report()}
            )

    def solve_observed(state, material, **kwargs):
        nonlocal initial_ions
        count = material.ni.size
        mobile = material.has_mobile_ions
        capacities = (material.P_lim_2d.ravel(),) if mobile else ()
        reference = np.asarray(state).copy()
        if mobile:
            if initial_ions is None:
                assert initial_ions_1d is not None
                initial_ions = (
                    np.broadcast_to(
                        initial_ions_1d[:, None],
                        material.ni.shape,
                    )
                    .ravel()
                    .copy()
                )
            reference[2 * count :] = initial_ions
        weights = (
            lateral_dual_cell_widths(material.grid.y)[:, None]
            * lateral_dual_cell_widths(material.grid.x)[None, :]
        ).ravel()
        monitor = AcceptedTrajectoryMonitor(
            count,
            reference,
            weights=weights,
            ion_capacities=capacities,
            inventory_rtol=lane.options["ion_inventory_rtol"],
        )
        accepted = False
        try:
            with observe_accepted_radau(solver_2d, monitor):
                result = original(state, material, **kwargs)
            accepted = True
            monitor.assert_valid()
            return result
        finally:
            record["runs"].append(
                {
                    "voltage_V": kwargs["V_app"],
                    "duration_s": kwargs["t_end"],
                    "solver_returned_successfully": accepted,
                    "trajectory": monitor.report(),
                }
            )

    monkeypatch.setattr(jv_sweep_2d, "run_transient_2d", solve_observed)
    monkeypatch.setattr(illuminated_ss, "run_transient", prepare_observed)
    measurement = run_twod_mobile_ion_interface_srh(lane, point, ROOT)
    quality = {item.name: item.values[0] for item in measurement.quality}
    comparisons = {"eq": operator.eq, "le": operator.le, "ge": operator.ge}
    for gate in lane.quality_gates:
        assert comparisons[gate.operator](quality[gate.metric], gate.limit), (
            gate,
            quality[gate.metric],
        )
    assert record["runs"]
    assert record["preparation"]
    assert all(item["trajectory"]["passed"] for item in record["preparation"])
    assert {0.0, 0.05, 0.1}.issubset({run["voltage_V"] for run in record["runs"]})
    assert all(
        run["trajectory"]["passed"]
        for run in record["runs"]
        if run["solver_returned_successfully"]
    )
    record["quality"] = quality
