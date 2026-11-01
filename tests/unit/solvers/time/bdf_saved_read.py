"""Solver-free readback of the retained BDF ionic histories and fit receipts."""

import argparse
import json
from pathlib import Path

import numpy as np


def read_evidence(root):
    import solarlab.solvers.nonlinear.newton as newton
    def forbidden(*args, **kwargs):
        raise AssertionError("saved BDF evidence read must not solve")
    newton.solve_newton = forbidden
    from examples.research_1d.ionic_multilayer import prepare_confined_ion_device
    from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
    from solarlab.equations.ionic_1d import Ionic1DProblem
    from solarlab.solvers.time.bdf import BDFStep, HistoryPoint
    from solarlab.solvers.time.bdf_1d import IonicStorage1D, bdf_accounting
    ionic = json.loads((root / "Ionic.json").read_bytes())
    device = compile_ionic_device_1d(prepare_confined_ion_device(diffusion_only=True),
        intervals_per_layer=(12, 12, 12), ion_domain_layer="absorber", ion_domain_endpoints=("blocking", "blocking"))
    assert device.identity == ionic["compiled_identity"]
    compared = 0
    maximum_residual_difference = 0.0
    maximum_maxwell_identity = 0.0
    for row in ionic["work_precision"]:
        if row["method"] != "adaptive BDF1/2":
            continue
        result = row["result"]
        assert result["completed"] and not result["restarts"]
        times = result["accepted_times_s"]
        coordinates = result["accepted_coordinates"]
        history = []
        for t, u in zip(times, coordinates):
            adapter = IonicStorage1D(Ionic1DProblem(device, .002 + .002 * t, 0.0))
            value = np.asarray(u)
            history.append(HistoryPoint(t, value, adapter.storage(value), adapter))
        accepted = [step for step in result["attempts"] if step["accepted"]]
        assert len(accepted) + 1 == len(history)
        for index, attempt in enumerate(accepted, start=1):
            endpoint = history[index]
            prior = tuple(reversed(history[max(0, index - 3):index]))
            step = BDFStep(endpoint.problem, prior, endpoint.time_s, attempt["order"])
            info = bdf_accounting(step, endpoint.u)
            saved = attempt["assessment"]["accounting"]
            key = "backward_euler_residual" if step.order == 1 else "bdf_residual"
            residual_difference = float(np.max(abs(np.asarray(info[key]) - np.asarray(saved[key]))))
            maximum_residual_difference = max(maximum_residual_difference, residual_difference)
            np.testing.assert_array_equal(info[key], saved[key])
            np.testing.assert_array_equal(info["ion_inventory_rate_m2_s"], saved["ion_inventory_rate_m2_s"])
            np.testing.assert_array_equal(info["total_current_density_A_m2"], saved["total_current_density_A_m2"])
            maximum_maxwell_identity = max(maximum_maxwell_identity,
                float(np.max(abs(info["maxwell_identity_error_A_m2"]))))
            if step.order == 2:
                np.testing.assert_array_equal(info["total_current_density_A_m2"],
                    info["conduction_current_density_A_m2"] + info["displacement_current_density_A_m2"])
                # Separate rounded readbacks include their explicit compensations.
                np.testing.assert_allclose(info["maxwell_cell_jump_A_m2"],
                    info["readback_predicted_maxwell_cell_jump_A_m2"], atol=1e-12, rtol=1e-10)
            compared += 1
    fits = json.loads((root / "CubicDiagnosticFits.json").read_bytes())
    assert all(value >= 10 * fits["measured_precision_floor_inventory"] for value in fits["eligible_differences"])
    assert abs(fits["observed_order"] - 2) < .1
    assert fits["first_failed_grid"] == 64 and fits["failed_candidate_preserved"]
    parity = json.loads((root / "C7ElectronicParity.json").read_bytes())
    assert all(parity["per_quantity_bitwise"].values())
    assert max(parity["per_quantity_maximum_differences"].values()) == 0
    return {"saved_ionic_steps_rechecked": compared, "maximum_original_residual_difference": maximum_residual_difference,
        "maximum_maxwell_identity_error_A_m2": maximum_maxwell_identity,
        "electronic_c7_parity": "bitwise all represented arrays", "cubic_observed_order": fits["observed_order"],
        "new_solver_calls": 0, "qualification": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    value = read_evidence(args.root)
    output = args.root / "SavedRead.json"
    text = json.dumps(value, indent=2) + "\n"
    if output.exists():
        assert output.read_text() == text
    else:
        output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
