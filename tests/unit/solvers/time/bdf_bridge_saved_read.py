"""No-solve parity/readback of the retained assembled BDF bridge witnesses."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csc_matrix


def _without_runtime(value):
    if isinstance(value, dict):
        return {key: _without_runtime(part) for key, part in value.items()
                if key not in ("wall_s", "controls", "assembled_full_residual")}
    if isinstance(value, list):
        return [_without_runtime(part) for part in value]
    return value


def _bits(left, right):
    left, right = np.asarray(left), np.asarray(right)
    assert left.shape == right.shape and left.dtype == right.dtype
    assert left.tobytes() == right.tobytes()
    return left.size


def _matrix_bits(left, right):
    left, right = csc_matrix(left), csc_matrix(right)
    left.sort_indices()
    right.sort_indices()
    assert left.shape == right.shape
    return sum(_bits(getattr(left, name), getattr(right, name))
               for name in ("data", "indices", "indptr"))


def read(root):
    import solarlab.solvers.nonlinear.newton as newton
    def forbidden(*args, **kwargs):
        raise AssertionError("saved bridge read must not solve")
    newton.solve_newton = forbidden
    from examples.research_1d.steady import load_device
    from solarlab.device.compile_1d import compile_device_1d
    from solarlab.equations.assembly.classical_1d import AssembledDC1DProblem
    from solarlab.equations.dc_1d import DC1DProblem
    from solarlab.solvers.time.assembled_storage import AssembledStorage1D
    from solarlab.solvers.time.bdf import BDFStep, HistoryPoint
    from solarlab.solvers.time.bdf_1d import BDFDrive, ElectronicStorage1D, bdf_accounting
    first = json.loads((root / "NativeParity01.json").read_text())
    second = json.loads((root / "NativeParityLight01.json").read_text())
    cases = {**first, **second}
    device = compile_device_1d(load_device(), intervals_per_layer=12)
    comparisons = []
    words = 0
    for generation in (0.0, 1.0):
        suffix = str(generation)
        for kind in ("BDF1", "adaptive"):
            original = cases[kind + "_legacy_" + suffix]
            assembled = cases[kind + "_assembled_" + suffix]
            assert json.dumps(_without_runtime(original), sort_keys=True) == json.dumps(
                _without_runtime(assembled), sort_keys=True), (kind, generation)
            assert original["controls"]["compiled_identity"] == device.identity
            drive_data = original["controls"]["drive"]
            drive = BDFDrive(np.array(drive_data["times_s"]), np.array(drive_data["left_values"]),
                             np.array(drive_data["right_values"]))
            histories, assembled_histories = [], []
            for time, coordinates in zip(original["accepted_times_s"], original["accepted_coordinates"]):
                u = np.asarray(coordinates)
                voltage, light = drive.values(time)
                old = ElectronicStorage1D(DC1DProblem(device, voltage, light))
                new = AssembledStorage1D(AssembledDC1DProblem(device, voltage, light))
                words += _bits(old.storage(u), new.storage(u))
                words += _matrix_bits(old.storage_tangent(u), new.storage_tangent(u))
                histories.append(HistoryPoint(time, u, old.storage(u), old))
                assembled_histories.append(HistoryPoint(time, u, new.storage(u), new))
            accepted = [a for a in original["attempts"] if a["accepted"]]
            accepted_new = [a for a in assembled["attempts"] if a["accepted"]]
            for index, (a, b) in enumerate(zip(accepted, accepted_new), start=1):
                h, h_new = histories[index], assembled_histories[index]
                previous = tuple(reversed(histories[max(0, index - 3):index]))
                previous_new = tuple(reversed(assembled_histories[max(0, index - 3):index]))
                step = BDFStep(h.problem, previous, h.time_s, a["order"])
                native = BDFStep(h_new.problem, previous_new, h_new.time_s, b["order"])
                words += _bits(step.residual(h.u), native.residual(h_new.u))
                words += _bits(native.residual(h_new.u), np.asarray(b["assessment"]["accounting"]["assembled_full_residual"]))
                words += _matrix_bits(step.jacobian(h.u), native.jacobian(h_new.u))
                info = bdf_accounting(step, h.u)
                saved = a["assessment"]["accounting"]
                for key, value in info.items():
                    if isinstance(value, np.ndarray):
                        words += _bits(value, np.asarray(saved[key], dtype=value.dtype))
            comparisons.append({"case": kind + "_" + suffix, "completed": original["completed"],
                "termination": original["termination"], "accepted_steps": len(accepted),
                "rejected_steps": sum(not a["accepted"] for a in original["attempts"]),
                "samples": len(original["samples"]), "identical_step_sequence": True,
                "Newton_records_and_accounting_serialized_bitwise": True,
                "current_source_original_rows_and_J_bitwise": True})
        be = cases["BE_" + suffix]
        be_new = cases["BE_assembled_" + suffix]
        assert json.dumps(_without_runtime(be), sort_keys=True) == json.dumps(
            _without_runtime(be_new), sort_keys=True), ("BE", generation)
    return {"case_comparisons": comparisons, "rechecked_array_entries": int(words),
        "changed_words": 0, "new_solver_calls": 0, "qualification": False,
        "dense_sample_qualification": "fixed endpoints checked; adaptive histories stopped before sampling"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    value = read(args.root)
    output = args.root / "SavedRead01.json"
    text = json.dumps(value, indent=2) + "\n"
    if output.exists():
        assert output.read_text() == text
    else:
        output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
