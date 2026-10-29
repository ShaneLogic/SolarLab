"""Unmodified steady_2d driver: original and assembled Newton paths, bitwise.

The two saved numerical seeds are the exact historical dark/light fields.
They must pass the current original gates before a small lateral
perturbation is applied. No cold preparation or voltage curve is repeated.
The factory substitution is test only; the driver and its controls are
untouched. Every callback state/F/J, Newton record, gate and physical
readout is compared, including signed zeros. Only elapsed wall time differs.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, is_dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.sparse import csc_matrix
from threadpoolctl import threadpool_limits

from examples.research_1d.steady import load_device
from examples.research_2d.dc_assembly import spatial_input
from solarlab.device.compile_2d import compile_device_2d
from solarlab.equations.assembly.two_dimensional import (
    AssembledDC2DProblem, assembled_problem_factory_2d,
)
from solarlab.equations.dc_2d import CompiledDevice2D, DC2DProblem
from solarlab.experiments.transient_results import _encode
from solarlab.solvers.nonlinear import steady_2d

SEEDS = Path(__file__).with_name("fixtures") / "dc2d_steady_seeds_v1.json"


@pytest.fixture(scope="module", autouse=True)
def blas1() -> Iterator[None]:
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(scope="module")
def device() -> CompiledDevice2D:
    return compile_device_2d(load_device(), spatial_input())


def words(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return (value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, (float, np.floating)):
        return np.float64(value).tobytes()
    if is_dataclass(value) and not isinstance(value, type):
        return words(asdict(value))
    if isinstance(value, dict):
        return tuple((key, words(item)) for key, item in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(words(item) for item in value)
    return value


def matrix_words(matrix: csc_matrix) -> Any:
    return (matrix.shape, words(matrix.data), matrix.indices.tobytes(), matrix.indptr.tobytes())


@pytest.mark.parametrize("name", ("dark", "light"))
def test_actual_195_row_steady_histories_are_bitwise(
    name: str, device: CompiledDevice2D, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest,
) -> None:
    document = json.loads(SEEDS.read_bytes())
    assert document["schema"] == "solarlab.assembly_2d.saved_numerical_seeds.v1"
    assert document["qualification"] is False
    seed = document["points"][name]
    # Compiled identity changed with the contact/source migration. The
    # saved acceptance remains attached to the historical identity; these
    # are numerical coordinates until the current original gates admit them.
    assert seed["compiled_identity"]
    voltage, light = seed["voltage_V"], seed["generation_scale"]
    u = np.frombuffer(bytes.fromhex(seed["u_words_le_hex"]), dtype="<f8").copy()
    assert hashlib.sha256(u.tobytes()).hexdigest() == seed["u_sha256"]
    state = device.state(u, voltage_V=voltage)
    for field, digest in seed["field_sha256"].items():
        assert hashlib.sha256(np.ascontiguousarray(getattr(state, field), dtype="<f8").tobytes()).hexdigest() == digest
    controls = steady_2d.Steady2DOptions()
    assert asdict(controls.electronic) == seed["original_controls"]["electronic"]
    original = DC2DProblem(device, voltage, light)
    admitted = steady_2d._diagnostics(original, u, controls)
    assert all(gate.passed for gate in steady_2d._checks(admitted, controls))
    assert admitted["original_rows"] == tuple(range(195))
    g, count = device.geometry, device.node_count
    y = (device.column.nodes_m - device.column.nodes_m[0]) / np.ptp(device.column.nodes_m)
    x = (g.x_m - g.x_m[0]) / g.lateral_length_m
    envelope = np.sin(np.pi * y)
    envelope[[0, -1]] = 0.0
    u[:count] += (1e-4 * envelope[:, None] * np.cos(np.pi * x)[None, :]).ravel()
    u[count:2 * count] += (.002 * envelope[:, None] * np.sin(np.pi * x)[None, :]).ravel()
    u[2 * count:] += (-.0015 * envelope[:, None] * np.cos(2 * np.pi * x)[None, :]).ravel()
    initial = device.state(u, voltage_V=voltage)
    assert np.max(abs(original.fluxes(u).electron.flux[:g.x_face_count])) > 0.0
    assert not all(gate.passed for gate in steady_2d._checks(steady_2d._diagnostics(original, u, controls), controls))
    provenance = {"kind": "current-gate-admitted archived fields with an explicit lateral test perturbation",
                  "seed_artifact_sha256": seed["artifact_sha256"], "seed_state_sha256": seed["u_sha256"],
                  "seed_compiled_identity": seed["compiled_identity"], "current_compiled_identity": device.identity,
                  "historical_acceptance_transferred": False,
                  "cold_preparation_calls": 0, "qualification": False}
    reference_trace: list[Any] = []
    candidate_trace: list[Any] = []
    old_residual, old_jacobian = DC2DProblem.residual, DC2DProblem.jacobian
    new_residual, new_jacobian = AssembledDC2DProblem.residual, AssembledDC2DProblem.jacobian
    assembly_calls = []
    assemble = AssembledDC2DProblem.assemble

    def reference_residual(self: DC2DProblem, values: Any) -> Any:
        output = old_residual(self, values)
        reference_trace.append(("F", words(np.asarray(values)), words(output)))
        return output

    def reference_jacobian(self: DC2DProblem, values: Any) -> Any:
        output = old_jacobian(self, values)
        reference_trace.append(("J", words(np.asarray(values)), matrix_words(output)))
        return output

    def candidate_residual(self: AssembledDC2DProblem, values: Any) -> Any:
        output = new_residual(self, values)
        candidate_trace.append(("F", words(np.asarray(values)), words(output)))
        return output

    def candidate_jacobian(self: AssembledDC2DProblem, values: Any) -> Any:
        output = new_jacobian(self, values)
        candidate_trace.append(("J", words(np.asarray(values)), matrix_words(output)))
        return output

    def counted_assemble(self: AssembledDC2DProblem, values: Any, *, derivatives: bool = True) -> Any:
        assembly_calls.append(derivatives)
        return assemble(self, values, derivatives=derivatives)

    with monkeypatch.context() as patch:
        patch.setattr(DC2DProblem, "residual", reference_residual)
        patch.setattr(DC2DProblem, "jacobian", reference_jacobian)
        reference = steady_2d.solve_steady_state_2d(device, voltage_V=voltage, generation_scale=light,
            options=controls, initial_state=initial, initial_provenance=provenance)
    with monkeypatch.context() as patch:
        patch.setattr(steady_2d, "DC2DProblem", assembled_problem_factory_2d(device))
        patch.setattr(AssembledDC2DProblem, "residual", candidate_residual)
        patch.setattr(AssembledDC2DProblem, "jacobian", candidate_jacobian)
        patch.setattr(AssembledDC2DProblem, "assemble", counted_assemble)
        candidate = steady_2d.solve_steady_state_2d(device, voltage_V=voltage, generation_scale=light,
            options=controls, initial_state=initial, initial_provenance=provenance)
    evidence = {"case": name, "rows": 195, "reference": asdict(reference), "candidate": asdict(candidate),
                "seed_admission": admitted, "qualification": False}
    output_directory = os.environ.get("SOLARLAB_ASSEMBLY_2D_EVIDENCE")
    if output_directory is not None:
        path = Path(output_directory) / f"Steady_{name}.json"
        with path.open("x") as stream:
            json.dump(_encode(evidence), stream, allow_nan=False, indent=2)
            stream.write("\n")
    assert reference.accepted and candidate.accepted, (reference.termination, candidate.termination)
    assert reference.newton is not None and len(reference.newton.iterations) > 1
    assert reference_trace == candidate_trace
    assert assembly_calls and any(assembly_calls) and not all(assembly_calls)
    old_result, new_result = asdict(reference), asdict(candidate)
    del old_result["wall_s"], new_result["wall_s"]
    assert words(old_result) == words(new_result)
    request.node.user_properties.append(("steady_parity", {"case": name, "rows": 195,
        "accepted": True, "iterations": len(reference.newton.iterations),
        "residual_evaluations": reference.newton.residual_evaluations,
        "jacobian_evaluations": reference.newton.jacobian_evaluations,
        "callback_arrays": len(reference_trace), "assembly_calls": len(assembly_calls),
        "callback_changed_words": 0, "newton_and_readout_changed_words": 0,
        "reference_wall_s": reference.wall_s, "candidate_wall_s": candidate.wall_s,
        "cold_preparation_calls": 0, "qualification": False}))
