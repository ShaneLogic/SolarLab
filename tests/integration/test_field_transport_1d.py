"""Actual CT assembly and saved native evidence; no repeated steady solves."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest

from examples.research_1d import field_transport as example
from examples.research_1d.steady import load_device
from solarlab.config.resolve_device import resolve_device
from solarlab.device.compile_1d import compile_device_1d
from solarlab.device.inputs import DeviceInput
from solarlab.device.resolved import PreparedDevice
from solarlab.equations.dc_1d import CapabilityError, CompiledDevice1D, DC1DProblem
from solarlab.materials.source import SourceDocument
from solarlab.mesh import RIGHT_LAYER_WINS
from solarlab.physics.transport import legacy_face_diffusivity
from solarlab.physics.transport.field_mobility import field_dependent_face_diffusivity
from solarlab.physics.transport.field_mobility_tangents import FieldMobilityCapabilityError


@pytest.fixture(scope="module")
def prepared() -> PreparedDevice:
    return example.prepare_case()


@pytest.fixture(scope="module")
def device(prepared: PreparedDevice) -> CompiledDevice1D:
    return compile_device_1d(prepared, intervals_per_layer=example.INTERVALS)


def edited(prepared: PreparedDevice, document: dict[str, Any]) -> PreparedDevice:
    raw = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    source = SourceDocument("field_transport_test", raw)
    sources = tuple(item for item in prepared.sources if item.id != source.id) + (source,)
    return resolve_device(DeviceInput.model_validate(document), prepared.defaults, prepared.resources, sources=sources)


def coordinates(device: CompiledDevice1D) -> np.ndarray:
    count = device.node_count
    phi = device.reference_potential_V + .012 * np.sin(np.linspace(0.0, 2.0 * np.pi, count))
    return np.r_[phi, np.linspace(-.15, .2, count), np.linspace(.1, -.25, count)]


def test_historical_face_rules_masks_and_actual_state_dependence(
    prepared: PreparedDevice, device: CompiledDevice1D,
) -> None:
    before = prepared.to_mapping()
    again = compile_device_1d(prepared, intervals_per_layer=example.INTERVALS)
    assert prepared.to_mapping() == before and again.identity == device.identity
    model = device.field_transport
    assert model is not None and model.field_measure == "face_field_1d_v1"
    assert device.geometry.material_owner_convention == RIGHT_LAYER_WINS
    np.testing.assert_array_equal(np.flatnonzero(model.electron.saturation_velocity_m_s), np.arange(11, 24))
    np.testing.assert_array_equal(model.electron.saturation_velocity_m_s[12:23], 1.0e4)
    np.testing.assert_array_equal(model.hole.saturation_velocity_m_s[12:23], 1.5e4)
    np.testing.assert_array_equal(model.electron.saturation_velocity_m_s[[11, 23]], 5.0e3)
    np.testing.assert_array_equal(model.electron.ct_exponent[[11, 23]], 1.5)
    assert not model.nodes_m.flags.writeable and not model.electron.ct_exponent.flags.writeable
    np.testing.assert_array_equal(model.base_electron_diffusivity_m2_s,
                                  legacy_face_diffusivity(device.parameters.electron_diffusivity_m2_s))
    phi = coordinates(device)[:device.node_count]
    actual = model.evaluate(phi)
    np.testing.assert_array_equal(actual.field_V_m, -np.diff(phi) / np.diff(device.nodes_m))
    for carrier, base, value in (
        (model.electron, model.base_electron_diffusivity_m2_s, actual.electron_diffusivity_m2_s),
        (model.hole, model.base_hole_diffusivity_m2_s, actual.hole_diffusivity_m2_s),
    ):
        expected = field_dependent_face_diffusivity(base, device.thermal_voltage_V, actual.field_V_m, carrier)
        np.testing.assert_array_equal(value, expected)
        assert np.min(value / base) < .99
    changed = model.evaluate(2.0 * phi)
    assert not np.array_equal(actual.electron_diffusivity_m2_s, changed.electron_diffusivity_m2_s)


def test_zero_field_is_smooth_without_regularization(device: CompiledDevice1D) -> None:
    model = device.field_transport
    assert model is not None
    value = model.evaluate(np.zeros(device.node_count))
    np.testing.assert_array_equal(value.electron_diffusivity_m2_s,
                                  model.base_electron_diffusivity_m2_s / device.thermal_voltage_V * device.thermal_voltage_V)
    np.testing.assert_array_equal(value.electron_d_diffusivity_d_field, 0.0)
    np.testing.assert_array_equal(value.hole_d_diffusivity_d_field, 0.0)
    phi = np.linspace(0.0, .1, device.node_count)
    positive, negative = model.evaluate(phi), model.evaluate(-phi)
    np.testing.assert_array_equal(positive.electron_diffusivity_m2_s, negative.electron_diffusivity_m2_s)
    np.testing.assert_array_equal(positive.electron_d_diffusivity_d_field, -negative.electron_d_diffusivity_d_field)


@pytest.mark.parametrize("potential_only", [False, True])
def test_full_dc_directional_jacobian_requires_mobility_chain(
    device: CompiledDevice1D, potential_only: bool,
) -> None:
    u = coordinates(device)
    # Generation is a state-independent source. Avoid subtracting that large
    # constant in finite differences at this low-density synthetic state.
    problem = DC1DProblem(device)
    matrix = problem.jacobian(u)
    np.testing.assert_array_equal(matrix.toarray(), DC1DProblem(device, generation_scale=1.0).jacobian(u).toarray())
    rng = np.random.default_rng(391)
    direction = np.r_[.02 * rng.normal(size=device.node_count), rng.normal(size=2 * device.node_count)]
    if potential_only:
        direction[device.node_count:] = 0.0
    step = 1.0e-5
    # Fourth order resolves the beta=1.5 interface face without increasing
    # the perturbation enough to cross its small nonzero electric field.
    numerical = (problem.residual(u - 2.0 * step * direction)
                 - 8.0 * problem.residual(u - step * direction)
                 + 8.0 * problem.residual(u + step * direction)
                 - problem.residual(u + 2.0 * step * direction)) / (12.0 * step)
    scale = np.maximum(abs(matrix) @ abs(direction), 1.0e-20)
    assert np.max(abs(numerical - matrix @ direction) / scale) < 2.0e-8
    model = device.field_transport
    assert model is not None
    field = model.evaluate(u[:device.node_count])
    # This diagnostic device holds the very same D values fixed, isolating
    # the omitted dD/dphi error. It is never passed to a physical solve.
    held = replace(device, field_transport=None,
                   electron_diffusivity_faces_m2_s=field.electron_diffusivity_m2_s,
                   hole_diffusivity_faces_m2_s=field.hole_diffusivity_m2_s)
    omitted = DC1DProblem(held).jacobian(u)
    assert np.max(abs(numerical - omitted @ direction) / scale) > 1.0e-4
    constant = DC1DProblem(replace(device, field_transport=None))
    assert np.any(problem.fluxes(u).electron.flux != constant.fluxes(u).electron.flux)
    assert np.any(problem.fluxes(u).hole.flux != constant.fluxes(u).hole.flux)
    assert np.any(problem.current_density(u) != constant.current_density(u))
    assert np.any(problem.residual(u)[device.node_count:] != constant.residual(u)[device.node_count:])
    np.testing.assert_array_equal(problem.residual(u)[:device.node_count], constant.residual(u)[:device.node_count])
    # Dirichlet contact constraints replace the full transport rows.
    rows = [0, device.node_count - 1, device.node_count, 2 * device.node_count - 1,
            2 * device.node_count, 3 * device.node_count - 1]
    np.testing.assert_array_equal(matrix.toarray()[rows], constant.jacobian(u).toarray()[rows])


@pytest.mark.parametrize("name,value", [("pf_gamma_n", 1e-4), ("pf_gamma_p", 1e-4),
                                        ("ct_beta_n", 1.0), ("ct_beta_p", .5)])
def test_unsupported_laws_refused_before_solver(
    prepared: PreparedDevice, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    name: str, value: float,
) -> None:
    document = prepared.to_input().editing_data()
    document["layers"][1]["parameters"][name] = value
    unsupported = edited(prepared, document)
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("unsupported transport reached the nonlinear solver")
    monkeypatch.setattr(example, "prepare_case", lambda config: unsupported)
    monkeypatch.setattr(example, "solve_steady_state", forbidden)
    monkeypatch.setattr(sys, "argv", ["field_transport", "--output", str(tmp_path / "result.json")])
    with pytest.raises(CapabilityError, match="unsupported_classical_1d_dc"):
        example.main()
    assert not (tmp_path / "result.json").exists()


def test_binding_rejects_partial_field_and_wrong_device(device: CompiledDevice1D) -> None:
    model = device.field_transport
    assert model is not None
    with pytest.raises(FieldMobilityCapabilityError):
        replace(model, field_measure="transverse_field_2d_v1")
    with pytest.raises(ValueError, match="grid, temperature"):
        replace(device, field_transport=replace(model, thermal_voltage_V=1.01 * model.thermal_voltage_V))


@pytest.mark.parametrize("parameter", ["v_sat_n", "v_sat_p", "ct_beta_n", "ct_beta_p"])
def test_active_law_identity_is_parameter_specific(prepared: PreparedDevice, parameter: str) -> None:
    document = prepared.to_input().editing_data()
    original = compile_device_1d(edited(prepared, document), intervals_per_layer=example.INTERVALS)
    document["layers"][1]["parameters"][parameter] *= 1.1
    changed = compile_device_1d(edited(prepared, document), intervals_per_layer=example.INTERVALS)
    assert original.identity != changed.identity
    assert original.field_transport is not None and changed.field_transport is not None
    assert original.field_transport.identity_sha256 != changed.field_transport.identity_sha256


def test_temperature_scaling_precedes_ct_without_repainting_parameters(prepared: PreparedDevice) -> None:
    document = prepared.to_input().editing_data()
    document["settings"]["T"] = 330.0
    active = compile_device_1d(edited(prepared, document), intervals_per_layer=example.INTERVALS)
    document["layers"][1]["parameters"].update(v_sat_n=0.0, v_sat_p=0.0)
    inactive = compile_device_1d(edited(prepared, document), intervals_per_layer=example.INTERVALS)
    assert active.field_transport is not None and inactive.field_transport is None
    assert active.thermal_voltage_V == inactive.thermal_voltage_V
    np.testing.assert_array_equal(active.parameters.electron_diffusivity_m2_s, inactive.parameters.electron_diffusivity_m2_s)
    np.testing.assert_array_equal(active.parameters.hole_diffusivity_m2_s, inactive.parameters.hole_diffusivity_m2_s)
    np.testing.assert_array_equal(active.field_transport.base_electron_diffusivity_m2_s, inactive.electron_diffusivity_faces_m2_s)
    np.testing.assert_array_equal(active.field_transport.electron.saturation_velocity_m_s[12:23], 1e4)


def test_no_ct_branch_and_saved_pre_edit_baseline_are_bitwise_exact() -> None:
    device = compile_device_1d(load_device(), intervals_per_layer=example.INTERVALS)
    assert device.field_transport is None
    baseline = os.environ.get("SOLARLAB_FIELD_TRANSPORT_BASELINE")
    if baseline is None:
        return
    directory = Path(baseline)
    arrays = np.load(directory / "NoCTBaseline.npz")
    metadata = json.loads((directory / "Baseline.json").read_text())
    assert device.identity == metadata["compiled_identity"]
    problem = DC1DProblem(device)
    for key, actual in (("residual", problem.residual(arrays["u"])),
                        ("jacobian", problem.jacobian(arrays["u"]).toarray()),
                        ("current", problem.current_density(arrays["u"])),
                        ("contacts", problem.contact_current_density(arrays["u"]))):
        np.testing.assert_array_equal(actual, arrays[key])


def test_saved_actual_ct_witness_reads_without_solving(monkeypatch: pytest.MonkeyPatch) -> None:
    path = os.environ.get("SOLARLAB_FIELD_TRANSPORT_RESULT")
    if path is None:
        pytest.skip("bind the saved native result explicitly; never repeat a physics solve")
        return
    raw = Path(path).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == os.environ["SOLARLAB_FIELD_TRANSPORT_RESULT_SHA256"]
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("reading evidence attempted a compile or solve")
    monkeypatch.setattr(example, "compile_device_1d", forbidden)
    monkeypatch.setattr(example, "solve_steady_state", forbidden)
    record = example.read_result(Path(path))
    summary = record["summary"]
    assert summary["accepted"] and summary["node_count"] == 37
    assert 0.0 < summary["minimum_electron_diffusivity_ratio"] < 1.0
    assert 0.0 < summary["minimum_hole_diffusivity_ratio"] < 1.0
    assert summary["maximum_abs_electron_dD_dE"] > 0.0 and summary["maximum_abs_hole_dD_dE"] > 0.0
    assert summary["current_spread_A_m2"] < 1e-8
    assert all(gate["passed"] for gate in record["result"]["newton"]["gates"])
    assert summary["physical_certificate_status"] == "not_evaluated" and not summary["qualification"]
    monkeypatch.setattr(sys, "argv", ["field_transport", "--read", path])
    assert example.main() == 0
