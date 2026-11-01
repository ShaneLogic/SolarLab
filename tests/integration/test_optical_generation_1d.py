"""Conservative typed optical binding; the one native result is read only."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest

from examples.research_1d import optical_generation as example
from examples.research_1d.steady import load_device
from solarlab.config.resolve_device import resolve_device
from solarlab.contracts.units import ELEMENTARY_CHARGE_C as Q
from solarlab.device.compile_1d import compile_device_1d
from solarlab.device.inputs import DeviceInput
from solarlab.device.optical_generation import bind_optical_generation_1d
from solarlab.device.resolved import PreparedDevice
from solarlab.equations.dc_1d import CapabilityError, CompiledDevice1D, DC1DProblem
from solarlab.materials.resources import ResourceLibrary, ResourceTable
from solarlab.materials.source import SourceDocument
from solarlab.mesh.geometry import half_endpoint_geometry
from solarlab.physics.generation.profile import StaleGenerationProfileError
from solarlab.solvers.nonlinear.newton import NewtonResult, Termination
from solarlab.solvers.nonlinear.steady_1d import SteadyStateError


@pytest.fixture(scope="module")
def prepared() -> PreparedDevice:
    return example.prepare_case()


@pytest.fixture(scope="module")
def device(prepared: PreparedDevice) -> CompiledDevice1D:
    return compile_device_1d(prepared, intervals_per_layer=example.INTERVALS)


def edited(base: PreparedDevice, document: dict[str, Any], resources: ResourceLibrary | None = None) -> PreparedDevice:
    raw = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    source = SourceDocument("optical_generation_test", raw)
    sources = tuple(item for item in base.sources if item.id != source.id) + (source,)
    return resolve_device(DeviceInput.model_validate(document), base.defaults,
                          base.resources if resources is None else resources, sources=sources)


def test_real_spectrum_cell_source_and_photon_budget(prepared: PreparedDevice, device: CompiledDevice1D) -> None:
    optical = bind_optical_generation_1d(prepared, device.geometry)
    g, b = device.generation, optical.budget
    assert g.method == "transfer_matrix" and device.node_count == 37
    assert np.max(abs(b.spectral_closure_residual())) < 1e-12
    assert b.incident_m2_s == pytest.approx(6e27 * 300e-9, rel=1e-15)
    assert b.reflected_m2_s > 0.0 and b.transmitted_m2_s > 0.0
    assert b.electrical_absorbed_m2_s > 0.0 and b.parasitic_absorbed_m2_s == 0.0
    assert b.reflected_m2_s + b.transmitted_m2_s + b.electrical_absorbed_m2_s == pytest.approx(b.incident_m2_s, rel=1e-12)
    np.testing.assert_array_equal(g.source_m3_s, optical.profile.source_m3_s)
    np.testing.assert_allclose(g.source_m3_s * device.geometry.cell_measures_m, g.absorbed_m2_s, rtol=3e-16)
    assert np.sum(g.source_m3_s * device.geometry.cell_measures_m) == pytest.approx(b.electrical_absorbed_m2_s, rel=1e-12)
    assert device.generation_limit_A_m2 == Q * g.areal_generation_m2_s
    # Cell24 straddles the true absorber/ETL plane despite ETL node ownership.
    assert g.source_m3_s[24] > 0.0
    np.testing.assert_array_equal(g.source_m3_s[25:], 0.0)
    assert not g.source_m3_s.flags.writeable


def test_generation_reaches_original_carrier_rows_without_a_state_derivative(device: CompiledDevice1D) -> None:
    N = device.node_count
    u = np.r_[device.reference_potential_V, np.zeros(2 * N)]
    dark, light = DC1DProblem(device), DC1DProblem(device, generation_scale=1.0)
    difference = light.residual(u) - dark.residual(u)
    np.testing.assert_array_equal(difference[:N], 0.0)
    for block in (difference[N:2 * N], difference[2 * N:]):
        np.testing.assert_allclose(block[1:-1], device.generation.absorbed_m2_s[1:-1], rtol=2e-15)
        np.testing.assert_array_equal(block[[0, -1]], 0.0)
    np.testing.assert_array_equal(light.jacobian(u).toarray(), dark.jacobian(u).toarray())


def test_lossless_bound_stack_matches_closed_form_airy(prepared: PreparedDevice) -> None:
    document = prepared.to_input().editing_data()
    for layer in document["layers"]:
        layer["parameters"].update(optical_material=None, n_optical=2.0, alpha=0.0)
    p = edited(prepared, document)
    device = compile_device_1d(p, intervals_per_layer=example.INTERVALS)
    optical = bind_optical_generation_1d(p, device.geometry)
    wavelengths = optical.binding.wavelengths_m
    r = (1.0 - 2.0) / (1.0 + 2.0)
    phase = np.cos(4.0 * np.pi * 2.0 * device.geometry.extent_m / wavelengths)
    expected = 2.0 * r**2 * (1.0 - phase) / (1.0 + r**4 - 2.0 * r**2 * phase)
    np.testing.assert_allclose(optical.budget.reflectance, expected, rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(optical.budget.transmittance, 1.0 - expected, rtol=0.0, atol=1e-13)
    np.testing.assert_array_equal(device.generation.source_m3_s, 0.0)


def test_historical_endpoint_weights_are_applied_once(prepared: PreparedDevice) -> None:
    document = prepared.to_input().editing_data()
    for index in (0, 2):
        document["layers"][index]["parameters"]["alpha"] = 1e6
    p = edited(prepared, document)
    device = compile_device_1d(p, intervals_per_layer=example.INTERVALS)
    whole = device.generation
    half = bind_optical_generation_1d(p, half_endpoint_geometry(device.nodes_m)).profile
    assert np.all(whole.source_m3_s[[0, -1]] > 0.0)
    np.testing.assert_array_equal(whole.absorbed_m2_s, half.absorbed_m2_s)
    np.testing.assert_array_equal(half.source_m3_s[[0, -1]], 2.0 * whole.source_m3_s[[0, -1]])
    np.testing.assert_array_equal(half.source_m3_s[1:-1], whole.source_m3_s[1:-1])


def test_coherent_optical_only_prefix_absorption_is_parasitic(prepared: PreparedDevice) -> None:
    document = prepared.to_input().editing_data()
    window = {"id": "window", "name": "Illustrative window", "role": "substrate",
              "thickness": 70e-9, "parameters": {"alpha": 5e5, "n_optical": 1.7, "incoherent": False}}
    document["layers"] = (window, *document["layers"])
    p = edited(prepared, document)
    device = compile_device_1d(p, intervals_per_layer=example.INTERVALS)
    optical = bind_optical_generation_1d(p, device.geometry)
    b = optical.budget
    assert device.node_count == 37 and optical.binding.electrical_offset_m == 70e-9
    assert not b.layer_electrical[0] and b.parasitic_absorbed_m2_s > 0.0
    assert device.generation.areal_generation_m2_s == pytest.approx(b.electrical_absorbed_m2_s, rel=1e-12)
    assert b.reflected_m2_s + b.transmitted_m2_s + b.parasitic_absorbed_m2_s + b.electrical_absorbed_m2_s == pytest.approx(b.incident_m2_s, rel=1e-12)


def test_absolute_spectrum_is_not_renormalized_by_scalar_phi(prepared: PreparedDevice, device: CompiledDevice1D) -> None:
    document = prepared.to_input().editing_data()
    document["settings"]["Phi"] = 9e24
    changed = compile_device_1d(edited(prepared, document), intervals_per_layer=example.INTERVALS)
    np.testing.assert_array_equal(changed.generation.source_m3_s, device.generation.source_m3_s)
    assert changed.identity != device.identity


def test_resource_source_identity_and_geometry_prevent_stale_reuse(prepared: PreparedDevice, device: CompiledDevice1D) -> None:
    original = prepared.resources.get("illustrative_photons", "spectrum")
    source = SourceDocument(original.source.id, b"# unchanged values, distinct supplied document\n" + original.source.content)
    replacement = ResourceTable(original.name, original.kind, source)
    resources = ResourceLibrary(tuple(replacement if item.name == original.name else item for item in prepared.resources.tables))
    p = edited(prepared, prepared.to_input().editing_data(), resources)
    changed = compile_device_1d(p, intervals_per_layer=example.INTERVALS)
    assert changed.generation.binding_sha256 != device.generation.binding_sha256
    np.testing.assert_array_equal(changed.generation.source_m3_s, device.generation.source_m3_s)
    with pytest.raises(StaleGenerationProfileError):
        device.generation.require_inputs(changed.generation.binding_sha256, device.geometry)
    other_grid = compile_device_1d(prepared, intervals_per_layer=(13, 12, 12))
    assert other_grid.generation.binding_sha256 == device.generation.binding_sha256
    with pytest.raises(StaleGenerationProfileError):
        device.generation.require_inputs(device.generation.binding_sha256, other_grid.geometry)


@pytest.mark.parametrize("kind", ["no_spectrum", "missing_index", "incoherent", "ambiguous_index", "outside_nk", "fixed_conflict"])
def test_unsupported_optical_inputs_stop_before_solver(
    prepared: PreparedDevice, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kind: str,
) -> None:
    document = prepared.to_input().editing_data()
    resources = prepared.resources
    if kind == "no_spectrum":
        document["spectrum"] = None
    elif kind == "missing_index":
        document["layers"][0]["parameters"]["n_optical"] = None
    elif kind == "incoherent":
        document["layers"][0]["parameters"]["incoherent"] = True
    elif kind == "ambiguous_index":
        document["layers"][1]["parameters"]["n_optical"] = 2.1
    elif kind == "outside_nk":
        replacement = ResourceTable("illustrative_absorber_nk", "nk", SourceDocument("short-nk", b"wavelength_nm,n,k\n450,2.5,0.1\n650,2.5,0.1\n"))
        resources = ResourceLibrary(tuple(replacement if item.kind == "nk" else item for item in resources.tables))
    else:
        table = ResourceTable("conflict", "fixed_generation", SourceDocument("conflict", b'{"coordinate_m":[0,1],"generation_m3_s":[1,1]}'))
        document["fixed_generation"] = table.name
        resources = ResourceLibrary((*resources.tables, table))
    bad = edited(prepared, document, resources)
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("unsupported optics reached the nonlinear solver")
    monkeypatch.setattr(example, "prepare_case", lambda: bad)
    monkeypatch.setattr(example, "solve_steady_state", forbidden)
    monkeypatch.setattr(sys, "argv", ["optical_generation", "--output", str(tmp_path / "result.json")])
    with pytest.raises(CapabilityError):
        example.main()
    assert not (tmp_path / "result.json").exists()


def test_irradiance_is_not_silently_interpreted_as_photon_flux() -> None:
    with pytest.raises(ValueError):
        ResourceTable("bad", "spectrum", SourceDocument("irradiance", b"wavelength_nm,irradiance_W_m-2_nm-1\n400,1\n700,1\n"))


def test_failed_example_serializes_optical_binding_and_preserves_candidate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    calls = []
    def injected(device: CompiledDevice1D, **kwargs: Any) -> Any:
        calls.append(kwargs)
        result = NewtonResult(np.zeros(3 * device.node_count), False, Termination.BUDGET_EXHAUSTED,
                              123.0, (), (), 0, 0, "a" * 64, "injected serialization test; no Newton executed")
        raise SteadyStateError("serialization_test_no_solve", result)
    monkeypatch.setattr(example, "solve_steady_state", injected)
    output = tmp_path / "failure.json"
    monkeypatch.setattr(sys, "argv", ["optical_generation", "--output", str(output)])
    assert example.main() == 2 and len(calls) == 1
    result = example.read_result(output)
    assert not result["summary"]["accepted"]
    assert result["failed_newton"]["termination"] == "budget_exhausted"
    assert len(result["failed_newton"]["state"]) == 111
    assert result["optical_binding"]["physical_layer_slices"] == [
        {"name": name, "start": index, "stop": index + 1} for index, name in enumerate(("htl", "absorber", "etl"))
    ]
    assert len(result["input_resources"]) == 2
    with pytest.raises(FileExistsError):
        example.main()
    assert len(calls) == 1


def test_mono_and_fixed_generation_preserve_pre_edit_bitwise_results() -> None:
    location = os.environ.get("SOLARLAB_OPTICAL_BASELINE")
    if location is None:
        pytest.skip("explicitly provide the captured pre-edit baseline")
        return
    directory = Path(location)
    metadata = json.loads((directory / "Baseline.json").read_text())
    arrays = np.load(directory / "BeforeOpticalBinding.npz")
    base = load_device()
    mono = compile_device_1d(base, intervals_per_layer=example.INTERVALS)
    fixed_doc = base.to_input().editing_data()
    fixed_doc["fixed_generation"] = "explicit_fixed"
    raw = json.dumps({"coordinate_m": mono.nodes_m.tolist(), "generation_m3_s": np.linspace(1e25, 3e25, mono.node_count).tolist()}, separators=(",", ":")).encode()
    table = ResourceTable("explicit_fixed", "fixed_generation", SourceDocument("optical_baseline.fixed.json", raw))
    raw_doc = json.dumps(fixed_doc, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    fixed = resolve_device(DeviceInput.model_validate(fixed_doc), base.defaults, ResourceLibrary((table,)),
        sources=(*base.sources, SourceDocument("optical_baseline.fixed_declaration", raw_doc)))
    for name, p in (("monochromatic", base), ("fixed", fixed)):
        device = compile_device_1d(p, intervals_per_layer=example.INTERVALS)
        assert device.identity == metadata["branches"][name]["compiled_identity"]
        assert device.generation.binding_sha256 == metadata["branches"][name]["generation_identity"]
        problem = DC1DProblem(device, generation_scale=1.0)
        u = arrays[name + "_u"]
        for key, value in (("generation", device.generation.source_m3_s), ("absorbed", device.generation.absorbed_m2_s),
                            ("residual", problem.residual(u)), ("jacobian", problem.jacobian(u).toarray()),
                            ("current", problem.current_density(u)), ("contacts", problem.contact_current_density(u))):
            np.testing.assert_array_equal(value, arrays[name + "_" + key])


def test_saved_optical_steady_reads_without_solving(monkeypatch: pytest.MonkeyPatch) -> None:
    location = os.environ.get("SOLARLAB_OPTICAL_RESULT")
    if location is None:
        pytest.skip("read the one saved native witness without repeating it")
        return
    path = Path(location)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == os.environ["SOLARLAB_OPTICAL_RESULT_SHA256"]
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("reading a saved result attempted execution")
    monkeypatch.setattr(example, "compile_device_1d", forbidden)
    monkeypatch.setattr(example, "solve_steady_state", forbidden)
    value = example.read_result(path)
    summary = value["summary"]
    assert summary["accepted"] and summary["node_count"] == 37
    assert summary["generation_method"] == "transfer_matrix"
    assert summary["electrical_absorbed_photons_m2_s"] > 0.0
    assert summary["current_spread_A_m2"] < 1e-8
    assert all(gate["passed"] for gate in value["result"]["newton"]["gates"])
    assert summary["optical_data_scope"] == "illustrative_declared_spectrum_and_indices"
    assert summary["physical_certificate_status"] == "not_evaluated" and not summary["qualification"]
    monkeypatch.setattr(sys, "argv", ["optical_generation", "--read", location])
    assert example.main() == 0
