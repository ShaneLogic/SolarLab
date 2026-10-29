"""Original photon counts, half cells, source identity and saved-only reading."""

import ast
from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np
import pytest

from examples.research_2d import optical_generation as example
from solarlab.config.resolve_device import resolve_device
from solarlab.device.inputs import DeviceInput
from solarlab.device.optical_generation import bind_optical_generation_1d
from solarlab.device.optical_generation_2d import bind_optical_generation_2d
from solarlab.mesh import INTEGER_ENDPOINT_V1, build_geometry
from solarlab.mesh.geometry import half_endpoint_geometry
from solarlab.mesh.geometry_2d import RectangularGeometry2D
from solarlab.physics.generation.profile import StaleGenerationProfileError


@pytest.fixture(scope="module")
def prepared():
    return example.prepare_1d_case()


def edited(prepared, data):
    return resolve_device(DeviceInput.model_validate(data), prepared.defaults, prepared.resources, sources=prepared.sources)


def absorbing_contacts(prepared):
    data = prepared.to_input().editing_data()
    for index in (0, 2):
        data["layers"][index]["parameters"]["alpha"] = 1e6
    return edited(prepared, data)


def test_half_endpoint_source_matches_legacy_conversion_exactly_once(prepared):
    prepared = absorbing_contacts(prepared)
    geometry = example.research_geometry(prepared)
    bound = bind_optical_generation_2d(prepared, geometry)
    old_geometry = build_geometry(INTEGER_ENDPOINT_V1, geometry.y_geometry.nodes_m)
    old = bind_optical_generation_1d(prepared, old_geometry).profile
    # Execute only the immutable legacy reweighting function, never its engine.
    root = Path(__file__).resolve().parents[2]
    path = root / "perovskite-sim/perovskite_sim/twod/solver_2d.py"
    function = next(node for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "_to_2d_dual_cell")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    expected = namespace["_to_2d_dual_cell"](old.source_m3_s)
    assert np.all(old.source_m3_s[[0, -1]] > 0.0)
    np.testing.assert_array_equal(bound.optical_y.profile.source_m3_s, expected)
    np.testing.assert_array_equal(bound.optical_y.profile.absorbed_m2_s, old.absorbed_m2_s)
    np.testing.assert_array_equal(bound.source_m3_s.reshape(13, 5), np.repeat(expected[:, None], 5, axis=1))
    assert np.sum(bound.source_m3_s * geometry.contract_geometry.volumes) == pytest.approx(
        geometry.lateral_length_m * old.areal_generation_m2_s, rel=1e-12)


@pytest.mark.parametrize("width", [1e-6, 2.7e-6])
def test_actual_irregular2d_cell_photons_and_per_depth_units(prepared, width):
    reference = example.research_geometry(prepared)
    geometry = RectangularGeometry2D(width * np.array([0.0, .07, .22, .61, 1.0]), reference.y_geometry)
    bound = bind_optical_generation_2d(prepared, geometry)
    budget = bound.optical_y.budget
    np.testing.assert_allclose(bound.source_m3_s * geometry.contract_geometry.volumes,
        bound.cell_generation_per_depth_m_s, rtol=5e-16, atol=0.0)
    assert budget.incident_m2_s == pytest.approx(6e27 * 300e-9, rel=1e-15)
    assert np.max(abs(budget.spectral_closure_residual())) < 1e-12
    assert bound.particle_generation_per_depth_m_s == pytest.approx(width * budget.electrical_absorbed_m2_s, rel=1e-12)
    assert bound.generation_current_limit_A_m2 == pytest.approx(bound.generation_current_limit_A_m / width, rel=1e-15)
    np.testing.assert_array_equal(bound.source_m3_s.reshape(13, 5)[:, 0], bound.source_m3_s.reshape(13, 5)[:, -1])
    for value in (bound.source_m3_s, bound.cell_generation_per_depth_m_s):
        assert not value.flags.writeable
        with pytest.raises(ValueError):
            value.flags.writeable = True


def test_front_optical_prefix_absorption_remains_parasitic(prepared):
    data = prepared.to_input().editing_data()
    prefix = {"id": "window", "name": "Illustrative optical window", "role": "substrate",
              "thickness": 70e-9, "parameters": {"alpha": 5e5, "n_optical": 1.7, "incoherent": False}}
    data["layers"] = [prefix, *data["layers"]]
    changed = edited(prepared, data)
    geometry = example.research_geometry(changed)
    bound = bind_optical_generation_2d(changed, geometry)
    b = bound.optical_y.budget
    assert bound.optical_y.binding.electrical_offset_m == 70e-9
    assert b.layer_electrical[0] is False and b.parasitic_absorbed_m2_s > 0.0
    assert b.reflected_m2_s + b.transmitted_m2_s + b.electrical_absorbed_m2_s + b.parasitic_absorbed_m2_s == pytest.approx(b.incident_m2_s, rel=1e-12)
    assert bound.particle_generation_per_depth_m_s / geometry.lateral_length_m == pytest.approx(b.electrical_absorbed_m2_s, rel=1e-12)


def test_absolute_spectrum_off_phi_and_source_mesh_admission(prepared):
    geometry = example.research_geometry(prepared)
    original = bind_optical_generation_2d(prepared, geometry)
    data = prepared.to_input().editing_data()
    data["settings"]["Phi"] = 9e24
    changed_prepared = edited(prepared, data)
    changed = bind_optical_generation_2d(changed_prepared, geometry)
    np.testing.assert_array_equal(original.source_m3_s, changed.source_m3_s)
    assert original.identity_sha256 != changed.identity_sha256
    with pytest.raises(StaleGenerationProfileError):
        original.require_inputs(changed_prepared, geometry)
    other = RectangularGeometry2D(geometry.x_m * 2.0, geometry.y_geometry)
    with pytest.raises(StaleGenerationProfileError):
        original.require_inputs(prepared, other)
    with pytest.raises(StaleGenerationProfileError):
        replace(original, geometry=RectangularGeometry2D(geometry.x_m,
            half_endpoint_geometry(geometry.y_geometry.nodes_m * .99)))


def test_lossless_stack_has_closed_form_airy_and_zero_generation(prepared):
    data = prepared.to_input().editing_data()
    for layer in data["layers"]:
        layer["parameters"].update(optical_material=None, n_optical=2.0, alpha=0.0)
    changed = edited(prepared, data)
    geometry = example.research_geometry(changed)
    bound = bind_optical_generation_2d(changed, geometry)
    b = bound.optical_y.budget
    r = (1.0 - 2.0) / (1.0 + 2.0)
    phase = np.cos(4.0 * np.pi * 2.0 * geometry.y_geometry.extent_m / bound.optical_y.binding.wavelengths_m)
    expected = 2.0 * r**2 * (1.0 - phase) / (1.0 + r**4 - 2.0 * r**2 * phase)
    np.testing.assert_allclose(b.reflectance, expected, rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(b.transmittance, 1.0 - expected, rtol=0.0, atol=1e-13)
    np.testing.assert_array_equal(bound.source_m3_s, 0.0)


@pytest.mark.parametrize("kind", ["no_spectrum", "incoherent", "missing_index", "ambiguous_index", "graded"])
def test_original_optical_refusals_remain_explicit(prepared, kind):
    data = prepared.to_input().editing_data()
    if kind == "no_spectrum":
        data["spectrum"] = None
    elif kind == "incoherent":
        data["layers"][0]["parameters"]["incoherent"] = True
    elif kind == "missing_index":
        data["layers"][0]["parameters"]["n_optical"] = None
    elif kind == "ambiguous_index":
        data["layers"][1]["parameters"]["n_optical"] = 2.1
    else:
        data["settings"]["graded_optics"] = True
    with pytest.raises(ValueError):
        changed = edited(prepared, data)
        bind_optical_generation_2d(changed, example.research_geometry(changed))


def test_saved_source_reader_performs_no_optical_or_carrier_solve(prepared, tmp_path, monkeypatch):
    output = tmp_path / "Optical.json"
    example.construct(output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    def forbidden(*args, **kwargs):
        raise AssertionError("saved optical source reader recomputed physics")
    monkeypatch.setattr(example, "bind_optical_generation_2d", forbidden)
    read = example.read_result(output, digest)
    assert read["summary"]["source_constructed"] and read["summary"]["nonlinear_solve_calls"] == 0
    assert not read["summary"]["coupled_2d_operating_point_accepted"] and not read["qualification"]
    with pytest.raises(ValueError):
        example.read_result(output, "0" * 64)
    with pytest.raises(FileExistsError):
        example.construct(output)


def test_resource_document_identity_is_not_replaced_by_equal_sample_values(prepared):
    from solarlab.materials.resources import ResourceLibrary, ResourceTable
    from solarlab.materials.source import SourceDocument
    original = prepared.resources.get("illustrative_photons", "spectrum")
    changed_table = ResourceTable(original.name, original.kind,
        SourceDocument(original.source.id, b"# independent supplied document\n" + original.source.content))
    resources = ResourceLibrary(tuple(changed_table if item.name == original.name else item
                                      for item in prepared.resources.tables))
    changed = resolve_device(prepared.to_input(), prepared.defaults, resources, sources=prepared.sources)
    geometry = example.research_geometry(prepared)
    before = bind_optical_generation_2d(prepared, geometry)
    after = bind_optical_generation_2d(changed, geometry)
    np.testing.assert_array_equal(before.source_m3_s, after.source_m3_s)
    assert before.identity_sha256 != after.identity_sha256
    with pytest.raises(StaleGenerationProfileError):
        before.require_inputs(changed, geometry)


def coupled_device(prepared):
    from examples.research_2d.dc_assembly import spatial_input
    from solarlab.device import compile_2d
    if "bind_optical_generation_2d" not in vars(compile_2d):
        pytest.skip("shared optical hooks are tested against the isolated candidate")
    return compile_2d.compile_device_2d(prepared, spatial_input())


def test_isolated_original195_rows_jacobian_and_contact_half_cell_photons(prepared):
    from solarlab.equations.dc_2d import DC2DProblem
    from solarlab.solvers.nonlinear.steady_2d import _scope
    prepared = absorbing_contacts(prepared)
    device = coupled_device(prepared)
    _scope(device)
    assert device.node_count == 65 and device.generation_y.method == "transfer_matrix"
    bound = bind_optical_generation_2d(prepared, device.geometry)
    np.testing.assert_array_equal(device.generation_source_m3_s, bound.source_m3_s)
    u = device.extrude_coordinates(np.r_[device.column.reference_potential_V, np.zeros(2 * device.geometry.ny)])
    dark, light = DC2DProblem(device, 0.0, 0.0), DC2DProblem(device, 0.0, 1.0)
    difference = light.residual(u) - dark.residual(u)
    assert difference.shape == (195,)
    np.testing.assert_array_equal(difference[:65], 0.0)
    expected = device.generation_source_m3_s * device.geometry.contract_geometry.volumes
    masked = expected.copy()
    masked[device.geometry.contact_nodes] = 0.0
    for block in (difference[65:130], difference[130:]):
        np.testing.assert_allclose(block, masked, rtol=1e-12, atol=0.0)
    np.testing.assert_array_equal(light.jacobian(u).toarray(), dark.jacobian(u).toarray())
    # Both contact fluxes point along+y. Generation in the lower half cell
    # decreases incoming particles; generation in the upper increases outgoing.
    flux_delta = np.stack(light.contact_particle_fluxes(u)) - np.stack(dark.contact_particle_fluxes(u))
    source_y = device.generation_y.source_m3_s
    contact_expected = np.repeat(np.array([[-source_y[0] * device.geometry.y_geometry.cell_measures_m[0]],
                                          [source_y[-1] * device.geometry.y_geometry.cell_measures_m[-1]]]), 5, axis=1)
    for carrier in flux_delta:
        np.testing.assert_allclose(carrier, contact_expected, rtol=1e-12, atol=0.0)


def test_isolated_active_optical_sources_are_captured_and_off_stays_old(prepared, monkeypatch):
    import sys
    from examples.research_1d.steady import load_device
    from solarlab.solvers.nonlinear.steady_2d import _sources
    active, off = coupled_device(prepared), coupled_device(load_device())
    capture = _sources(active)
    names = ("solarlab.device.optical_generation_2d", "solarlab.device.optical_generation",
             "solarlab.physics.generation.optical_binding", "solarlab.physics.generation.transfer_matrix")
    for name in names:
        assert capture[name] == hashlib.sha256(Path(sys.modules[name].__file__).read_bytes()).hexdigest()
        assert name not in _sources(off)
    assert _sources(off) == _sources()
    old = capture[names[0]]
    monkeypatch.setattr(sys.modules[names[0]], "__file__", sys.modules[names[1]].__file__)
    assert _sources(active)[names[0]] != old


def test_optical_prefix_is_source_only_until_electrical_compiler_admission(prepared):
    data = prepared.to_input().editing_data()
    data["layers"] = [{"id": "window", "name": "Optical prefix", "role": "substrate", "thickness": 70e-9,
                       "parameters": {"alpha": 5e5, "n_optical": 1.7, "incoherent": False}}, *data["layers"]]
    with pytest.raises(ValueError, match="optical-only/index"):
        coupled_device(edited(prepared, data))
