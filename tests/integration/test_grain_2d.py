"""Fractional neutral grain SRH: independent geometry/rate/assembly oracles."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from examples.research_1d.steady import load_device
from examples.research_2d.dc_assembly import coordinates, spatial_input
from solarlab.config.resolve_device import resolve_device
from solarlab.device.compile_2d import compile_device_2d
from solarlab.device.grain_2d import bind_grain_regions_2d, grain_declarations
from solarlab.device.inputs import DeviceInput
from solarlab.device.resolved import PreparedDevice
from solarlab.equations.dc_2d import CompiledDevice2D, DC2DProblem
from solarlab.experiments.two_dimensional.inputs import JV2DInput
from solarlab.mesh.geometry import half_endpoint_geometry
from solarlab.mesh.geometry_2d import RectangularGeometry2D
from solarlab.physics.recombination.reaction import BulkRecombinationCoefficients, BulkRecombinationReaction


def grain(**changes: Any) -> dict[str, Any]:
    return {"id": "grain_a", "layer_ids": ["absorber"], "x_position": "0.4 um",
            "width": "0.3 um", "tau_n": "0.5 us", "tau_p": "0.25 us", **changes}


def spatial(grains: list[dict[str, Any]]) -> JV2DInput:
    return JV2DInput.model_validate({**spatial_input().editing_data(), "microstructure": {"grain_boundaries": grains}})


@pytest.fixture(scope="module")
def prepared() -> PreparedDevice:
    return load_device()


@pytest.fixture(scope="module")
def baseline(prepared: PreparedDevice) -> CompiledDevice2D:
    return compile_device_2d(prepared, spatial_input())


def with_device_grains(prepared: PreparedDevice) -> PreparedDevice:
    data = prepared.to_input().editing_data()
    data["grain_boundaries"] = [grain()]
    return resolve_device(DeviceInput.model_validate(data), prepared.defaults, prepared.resources, sources=prepared.sources)


def test_fractional_geometry_and_original_interface_ownership(prepared: PreparedDevice) -> None:
    # Explicit dual cells: x bounds [0,.1,.45,.85,1] um; grain [.25,.55] um.
    y = np.array([0, 40, 80, 200, 380, 420, 460]) * 1e-9
    geometry = RectangularGeometry2D(np.array([0, .2, .7, 1]) * 1e-6, half_endpoint_geometry(y))
    binding = bind_grain_regions_2d(prepared, spatial([grain()]), geometry)
    assert binding is not None
    region = binding.regions[0]
    expected_x = np.array([0, .2/.35, .1/.4, 0])
    expected_y = np.array([False, False, False, True, True, False, False])
    np.testing.assert_allclose(region.x_overlap_fraction, expected_x, rtol=2e-15, atol=0)
    np.testing.assert_array_equal(region.y_mask, expected_y)
    # y=80 nm stays HTL, y=380 nm stays absorber for grain selection.
    # The represented height is 150+110=260 nm, not nominal 300 nm.
    assert region.represented_height_m == pytest.approx(260e-9, rel=2e-15)
    assert region.represented_width_m == pytest.approx(300e-9, rel=2e-15)
    assert region.represented_area_m2 == pytest.approx(300e-9 * 260e-9, rel=2e-15)
    for value in (region.fractions, region.x_overlap_fraction, region.y_mask):
        with pytest.raises(ValueError):
            value.setflags(write=True)


def test_subcell_regions_keep_physical_width_and_can_share_a_cell(prepared: PreparedDevice, baseline: CompiledDevice2D) -> None:
    grains = [grain(x_position="0.48 um", width="0.02 um"),
              grain(id="grain_b", x_position="0.52 um", width="0.02 um")]
    binding = bind_grain_regions_2d(prepared, spatial(grains), baseline.geometry)
    assert binding is not None
    for region in binding.regions:
        assert np.count_nonzero(region.x_overlap_fraction) == 1
        assert np.max(region.x_overlap_fraction) == pytest.approx(.08)
        assert region.represented_width_m == pytest.approx(20e-9)
    assert np.any(binding.regions[0].fractions * binding.regions[1].fractions > 0)


def test_fractional_srh_rate_and_exact_partials_independent_oracle(prepared: PreparedDevice, baseline: CompiledDevice2D) -> None:
    binding = bind_grain_regions_2d(prepared, spatial([grain()]), baseline.geometry)
    assert binding is not None
    size = baseline.node_count
    # Deliberately unequal coefficients and nonzero radiative/Auger channels.
    n, p = np.linspace(2e19, 3e19, size), np.linspace(4e19, 7e19, size)
    coeff = BulkRecombinationCoefficients(size, np.full(size, 1e30), np.full(size, 2e-6),
        np.full(size, 3e-6), np.full(size, 5e14), np.full(size, 7e14), np.full(size, 1e-18),
        np.full(size, 2e-42), np.full(size, 3e-42))
    evaluation = BulkRecombinationReaction(coeff).evaluate(n, p)
    actual = binding.evaluate(evaluation, coeff)
    f = binding.regions[0].fractions
    numerator = n*p - coeff.ni_sq_m6
    db = 3e-6*(n+5e14) + 2e-6*(p+7e14)
    dg = .25e-6*(n+5e14) + .5e-6*(p+7e14)
    independent = (1-f)*numerator/db + f*numerator/dg + (1e-18+2e-42*n+3e-42*p)*numerator
    dn = ((1-f)*(p*db - numerator*3e-6)/db**2 + f*(p*dg - numerator*.25e-6)/dg**2
          + 1e-18*p + 2e-42*numerator + (2e-42*n+3e-42*p)*p)
    dp = ((1-f)*(n*db - numerator*2e-6)/db**2 + f*(n*dg - numerator*.5e-6)/dg**2
          + 1e-18*n + 3e-42*numerator + (2e-42*n+3e-42*p)*n)
    for result, expected in ((actual.rate, independent), (actual.d_rate_dn, dn), (actual.d_rate_dp, dp)):
        np.testing.assert_allclose(result, expected, rtol=6e-16)
    np.testing.assert_array_equal(actual.rate[f == 0], evaluation.total.rate[f == 0])
    empty = replace(binding, regions=())
    assert empty.evaluate(evaluation, coeff) is evaluation.total
    no_derivative = BulkRecombinationReaction(coeff).evaluate(n, p, derivatives=False)
    assert not binding.evaluate(no_derivative, coeff).has_derivatives


@pytest.mark.parametrize("override", [None, {}])
def test_null_and_empty_object_inherit_device_inventory(prepared: PreparedDevice, override: Any) -> None:
    d = with_device_grains(prepared)
    request = JV2DInput.model_validate({**spatial_input().editing_data(), "microstructure": override})
    assert len(grain_declarations(d, request)) == 1
    assert len(grain_declarations(d, spatial_input())) == 1
    assert grain_declarations(d, spatial([])) == ()


@pytest.mark.parametrize("changes", [
    {"layer_ids": []}, {"layer_ids": ["missing"]}, {"layer_ids": ["absorber", "absorber"]},
    {"x_position": "0.01 um"}, {"x_position": "0.99 um"}, {"tau_n": 0},
    {"tau_p": float("inf")}, {"width": float("nan")}, {"charge": 1},
])
def test_invalid_declarations_refuse_before_column_compilation(prepared: PreparedDevice, monkeypatch: pytest.MonkeyPatch,
                                                             changes: dict[str, Any]) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("invalid grain reached coefficient compilation")
    monkeypatch.setattr("solarlab.device.compile_2d.compile_device_1d", forbidden)
    with pytest.raises(ValueError):
        compile_device_2d(prepared, spatial([grain(**changes)]))


@pytest.mark.parametrize("other", [grain(), grain(id="grain_b", x_position="0.45 um")])
def test_duplicate_ids_and_physical_overlap_refused(prepared: PreparedDevice, other: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="duplicate|overlapping"):
        grain_declarations(prepared, spatial([grain(), other]))


def test_layer_ids_are_authoritative_and_role_is_retained_provenance(prepared: PreparedDevice, baseline: CompiledDevice2D) -> None:
    binding = bind_grain_regions_2d(prepared, spatial([grain(source_layer_role="HTL")]), baseline.geometry)
    assert binding is not None
    assert binding.regions[0].declaration.source_layer_role == "HTL"
    np.testing.assert_array_equal(binding.regions[0].y_mask, np.isin(np.arange(13), [5, 6, 7, 8]))


def test_compile_binds_grains_and_retains_physical_material_and_charge(prepared: PreparedDevice, baseline: CompiledDevice2D) -> None:
    d = compile_device_2d(prepared, spatial([grain()]))
    assert d.grain_recombination is not None
    assert d.identity != baseline.identity and d.column.identity == baseline.column.identity
    assert d.grain_recombination.regions[0].y_mask[8]
    assert baseline.column.parameters.layer_masks[2][8]  # RIGHT_LAYER_WINS: ETL material.
    for name in ("ni_sq_m6", "n1_m3", "p1_m3", "tau_n_s", "tau_p_s", "radiative_m3_s", "auger_n_m6_s", "auger_p_m6_s"):
        np.testing.assert_array_equal(getattr(d.recombination.coefficients, name), getattr(baseline.recombination.coefficients, name))
    _, u = coordinates(d, lateral=True)
    _, n, p = d.fields(u, voltage_V=.002)
    np.testing.assert_array_equal(d.charge_ledger(n, p).space_charge_C_m3(), baseline.charge_ledger(n, p).space_charge_C_m3())
    np.testing.assert_array_equal(d.generation_source_m3_s, baseline.generation_source_m3_s)
    inherited = compile_device_2d(with_device_grains(prepared), spatial_input())
    assert inherited.grain_recombination is not None
    np.testing.assert_array_equal(inherited.grain_recombination.regions[0].fractions, d.grain_recombination.regions[0].fractions)


def test_no_grain_path_has_exact_original_identity_and_algebra(prepared: PreparedDevice, baseline: CompiledDevice2D) -> None:
    # Independent reconstruction of the original no-grain identity expression.
    declaration = spatial_input().model_dump_json()
    expected = hashlib.sha256(json.dumps(("classical_mb_fixed_reservoir_neumann_2d_dc_v1", prepared.content_sha256,
        baseline.column.identity, baseline.geometry.sha256, baseline.generation_y.binding_sha256, declaration),
        separators=(",", ":")).encode()).hexdigest()
    assert baseline.identity == expected and baseline.grain_recombination is None
    cleared = compile_device_2d(with_device_grains(prepared), spatial([]))
    assert cleared.grain_recombination is None
    _, u = coordinates(baseline, lateral=True)
    p0, p1 = DC2DProblem(baseline, .002, 1), DC2DProblem(cleared, .002, 1)
    for a, b in ((p0.residual(u), p1.residual(u)), (p0.jacobian(u).toarray(), p1.jacobian(u).toarray()),
                 (p0.terminal_current_A_m(u), p1.terminal_current_A_m(u))):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("potential_only", [False, True])
def test_full_grain_jacobian_and_volumetric_source_units(prepared: PreparedDevice, baseline: CompiledDevice2D,
                                                      potential_only: bool) -> None:
    d = compile_device_2d(prepared, spatial([grain()]))
    problem, bulk = DC2DProblem(d, .002, 0), DC2DProblem(baseline, .002, 0)
    _, u = coordinates(d, lateral=True)
    # Constant quasi-Fermi coordinates make SG transport exactly zero. This
    # isolates the volumetric source without subtracting a small SRH change
    # from large off-steady transport rows (a cancellation-prone oracle).
    source_u = u.copy()
    source_u[d.node_count:2*d.node_count] = .3
    source_u[2*d.node_count:] = -.2
    _, n, p = d.fields(source_u, voltage_V=.002)
    delta = problem.recombination_rate(n, p).rate - bulk.recombination_rate(n, p).rate
    expected = -delta * d.geometry.contract_geometry.volumes  # [m^-3/s]*[m²] -> [1/(m s)].
    expected[d.geometry.contact_nodes] = 0
    residual_delta = (problem.residual(source_u) - bulk.residual(source_u)).reshape(3, -1)
    np.testing.assert_array_equal(residual_delta[0], np.zeros(d.node_count))
    for block in residual_delta[1:]:
        np.testing.assert_allclose(block, expected, rtol=2e-14, atol=0)
    matrix = problem.jacobian(u)
    rng = np.random.default_rng(772)
    direction = np.r_[.02*rng.normal(size=d.node_count), rng.normal(size=2*d.node_count)]
    if potential_only:
        direction[d.node_count:] = 0
    h = 2e-4
    numerical = (problem.residual(u-2*h*direction)-8*problem.residual(u-h*direction)
                 + 8*problem.residual(u+h*direction)-problem.residual(u+2*h*direction))/(12*h)
    scale = np.maximum(abs(matrix) @ abs(direction), 1e-20)
    assert np.max(abs(numerical-matrix@direction)/scale) < 2e-8


@pytest.mark.parametrize("changes", [{"width": "0.2 um"}, {"x_position": "0.6 um"},
                                     {"tau_n": "0.3 us"}, {"tau_p": "0.6 us"}, {"layer_ids": ["etl"]}])
def test_active_identity_tracks_every_physical_grain_input(prepared: PreparedDevice, changes: dict[str, Any]) -> None:
    first = compile_device_2d(prepared, spatial([grain()]))
    changed = compile_device_2d(prepared, spatial([grain(**changes)]))
    assert first.identity != changed.identity
    assert first.grain_recombination.model_id != changed.grain_recombination.model_id


def test_example_algebra_and_lossless_reader_without_solver(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from examples.research_2d import grains as example
    from solarlab.experiments.transient_results import _encode
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("algebra/read must not launch any Newton solve")
    monkeypatch.setattr(example, "solve_steady_state", forbidden)
    monkeypatch.setattr(example, "solve_steady_state_2d", forbidden)
    path = tmp_path / "GrainAlgebra.json"
    result = example.algebra(path)
    assert result["newton_calls"] == 0 and result["directional_relative_error"] < 2e-8
    monkeypatch.setattr(example, "compile_device_2d", forbidden)
    reread = example.read_result(path, hashlib.sha256(path.read_bytes()).hexdigest())
    assert json.dumps(_encode(reread), sort_keys=True) == json.dumps(_encode(result), sort_keys=True)
    reread["grain_binding"]["regions"][0]["represented_area_m2"] *= 2
    raw = json.dumps(_encode(reread)).encode()
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="represented area"):
        example.read_result(path, hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError, match="SHA-256"):
        example.read_result(path, "0"*64)


def test_saved_native_grain_result_reuse_without_resimulation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit opt-in saved witness; the default test path never simulates it."""
    from examples.research_2d import grains as example
    saved = os.environ.get("SOLARLAB_GRAIN_2D_SAVED_RESULT")
    digest = os.environ.get("SOLARLAB_GRAIN_2D_SAVED_SHA256")
    if saved is None or digest is None:
        pytest.skip("explicit saved grain2D result and SHA are required; no automatic native run")
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("saved-result tests must not compile or solve")
    for name in ("compile_device_2d", "solve_steady_state", "solve_steady_state_2d"):
        monkeypatch.setattr(example, name, forbidden)
    result = example.read_result(Path(saved), digest)
    assert result["completed"] and result["accepted"] and not result["qualification"]
    assert result["full_2d_newton_calls"] == 1 and result["auxiliary_1d_seed_calls"] == 1
    assert not result["auxiliary_seed_is_2d_completion"]
    diagnostics = result["result_2d"]["diagnostics"]
    assert np.max(abs(diagnostics["lateral_current_density_A_m2"])) > 0
    shape = tuple(result["geometry"]["shape_Ny_Nx"])
    assert len(diagnostics["original_rows"]) == 3*int(np.prod(shape))
