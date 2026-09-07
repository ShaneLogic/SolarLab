"""Contracts for analyzing captured frontend control runs."""
import copy
import importlib.util
from pathlib import Path

import pytest
import numpy as np


@pytest.fixture(scope="module")
def mod():
    source = Path(__file__).resolve().parents[3] / "scripts/summarize_ion_frontend_matrix.py"
    spec = importlib.util.spec_from_file_location("ion_frontend_matrix", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def record():
    frame = {name: [0.0, 1.0] for name in ("x", "phi", "E", "n", "p", "P", "rho")}
    frame["V_app"] = 0.0
    return {"request": {"params": {"v_rate": 0.04}}, "result": {"data": {
        "V_fwd": [0.0], "V_rev": [0.0], "snapshots_fwd": [frame],
        "snapshots_rev": [copy.deepcopy(frame)], "hysteresis_index_paper": 3e-7,
    }}}


def test_neutral_controls_compare_electronic_state_not_ionic_population(mod, record):
    empty = copy.deepcopy(record)
    empty["result"]["data"]["snapshots_fwd"][0]["P"] = [0.0, 0.0]
    report = mod.compare_neutral_controls(record, empty)
    assert report["HI_absolute_difference"] == 0
    assert set(report["maximum_absolute_state_difference"].values()) == {0.0}


def test_comparison_refuses_different_protocols(mod, record):
    other = copy.deepcopy(record)
    other["request"]["params"]["v_rate"] = 0.08
    with pytest.raises(ValueError, match="settings differ"):
        mod.compare_neutral_controls(record, other)


def test_comparison_reports_different_potential(mod, record):
    other = copy.deepcopy(record)
    other["result"]["data"]["snapshots_rev"][0]["phi"][0] = 1e-6
    report = mod.compare_neutral_controls(record, other)
    assert report["maximum_absolute_state_difference"]["phi"] == 1e-6
    assert not report["within_diagnostic_tolerance"]["phi"]


def test_roundoff_is_reported_without_claiming_bitwise_equality(mod, record):
    other = copy.deepcopy(record)
    other["result"]["data"]["snapshots_rev"][0]["phi"][0] = 1e-14
    report = mod.compare_neutral_controls(record, other)
    assert report["maximum_absolute_state_difference"]["phi"] == 1e-14
    assert report["within_diagnostic_tolerance"]["phi"]


def test_snapshot_voltage_must_match_scan(mod, record):
    record["result"]["data"]["V_fwd"] = [1.0]
    with pytest.raises(AssertionError):
        mod.snapshots(record)


def test_nonfinite_values_are_rejected(mod, record):
    record["result"]["data"]["snapshots_fwd"][0]["P"][0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        mod.snapshots(record)


@pytest.fixture
def matrix_request():
    return {"request": {"device": {"device": {"mode": "full"}, "layers": [
        {"P0": 1e25, "D_ion": 2.585e-18, "mu_n": 0.002},
    ]}, "params": {"v_rate": 0.04, "waveform": {"branch_dwell_s": 0.5}}}}


def test_ion_axis_allows_population_and_diffusivity_only(mod, matrix_request):
    other = copy.deepcopy(matrix_request)
    other["request"]["device"]["layers"][0].update(P0=0, D_ion=0)
    mod.validate_matrix({"base": matrix_request, "control": other}, "ions")
    other["request"]["device"]["layers"][0]["mu_n"] = 0.004
    with pytest.raises(ValueError, match="outside"):
        mod.validate_matrix({"base": matrix_request, "control": other}, "ions")


def test_rate_axis_requires_unchanged_ions_and_dwell(mod, matrix_request):
    other = copy.deepcopy(matrix_request)
    other["request"]["params"]["v_rate"] = 0.08
    mod.validate_matrix({"base": matrix_request, "fast": other}, "scan_rate")
    other["request"]["params"]["waveform"]["branch_dwell_s"] = 0.25
    with pytest.raises(ValueError, match="outside"):
        mod.validate_matrix({"base": matrix_request, "fast": other}, "scan_rate")


def test_rate_change_is_rejected_in_ion_matrix(mod, matrix_request):
    other = copy.deepcopy(matrix_request)
    other["request"]["params"]["v_rate"] = 0.08
    with pytest.raises(ValueError, match="outside"):
        mod.validate_matrix({"base": matrix_request, "other": other}, "ions")


def test_branch_memory_uses_si_field_and_nanometre_centroid(mod):
    from perovskite_sim.models.config_loader import load_device_from_yaml
    from perovskite_sim.physics.generation import dual_cell_widths
    stack = load_device_from_yaml("configs/calado2016_fig1f.yaml")
    x = np.array([0, 200e-9, 600e-9, 800e-9])
    data = {"V_fwd": [0.4], "V_rev": [0.4],
        "snapshots_fwd": [{"phi": [0, 0.2, 0.6, 0.8], "P": [0, 1, 1, 0]}],
        "snapshots_rev": [{"phi": [0, 0.2, 0.4, 0.6], "P": [0, 1, 1, 0]}]}
    record = {"result": {"data": data}}
    report = mod.branch_memory(record, stack, x, dual_cell_widths(x))
    assert report["fwd"]["absorber_mean_field_V_m"] == pytest.approx(-1e6)
    assert report["reverse_minus_forward_mean_field_V_m"] == pytest.approx(5e5)
    assert report["rev"]["ion_centroid_nm"] == pytest.approx(400)
    data["snapshots_rev"][0]["P"] = [0, 0, 0, 0]
    assert mod.branch_memory(record, stack, x, dual_cell_widths(x))["rev"]["ion_centroid_nm"] is None
    assert mod.branch_memory(record, stack, x, dual_cell_widths(x), voltage=0.3) is None
    data["snapshots_fwd"][0]["P_neg"] = [0, 0, 1, 0]
    data["snapshots_rev"][0]["P_neg"] = [0, 1, 0, 0]
    negative = mod.branch_memory(record, stack, x, dual_cell_widths(x))
    assert negative["fwd"]["negative_ion_centroid_nm"] == pytest.approx(600)
    assert negative["rev"]["negative_ion_centroid_nm"] == pytest.approx(200)


def test_partial_negative_snapshots_are_rejected(mod, record):
    record["result"]["data"]["snapshots_fwd"][0]["P_neg"] = [1, 2]
    with pytest.raises(ValueError, match="incomplete"):
        mod.snapshots(record)


def test_population_summary_checks_mass_and_frozen_state(mod):
    reference, widths = np.array([1.0, 1.0]), np.array([1.0, 2.0])
    mobile = np.array([[2.0, 0.5]])
    assert mod.population_summary(mobile, reference, np.ones(2), widths)["inventory_relative_drift"] == 0
    with pytest.raises(AssertionError):
        mod.population_summary(mobile, reference, np.zeros(2), widths)
    with pytest.raises(ValueError, match="conserved"):
        mod.population_summary(np.array([[2.0, 1.0]]), reference, np.ones(2), widths)
    with pytest.raises(AssertionError):
        mod.population_summary(np.array([[-1.0, 1.0]]), np.zeros(2), np.ones(2), np.ones(2))


def test_optical_budget_matches_photon_balance_and_uniform_override(mod):
    from types import SimpleNamespace
    from perovskite_sim.models.config_loader import load_device_from_yaml
    from perovskite_sim.physics.generation import beer_lambert_generation, dual_cell_widths
    stack = load_device_from_yaml("configs/calado2016_fig1f.yaml")
    x = np.array([0.0, 0.1e-6, 0.8e-6])
    mat = SimpleNamespace(alpha=np.array([0.0, 1e5, 0.0]), G_optical=None)
    source = SimpleNamespace(uniform_generation_rate_m3_s=None)
    expected = mod.expected_generation_budget(stack, source, mat, x, True)
    actual = mod.Q * (beer_lambert_generation(x, mat.alpha, stack.Phi) @ dual_cell_widths(x))
    assert actual == pytest.approx(expected, rel=1e-12)
    source.uniform_generation_rate_m3_s = 2.5e27
    assert mod.expected_generation_budget(stack, source, mat, x, True) == pytest.approx(mod.Q * 2.5e27 * 400e-9)
    assert mod.expected_generation_budget(stack, source, mat, x, False) == 0


def test_raw_current_summary_sorts_reverse_and_checks_hi(mod):
    data = {"V_fwd": [-1, 0, 0.5, 1], "J_fwd": [100, 100, 50, -10],
        "V_rev": [1, 0.5, 0, -1], "J_rev": [-10, 60, 100, 100], "hysteresis_index_paper": 0.2}
    summary = mod.current_summary(data)
    assert summary["J_sc_rev_A_m2"] == 100
    assert summary["P_max_fwd_W_m2"] == 25
    data["hysteresis_index_paper"] = 0.1
    with pytest.raises(AssertionError):
        mod.current_summary(data)


def test_optics_axis_rejects_ionic_changes(mod, matrix_request):
    other = copy.deepcopy(matrix_request)
    other["request"]["device"]["layers"][0]["alpha"] = 2e5
    other["request"]["device"]["device"]["Phi"] = 1.25e22
    mod.validate_matrix({"base": matrix_request, "optics": other}, "optics")
    other["request"]["device"]["layers"][0]["P0"] = 0
    with pytest.raises(ValueError, match="outside"):
        mod.validate_matrix({"base": matrix_request, "optics": other}, "optics")
