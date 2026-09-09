"""Replay selected internal figure curves without fitting physical inputs."""
from dataclasses import fields
import json
from pathlib import Path

import numpy as np
import pytest

import backend.main as backend
from perovskite_sim.experiments import jv_sweep


REFERENCE = json.loads(
    (Path(__file__).parents[1] / "fixtures/IonDiffusivityFigureReference.json").read_text()
)


def figure_config():
    return backend.get_config("calado2016_ion_sweep.yaml")["config"]


def test_figure_preset_preserves_captured_inputs_and_numerical_defaults():
    config = figure_config()
    captured = REFERENCE["request"]
    defaults = config["simulation_hints"]["jv_sweep"]
    assert defaults == {key: captured["params"][key] for key in defaults}
    assert config["device"]["mode"] == "full"
    assert all(config["device"][key] is None for key in (
        "S_n_left", "S_p_left", "S_n_right", "S_p_right",
    ))
    actual = backend.stack_from_dict(config)
    expected = backend.stack_from_dict(captured["device"])
    x = jv_sweep.build_electrical_grid(actual, defaults["N_grid"])
    expected_x = jv_sweep.build_electrical_grid(expected, captured["params"]["N_grid"])
    np.testing.assert_array_equal(x, expected_x)
    assert len(x) == 61
    actual_mat = jv_sweep.build_material_arrays(x, actual)
    expected_mat = jv_sweep.build_material_arrays(x, expected)
    for field in fields(actual_mat):
        left, right = getattr(actual_mat, field.name), getattr(expected_mat, field.name)
        if isinstance(left, np.ndarray):
            np.testing.assert_allclose(left, right, rtol=1e-14, atol=0, err_msg=field.name)
        elif isinstance(left, (str, bool)) or left is None:
            assert left == right, field.name
        elif isinstance(left, (int, float)):
            assert left == pytest.approx(right, rel=1e-14, abs=0), field.name


@pytest.mark.slow
@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: f"D{case['ratio']:g}")
def test_figure_subcurve_matches_archived_current_and_hi(case):
    config = figure_config()
    absorber = next(layer for layer in config["layers"] if layer["role"] == "absorber")
    absorber["D_ion"] = REFERENCE["reference_D_ion_m2_s"] * case["ratio"]
    settings = config["simulation_hints"]["jv_sweep"]
    result = backend._run_jv_dispatch(
        backend.stack_from_dict(config), **settings, solver="transient", illuminated=True,
    )
    for branch in ("fwd", "rev"):
        reference = case["curves"][branch]
        np.testing.assert_allclose(
            getattr(result, f"V_{branch}")[1:], reference["V"], rtol=0, atol=1e-12,
        )
        np.testing.assert_allclose(
            getattr(result, f"J_{branch}")[1:], reference["J_A_m2"], rtol=1e-5, atol=1e-3,
        )
    assert result.hysteresis_index_paper == pytest.approx(
        case["hysteresis_index_paper"], rel=2e-4, abs=1e-5,
    )
    assert result.numerical_controls["actual_N_grid"] == 61
