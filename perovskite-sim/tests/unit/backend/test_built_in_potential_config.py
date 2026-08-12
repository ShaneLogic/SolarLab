from __future__ import annotations

import dataclasses

import pytest

from backend import main as backend_main
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.scaps_compat import load_scaps_yaml


def test_compatibility_stack_roundtrip_keeps_legacy_vbi_shape():
    stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    config = backend_main._stack_to_config_dict(stack)

    assert config["device"]["V_bi"] == pytest.approx(stack.V_bi)
    assert "built_in_potential_mode" not in config["device"]
    rebuilt = backend_main.stack_from_dict(config)
    assert rebuilt.built_in_potential_mode is None
    assert rebuilt.V_bi == pytest.approx(stack.V_bi)


def test_explicit_metal_work_functions_roundtrip_without_manual_vbi():
    base = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    stack = dataclasses.replace(
        base,
        built_in_potential_mode="metal_work_function",
        work_function_left_eV=5.2,
        work_function_right_eV=4.1,
    )
    config = backend_main._stack_to_config_dict(stack)

    assert "V_bi" not in config["device"]
    assert config["device"]["built_in_potential_mode"] == "metal_work_function"
    rebuilt = backend_main.stack_from_dict(config)
    assert rebuilt.poisson_built_in_potential() == pytest.approx(1.1)
    assert rebuilt.work_function_left_eV == pytest.approx(5.2)
    assert rebuilt.work_function_right_eV == pytest.approx(4.1)


def test_semiconductor_work_function_roundtrip_preserves_dos_inputs():
    base = load_scaps_yaml("configs/scaps_mirror_v2.yaml")
    stack = dataclasses.replace(
        base,
        built_in_potential_mode="semiconductor_work_function",
    )
    config = backend_main._stack_to_config_dict(stack)
    rebuilt = backend_main.stack_from_dict(config)

    assert "V_bi" not in config["device"]
    assert rebuilt.compute_semiconductor_V_bi() == pytest.approx(
        stack.compute_semiconductor_V_bi(), abs=1e-12
    )
