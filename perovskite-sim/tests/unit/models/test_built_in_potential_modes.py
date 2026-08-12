from __future__ import annotations

import dataclasses

import pytest

from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.models.config_loader import (
    built_in_potential_fields_from_device_dict,
    load_device_from_yaml,
)
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.scaps_compat import load_scaps_yaml
from perovskite_sim.solver.mol import build_material_arrays, poisson_right_boundary
from perovskite_sim.twod.grid_2d import build_grid_2d
from perovskite_sim.twod.microstructure import Microstructure
from perovskite_sim.twod.solver_2d import build_material_arrays_2d


def _build(stack):
    layers = electrical_layers(stack)
    grid = multilayer_grid(
        [Layer(thickness=layer.thickness, N=10) for layer in layers]
    )
    return build_material_arrays(grid, stack)


def test_existing_vbi_yaml_retains_compatibility_mode_and_boundary():
    stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    mat = _build(stack)

    assert stack.built_in_potential_mode is None
    assert stack.resolved_built_in_potential_mode() == "legacy_manual"
    assert mat.V_bi_bc == pytest.approx(stack.V_bi)


def test_new_config_without_manual_value_defaults_to_physical_mode():
    fields = built_in_potential_fields_from_device_dict({"Phi": 2.5e21})

    assert fields["built_in_potential_mode"] == "semiconductor_work_function"
    assert fields["V_bi"] == pytest.approx(1.1)  # inert dataclass fallback


def test_explicit_legacy_override_uses_new_name():
    fields = built_in_potential_fields_from_device_dict(
        {
            "built_in_potential_mode": "legacy_manual",
            "V_bi_override": 1.07,
        }
    )

    assert fields["built_in_potential_mode"] == "legacy_manual"
    assert fields["V_bi"] == pytest.approx(1.07)


def test_override_rejected_outside_legacy_mode():
    with pytest.raises(ValueError, match="only valid"):
        built_in_potential_fields_from_device_dict(
            {
                "built_in_potential_mode": "semiconductor_work_function",
                "V_bi_override": 1.1,
            }
        )


def test_legacy_vbi_rejected_with_explicit_physical_mode():
    with pytest.raises(ValueError, match="legacy compatibility input"):
        built_in_potential_fields_from_device_dict(
            {
                "built_in_potential_mode": "metal_work_function",
                "V_bi": 1.1,
                "work_function_left_eV": 5.2,
                "work_function_right_eV": 4.1,
            }
        )


def test_semiconductor_work_function_is_dos_and_temperature_consistent():
    base = load_scaps_yaml("configs/scaps_mirror_v2.yaml")
    stack = dataclasses.replace(
        base,
        built_in_potential_mode="semiconductor_work_function",
    )
    mat = _build(stack)

    assert stack.compute_semiconductor_V_bi() == pytest.approx(
        1.2939750419068696, abs=1e-12
    )
    assert mat.V_bi_bc == pytest.approx(stack.compute_semiconductor_V_bi())
    assert mat.has_selective_contacts is False


def test_semiconductor_work_function_fails_closed_without_contact_dos():
    base = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    stack = dataclasses.replace(
        base,
        built_in_potential_mode="semiconductor_work_function",
    )

    with pytest.raises(ValueError, match="Nc300, Nv300"):
        _build(stack)


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (5.2, 4.1, 1.1),
        (4.1, 5.2, -1.1),
    ],
)
def test_explicit_metal_work_functions_set_signed_poisson_boundary(
    left, right, expected
):
    base = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    stack = dataclasses.replace(
        base,
        built_in_potential_mode="metal_work_function",
        work_function_left_eV=left,
        work_function_right_eV=right,
    )
    mat = _build(stack)

    assert mat.V_bi_bc == pytest.approx(expected)
    assert mat.junction_polarity == (-1.0 if expected < 0.0 else 1.0)
    assert poisson_right_boundary(mat, 0.4) == pytest.approx(
        expected - mat.junction_polarity * 0.4
    )
    assert mat.has_selective_contacts is False


def test_metal_work_function_requires_both_contacts():
    base = load_device_from_yaml("configs/ionmonger_benchmark.yaml")

    with pytest.raises(ValueError, match="requires work_function_left_eV"):
        dataclasses.replace(
            base,
            built_in_potential_mode="metal_work_function",
            work_function_left_eV=5.2,
        )


def test_explicit_mode_decouples_robin_kinetics_from_manual_potential():
    base = load_scaps_yaml("configs/scaps_mirror_v2.yaml")
    stack = dataclasses.replace(
        base,
        built_in_potential_mode="legacy_manual",
        flat_band_contacts=True,
    )
    mat = _build(stack)

    assert mat.V_bi_bc == pytest.approx(base.V_bi)
    assert mat.has_selective_contacts is True
    assert mat.S_n_L == pytest.approx(1.0e5)
    assert mat.S_p_R == pytest.approx(1.0e5)


def test_2d_reuses_signed_1d_contact_electrostatics():
    base = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    stack = dataclasses.replace(
        base,
        built_in_potential_mode="metal_work_function",
        work_function_left_eV=4.1,
        work_function_right_eV=5.2,
    )
    layers = electrical_layers(stack)
    grid = build_grid_2d(
        [Layer(layer.thickness, N=6) for layer in layers],
        lateral_length=1.0e-6,
        Nx=2,
        lateral_uniform=True,
    )
    mat = build_material_arrays_2d(grid, stack, Microstructure())

    assert mat.V_bi == pytest.approx(-1.1)
    assert mat.junction_polarity == -1.0
