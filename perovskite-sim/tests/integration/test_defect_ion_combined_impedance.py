"""Joint device-level closure for explicit defects and mobile ions."""

from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from perovskite_sim.experiments.defect_ion_combined_impedance import (
    DEFECT_ION_COMBINED_SCOPE,
    DefectIonCombinedCertificationError,
    DefectIonCombinedError,
    run_defect_ion_combined_impedance,
)
from perovskite_sim.models.defects import (
    EFFECTIVE_LIFETIME,
    EXPLICIT_DEFECT_SCHEMA_VERSION,
    EXPLICIT_QUASI_STEADY,
)
from perovskite_sim.physics.temperature import thermal_voltage
from tests.integration.test_charged_explicit_defects_qf import (
    _grid as _bulk_grid,
)
from tests.integration.test_charged_explicit_defects_qf import (
    _species,
    _stack as _bulk_defect_stack,
)
from tests.integration.test_interface_defect_aware_impedance import (
    _grid as _interface_grid,
)
from tests.integration.test_interface_defect_aware_impedance import (
    _research_interface_stack,
)


BULK_FREQUENCIES_HZ = np.logspace(-3.0, 6.0, 19)
INTERFACE_FREQUENCIES_HZ = np.logspace(-3.0, 2.0, 11)
ION_DIFFUSION_M2_S = 1.0e-14
ION_DENSITY_M3 = 1.0e22
ION_LIMIT_M3 = 2.0e22


def _with_mobile_ions(
    stack,
    *,
    positive: bool = True,
    negative: bool = False,
):
    site_limit = ION_LIMIT_M3 * (2.0 if positive and negative else 1.0)
    layers = []
    for layer in stack.layers:
        params = layer.params
        assert params is not None
        params = replace(
            params,
            D_ion=ION_DIFFUSION_M2_S if positive else 0.0,
            P0=ION_DENSITY_M3 if positive else 0.0,
            P_lim=site_limit,
            D_ion_neg=ION_DIFFUSION_M2_S if negative else 0.0,
            P0_neg=ION_DENSITY_M3 if negative else 0.0,
            P_lim_neg=site_limit,
        )
        layers.append(replace(layer, params=params))
    return replace(
        stack,
        layers=tuple(layers),
        mode="full" if negative else stack.mode,
    )


def _contact_consistent_interface_stack():
    stack = _research_interface_stack()
    layers = []
    for layer in stack.layers:
        params = layer.params
        assert params is not None
        intrinsic = math.sqrt(
            params.Nc300 * params.Nv300 * math.exp(-params.Eg / thermal_voltage(300.0))
        )
        layers.append(
            replace(
                layer,
                params=replace(
                    params,
                    ni=intrinsic,
                    n1=intrinsic,
                    p1=intrinsic,
                ),
            )
        )
    return replace(
        stack,
        layers=tuple(layers),
        built_in_potential_mode="semiconductor_work_function",
    )


def _bulk_interface_ion_stack():
    stack = _contact_consistent_interface_stack()
    layers = []
    for index, layer in enumerate(stack.layers):
        params = layer.params
        assert params is not None
        updates = {
            "D_ion": ION_DIFFUSION_M2_S,
            "P0": ION_DENSITY_M3,
            "P_lim": ION_LIMIT_M3,
        }
        if index == 0:
            updates.update(
                defect_schema_version=EXPLICIT_DEFECT_SCHEMA_VERSION,
                defect_model=EXPLICIT_QUASI_STEADY,
                bulk_defects=(_species("acceptor"),),
            )
        layers.append(replace(layer, params=replace(params, **updates)))
    return replace(stack, layers=tuple(layers))


def _assert_current_decomposition(result) -> None:
    expected = (
        result.electron_admittance_faces_S_m2
        + result.hole_admittance_faces_S_m2
        + result.positive_ion_admittance_faces_S_m2
        + result.displacement_admittance_faces_S_m2
    )
    if result.negative_ion_admittance_faces_S_m2 is not None:
        expected = expected + result.negative_ion_admittance_faces_S_m2
    np.testing.assert_allclose(
        result.admittance_faces_S_m2,
        expected,
        rtol=2.0e-15,
        atol=1.0e-12,
    )


def test_bulk_defect_and_positive_ion_share_one_certified_operator():
    stack = _with_mobile_ions(_bulk_defect_stack())
    grid = _bulk_grid(stack, 4)

    result = run_defect_ion_combined_impedance(
        grid,
        stack,
        BULK_FREQUENCIES_HZ,
    )

    assert result.scope == DEFECT_ION_COMBINED_SCOPE
    assert result.certificate.certified
    assert result.certificate.reasons == ()
    assert result.certificate.capability == "bulk_defect_plus_ions"
    assert result.certificate.frequency_window.certified
    assert result.dc_state.certificate.certified
    assert result.dc_state.certificate.contact_thermodynamics.certified
    assert result.layout.bulk_trap_layout is not None
    assert result.layout.interface_count == 0
    assert result.interface_current_observation == "ordinary_finite_volume_faces"
    assert result.bulk_trap_charge_storage_response_F_m2 is not None
    assert result.interface_sheet_charge_storage_response_F_m2 is None
    assert np.max(np.abs(result.positive_ion_admittance_faces_S_m2)) > 0.0
    assert result.certificate.maximum_ion_inventory_response_relative < 1.0e-8
    assert result.certificate.maximum_bulk_trap_balance_relative_error < 1.0e-3
    assert result.certificate.low_frequency_qss_relative_error < 3.0e-2
    assert result.certificate.high_frequency_frozen_relative_error < 3.0e-2
    _assert_current_decomposition(result)


@pytest.mark.parametrize(
    ("positive", "negative", "expected_species"),
    [
        (True, True, {"positive", "negative"}),
        (False, True, {"negative"}),
    ],
)
def test_dual_and_negative_only_ion_blocks_keep_independent_inventories(
    positive,
    negative,
    expected_species,
):
    stack = _with_mobile_ions(
        _bulk_defect_stack(),
        positive=positive,
        negative=negative,
    )
    grid = _bulk_grid(stack, 4)

    result = run_defect_ion_combined_impedance(
        grid,
        stack,
        BULK_FREQUENCIES_HZ,
    )

    assert result.certificate.certified
    assert result.negative_ion_admittance_faces_S_m2 is not None
    assert result.negative_ion_storage_response_F_m2 is not None
    assert bool(result.layout.ion_layout.positive_nodes) is positive
    assert result.layout.ion_layout.negative_nodes
    assert {
        item.species
        for item in result.certificate.frequency_window.ionic.ionic_timescales
    } == expected_species
    assert result.certificate.maximum_ion_inventory_response_relative < 1.0e-8
    if not positive:
        assert result.layout.positive_ion_slice.start == (
            result.layout.positive_ion_slice.stop
        )
        np.testing.assert_array_equal(
            result.positive_ion_admittance_faces_S_m2,
            np.zeros_like(result.positive_ion_admittance_faces_S_m2),
        )
    _assert_current_decomposition(result)


def test_interface_defect_and_ions_close_sheet_charge_and_four_leg_capture():
    stack = _with_mobile_ions(_contact_consistent_interface_stack())
    grid = _interface_grid(stack)

    result = run_defect_ion_combined_impedance(
        grid,
        stack,
        INTERFACE_FREQUENCIES_HZ,
    )

    assert result.certificate.certified
    assert result.certificate.capability == "interface_defect_plus_ions"
    assert result.layout.bulk_trap_layout is None
    assert result.layout.interface_count == 1
    assert result.interface_current_observation == ("symmetric_adjacent_physical_faces")
    assert result.interface_sheet_charge_storage_response_F_m2 is not None
    assert result.interface_occupancy_response_per_V is not None
    assert result.bulk_trap_charge_storage_response_F_m2 is None
    assert np.max(np.abs(result.interface_occupancy_response_per_V)) > 0.0
    assert result.dc_state.certificate.maximum_interface_residual < 1.0e-7
    assert result.dc_state.certificate.maximum_interface_gauss_residual < 1.0e-7
    assert result.certificate.maximum_interface_trap_balance_relative_error < 1.0e-3
    assert result.dc_state.certificate.contact_thermodynamics.certified
    _assert_current_decomposition(result)


def test_bulk_interface_defects_and_ions_close_one_triple_coupled_system():
    stack = _bulk_interface_ion_stack()
    grid = _interface_grid(stack)

    result = run_defect_ion_combined_impedance(
        grid,
        stack,
        BULK_FREQUENCIES_HZ,
    )

    assert result.certificate.certified
    assert result.certificate.capability == "bulk_interface_defect_plus_ions"
    assert result.layout.bulk_trap_layout is not None
    assert result.layout.interface_count == 1
    assert result.interface_current_observation == ("symmetric_adjacent_physical_faces")
    assert result.bulk_trap_charge_storage_response_F_m2 is not None
    assert result.interface_sheet_charge_storage_response_F_m2 is not None
    assert result.certificate.qss_embedding_relative_error < 1.0e-8
    assert result.certificate.maximum_all_face_admittance_spread < 5.0e-4
    assert result.certificate.maximum_bulk_trap_balance_relative_error < 1.0e-3
    assert result.certificate.maximum_interface_trap_balance_relative_error < 1.0e-3
    _assert_current_decomposition(result)


def test_combined_frequency_window_fails_closed():
    stack = _with_mobile_ions(_bulk_defect_stack())
    grid = _bulk_grid(stack, 4)
    frequencies = np.logspace(1.0, 2.0, 3)

    partial = run_defect_ion_combined_impedance(
        grid,
        stack,
        frequencies,
        require_certificate=False,
    )

    assert not partial.certificate.certified
    assert not partial.certificate.frequency_window.certified
    assert "combined_frequency_window_not_covered" in partial.certificate.reasons
    with pytest.raises(DefectIonCombinedCertificationError) as exc_info:
        run_defect_ion_combined_impedance(grid, stack, frequencies)
    assert exc_info.value.result.certificate.reasons == partial.certificate.reasons


def test_combined_capability_requires_both_a_defect_and_active_ions():
    defect_stack = _bulk_defect_stack()
    with pytest.raises(DefectIonCombinedError, match="active mobile-ion species"):
        run_defect_ion_combined_impedance(
            _bulk_grid(defect_stack, 4),
            defect_stack,
            BULK_FREQUENCIES_HZ,
        )

    ion_stack = _with_mobile_ions(defect_stack)
    params = ion_stack.layers[0].params
    assert params is not None
    no_defect_stack = replace(
        ion_stack,
        layers=(
            replace(
                ion_stack.layers[0],
                params=replace(
                    params,
                    defect_schema_version=None,
                    defect_model=EFFECTIVE_LIFETIME,
                    bulk_defects=(),
                ),
            ),
        ),
    )
    with pytest.raises(DefectIonCombinedError, match="explicit bulk or interface"):
        run_defect_ion_combined_impedance(
            _bulk_grid(no_defect_stack, 4),
            no_defect_stack,
            BULK_FREQUENCIES_HZ,
        )


def test_combined_capability_rejects_inconsistent_contact_thermodynamics():
    stack = _with_mobile_ions(_research_interface_stack())

    with pytest.raises(DefectIonCombinedError, match="contact thermodynamic"):
        run_defect_ion_combined_impedance(
            _interface_grid(stack),
            stack,
            INTERFACE_FREQUENCIES_HZ,
        )
