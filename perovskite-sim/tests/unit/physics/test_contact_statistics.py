"""Contact thermodynamics for opt-in bulk carrier statistics."""

from __future__ import annotations

from dataclasses import replace

import pytest

from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.models.device import DeviceStack, LayerSpec
from perovskite_sim.models.parameters import MaterialParams
from perovskite_sim.physics.contacts import build_semiconductor_contact_state
from perovskite_sim.physics.statistics import FERMI_DIRAC, MAXWELL_BOLTZMANN
from perovskite_sim.solver.mol import (
    BulkCarrierStatisticsCapabilityError,
    build_material_arrays,
)


def _silicon(
    *,
    acceptors: float = 0.0,
    donors: float = 0.0,
    statistics: str = MAXWELL_BOLTZMANN,
) -> MaterialParams:
    return MaterialParams(
        eps_r=11.7,
        mu_n=0.135,
        mu_p=0.048,
        D_ion=0.0,
        P_lim=1.0e30,
        P0=0.0,
        ni=1.0e16,
        tau_n=1.0e-6,
        tau_p=1.0e-6,
        n1=1.0e16,
        p1=1.0e16,
        B_rad=1.0e-20,
        C_n=2.8e-43,
        C_p=9.9e-44,
        alpha=0.0,
        N_A=acceptors,
        N_D=donors,
        chi=4.05,
        Eg=1.124,
        Nc300=2.8e25,
        Nv300=1.04e25,
        carrier_statistics=statistics,
    )


@pytest.mark.parametrize(
    ("acceptors", "donors"),
    ((0.0, 1.0e27), (8.0e26, 0.0)),
)
def test_degenerate_contact_state_closes_neutrality_and_changes_work_function(
    acceptors,
    donors,
):
    fd_params = _silicon(
        acceptors=acceptors,
        donors=donors,
        statistics=FERMI_DIRAC,
    )
    fd = build_semiconductor_contact_state(
        fd_params,
        temperature_K=300.0,
        use_temperature_scaling=True,
    )
    mb = build_semiconductor_contact_state(
        replace(fd_params, carrier_statistics=MAXWELL_BOLTZMANN),
        temperature_K=300.0,
        use_temperature_scaling=True,
    )

    assert fd.statistics == FERMI_DIRAC
    assert fd.neutrality.normalized_charge_residual < 2.0e-13
    assert abs(fd.work_function_eV - mb.work_function_eV) > 0.05
    if donors > acceptors:
        assert fd.electron_density_m3 == pytest.approx(donors, rel=1.0e-8)
        assert fd.reduced_electron_fermi_level > 1.0
    else:
        assert fd.hole_density_m3 == pytest.approx(acceptors, rel=1.0e-8)
        assert fd.reduced_hole_fermi_level > 1.0


def test_fd_device_work_function_uses_the_same_contact_neutrality_states():
    left = _silicon(acceptors=8.0e26, statistics=FERMI_DIRAC)
    right = _silicon(donors=1.0e27, statistics=FERMI_DIRAC)
    stack = DeviceStack(
        layers=(
            LayerSpec("p_plus", 1.0e-6, left, "HTL"),
            LayerSpec("n_plus", 1.0e-6, right, "ETL"),
        ),
        built_in_potential_mode="semiconductor_work_function",
    )
    left_state = build_semiconductor_contact_state(
        left,
        temperature_K=300.0,
        use_temperature_scaling=True,
    )
    right_state = build_semiconductor_contact_state(
        right,
        temperature_K=300.0,
        use_temperature_scaling=True,
    )

    assert stack.compute_semiconductor_V_bi() == pytest.approx(
        left_state.work_function_eV - right_state.work_function_eV,
        abs=2.0e-14,
    )


@pytest.mark.parametrize(
    "mode_fields",
    (
        {},
        {"built_in_potential_mode": "legacy_manual"},
        {
            "built_in_potential_mode": "metal_work_function",
            "work_function_left_eV": 5.0,
            "work_function_right_eV": 4.0,
        },
    ),
)
def test_fd_stack_rejects_nonphysical_contact_potential_modes(mode_fields):
    params = _silicon(donors=1.0e27, statistics=FERMI_DIRAC)
    with pytest.raises(ValueError, match="explicit built_in_potential_mode"):
        DeviceStack(
            layers=(LayerSpec("n_plus", 1.0e-6, params, "ETL"),),
            **mode_fields,
        )


def test_bulk_solver_fails_closed_until_fd_transport_is_enabled():
    params = _silicon(donors=1.0e27, statistics=FERMI_DIRAC)
    stack = DeviceStack(
        layers=(LayerSpec("n_plus", 1.0e-6, params, "ETL"),),
        built_in_potential_mode="semiconductor_work_function",
    )
    grid = multilayer_grid((Layer(1.0e-6, N=8),))

    with pytest.raises(
        BulkCarrierStatisticsCapabilityError,
        match="bulk Fermi-Dirac transport closure is not enabled",
    ):
        build_material_arrays(grid, stack)


@pytest.mark.parametrize(
    "updates",
    (
        {"Nc300": None},
        {"Nv300": 0.0},
        {"Eg": -1.0},
        {"N_D": -1.0},
    ),
)
def test_fd_material_requires_complete_physical_inputs(updates):
    base = _silicon(donors=1.0e27)
    with pytest.raises(ValueError, match="fermi_dirac carrier statistics"):
        replace(base, carrier_statistics=FERMI_DIRAC, **updates)
