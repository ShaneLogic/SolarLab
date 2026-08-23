"""Restricted high-doping p-n equilibrium transport closure tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.models.device import DeviceStack, InterfaceDefect, LayerSpec
from perovskite_sim.models.parameters import MaterialParams
from perovskite_sim.physics.contacts import build_semiconductor_contact_state
from perovskite_sim.physics.statistics import FERMI_DIRAC, MAXWELL_BOLTZMANN
from perovskite_sim.solver.degenerate_equilibrium import (
    solve_degenerate_pn_equilibrium,
)
from perovskite_sim.solver.mol import (
    DEGENERATE_TRANSPORT_RESEARCH_RECOMBINATION_OFF,
    BulkCarrierStatisticsCapabilityError,
    build_material_arrays,
)


LAYER_THICKNESS_M = 100.0e-9


def _silicon(
    *,
    acceptors: float = 0.0,
    donors: float = 0.0,
    statistics: str = FERMI_DIRAC,
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
        B_rad=0.0,
        C_n=0.0,
        C_p=0.0,
        alpha=0.0,
        N_A=acceptors,
        N_D=donors,
        chi=4.05,
        Eg=1.124,
        Nc300=2.8e25,
        Nv300=1.04e25,
        carrier_statistics=statistics,
    )


def _stack(
    *,
    statistics: str = FERMI_DIRAC,
    left: MaterialParams | None = None,
    right: MaterialParams | None = None,
    **updates,
) -> DeviceStack:
    p_type = left or _silicon(
        acceptors=3.0e25,
        statistics=statistics,
    )
    n_type = right or _silicon(
        donors=3.0e25,
        statistics=statistics,
    )
    values = {
        "layers": (
            LayerSpec("p_plus", LAYER_THICKNESS_M, p_type, "absorber"),
            LayerSpec("n_plus", LAYER_THICKNESS_M, n_type, "absorber"),
        ),
        "Phi": 0.0,
        "interfaces": ((0.0, 0.0),),
        "built_in_potential_mode": "semiconductor_work_function",
        "mode": "full",
    }
    values.update(updates)
    return DeviceStack(**values)


def _grid(intervals_per_layer: int) -> np.ndarray:
    return multilayer_grid(
        (
            Layer(LAYER_THICKNESS_M, intervals_per_layer),
            Layer(LAYER_THICKNESS_M, intervals_per_layer),
        ),
        alpha=3.0,
    )


def test_research_material_uses_the_same_fd_contact_reservoirs():
    stack = _stack()
    material = build_material_arrays(
        _grid(20),
        stack,
        carrier_statistics_transport=(
            DEGENERATE_TRANSPORT_RESEARCH_RECOMBINATION_OFF
        ),
    )
    left = build_semiconductor_contact_state(
        stack.layers[0].params,
        temperature_K=300.0,
        use_temperature_scaling=True,
    )
    right = build_semiconductor_contact_state(
        stack.layers[-1].params,
        temperature_K=300.0,
        use_temperature_scaling=True,
    )

    assert material.carrier_statistics == FERMI_DIRAC
    assert material.degenerate_recombination_model == "off"
    assert material.n_L == left.electron_density_m3
    assert material.p_L == left.hole_density_m3
    assert material.n_R == right.electron_density_m3
    assert material.p_R == right.hole_density_m3


def test_default_material_path_still_rejects_fd_transport():
    with pytest.raises(
        BulkCarrierStatisticsCapabilityError,
        match="bulk Fermi-Dirac transport closure is not enabled",
    ):
        build_material_arrays(_grid(10), _stack())


@pytest.mark.parametrize(
    ("stack", "message"),
    (
        (
            _stack(right=replace(_silicon(donors=3.0e25), eps_r=12.0)),
            "non-homojunction fields",
        ),
        (_stack(interfaces=((1.0, 0.0),)), "interface recombination"),
        (
            _stack(
                interface_defects=(InterfaceDefect(E_t_eV=0.5),),
            ),
            "interface recombination",
        ),
        (_stack(Phi=1.0), "optical generation"),
        (_stack(band_grading=True), "band grading"),
        (
            _stack(
                left=replace(
                    _silicon(acceptors=3.0e25),
                    N_A_bulk=1.0e24,
                    doping_profile_shape="gaussian",
                    doping_decay_length=10.0e-9,
                )
            ),
            "spatial doping profiles",
        ),
        (_stack(flat_band_contacts=True), "calibrated contact floors"),
        (_stack(S_n_left=1.0e5), "selective contacts"),
        (
            _stack(
                left=replace(
                    _silicon(acceptors=3.0e25),
                    D_ion=1.0e-16,
                    P0=1.0e24,
                )
            ),
            "mobile ions",
        ),
        (
            _stack(
                left=replace(
                    _silicon(acceptors=3.0e25),
                    v_sat_n=1.0e5,
                )
            ),
            "field-dependent mobility",
        ),
        (
            _stack(interface_plane_projection=True),
            "advanced interface closure",
        ),
    ),
)
def test_research_transport_rejects_uncertified_topologies(stack, message):
    with pytest.raises(BulkCarrierStatisticsCapabilityError, match=message):
        build_material_arrays(
            _grid(10),
            stack,
            carrier_statistics_transport=(
                DEGENERATE_TRANSPORT_RESEARCH_RECOMBINATION_OFF
            ),
        )


def test_research_transport_rejects_behavior_environment_override(monkeypatch):
    monkeypatch.setenv("SOLARLAB_BAND_GRADING", "1")
    with pytest.raises(
        BulkCarrierStatisticsCapabilityError,
        match="environment overrides=SOLARLAB_BAND_GRADING",
    ):
        build_material_arrays(
            _grid(10),
            _stack(),
            carrier_statistics_transport=(
                DEGENERATE_TRANSPORT_RESEARCH_RECOMBINATION_OFF
            ),
        )


def test_mb_research_transport_is_exactly_the_classical_statistics_limit():
    stack = _stack(statistics=MAXWELL_BOLTZMANN)
    result = solve_degenerate_pn_equilibrium(_grid(40), stack)

    assert result.maximum_normalized_poisson_residual < 1.0e-9
    assert result.maximum_relative_face_current < 1.0e-12
    assert result.maximum_normalized_carrier_rate < 1.0e-12


@pytest.mark.parametrize("poisson_tolerance", (1.0e-8, 1.0e-10, 1.0e-12))
def test_high_doping_fd_pn_equilibrium_closes_across_grid_and_tolerance(
    poisson_tolerance,
):
    stack = _stack()
    results = [
        solve_degenerate_pn_equilibrium(
            _grid(intervals),
            stack,
            poisson_tolerance=poisson_tolerance,
        )
        for intervals in (40, 80, 160)
    ]

    for result in results:
        assert np.all(np.isfinite(result.state))
        assert np.all(result.electron_density_m3 > 0.0)
        assert np.all(result.hole_density_m3 > 0.0)
        assert result.maximum_normalized_poisson_residual <= poisson_tolerance
        assert result.maximum_relative_face_current < 1.0e-12
        assert result.maximum_normalized_carrier_rate < 1.0e-12
        assert result.depletion_width_relative_error < 0.05
        assert result.peak_field_relative_error < 0.05

    assert results[-1].charge_balance_relative_error < 0.005
    assert (
        results[-1].charge_balance_relative_error
        < results[0].charge_balance_relative_error
    )
    assert abs(
        results[-1].depletion_width_m - results[-2].depletion_width_m
    ) / results[-1].depletion_width_m < 0.002
