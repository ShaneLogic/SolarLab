from __future__ import annotations

import numpy as np

from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import DeviceStack, InterfaceDefect, LayerSpec
from perovskite_sim.models.parameters import MaterialParams
from perovskite_sim.solver.tolerances import ComponentwiseAtol
from perovskite_sim.twod.experiments.jv_sweep_2d import (
    build_jv_2d_execution_protocol,
    run_jv_sweep_2d,
)
from perovskite_sim.twod.microstructure import (
    GrainBoundary,
    Microstructure,
)


def _combined_stack() -> DeviceStack:
    material = MaterialParams(
        eps_r=10.0,
        mu_n=1.0e-3,
        mu_p=1.0e-3,
        D_ion=1.0e-16,
        P_lim=1.0e24,
        P0=1.0e22,
        ni=1.0e12,
        tau_n=1.0e30,
        tau_p=1.0e30,
        n1=1.0e12,
        p1=1.0e12,
        B_rad=0.0,
        C_n=0.0,
        C_p=0.0,
        alpha=0.0,
        N_A=0.0,
        N_D=0.0,
        chi=4.0,
        Eg=1.5,
        Nc300=1.0e25,
        Nv300=1.0e25,
    )
    return DeviceStack(
        layers=(
            LayerSpec("left", 1.0e-7, material, role="absorber"),
            LayerSpec("right", 1.0e-7, material, role="ETL"),
        ),
        interfaces=((0.03, 0.05),),
        interface_defects=(InterfaceDefect(E_t_eV=0.5),),
        interface_two_sided=True,
        V_bi=0.0,
        Phi=0.0,
        mode="full",
    )


def test_real_strict_mobile_ion_jv_returns_complete_point_evidence():
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    common = {
        "lateral_length": 1.0e-7,
        "Nx": 2,
        "V_max": 0.01,
        "V_step": 0.01,
        "illuminated": False,
        "lateral_bc": "neumann",
        "Ny_per_layer": 2,
        "settle_t": 1.0e-12,
        "save_snapshots": True,
        "ion_dynamics": "single_mobile",
        "atol": ComponentwiseAtol(),
        "max_nfev_per_solve": 20_000,
        "max_bisect": 2,
        "ion_inventory_rtol": 1.0e-10,
    }
    protocol = build_jv_2d_execution_protocol(
        stack,
        Microstructure(),
        **common,
    )

    result = run_jv_sweep_2d(
        stack,
        Microstructure(),
        **common,
        jv_2d_protocol=protocol,
        protocol_mode="research_strict",
    )

    np.testing.assert_array_equal(result.V, [0.0, 0.01])
    assert np.all(np.isfinite(result.J))
    assert len(result.snapshots) == 2
    assert len(result.current_components) == 2
    assert len(result.ion_diagnostics) == 2
    assert all(snapshot.P_ion is not None for snapshot in result.snapshots)
    assert all(report.passed for report in result.ion_diagnostics)
    assert max(
        report.max_relative_face_spread for report in result.current_components
    ) < 1.0e-10
    assert result.protocol == protocol


def test_real_combined_gb_interface_mobile_jv_returns_each_evidence_axis():
    stack = _combined_stack()
    microstructure = Microstructure(
        (
            GrainBoundary(
                x_position=5.0e-8,
                width=1.0e-8,
                tau_n=1.0e-9,
                tau_p=1.0e-9,
            ),
        )
    )
    common = {
        "lateral_length": 1.0e-7,
        "Nx": 2,
        "V_max": 0.01,
        "V_step": 0.01,
        "illuminated": False,
        "lateral_bc": "neumann",
        "Ny_per_layer": 2,
        "settle_t": 1.0e-12,
        "save_snapshots": True,
        "ion_dynamics": "single_mobile",
        "interface_srh": "two_sided_cross_node",
        "atol": ComponentwiseAtol(),
        "max_nfev_per_solve": 20_000,
        "max_bisect": 2,
        "ion_inventory_rtol": 1.0e-10,
    }
    protocol = build_jv_2d_execution_protocol(
        stack,
        microstructure,
        **common,
    )

    result = run_jv_sweep_2d(
        stack,
        microstructure,
        **common,
        jv_2d_protocol=protocol,
        protocol_mode="research_strict",
    )

    assert len(result.current_components) == 2
    assert len(result.ion_diagnostics) == 2
    assert len(result.interface_srh_diagnostics) == 2
    assert len(result.protocol.grain_boundaries) == 1
    assert result.protocol.interface_srh == "two_sided_cross_node"
    assert all(
        np.all(np.isfinite(report.total_surface_rate_m2_s))
        for report in result.interface_srh_diagnostics
    )
