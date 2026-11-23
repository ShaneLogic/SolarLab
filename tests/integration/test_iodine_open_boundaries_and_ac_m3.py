"""Integration tests for open escape, reactive contacts, and AC impedance (M3)."""

from __future__ import annotations

import numpy as np
import pytest

from solarlab.contracts import ContractError
from solarlab.equations.dc_1d import CapabilityError
from solarlab.device.compile_iodine_1d import (
    IodineBoundaryDeclaration,
    IodineChemistryDeclaration,
    bitton_tessler_2023_table_s1_chemistry,
    compile_iodine_device_1d,
)
from solarlab.solvers.linear.iodine_impedance import solve_iodine_small_signal_1d
from solarlab.solvers.nonlinear.iodine_1d import (
    solve_frozen_chemistry_electronic_1d,
    solve_iodine_bdf_transient,
    solve_iodine_steady_state,
)
from solarlab.solvers.time.bdf import BDFOptions
from tests.integration.test_iodine_assembly_and_steady_m2 import (
    _bitton_tessler_prepared_device,
)


def test_open_i2_escape_vs_closed_dark_recovery_and_steady_refusal() -> None:
    prepared = _bitton_tessler_prepared_device()
    closed_chem = bitton_tessler_2023_table_s1_chemistry(
        domain_mode="absorber_blocking",
        rate_mode="smoluchowski_capture_radius",
        perovskite_ka_cm3_s=5.0e-11,
        right_escape_m_s=0.0,
    )
    open_chem = bitton_tessler_2023_table_s1_chemistry(
        domain_mode="absorber_blocking",
        rate_mode="smoluchowski_capture_radius",
        perovskite_ka_cm3_s=5.0e-11,
        right_escape_m_s=5.0e-2,
    )
    closed_dev = compile_iodine_device_1d(
        prepared, closed_chem, intervals_per_layer=(6, 20, 6)
    )
    open_dev = compile_iodine_device_1d(
        prepared, open_chem, intervals_per_layer=(6, 20, 6)
    )

    # Steady-state solver must refuse open outgassing boundary conditions
    with pytest.raises(CapabilityError):
        solve_iodine_steady_state(open_dev, voltage_V=0.0, generation_scale=0.0)

    dark_eq = solve_iodine_steady_state(closed_dev, voltage_V=0.0, generation_scale=0.0)
    assert dark_eq.converged and dark_eq.state is not None
    inv_0 = closed_dev.reference_iodine_inventory_m2

    bdf_stress_opts = BDFOptions(
        initial_step_s=1.0e-6,
        min_step_s=1.0e-14,
        max_step_s=20.0,
        relative_tolerance=1.0e-3,
        absolute_tolerance=1.0e11,
        max_steps=300,
    )
    bdf_rec_opts = BDFOptions(
        initial_step_s=1.0e-6,
        min_step_s=1.0e-14,
        max_step_s=50.0,
        relative_tolerance=1.0e-3,
        absolute_tolerance=1.0e11,
        max_steps=300,
    )

    # 1. Closed device: 60 s light stress (0.9 V, 1 Sun) -> 200 s dark recovery (0 V)
    res_c_stress, led_c_stress = solve_iodine_bdf_transient(
        closed_dev,
        dark_eq.state.u,
        (0.0, 60.0),
        voltage_fn=lambda _t: 0.9,
        generation_fn=lambda _t: 1.0,
        options=bdf_stress_opts,
    )
    assert res_c_stress.completed
    stressed_c = closed_dev.state(res_c_stress.accepted_coordinates[-1], voltage_V=0.9)

    res_c_rec, led_c_rec = solve_iodine_bdf_transient(
        closed_dev,
        stressed_c.u,
        (60.0, 260.0),
        voltage_fn=lambda _t: 0.0,
        generation_fn=lambda _t: 0.0,
        options=bdf_rec_opts,
    )
    assert res_c_rec.completed
    recovered_c = closed_dev.state(res_c_rec.accepted_coordinates[-1], voltage_V=0.0)
    assert recovered_c.total_iodine_inventory_m2 == pytest.approx(inv_0, rel=5.0e-10)
    # Dark recovery reduces photogenerated neutral I^0 and I_2 by orders of magnitude
    assert (
        recovered_c.species_inventories_m2["iodine_atom"]
        < 1.0e-2 * stressed_c.species_inventories_m2["iodine_atom"]
    )
    assert sum(s.step_escaped_iodine_atoms_m2 for s in led_c_stress + led_c_rec) == 0.0

    # 2. Open device: volatile I_2 outgassing permanently depletes elemental iodine
    # while BDF1/2 step ledger satisfies (in_device + cum_escaped == inv_0) identically!
    res_o_stress, led_o_stress = solve_iodine_bdf_transient(
        open_dev,
        dark_eq.state.u,
        (0.0, 60.0),
        voltage_fn=lambda _t: 0.9,
        generation_fn=lambda _t: 1.0,
        options=bdf_stress_opts,
    )
    assert res_o_stress.completed and len(led_o_stress) > 0
    cum_escaped = 0.0
    for acc in led_o_stress:
        cum_escaped += acc.step_escaped_iodine_atoms_m2
        assert abs(acc.step_conservation_residual_m2) / inv_0 <= 1.0e-10
        assert abs((acc.in_device_iodine_m2 + cum_escaped) - inv_0) / inv_0 <= 1.0e-10
    assert cum_escaped > 0.0
    final_open_inv = led_o_stress[-1].in_device_iodine_m2
    assert final_open_inv < inv_0


def test_charge_conserving_contact_oxidation_modes_and_contract_refusals() -> None:
    prepared = _bitton_tessler_prepared_device()
    base = bitton_tessler_2023_table_s1_chemistry(
        domain_mode="full_stack_permeable",
        rate_mode="smoluchowski_capture_radius",
    )

    # Non-conserving or negative boundary rates must be rejected by ContractError
    with pytest.raises(ContractError):
        compile_iodine_device_1d(
            prepared,
            IodineChemistryDeclaration(
                layers=base.layers,
                domain_mode="full_stack_permeable",
                right_boundary=IodineBoundaryDeclaration(mol_escape_velocity_m_s=-1.0),
            ),
            intervals_per_layer=(4, 10, 4),
        )
    with pytest.raises(ContractError):
        compile_iodine_device_1d(
            prepared,
            IodineChemistryDeclaration(
                layers=base.layers,
                domain_mode="full_stack_permeable",
                right_boundary=IodineBoundaryDeclaration(
                    contact_reaction_mode="none",
                    contact_oxidation_rate_m4_s=1.0e-25,
                ),
            ),
            intervals_per_layer=(4, 10, 4),
        )

    bdf_opts = BDFOptions(
        initial_step_s=1.0e-6,
        min_step_s=1.0e-14,
        max_step_s=5.0,
        relative_tolerance=1.0e-3,
        absolute_tolerance=1.0e11,
        max_steps=200,
    )

    # Mode A: hole_oxidation_to_I0 conserves in-device iodine AND Maxwell current!
    chem_rev_contact = IodineChemistryDeclaration(
        layers=base.layers,
        domain_mode="full_stack_permeable",
        right_boundary=IodineBoundaryDeclaration(
            contact_reaction_mode="hole_oxidation_to_I0",
            contact_oxidation_rate_m4_s=1.0e-24,
        ),
        initialize_equilibrium_split=True,
    )
    dev_rev = compile_iodine_device_1d(
        prepared, chem_rev_contact, intervals_per_layer=(5, 14, 5)
    )
    u0_rev = solve_iodine_steady_state(dev_rev, voltage_V=0.0, generation_scale=0.0)
    assert u0_rev.converged and u0_rev.state is not None
    res_rev, led_rev = solve_iodine_bdf_transient(
        dev_rev,
        u0_rev.state.u,
        (0.0, 10.0),
        voltage_fn=lambda _t: 0.95,
        generation_fn=lambda _t: 1.0,
        options=bdf_opts,
    )
    assert res_rev.completed
    inv_ref = dev_rev.reference_iodine_inventory_m2
    for acc in led_rev:
        assert abs(acc.in_device_iodine_m2 - inv_ref) / inv_ref <= 1.0e-10
        assert acc.step_immobilized_iodine_atoms_m2 == 0.0

    # Mode B: immobilized_product_sheet tracks irreversible contact corrosion sheet!
    chem_immob = IodineChemistryDeclaration(
        layers=base.layers,
        domain_mode="full_stack_permeable",
        right_boundary=IodineBoundaryDeclaration(
            contact_reaction_mode="immobilized_product_sheet",
            contact_oxidation_rate_m4_s=1.0e-22,
        ),
        initialize_equilibrium_split=True,
    )
    dev_immob = compile_iodine_device_1d(
        prepared, chem_immob, intervals_per_layer=(5, 14, 5)
    )
    res_im, led_im = solve_iodine_bdf_transient(
        dev_immob,
        u0_rev.state.u,
        (0.0, 10.0),
        voltage_fn=lambda _t: 0.95,
        generation_fn=lambda _t: 1.0,
        options=bdf_opts,
    )
    assert res_im.completed and len(led_im) > 0
    cum_immob = 0.0
    for acc in led_im:
        cum_immob += acc.step_immobilized_iodine_atoms_m2
        assert abs((acc.in_device_iodine_m2 + cum_immob) - inv_ref) / inv_ref <= 1.0e-10
    assert cum_immob > 0.0


def test_iodine_small_signal_ac_impedance_limits_and_dc_slope_consistency() -> None:
    prepared = _bitton_tessler_prepared_device()
    chem = bitton_tessler_2023_table_s1_chemistry(
        domain_mode="absorber_blocking",
        perovskite_ka_cm3_s=1.0e-11,
    )
    device = compile_iodine_device_1d(prepared, chem, intervals_per_layer=(6, 20, 6))

    v_op = 0.75
    g_op = 1.0
    steady = solve_iodine_steady_state(device, voltage_V=v_op, generation_scale=g_op)
    assert steady.converged and steady.state is not None

    freqs = np.r_[0.0, np.logspace(-3, 8, 23)]
    ac_mobile = solve_iodine_small_signal_1d(
        device,
        steady.state,
        freqs,
        generation_scale=g_op,
        chem_mode="mobile",
    )
    u_frozen_op, _ = solve_frozen_chemistry_electronic_1d(
        device, steady.state.u, voltage_V=v_op, generation_scale=g_op
    )
    frozen_op_state = device.state(u_frozen_op, voltage_V=v_op)
    ac_frozen = solve_iodine_small_signal_1d(
        device,
        frozen_op_state,
        freqs,
        generation_scale=g_op,
        chem_mode="frozen",
    )

    assert ac_mobile.scientific_qualification is False
    assert ac_mobile.max_backward_error <= 1.0e-11
    assert ac_frozen.max_backward_error <= 1.0e-11
    scale_m = float(np.max(np.abs(ac_mobile.admittance_S_m2)))
    scale_f = float(np.max(np.abs(ac_frozen.admittance_S_m2)))
    assert ac_mobile.max_maxwell_spread_A_m2_V <= 1.0e-8 + 1.0e-10 * scale_m
    assert ac_frozen.max_maxwell_spread_A_m2_V <= 1.0e-8 + 1.0e-10 * scale_f

    # 1. High-frequency geometric capacitance limit (100 MHz):
    # Series dielectric capacitance C_geom = 1 / sum(1 / C_face)
    inv_c = float(np.sum(1.0 / device.electronic_component.poisson.face_capacitance_F_m2))
    c_geom = 1.0 / inv_c
    assert ac_mobile.apparent_capacitance_F_m2[-1] == pytest.approx(c_geom, rel=2.0e-2)
    assert ac_frozen.apparent_capacitance_F_m2[-1] == pytest.approx(c_geom, rel=2.0e-2)
    assert ac_mobile.impedance_Ohm_m2[-1] == pytest.approx(
        ac_frozen.impedance_Ohm_m2[-1], rel=1.0e-2
    )

    # 2. Exact 0 Hz differential conductance vs central finite-difference DC slope!
    dv = 1.0e-4
    sp = solve_iodine_steady_state(device, voltage_V=v_op + dv, generation_scale=g_op)
    sm = solve_iodine_steady_state(device, voltage_V=v_op - dv, generation_scale=g_op)
    assert sp.converged and sm.converged
    pol = device.electronic_component.voltage_polarity
    # Input current into positive electrode is polarity * J_mean
    g_dc_mobile_fd = pol * (sp.current_density_A_m2 - sm.current_density_A_m2) / (2.0 * dv)
    assert ac_mobile.admittance_S_m2[0].real == pytest.approx(g_dc_mobile_fd, rel=5.0e-4)
    assert abs(ac_mobile.admittance_S_m2[0].imag) <= 1.0e-12

    _, jp_f = solve_frozen_chemistry_electronic_1d(
        device, steady.state.u, voltage_V=v_op + dv, generation_scale=g_op
    )
    _, jm_f = solve_frozen_chemistry_electronic_1d(
        device, steady.state.u, voltage_V=v_op - dv, generation_scale=g_op
    )
    g_dc_frozen_fd = pol * (jp_f - jm_f) / (2.0 * dv)
    assert ac_frozen.admittance_S_m2[0].real == pytest.approx(g_dc_frozen_fd, rel=5.0e-4)

    # 3. Low-frequency (1 mHz) mobile chemical/ionic arc differs from frozen response!
    assert abs(
        ac_mobile.apparent_capacitance_F_m2[1] - ac_frozen.apparent_capacitance_F_m2[1]
    ) > 10.0 * c_geom
    for sp_name in ("iodide", "iodine_atom", "iodine_molecule"):
        assert ac_frozen.responses[1].species_inventory_perturbation_m2[sp_name] == 0.0j
    # Closed-domain mobile AC conserves elemental iodine phasor delta(M_- + M_0 + 2 M_2) == 0
    inv_p = ac_mobile.responses[1].species_inventory_perturbation_m2
    delta_elem = (
        inv_p["iodide"] + inv_p["iodine_atom"] + 2.0 * inv_p["iodine_molecule"]
    )
    assert abs(delta_elem) / device.reference_iodine_inventory_m2 <= 1.0e-8
