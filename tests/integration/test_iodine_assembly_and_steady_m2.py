"""Integration tests for 1D iodine sparse assembly, steady state, and BDF (M2)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from solarlab.config.resolve_device import resolve_device
from solarlab.device.compile_iodine_1d import (
    IodineChemistryDeclaration,
    IodineLayerSpec,
    IodineRateParameterization,
    bitton_tessler_2023_table_s1_chemistry,
    compile_iodine_device_1d,
)
from solarlab.device.defaults import DefaultCatalog
from solarlab.device.inputs import DeviceInput
from solarlab.device.resolved import PreparedDevice
from solarlab.equations.assembly.iodine_1d import (
    AssembledConstrainedIodine1DProblem,
    AssembledIodine1DProblem,
)
from solarlab.equations.iodine_1d import IodineDomainMode
from solarlab.materials.resources import ResourceLibrary
from solarlab.materials.source import SourceDocument
from solarlab.solvers.nonlinear.iodine_1d import (
    solve_iodine_bdf_transient,
    solve_iodine_steady_state,
)
from solarlab.solvers.time.bdf import BDFOptions


ROOT = Path(__file__).resolve().parents[2]


def _bitton_tessler_prepared_device() -> PreparedDevice:
    """Construct the Bitton & Tessler (2023) SI Table S1 3-layer stack."""
    base_path = ROOT / "examples" / "research_1d" / "device.json"
    source = SourceDocument(str(base_path), base_path.read_bytes())
    doc = json.loads(source.content)
    catalog = {
        **doc["default_catalog"],
        "evidence": [[source.id, source.sha256]],
    }
    q_c = 1.602176634e-19
    kb_j_k = 1.380649e-23
    vt = (kb_j_k * 300.0) / q_c
    # Langevin B_rad = q * (mu_n + mu_p) / (eps_0 * eps_r) for eps_r=3, mu=1e-3 cm^2/Vs
    eps0 = 8.8541878128e-12
    mu_bl = 1.0e-3 * 1.0e-4  # 1e-7 m^2/(V s)
    b_langevin = q_c * (2.0 * mu_bl) / (eps0 * 3.0)

    ni_bl = float(np.exp(0.5 * (np.log(1.0e27) + np.log(1.0e27) - 2.5 / vt)))
    ni_pvk = float(np.exp(0.5 * (np.log(7.0e24) + np.log(2.5e24) - 1.6 / vt)))

    device_dict = {
        "schema_version": "solarlab.device-preparation.v1",
        "id": "bitton_tessler_2023_stack",
        "name": "Bitton & Tessler 2023 HBL/PVK/EBL stack",
        "source_format": "canonical",
        "settings": {
            "built_in_potential_mode": "semiconductor_work_function",
            "dos_band_potentials": True,
            "T": 300.0,
            "Phi": 2.5e21,
        },
        "layers": [
            {
                "id": "hbl",
                "name": "HBL",
                "role": "ETL",
                "thickness": 50.0e-9,
                "parameters": {
                    "chi": 4.3,
                    "Eg": 2.5,
                    "Nc300": 1.0e27,
                    "Nv300": 1.0e27,
                    "eps_r": 3.0,
                    "N_D": 2.0e22,
                    "N_A": 0.0,
                    "ni": ni_bl,
                    "n1": ni_bl,
                    "p1": ni_bl,
                    "tau_n": 1.0,
                    "tau_p": 1.0,
                    "B_rad": b_langevin,
                    "C_n": 0.0,
                    "C_p": 0.0,
                    "mu_n": mu_bl,
                    "mu_p": mu_bl,
                    "alpha": 0.0,
                },
            },
            {
                "id": "absorber",
                "name": "Perovskite",
                "role": "absorber",
                "thickness": 500.0e-9,
                "parameters": {
                    "chi": 4.3,
                    "Eg": 1.6,
                    "Nc300": 7.0e24,
                    "Nv300": 2.5e24,
                    "eps_r": 50.0,
                    "N_D": 0.0,
                    "N_A": 0.0,
                    "ni": ni_pvk,
                    "n1": ni_pvk,
                    "p1": ni_pvk,
                    "tau_n": 1.0e-5,
                    "tau_p": 1.0e-5,
                    "B_rad": 1.0e-10 * 1.0e-6,
                    "C_n": 0.0,
                    "C_p": 0.0,
                    "mu_n": 2.0e-4,
                    "mu_p": 2.0e-4,
                    "alpha": 1.0e7,
                },
            },
            {
                "id": "ebl",
                "name": "EBL",
                "role": "HTL",
                "thickness": 50.0e-9,
                "parameters": {
                    "chi": 3.4,
                    "Eg": 2.5,
                    "Nc300": 1.0e27,
                    "Nv300": 1.0e27,
                    "eps_r": 3.0,
                    "N_D": 0.0,
                    "N_A": 2.0e22,
                    "ni": ni_bl,
                    "n1": ni_bl,
                    "p1": ni_bl,
                    "tau_n": 1.0,
                    "tau_p": 1.0,
                    "B_rad": b_langevin,
                    "C_n": 0.0,
                    "C_p": 0.0,
                    "mu_n": mu_bl,
                    "mu_p": mu_bl,
                    "alpha": 0.0,
                },
            },
        ],
    }
    return resolve_device(
        DeviceInput.model_validate(device_dict),
        DefaultCatalog.from_mapping(catalog),
        ResourceLibrary(),
        sources=(source,),
    )


@pytest.mark.parametrize(
    ("domain_mode", "use_partition"),
    [
        ("absorber_blocking", False),
        ("full_stack_permeable", False),
        ("full_stack_permeable", True),
    ],
)
def test_assembled_iodine_sparse_jacobians_match_central_finite_differences(
    domain_mode: IodineDomainMode,
    use_partition: bool,
) -> None:
    prepared = _bitton_tessler_prepared_device()
    chem = bitton_tessler_2023_table_s1_chemistry(domain_mode=domain_mode)
    if use_partition:
        hbl_s, pvk_s, ebl_s = chem.layers
        chem = IodineChemistryDeclaration(
            layers=(
                IodineLayerSpec(
                    **{
                        **{f: getattr(hbl_s, f) for f in hbl_s.__dataclass_fields__},
                        "partition_minus": 0.05,
                        "partition_atom": 0.2,
                        "partition_mol": 0.5,
                    }
                ),
                pvk_s,
                IodineLayerSpec(
                    **{
                        **{f: getattr(ebl_s, f) for f in ebl_s.__dataclass_fields__},
                        "partition_minus": 0.05,
                        "partition_atom": 0.2,
                        "partition_mol": 0.5,
                    }
                ),
            ),
            domain_mode="full_stack_permeable",
            initialize_equilibrium_split=True,
        )

    device = compile_iodine_device_1d(prepared, chem, intervals_per_layer=(4, 6, 4))
    u0 = np.array(device.initial_coordinates(voltage_V=0.45), dtype=np.float64, copy=True)
    rng = np.random.default_rng(42)
    u0 += 0.05 * rng.standard_normal(u0.size)

    unconstrained = AssembledIodine1DProblem(device, voltage_V=0.45, generation_scale=0.8)
    constrained = AssembledConstrainedIodine1DProblem(
        device, voltage_V=0.45, generation_scale=0.8, inventory_node=2
    )

    for prob in (unconstrained, constrained):
        j_exact = prob.jacobian(u0).toarray()
        f0 = prob.residual(u0)
        eps = np.finfo(np.float64).eps
        j_fd = np.zeros_like(j_exact)
        for col in range(u0.size):
            h = 1.0e-6 * max(abs(float(u0[col])), 1.0)
            up, um = u0.copy(), u0.copy()
            up[col] += h
            um[col] -= h
            j_fd[:, col] = (prob.residual(up) - prob.residual(um)) / (2.0 * h)
            row_round = 128.0 * eps * (np.abs(f0) + np.max(np.abs(j_exact), axis=1)) / h
            assert np.all(
                np.abs(j_exact[:, col] - j_fd[:, col])
                <= 1.0e-6 * np.abs(j_fd[:, col]) + row_round + 1.0e-12
            )

    # Storage tangent S_u check
    s_tan = unconstrained.storage_tangent(u0).toarray()
    s0 = unconstrained.storage(u0)
    for col in range(u0.size):
        h = 1.0e-6 * max(abs(float(u0[col])), 1.0)
        up, um = u0.copy(), u0.copy()
        up[col] += h
        um[col] -= h
        fd_s = (unconstrained.storage(up) - unconstrained.storage(um)) / (2.0 * h)
        s_round = 64.0 * eps * np.abs(s0) / h
        assert np.all(
            np.abs(s_tan[:, col] - fd_s) <= 1.0e-7 * np.abs(fd_s) + s_round + 1.0e-18
        )


def test_unreactive_limit_preserves_each_species_and_boltzmann_iodide_profile() -> None:
    prepared = _bitton_tessler_prepared_device()
    zero_chem = IodineChemistryDeclaration(
        layers=(
            IodineLayerSpec.from_cgs(
                "absorber",
                D_minus_cm2_s=1.0e-8,
                D_atom_cm2_s=1.0e-8,
                D_mol_cm2_s=1.0e-9,
                initial_iodide_cm3=1.0e18,
                initial_atom_cm3=1.0e12,
                initial_mol_cm3=1.0e10,
                compensating_donor_cm3=1.0e18,
                k_a_cm3_s=0.0,
                k_b_s_1=0.0,
                k_c_cm3_s=0.0,
                k_d_cm3_s=0.0,
                k_e_cm3_s=0.0,
            ),
        ),
        domain_mode="absorber_blocking",
        thermodynamic_mode="wegscheider_detailed_balance",
    )
    device = compile_iodine_device_1d(
        prepared, zero_chem, intervals_per_layer=(8, 24, 8)
    )
    assert device.is_unreactive_limit
    res = solve_iodine_steady_state(device, voltage_V=0.5, generation_scale=1.0)
    assert res.converged and res.state is not None

    ref_inv = device.reference_species_inventories_m2
    for sp in ("iodide", "iodine_atom", "iodine_molecule"):
        assert res.state.species_inventories_m2[sp] == pytest.approx(
            ref_inv[sp], rel=1.0e-12
        )
    # Neutral species have zero drift and zero reaction -> strictly flat profile!
    c0 = res.state.iodine_atom_density_m3
    c2 = res.state.iodine_molecule_density_m3
    np.testing.assert_allclose(c0, c0[0], rtol=1.0e-10)
    np.testing.assert_allclose(c2, c2[0], rtol=1.0e-10)

    # Zero-flux I^- satisfies c_-(x) / c_-(0) = exp((phi(x) - phi(0)) / V_T)
    phi_abs = res.state.electronic_state.potential_V[device.chem_node_indices]
    vt = device.electronic_component.thermal_voltage_V
    cm = res.state.iodide_density_m3
    expected_ratio = np.exp((phi_abs - phi_abs[0]) / vt)
    np.testing.assert_allclose(cm / cm[0], expected_ratio, rtol=1.0e-9)


@pytest.mark.parametrize(
    ("domain_mode", "rate_mode"),
    [
        ("absorber_blocking", "explicit_rates"),
        ("full_stack_permeable", "explicit_rates"),
        ("absorber_blocking", "smoluchowski_capture_radius"),
    ],
)
def test_bitton_tessler_steady_and_transient_bdf_conservation_and_fig_s2_trends(
    domain_mode: IodineDomainMode,
    rate_mode: IodineRateParameterization,
) -> None:
    prepared = _bitton_tessler_prepared_device()
    chem = bitton_tessler_2023_table_s1_chemistry(
        domain_mode=domain_mode,
        rate_mode=rate_mode,
        perovskite_ka_cm3_s=1.0e-11,
    )
    device = compile_iodine_device_1d(prepared, chem, intervals_per_layer=(8, 30, 8))

    # 1. Dark equilibrium (0 V, Dark): exact iodine inventory & tiny dark current
    dark_eq = solve_iodine_steady_state(device, voltage_V=0.0, generation_scale=0.0)
    assert dark_eq.converged and dark_eq.state is not None
    assert dark_eq.state.total_iodine_inventory_m2 == pytest.approx(
        device.reference_iodine_inventory_m2, rel=1.0e-12
    )
    assert abs(dark_eq.current_density_A_m2) < 1.0e-7

    # 2. Illuminated forward bias (1 Sun, 0.9 V -- Bitton & Tessler Fig. S2):
    # Photo-holes oxidize I^- -> I^0 -> I_2, elevating neutral I^0 and I_2 by orders
    # of magnitude above dark equilibrium while conserving total iodine to 1e-12!
    illum_steady = solve_iodine_steady_state(
        device, voltage_V=0.9, generation_scale=1.0
    )
    assert illum_steady.converged and illum_steady.state is not None
    assert illum_steady.state.total_iodine_inventory_m2 == pytest.approx(
        device.reference_iodine_inventory_m2, rel=1.0e-12
    )
    assert (
        illum_steady.state.species_inventories_m2["iodine_atom"]
        > 1.0e4 * dark_eq.state.species_inventories_m2["iodine_atom"]
    )
    assert (
        illum_steady.state.species_inventories_m2["iodine_molecule"]
        > 50.0 * dark_eq.state.species_inventories_m2["iodine_molecule"]
    )

    # 3. Transient BDF1/2 from dark equilibrium into (1 Sun, 0.9 V) stress:
    # Conserves elemental iodine at every step and converges to steady state!
    bdf_opts = BDFOptions(
        initial_step_s=1.0e-6,
        min_step_s=1.0e-14,
        max_step_s=200.0,
        relative_tolerance=1.0e-3,
        absolute_tolerance=1.0e11,
        max_steps=600,
    )
    bdf_res, ledger = solve_iodine_bdf_transient(
        device,
        dark_eq.state.u,
        (0.0, 1000.0),
        voltage_fn=lambda _t: 0.9,
        generation_fn=lambda _t: 1.0,
        options=bdf_opts,
    )
    assert bdf_res.completed and len(ledger) > 0
    inv_ref = device.reference_iodine_inventory_m2
    for step_acc in ledger:
        assert abs(step_acc.in_device_iodine_m2 - inv_ref) / inv_ref <= 1.0e-11
        assert abs(step_acc.step_conservation_residual_m2) / inv_ref <= 5.0e-11

    final_state = device.state(bdf_res.accepted_coordinates[-1], voltage_V=0.9)
    assert final_state.species_inventories_m2["iodine_atom"] == pytest.approx(
        illum_steady.state.species_inventories_m2["iodine_atom"], rel=2.0e-2
    )
