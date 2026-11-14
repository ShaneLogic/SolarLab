"""Live numerical parity tests (T4) for photon recycling & bounded WKB tunnelling (P04-12).

Verifies end-to-end numerical and structural parity between new-core ``solarlab`` and
legacy ``perovskite_sim`` across:
1. Optical escape probability primitives (``wavelength_at_gap``, ``compute_p_esc``,
   ``compute_p_esc_for_absorber``) and static thermionic-field emission ``tfe_gamma``.
2. Nonlocal radiative reabsorption operator parity against ``perovskite_sim.solver.mol``,
   coherent spectral TMM escape-probability binding, photon-budget closure, and
   radiative-limit ``V_oc(P_esc)`` enhancement ``V_T * ln(1 / P_esc)``.
3. Live WKB compilation identity, path-flux operator parity, and end-to-end steady-state
   device solves on ``perovskite-sim/tests/fixtures/configs/wkb_resolved_electron_barrier.yaml``
   under both ``PREALLOCATED_RESERVOIR_SUPERSET_V1`` and ``STRICT_WINDOW_REBIND_V1``.
"""

from __future__ import annotations

from collections.abc import Iterator
import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pytest
from threadpoolctl import threadpool_limits

from perovskite_sim.constants import K_B, Q
from perovskite_sim.discretization.grid import Layer as LegacyLayer, multilayer_grid as legacy_grid
from perovskite_sim.experiments.quasi_fermi_steady_state import solve_quasi_fermi_steady_state
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import DeviceStack, LayerSpec
from perovskite_sim.models.parameters import MaterialParams
from perovskite_sim.physics import photon_recycling as legacy_pr
from perovskite_sim.physics import tunneling as legacy_tun
from perovskite_sim.physics.tunneling_channel_device import (
    evaluate_tunnelling_channels as legacy_evaluate_tunnelling,
    physical_quasi_fermi_levels_eV as legacy_ef_levels,
)
from perovskite_sim.solver.mol import StateVec, assemble_rhs, build_material_arrays

from examples.research_1d.steady import load_device
from solarlab.config.device_import import import_standard_device
from solarlab.config.legacy_defaults import read_legacy_default_catalog
from solarlab.config.resolve_device import resolve_device
from solarlab.device.compile_nonlocal_1d import (
    CompiledNonlocalDevice1D,
    PHOTON_RECYCLING_LOCAL_ESCAPE_SCALED,
    compile_nonlocal_device_1d,
    solve_nonlocal_steady_state_1d,
)
from solarlab.device.defaults import DefaultCatalog
from solarlab.device.inputs import DeviceInput
from solarlab.equations.assembly.nonlocal_1d import AssembledNonlocalDC1DProblem
from solarlab.equations.nonlocal_1d import (
    evaluate_device_tunnelling_1d,
    physical_quasi_fermi_levels_eV,
)
from solarlab.materials.resources import ResourceLibrary, ResourceTable
from solarlab.materials.source import SourceDocument
from solarlab.physics.generation.photon_recycling import (
    TMM_INTEGRATED_ABSORBANCE_CONTROL_VOLUME_V1,
    TMM_INTEGRATED_ABSORBANCE_TRAPEZOID_V1,
    YABLONOVITCH_INTRINSIC_ALPHA_D_V1,
    compute_p_esc,
    compute_p_esc_for_absorber,
    wavelength_at_gap,
)
from solarlab.physics.generation.radiative_reabsorption import (
    CONTROL_VOLUME_CONSERVATIVE_V1,
    AbsorberReabsorptionSpec,
    evaluate_radiative_reabsorption,
)
from solarlab.physics.transport.tunneling_channels import (
    PREALLOCATED_RESERVOIR_SUPERSET_V1,
    STRICT_WINDOW_REBIND_V1,
)
from solarlab.physics.transport.wkb_tunneling import tfe_gamma

ROOT = Path(__file__).resolve().parents[2]
LEGACY_PKG = ROOT / "perovskite-sim" / "perovskite_sim"
WKB_FIXTURE_YAML = ROOT / "perovskite-sim" / "tests" / "fixtures" / "configs" / "wkb_resolved_electron_barrier.yaml"

LEGACY_DEFAULTS = (
    "models/parameters.py",
    "models/device.py",
    "models/defects.py",
    "physics/statistics.py",
    "physics/band_gap_narrowing.py",
    "scaps_compat/loader.py",
    "constants.py",
    "twod/microstructure.py",
    "models/tandem_config.py",
)
LEGACY_MODEL_SOURCES = (
    "physics/cigs_optics.py",
    "models/tunneling_channels.py",
)

Vector = NDArray[np.float64]


@pytest.fixture(scope="module", autouse=True)
def blas1() -> Iterator[None]:
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(scope="module")
def legacy_catalog() -> DefaultCatalog:
    sources = tuple(
        SourceDocument(name, (LEGACY_PKG / name).read_bytes()) for name in LEGACY_DEFAULTS
    )
    model_sources = tuple(
        SourceDocument(name, (LEGACY_PKG / name).read_bytes()) for name in LEGACY_MODEL_SOURCES
    )
    return read_legacy_default_catalog(sources, model_sources=model_sources)


def _find_voc(dev: CompiledNonlocalDevice1D, v_lo: float, v_max: float) -> tuple[float, Any]:
    """Stepped bracket + secant/bisection root find for illuminated open-circuit voltage."""
    s_lo = solve_nonlocal_steady_state_1d(dev, voltage_V=v_lo, generation_scale=1.0)
    j_lo = float(np.mean(s_lo.current_density_A_m2))
    v_cur = v_lo
    s_hi = s_lo
    j_hi = j_lo
    while v_cur < v_max and j_lo * j_hi > 0.0:
        v_next = min(v_max, v_cur + 0.02)
        s_next = solve_nonlocal_steady_state_1d(
            dev, initial_state=s_hi.state, voltage_V=v_next, generation_scale=1.0
        )
        j_next = float(np.mean(s_next.current_density_A_m2))
        if j_lo * j_next <= 0.0:
            v_hi = v_next
            s_hi = s_next
            j_hi = j_next
            break
        v_lo, s_lo, j_lo = v_next, s_next, j_next
        v_cur = v_next
        s_hi, j_hi = s_next, j_next
    else:
        v_hi = v_cur
    assert j_lo * j_hi <= 0.0, f"V_oc not bracketed up to {v_max}: J=({j_lo}, {j_hi})"

    state_seed = s_lo.state
    a, b = v_lo, v_hi
    fa, fb = j_lo, j_hi
    best = s_lo
    for _ in range(14):
        if abs(fb - fa) > 0.0:
            v_sec = b - fb * (b - a) / (fb - fa)
            margin = 0.05 * (b - a)
            v_mid = v_sec if (a + margin) < v_sec < (b - margin) else 0.5 * (a + b)
        else:
            v_mid = 0.5 * (a + b)
        sol = solve_nonlocal_steady_state_1d(
            dev, initial_state=state_seed, voltage_V=v_mid, generation_scale=1.0
        )
        state_seed = sol.state
        best = sol
        fm = float(np.mean(sol.current_density_A_m2))
        if abs(fm) <= 1.0e-7 or abs(b - a) <= 1.0e-7:
            return v_mid, sol
        if fa * fm <= 0.0:
            b, fb = v_mid, fm
        else:
            a, fa = v_mid, fm
    return float(best.voltage_V), best


# ---------------------------------------------------------------------------
# 1. Optical escape & static TFE primitives parity vs perovskite_sim
# ---------------------------------------------------------------------------


def test_optical_escape_and_tfe_primitives_match_legacy_bit_for_bit() -> None:
    for eg in (1.12, 1.50, 1.60, 2.10):
        assert wavelength_at_gap(eg) == pytest.approx(legacy_pr.wavelength_at_gap(eg), rel=1e-15)

    for alpha in (0.0, 2.0e4, 1.5e6, 8.0e6):
        for d_m in (50.0e-9, 450.0e-9, 1.2e-6):
            for n_r in (1.5, 2.4, 2.6, 3.5):
                p_new = compute_p_esc(alpha, d_m, n_r)
                p_leg = legacy_pr.compute_p_esc(alpha, d_m, n_r)
                assert p_new == pytest.approx(p_leg, rel=1e-15, abs=0.0)

    x_abs = np.linspace(0.0, 500.0e-9, 65)
    wl_m = np.linspace(400.0e-9, 900.0e-9, 41)
    for k_gap, n_gap in ((0.0, 2.5), (0.015, 2.55), (0.12, 2.65)):
        alpha_wl = 4.0 * math.pi * k_gap / wl_m
        a_xl = alpha_wl[None, :] * np.exp(-0.7 * x_abs[:, None] * alpha_wl[None, :])
        p_new = compute_p_esc_for_absorber(
            a_xl,
            wl_m,
            x_abs,
            1.60,
            n_gap,
            mode=TMM_INTEGRATED_ABSORBANCE_TRAPEZOID_V1,
        )
        p_leg = legacy_pr.compute_p_esc_for_absorber(a_xl, wl_m, x_abs, 1.60, n_gap)
        assert p_new == pytest.approx(p_leg, rel=1e-15, abs=0.0)

    vt_300 = K_B * 300.0 / Q
    for delta_e in (-0.05, 0.0, 0.08, 0.25, 0.45):
        for doping in (0.0, 1.0e22, 5.0e23, 2.0e24):
            for m_rel, eps_r in ((0.15, 10.0), (0.25, 24.0)):
                g_new = tfe_gamma(delta_e, doping, m_rel, eps_r, vt_300)
                g_leg = legacy_tun.tfe_gamma(delta_e, doping, m_rel, eps_r, vt_300)
                assert g_new == pytest.approx(g_leg, rel=1e-15, abs=0.0)


# ---------------------------------------------------------------------------
# 2. Radiative reabsorption operator parity vs mol.py & steady-state V_oc(P_esc)
# ---------------------------------------------------------------------------


def test_radiative_reabsorption_operator_parity_with_legacy_mol_and_voc_enhancement() -> None:
    # (a) Live operator parity against perovskite_sim.solver.mol.assemble_rhs
    vt = K_B * 300.0 / Q
    nc, nv = 1.0e24, 1.0e24
    ni_htl = math.sqrt(nc * nv * math.exp(-1.8 / vt))
    ni_abs = math.sqrt(nc * nv * math.exp(-1.6 / vt))
    ni_etl = math.sqrt(nc * nv * math.exp(-2.0 / vt))

    def _mp(eg: float, chi: float, ni_val: float, **kw: Any) -> MaterialParams:
        base = dict(
            eps_r=20.0,
            mu_n=1.0e-4,
            mu_p=1.0e-4,
            D_ion=0.0,
            P_lim=1.0e30,
            P0=0.0,
            ni=ni_val,
            tau_n=2.0e-6,
            tau_p=2.0e-6,
            n1=ni_val,
            p1=ni_val,
            B_rad=0.0,
            C_n=0.0,
            C_p=0.0,
            alpha=0.0,
            N_A=0.0,
            N_D=0.0,
            chi=chi,
            Eg=eg,
            Nc300=nc,
            Nv300=nv,
        )
        base.update(kw)
        return MaterialParams(**base)

    stack_off = DeviceStack(
        layers=(
            LayerSpec("htl", 100.0e-9, _mp(1.8, 3.6, ni_htl, N_A=1.0e23), "HTL"),
            LayerSpec("absorber", 450.0e-9, _mp(1.6, 3.9, ni_abs, B_rad=2.5e-15, alpha=2.0e6), "absorber"),
            LayerSpec("etl", 100.0e-9, _mp(2.0, 4.0, ni_etl, N_D=1.0e23), "ETL"),
        ),
        V_bi=1.0,
        Phi=1.0e21,
        interfaces=((0.0, 0.0), (0.0, 0.0)),
        mode="full",
    )
    x_grid = legacy_grid([LegacyLayer(layer.thickness, 12) for layer in stack_off.layers], alpha=3.0)
    n_nodes = x_grid.size

    mat_off = build_material_arrays(x_grid, stack_off)
    abs_mask = (x_grid > 100.0e-9 - 1.0e-15) & (x_grid <= 550.0e-9 - 1.0e-15)
    mat_on = dataclasses.replace(
        mat_off,
        has_radiative_reabsorption=True,
        absorber_masks=(abs_mask,),
        absorber_p_esc=(0.14,),
        absorber_thicknesses=(450.0e-9,),
    )

    rng = np.random.default_rng(20260412)
    n_prof = 1.0e20 * np.exp(0.6 * np.sin(np.linspace(0.0, 2.5, n_nodes))) + rng.uniform(1e18, 5e19, n_nodes)
    p_prof = 1.5e20 * np.exp(0.5 * np.cos(np.linspace(0.0, 2.0, n_nodes))) + rng.uniform(1e18, 5e19, n_nodes)
    n_prof[0] = mat_off.n_L
    n_prof[-1] = mat_off.n_R
    p_prof[0] = mat_off.p_L
    p_prof[-1] = mat_off.p_R
    y_state = StateVec.pack(n_prof, p_prof, np.zeros(n_nodes, dtype=float))

    rhs_off = assemble_rhs(0.0, y_state, x_grid, stack_off, mat_off, illuminated=False, V_app=0.5)
    rhs_on = assemble_rhs(0.0, y_state, x_grid, stack_off, mat_on, illuminated=False, V_app=0.5)
    dn_dt_diff = rhs_on[1 : n_nodes - 1] - rhs_off[1 : n_nodes - 1]
    dp_dt_diff = rhs_on[n_nodes + 1 : 2 * n_nodes - 1] - rhs_off[n_nodes + 1 : 2 * n_nodes - 1]

    spec_hist = AbsorberReabsorptionSpec.historical_trapezoid_uniform(
        layer_id="absorber",
        nodes_m=x_grid,
        absorber_mask=mat_on.absorber_masks[0],
        thickness_m=mat_on.absorber_thicknesses[0],
        intrinsic_radiative_B_m3_s=mat_on.B_rad,
        intrinsic_product_m6=mat_on.ni_sq,
        escape_probability=0.14,
    )
    ev_hist = evaluate_radiative_reabsorption([spec_hist], n_prof, p_prof, derivatives=False)
    # Near heterointerface clustered faces, subtracting ~1e34 s^-1 drift-diffusion divergences
    # in (rhs_on - rhs_off) has ~6e-9 IEEE-754 cancellation; interior absorber nodes match < 1e-12.
    np.testing.assert_allclose(ev_hist.generation_m3_s[1:-1], dn_dt_diff, rtol=1e-8, atol=0.0)
    np.testing.assert_allclose(ev_hist.generation_m3_s[1:-1], dp_dt_diff, rtol=1e-8, atol=0.0)
    np.testing.assert_allclose(
        ev_hist.generation_m3_s[15:22], dn_dt_diff[14:21], rtol=1e-12, atol=0.0
    )

    # (b) Coherent spectral TMM escape binding + steady-state V_oc(P_esc) enhancement
    base = load_device()
    tdoc_defaults = base.defaults
    from solarlab.physics.transport.tunneling_channels import TunnellingChannelDocument

    tdoc = TunnellingChannelDocument()
    defaults = dataclasses.replace(
        tdoc_defaults,
        complex_defaults=(
            ("cigs_graded_optics", (("model", "minoura_2015"), ("slices", 12), ("kk_quadrature_order", 160))),
            ("band_to_band", tuple(tdoc.band_to_band.to_dict().items())),
            ("intraband", tuple(tdoc.intraband.to_dict().items())),
            ("interface_defect_assisted", tuple(tdoc.interface_defect_assisted.to_dict().items())),
            ("contact", tuple(tdoc.contact.to_dict().items())),
        ),
    )

    wavelengths = (400, 450, 500, 550, 600, 650, 700, 750, 800, 850)
    photon_csv = "wavelength_nm,spectral_flux_m-2_s-1_m-1\n" + "".join(
        f"{wl},{6.0e27 if wl <= 760 else 0.0}\n" for wl in wavelengths
    )
    nk_csv = "wavelength_nm,n,k\n" + "".join(
        f"{wl},2.55,{0.18 if wl <= 780 else 0.02}\n" for wl in wavelengths
    )
    tables = (
        ResourceTable(
            "pr_test_photons",
            "spectrum",
            SourceDocument("pr_test_photons.csv", photon_csv.encode()),
        ),
        ResourceTable(
            "pr_test_absorber_nk",
            "nk",
            SourceDocument("pr_test_absorber_nk.csv", nk_csv.encode()),
        ),
    )
    decl = base.to_input().editing_data()
    decl["spectrum"] = tables[0].name
    decl["settings"]["Phi"] = 0.0
    for layer in decl["layers"]:
        p = layer["parameters"]
        p.update(alpha=0.0, incoherent=False, tau_n=1.0e-3, tau_p=1.0e-3, C_n=0.0, C_p=0.0)
        if layer["role"] == "absorber":
            abs_id = layer["id"]
            p.update(optical_material=tables[1].name, n_optical=None, B_rad=5.0e-15)
        else:
            p.update(optical_material=None, n_optical=2.0, B_rad=0.0)
    raw_decl = json.dumps(decl, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    src_decl = SourceDocument("pr_test.declaration", raw_decl)
    prep_tmm = resolve_device(
        DeviceInput.model_validate(decl),
        defaults,
        ResourceLibrary(tables),
        sources=(*base.sources, src_decl),
    )

    # Compile with live coherent spectral TMM escape binding across all 3 modes!
    dev_tmm_yab = compile_nonlocal_device_1d(
        prep_tmm,
        intervals_per_layer=8,
        photon_recycling_mode=CONTROL_VOLUME_CONSERVATIVE_V1,
        escape_probability_mode=YABLONOVITCH_INTRINSIC_ALPHA_D_V1,
    )
    dev_tmm_trap = compile_nonlocal_device_1d(
        prep_tmm,
        intervals_per_layer=8,
        photon_recycling_mode=CONTROL_VOLUME_CONSERVATIVE_V1,
        escape_probability_mode=TMM_INTEGRATED_ABSORBANCE_TRAPEZOID_V1,
    )
    dev_tmm_cv = compile_nonlocal_device_1d(
        prep_tmm,
        intervals_per_layer=8,
        photon_recycling_mode=CONTROL_VOLUME_CONSERVATIVE_V1,
        escape_probability_mode=TMM_INTEGRATED_ABSORBANCE_CONTROL_VOLUME_V1,
    )
    assert 0.0 < dev_tmm_yab.escape_bindings[0].escape_probability < 1.0
    assert 0.0 < dev_tmm_trap.escape_bindings[0].escape_probability < 1.0
    assert 0.0 < dev_tmm_cv.escape_bindings[0].escape_probability < 1.0

    # Sweep P_esc in (1.0, 0.3, 0.05) and verify radiative-limit V_oc enhancement!
    voc_map: dict[float, float] = {}
    for p_esc in (1.0, 0.3, 0.05):
        dev_cv = compile_nonlocal_device_1d(
            prep_tmm,
            intervals_per_layer=8,
            photon_recycling_mode=CONTROL_VOLUME_CONSERVATIVE_V1,
            explicit_escape_probabilities={abs_id: p_esc},
        )
        voc_cv, sol_cv = _find_voc(dev_cv, 1.00, 1.32)
        voc_map[p_esc] = voc_cv

        prob_cv = AssembledNonlocalDC1DProblem(dev_cv, voc_cv, 1.0)
        budgets = prob_cv.photon_recycling_budgets(sol_cv.state.u)
        assert len(budgets) == 1
        budget = budgets[0]
        assert budget.emitted_flux_m2_s > 0.0
        assert abs(budget.closure_residual_m2_s) <= 1.0e-14 * budget.emitted_flux_m2_s

        dev_loc = compile_nonlocal_device_1d(
            prep_tmm,
            intervals_per_layer=8,
            photon_recycling_mode=PHOTON_RECYCLING_LOCAL_ESCAPE_SCALED,
            explicit_escape_probabilities={abs_id: p_esc},
        )
        voc_loc, _ = _find_voc(dev_loc, 1.00, 1.32)
        assert abs(voc_cv - voc_loc) <= 2.5e-3

    assert voc_map[0.05] > voc_map[0.3] > voc_map[1.0]
    vt_dev = dev_tmm_cv.thermal_voltage_V
    for p_esc in (0.3, 0.05):
        expected_shift = vt_dev * math.log(1.0 / p_esc)
        actual_shift = voc_map[p_esc] - voc_map[1.0]
        assert abs(actual_shift - expected_shift) <= 3.0e-3


# ---------------------------------------------------------------------------
# 3. Live WKB compilation, operator, and steady-state parity on fixture YAML
# ---------------------------------------------------------------------------


def test_wkb_resolved_electron_barrier_live_operator_and_steady_state_parity(
    legacy_catalog: DefaultCatalog,
) -> None:
    stack = load_device_from_yaml(WKB_FIXTURE_YAML)
    counts = (10, 10, 10)
    x_leg = legacy_grid([LegacyLayer(layer.thickness, c) for layer, c in zip(stack.layers, counts)], alpha=3.0)
    mat_leg = build_material_arrays(x_leg, stack)
    assert mat_leg.tunnelling_channels is not None

    src_yaml = SourceDocument("wkb_resolved_electron_barrier.yaml", WKB_FIXTURE_YAML.read_bytes())
    dev_input = import_standard_device(
        src_yaml, id="wkb_resolved_electron_barrier", defaults=legacy_catalog
    )
    prepared = resolve_device(dev_input, legacy_catalog, ResourceLibrary(()), sources=(src_yaml,))

    dev_sup = compile_nonlocal_device_1d(
        prepared,
        intervals_per_layer=counts,
        grid_alpha=3.0,
        wkb_sparsity_policy=PREALLOCATED_RESERVOIR_SUPERSET_V1,
    )
    dev_strict = compile_nonlocal_device_1d(
        prepared,
        intervals_per_layer=counts,
        grid_alpha=3.0,
        wkb_sparsity_policy=STRICT_WINDOW_REBIND_V1,
    )
    assert dev_sup.tunnelling_channels is not None

    # Exact grid & SHA-256 compiled tunnelling identity parity!
    np.testing.assert_array_equal(dev_sup.nodes_m, x_leg)
    assert dev_sup.tunnelling_channels.identity_sha256 == mat_leg.tunnelling_channels.identity_sha256
    assert dev_sup.tunnelling_channels.enabled_channels == mat_leg.tunnelling_channels.enabled_channels

    # Live WKB path-flux operator parity at an explicit non-equilibrium state!
    s_probe = solve_nonlocal_steady_state_1d(dev_sup, voltage_V=0.05, generation_scale=0.0)
    phi_p = s_probe.state.potential_V
    n_p = s_probe.state.electron_density_m3
    p_p = s_probe.state.hole_density_m3
    tun_new = evaluate_device_tunnelling_1d(dev_sup, phi_p, n_p, p_p, derivatives=False)
    chi_leg = mat_leg.chi_phys if mat_leg.chi_phys is not None else mat_leg.chi
    eg_leg = mat_leg.Eg_phys if mat_leg.Eg_phys is not None else mat_leg.Eg
    assert mat_leg.N_C_physical is not None and mat_leg.N_V_physical is not None
    efn_leg, efp_leg = legacy_ef_levels(
        potential_V=phi_p,
        affinity_eV=chi_leg,
        band_gap_eV=eg_leg,
        electron_density_m3=n_p,
        hole_density_m3=p_p,
        conduction_dos_m3=mat_leg.N_C_physical,
        valence_dos_m3=mat_leg.N_V_physical,
        thermal_voltage_V=mat_leg.V_T_device,
    )
    tun_leg = legacy_evaluate_tunnelling(
        mat_leg.tunnelling_channels,
        positions_m=x_leg,
        potential_V=phi_p,
        affinity_eV=chi_leg,
        band_gap_eV=eg_leg,
        electron_quasi_fermi_eV=efn_leg,
        hole_quasi_fermi_eV=efp_leg,
        thermal_voltage_V=mat_leg.V_T_device,
    )
    assert tun_new is not None and tun_new.intraband_path is not None
    assert tun_leg.intraband_path is not None

    np.testing.assert_allclose(
        tun_new.intraband_path.actions, tun_leg.intraband_path.actions, rtol=1e-14, atol=0.0
    )
    np.testing.assert_allclose(
        tun_new.intraband_path.particle_face_flux_m2_s,
        tun_leg.intraband_path.particle_face_flux_m2_s,
        rtol=1e-14,
        atol=0.0,
    )
    np.testing.assert_allclose(
        tun_new.electron_face_current_A_m2, tun_leg.electron_face_current_A_m2, rtol=1e-14, atol=0.0
    )
    np.testing.assert_allclose(
        tun_new.channel_net_flux_m2_s,
        tun_leg.channel_net_flux_m2_s,
        rtol=1e-14,
        atol=0.0,
    )
    assert abs(float(np.sum(tun_new.electron_nodal_transfer_m2_s))) <= 1.0e-12 * float(
        np.max(np.abs(tun_new.electron_nodal_transfer_m2_s))
    )

    # Channel-disabled reference device to verify observable WKB tunnelling enhancement!
    doc_no_wkb = prepared.to_input().editing_data()
    doc_no_wkb["tunnelling_channels"]["intraband"]["enabled"] = False
    prep_no_wkb = resolve_device(
        DeviceInput.model_validate(doc_no_wkb), legacy_catalog, ResourceLibrary(())
    )
    dev_no_wkb = compile_nonlocal_device_1d(
        prep_no_wkb, intervals_per_layer=counts, grid_alpha=3.0
    )

    # End-to-end steady-state parity across reverse, equilibrium, and forward biases!
    for v_app in (-0.02, 0.0, 0.02, 0.10):
        leg_res = solve_quasi_fermi_steady_state(x_leg, stack, V_app=v_app, illuminated=False)
        sol_sup = solve_nonlocal_steady_state_1d(dev_sup, voltage_V=v_app, generation_scale=0.0)
        sol_str = solve_nonlocal_steady_state_1d(dev_strict, voltage_V=v_app, generation_scale=0.0)

        # Superset vs strict rebind policies agree to machine precision!
        np.testing.assert_allclose(
            sol_str.current_density_A_m2, sol_sup.current_density_A_m2, rtol=1e-10, atol=1e-12
        )

        # Electrostatic potential and terminal current parity against perovskite_sim!
        np.testing.assert_allclose(sol_sup.state.potential_V, leg_res.phi, rtol=0.0, atol=1e-7)

        j_term_new = -float(np.mean(sol_sup.current_density_A_m2))
        if v_app == 0.0:
            assert abs(j_term_new) <= 1.0e-10
            assert abs(leg_res.current_A_m2) <= 1.0e-10
        else:
            assert j_term_new == pytest.approx(leg_res.current_A_m2, rel=1e-6)

            prob_v = AssembledNonlocalDC1DProblem(dev_sup, v_app, 0.0)
            tun_v = prob_v.tunnelling_evaluation(sol_sup.state.u)
            assert tun_v is not None
            assert leg_res.tunnelling_channel_diagnostics is not None
            np.testing.assert_allclose(
                tun_v.electron_face_current_A_m2,
                leg_res.tunnelling_channel_diagnostics.electron_face_current_A_m2,
                rtol=1e-6,
                atol=1e-10,
            )

            # Strictly dissipative WKB chemical work!
            bands = dev_sup.parameters.band_structure
            efn_eV, _ = physical_quasi_fermi_levels_eV(
                potential_V=sol_sup.state.potential_V,
                affinity_eV=bands.physical_electron_affinity_eV,
                band_gap_eV=bands.physical_band_gap_eV,
                electron_density_m3=sol_sup.state.electron_density_m3,
                hole_density_m3=sol_sup.state.hole_density_m3,
                conduction_dos_m3=dev_sup.parameters.conduction_dos_m3,
                valence_dos_m3=dev_sup.parameters.valence_dos_m3,
                thermal_voltage_V=dev_sup.thermal_voltage_V,
            )
            chem_work = float(np.dot(efn_eV, tun_v.electron_nodal_transfer_m2_s))
            assert chem_work < 0.0

            # Observable WKB current enhancement over channel-disabled barrier!
            sol_no_wkb = solve_nonlocal_steady_state_1d(
                dev_no_wkb, voltage_V=v_app, generation_scale=0.0
            )
            j_no_wkb = -float(np.mean(sol_no_wkb.current_density_A_m2))
            assert abs(j_term_new) > abs(j_no_wkb)
            assert abs(j_term_new) - abs(j_no_wkb) > 1.0e-3
