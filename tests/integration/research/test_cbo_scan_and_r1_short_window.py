"""Integration tests for ``solarlab_research.cbo_scan`` and ``solarlab_research.r1`` (P07-01, P07-02).

Covers:
* Public 1D CBO scan orchestration with ``recomputed_built_in`` and ``fixed_contacts``
  boundary policies, adaptive midpoint bridge continuation, onset critical intervals,
  nested voltage-grid convergence certification, spatial mesh convergence certification,
  and ``D20`` pre-``P08`` model-pinning enforcement.
* R1 controls ``A-D`` preparation from a common dark ``0-`` equilibrium anchor with
  exact replication and exact zero-flux / zero-rate suppression of disabled channels.
* Minimal ``183/309`` public-interface witness evaluation preserving historical V53
  step ``309`` ``failed`` status without relabeling.
* Conservative ``4/8/16`` baseline-vs-compensated short-window coupled pairing with
  all 16 V53 certificate gates, 13 operator channels, sub-ULP storage precision check,
  and deterministic ``R1ReplayReceipt`` (``r1_l2_certified=False``,
  ``r1_l3_device_qualified=False``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from solarlab.physics.registry import ModelRef

from solarlab_research.cbo_scan import (
    CBOScanConfig,
    CBOVoltageGridConvergencePolicy,
    FIXED_CONTACTS,
    InterfaceCBOScanError,
    RECOMPUTED_BUILT_IN,
    certify_cbo_grid_convergence,
    certify_cbo_voltage_grid_convergence,
    default_cbo_prepared_device,
    enforce_pre_p08_cbo_pinning,
    run_cbo_scan,
)
from solarlab_research.r1 import (
    R1_OPERATOR_CHANNELS,
    R1_V53_CERTIFICATE_LIMITS,
    evaluate_r1_witness_minimal_input,
    run_r1_short_window_pairing,
    synthetic_research_release_bundle,
    verify_r1_controls_abcd,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
HIST_WITNESS = REPO_ROOT / "tests" / "fixtures" / "refactor" / "R1FailureWitnessV1.json"


def test_cbo_scan_boundary_policies_adaptive_bridge_and_d20_pinning() -> None:
    baseline = default_cbo_prepared_device()

    # D20 pre-P08 model pinning rejects prospective P08 models unless explicitly opted in
    p08_ref = ModelRef("spectral_control_volume_beer_lambert", 1)
    with pytest.raises(InterfaceCBOScanError, match="D20 model-pinning violation"):
        enforce_pre_p08_cbo_pinning(baseline, requested_models=(p08_ref,))
    opted_in = enforce_pre_p08_cbo_pinning(
        baseline, requested_models=(p08_ref,), allow_prospective_p08_models=True
    )
    assert opted_in == (p08_ref,)

    coarse_v = tuple(np.linspace(0.0, 1.125, 17).tolist())
    fine_v = tuple(np.linspace(0.0, 1.125, 33).tolist())

    # 1. recomputed_built_in with nested voltage grids + adaptive midpoint bridges
    cfg_recomputed = CBOScanConfig(
        delta_ec_eV=(-0.10, 0.0, 0.15, 0.25),
        boundary_policy=RECOMPUTED_BUILT_IN,
        intervals_per_layer=(4, 8, 4),
        voltage_grids_V=(coarse_v, fine_v),
        maximum_delta_step_eV=0.06,
        relative_drop_fraction=0.02,
        voltage_grid_policy=CBOVoltageGridConvergencePolicy(
            minimum_voltage_grids=2,
            maximum_voc_change_V=2.0e-2,
            maximum_ff_change=3.0e-2,
            maximum_pce_change=3.0e-2,
        ),
    )
    res_recomputed = run_cbo_scan(baseline, cfg_recomputed)
    assert res_recomputed.complete is True
    assert res_recomputed.certified is True
    assert res_recomputed.sync_vbi is True
    assert res_recomputed.bridge_point_count >= 2

    req_pts_recomp = [pt for pt in res_recomputed.points if pt.requested]
    assert len(req_pts_recomp) == 4
    # Under recomputed_built_in, shifting ETL chi changes applied V_bi linearly
    vbi_vals_recomp = [pt.applied_V_bi_V for pt in req_pts_recomp]
    assert max(vbi_vals_recomp) - min(vbi_vals_recomp) > 0.30

    vgrid_cert = certify_cbo_voltage_grid_convergence(res_recomputed)
    assert vgrid_cert.certified is True, f"voltage grid convergence failed: {vgrid_cert.reasons}"
    assert vgrid_cert.voltage_point_counts == (17, 33)

    # 2. fixed_contacts keeps applied_V_bi_V and grid_node_count strictly invariant
    cfg_fixed = CBOScanConfig(
        delta_ec_eV=(-0.10, 0.0, 0.15),
        boundary_policy=FIXED_CONTACTS,
        intervals_per_layer=(4, 8, 4),
        voltages_V=coarse_v,
        calculate_jv_metrics=False,
    )
    res_fixed = run_cbo_scan(baseline, cfg_fixed)
    assert res_fixed.complete is True
    assert res_fixed.sync_vbi is False
    vbi_fixed = [pt.applied_V_bi_V for pt in res_fixed.points]
    assert np.allclose(vbi_fixed, vbi_fixed[0], rtol=0.0, atol=1.0e-12)
    assert res_fixed.grid_node_count == res_recomputed.grid_node_count

    # 3. Spatial mesh convergence across two meshes sharing identical protocol_sha256
    cfg_mesh_fine = CBOScanConfig(
        delta_ec_eV=(-0.10, 0.0, 0.15, 0.25),
        boundary_policy=RECOMPUTED_BUILT_IN,
        intervals_per_layer=(6, 12, 6),
        voltage_grids_V=(coarse_v, fine_v),
        maximum_delta_step_eV=0.06,
        relative_drop_fraction=0.02,
        voltage_grid_policy=cfg_recomputed.voltage_grid_policy,
    )
    res_mesh_fine = run_cbo_scan(baseline, cfg_mesh_fine)
    assert res_mesh_fine.protocol_sha256 == res_recomputed.protocol_sha256

    ff_ci = next(c for c in res_recomputed.critical_intervals if c.metric == "FF")
    assert ff_ci.resolved is True
    mesh_cert = certify_cbo_grid_convergence(
        (res_recomputed, res_mesh_fine),
        metric="FF",
        maximum_envelope_width_eV=0.20,
        maximum_reference_relative_spread=0.05,
    )
    assert mesh_cert.certified is True, f"spatial mesh convergence failed: {mesh_cert.reasons}"
    assert mesh_cert.grid_interval_counts == (16, 24)


def test_r1_controls_abcd_witness_and_short_window_pairing() -> None:
    # 1. Controls A-D from a common replicated 0- anchor
    suite = verify_r1_controls_abcd(intervals_per_layer=(4, 3))
    assert suite.passed is True
    assert set(suite.controls) == {"A", "B", "C", "D"}
    for code in ("A", "B", "C", "D"):
        audit = suite.controls[code]
        assert audit.passed is True
        assert audit.zero_minus_replicated is True
        assert audit.ion_inventory_relative_drift <= 1.0e-12

    assert suite.controls["A"].max_ion_flux_m2_s == 0.0
    assert suite.controls["A"].max_ion_rate_m2_s == 0.0
    assert suite.controls["A"].trap_occupancy_rate_s == 0.0
    assert suite.controls["A"].sheet_charge_C_m2 == 0.0

    assert suite.controls["B"].max_ion_flux_m2_s > 0.0
    assert suite.controls["B"].trap_occupancy_rate_s == 0.0
    assert suite.controls["B"].sheet_charge_C_m2 == 0.0

    assert suite.controls["C"].max_ion_flux_m2_s == 0.0
    assert suite.controls["C"].trap_occupancy_rate_s > 0.0
    assert abs(suite.controls["C"].sheet_charge_C_m2) > 0.0

    assert suite.controls["D"].max_ion_flux_m2_s > 0.0
    assert suite.controls["D"].trap_occupancy_rate_s > 0.0
    assert abs(suite.controls["D"].sheet_charge_C_m2) > 0.0

    # 2. Minimal 183/309 public-interface witness evaluation (non-relabeling)
    witness_eval = evaluate_r1_witness_minimal_input(HIST_WITNESS, intervals_per_layer=(4, 3))
    assert witness_eval.parsed_fixture_layer_count >= 2
    assert witness_eval.public_step183_accepted is True
    assert witness_eval.public_step309_accepted is True
    assert witness_eval.public_step183_jacobian_error < R1_V53_CERTIFICATE_LIMITS["analytic_jacobian_error"]
    assert witness_eval.public_step309_jacobian_error < R1_V53_CERTIFICATE_LIMITS["analytic_jacobian_error"]
    assert witness_eval.public_step309_nonlinear_residual <= R1_V53_CERTIFICATE_LIMITS["nonlinear_residual"]
    # Historical V53 step 309 remains failed and unrelabelled
    assert witness_eval.historical_step309_status == "failed"
    assert witness_eval.relabelled is False

    # 3. Conservative 4/8/16 baseline-vs-compensated short-window pairing + replay receipt
    bundle = synthetic_research_release_bundle()
    pairing = run_r1_short_window_pairing(
        bundle=bundle,
        control="D",
        times_s=(0.0, 1.0e-7, 1.0e-6),
        intervals_per_layer=(4, 3),
    )
    assert pairing.baseline.accepted_rows == 59
    assert pairing.compensated.accepted_rows == 59
    assert dict(pairing.baseline.substep_row_counts) == {4: 9, 8: 17, 16: 33}
    assert dict(pairing.compensated.substep_row_counts) == {4: 9, 8: 17, 16: 33}
    assert pairing.baseline.certificate.passed is True, pairing.baseline.certificate.failed_items
    assert pairing.compensated.certificate.passed is True, pairing.compensated.certificate.failed_items
    assert set(pairing.compensated.operator_channels) == set(R1_OPERATOR_CHANNELS)
    assert pairing.sub_ulp_precision_ratio > 1.0e6

    receipt = pairing.replay_receipt
    assert receipt.deterministic_match is True
    assert receipt.compensated_final_state_sha256 == receipt.replay_compensated_state_sha256
    assert receipt.qualification_level == "short_window_interface_verification_only"
    assert receipt.r1_l2_certified is False
    assert receipt.r1_l3_device_qualified is False
    assert pairing.qualification.r1_l2_certified is False
    assert pairing.qualification.r1_l3_device_qualified is False
