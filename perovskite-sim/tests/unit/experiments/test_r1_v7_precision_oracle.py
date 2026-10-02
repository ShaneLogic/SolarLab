"""Same-equation oracle and actual production-call fault controls for V7."""
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal, localcontext
import inspect
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from perovskite_sim.experiments import defect_ion_combined_impedance as assembly
from perovskite_sim.experiments.one_dimensional_mechanism_r1 import build_r1_material
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.physics import ion_migration, physical_control_volume
from scripts import verify_r1_v7_precision as oracle


@pytest.fixture(scope="module")
def healthy_material():
    fixture = Path(__file__).parents[2] / "fixtures/configs/dynamic_interface_defect_ion_transient_absorber_only.yaml"
    return build_r1_material(load_device_from_yaml(fixture), 16)


@pytest.fixture
def sample(healthy_material):
    grid, material = healthy_material
    frozen = oracle.freeze_healthy_material(grid, material, source_identity="unit-frozen-healthy-material")
    phi = np.linspace(0.0, 0.003, len(grid))
    state = {"phi_V": phi.tolist(), "n_m3": [1e15] * len(grid), "p_m3": [1e15] * len(grid),
             "positive_m3": material.P_ion0.tolist(),
             "sheet_charge_C_m2": [0.] * len(material.iface_qss_left_nodes)}
    return grid, material, frozen, state


def production_observation(grid, material, state, *, boundaries=(0., 0.)):
    """Call the actual production rate and flux assemblers after fault install."""
    rate, _, flux, _ = assembly._ion_fields(
        grid, material, np.asarray(state["positive_m3"]), None, np.asarray(state["phi_V"]))
    return {"positive_flux_m2_s": flux, "positive_rate_m3_s": rate,
            "boundary_flux_m2_s": list(boundaries)}


def test_independently_rebuilt_actual_r1_coefficients_match_the_frozen_discrete_problem(sample):
    _, material, frozen, _ = sample
    assert frozen["healthy_cached_coefficient_identity"] == "bit_identical"
    assert np.array_equal(np.asarray(frozen["coefficients"]["poisson_capacitance_F_m2"], float),
                          material.poisson_factor.C)
    assert np.array_equal(np.asarray(frozen["coefficients"]["physical_width_m"], float), material.dx_cell)
    assert np.array_equal(np.asarray(frozen["coefficients"]["D_ion_face_m2_s"], float), material.D_ion_face)


def test_faulted_material_cannot_be_frozen_as_if_it_were_the_declared_healthy_law(sample):
    grid, material, _, _ = sample
    with pytest.raises(ValueError, match="D_ion_face differs"):
        oracle.freeze_healthy_material(grid, replace(material, D_ion_face=material.D_ion_face * 1.01),
                                       source_identity="wrongly-frozen-fault")


def test_healthy_production_calls_pass_independent_per_side_checks(sample):
    grid, material, frozen, state = sample
    observed = production_observation(grid, material, state)
    result = oracle.verify_pair(frozen, state, observed, state, observed)
    assert result["qualified"]
    assert all(item["passed"] for item in result["direct"]["oracle_stability"].values())
    assert result["direct"]["poisson"]["qualified"] is None


@pytest.mark.parametrize("mutation", ["diffusion", "thermal_voltage", "drift_sign", "omitted_ion", "one_face_sign"])
def test_common_faults_are_injected_into_production_calls_and_caught_per_side(sample, monkeypatch, mutation):
    grid, material, frozen, state = sample
    baseline = production_observation(grid, material, state)
    assert oracle.verify_side(frozen, state, baseline)["qualified"]
    selected = int(np.flatnonzero(material.D_ion_face)[0])
    changed = material
    if mutation == "diffusion":
        changed = replace(material, D_ion_face=material.D_ion_face * 1.01)
    elif mutation == "thermal_voltage":
        changed = replace(material, V_T_device=material.V_T_device * 1.01)
    else:
        original_kernel = ion_migration._steric_diffusion_only_flux

        def faulty_local_face_assembly(P, phi, dx, diffusion, vt, limit, node_limit, partner, drift_sign):
            # This is the shared implementation called independently by BOTH
            # production flux and production continuity assembly, not a
            # transformation of one observation after its return.
            if mutation == "omitted_ion":
                return np.zeros(len(phi) - 1)
            if mutation == "drift_sign":
                drift_sign = -drift_sign
            result = original_kernel(P, phi, dx, diffusion, vt, limit, node_limit, partner, drift_sign)
            if mutation == "one_face_sign":
                result[selected] = -result[selected]
            return result

        monkeypatch.setattr(ion_migration, "_steric_diffusion_only_flux", faulty_local_face_assembly)
    direct = production_observation(grid, changed, state)
    eliminated = production_observation(grid, changed, state)
    checked = oracle.verify_pair(frozen, state, direct, state, eliminated)
    assert checked["original_gate"]["passed"]  # Shared errors are invisible here.
    assert not checked["qualified"]
    assert not checked["direct"]["checks"]["positive_flux_m2_s"]["passed"]
    assert not checked["eliminated"]["checks"]["positive_flux_m2_s"]["passed"]


def test_boundary_leak_is_injected_before_actual_continuity_assembly(sample, monkeypatch):
    grid, material, frozen, state = sample
    actual_boundaries = []
    original = physical_control_volume.flux_divergence

    def leaky_boundary(flux, widths, *, left_flux=0., right_flux=0.):
        left_flux = 1e5
        actual_boundaries[:] = [left_flux, right_flux]
        return original(flux, widths, left_flux=left_flux, right_flux=right_flux)

    monkeypatch.setattr(physical_control_volume, "flux_divergence", leaky_boundary)
    observed = production_observation(grid, material, state)
    observed["boundary_flux_m2_s"] = actual_boundaries
    report = oracle.verify_side(frozen, state, observed)
    assert report["checks"]["positive_flux_m2_s"]["passed"]
    assert not report["checks"]["boundary_flux_m2_s"]["passed"]
    assert not report["qualified"]


def test_healthy_frozen_snapshot_survives_later_material_cache_mutation(sample):
    grid, material, frozen, state = sample
    before = oracle.evaluate_state(frozen, state)
    different_material = replace(material, D_ion_face=material.D_ion_face * 1.01,
                                 V_T_device=material.V_T_device * 1.01)
    assert different_material.V_T_device != material.V_T_device
    assert oracle.evaluate_state(frozen, state) == before


@pytest.mark.parametrize("form", ["flat", "nested", "explicit"])
def test_hi_plus_lo_is_consumed_and_not_cast_back_to_float64(sample, form):
    _, _, frozen, state = sample
    size = len(state["phi_V"])
    state["phi_V"] = [0.4] * size
    split = {name: {"hi": list(state[name]), "lo": [0.] * len(state[name])}
             for name in oracle.STATE_FIELDS}
    active = next(j for j, value in enumerate(frozen["coefficients"]["D_ion_face_m2_s"])
                  if Decimal(value) > 0)
    split["phi_V"]["lo"][active + 1] = 1e-20
    if form == "flat":
        precise = {**state, **{f"precision_{name}_{part}": values[part]
                              for name, values in split.items() for part in ("hi", "lo")}}
    elif form == "nested":
        precise = {**state, "precision": {"representation": "float64-pair-v1", "fields": split}}
    else:
        precise = split
    regular = oracle.evaluate_state(frozen, state)
    result = oracle.evaluate_state(frozen, precise)
    assert float(0.4 + 1e-20) == 0.4
    assert Decimal(regular["positive_flux_m2_s"][active]) == 0
    assert Decimal(result["positive_flux_m2_s"][active]) < 0
    assert result["state_input"]["explicit_low_fields"] == list(oracle.STATE_FIELDS)
    assert not oracle.compare_vectors(result["positive_flux_m2_s"], regular["positive_flux_m2_s"],
                                       units="m^-2 s^-1")["passed"]


def test_single_path_potential_input_error_is_seen_by_original_difference(sample):
    grid, material, frozen, state = sample
    changed = deepcopy(state)
    active = int(np.flatnonzero(material.D_ion_face)[0])
    changed["phi_V"][active + 1] += 1e-6
    first = production_observation(grid, material, state)
    second = production_observation(grid, material, changed)
    report = oracle.verify_pair(frozen, state, first, changed, second)
    assert report["direct"]["qualified"] and report["eliminated"]["qualified"]
    assert not report["original_gate"]["passed"]
    assert not report["qualified"]


def test_frozen_controls_have_zero_rates_without_removing_ion_charge(sample):
    grid, material, _, state = sample
    frozen = oracle.freeze_healthy_material(grid, material, source_identity="healthy-A", control_label="A")
    result = oracle.evaluate_state(frozen, state)
    assert all(Decimal(value) == 0 for value in result["positive_flux_m2_s"])
    assert all(Decimal(value) == 0 for value in result["positive_rate_m3_s"])
    assert "charge_density_C_m3" in result["poisson"]


def test_poisson_independently_accounts_for_a_sheet_and_carrier_low_part(sample):
    _, _, frozen, state = sample
    size = len(state["phi_V"])
    state["phi_V"] = [0.] * size
    before = oracle.evaluate_state(frozen, state)
    precise = {name: {"hi": list(state[name]), "lo": [0.] * len(state[name])}
               for name in oracle.STATE_FIELDS}
    nodes = frozen["coefficients"]["interface_nodes"][0]
    precise["sheet_charge_C_m2"]["lo"][0] = "1e-19"
    precise["p_m3"]["lo"][nodes[0]] = "0.001"
    after = oracle.evaluate_state(frozen, precise)
    with localcontext() as context:
        context.prec = 80
        for side, node in enumerate(nodes):
            difference = (Decimal(after["poisson"]["residual_C_m2"][node - 1])
                          - Decimal(before["poisson"]["residual_C_m2"][node - 1]))
            expected = Decimal(frozen["coefficients"]["sheet_weights"][0][side]) * Decimal("1e-19")
            if side == 0:
                expected += (Decimal(frozen["constants"]["q_C"]) * Decimal("0.001")
                             * Decimal(frozen["coefficients"]["physical_width_m"][node]))
            assert abs(difference - expected) < Decimal("1e-70")


def test_missing_rate_or_boundary_never_becomes_inferred_success(sample):
    grid, material, frozen, state = sample
    observed = production_observation(grid, material, state)
    del observed["boundary_flux_m2_s"]
    del observed["positive_rate_m3_s"]
    result = oracle.verify_side(frozen, state, observed)
    assert not result["qualified"]
    assert result["missing_observed"] == ["positive_rate_m3_s", "boundary_flux_m2_s"]


def test_flat_precision_has_precedence_and_missing_low_field_fails_closed(sample):
    _, _, frozen, state = sample
    state["precision_phi_V_hi"] = state["phi_V"]
    with pytest.raises(ValueError, match="missing a required precision field"):
        oracle.evaluate_state(frozen, state)


def test_frozen_hash_and_observation_shape_are_enforced(sample):
    _, _, frozen, state = sample
    altered = deepcopy(frozen)
    altered["coefficients"]["D_ion_face_m2_s"][0] = "123"
    with pytest.raises(ValueError, match="hash mismatch"):
        oracle.evaluate_state(altered, state)
    with pytest.raises(ValueError, match="lengths differ"):
        oracle.compare_vectors([1, 2], [1], units="m^-2 s^-1")


def test_oracle_does_not_import_or_call_a_production_numerical_implementation():
    source = inspect.getsource(oracle)
    assert "from perovskite_sim" not in source
    assert "import perovskite_sim" not in source
    assert "ion_face_flux(" not in source
    assert "poisson_project" not in source


@pytest.mark.parametrize("drive", ["-1e-20", "0", "1e-20"])
def test_zero_and_signed_sub_ulp_drive_have_stable_oracle_values(sample, drive):
    _, _, frozen, state = sample
    state["phi_V"] = [0.4] * len(state["phi_V"])
    active = next(j for j, value in enumerate(frozen["coefficients"]["D_ion_face_m2_s"])
                  if Decimal(value) > 0)
    split = {name: {"hi": list(state[name]), "lo": [0.] * len(state[name])}
             for name in oracle.STATE_FIELDS}
    split["phi_V"]["lo"][active + 1] = drive
    low, high = oracle.evaluate_state(frozen, split), oracle.evaluate_state(frozen, split, precision=100)
    assert oracle.compare_vectors(low["positive_flux_m2_s"], high["positive_flux_m2_s"],
                                   limit=Decimal("1e-20"), units="m^-2 s^-1")["passed"]
    value = Decimal(low["positive_flux_m2_s"][active])
    assert value == 0 if drive == "0" else value * Decimal(drive) < 0


def test_real_v5_row851_retains_its_measured_equation_residual_and_fails_the_oracle(healthy_material):
    # Exact selected state from V5 Long/N16/D/AttemptV1/AcceptedStepsV1.jsonl,
    # row 851, t=1.9663570500354033 s, substeps=4. The complete source was
    # SHA256-verified while extracting this bounded, self-contained regression:
    # 652b4b6e3c15ac0473b3743e3d0811d5a10eff0c7559d9955fa80f7e4838d9cd.
    # No /tmp, iCloud or historical worktree dependency is required by the test.
    state = {
        "phi_V": [0.0, -0.0015992313103144649, -0.005247760024261007, -0.011919210450597729,
                  -0.020765467606613592, -0.029615594028103442, -0.036293969219870895,
                  -0.03994716739289411, -0.04320994222643342, -0.047187457284047093,
                  -0.05548519514143226, -0.06828927373219866, -0.08109335231636862,
                  -0.08939109016155034, -0.09336860521065585, -0.09499999999999964],
        "n_m3": [2515095827087.2163, 2359611296071.039, 2038211316541.4773, 1553287606426.209,
                 1071549586497.4185, 729300343304.1593, 541944360611.93115, 459709918772.4196,
                 19129658191102.22, 16390913221345.035, 11869998885959.771, 7204106249196.3,
                 4360720932065.6895, 3142762755112.4897, 2683836892476.661, 2515095827087.2163],
        "p_m3": [2515095827087.2163, 2679263198583.6206, 3094685336733.084, 4026456844991.052,
                 5702629273915.268, 8063927124176.771, 10461544458944.533, 12058749782125.96,
                 290511554604.84656, 348201164494.0228, 501305252929.11804, 858734030491.041,
                 1445261480386.5796, 2013564395972.9153, 2357839181089.977, 2515095827087.2163],
        "positive_m3": [6.189821222883317e21, 6.457289960479146e21, 7.089128535970169e21,
                        8.309174794020064e21, 1.0003722211504356e22, 1.169878265224868e22,
                        1.2919593453686993e22, 1.355183215973379e22,
                        1e22, 1e22, 1e22, 1e22, 1e22, 1e22, 1e22, 1e22],
        "sheet_charge_C_m2": [4.2776189953869995e-10],
    }
    state_hash = hashlib.sha256(json.dumps(state, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert state_hash == "a8b8adfb1d442998fa74a204c70e4ad9938e13f93afa02dac6783d18fa4c2ee2"
    grid, material = healthy_material
    frozen = oracle.freeze_healthy_material(grid, material, source_identity="a71f1fa/V5-row851")
    observed = {
        "positive_flux_m2_s": [3.4336540347180833, 0., 0., -0.4374891379489823, -2.624934827693893,
                               -2.7003240668443604, 8.449967225552664, 0., 0., 0., 0., 0., 0., 0., 0.],
        "positive_rate_m3_s": [-2248760229.1366143, 654069297.952255, -0., 22150989.036260884,
                               91265080.36772898, 3817114.677870683, -970513214.0351827,
                               1246935166.9970784, -0., -0., -0., -0., -0., -0., -0., -0.],
        "boundary_flux_m2_s": [0., 0.],
    }
    result = oracle.verify_side(frozen, state, observed)
    assert float(result["poisson"]["maximum_absolute_C_m2"]) == pytest.approx(4.345998333683578e-18, rel=2e-14)
    assert not result["qualified"]
    assert not result["checks"]["positive_flux_m2_s"]["passed"]
    assert not result["checks"]["positive_rate_m3_s"]["passed"]
    assert all(item["passed"] for item in result["oracle_stability"].values())
