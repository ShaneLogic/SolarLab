"""Production pair formats reject incomplete state and numeric sidecars."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from perovskite_sim.experiments import one_dimensional_mechanism_r1_pair_codec as codec
from perovskite_sim.experiments.one_dimensional_mechanism_r1_backend import get_backend
from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import R1StateError, digest


def snapshot():
    shapes = {**dict.fromkeys(("n_m3", "p_m3", "positive_m3", "phi_V", "dqfn_V", "dqfp_V", "positive_rate_m3_s"), (4,)),
        **dict.fromkeys(("occupancy", "sheet_charge_C_m2", "positive_inventory_m2"), (1,)),
        **dict.fromkeys(("electron_current_A_m2", "hole_current_A_m2", "positive_flux_m2_s"), (3,)),
        "trace_potential_V": (1, 2), "trace_state_m3": (1, 4), "capture_m2_s": (1, 4),
        "local_residual": (6,), "poisson_residual_C_m2": (2,), "boundary_flux_m2_s": (2,), "storage": (7,)}
    result = {name: np.ones(shapes[name]).tolist() for name in codec.LEGACY_SNAPSHOT_FIELDS}
    for name in codec.FINE_FIELDS:
        result["precision_" + name + "_hi"] = np.ones(shapes[name]).tolist()
        result["precision_" + name + "_lo"] = np.zeros(shapes[name]).tolist()
    return result


def rows(count=2, *, pair=True):
    state = snapshot()
    if not pair:
        state = {name: state[name] for name in codec.LEGACY_SNAPSHOT_FIELDS}
    return [{"substeps": 1, "time_s": index * 1e-9, "dt_s": 1e-9 if index else 0.,
             "state": deepcopy(state)} for index in range(count)]


@pytest.mark.parametrize("mutation", ["missing_low", "extra", "shape", "nonfinite", "unnormalized", "high_mismatch"])
def test_snapshot_rejects_real_structural_mutations(mutation):
    value = snapshot()
    codec.validate_snapshot(value)
    if mutation == "missing_low":
        value.pop("precision_n_m3_lo")
    elif mutation == "extra":
        value["unclassified_precision"] = [0.]
    elif mutation == "shape":
        value["precision_n_m3_lo"] = [0.]
    elif mutation == "nonfinite":
        value["precision_n_m3_lo"][0] = float("nan")
    elif mutation == "unnormalized":
        value["precision_n_m3_lo"][0] = 1.
    else:
        value["n_m3"][0] = 2.
    with pytest.raises(R1StateError):
        codec.validate_snapshot(value)


@pytest.mark.parametrize("pair,expected", [(False, 16), (True, 50)])
def test_sidecar_rows_are_stacked_and_count_does_not_grow(tmp_path, pair, expected):
    short, longer = rows(2, pair=pair), rows(20, pair=pair)
    a, b = codec.numeric_arrays(codec.state_sidecar_payload(short)), codec.numeric_arrays(codec.state_sidecar_payload(longer))
    assert set(a) == set(b)
    assert len(a) == expected
    assert a["data.accepted_states.n_m3"].shape == (2, 4)
    assert b["data.accepted_states.n_m3"].shape == (20, 4)
    path = tmp_path / "StateArraysV1.npz"
    codec.write_numeric_sidecar(path, short)
    assert codec.verify_numeric_sidecar(path, short)["exact_key_coverage"]


@pytest.mark.parametrize("mutation", ["missing_low", "extra", "shape", "nonfinite", "changed_low", "float32", "signed_zero"])
def test_sidecar_rejects_resealed_numeric_mutations(tmp_path, mutation):
    record = rows()
    path = tmp_path / "StateArraysV1.npz"
    codec.write_numeric_sidecar(path, record)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    name = "data.accepted_states.precision_n_m3_lo"
    if mutation == "missing_low":
        arrays.pop(name)
    elif mutation == "extra":
        arrays["data.unclassified"] = np.array([0.])
    elif mutation == "shape":
        arrays[name] = arrays[name][:1]
    elif mutation == "nonfinite":
        arrays[name][0, 0] = np.nan
    elif mutation == "float32":
        arrays[name] = arrays[name].astype(np.float32)
    elif mutation == "signed_zero":
        arrays[name][0, 0] = -0.0
    else:
        arrays[name][0, 0] = 1e-30
    np.savez_compressed(path, **arrays)
    with pytest.raises(R1StateError):
        codec.verify_numeric_sidecar(path, record)


@pytest.mark.parametrize("schema,backend", [("R1CommonStatePairV1", "pair"),
                                          ("R1CommonStatePairV2", "legacy")])
def test_history_and_production_decoders_do_not_silently_upgrade(schema, backend):
    record = {"schema": schema, "representation": codec.REPRESENTATION}
    record["sha256"] = digest(record)
    with pytest.raises(R1StateError):
        get_backend(backend).decode_prepared(record)


def test_stage_producer_declares_revision_by_explicit_representation():
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_evidence import _producer_evidence_revision
    path = Path(__file__).resolve().parents[3] / "scripts/run_one_dimensional_mechanism_r1_stage_one.py"
    assert _producer_evidence_revision(path.read_bytes()) == 6
    assert _producer_evidence_revision(path.read_bytes(), codec.REPRESENTATION) == 7


def test_finite_failure_sidecar_contains_previous_and_attempted_low_words(tmp_path):
    record = {"schema": codec.STEP_SCHEMA, "accepted_steps": rows(),
              "failure": {"numerical_evidence": {"previous": snapshot(), "attempted": snapshot()}}}
    path = tmp_path / "FailedResultV1.npz"
    codec.write_numeric_sidecar(path, record)
    codec.verify_numeric_sidecar(path, record)
    with np.load(path, allow_pickle=False) as arrays:
        for phase in ("previous", "attempted"):
            for field in codec.FINE_FIELDS:
                assert "data.failure.numerical_evidence." + phase + ".precision_" + field + "_lo" in arrays


def test_initial_scalar_pair_words_have_bitwise_float64_sidecars(tmp_path):
    record = {"schema": codec.STEP_SCHEMA, "accepted_steps": rows(),
              "initial_event": {"trace_residual": {"hi": 1., "lo": 1e-30}, "ordinary_scalar": 3.}}
    path = tmp_path / "StateArraysV1.npz"
    codec.write_numeric_sidecar(path, record)
    codec.verify_numeric_sidecar(path, record)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    assert "data.initial_event.ordinary_scalar" not in arrays
    for word in ("hi", "lo"):
        key = "data.initial_event.trace_residual." + word
        assert arrays[key].shape == () and arrays[key].dtype == np.dtype("float64")
    arrays.pop("data.initial_event.trace_residual.lo")
    np.savez_compressed(path, **arrays)
    with pytest.raises(R1StateError, match="key coverage"):
        codec.verify_numeric_sidecar(path, record)


def test_failed_fine_preparation_retains_full_state_seed_and_original_gates(monkeypatch):
    from types import SimpleNamespace
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import R1PreparedState
    legacy = {name: value for name, value in snapshot().items() if name in codec.LEGACY_SNAPSHOT_FIELDS}
    seed_record = {"schema": "R1CommonStateV1", "source": {}, "state": legacy,
                   "qf_references_V": {}, "preparation_policy": {}, "preparation_checks": {}, "verification": {}}
    seed_record["sha256"] = digest(seed_record)
    seed = R1PreparedState.from_dict(seed_record)
    system = SimpleNamespace(grid=np.arange(4.), controls={"nu_I": 1, "nu_t": 1},
                             qfn_reference=np.ones(4), qfp_reference=np.ones(4))
    checks = {"certified": False, "reasons": ["deliberate_gate_failure"], "measured": 2.}
    monkeypatch.setattr(codec.states, "execution_source", lambda: {})
    monkeypatch.setattr(codec.states, "_preparation_policy", lambda value: None)
    monkeypatch.setattr(codec.states, "equilibrium_checks", lambda *args: checks)
    with pytest.raises(R1StateError) as error:
        codec.prepare_pair_state(None, 16, {}, legacy_prepare=lambda *args, **kwargs: seed,
            legacy_verify=lambda *args, **kwargs: (system, None),
            fine_factory=lambda base, before: (base, None), snapshot=lambda *args: snapshot())
    payload = error.value.result
    assert payload["schema"] == codec.FAILED_PREPARATION_SCHEMA
    assert payload["certified"] is False
    assert payload["raw_preparation"]["preparation_checks"] == checks
    assert payload["raw_preparation"]["seed_preparation"] == seed_record
    codec.R1CommonStatePairV2.from_dict(payload["raw_preparation"])


def test_nonfinite_failure_keeps_original_error_and_lossless_prefix(tmp_path, monkeypatch):
    import importlib.util
    project = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(project / "scripts"))
    spec = importlib.util.spec_from_file_location("pair_failure_runner", project / "scripts/run_one_dimensional_mechanism_r1_stage_one.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(codec.states, "execution_source", lambda: {"source": "unit"})
    record = {"accepted_steps": rows(), "certificate": {"certified": False, "reasons": ["original_failure"]},
              "failure": {"type": "OriginalFailure", "message": "must survive finalization",
                          "numerical_evidence": {"attempted": np.asarray([np.nan, np.inf])}}}
    finalized = codec.finalize_record(record)
    assert finalized["failure"]["type"] == "OriginalFailure"
    assert finalized["certificate"]["certified"] is False
    assert finalized["certificate"]["finite_numeric_evidence"]["passed"] is False
    assert "sha256" not in finalized
    runner._record_failure_result(tmp_path, finalized)
    saved = runner.read_json(tmp_path / "FailedResultV1.json")
    assert saved["failure"]["numerical_evidence"]["attempted"] == [{"nonfinite": "nan"}, {"nonfinite": "inf"}]
    sidecar = tmp_path / "FailedResultV1.npz"
    assert codec.verify_failed_numeric_sidecar(sidecar, saved)["exact_key_coverage"]
    with np.load(sidecar, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    key = "data.accepted_steps.0.state.precision_n_m3_lo"
    assert key in arrays
    arrays.pop(key)
    np.savez_compressed(sidecar, **arrays)
    with pytest.raises(R1StateError, match="key coverage"):
        codec.verify_failed_numeric_sidecar(sidecar, saved)
