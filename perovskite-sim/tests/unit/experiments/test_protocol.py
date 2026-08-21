from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from perovskite_sim.experiments.eqe import (
    build_eqe_experiment_protocol,
    compute_eqe,
)
from perovskite_sim.experiments.impedance import (
    build_impedance_experiment_protocol,
    run_impedance,
)
from perovskite_sim.experiments.jv_sweep import (
    build_jv_experiment_protocol,
    run_jv_sweep,
)
from perovskite_sim.experiments.protocol import (
    ACExcitation,
    DCSettleCriterion,
    ExperimentProtocol,
    IlluminationStep,
    ImplicitProtocolError,
    ProtocolMismatchError,
    SamplingProtocol,
    ScanProtocol,
    VocSearchProtocol,
    resolve_experiment_protocol,
)
from perovskite_sim.experiments.suns_voc import (
    build_suns_voc_experiment_protocol,
    run_suns_voc,
)
from perovskite_sim.experiments.tpv import (
    _find_voc,
    build_tpv_experiment_protocol,
    run_tpv,
)
from perovskite_sim.models.config_loader import load_device_from_yaml


@pytest.fixture(scope="module")
def stack():
    return load_device_from_yaml("configs/nip_MAPbI3.yaml")


@pytest.fixture(scope="module")
def tmm_stack():
    return load_device_from_yaml("configs/nip_MAPbI3_tmm.yaml")


@pytest.fixture
def legacy_protocol(stack) -> ExperimentProtocol:
    return build_jv_experiment_protocol(
        stack,
        n_points=3,
        V_max=1.0,
        implicit_legacy_protocol=True,
    )


def test_protocol_round_trip_preserves_canonical_hash(legacy_protocol):
    encoded = legacy_protocol.canonical_json()
    restored = ExperimentProtocol.from_json(encoded)

    assert restored == legacy_protocol
    assert restored.protocol_hash == legacy_protocol.protocol_hash
    assert restored.canonical_json() == encoded
    assert restored.to_json() == encoded
    assert restored.sha256 == restored.protocol_hash
    assert json.loads(encoded) == legacy_protocol.to_dict()


def test_mapping_order_does_not_change_canonical_hash(legacy_protocol):
    payload = legacy_protocol.to_dict()
    reversed_payload = dict(reversed(tuple(payload.items())))

    restored = ExperimentProtocol.from_dict(reversed_payload)

    assert restored.protocol_hash == legacy_protocol.protocol_hash


def test_protocol_and_nested_sampling_are_immutable(legacy_protocol):
    with pytest.raises(dataclasses.FrozenInstanceError):
        legacy_protocol.temperature_K = 350.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        legacy_protocol.sampling.values = (0.0,)
    assert isinstance(legacy_protocol.illumination_history, tuple)
    assert isinstance(legacy_protocol.sampling.values, tuple)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SamplingProtocol("time_s", "declared", (0.0, np.nan)),
        lambda: ScanProtocol("time_s", "forward_time", 0.0, np.inf),
        lambda: ACExcitation(0.5, np.inf),
        lambda: DCSettleCriterion("finite_time", duration_s=np.nan),
        lambda: IlluminationStep("probe", "baseline", intensity_suns=np.inf),
        lambda: VocSearchProtocol(coarse_dwell_s=np.nan),
    ],
)
def test_protocol_rejects_nan_and_infinity(factory):
    with pytest.raises(ValueError, match="finite"):
        factory()


def test_research_strict_rejects_implicit_history(legacy_protocol):
    with pytest.raises(ImplicitProtocolError, match="implicit_legacy_protocol"):
        resolve_experiment_protocol(
            None,
            legacy_protocol,
            mode="research_strict",
        )


def test_explicit_protocol_must_match_execution(legacy_protocol):
    explicit = legacy_protocol.as_explicit()
    assert resolve_experiment_protocol(
        explicit,
        legacy_protocol,
        mode="research_strict",
    ) is explicit

    mismatched = dataclasses.replace(explicit, temperature_K=310.0)
    with pytest.raises(ProtocolMismatchError, match="temperature_K"):
        resolve_experiment_protocol(
            mismatched,
            legacy_protocol,
            mode="research_strict",
        )


def test_unknown_schema_keys_fail_closed(legacy_protocol):
    payload = legacy_protocol.to_dict()
    payload["unregistered_history"] = "hidden"

    with pytest.raises(ValueError, match="extra"):
        ExperimentProtocol.from_dict(payload)


def test_jv_fixed_generation_is_part_of_protocol_hash(stack):
    first = build_jv_experiment_protocol(
        stack,
        n_points=3,
        V_max=1.0,
        fixed_generation=np.array([1.0, 2.0, 3.0]),
    )
    second = build_jv_experiment_protocol(
        stack,
        n_points=3,
        V_max=1.0,
        fixed_generation=np.array([1.0, 2.0, 4.0]),
    )

    assert first.protocol_hash != second.protocol_hash
    assert first.illumination_history[0].source_reference.startswith("sha256:")


def test_builders_cover_all_phase_1_experiment_fields(stack, tmm_stack):
    wavelengths = np.array([450.0, 650.0])
    frequencies = np.array([1.0, 10.0, 100.0])
    protocols = (
        build_jv_experiment_protocol(stack, n_points=3, V_max=1.0),
        build_tpv_experiment_protocol(stack, n_points=20),
        build_eqe_experiment_protocol(tmm_stack, wavelengths),
        build_suns_voc_experiment_protocol(stack, (0.1, 1.0)),
        build_impedance_experiment_protocol(
            stack,
            frequencies,
            V_dc=0.8,
            delta_V=5e-3,
            n_cycles=6,
            n_extract=3,
            points_per_cycle=80,
        ),
    )

    assert tuple(protocol.experiment for protocol in protocols) == (
        "jv_hysteresis",
        "tpv",
        "eqe",
        "suns_voc",
        "impedance",
    )
    for protocol in protocols:
        assert protocol.initial_state_source
        assert protocol.illumination_history
        assert protocol.temperature_K == pytest.approx(300.0)
        assert protocol.scan is not None
        assert protocol.dc_settle is not None
        assert protocol.sampling.values
        assert not protocol.implicit_legacy_protocol
    assert protocols[0].scan.rate_V_s == pytest.approx(0.1)
    scan_history = next(
        step
        for step in protocols[0].illumination_history
        if step.phase == "forward_reverse_scan"
    )
    assert scan_history.duration_s == pytest.approx(
        2.0 * 3 * protocols[0].dwell_duration_s
    )
    assert protocols[1].soak_duration_s == pytest.approx(1e-3)
    assert protocols[1].voc_search == VocSearchProtocol()
    assert protocols[2].dwell_duration_s == pytest.approx(1e-1)
    assert protocols[3].sampling.values == (0.1, 1.0)
    assert protocols[3].voc_search == VocSearchProtocol(minimum_guess_V=0.3)
    assert protocols[4].ac_excitation == ACExcitation(
        dc_bias_V=0.8,
        amplitude_V=5e-3,
        cycles=6,
        extraction_cycles=3,
        points_per_cycle=80,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("coarse_start_V", 0.01),
        ("coarse_upper_guess_factor", 1.6),
        ("minimum_guess_V", 0.2),
        ("coarse_points", 21),
        ("coarse_dwell_s", 2.0e-4),
        ("bisection_tolerance_V", 2.0e-3),
        ("bisection_max_steps", 16),
        ("bisection_dwell_s", 2.0e-4),
        ("final_settle_s", 2.0e-4),
    ],
)
def test_voc_search_fields_are_canonical_and_change_protocol_hash(
    stack, field, value
):
    base_search = VocSearchProtocol()
    changed_search = dataclasses.replace(base_search, **{field: value})
    base = build_tpv_experiment_protocol(
        stack, n_points=20, voc_search=base_search
    )
    changed = build_tpv_experiment_protocol(
        stack, n_points=20, voc_search=changed_search
    )

    assert changed.protocol_hash != base.protocol_hash
    assert ExperimentProtocol.from_json(changed.canonical_json()) == changed


def test_find_voc_executes_declared_search_dwell_and_counts(monkeypatch, stack):
    calls: list[tuple[float, float, float]] = []

    def fake_integrate(_x, state, _stack, _mat, voltage, t_lo, t_hi, *_args):
        calls.append((float(voltage), float(t_lo), float(t_hi)))
        return state.copy()

    monkeypatch.setattr(
        "perovskite_sim.experiments.tpv._integrate_step", fake_integrate
    )
    monkeypatch.setattr(
        "perovskite_sim.experiments.tpv._compute_current",
        lambda _x, _state, _stack, voltage, **_kwargs: 1.0 - float(voltage),
    )
    search = VocSearchProtocol(
        coarse_upper_guess_factor=1.5,
        coarse_points=4,
        coarse_dwell_s=0.02,
        bisection_tolerance_V=0.2,
        bisection_max_steps=2,
        bisection_dwell_s=0.03,
        final_settle_s=0.04,
    )

    _find_voc(
        np.array([0.0, 0.5, 1.0]),
        np.ones(9),
        stack,
        object(),
        V_guess=1.0,
        search=search,
    )

    assert len(calls) == (
        search.coarse_points + search.bisection_max_steps + 1
    )
    assert all(
        t_hi - t_lo == pytest.approx(search.coarse_dwell_s)
        for _voltage, t_lo, t_hi in calls[: search.coarse_points]
    )
    assert calls[-2][2] - calls[-2][1] == pytest.approx(
        search.bisection_dwell_s
    )
    assert calls[-1][2] - calls[-1][1] == pytest.approx(search.final_settle_s)


def test_legacy_builders_mark_implicit_history(stack, tmm_stack):
    protocols = (
        build_jv_experiment_protocol(
            stack, n_points=3, implicit_legacy_protocol=True
        ),
        build_tpv_experiment_protocol(
            stack, n_points=20, implicit_legacy_protocol=True
        ),
        build_eqe_experiment_protocol(
            tmm_stack,
            np.array([500.0]),
            implicit_legacy_protocol=True,
        ),
        build_suns_voc_experiment_protocol(
            stack, (1.0,), implicit_legacy_protocol=True
        ),
        build_impedance_experiment_protocol(
            stack,
            np.array([1e3]),
            implicit_legacy_protocol=True,
        ),
    )

    assert all(protocol.implicit_legacy_protocol for protocol in protocols)


@pytest.mark.parametrize("experiment", ["jv", "tpv", "eqe", "suns_voc", "impedance"])
def test_public_experiments_reject_implicit_history_before_solving(
    experiment,
    stack,
    tmm_stack,
):
    calls = {
        "jv": lambda: run_jv_sweep(
            stack, N_grid=6, n_points=3, protocol_mode="research_strict"
        ),
        "tpv": lambda: run_tpv(
            stack, N_grid=6, n_points=20, protocol_mode="research_strict"
        ),
        "eqe": lambda: compute_eqe(
            tmm_stack,
            wavelengths_nm=np.array([500.0]),
            N_grid=6,
            protocol_mode="research_strict",
        ),
        "suns_voc": lambda: run_suns_voc(
            stack,
            suns_levels=(1.0,),
            N_grid=6,
            protocol_mode="research_strict",
        ),
        "impedance": lambda: run_impedance(
            stack,
            np.array([1e3]),
            N_grid=6,
            protocol_mode="research_strict",
        ),
    }

    with pytest.raises(ImplicitProtocolError):
        calls[experiment]()


def test_default_and_explicit_protocols_describe_identical_execution(stack):
    legacy = build_jv_experiment_protocol(
        stack,
        v_rate=0.2,
        n_points=4,
        V_max=1.1,
        implicit_legacy_protocol=True,
    )
    explicit = build_jv_experiment_protocol(
        stack,
        v_rate=0.2,
        n_points=4,
        V_max=1.1,
    )

    resolved = resolve_experiment_protocol(
        explicit,
        legacy,
        mode="research_strict",
    )

    legacy_execution = legacy.to_dict()
    explicit_execution = resolved.to_dict()
    legacy_execution.pop("implicit_legacy_protocol")
    explicit_execution.pop("implicit_legacy_protocol")
    assert explicit_execution == legacy_execution
