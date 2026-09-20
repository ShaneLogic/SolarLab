"""Reconstruct execution settings from pinned input and compare to saved data."""
from __future__ import annotations

import math

from perovskite_sim.experiments.one_dimensional_mechanism_r1_evidence import read_json, _canonical
from perovskite_sim.experiments.one_dimensional_mechanism_r1_convergence import observation_times
from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import r1_policy
from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import json_data

NUMERICAL_SCOPE = "single setting; no three-axis convergence, amplitude linearity or dual-domain acceptance asserted"


def _same(actual, expected, name):
    if _canonical(actual) != _canonical(expected):
        raise ValueError("execution parameter mismatch: " + name)


def verify_execution_parameters(output, completion):
    protocol = read_json(output / "ProtocolV1.json")
    representation = completion.get("representation", "float64-baseline")
    _same(protocol.get("representation", "float64-baseline"), representation, "numerical representation")
    study = read_json(output / "StudyInputV1.json")
    additional = read_json(output / "AdditionalFailuresV1.json")
    additional_v2 = read_json(output / "AdditionalFailuresV2.json")
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_evidence import sha256
    _same(protocol.get("additional_failures_v2_sha256"), sha256(output / "AdditionalFailuresV2.json"), "additional failure registry V2")
    interval = protocol.get("intervals")
    if type(interval) is not int or interval not in (16, 32, 64, 128, 256):
        raise ValueError("execution intervals outside declared R1 set")
    _same(protocol.get("control_definitions"), study["controls"], "control definitions")
    _same(protocol.get("claims_excluded"), study["claims_excluded"], "excluded claims")
    _same(protocol.get("numerical_validation_scope"), NUMERICAL_SCOPE, "numerical scope")
    count = sum(len(study.get(k, [])) + len(additional.get(k, [])) + len(additional_v2.get(k, []))
                for k in ("known_nonconvergence", "known_physical_gate_failures"))
    _same(protocol.get("historical_failure_case_count"), count, "historical failure count")
    factor = protocol.get("nonlinear_factor")
    if type(factor) not in (int, float) or not math.isfinite(factor):
        raise ValueError("invalid nonlinear factor")
    substeps = protocol.get("time_substeps")
    policy = r1_policy(factor, time_substeps=substeps)
    _same(protocol.get("policy"), json_data(policy), "numerical policy")
    control = protocol.get("control")
    stage = completion["stage"]
    if (not isinstance(control, str) or not control or len(set(control)) != len(control)
            or any(c not in "ABCD" for c in control)
            or (stage == "step" and len(control) != 1)
            or (stage == "prepare" and control != "D")):
        raise ValueError("invalid executed control selection")
    extended = interval > 64 or tuple(substeps) != (1, 2, 4)
    if stage == "step":
        amplitude = protocol.get("amplitude_V")
        if type(amplitude) not in (float, int) or not math.isfinite(amplitude) or not 0 < amplitude < .02:
            raise ValueError("invalid declared voltage step")
        window = protocol.get("observation_window")
        if not isinstance(window, dict):
            raise ValueError("missing observation window")
        if window.get("kind") == "functional":
            times = study["functional_times_s"]
            spacing = None
        elif window.get("kind") == "full":
            times = observation_times(first_time_s=window.get("first_positive_time_s"),
                                      last_time_s=window.get("last_time_s")).tolist()
            spacing = 12
            extended = True
        elif window.get("kind") == "diagnostic":
            if representation != "float64-pair-v1" or interval != 16 or tuple(substeps) != (1, 2, 4):
                raise ValueError("diagnostic window requires the fixed N16 production pair short chain")
            times, spacing, extended = [0.0, 1e-9], None, True
        else:
            raise ValueError("unknown observation window")
        expected_window = {"kind": window["kind"], "first_positive_time_s": times[1],
                           "last_time_s": times[-1], "output_point_count": len(times),
                           "logarithmic_intervals_per_decade": spacing}
        _same(window, expected_window, "observation window")
        _same(protocol.get("times_s"), times, "actual observation times")
        if protocol.get("physics_evidence") is not True:
            raise ValueError("current physical acceptance requires saved state-equation evidence")
    else:
        times = None
        _same(protocol.get("observation_window"), None, "non-transient window")
        _same(protocol.get("times_s"), None, "non-transient times")
        _same(protocol.get("amplitude_V"), 0.0, "non-transient amplitude")
        _same(substeps, [1, 2, 4], "non-transient time setting")
    scope = ("R1-2-physics" if completion.get("run_class") == "formal" else "R1-2-development") if extended else "R1-1"
    _same(protocol.get("stage_scope"), scope, "execution scope")
    _same(completion.get("stage_scope"), scope, "completion scope")
    prepared_path = output / "PreparedStateV1.json"
    if prepared_path.exists():
        prepared = read_json(prepared_path)
        _same(read_json(output / "ResolvedStackV1.json"), prepared.get("physical_stack"), "resolved material stack")
    return {"intervals": interval, "time_substeps": list(substeps), "times_s": times,
            "control": control, "nonlinear_factor": factor, "scope": scope,
            "representation": representation,
            "diagnostic_only": stage == "step" and protocol["observation_window"]["kind"] == "diagnostic",
            "convergence_claimed": False, "mechanism_identification_claimed": False}
