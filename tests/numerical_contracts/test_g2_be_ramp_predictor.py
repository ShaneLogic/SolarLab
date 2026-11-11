"""Regression contracts for boundary-consistent BE voltage-ramp prediction on B.

Constructs DynamicAcceptorIonPublicDeviceV1 (B, intervals=8) strictly from the
portable qualification_origins.json input fixture and public benchmark modules,
without external archive paths or captured floating-point output oracles.
"""

from __future__ import annotations

from dataclasses import asdict
from fractions import Fraction
import json
import math
from pathlib import Path

import numpy as np
import pytest

from perovskite_sim.constants import Q
from scripts.benchmarks import coupled_device_prototype as cdp
from scripts.benchmarks import g2_reference_be as be
from scripts.benchmarks.conservative_device import EquationBudget, conservative_be_step
from scripts.benchmarks.contract_prototype import ContractError, PhysicalScaling, SECOND
from scripts.benchmarks.precision_prototype import (
    FrameInputExpansion,
    PrimitiveExpansion,
    encode_point,
)

FIXTURES = Path(__file__).with_name("fixtures")
B_ORIGIN = json.loads((FIXTURES / "qualification_origins.json").read_text())["B"]


def _round_down_positive(value: Fraction) -> float:
    approx = float(value)
    if Fraction(approx) > value:
        approx = math.nextafter(approx, 0.0)
    assert approx > 0.0 and Fraction(approx) <= value
    return approx


def _build_b_probe_context():
    definition = cdp.SlabDefinition(
        source_path="portable-qualification-origin",
        **B_ORIGIN["definition"],
    )
    model = cdp.AffineCoupledSlab(definition, 8)
    mapping = cdp.AffineVoltageMap(
        model, mapped_input_profile="frame-input-expansion12-v1"
    )
    all_segments = tuple(cdp.ProtocolSegment(**s) for s in B_ORIGIN["segments"])
    probe_segments = all_segments[:2]
    request_stub = {
        "segments": B_ORIGIN["segments"],
        "observation_times": B_ORIGIN["observation_times"],
    }
    probe_tags = be.tags_from_request(request_stub, 2)

    zero_handoff = FrameInputExpansion(
        tuple(np.zeros(model.layout.size, dtype=float) for _ in range(12))
    )
    inputs0, _ = probe_segments[0].inputs(probe_segments[0].start)
    initial, _ = model.coordinates.trial(
        model.reference,
        zero_handoff,
        probe_segments[0].start,
        inputs0,
        predecessor=model.reference,
    )

    # Exact frozen CommanderBEConfigurationV1 formulas (tau = alpha * r_strict / 64).
    tau = Fraction(1e-10) * Fraction(1e-6) / 64
    q_ref = (
        Fraction(Q)
        * Fraction(definition.area)
        * Fraction(definition.length)
        * Fraction(definition.ion_initial)
    )
    vt = float(mapping.columns[-1])
    vols = [Fraction(float(v)) for v in model.geometry.volumes]
    budget_Ceiling_rows = {
        "a_n": [
            _round_down_positive(tau * vols[i] * Fraction(definition.n_eq))
            for i in range(1, 8)
        ],
        "b_p": [
            _round_down_positive(tau * vols[i] * Fraction(definition.p_eq))
            for i in range(1, 8)
        ],
        "c_ion": [
            _round_down_positive(tau * vols[i] * Fraction(definition.ion_initial))
            for i in range(9)
        ],
        "d_trap": [
            _round_down_positive(tau * vols[i] * Fraction(definition.trap_density))
            for i in range(9)
        ],
        "x_poisson": [_round_down_positive(tau * q_ref / 7)] * 7,
        "y_phi_contact": [_round_down_positive(tau * Fraction(vt))] * 2,
        "z_n_contact": [_round_down_positive(tau * Fraction(definition.n_eq))] * 2,
        "zz_p_contact": [_round_down_positive(tau * Fraction(definition.p_eq))] * 2,
    }
    budgets = {}
    equation_budgets_cfg = {}
    for eq in model.layout.equations:
        unit = eq.unit * SECOND if eq.role == "storage" else eq.unit
        values = budget_Ceiling_rows[eq.id]
        budgets[eq.id] = EquationBudget(unit, values)
        equation_budgets_cfg[eq.id] = {
            "unit": json.loads(json.dumps(asdict(unit))),
            "absolute_hex": [float(x).hex() for x in values],
        }

    pkt = model.numeric_packet()
    eq_scaling = (
        [float(vols[i] * Fraction(definition.ni)) for i in range(1, 8)]
        + [float(vols[i] * Fraction(definition.ni)) for i in range(1, 8)]
        + [float(vols[i] * Fraction(definition.ion_initial)) for i in range(9)]
        + [float(vols[i] * Fraction(definition.trap_density)) for i in range(9)]
        + [float(Fraction(1) / Fraction(r)) for r in pkt["row_scaling"][32:]]
    )
    state_scaling = [float(v) for v in mapping.columns]
    scaling = PhysicalScaling(eq_scaling, state_scaling)

    def feasible(point):
        values = be._physical_exact(point)
        for spec in model.layout.variables:
            row = values[spec.id]
            if spec.id in {"n_m3", "p_m3"} and min(row) <= 0:
                return False
            if spec.lower is not None and min(row) < Fraction(spec.lower):
                return False
            if spec.upper is not None and max(row) > Fraction(spec.upper):
                return False
            if spec.id == "c_m3" and max(row) >= Fraction(definition.ion_capacity):
                return False
        return True

    def checks(point):
        result = cdp.affine_state_quality_evidence(model, point, B_ORIGIN["budgets"])
        return {
            "safety_pass": feasible(point),
            "scientific_failed": result.get("passed") is False,
            "same_state_evidence": result,
        }

    lift_hex = [[float(v).hex() for v in row] for row in mapping.lift]
    predictor_spec = {"policy": "affine-input-lift-v1", "lift_hex": lift_hex}
    return {
        "model": model,
        "mapping": mapping,
        "all_segments": all_segments,
        "probe_segments": probe_segments,
        "probe_tags": probe_tags,
        "initial": initial,
        "budgets": budgets,
        "equation_budgets_cfg": equation_budgets_cfg,
        "scaling": scaling,
        "eq_scaling": eq_scaling,
        "state_scaling": state_scaling,
        "feasible": feasible,
        "checks": checks,
        "predictor_spec": predictor_spec,
    }


def _lightweight_b_backend(ctx: dict, *, initial_predictor: dict | None) -> be.Backend:
    """Fast deterministic observer wrapper for multi-hundred-step BE convergence tests."""
    model = ctx["model"]

    def enclosure(val: Fraction) -> dict:
        return {"center": be._pair(val), "radius": [0, 1]}

    def observe_point(point, segment):
        charges = [
            sum(
                Fraction(float(w[0]))
                for w in model.linear_action(name, point).value.words
            )
            for name in ("body_charge",)
        ]
        metal = model.linear_action("metal_charge", point).value
        m_left = sum(Fraction(float(w[0])) for w in metal.words)
        m_right = sum(Fraction(float(w[1])) for w in metal.words)
        return {
            "ports": {
                "left": enclosure(Fraction(0)),
                "right": enclosure(Fraction(0)),
                "unit": "A",
            },
            "charges": {
                "device": enclosure(charges[0]),
                "left_metal": enclosure(m_left),
                "right_metal": enclosure(m_right),
                "unit": "C",
            },
            "observation_uncertainty": "exact affine charge action",
            "derivative_origin": "same_state_affine_tangent",
            "evidence_status": "represented_point_only",
        }

    def observe_interval(left, right, step, segment, target):
        return {
            "physical_flux": [enclosure(Fraction(0)) for _ in range(3)],
            "unit": "C",
            "storage_secant_used_as_flux": False,
            "origin": "declared_reconstruction",
            "reference_error_status": "unknown_reference_error",
        }

    return be.Backend(
        model.public_problem(),
        model.coordinates,
        ctx["initial"],
        ctx["scaling"],
        ctx["budgets"],
        ctx["probe_segments"],
        ctx["feasible"],
        ctx["checks"],
        observe_point,
        observe_interval,
        {
            "kind": "portable_B_probe",
            "source_identity": model.source_identity,
            "observation_controls": {"kind": "lightweight_b_probe"},
        },
        initial_predictor=initial_predictor,
    )


def test_b_small_voltage_step_solution_invariance_and_contact_lift():
    ctx = _build_b_probe_context()
    model = ctx["model"]
    problem = model.public_problem()
    left = ctx["initial"]
    h = 2.0**-13
    # Non-dyadic voltage step small enough that the unlifted trial also converges.
    right_inputs = (3e-4, 0.0)

    lifted_inc = be._predict_initial_increment(
        ctx["predictor_spec"], model.layout.size, left.inputs, right_inputs
    )
    zero_inc = PrimitiveExpansion.from_value(np.zeros(model.layout.size))

    # 1. Verify exact boundary lift and unchanged positive/bounded fields at trial 0.
    zero_trial, zero_step_inc = model.coordinates.trial(
        model.reference,
        FrameInputExpansion(tuple(np.zeros(model.layout.size) for _ in range(12))).add(
            zero_inc
        ),
        left.time + h,
        right_inputs,
        predecessor=left,
    )
    lifted_trial, lifted_step_inc = model.coordinates.trial(
        model.reference,
        FrameInputExpansion(tuple(np.zeros(model.layout.size) for _ in range(12))).add(
            lifted_inc
        ),
        left.time + h,
        right_inputs,
        predecessor=left,
    )
    assert ctx["feasible"](lifted_trial) is True
    before_exact = be._physical_exact(left)
    lifted_exact = be._physical_exact(lifted_trial)
    for field in ("n_m3", "p_m3", "c_m3", "f"):
        assert lifted_exact[field] == before_exact[field]
    assert lifted_exact["phi_V"][0] == Fraction(0)
    assert lifted_exact["phi_V"][-1] + Fraction(right_inputs[0]) == Fraction(0)

    res_zero_0 = problem.conservative_residual(left, zero_trial, zero_step_inc)
    res_lift_0 = problem.conservative_residual(left, lifted_trial, lifted_step_inc)
    phi_contact_offset = model.graph.row_offsets["y_phi_contact"]
    zero_contact_res = [
        sum(Fraction(float(w[i])) for w in res_zero_0.words)
        for i in range(phi_contact_offset.start, phi_contact_offset.stop)
    ]
    lift_contact_res = [
        sum(Fraction(float(w[i])) for w in res_lift_0.words)
        for i in range(phi_contact_offset.start, phi_contact_offset.stop)
    ]
    contact_budget = Fraction(ctx["budgets"]["y_phi_contact"].absolute[1])
    assert abs(zero_contact_res[1]) / contact_budget > 10**15
    assert lift_contact_res == [Fraction(0), Fraction(0)]

    # 2. Both predictors solve the identical BE equation and converge to the same state.
    step_unlifted = conservative_be_step(
        problem,
        model.coordinates,
        left,
        h,
        scaling=ctx["scaling"],
        budgets=ctx["budgets"],
        inputs=right_inputs,
        initial_increment=None,
        feasible=ctx["feasible"],
        **be.STEP_OPTIONS,
    )
    step_lifted = conservative_be_step(
        problem,
        model.coordinates,
        left,
        h,
        scaling=ctx["scaling"],
        budgets=ctx["budgets"],
        inputs=right_inputs,
        initial_increment=lifted_inc,
        feasible=ctx["feasible"],
        **be.STEP_OPTIONS,
    )
    assert max(step_unlifted.residual_ratios) <= 1
    assert max(step_lifted.residual_ratios) <= 1
    assert len(step_lifted.iterations) <= len(step_unlifted.iterations)

    state_u = be._physical_exact(step_unlifted.right)
    state_l = be._physical_exact(step_lifted.right)
    for spec in model.layout.variables:
        offset = model.layout.offsets[spec.id]
        for j, col in enumerate(range(offset.start, offset.stop)):
            scale = Fraction(ctx["state_scaling"][col])
            scaled_diff = abs(state_u[spec.id][j] - state_l[spec.id][j]) / scale
            assert scaled_diff < Fraction(1, 10**18), (spec.id, j, float(scaled_diff))


def test_prepare_b_backend_wires_affine_lift_predictor_and_real_slab_observer(
    tmp_path, monkeypatch
):
    pytest.importorskip("flint")
    ctx = _build_b_probe_context()
    model = ctx["model"]
    mapping = ctx["mapping"]

    t1 = 2.0**-8
    t2 = 2.0**-7
    t3 = 3.0 * 2.0**-8
    v_end = 5.0 * (t2 - t1)
    short_segments = (
        cdp.ProtocolSegment("dark_equilibrium_hold", 0.0, t1, (0.0, 0.0), (0.0, 0.0)),
        cdp.ProtocolSegment("voltage_ramp", t1, t2, (0.0, v_end), (0.0, 0.0)),
        cdp.ProtocolSegment("biased_light_soak", t2, t3, (v_end, v_end), (0.0, 0.0)),
    )
    pkt = model.numeric_packet()
    pkt["initial_reference"] = encode_point(model.reference)
    pkt["definition_identity"] = model.definition.identity
    pkt["layout_identity"] = model.layout.identity
    request = {
        "case_id": "DynamicAcceptorIonPublicDeviceV1",
        "numeric_packet": pkt,
        "voltage_lift_map": mapping.payload(),
        "segments": [asdict(s) for s in short_segments],
        "observation_times": {
            "dark_equilibrium_hold": [0.0, t1],
            "voltage_ramp": [t1, t2],
            "biased_light_soak": [t2, t3],
        },
        "initial_segment_frame": {
            "physical_handoff_words_hex": [
                [(0.0).hex()] * model.layout.size for _ in range(12)
            ]
        },
        "initial_preparation": {"point_identity": ctx["initial"].identity},
        "budgets": B_ORIGIN["budgets"],
    }
    configuration = {
        "scientific_execution_authorized": True,
        "source_review_accepted": True,
        "method_status": "frozen",
        "commander_admission_identity": "test-admission-pin",
        "segment_count": 2,
        "equation_budgets": ctx["equation_budgets_cfg"],
        "equation_scaling": ctx["eq_scaling"],
        "state_scaling": ctx["state_scaling"],
        "observation_absolute_error_C": [1, 10**18],
    }
    # Bypass only the external IDA parent-request lineage check while running
    # the real prepare_B_backend, real SlabObserver, and real B kernel.
    monkeypatch.setattr(
        cdp, "validate_voltage_lift_native_request", lambda *a, **k: None
    )

    backend = be.prepare_B_backend(request, configuration)
    assert backend.initial_predictor == ctx["predictor_spec"]
    assert be.effective_controls(backend)["initial_predictor"] == ctx["predictor_spec"]

    tags = be.tags_from_request(request, 2)
    schedule = be.compile_schedule(
        backend.segments, tags, level=0, seed=Fraction(1, 512)
    )
    identity = be.RunIdentity(
        "B_short_real_observer",
        be.digest(request),
        "1" * 64,
        "2" * 64,
        "3" * 40,
        "test-env",
        schedule.identity,
        backend.problem.source_identity,
        backend.initial.identity,
        be.digest(be.effective_controls(backend)),
    )
    limits = be.Limits(120.0, output_bytes=64 * 1048576, logical_bytes=64 * 1048576)
    out_dir = tmp_path / "real_observer_run"
    result = be.run_reference(backend, schedule, identity, request, out_dir, limits)
    readback = be.read_reference_history(out_dir, backend.problem.layout, identity)
    assert result["numerical_complete"] is True
    assert readback["structural_pass"] is True
    assert readback["numerical_complete"] is True
    assert readback["accepted_steps"] == len(schedule.steps)


def test_b_short_window_l0_l1_l2_ramp_convergence_and_first_order_refinement(tmp_path):
    ctx = _build_b_probe_context()
    model = ctx["model"]
    request = {
        "case_id": "DynamicAcceptorIonPublicDeviceV1",
        "segments": B_ORIGIN["segments"],
        "observation_times": B_ORIGIN["observation_times"],
        "voltage_lift_map": {"lift_hex": ctx["predictor_spec"]["lift_hex"]},
    }
    limits = be.Limits(300.0, output_bytes=128 * 1048576, logical_bytes=128 * 1048576)

    # 1. Historical unlifted behavior remains reproduced when predictor is absent:
    # first voltage_ramp step (step 65 of L0) exhausts the 12-iteration budget.
    unlifted_backend = _lightweight_b_backend(ctx, initial_predictor=None)
    l0_schedule = be.compile_schedule(ctx["probe_segments"], ctx["probe_tags"], level=0)
    unlifted_id = be.RunIdentity(
        "B_Probe_Unlifted_L0",
        be.digest(request),
        "1" * 64,
        "2" * 64,
        "3" * 40,
        "test-env",
        l0_schedule.identity,
        unlifted_backend.problem.source_identity,
        unlifted_backend.initial.identity,
        be.digest(be.effective_controls(unlifted_backend)),
    )
    with pytest.raises(ContractError, match="conservative_iteration_budget_exhausted"):
        be.run_reference(
            unlifted_backend,
            l0_schedule,
            unlifted_id,
            request,
            tmp_path / "unlifted_l0",
            limits,
        )
    failure = json.loads((tmp_path / "unlifted_l0" / "FirstFailure.json").read_text())
    assert failure["location"]["target_step"]["index"] == 65
    assert failure["location"]["target_step"]["segment"] == "voltage_ramp"
    assert be.read_reference_history(
        tmp_path / "unlifted_l0", model.layout, unlifted_id
    )["structural_pass"]

    # 2. With affine-input-lift-v1, L0/L1/L2 (128/256/512 steps) over 0-0.2 s
    # converge cleanly within STEP_OPTIONS and pass readback + state checks.
    lifted_backend = _lightweight_b_backend(
        ctx, initial_predictor=ctx["predictor_spec"]
    )
    terminal_points = []
    expected_steps = (128, 256, 512)
    for level, step_count in enumerate(expected_steps):
        schedule = be.compile_schedule(
            ctx["probe_segments"], ctx["probe_tags"], level=level
        )
        assert len(schedule.steps) == step_count
        run_dir = tmp_path / f"lifted_l{level}"
        identity = be.RunIdentity(
            f"B_Probe_L{level}",
            be.digest(request),
            "1" * 64,
            "2" * 64,
            "3" * 40,
            "test-env",
            schedule.identity,
            lifted_backend.problem.source_identity,
            lifted_backend.initial.identity,
            be.digest(be.effective_controls(lifted_backend)),
        )
        result = be.run_reference(
            lifted_backend, schedule, identity, request, run_dir, limits
        )
        readback = be.read_reference_history(run_dir, model.layout, identity)
        assert result["numerical_complete"] is True
        assert result["scientific_status"] == "unknown_reference_error"
        assert readback["structural_pass"] is True
        assert readback["numerical_complete"] is True
        assert readback["accepted_steps"] == step_count
        assert readback["tagged_observations"] == 66

        rows = [
            json.loads(line)
            for line in (run_dir / "History/BE.jsonl").read_text().splitlines()
        ]
        step_rows = [r for r in rows if r["kind"] == "be_accepted_step"]
        max_iters = max(len(r["newton_iterations"]) for r in step_rows)
        min_damping = min(
            (it["damping"] for r in step_rows for it in r["newton_iterations"]),
            default=1.0,
        )
        if level == 0:
            assert max_iters <= 4 and min_damping >= 0.5
        else:
            assert max_iters <= 3 and min_damping == 1.0

        header_ref = rows[0]["original_reference"]
        terminal_points.append(
            be.unpack_point(step_rows[-1]["right"], header_ref, model.layout)
        )

    # 3. Verify asymptotic first-order temporal refinement (L0 -> L1 -> L2) at t = 0.2 s
    # across all dynamic state DOFs and terminal physical port/charge observables.
    states = [be._physical_exact(pt) for pt in terminal_points]
    dynamic_checked = 0
    invariant_checked = 0
    for field in ("n_m3", "p_m3", "phi_V", "c_m3", "f"):
        offset = model.layout.offsets[field]
        for idx in range(model.count):
            col = offset.start + idx
            scale = Fraction(ctx["state_scaling"][col])
            v0, v1, v2 = (s[field][idx] for s in states)
            d01, d12 = v0 - v1, v1 - v2
            if field in {"n_m3", "p_m3", "phi_V", "f"} and idx in (
                0,
                model.count - 1,
            ):
                assert abs(d01) / scale < Fraction(1, 10**24), (field, idx, d01)
                assert abs(d12) / scale < Fraction(1, 10**24), (field, idx, d12)
                invariant_checked += 1
            else:
                verdict = be.order_check((d01, Fraction(0)), (d12, Fraction(0)))
                assert verdict["status"] == "passed", (field, idx, verdict)
                dynamic_checked += 1
    assert dynamic_checked == 37
    assert invariant_checked == 8

    # Check real SlabObserver point currents & charges at t = 0.2 s when Arb is present.
    try:
        import flint  # noqa: F401
    except ImportError:
        pass
    else:
        observer = be.SlabObserver(model, Fraction(1, 10**18))
        ramp_segment = ctx["probe_segments"][1]
        point_obs = [observer.point(pt, ramp_segment) for pt in terminal_points]
        for group, keys in (
            ("ports", ("left", "right")),
            ("charges", ("device", "left_metal", "right_metal")),
        ):
            for key in keys:
                centers = [be._unpair(o[group][key]["center"]) for o in point_obs]
                radii = [be._unpair(o[group][key]["radius"]) for o in point_obs]
                d01 = (centers[0] - centers[1], radii[0] + radii[1])
                d12 = (centers[1] - centers[2], radii[1] + radii[2])
                verdict = be.order_check(d01, d12)
                assert verdict["status"] == "passed", (group, key, verdict)
