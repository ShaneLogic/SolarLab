"""Analytic E1/E3 checks and manufactured counterexamples; no B integration."""

from dataclasses import asdict
from copy import deepcopy
from fractions import Fraction
from hashlib import sha256
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.benchmarks import g2_reference_error as error
from scripts.benchmarks.contract_prototype import (
    ONE,
    SECOND,
    VOLT,
    EquationSpec,
    Layout,
    Support,
    VariableSpec,
)


def premises(names):
    return {
        name: {
            "checked": True,
            "source_identity": "manufactured_analytic_" + name,
            "statement": "Explicit analytic fixture premise, not a B qualification",
        }
        for name in names
    }


def axes(**amounts):
    return {name: amounts.get(name, Fraction()) for name in error.AXES}


def scalar_scaling():
    unit = json.loads(json.dumps(asdict(ONE)))
    return error.Scaling((Fraction(1),), (Fraction(1),), (unit,), (unit,))


def envelope(size=1, *, radius=Fraction(1)):
    return {
        "premises": premises(error.E3_PREMISES),
        "source_identity": "analytic_linear_envelope",
        "A_radius": [[0] * size for _ in range(size)],
        "B_radius": [[0] * size for _ in range(size)],
        "equation_error_by_axis": {a: [0] * size for a in error.AXES},
        "direct_state_error_by_axis": axes(),
        "box_radius": radius,
    }


def test_E1_three_levels_bound_the_known_decay_solution():
    values = [
        (Fraction(1) / (1 + h)) ** int(1 / h)
        for h in (Fraction(1, 4), Fraction(1, 8), Fraction(1, 16))
    ]
    out = error.estimate_e1(values, [0] * 3, premises=premises(error.E1_PREMISES))
    assert (
        out["status"] == "computed_pending_review" and out["method_status"] == "passed"
    )
    # Alternating exponential series gives an independent rational enclosure.
    n = 80
    high = sum(
        (Fraction((-1) ** k, math.factorial(k)) for k in range(n + 1)), Fraction()
    )
    low = high - Fraction(1, math.factorial(n + 1))
    assert error.q(out["E_R"]) >= values[2] - low > values[2] - high > 0
    assert error.q(out["ratio_lower"]) ** 5 >= 16
    assert error.q(out["ratio_upper"]) ** 5 <= 64


def test_frozen_coefficient_is_upward_and_weights_include_contamination():
    assert error.C == Fraction(42167, 25000)
    assert (1 + error.FS / error.C) ** 5 <= 16
    assert (1 + error.FS / Fraction(42166, 25000)) ** 5 > 16
    u = [Fraction(1, 1000)] * 3
    result = error.estimate_e1([3, 1, 0], u, premises=premises(error.E1_PREMISES))
    assert error.q(result["E_R"]) == error.C + error.C * u[1] + (error.C + 1) * u[2]


@pytest.mark.parametrize(
    "values,uncertainties,status,reason",
    [
        ([0, 2, 1], [0, 0, 0], "failed", "oscillatory"),
        ([4, 1, 0], [0, 0, 0], "failed", "nonasymptotic"),
        ([3, 1, 0], [Fraction(1, 5)] * 3, "unknown", "crosses"),
        ([3, 1, 1], [0, 0, 0], "unknown", "denominator"),
        ([3, 1, 0], [0, None, 0], "unknown", "number"),
    ],
)
def test_E1_counterexamples_remain_unknown_reference_error(
    values, uncertainties, status, reason
):
    result = error.estimate_e1(
        values, uncertainties, premises=premises(error.E1_PREMISES)
    )
    assert result["status"] == "unknown_reference_error"
    assert result["method_status"] == status and reason in result["reason"]
    assert result["E_R"] is None


def test_three_equal_values_and_contraction_are_not_proofs():
    assert (
        error.estimate_e1([1, 1, 1], [0, 0, 0], premises=premises(error.E1_PREMISES))[
            "E_R"
        ]
        is None
    )
    assert error.estimate_e1([3, 1, 0], [0, 0, 0], premises={})["E_R"] is None
    proof = {
        "kind": "same_initial_point",
        "checked": True,
        "source_identity": "same-public-Point",
        "statement": "Only time approximation vanishes; initial uncertainty remains",
    }
    result = error.estimate_e1(
        [1, 1, 1],
        [Fraction(1, 10)] * 3,
        premises=premises(error.E1_PREMISES),
        zero_time_proof=proof,
    )
    assert error.q(result["E_R"]) == Fraction(1, 10)


def test_scalar_E3_covers_exact_implicit_error():
    result = error.evaluate_e3_step(
        A=[[Fraction(5, 4)]],
        B=[[1]],
        residual=[Fraction(1, 100)],
        scaling=scalar_scaling(),
        previous=axes(initial_and_algebraic=Fraction(1, 50)),
        envelope=envelope(),
    )
    expected = Fraction(3, 100) / Fraction(5, 4)
    assert result["status"] == "computed_pending_review"
    assert expected <= error.q(result["error_upper"]) < expected + Fraction(1, 10**12)
    assert error.q(result["eta_upper"]) < 1


def test_E3_recurrence_retains_amplification_at_every_step():
    previous = axes()
    for _ in range(20):
        out = error.evaluate_e3_step(
            A=[[Fraction(1, 2)]],
            B=[[1]],
            residual=[Fraction(1, 2 * 10**12)],
            scaling=scalar_scaling(),
            previous=previous,
            envelope=envelope(),
        )
        assert out["status"] == "computed_pending_review"
        previous = out["error_by_axis"]
    exact = Fraction(2**20 - 1, 10**12)
    assert exact <= error.q(out["error_upper"]) < exact + Fraction(1, 10**15)
    assert error.q(out["error_upper"]) > 20 * Fraction(1, 10**12)


def test_DAE_scaling_uses_conservative_amount_units_and_algebraic_rows():
    layout = Layout(
        (Support("state", "global", (1,)),),
        (
            VariableSpec("a_x", "analytic", "state", (1,), ONE),
            VariableSpec("b_phi", "analytic", "state", (1,), VOLT, role="constraint"),
        ),
        (
            EquationSpec("a_balance", "analytic", "state", (1,), ONE / SECOND),
            EquationSpec(
                "b_constraint", "analytic", "state", (1,), VOLT, role="constraint"
            ),
        ),
    )

    def si(u):
        return json.loads(json.dumps(asdict(u)))

    spec = {
        "state": [2, 4],
        "equation": [3, 5],
        "state_units": [si(ONE), si(VOLT)],
        "equation_units": [si(ONE), si(VOLT)],
    }
    scaling = error.Scaling.from_spec(spec, layout)
    result = error.evaluate_e3_step(
        A=[[Fraction(5, 4), 0], [-2, 1]],
        B=[[1, 0], [0, 0]],
        residual=[Fraction(1, 100), Fraction(1, 200)],
        scaling=scaling,
        previous=axes(initial_and_algebraic=Fraction(1, 100)),
        envelope=envelope(2),
    )
    assert result["status"] == "computed_pending_review"
    # Independent exact inverse applied to a worst signed previous state error.
    dx = (Fraction(2, 100) + Fraction(1, 100)) / Fraction(5, 4)
    dphi = 2 * dx + Fraction(1, 200)
    assert error.q(result["error_upper"]) >= max(dx / 2, dphi / 4)
    spec["equation_units"][0] = si(ONE / SECOND)
    with pytest.raises(error.MissingEvidence, match="physical_scaling_units"):
        error.Scaling.from_spec(spec, layout)


@pytest.mark.parametrize("kind", ["radius", "eta", "remainder", "initial", "infinite"])
def test_missing_or_invalid_E3_data_never_becomes_zero(kind):
    bound = envelope()
    previous = axes()
    if kind == "radius":
        bound["box_radius"] = None
    elif kind == "eta":
        bound["A_radius"] = [[2]]
    elif kind == "remainder":
        del bound["equation_error_by_axis"]["trajectory_nonlinear"]
    elif kind == "initial":
        previous = None
    elif kind == "infinite":
        bound["direct_state_error_by_axis"]["evaluation"] = float("inf")
    result = error.evaluate_e3_step(
        A=[[1]],
        B=[[1]],
        residual=[Fraction(1, 10)],
        scaling=scalar_scaling(),
        previous=previous,
        envelope=bound,
    )
    assert result["status"] == "unknown_reference_error" and result["E_R"] is None


def test_finite_wrong_inverse_cannot_certify_a_step(monkeypatch):
    monkeypatch.setattr(np.linalg, "inv", lambda a: np.zeros_like(a))
    out = error.evaluate_e3_step(
        A=[[1]],
        B=[[1]],
        residual=[0],
        scaling=scalar_scaling(),
        previous=axes(),
        envelope=envelope(),
    )
    assert (
        out["status"] == "unknown_reference_error"
        and out["reason"] == "eta_not_below_one"
    )


def test_outward_matrix_product_encloses_exact_rational_product():
    z = np.array([[0.1, -0.3], [1.2, 2.3]])
    a = np.array([[1e-200, 7.1], [-3.7, 1e200]])
    low, high = error._left_product(z, a, a)
    for i in range(2):
        for j in range(2):
            exact = sum(
                (Fraction(float(z[i, k])) * Fraction(float(a[k, j])) for k in range(2)),
                Fraction(),
            )
            assert Fraction(float(low[i, j])) <= exact <= Fraction(float(high[i, j]))


@pytest.mark.parametrize("kind", ["int64", "uint64", "rational", "integer_list"])
def test_exact_operator_endpoints_survive_cancellation(kind):
    # Science RE-F01: rounding 2**53+1 before addition loses the true endpoint 1.
    center, radius = -(2**53), 2**53 + 1
    if kind == "int64":
        a, r = (
            np.array([[center]], dtype=np.int64),
            np.array([[radius]], dtype=np.int64),
        )
    elif kind == "uint64":
        center, radius = 2**63 + 1, 2**63
        a, r = (
            np.array([[center]], dtype=np.uint64),
            np.array([[radius]], dtype=np.uint64),
        )
    elif kind == "rational":
        center, radius = Fraction(center, 3), Fraction(radius, 3)
        a, r = np.array([[center]], dtype=object), np.array([[radius]], dtype=object)
    else:
        a, r = [[center]], [[radius]]
    low, high = scalar_scaling().operators(a, r)
    assert Fraction(float(low[0, 0])) <= center - radius
    assert Fraction(float(high[0, 0])) >= center + radius
    if kind == "uint64":
        assert low[0, 0] == 1.0
    elif kind != "rational":
        assert high[0, 0] == 1.0


def test_float64_operator_path_keeps_existing_outward_arithmetic():
    a = np.array([[0.1]], dtype=np.float64)
    r = np.array([[0.2]], dtype=np.float64)
    low, high = scalar_scaling().operators(a, r)
    expected_low = np.nextafter(np.nextafter(a - r, -np.inf), -np.inf)
    expected_high = np.nextafter(np.nextafter(a + r, np.inf), np.inf)
    assert np.array_equal(low.view(np.uint64), expected_low.view(np.uint64))
    assert np.array_equal(high.view(np.uint64), expected_high.view(np.uint64))


def test_integer_sparse_and_mixed_exact_operator_inputs():
    from scipy.sparse import csr_matrix

    a = csr_matrix(np.array([[-(2**53)]], dtype=np.int64))
    low, high = scalar_scaling().operators(a, np.array([[2**53 + 1]], dtype=object))
    assert Fraction(float(low[0, 0])) <= -(2**54) - 1
    assert high[0, 0] == 1.0
    low, high = scalar_scaling().operators(np.array([[-float(2**53)]]), [[2**53 + 1]])
    assert high[0, 0] == 1.0


def test_wide_float_matrix_is_not_silently_narrowed():
    if np.dtype(np.longdouble).itemsize <= 8:
        pytest.skip("platform longdouble has only binary64 precision")
    with pytest.raises(error.MissingEvidence, match="matrix_float_precision"):
        scalar_scaling().operators(np.array([[1]], dtype=np.longdouble), [[0]])


def test_zero_balance_is_not_a_q_or_f_error_oracle():
    proof = premises(
        (
            "fixed_B_continuity",
            "complete_boundary_storage",
            "fixed_parameters_and_initial",
            "no_reset_or_impulse",
        )
    )
    coverage = {
        "all_pieces_independently_integrated": True,
        "storage_delta_substitution": False,
        "history_sha256": "manufactured-math-only",
    }
    zero = {"center": [0, 1], "radius": [0, 1]}
    assert (
        error.flux_error(zero, zero, {}, identity_proof=proof, path_coverage=coverage)[
            "E_R"
        ]
        is None
    )
    charge = error.charge_error(
        [1, -2],
        [axes(trajectory_nonlinear=Fraction(1, 10))] * 2,
        [axes(initial_and_algebraic=Fraction(1, 20))] * 2,
        axes(),
    )
    assert error.q(charge["E_R"]) == Fraction(9, 20)
    flux = error.flux_error(
        zero, zero, charge, identity_proof=proof, path_coverage=coverage
    )
    assert error.q(flux["E_R"]) == Fraction(9, 20)
    coverage["storage_delta_substitution"] = True
    assert (
        error.flux_error(
            zero, zero, charge, identity_proof=proof, path_coverage=coverage
        )["E_R"]
        is None
    )
    assert (
        error.flux_error(zero, zero, charge, identity_proof={}, path_coverage=coverage)[
            "E_R"
        ]
        is None
    )


@pytest.mark.parametrize("x", [True, float("nan"), float("inf"), [1, 0], [-1, -2]])
def test_invalid_scalar_encodings_are_rejected(x):
    with pytest.raises(error.MissingEvidence):
        error.q(x)


def test_upward_bound_storage_is_finite_size_and_monotone():
    for v in (Fraction(1, 3), Fraction(1, 10**150), Fraction(10**150, 3)):
        rounded = error.upper(v)
        assert rounded >= v and (rounded - v) / v <= Fraction(1, 2**158)


def manufactured_campaign(tmp_path):
    # Reuse the existing public-BE manufactured fixture, not any device run.
    location = Path(__file__).with_name("test_g2_reference_be.py")
    spec = importlib.util.spec_from_file_location("g2_be_math_fixtures", location)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    segment = helper.ProtocolSegment("analytic", 0.0, 1.0, (0.0, 0.0), (0.0, 0.0))
    backend = helper.manufactured("driven", segments=(segment,))
    tags = helper.tags_for(backend.segments, [[0.0, 1.0]])
    levels = []
    for level in range(3):
        schedule = helper.be.compile_schedule(
            backend.segments, tags, level=level, seed=Fraction(1, 8)
        )
        _, _, identity, _, _ = helper.execute(
            tmp_path / str(level), backend=backend, schedule=schedule
        )
        levels.append(
            error.load_level(
                {
                    "directory": str(tmp_path / str(level)),
                    "identity": identity.payload(),
                },
                backend.problem.layout,
            )
        )

    def units(u):
        return json.loads(json.dumps(asdict(u)))

    n_unit = backend.problem.layout.variables[0].unit
    scaling = {
        "state": [1, 1, 1],
        "equation": [1, 1, 1],
        "state_units": [units(n_unit), units(n_unit), units(VOLT)],
        "equation_units": [units(n_unit), units(n_unit), units(VOLT)],
    }
    provider = SimpleNamespace(
        layout=backend.problem.layout,
        source_identity=backend.problem.source_identity,
        charge_coefficients=[1, 1, 0],
        charge_form_identity="manufactured_body_form",
    )
    provider.operators = lambda left, right: (
        backend.problem.conservative_jacobian(left, right),
        [[1, 0, 0], [0, 1, 0], [0, 0, 0]],
    )
    provider.feasible_box = lambda point, radius, scale: (
        min(helper.exact(point, "n")) > radius
    )
    bound = envelope(3)
    bound["source_identity"] = provider.source_identity
    support = {
        "source_identity": provider.source_identity,
        "scaling": scaling,
        "initial_point_identity": backend.initial.identity,
        "initial_error_by_axis": axes(),
        "initial_premises": premises(("initial_error_and_DAE_meaning",)),
        "step_envelope": bound,
        "port_sensitivities": {
            "port/" + side: {
                "premises": premises(
                    ("uniform_physical_sensitivity", "direct_input_coverage")
                ),
                "unit": "A",
                "absolute_gradient": [0, 0, 0],
                "direct_error_by_axis": axes(),
                "state_units": scaling["state_units"],
            }
            for side in ("left", "right")
        },
    }
    target = error.target_identity(
        provider.source_identity, backend.initial.identity, [asdict(segment)]
    )
    invariants = {
        key: {
            "kind": "reviewed_constant_or_invariant",
            "checked": True,
            "source_identity": "analytic-linear-time-invariant",
            "statement": "phi=9; I_left=0; I_right+t=0",
            "labels_sha256": error.digest(tags),
            "target_problem_identity": target,
        }
        for key in ("state/phi/0", "port/left", "port/right")
    }
    parameters = {
        "schema": "solarlab.g2-reference-error-parameters.v1",
        "target_problem_identity": target,
        "method_contract_sha256": "1" * 64,
        "E1_premises": premises(error.E1_PREMISES),
        "levels": [deepcopy(support) for _ in range(3)],
        "invariants": invariants,
        "charge": {
            "form_identity": provider.charge_form_identity,
            "extra_by_axis": axes(),
            "premises": premises(
                ("complete_body_charge_form", "fixed_coefficients_and_background")
            ),
            "identity_proof": premises(
                (
                    "fixed_B_continuity",
                    "complete_boundary_storage",
                    "fixed_parameters_and_initial",
                    "no_reset_or_impulse",
                )
            ),
        },
    }
    context = {
        "target_problem_identity": target,
        "method_contract": {"path": "manufactured-method", "sha256": "1" * 64},
        "lineage_source": {"path": "manufactured-lineage", "sha256": "2" * 64},
        "author": "analytic_test",
        "premises": parameters["E1_premises"],
        "reference_identity": {
            "case_id": "manufactured",
            "run_id": error.digest(levels[2].identity.payload()),
            "source_identity": provider.source_identity,
            "request_sha256": levels[2].identity.input_sha256,
        },
    }
    return levels, provider, parameters, context


def test_offline_history_to_complete_conditional_records(tmp_path):
    levels, provider, parameters, context = manufactured_campaign(tmp_path)
    log = []
    result = error.analyze_levels(
        levels, provider, parameters, context, emit=log.append
    )
    assert (
        result["status"] == "computed_pending_review"
        and result["scientific_pass"] is False
    )
    assert len(log) == 8 + 16 + 32
    assert {x["level"] for x in log} == {0, 1, 2}
    assert all(
        r["claim_kind"] == "conditional_uncertainty_estimate"
        for r in result["quantities"].values()
    )
    q = result["quantities"]["q"]
    f = result["quantities"]["f"]
    assert error.q(q["conditional_estimate_value"]) > 0
    assert error.q(f["conditional_estimate_value"]) >= error.q(
        q["conditional_estimate_value"]
    )
    assert f["rigorous_bound_value"] is None and f["diagnostic_value"] is None
    assert result["E2_status"] == "pending_Main_Strict_transfer"
    error.canonical(result)


def test_offline_missing_remainder_stays_unknown_and_keeps_all_steps(tmp_path):
    levels, provider, parameters, context = manufactured_campaign(tmp_path)
    del parameters["levels"][1]["step_envelope"]["equation_error_by_axis"]["evaluation"]
    log = []
    result = error.analyze_levels(
        levels, provider, parameters, context, emit=log.append
    )
    assert result["status"] == "unknown_reference_error"
    assert result["quantities"]["q"]["conditional_estimate_value"] is None
    assert result["E3_summary"][1]["first_failure"]["step_index"] == 1
    assert len(log) == 56


def test_changed_quantity_clock_is_not_compared_by_array_position(tmp_path):
    levels, provider, parameters, context = manufactured_campaign(tmp_path)
    levels[1].readback["observations"][-1]["tag"]["time_hex"] = (0.5).hex()
    with pytest.raises(error.MissingEvidence, match="different_time_labels"):
        error.analyze_levels(levels, provider, parameters, context)


def test_native_history_is_rejected_before_reader_opens_it(tmp_path, monkeypatch):
    (tmp_path / "RunResult.json").write_text(
        json.dumps({"schema": "native_history", "method": "IDA"})
    )
    identity = {
        "case_id": "old",
        "input_sha256": "1" * 64,
        "method_contract_sha256": "2" * 64,
        "plan_sha256": "3" * 64,
        "source_commit": "4" * 40,
        "environment_identity": "test",
        "schedule_identity": "5" * 64,
        "source_identity": "6" * 64,
        "initial_point_identity": "7" * 64,
        "controls_sha256": "8" * 64,
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("old history must never be opened")

    monkeypatch.setattr(error, "read_reference_history", forbidden)
    with pytest.raises(error.MissingEvidence, match="not_BE_history"):
        error.load_level({"directory": str(tmp_path), "identity": identity}, None)


def test_method_loader_accepts_actual_top_level_freeze_structure(tmp_path):
    author = {
        "schema": error.METHOD_SCHEMA,
        "version": 1,
        "status": "candidate",
        "authority": {"freeze_record": None},
        "E1": {
            "Fs": [5, 4],
            "p_use": [4, 5],
            "coefficient_upper": [42167, 25000],
            "order_interval": [[4, 5], [6, 5]],
        },
        "time_ladder": {"levels": [0, 1, 2]},
    }

    def save(name, value):
        p = tmp_path / name
        raw = error.canonical(value)
        p.write_bytes(raw)
        return {"path": str(p), "sha256": sha256(raw).hexdigest(), "bytes": len(raw)}

    a = save("Author.json", author)
    r = save("Review.json", {"review": "manufactured-method-test-only"})
    frozen = {
        **author,
        "status": "frozen",
        "freeze": {
            "frozen_at": "2026-10-15T00:00:00Z",
            "frozen_by": "manufactured-test",
            "reviewed_path": a["path"],
            "reviewed_sha256": a["sha256"],
            "reviewed_bytes": a["bytes"],
            "review_records": [r],
            "user_records": ["test"],
            "notes": ["three levels, two refinements"],
            "plan_rule": "test only",
        },
    }
    reference = save("Frozen.json", frozen)
    assert error.load_method(reference)["authority"]["freeze_record"] is None
    frozen["E1"]["coefficient_upper"] = [1, 1]
    with pytest.raises(error.MissingEvidence, match="changed_reviewed_method_body"):
        error.load_method(save("Changed.json", frozen))


def test_nonnumeric_matrices_cannot_drop_imaginary_or_boolean_parts():
    for matrix in (np.array([[1j]]), np.array([[True]])):
        out = error.evaluate_e3_step(
            A=matrix,
            B=[[1]],
            residual=[0],
            scaling=scalar_scaling(),
            previous=axes(),
            envelope=envelope(),
        )
        assert out["status"] == "unknown_reference_error"


def test_zero_charge_coefficients_do_not_require_unrelated_state_error(tmp_path):
    levels, provider, parameters, context = manufactured_campaign(tmp_path)
    del parameters["invariants"]["state/phi/0"]
    result = error.analyze_levels(levels, provider, parameters, context)
    assert result["quantities"]["state/phi/0"]["status"] == "unknown_reference_error"
    assert result["quantities"]["q"]["status"] == "computed_pending_review"
    assert (
        result["status"] == "unknown_reference_error"
    )  # Other required state gates still matter.


@pytest.mark.parametrize("mutation", ["seed", "input"])
def test_time_family_and_physical_protocol_must_match(tmp_path, mutation):
    levels, provider, parameters, context = manufactured_campaign(tmp_path)
    if mutation == "seed":
        levels[1].header["schedule"]["seed"] = [1, 4]
        reason = "different_time_family"
    else:
        for level in levels:
            level.header["schedule"]["segments"][0]["voltage"][1] = 1.0
        reason = "history_protocol_differs_from_source_request"
    with pytest.raises(error.MissingEvidence, match=reason):
        error.analyze_levels(levels, provider, parameters, context)


def test_public_factored_form_keeps_area_nt_sign_and_divisor():
    from scripts.benchmarks.contract_prototype import (
        COULOMB,
        PARTICLE,
        LinearFactor,
        LinearTerm,
        PhysicalLinearForm,
    )

    layout = Layout(
        (Support("cells", "cell", (2,)),),
        (VariableSpec("population", "analytic", "cells", (2,), PARTICLE),),
        (EquationSpec("balance", "analytic", "cells", (2,), PARTICLE / SECOND),),
    )
    source = "analytic_factor_source"
    charge = LinearFactor("q", 0.1, COULOMB / PARTICLE, source)
    factor = LinearFactor("area_times_Nt_fixture", 3.0, ONE, source)
    divisor = LinearFactor("thickness_fixture", 7.0, ONE, source)
    form = PhysicalLinearForm(
        layout,
        ("body",),
        (COULOMB,),
        (
            LinearTerm(0, "population", 0, (charge, factor), -1, divisor),
            LinearTerm(0, "population", 1, (charge,), 1),
        ),
        source,
        "state",
        2,
    )
    (row,) = error.form_matrix(form)
    assert row == [-Fraction(0.1) * 3 / 7, Fraction(0.1)]


def test_error_output_respects_cap_and_never_replaces_existing_bundle(tmp_path):
    from scripts.benchmarks.g2_reference_be import Limits

    output = error.Output(
        tmp_path / "bundle",
        Limits(60.0, output_bytes=1024, metadata_reserve=128, record_bytes=128),
    )
    output.emit({"kind": "manufactured"})
    with pytest.raises(error.MissingEvidence, match="error_output_budget"):
        output.emit({"oversized": "x" * 200})
    output.close()
    with pytest.raises(FileExistsError):
        error.Output(tmp_path / "bundle", Limits(60.0))
