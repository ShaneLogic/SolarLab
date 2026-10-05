"""Independent Decimal references for actual public physical precision consumers."""

from __future__ import annotations

from decimal import Decimal, localcontext
from hashlib import sha256
from math import factorial
from pathlib import Path
import json
import os

import numpy as np
import pytest

from perovskite_sim.constants import EPS_0, K_B, Q
from scripts.benchmarks.contract_prototype import (
    AREA, COULOMB, ONE, PARTICLE, SECOND, VOLT, VOLUME, ContractError,
    EquationSpec, Geometry, Layout, Point, StateIncrement, StateView, Support, VariableSpec,
)
from scripts.benchmarks.precision_prototype import (
    DD, DoubleArray, RelativeCoordinates, decode_point, encode_point,
)
from scripts.benchmarks.device_prototype import (
    ElectronTrapCapture, FaceGeometry, _bernoulli_log_change, assemble_charge_rate, displacement_current,
    displacement_current_from_potential, electric_displacement, lift_physical_inputs,
    sg_current, sg_current_increment,
)


REPO = Path(__file__).resolve().parents[2]
GATE_PATH = Path(__file__).with_name("AnalyticGatesV1.json")
assert sha256(GATE_PATH.read_bytes()).hexdigest() == "43c9ba4ab1b6a95b3e278efcd8eed9a67dc4d5e258295c8e6961c53f7ee57201"
NC08 = next(g for g in json.loads(GATE_PATH.read_text())["gates"] if g["id"] == "NC08")
TEXTS = NC08["decimal_inputs"]["delta_log_n"]
VT = K_B * 300.0 / Q
FIXTURE = REPO / "tests/fixtures/refactor/R1FailureWitnessV1.json"
FIXTURE_SHA = "29e4981bd4d5d11d19f3120febbd47d853d39bfb45a72a44ecec91566757db1d"


def binary(value):
    return Decimal.from_float(float(value))


def words(value: DoubleArray, precision=80):
    with localcontext() as ctx:
        ctx.prec = precision
        return [binary(h) + binary(l) for h, l in zip(value.high.ravel(), value.low.ravel(), strict=True)]


def series_expm1(x):
    return sum(x ** k / Decimal(factorial(k)) for k in range(1, 6))


def binary_inputs(**values):
    return {name: {"hex": float(value).hex(), "decimal": str(binary(value))}
            for name, value in values.items()}


def base_reference(kind, text, precision):
    with localcontext() as ctx:
        ctx.prec = precision
        dn = binary(float(NC08["decimal_inputs"]["n0"])) * series_expm1(binary(float(text)))
        if kind == "SG":
            g = NC08["weak_flux"]
            return binary(float(g["q_C"])) * binary(g["D_m2_s"]) / binary(g["dx_m"]) * dn
        g = NC08["capture"]
        return binary(g["capture_coefficient"]) * binary(g["trap_density"]) * (1 - binary(g["occupancy"])) * dn


def close_record(candidate: DoubleArray, reference, atol, rtol):
    with localcontext() as ctx:
        ctx.prec = 80
        actual = words(candidate)
        assert len(actual) == len(reference)
        errors = [abs(a - b) for a, b in zip(actual, reference, strict=True)]
        limits = [Decimal(str(atol)) + Decimal(str(rtol)) * abs(b) for b in reference]
    signs = [(a > 0) - (a < 0) == (b > 0) - (b < 0)
             for a, b in zip(actual, reference, strict=True)]
    return {"reference": [str(v) for v in reference], "candidate_words_sum": [str(v) for v in actual],
            "absolute_error": [float(v) for v in errors], "limits": [float(v) for v in limits],
            "sign_and_exact_zero_match": all(signs),
            "passed": all(a <= b for a, b in zip(errors, limits, strict=True))}


def reference_uncertainty(reference80, reference100, atol, rtol):
    with localcontext() as ctx:
        ctx.prec = 100
        error = [abs(a - b) for a, b in zip(reference80, reference100, strict=True)]
        budgets = [(Decimal(str(atol)) + Decimal(str(rtol)) * abs(b)) / 3 for b in reference100]
    return {"absolute_error": [str(v) for v in error], "one_third_gate": [str(v) for v in budgets],
            "passed": all(a <= b for a, b in zip(error, budgets, strict=True))}


def conclude(request, data, checks):
    def encode(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(type(value).__name__)
    data.update(test=request.node.nodeid, checks={key: bool(v) for key, v in checks.items()},
                passed=bool(all(checks.values())))
    if path := os.environ.get("DEVICE_CASE_LOG"):
        with Path(path).open("a") as stream:
            stream.write(json.dumps(data, allow_nan=False, default=encode, separators=(",", ":")) + "\n")
    assert all(checks.values()), {key: bool(v) for key, v in checks.items() if not v}


def base_point(n0=None, gauge=0.0):
    n0 = float(NC08["decimal_inputs"]["n0"]) if n0 is None else n0
    supports = (Support("cells", "cell", (2,)), Support("faces", "face", (1,)))
    variables = (VariableSpec("n_m3", "carrier", "cells", (2,), PARTICLE / VOLUME, lower=0),
                 VariableSpec("p_m3", "carrier", "cells", (2,), PARTICLE / VOLUME, lower=0),
                 VariableSpec("phi_V", "electrostatics", "cells", (2,), VOLT),
                 VariableSpec("occupancy", "trap", "cells", (2,), ONE, lower=0, upper=1),
                 VariableSpec("D_C_m2", "electrostatics", "faces", (1,), COULOMB / AREA))
    equations = (EquationSpec("charge", "electrostatics", "cells", (2,), COULOMB / SECOND,
                              derivative_support=("n_m3", "p_m3", "phi_V")),
                 EquationSpec("carrier", "capture", "cells", (2,), PARTICLE / SECOND,
                              derivative_support=("n_m3", "occupancy")),
                 EquationSpec("trap", "capture", "cells", (2,), PARTICLE / SECOND,
                              derivative_support=("n_m3", "occupancy")))
    layout = Layout(supports, variables, equations)
    fields = [("n_m3", DoubleArray([n0, n0])), ("p_m3", DoubleArray([n0, n0])),
              ("phi_V", DoubleArray([gauge, gauge])), ("occupancy", DoubleArray([0.5, 0.5])),
              ("D_C_m2", DoubleArray([float(NC08["displacement"]["D0_C_m2"])]))]
    modes = {v.id: ("log" if n0 > 0 else "inactive") if v.id in {"n_m3", "p_m3"}
             else "linear" for v in variables}
    coordinates = RelativeCoordinates(layout, modes)
    return coordinates, coordinates.initial(StateView(layout, fields))


def advance(coordinates, point, changes, time=1.0):
    dy = np.zeros(point.state.layout.size)
    for name, value in changes.items():
        dy[point.state.layout.offsets[name]] = np.asarray(value).ravel()
    return coordinates.advance(point, dy, time, point.inputs)


def restored_increment(left, right):
    return StateIncrement.from_points(left, right)


def exact_root_fields(point):
    """Only an unchanged, exactly resolved test seed may become another root."""
    fields = []
    for name in point.state.fields:
        projection = point.state.project_field(name)
        projection.require_error(np.zeros(projection.value.shape))
        fields.append((name, projection.value))
    return fields


def relative_reference(left, changes, density_id, precision):
    """Evaluate the specified map from its binary inputs, before word projection."""
    with localcontext() as ctx:
        ctx.prec = precision
        n0 = words(left.state.field(density_id), precision)
        phi0 = words(left.state.field("phi_V"), precision)
        dn = changes.get(density_id, np.zeros(len(n0)))
        dp = changes.get("phi_V", np.zeros(len(phi0)))
        return ([v * binary(z).exp() for v, z in zip(n0, dn, strict=True)],
                [v + binary(z) for v, z in zip(phi0, dp, strict=True)])


def bernoulli_reference(x):
    if abs(x) < Decimal("1e-8"):
        return 1 - x / 2 + x*x / 12 - x**4 / 720 + x**6 / 30240 - x**8 / 1209600 + x**10 / 47900160
    return x / (x.exp() - 1)


def sg_reference(n, phi, pairs, spacing, diffusion, charge, vt, carrier, precision=80):
    with localcontext() as ctx:
        ctx.prec = precision
        result = []
        for index, (i, j) in enumerate(pairs):
            xi = (phi[j] - phi[i]) / vt
            b, bm = bernoulli_reference(xi), bernoulli_reference(-xi)
            value = b * n[j] - bm * n[i] if carrier == "electron" else b * n[i] - bm * n[j]
            result.append(charge * diffusion[index] / spacing[index] * value)
        return result


@pytest.mark.parametrize("text", TEXTS)
def test_covered_nc08_actual_SG_and_charge_assembly(text, request):
    gate = NC08["weak_flux"]
    coordinates, left = base_point()
    right, increment = advance(coordinates, left, {"n_m3": [0, float(text)]})
    face = FaceGeometry(np.array([[0, 1]]), np.array([gate["dx_m"]]))
    args = dict(density_id="n_m3", thermal_voltage_V=VT, diffusion_m2_s=gate["D_m2_s"], charge_C=float(gate["q_C"]))
    current = sg_current(right.state, face, **args)
    change = sg_current_increment(left, right, increment, face, **args)
    with localcontext() as ctx:
        ctx.prec = 80
        dn = Decimal(NC08["decimal_inputs"]["n0"]) * series_expm1(Decimal(text))
        reference = Decimal(gate["q_C"]) * Decimal(str(gate["D_m2_s"])) / Decimal(str(gate["dx_m"])) * dn
    check = close_record(current, [reference], gate["atol_A_m2"], gate["rtol"])
    delta_check = close_record(change, [reference], gate["atol_A_m2"], gate["rtol"])
    actual_reference = [base_reference("SG", text, p) for p in [80, 100]]
    binary_check = close_record(current, [actual_reference[0]], gate["atol_A_m2"], gate["rtol"])
    uncertainty = reference_uncertainty([actual_reference[0]], [actual_reference[1]], gate["atol_A_m2"], gate["rtol"])
    geometry = Geometry([gate["dx_m"] / 2] * 2, np.array([[0, 1]]), [1.0])
    assembled = assemble_charge_rate(right.state, "charge", geometry, current)
    assembly_check = close_record(assembled, [-reference, reference], gate["atol_A_m2"], gate["rtol"])
    restored = decode_point(json.loads(json.dumps(encode_point(right))), right.state.layout)
    restored_current = sg_current(restored.state, face, **args)
    conclude(request, {"family": "covered_NC08_SG", "input": text, "current": check,
                       "finite_current_change": delta_check, "assembled_charge_rate": assembly_check,
                       "binary_input_reference": binary_check, "reference_uncertainty": uncertainty,
                       "binary_inputs": binary_inputs(n0=float(NC08["decimal_inputs"]["n0"]), delta_log=float(text),
                                                       q=float(gate["q_C"]), D=gate["D_m2_s"], dx=gate["dx_m"], VT=VT),
                       "decimal_to_binary_quantization_effect": str(actual_reference[0] - reference),
                       "source_area_m2": 1.0, "restored_identity": restored.identity},
             {"SG_value": check["passed"], "finite_increment": delta_check["passed"],
              "charge_assembly": assembly_check["passed"], "zero_nonzero": (words(current)[0] == 0) == (text == "0"),
              "signed_current": check["sign_and_exact_zero_match"],
              "binary_reference": binary_check["passed"] and uncertainty["passed"],
              "codec_identity": restored.identity == right.identity,
              "restored_physics_same_words": restored_current.identity_bytes() == current.identity_bytes()})


@pytest.mark.parametrize("text", TEXTS)
def test_covered_nc08_actual_net_capture_and_sources(text, request):
    gate = NC08["capture"]
    coordinates, left = base_point()
    right, increment = advance(coordinates, left, {"n_m3": [0, float(text)]})
    closure = ElectronTrapCapture(gate["capture_coefficient"], gate["trap_density"],
                                   float(NC08["decimal_inputs"]["n0"]))
    rate = closure.net_rate(right.state)
    rate_change = closure.finite_change(left, right, increment)
    with localcontext() as ctx:
        ctx.prec = 80
        dn = Decimal(NC08["decimal_inputs"]["n0"]) * series_expm1(Decimal(text))
        expected = Decimal(str(gate["capture_coefficient"])) * Decimal(str(gate["trap_density"])) * (1 - Decimal(str(gate["occupancy"]))) * dn
    check = close_record(rate, [Decimal(0), expected], gate["atol"], gate["rtol"])
    delta_check = close_record(rate_change, [Decimal(0), expected], gate["atol"], gate["rtol"])
    actual_reference = [base_reference("capture", text, p) for p in [80, 100]]
    binary_check = close_record(rate, [Decimal(0), actual_reference[0]], gate["atol"], gate["rtol"])
    uncertainty = reference_uncertainty([actual_reference[0]], [actual_reference[1]], gate["atol"], gate["rtol"])
    dx = NC08["weak_flux"]["dx_m"]
    geometry = Geometry([dx / 2] * 2, np.array([[0, 1]]), [1.0])
    carrier, trapped = closure.assembled_inventory_sources(right.state, geometry, "carrier", "trap")
    balanced = carrier.as_dd() + trapped.as_dd()
    restored = decode_point(encode_point(right), right.state.layout)
    conclude(request, {"family": "covered_NC08_capture", "input": text, "net_rate": check,
                       "finite_rate_change": delta_check, "assembled_carrier": [str(v) for v in words(carrier)],
                       "binary_input_reference": binary_check, "reference_uncertainty": uncertainty,
                       "binary_inputs": binary_inputs(n0=float(NC08["decimal_inputs"]["n0"]), delta_log=float(text),
                                                       cn=gate["capture_coefficient"], Nt=gate["trap_density"], f=gate["occupancy"]),
                       "decimal_to_binary_quantization_effect": str(actual_reference[0] - expected),
                       "assembled_trapped": [str(v) for v in words(trapped)]},
             {"actual_capture_emission": check["passed"], "finite_change": delta_check["passed"],
              "shared_reaction_inventory": np.all(balanced == 0),
              "signed_capture": check["sign_and_exact_zero_match"],
              "binary_reference": binary_check["passed"] and uncertainty["passed"],
              "zero_nonzero": (words(rate)[1] == 0) == (text == "0"),
              "restored_consumer_words": closure.net_rate(restored.state).identity_bytes() == rate.identity_bytes()})


def test_covered_nc08_actual_displacement_consumer(request):
    gate = NC08["displacement"]
    coordinates, left = base_point()
    right, increment = advance(coordinates, left, {"D_C_m2": [float(gate["increment_C_m2"])]}, float(gate["dt_s"]))
    current = displacement_current(left, right, increment)
    expected = Decimal(gate["expected_current_A_m2"])
    check = close_record(current, [expected], gate["atol_A_m2"], gate["rtol"])
    restored = decode_point(encode_point(right), right.state.layout)
    recovered = displacement_current(left, restored, restored_increment(left, restored))
    recovered_check = close_record(recovered, [expected], gate["atol_A_m2"], gate["rtol"])
    conclude(request, {"family": "covered_NC08_displacement", "current": check,
                       "binary_inputs": binary_inputs(D0=float(gate["D0_C_m2"]), delta_D=float(gate["increment_C_m2"]), dt=float(gate["dt_s"])),
                       "restored_endpoint_current": recovered_check},
             {"increment_consumer": check["passed"], "restored_endpoint_consumer": recovered_check["passed"],
              "same_rounded_absolute_D": np.array_equal(left.state.field("D_C_m2").high, right.state.field("D_C_m2").high),
              "signed_nonzero": check["sign_and_exact_zero_match"] and recovered_check["sign_and_exact_zero_match"]})


@pytest.mark.parametrize("carrier", ["electron", "hole"])
@pytest.mark.parametrize("drive", ["density", "potential", "mixed"])
@pytest.mark.parametrize("gauge", [0.0, 1.0])
@pytest.mark.parametrize("text", TEXTS)
def test_extended_SG_physical_potential_density_and_codec(carrier, drive, gauge, text, request):
    """The authoritative input is now the map; old projected-word failures remain archived."""
    gate = NC08["weak_flux"]
    coordinates, left = base_point(gauge=gauge)
    changes = {}
    density_id = "n_m3" if carrier == "electron" else "p_m3"
    if drive != "potential":
        changes[density_id] = [0, float(text)]
    if drive != "density":
        changes["phi_V"] = [0, -VT * float(text)]
    right, increment = advance(coordinates, left, changes)
    face = FaceGeometry(np.array([[0, 1]]), np.array([gate["dx_m"]]))
    args = dict(density_id=density_id, thermal_voltage_V=VT, diffusion_m2_s=gate["D_m2_s"], charge_C=float(gate["q_C"]), carrier=carrier)
    actual = sg_current(right.state, face, **args)
    delta = sg_current_increment(left, right, increment, face, **args)
    refs = []
    for precision in [80, 100]:
        density, potential = relative_reference(left, changes, density_id, precision)
        refs.append(sg_reference(density, potential,
                                 face.pairs, [binary(gate["dx_m"])], [binary(gate["D_m2_s"])],
                                 binary(float(gate["q_C"])), binary(VT), carrier, precision))
    check = close_record(actual, refs[0], gate["atol_A_m2"], gate["rtol"])
    delta_check = close_record(delta, refs[0], gate["atol_A_m2"], gate["rtol"])
    restored = decode_point(json.loads(json.dumps(encode_point(right))), right.state.layout)
    restored_check = close_record(sg_current(restored.state, face, **args), refs[0], gate["atol_A_m2"], gate["rtol"])
    ref_error = max(abs(a - b) for a, b in zip(*refs))
    nonzero_expected = any(ref != 0 for ref in refs[0])
    conclude(request, {"family": "extended_SG", "carrier": carrier, "drive": drive, "gauge_V": gauge,
                       "input": text, "current": check, "finite_change": delta_check,
                       "actual_state": encode_point(right),
                       "binary_inputs": binary_inputs(q=float(gate["q_C"]), D=gate["D_m2_s"], dx=gate["dx_m"], VT=VT),
                       "restored": restored_check, "reference_80_vs_100": str(ref_error)},
             {"SG": check["passed"], "increment": delta_check["passed"], "restored": restored_check["passed"],
              "zero_nonzero": any(v != 0 for v in words(actual)) == nonzero_expected,
              "current_sign": check["sign_and_exact_zero_match"],
              "increment_sign": delta_check["sign_and_exact_zero_match"],
              "reference": ref_error <= Decimal(str(NC08["reference_error_bound"]))})


@pytest.mark.parametrize("text", ["0", "1e-22", "-1e-22"])
@pytest.mark.parametrize("gauge", [0.0, 1.0])
def test_extended_constitutive_displacement_chain(text, gauge, request):
    gate = NC08["displacement"]
    dx, eps = NC08["weak_flux"]["dx_m"], EPS_0 * 20.0
    coords, base = base_point(gauge=gauge)
    slope = -DD(float(gate["D0_C_m2"])) * dx / eps
    fields = [(key, value) for key, value in exact_root_fields(base) if key != "phi_V"]
    fields.append(("phi_V", DoubleArray.from_dd(DD([gauge, gauge]) + DD([0.0, float(slope.hi)], [0.0, float(slope.lo)]))))
    left = coords.initial(StateView(base.state.layout, fields))
    proposed_dphi = -DD(float(text)) * dx / eps
    # The public coordinate is binary64; retain its quantization explicitly in
    # the reference, then compare it separately with the nominal DeltaD target.
    coordinate_dphi = float(proposed_dphi.hi)
    right, increment = advance(coords, left, {"phi_V": [0, coordinate_dphi]}, float(gate["dt_s"]))
    face = FaceGeometry(np.array([[0, 1]]), np.array([dx]))
    current = displacement_current_from_potential(left, right, increment, face, eps)
    with localcontext() as ctx:
        ctx.prec = 80
        realized = -binary(eps) * binary(coordinate_dphi) / binary(dx) / binary(float(gate["dt_s"]))
        nominal = Decimal(text) / Decimal(gate["dt_s"])
        quantization = abs(realized - nominal)
    with localcontext() as ctx:
        ctx.prec = 100
        realized100 = -binary(eps) * binary(coordinate_dphi) / binary(dx) / binary(float(gate["dt_s"]))
    uncertainty = reference_uncertainty([realized], [realized100], gate["atol_A_m2"], gate["rtol"])
    check = close_record(current, [nominal], gate["atol_A_m2"], gate["rtol"])
    restored = decode_point(encode_point(right), right.state.layout)
    restored_current = displacement_current_from_potential(left, restored, restored_increment(left, restored), face, eps)
    restore_check = close_record(restored_current, [nominal], gate["atol_A_m2"], gate["rtol"])
    D0, D1 = electric_displacement(left.state, face, eps), electric_displacement(right.state, face, eps)
    conclude(request, {"family": "extended_displacement", "increment_C_m2": text, "gauge_V": gauge,
                       "current": check, "restored_current": restore_check,
                       "coordinate_dphi_V": coordinate_dphi, "input_quantization_current": str(quantization),
                       "reference_uncertainty": uncertainty,
                       "binary_inputs": binary_inputs(epsilon=eps, dx=dx, delta_phi=coordinate_dphi, dt=float(gate["dt_s"])),
                       "D0_words": [str(v) for v in words(D0)], "D1_words": [str(v) for v in words(D1)]},
             {"constitutive_chain": check["passed"], "restored_chain": restore_check["passed"],
              "zero_nonzero": (words(current)[0] == 0) == (text == "0"),
              "signed_current": check["sign_and_exact_zero_match"] and restore_check["sign_and_exact_zero_match"],
              "reference_uncertainty": uncertainty["passed"],
              "input_quantization": quantization <= (Decimal(str(gate["atol_A_m2"])) + Decimal(str(gate["rtol"])) * abs(nominal)) / 3})


def source_point(case_name):
    assert sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA
    fixture = json.loads(FIXTURE.read_text())
    case = fixture["cases"][case_name]
    if case_name == "step183_accepted":
        raw = case["saved_row"]["state"]
        coordinate = case["saved_row"]["physics_reconstruction"]["coordinate"]
        timestamp = case["saved_row"]["time_s"]
    else:
        raw, coordinate = case["saved_witness"]["attempted_state"], case["saved_witness"]["coordinate"]
        timestamp = case["saved_witness"]["previous_time_s"] + case["saved_witness"]["dt_s"]
    units = {"n_m3": PARTICLE / VOLUME, "p_m3": PARTICLE / VOLUME, "phi_V": VOLT,
             "positive_m3": PARTICLE / VOLUME, "occupancy": ONE,
             "trace_state_m3": PARTICLE / VOLUME, "trace_potential_V": VOLT}
    supports, variables, fields = [], [], []
    for name, unit in units.items():
        value = np.asarray(raw[name], float)
        supports.append(Support(name, "global", value.shape))
        variables.append(VariableSpec(name, "original-physical-input", name, value.shape, unit))
        fields.append((name, DoubleArray(value)))
    layout = Layout(tuple(supports), tuple(variables), ())
    return fixture, Point(timestamp, np.asarray(coordinate), np.empty(0), StateView(layout, fields),
                          f"original-physical-snapshot:{FIXTURE_SHA}:{case_name}")


def lift_reference(left, changes, precision):
    deltas, fields = {}, {}
    with localcontext() as ctx:
        ctx.prec = precision
        for name in left.state.fields:
            origin = words(left.state.field(name), precision)
            if name in {"n_m3", "p_m3", "positive_m3", "trace_state_m3"}:
                mode = {"n_m3": "electron", "p_m3": "hole", "positive_m3": "positive", "trace_state_m3": "trace_density"}[name]
                drives = [binary(v) for v in changes[mode].ravel()]
                if name in {"n_m3", "p_m3"}:
                    sign = 1 if name == "n_m3" else -1
                    drives = [d + sign * binary(z) for d, z in zip(drives, changes["potential"].ravel(), strict=True)]
                delta = [v * series_expm1(d) for v, d in zip(origin, drives, strict=True)]
            elif name == "occupancy":
                odds = [series_expm1(binary(v)) for v in changes["occupancy"].ravel()]
                delta = [f * (1 - f) * e / (1 + f * e) for f, e in zip(origin, odds, strict=True)]
            elif name in {"phi_V", "trace_potential_V"}:
                mode = "potential" if name == "phi_V" else "trace_potential"
                delta = [binary(VT) * binary(z) for z in changes[mode].ravel()]
            else:
                raise AssertionError("uncovered physical field")
            deltas[name] = delta
            fields[name] = [v + d for v, d in zip(origin, delta, strict=True)]
    return deltas, fields


@pytest.mark.parametrize("case_name", ["step183_failed", "step183_accepted", "step309_failed"])
@pytest.mark.parametrize("drive", ["density", "potential", "mixed"])
@pytest.mark.parametrize("text", TEXTS)
def test_extended_saved_R1_fields_consume_new_input_lift(case_name, drive, text, request):
    fixture, left = source_point(case_name)
    x = float(text)
    changes = {mode: np.zeros(left.state.field(field).shape) for mode, field in {
        "electron": "n_m3", "hole": "p_m3", "potential": "phi_V", "positive": "positive_m3",
        "occupancy": "occupancy", "trace_density": "trace_state_m3", "trace_potential": "trace_potential_V"}.items()}
    if drive != "potential":
        for mode in ["electron", "positive", "occupancy", "trace_density"]:
            changes[mode][:] = x
        changes["hole"][:] = -x
        inactive = np.ones(changes["positive"].size, dtype=bool)
        inactive[fixture["prepared"]["grid"]["positive_nodes"]] = False
        changes["positive"][inactive] = 0.0
    if drive != "density":
        changes["potential"][:] = x
        changes["trace_potential"][:] = x
    right, inc = lift_physical_inputs(left, VT, changes, left.time + 1.0)
    restored = decode_point(json.loads(json.dumps(encode_point(right))), right.state.layout)
    reconstructed = restored_increment(left, restored)
    deltas80, refs80 = lift_reference(left, changes, 80)
    deltas100, refs100 = lift_reference(left, changes, 100)
    comparisons, uncertainties = {}, {}
    for name in left.state.fields:
        with localcontext() as ctx:
            ctx.prec = 100
            scale = Decimal(1)
            if name == "occupancy":
                scale = Decimal(NC08["decimal_inputs"]["n0"])
            elif name in {"phi_V", "trace_potential_V"}:
                scale = Decimal(NC08["decimal_inputs"]["n0"]) / binary(VT)
            reference80 = [scale * v for v in deltas80[name]]
            reference100 = [scale * v for v in deltas100[name]]
        for label, change in [(name, inc), (name + "_restored", reconstructed)]:
            actual = DoubleArray.from_dd(change.field(name).as_dd() * float(scale))
            comparisons[label] = close_record(actual, reference80, NC08["density_increment_atol"], NC08["density_increment_rtol"])
        uncertainties[name] = reference_uncertainty(reference80, reference100, NC08["density_increment_atol"], NC08["density_increment_rtol"])
    grid = fixture["prepared"]["grid"]
    nodes = np.asarray(grid["coordinates_m"])
    selected = np.ones(nodes.size - 1, dtype=bool)
    for interface in grid["interface_positions_m"]:
        selected &= ~((nodes[:-1] <= interface) & (nodes[1:] >= interface))
    indices = np.flatnonzero(selected)
    pairs = np.column_stack((indices, indices + 1))
    face = FaceGeometry(pairs, np.diff(nodes)[indices])
    layers = fixture["prepared"]["physical_stack"]["layers"]
    assert fixture["prepared"]["physical_stack"]["T"] == 300.0
    current_comparisons, current_uncertainties = {}, {}
    for carrier, density_id in [("electron", "n_m3"), ("hole", "p_m3")]:
        mobility = [layer["params"]["mu_n" if carrier == "electron" else "mu_p"] for layer in layers]
        assert len(set(mobility)) == 1
        diffusion = mobility[0] * VT
        args = dict(density_id=density_id, thermal_voltage_V=VT, diffusion_m2_s=diffusion, charge_C=Q, carrier=carrier)
        value = sg_current_increment(left, right, inc, face, **args)
        restored_value = sg_current_increment(left, restored, reconstructed, face, **args)
        refs = []
        for precision, fields in [(80, refs80), (100, refs100)]:
            before = sg_reference(words(left.state.field(density_id), precision), words(left.state.field("phi_V"), precision), pairs,
                                  [binary(v) for v in face.spacing_m], [binary(diffusion)] * len(pairs), binary(Q), binary(VT), carrier, precision)
            after = sg_reference(fields[density_id], fields["phi_V"], pairs,
                                 [binary(v) for v in face.spacing_m], [binary(diffusion)] * len(pairs), binary(Q), binary(VT), carrier, precision)
            with localcontext() as ctx:
                ctx.prec = precision
                refs.append([a - b for a, b in zip(after, before, strict=True)])
        gate = NC08["weak_flux"]
        current_comparisons[carrier] = close_record(value, refs[0], gate["atol_A_m2"], gate["rtol"])
        current_comparisons[carrier + "_restored"] = close_record(restored_value, refs[0], gate["atol_A_m2"], gate["rtol"])
        current_uncertainties[carrier] = reference_uncertainty(*refs, gate["atol_A_m2"], gate["rtol"])
    conclude(request, {"family": "extended_R1_origin_lift", "origin": case_name, "drive": drive, "input": text,
                       "probe_is_new_not_historical_coordinate_replay": True, "fixture_sha256": FIXTURE_SHA,
                       "source_words": "original exported binary64 physical fields; no fabricated historical DD/QF reconstruction",
                       "origin_identity": left.identity, "lifted_identity": right.identity,
                       "field_comparisons": comparisons, "field_reference_uncertainty": uncertainties,
                       "current_comparisons": current_comparisons, "current_reference_uncertainty": current_uncertainties,
                       "selected_bulk_faces": indices.tolist(), "excluded_TE_faces": np.flatnonzero(~selected).tolist(),
                       "Fermi_table_or_full_R1_qualification": False},
             {"lifted_physical_fields": all(v["passed"] for v in comparisons.values()),
              "actual_bulk_SG_consumption": all(v["passed"] for v in current_comparisons.values()),
              "physical_sign_and_exact_zero": all(v["sign_and_exact_zero_match"] for v in comparisons.values()),
              "SG_sign_and_exact_zero": all(v["sign_and_exact_zero_match"] for v in current_comparisons.values()),
              "reference_uncertainty": all(v["passed"] for v in [*uncertainties.values(), *current_uncertainties.values()]),
              "codec_identity": restored.identity == right.identity,
              "all_physical_words_preserved": all(restored.state.field(k).identity_bytes() == right.state.field(k).identity_bytes() for k in right.state.fields)})


def test_extended_coupled_voltage_lift_retains_carrier_history(request):
    _, anchor = base_point()
    extra_support = Support("qf", "global", (2,))
    layout = Layout((*anchor.state.layout.supports, extra_support),
                    (*anchor.state.layout.variables,
                     VariableSpec("dqfn_V", "carrier", "qf", (2,), VOLT),
                     VariableSpec("dqfp_V", "carrier", "qf", (2,), VOLT)), anchor.state.layout.equations)
    state = StateView(layout, [*exact_root_fields(anchor), ("dqfn_V", DoubleArray([0.0, 0.0])),
                               ("dqfp_V", DoubleArray([0.0, 0.0]))])
    left = Point(0.0, np.zeros(layout.size), np.array([0.0]), state, "voltage-lift-physical-origin")
    lift = np.array([VT * 1e-17, -VT * 1e-17])
    right, inc = lift_physical_inputs(left, VT, {}, 1.0, voltage_lift_V=lift, inputs=np.array([lift[0]]))
    nsum = inc.field("dqfn_V").as_dd() + inc.field("phi_V").as_dd()
    psum = inc.field("dqfp_V").as_dd() - inc.field("phi_V").as_dd()
    conclude(request, {"family": "extended_voltage_lift", "lift_V": lift.tolist(),
                       "left_identity": left.identity, "right_identity": right.identity},
             {"carrier_n_unchanged": np.all(inc.field("n_m3").as_dd() == 0),
              "carrier_p_unchanged": np.all(inc.field("p_m3").as_dd() == 0),
              "n_QF_potential_cancel": np.all(nsum == 0), "p_QF_potential_cancel": np.all(psum == 0),
              "voltage_lift_remains_nonzero": np.any(inc.field("phi_V").as_dd() != 0),
              "public_input_is_recorded": right.inputs[0] == lift[0] and left.inputs[0] == 0.0})


def test_extended_strict_zero_and_invalid_input_guards(request):
    coords, left = base_point(n0=0)
    origin = Point(left.time, left.y, left.inputs, StateView(left.state.layout, exact_root_fields(left)),
                   "external-zero-input")
    face = FaceGeometry(np.array([[0, 1]]), np.array([1e-8]))
    value = sg_current(left.state, face, density_id="n_m3", thermal_voltage_V=VT,
                       diffusion_m2_s=1e-4, charge_C=Q)
    disabled = sg_current(base_point()[1].state, face, density_id="n_m3", thermal_voltage_V=VT,
                          diffusion_m2_s=0.0, charge_C=Q)
    no_sites = ElectronTrapCapture(1e-16, 0.0, 1e22).net_rate(left.state)
    no_capture = ElectronTrapCapture(0.0, 1e16, 1e22).net_rate(left.state)
    metadata = face.pairs; metadata.shape = (2, 1)
    errors = []
    for function in [lambda: FaceGeometry(np.array([[0, 1]]), np.array([0.0])),
                     lambda: lift_physical_inputs(origin, VT, {"electron": [1e-17]}, 1.0),
                     lambda: lift_physical_inputs(origin, VT, {"potential": [np.nan, 0.0]}, 1.0),
                     lambda: lift_physical_inputs(left, VT, {}, 1.0)]:
        with pytest.raises(ContractError) as caught:
            function()
        errors.append(caught.value.reason)
    conclude(request, {"family": "extended_zero_guards", "errors": errors},
             {"zero_species": np.all(value.as_dd() == 0), "zero_diffusivity": np.all(disabled.as_dd() == 0),
              "disabled_capture_exact_zero": np.all(no_sites.as_dd() == 0) and np.all(no_capture.as_dd() == 0),
              "geometry_metadata_owned": face.pairs.shape == (1, 2),
              "expected_rejections": errors == ["invalid_face_geometry", "input_lift_coordinate_shape_mismatch", "nonfinite_array", "incompatible_state_authority"]})


@pytest.mark.parametrize("text", TEXTS)
@pytest.mark.parametrize("gauge", [0.0, 1.0])
def test_diagnostic_declared_increment_and_endpoint_references(text, gauge, request):
    """Map, stored finite-delta words and external endpoint words are distinct inputs.

    Old Attempt02/03 cross-representation failures stay failed. Each current
    comparison binds its own Point authority and its independent exact input.
    """
    gate = NC08["weak_flux"]
    coordinates, left = base_point(gauge=gauge)
    changes = {"p_m3": [0, float(text)], "phi_V": [0, -VT * float(text)]}
    right, inc = advance(coordinates, left, changes)
    face = FaceGeometry(np.array([[0, 1]]), np.array([gate["dx_m"]]))
    args = dict(density_id="p_m3", thermal_voltage_V=VT, diffusion_m2_s=gate["D_m2_s"],
                charge_C=float(gate["q_C"]), carrier="hole")
    restored_left = decode_point(json.loads(json.dumps(encode_point(left))), left.state.layout)
    restored_right = decode_point(json.loads(json.dumps(encode_point(right))), right.state.layout)
    restored_inc = StateIncrement.from_points(restored_left, restored_right)

    # The finite words define a linear map, not a guessed delta for the original map.
    linear = RelativeCoordinates(left.state.layout, {name: "linear" for name in left.state.fields})
    stored_left = linear.initial(StateView(left.state.layout, exact_root_fields(left)))
    step = DoubleArray(np.concatenate([inc.field(v.id).high.ravel() for v in left.state.layout.variables]),
                       np.concatenate([inc.field(v.id).low.ravel() for v in left.state.layout.variables]))
    stored_right, stored_inc = linear.advance(stored_left, step, right.time, right.inputs)

    # Explicit external inputs test the old rounded-word meaning; this is not a rebase.
    endpoint_left = Point(left.time, left.y, left.inputs,
                          StateView(left.state.layout, exact_root_fields(left)), "external-endpoint-left")
    endpoint_right = Point(right.time, right.y, right.inputs, StateView(right.state.layout, [
        (name, DoubleArray(value.high, value.low)) for name, value in right.state.fields.items()
    ]), "external-endpoint-right")
    endpoint_inc = StateIncrement.from_points(endpoint_left, endpoint_right)
    pairs = {"mapped": (left, right, inc), "stored_delta": (stored_left, stored_right, stored_inc),
             "encoded_endpoint": (endpoint_left, endpoint_right, endpoint_inc)}
    references = {name: [] for name in pairs}
    for precision in (80, 100):
        with localcontext() as ctx:
            ctx.prec = precision
            n0 = words(left.state.field("p_m3"), precision)
            phi0 = words(left.state.field("phi_V"), precision)
            mapped_n, mapped_phi = relative_reference(left, changes, "p_m3", precision)
            stored_n = [v+d for v,d in zip(n0, words(inc.field("p_m3"), precision), strict=True)]
            stored_phi = [v+d for v,d in zip(phi0, words(inc.field("phi_V"), precision), strict=True)]
            for name, density, potential in [
                ("mapped", mapped_n, mapped_phi), ("stored_delta", stored_n, stored_phi),
                ("encoded_endpoint", words(right.state.field("p_m3"), precision),
                 words(right.state.field("phi_V"), precision)),
            ]:
                references[name].append(sg_reference(density, potential, face.pairs,
                    [binary(gate["dx_m"])], [binary(gate["D_m2_s"])], binary(float(gate["q_C"])),
                    binary(VT), "hole", precision))
    checks = {}
    for name, (before, after, delta) in pairs.items():
        checks[name] = close_record(sg_current(after.state, face, **args), references[name][0],
                                   gate["atol_A_m2"], gate["rtol"])
        checks[name+"_increment"] = close_record(sg_current_increment(before, after, delta, face, **args),
                                                references[name][0], gate["atol_A_m2"], gate["rtol"])
    checks["restored"] = close_record(sg_current_increment(restored_left, restored_right, restored_inc, face, **args),
                                     references["mapped"][0], gate["atol_A_m2"], gate["rtol"])
    uncertainties = {name: reference_uncertainty(*refs, gate["atol_A_m2"], gate["rtol"])
                     for name, refs in references.items()}
    conclude(request, {"family": "diagnostic_distinct_input_authorities", "input": text, "gauge_V": gauge,
                       "values": checks, "reference_uncertainty": uncertainties,
                       "endpoint_identities": {name: after.identity for name, (_,after,_) in pairs.items()},
                       "prior_failures": "Original DevicePrototypeAttempt02/03 remain failed; current maps have explicit distinct meanings."},
             {"separate_exact_inputs": all(v["passed"] and v["sign_and_exact_zero_match"] for v in checks.values()),
              "independent_reference_precision": all(v["passed"] for v in uncertainties.values()),
              "distinct_identities": len({after.identity for _,after,_ in pairs.values()}) == 3,
              "restored_identity": restored_right.identity == right.identity})


@pytest.mark.parametrize("carrier", ["electron", "hole"])
@pytest.mark.parametrize("weak", [0.0, 1e-40, -1e-40])
def test_localized_input_lift_SG_consumes_weak_activity(carrier, weak, request):
    _, seed = base_point(gauge=1.0)
    origin = Point(seed.time, seed.y, seed.inputs, StateView(seed.state.layout, exact_root_fields(seed)),
                   "localized-r1-physical-input")
    changes = {"potential": [0.1, 0.3], "electron": [0.0, weak], "hole": [0.0, -weak]}
    right, inc = lift_physical_inputs(origin, VT, changes, 1.0)
    gate = NC08["weak_flux"]
    face = FaceGeometry(np.array([[0, 1]]), np.array([gate["dx_m"]]))
    density_id = "n_m3" if carrier == "electron" else "p_m3"
    args = dict(density_id=density_id, thermal_voltage_V=VT, diffusion_m2_s=gate["D_m2_s"],
                charge_C=float(gate["q_C"]), carrier=carrier)
    refs = []
    for precision in (100, 120):
        with localcontext() as ctx:
            ctx.prec = precision
            role = changes[carrier]
            sign = 1 if carrier == "electron" else -1
            density = [binary(float(NC08["decimal_inputs"]["n0"])) *
                       (binary(z) + sign*binary(p)).exp() for z,p in zip(role, changes["potential"], strict=True)]
            potential = [Decimal(1) + binary(VT)*binary(p) for p in changes["potential"]]
            refs.append(sg_reference(density, potential, face.pairs, [binary(gate["dx_m"])],
                        [binary(gate["D_m2_s"])], binary(float(gate["q_C"])), binary(VT), carrier, precision))
    # Exactly zero drive has exactly zero physical current; Decimal exp subtraction
    # can leave a precision-dependent residue, which is retained as reference error.
    if weak == 0:
        assert max(abs(v) for values in refs for v in values) < Decimal("1e-70")
        refs = [[Decimal(0)], [Decimal(0)]]
    current = close_record(sg_current(right.state, face, **args), refs[0], 0.0, gate["rtol"])
    delta = close_record(sg_current_increment(origin, right, inc, face, **args), refs[0], 0.0, gate["rtol"])
    restored = decode_point(encode_point(right), right.state.layout)
    restored_value = close_record(sg_current(restored.state, face, **args), refs[0], 0.0, gate["rtol"])
    uncertainty = reference_uncertainty(*refs, 0.0, gate["rtol"])
    conclude(request, {"family":"localized_R1_weak_SG","carrier":carrier,"weak":weak,
                       "current":current,"finite_change":delta,"restored":restored_value,
                       "reference_uncertainty":uncertainty},
             {"current":current["passed"] and current["sign_and_exact_zero_match"],
              "finite_change":delta["passed"] and delta["sign_and_exact_zero_match"],
              "codec":restored_value["passed"] and restored_value["sign_and_exact_zero_match"],
              "reference_precision":uncertainty["passed"]})


@pytest.mark.parametrize("carrier", ["electron", "hole"])
@pytest.mark.parametrize("weak", [1e-40, -1e-40])
@pytest.mark.parametrize("step", [0.0, 0.1, 2.0**-24, 2.0**-25, 2.0**-26, 1e-8, -1e-8, 1e-9, 1e-17])
def test_nonzero_weak_current_survives_common_potential_step(carrier, weak, step, request):
    """A common potential primitive scales the populations and their nonzero flux."""
    _, seed = base_point(gauge=1.0)
    origin = Point(seed.time, seed.y, seed.inputs, StateView(seed.state.layout, exact_root_fields(seed)),
                   "two-transition-r1-origin")
    initial = {"potential": [0.1, 0.3], "electron": [0.0, weak], "hole": [0.0, -weak]}
    left, _ = lift_physical_inputs(origin, VT, initial, 1.0)
    right, inc = lift_physical_inputs(left, VT, {"potential": [step, step]}, 2.0)
    gate = NC08["weak_flux"]
    face = FaceGeometry(np.array([[0, 1]]), np.array([gate["dx_m"]]))
    args = dict(density_id="n_m3" if carrier == "electron" else "p_m3", thermal_voltage_V=VT,
                diffusion_m2_s=gate["D_m2_s"], charge_C=float(gate["q_C"]), carrier=carrier)
    refs = []
    for precision in (100, 120):
        with localcontext() as ctx:
            ctx.prec = precision
            sign = 1 if carrier == "electron" else -1
            densities = [binary(float(NC08["decimal_inputs"]["n0"])) *
                         (binary(z)+sign*binary(p)).exp()
                         for z,p in zip(initial[carrier], initial["potential"], strict=True)]
            potentials = [Decimal(1)+binary(VT)*binary(p) for p in initial["potential"]]
            current = sg_reference(densities, potentials, face.pairs, [binary(gate["dx_m"])],
                                   [binary(gate["D_m2_s"])], binary(float(gate["q_C"])),
                                   binary(VT), carrier, precision)[0]
            refs.append([current*((sign*binary(step)).exp()-1)])
    delta = close_record(sg_current_increment(left, right, inc, face, **args),
                         refs[0], 0.0, gate["rtol"])
    restored_left = decode_point(encode_point(left), left.state.layout)
    restored_right = decode_point(encode_point(right), right.state.layout)
    restored = close_record(sg_current_increment(restored_left, restored_right,
        StateIncrement.from_points(restored_left, restored_right), face, **args), refs[0], 0.0, gate["rtol"])
    uncertainty = reference_uncertainty(*refs, 0.0, gate["rtol"])
    conclude(request, {"family":"nonzero_R1_weak_current_change", "carrier":carrier, "weak":weak, "step":step,
                       "finite_change":delta, "restored":restored, "reference_uncertainty":uncertainty},
             {"finite_change":delta["passed"] and delta["sign_and_exact_zero_match"],
              "codec":restored["passed"] and restored["sign_and_exact_zero_match"],
              "reference_precision":uncertainty["passed"]})


@pytest.mark.parametrize("x", [-50.0, -1.0, -0.250000001, -0.25, 0.0, 0.25, 0.250000001, 1.0, 50.0])
@pytest.mark.parametrize("step", [0.0, -1e-40, 1e-40, -1e-8, 0.1])
def test_bernoulli_finite_log_ratio_preserves_tiny_input(x, step, request):
    from scripts.benchmarks.precision_prototype import bernoulli

    x0, dx = DD([x]), DD([step])
    x1 = x0 + dx
    b0, b1 = [bernoulli(DoubleArray.from_dd(z)).as_dd() for z in (x0, x1)]
    actual = DoubleArray.from_dd(_bernoulli_log_change(x0, x1, dx, b0, b1))
    references = []
    # The direct oracle at x=0, dx=1e-40 loses about 80 decimal digits
    # through exp(x)-1 followed by log(B). Attempt12 retained the failing
    # 100-digit reference-uncertainty check; increase only oracle precision.
    for precision in (140, 180):
        with localcontext() as ctx:
            ctx.prec = precision
            # Independent direct Decimal transcendental evaluation, including
            # x=0 analytically. Preserve the input dx before adding endpoints.
            def direct_bernoulli(z):
                return Decimal(1) if z == 0 else z / (z.exp() - 1)
            references.append([(direct_bernoulli(binary(x) + binary(step)) /
                                direct_bernoulli(binary(x))).ln()])
    check = close_record(actual, references[1], 0.0, 1e-24)
    uncertainty = reference_uncertainty(*references, 0.0, 1e-24)
    conclude(request, {"family": "Bernoulli_finite_log_ratio", "x": x, "step": step,
                       "comparison": check, "reference_uncertainty": uncertainty},
             {"finite_ratio": check["passed"] and check["sign_and_exact_zero_match"],
              "reference_precision": uncertainty["passed"]})


@pytest.mark.parametrize("carrier", ["electron", "hole"])
@pytest.mark.parametrize("weak", [1e-40, -1e-40])
@pytest.mark.parametrize("potential_step", [1e-40, -1e-40])
def test_weak_current_finite_change_from_nonuniform_potential(carrier, weak, potential_step, request):
    """A retained tiny field change changes an already tiny net current."""
    _, seed = base_point(gauge=1.0)
    origin = Point(seed.time, seed.y, seed.inputs, StateView(seed.state.layout, exact_root_fields(seed)),
                   "two-transition-r1-nonuniform-origin")
    initial = {"potential": [0.1, 0.3], "electron": [0.0, weak], "hole": [0.0, -weak]}
    left, _ = lift_physical_inputs(origin, VT, initial, 1.0)
    right, inc = lift_physical_inputs(left, VT, {"potential": [0.0, potential_step]}, 2.0)
    gate = NC08["weak_flux"]
    face = FaceGeometry([[0, 1]], [gate["dx_m"]])
    args = dict(density_id="n_m3" if carrier == "electron" else "p_m3", thermal_voltage_V=VT,
                diffusion_m2_s=gate["D_m2_s"], charge_C=float(gate["q_C"]), carrier=carrier)
    references = []
    for precision in (140, 180):
        with localcontext() as ctx:
            ctx.prec = precision
            sign = 1 if carrier == "electron" else -1
            currents = []
            for delta in (0.0, potential_step):
                primitive = [binary(0.1), binary(0.3) + binary(delta)]
                densities = [binary(float(NC08["decimal_inputs"]["n0"])) *
                             (binary(z) + sign*p).exp()
                             for z, p in zip(initial[carrier], primitive, strict=True)]
                potentials = [Decimal(1) + binary(VT)*p for p in primitive]
                currents.append(sg_reference(densities, potentials, face.pairs, [binary(gate["dx_m"])],
                    [binary(gate["D_m2_s"])], binary(float(gate["q_C"])), binary(VT), carrier, precision)[0])
            references.append([currents[1] - currents[0]])
    actual = sg_current_increment(left, right, inc, face, **args)
    check = close_record(actual, references[1], 0.0, gate["rtol"])
    uncertainty = reference_uncertainty(*references, 0.0, gate["rtol"])
    restored_left, restored_right = [decode_point(encode_point(p), p.state.layout) for p in (left, right)]
    restored = sg_current_increment(restored_left, restored_right,
                                   StateIncrement.from_points(restored_left, restored_right), face, **args)
    conclude(request, {"family": "nonuniform_R1_weak_current_change", "carrier": carrier,
                       "weak": weak, "potential_step": potential_step, "comparison": check,
                       "reference_uncertainty": uncertainty},
             {"finite_change": check["passed"] and check["sign_and_exact_zero_match"],
              "reference_precision": uncertainty["passed"],
              "codec": actual.identity_bytes() == restored.identity_bytes()})


@pytest.mark.parametrize("carrier", ["electron", "hole"])
@pytest.mark.parametrize("potential_step", [0.0, 1e-18, -1e-18])
def test_nonzero_affine_density_current_increment(carrier, potential_step, request):
    """Linear physical-density increments remain valid away from equilibrium."""
    _, seed = base_point()
    density_id = "n_m3" if carrier == "electron" else "p_m3"
    fields = dict(exact_root_fields(seed))
    n0 = float(NC08["decimal_inputs"]["n0"])
    fields[density_id] = DoubleArray([n0, 1.2*n0])
    fields["phi_V"] = DoubleArray([-0.1, 0.2])
    coordinates = RelativeCoordinates(seed.state.layout, {name: "linear" for name in fields})
    left = coordinates.initial(StateView(seed.state.layout, fields.items()))
    changes = {density_id: [n0*1e-17, 1.2*n0*1e-17], "phi_V": [0.0, potential_step]}
    right, inc = advance(coordinates, left, changes)
    gate = NC08["weak_flux"]
    face = FaceGeometry([[0, 1]], [gate["dx_m"]])
    args = dict(density_id=density_id, thermal_voltage_V=VT, diffusion_m2_s=gate["D_m2_s"],
                charge_C=float(gate["q_C"]), carrier=carrier)
    references = []
    for precision in (100, 140):
        with localcontext() as ctx:
            ctx.prec = precision
            n, phi = words(left.state.field(density_id), precision), words(left.state.field("phi_V"), precision)
            next_n = [v+binary(d) for v, d in zip(n, changes[density_id], strict=True)]
            next_phi = [v+binary(d) for v, d in zip(phi, changes["phi_V"], strict=True)]
            currents = [sg_reference(density, potential, face.pairs, [binary(gate["dx_m"])],
                [binary(gate["D_m2_s"])], binary(float(gate["q_C"])), binary(VT), carrier, precision)[0]
                for density, potential in ((n, phi), (next_n, next_phi))]
            references.append([currents[1]-currents[0]])
    actual = sg_current_increment(left, right, inc, face, **args)
    check = close_record(actual, references[1], 0.0, gate["rtol"])
    uncertainty = reference_uncertainty(*references, 0.0, gate["rtol"])
    conclude(request, {"family": "nonzero_affine_density_current_increment", "carrier": carrier,
                       "potential_step": potential_step, "comparison": check,
                       "reference_uncertainty": uncertainty},
             {"finite_change": check["passed"] and check["sign_and_exact_zero_match"],
              "reference_precision": uncertainty["passed"]})


@pytest.mark.parametrize("carrier", ["electron", "hole"])
@pytest.mark.parametrize("common_change", [1e21, -1e21])
def test_affine_common_density_preserves_diffusion_current(carrier, common_change, request):
    """At zero electric field, constant density gradients give constant flux."""
    _, seed = base_point()
    density_id = "n_m3" if carrier == "electron" else "p_m3"
    fields = dict(exact_root_fields(seed))
    n0 = float(NC08["decimal_inputs"]["n0"])
    weak = 1e-18
    fields[density_id] = DoubleArray([n0, n0], [0.0, weak])
    coordinates = RelativeCoordinates(seed.state.layout, {name: "linear" for name in fields})
    left = coordinates.initial(StateView(seed.state.layout, fields.items()))
    right, inc = advance(coordinates, left, {density_id: [common_change, common_change]})
    gate = NC08["weak_flux"]
    face = FaceGeometry([[0, 1]], [gate["dx_m"]])
    args = dict(density_id=density_id, thermal_voltage_V=VT, diffusion_m2_s=gate["D_m2_s"],
                charge_C=float(gate["q_C"]), carrier=carrier)
    with localcontext() as ctx:
        ctx.prec = 100
        expected = binary(float(gate["q_C"]))*binary(gate["D_m2_s"])/binary(gate["dx_m"])*binary(weak)
        if carrier == "hole":
            expected = -expected
    values = [close_record(sg_current(p.state, face, **args), [expected], 0.0, gate["rtol"])
              for p in (left, right)]
    delta = sg_current_increment(left, right, inc, face, **args)
    conclude(request, {"family": "affine_constant_diffusion_current", "carrier": carrier,
                       "common_density_change": common_change, "currents": values,
                       "finite_change_words": [str(v) for v in words(delta, 100)]},
             {"physical_values": all(v["passed"] and v["sign_and_exact_zero_match"] for v in values),
              "exact_constant_gradient_flux": np.all(delta.as_dd() == 0)})
