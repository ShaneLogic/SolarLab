"""Independent same-discrete-equation Decimal checks for the V7 prototype.

Freeze healthy inputs BEFORE installing implementation faults.  Coefficients
are independently assembled in the original float64 order and checked against
the healthy material.  Decimal then evaluates that frozen discrete problem;
replacing it by exact-real geometry would introduce a different Poisson root.

This module deliberately imports no production solver, Bernoulli, flux,
divergence or Poisson helper.  A passing side/pair check is an operator check,
not a trajectory, fault-campaign, replay, cost or D3 qualification.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping


MATERIAL_SCHEMA = "R1V7FrozenHealthyMaterialV1"
ORACLE_TARGET = Decimal("1e-10")
ORIGINAL_GATE = Decimal("1e-6")
PHYSICAL_FLOOR = Decimal(1)
# Pinned original source literals, interpreted as the same binary64 values.
# These are not imported from mutable implementation modules during a fault.
Q_C = 1.602176634e-19
EPS0_F_M = 8.854187817e-12
STATE_FIELDS = ("phi_V", "n_m3", "p_m3", "positive_m3", "sheet_charge_C_m2")
OBSERVED_UNITS = {
    "positive_flux_m2_s": "m^-2 s^-1",
    "positive_rate_m3_s": "m^-3 s^-1",
    "boundary_flux_m2_s": "m^-2 s^-1",
}


def dec(value):
    """Floats mean exact binary values; strings mean exact decimal values."""
    if isinstance(value, bool):
        raise TypeError("booleans are not numerical evidence")
    if isinstance(value, Decimal):
        result = value
    elif isinstance(value, (str, int)):
        result = Decimal(value)
    else:
        result = Decimal.from_float(float(value))
    if not result.is_finite():
        raise ValueError("numerical evidence must be finite")
    return result


def _vector(value):
    if isinstance(value, Mapping):
        if set(value) != {"hi", "lo"}:
            raise ValueError("a split vector requires exactly hi and lo")
        high, low = list(value["hi"]), list(value["lo"])
        if len(high) != len(low):
            raise ValueError("split vector hi/lo lengths differ")
        return [dec(a) + dec(b) for a, b in zip(high, low)]
    if isinstance(value, (str, bytes)):
        raise ValueError("a vector cannot be a scalar string")
    return [dec(item) for item in value]


def _strings(values):
    return [str(value) for value in values]


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def _float_vector(value, name, length=None):
    values = [float(item) for item in value]
    if length is not None and len(values) != length:
        raise ValueError(f"{name} has the wrong length")
    if not all(math.isfinite(item) for item in values):
        raise ValueError(f"{name} must contain finite values")
    return values


def _assert_binary_equal(actual, expected, name):
    actual = _float_vector(actual, name, len(expected))
    mismatches = [index for index, (a, b) in enumerate(zip(actual, expected))
                  if a.hex() != b.hex()]
    if mismatches:
        raise ValueError(f"healthy {name} differs from independent coefficient law "
                         f"at indices {mismatches[:8]}")


def freeze_healthy_material(grid, material, *, source_identity, control_label="D"):
    """Copy and hash healthy raw inputs, independently checking coefficient law.

    Only this function reads a material object.  After it returns, mutations to
    that object or its cached arrays cannot change any oracle calculation.
    A/C have frozen ions; B/D have the declared positive-ion transport law.
    """
    if not isinstance(source_identity, str) or not source_identity.strip():
        raise ValueError("an explicit healthy source identity is required")
    if control_label not in ("A", "B", "C", "D"):
        raise ValueError("control_label must be A, B, C or D")
    if not bool(material.ion_steric_diffusion_only):
        raise ValueError("V7 oracle covers the frozen diffusion-only steric law")
    if bool(getattr(material, "has_dual_ions", False)):
        raise ValueError("V7 oracle covers the declared positive species only")
    x = _float_vector(grid, "grid_m")
    size = len(x)
    if size < 3 or any(b <= a for a, b in zip(x, x[1:])):
        raise ValueError("grid must be strictly increasing with at least 3 nodes")
    eps = _float_vector(material.eps_r, "eps_r", size)
    diffusion = _float_vector(material.D_ion_node, "D_ion_node", size)
    limits = _float_vector(material.P_lim_node, "P_lim_node", size)
    if min(eps) <= 0 or min(diffusion) < 0 or min(limits) <= 0:
        raise ValueError("invalid permittivity, diffusion or site limit")
    positions = _float_vector(material.iface_qss_interface_positions_m, "interfaces")
    if any(b <= a for a, b in zip(positions, positions[1:])):
        raise ValueError("interface positions must be ordered")
    distances = [b - a for a, b in zip(x, x[1:])]
    faces = [x[0], *[(a + b) / 2 for a, b in zip(x, x[1:])], x[-1]]
    capacitance = [EPS0_F_M * (2.0 * a * b / (a + b)) / dx
                   for a, b, dx in zip(eps, eps[1:], distances)]
    interface_nodes, sheet_weights = [], []
    used = set()
    for position in positions:
        right = next((j for j, node in enumerate(x) if node >= position), size)
        if right == 0 or right == size or x[right] == position or right in used:
            raise ValueError("each interface must occupy its own grid gap")
        used.add(right)
        left = right - 1
        dl, dr = position - x[left], x[right] - position
        faces[right] = position
        capacitance[left] = EPS0_F_M / (dl / eps[left] + dr / eps[right])
        cl, cr = EPS0_F_M * eps[left] / dl, EPS0_F_M * eps[right] / dr
        sheet_weights.append([cl / (cl + cr), cr / (cl + cr)])
        interface_nodes.append([left, right])
    widths = [b - a for a, b in zip(faces, faces[1:])]
    if min(widths) <= 0:
        raise ValueError("physical control volumes must be positive")
    diffusion_faces = [2.0 * a * b / (a + b) if a + b > 0 else 0.0
                       for a, b in zip(diffusion, diffusion[1:])]
    # These are checked BEFORE faults and never read by evaluate_state.
    _assert_binary_equal(material.D_ion_face, diffusion_faces, "D_ion_face")
    _assert_binary_equal(material.dx_cell, widths, "dx_cell")
    _assert_binary_equal(material.poisson_factor.C, capacitance, "Poisson C")
    _assert_binary_equal(material.poisson_factor.h_cell, widths[1:-1], "Poisson h_cell")
    _assert_binary_equal(material.physical_cell_faces_m, faces, "physical faces")
    if list(material.iface_qss_left_nodes) != [pair[0] for pair in interface_nodes] or list(
            material.iface_qss_right_nodes) != [pair[1] for pair in interface_nodes]:
        raise ValueError("healthy interface nodes differ from independent geometry")
    vt = float(material.V_T_device)
    if not math.isfinite(vt) or vt <= 0:
        raise ValueError("thermal voltage must be finite and positive")
    raw = {"grid_m": x, "eps_r": eps, "D_ion_node_m2_s": diffusion,
           "P_lim_node_m3": limits, "interface_positions_m": positions,
           "N_A_m3": _float_vector(material.N_A, "N_A", size),
           "N_D_m3": _float_vector(material.N_D, "N_D", size),
           "P_ion0_m3": _float_vector(material.P_ion0, "P_ion0", size)}
    result = {
        "schema": MATERIAL_SCHEMA, "source_identity": source_identity,
        "control_label": control_label, "ion_enabled": control_label in ("B", "D"),
        "raw": {name: _strings(map(dec, values)) for name, values in raw.items()},
        "constants": {"q_C": str(dec(Q_C)), "eps0_F_m": str(dec(EPS0_F_M)),
                      "thermal_voltage_V": str(dec(vt))},
        "coefficients": {
            "face_distance_m": _strings(map(dec, distances)),
            "physical_width_m": _strings(map(dec, widths)),
            "poisson_capacitance_F_m2": _strings(map(dec, capacitance)),
            "D_ion_face_m2_s": _strings(map(dec, diffusion_faces)),
            "sheet_weights": [_strings(map(dec, pair)) for pair in sheet_weights],
            "interface_nodes": interface_nodes,
        },
        "coefficient_contract": "independent_original_float64_assembly_then_exact_inputs",
        "healthy_cached_coefficient_identity": "bit_identical",
        "boundary_contract": "blocking_positive_particle_flux_at_both_contacts",
        "scope": "same_discrete_coefficients_not_exact_real_geometry",
    }
    result["content_sha256"] = _digest(result)
    return result


def _check_frozen(frozen):
    if frozen.get("schema") != MATERIAL_SCHEMA:
        raise ValueError("unsupported frozen material schema")
    content = {key: value for key, value in frozen.items() if key != "content_sha256"}
    if frozen.get("content_sha256") != _digest(content):
        raise ValueError("frozen healthy material content hash mismatch")


def _state_vectors(state, size, sheet_count):
    precise = state.get("precision")
    if any(key.startswith("precision_") for key in state):
        fields = {}
        for name in STATE_FIELDS:
            high, low = f"precision_{name}_hi", f"precision_{name}_lo"
            if high not in state or low not in state:
                raise ValueError("declared flat precise snapshot is missing a required precision field")
            fields[name] = {"hi": state[high], "lo": state[low]}
    elif precise is not None:
        if precise.get("representation") != "float64-pair-v1":
            raise ValueError("unsupported precision representation")
        fields = precise.get("fields", {})
        if any(name not in fields for name in STATE_FIELDS):
            raise ValueError("declared precise snapshot is missing a required precision field")
    else:
        fields = state
    values, explicit_low = {}, []
    for name in STATE_FIELDS:
        if name not in fields:
            raise ValueError(f"state field {name} is required")
        values[name] = _vector(fields[name])
        if len(values[name]) != (sheet_count if name == "sheet_charge_C_m2" else size):
            raise ValueError(f"state field {name} has the wrong length")
        if isinstance(fields[name], Mapping):
            explicit_low.append(name)
    if min(values["n_m3"] + values["p_m3"] + values["positive_m3"]) < 0:
        raise ValueError("populations must be nonnegative")
    return values, explicit_low


def _bernoulli_decimal(value):
    # exp() - 1 is resolved by the independently checked 80/100-digit ladder.
    return Decimal(1) if value == 0 else value / (value.exp() - Decimal(1))


def _evaluate(frozen, state, precision):
    with localcontext() as context:
        context.prec = precision
        raw, coeff = frozen["raw"], frozen["coefficients"]
        size = len(raw["grid_m"])
        values, explicit_low = _state_vectors(state, size, len(coeff["interface_nodes"]))
        phi, n, p, ions, sheets = (values[name] for name in STATE_FIELDS)
        limits = _vector(raw["P_lim_node_m3"])
        dx, widths = _vector(coeff["face_distance_m"]), _vector(coeff["physical_width_m"])
        diffusion = _vector(coeff["D_ion_face_m2_s"])
        vt, q = dec(frozen["constants"]["thermal_voltage_V"]), dec(frozen["constants"]["q_C"])
        flux = []
        for j, (distance, d) in enumerate(zip(dx, diffusion)):
            if d == 0 or not frozen["ion_enabled"]:
                flux.append(Decimal(0))
                continue
            if (ions[j] >= limits[j] * Decimal("0.999999") or
                    ions[j + 1] >= limits[j + 1] * Decimal("0.999999")):
                raise ValueError("oracle refuses clipping/non-differentiable occupancy")
            mu_left = -(Decimal(1) - ions[j] / limits[j]).ln()
            mu_right = -(Decimal(1) - ions[j + 1] / limits[j + 1]).ln()
            xi = (phi[j + 1] - phi[j]) / vt + mu_right - mu_left
            # Direct independent SG formula, not the compensated production
            # log-force form. High precision resolves its cancellation.
            flux.append(d / distance * (_bernoulli_decimal(xi) * ions[j]
                                        - _bernoulli_decimal(-xi) * ions[j + 1]))
        boundaries = [Decimal(0), Decimal(0)]
        all_faces = [boundaries[0], *flux, boundaries[1]]
        rate = [(all_faces[j] - all_faces[j + 1]) / widths[j] for j in range(size)]
        na, nd, background = (_vector(raw[name]) for name in ("N_A_m3", "N_D_m3", "P_ion0_m3"))
        rho = [q * (p[j] - n[j] + ions[j] - background[j] + nd[j] - na[j])
               for j in range(size)]
        capacitance = _vector(coeff["poisson_capacitance_F_m2"])
        displacement = [-capacitance[j] * (phi[j + 1] - phi[j]) for j in range(size - 1)]
        sheet_rhs = [Decimal(0)] * size
        for nodes, weights, charge in zip(coeff["interface_nodes"], coeff["sheet_weights"], sheets):
            for node, weight in zip(nodes, weights):
                sheet_rhs[node] += dec(weight) * charge
        residual = [displacement[j - 1] - displacement[j] + rho[j] * widths[j] + sheet_rhs[j]
                    for j in range(1, size - 1)]
        return {
            "schema": "R1V7PrecisionOracleV1", "precision_digits": precision,
            "frozen_material_sha256": frozen["content_sha256"],
            "positive_flux_m2_s": _strings(flux), "positive_rate_m3_s": _strings(rate),
            "boundary_flux_m2_s": _strings(boundaries),
            "poisson": {"residual_C_m2": _strings(residual),
                        "maximum_absolute_C_m2": str(max(map(abs, residual), default=Decimal(0))),
                        "charge_density_C_m3": _strings(rho),
                        "face_displacement_C_m2": _strings(displacement),
                        "physical_boundary_displacement_C_m2": _strings([
                            displacement[0] - rho[0] * widths[0] - sheet_rhs[0],
                            displacement[-1] + rho[-1] * widths[-1] + sheet_rhs[-1]]),
                        "qualified": None, "scope": "absolute_equation_residual_no_new_gate"},
            "state_input": {"explicit_low_fields": explicit_low,
                            "implicit_zero_low_fields": [name for name in STATE_FIELDS if name not in explicit_low]},
            "scope": "independent_operator_values_not_a_projected_or_accepted_state",
        }


def evaluate_state(frozen, state, *, precision=80):
    """Evaluate represented hi+lo state without changing it or solving Poisson."""
    _check_frozen(frozen)
    if not isinstance(precision, int) or not 50 <= precision <= 300:
        raise ValueError("Decimal precision must be between 50 and 300 digits")
    return _evaluate(frozen, state, precision)


def compare_vectors(actual, expected, *, limit=ORACLE_TARGET, units, precision=100):
    """Absolute error and symmetric peak/floor-1 relative score, never RMS."""
    with localcontext() as context:
        context.prec = precision
        a, b = _vector(actual), _vector(expected)
        if len(a) != len(b):
            raise ValueError("compared vector lengths differ")
        difference = max((abs(x - y) for x, y in zip(a, b)), default=Decimal(0))
        scale = max([PHYSICAL_FLOOR, *map(abs, a), *map(abs, b)])
        relative = difference / scale
        limit = dec(limit)
        if limit <= 0:
            raise ValueError("comparison limit must be positive")
        return {"absolute": str(difference), "scale": str(scale), "relative": str(relative),
                "limit": str(limit), "floor": "1", "passed": relative <= limit, "units": units}


def verify_side(frozen, state, observed, *, precision=80):
    """Compare actual implementation outputs with independently frozen health.

    Observations must come from an actual implementation call. This function
    has no mutation/fault parameter and cannot certify how a fault was injected.
    Missing rate or contact flux is incomplete evidence, never an inferred zero.
    """
    oracle = evaluate_state(frozen, state, precision=precision)
    refined = _evaluate(frozen, state, precision + 20)
    stability = {name: compare_vectors(oracle[name], refined[name], limit=Decimal("1e-20"), units=units)
                 for name, units in OBSERVED_UNITS.items()}
    checks, missing = {}, []
    for name, units in OBSERVED_UNITS.items():
        if name not in observed:
            missing.append(name)
            continue
        checks[name] = compare_vectors(observed[name], oracle[name], units=units)
    return {"schema": "R1V7PrecisionSideCheckV1",
            "qualified": not missing and all(item["passed"] for item in checks.values())
                         and all(item["passed"] for item in stability.values()),
            "checks": checks, "missing_observed": missing, "oracle_stability": stability,
            "poisson": oracle["poisson"], "oracle": oracle,
            "scope": "operator_only_fault_injection_provenance_verified_by_runner"}


def verify_pair(frozen, direct_state, direct_observed, eliminated_state, eliminated_observed, *, precision=80):
    """Keep original two-path gate separate from per-side implementation checks."""
    direct = verify_side(frozen, direct_state, direct_observed, precision=precision)
    eliminated = verify_side(frozen, eliminated_state, eliminated_observed, precision=precision)
    original, missing = {}, []
    for name in ("positive_flux_m2_s", "positive_rate_m3_s"):
        if name not in direct_observed or name not in eliminated_observed:
            missing.append(name)
        else:
            original[name] = compare_vectors(direct_observed[name], eliminated_observed[name],
                                             limit=ORIGINAL_GATE, units=OBSERVED_UNITS[name])
    original_passed = not missing and all(item["passed"] for item in original.values())
    return {"schema": "R1V7PrecisionPairCheckV1",
            "qualified": direct["qualified"] and eliminated["qualified"] and original_passed,
            "direct": direct, "eliminated": eliminated,
            "original_gate": {"passed": original_passed, "checks": original, "missing_observed": missing},
            "scope": "operator_pair_only_not_P1_D3_or_fault_campaign_qualification"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path, help="JSON with frozen, state and observed")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    report = verify_side(request["frozen"], request["state"], request["observed"],
                         precision=request.get("precision", 80))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
