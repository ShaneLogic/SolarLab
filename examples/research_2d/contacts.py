"""Four independent carrier-contact channels in connected 2D DC algebra.

    python -m examples.research_2d.contacts --output contacts.json
    python -m examples.research_2d.contacts --read contacts.json --sha256 SHA256
    python -m examples.research_2d.contacts --prepare prepared.json --seed FIXED_2D.json --seed-sha256 SHA256
    python -m examples.research_2d.contacts --solve result.json --seed FIXED_2D.json --seed-sha256 SHA256

The declared mixed case uses blocking n-left, fixed p-left and finite n/p-right
exchange. This command records equations and proposes one bounded steady
witness. --prepare performs only explicit contact-density initialization;
--solve runs one bounded full2D Newton from that numerical seed.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from examples.research_1d.steady import load_device
from examples.research_1d.transient import json_value
from examples.research_2d.dc_assembly import DIRECTIONAL_RTOL, DIRECTIONAL_STEP, VOLTAGE_V, coordinates, spatial_input
from examples.research_2d.steady import read_result as read_fixed_result
from solarlab.config.resolve_device import resolve_device
from solarlab.device.compile_2d import compile_device_2d
from solarlab.device.inputs import DeviceInput
from solarlab.device.resolved import PreparedDevice
from solarlab.equations.dc_2d import CompiledDevice2D, DC2DProblem, State2D
from solarlab.solvers.nonlinear.newton import _Factor
from solarlab.solvers.nonlinear.problem import GateOutcome, state_digest
from solarlab.solvers.nonlinear.steady_2d import Steady2DOptions, _checks, _diagnostics, _settings, solve_steady_state_2d

SCHEMA = "solarlab.carrier_contacts_2d.algebra.v1"
PREPARATION_SCHEMA = "solarlab.carrier_contacts_2d.preparation.v1"
STEADY_SCHEMA = "solarlab.contacts2d.steady-witness.v1"
VELOCITIES: tuple[float | None, float | None, float | None, float | None] = (0.0, None, 1e5, 1e3)


def contact_device(velocities: tuple[float | None, float | None, float | None, float | None] = VELOCITIES,
                   *, prepared: PreparedDevice | None = None) -> PreparedDevice:
    """Declare the four existing channels in n-left,p-left,n-right,p-right order."""
    original = load_device() if prepared is None else prepared
    data = original.to_input().editing_data()
    data["contacts"] = deepcopy(original.to_mapping()["contacts"])
    contacts = {entry["side"]: entry for entry in data["contacts"]}
    for side, pair in zip(("left", "right"), (velocities[:2], velocities[2:])):
        contacts[side]["S_n"], contacts[side]["S_p"] = pair
    return resolve_device(DeviceInput.model_validate(data), original.defaults, original.resources, sources=original.sources)


def algebra_coordinates(device: CompiledDevice2D) -> np.ndarray:
    """A finite non-steady probe with both lateral and boundary perturbations."""
    _, u = coordinates(device, lateral=True)
    contact = device.geometry.contact_nodes
    u[contact] += np.linspace(-.0003, .0004, contact.size)
    u[device.node_count + contact] += np.linspace(-.012, .016, contact.size)
    u[2 * device.node_count + contact] += np.linspace(.009, -.013, contact.size)
    return u


def prepare_seed(path: Path, sha256: str) -> tuple[CompiledDevice2D, State2D, dict[str, Any]]:
    """One five-node density correction; all other free changes are explicit contact kinetics.

    This is numerical preparation, using the same physical-density chain as
    steady_1d._density_basin_seed on only the blocking n-left block. It is
    never an accepted 2D state or an alternative equation/solver.
    """
    prior = read_fixed_result(path, sha256)
    base = load_device()
    if prior["prepared_declaration"]["declaration"] != base.to_mapping()["declaration"]:
        raise ValueError("saved seed is not the declared fixed-contact example")
    d = compile_device_2d(contact_device(prepared=base), spatial_input())
    g, count = d.geometry, d.node_count
    for name, expected in (("x_m", g.x_m), ("y_m", d.column.nodes_m), ("cell_areas_m2", g.contract_geometry.volumes),
                           ("face_pairs", g.contract_geometry.pairs), ("face_lengths_m", g.contract_geometry.face_measures),
                           ("face_spacing_m", g.face_spacing_m)):
        if not np.array_equal(prior["geometry"][name], expected):
            raise ValueError("saved seed geometry differs: " + name)
    selected = [a for a in prior["attempts"] if a["accepted"] and a["target"] == {"voltage_V": VOLTAGE_V, "generation_scale": 1.0}]
    if len(selected) != 1:
        raise ValueError("one accepted fixed-contact2D 2mV/G1 seed is required")
    old = selected[0]["result_2d"]
    saved_u = np.asarray(old["state"]["u"], dtype=float)
    saved = d.state(saved_u, voltage_V=VOLTAGE_V)
    for name in ("potential_V", "electron_density_m3", "hole_density_m3"):
        if not np.array_equal(getattr(saved, name), old["state"][name]):
            raise ValueError("saved physical fields/reference mapping differs: " + name)
    fluxes = np.asarray(old["diagnostics"]["terminal_particle_flux_density_m2_s"], dtype=float)
    if fluxes.shape != (2, 2, g.nx) or not np.all(np.isfinite(fluxes)):
        raise ValueError("saved directed terminal particle fluxes are invalid")
    changed: np.ndarray
    u, changed, kinetics = saved_u.copy(), np.zeros(3 * count, dtype=bool), []
    refs = d.reference_potentials(VOLTAGE_V)
    for channel in d.column.channels:
        if channel.kind != "robin":
            continue
        assert channel.velocity_m_s is not None
        carrier = 0 if channel.carrier == "n" else 1
        side = 0 if channel.side == "left" else 1
        nodes = g.contact_nodes.reshape(2, g.nx)[side]
        ratio = fluxes[carrier, side] / (channel.outward_normal * channel.velocity_m_s * channel.reservoir_density_m3)
        proposed = channel.reservoir_density_m3 * (1.0 + ratio)
        if not np.all(np.isfinite(proposed)) or np.any(proposed <= 0.0):
            raise ValueError("finite current matching requires positive finite densities; no clipping")
        offset, sign = (carrier + 1) * count, 1.0 if carrier == 0 else -1.0
        logarithm = np.log1p(ratio)
        u[offset + nodes] = sign * logarithm - (u[nodes] - refs[carrier][nodes]) / d.thermal_voltage_V
        changed[offset + nodes] = True
        kinetics.append({"channel": asdict(channel), "nodes": nodes, "saved_flux_m2_s": fluxes[carrier, side],
                         "proposed_density_m3": proposed, "log_excess_ratio": logarithm})
    problem = DC2DProblem(d, VOLTAGE_V, 1.0)
    if d.column.channel("n", "left").kind != "blocking":
        raise ValueError("this preparation is the explicit mixed contact example only")
    nodes = g.contact_nodes[:g.nx]
    active = count + nodes
    kinetic_u = u.copy()
    density = d.fields(u, voltage_V=VOLTAGE_V)[1][nodes]
    before = problem.residual(u)
    # dc/dz_n=c: divide each Jacobian COLUMN by its own physical density.
    block = problem.jacobian(u)[active][:, active].toarray() / density[None, :]
    increment = np.linalg.solve(block, -before[active])
    proposed = density + increment
    if not np.all(np.isfinite(proposed)) or np.any(proposed <= 0.0):
        raise ValueError("blocking density preparation is not finite/positive: " + repr(proposed.tolist()))
    change = np.log1p(increment / density)
    if not np.all(np.isfinite(change)):
        raise ValueError("blocking log1p mapping is nonfinite; no fallback")
    u[active] += change
    changed[active] = True
    if not np.array_equal(u[~changed], saved_u[~changed]):
        raise ValueError("preparation changed a fixed/interior coordinate")
    after = problem.residual(u)
    settings, _ = _settings(d, Steady2DOptions(), lambda values: GateOutcome("unused", True, 0.0, 1.0))
    step = _Factor(problem.jacobian(u)).solve(-after)
    fraction = settings.step_control.step_fraction(u, step)
    data = _diagnostics(problem, u, Steady2DOptions())
    info = {"schema": PREPARATION_SCHEMA, "seed_path": str(path), "seed_sha256": sha256,
        "original_compiled_identity": prior["compiled_identity"], "compiled_identity": d.identity,
        "original_source_sha256": old["source_sha256"], "original_acceptance_checks": old["checks"],
        "original_seed_state_sha256": state_digest(saved_u), "original_reference_mapping_exact": True,
        "saved_coordinates": saved_u, "finite_matched_coordinates": kinetic_u, "finite_contacts": kinetics,
        "blocking_nodes": nodes, "blocking_rows": active, "blocking_density_m3": density,
        "blocking_density_increment_m3": increment, "blocking_proposed_density_m3": proposed,
        "blocking_physical_jacobian": block, "blocking_residual_before": before[active], "blocking_residual_after": after[active],
        "unchanged_coordinates": np.flatnonzero(~changed), "preserved_coordinates_exact": True,
        "full_residual_before": before, "full_residual_after": after, "full_newton_direction": step,
        "full_step_inf": float(np.max(abs(step))), "full_step_fraction": fraction,
        "full_step_can_progress": bool(np.all(np.isfinite(step)) and fraction > 0.0 and 1.0 - fraction != 1.0),
        "initial_diagnostics": data, "initial_checks": [asdict(c) for c in _checks(data, Steady2DOptions())],
        "contact_density_linear_solves": 1, "full_direction_linear_solves": 1, "nonlinear_solves": 0,
        "accepted_for_contact_model": False, "qualification": False}
    return d, d.state(u, voltage_V=VOLTAGE_V), info


def solve_from_seed(output: Path, seed_path: Path, seed_sha256: str) -> dict[str, Any]:
    """One existing full2D Newton; preserve failure and require explicit output."""
    if output.exists():
        raise FileExistsError("preserve the original contact result")
    d, seed, info = prepare_seed(seed_path, seed_sha256)
    if not info["full_step_can_progress"]:
        raise ValueError("prepared full Newton correction remains ineffective; no nonlinear trial")
    result = solve_steady_state_2d(d, voltage_V=VOLTAGE_V, generation_scale=1.0,
        options=Steady2DOptions(wall_s=25.0), initial_state=seed,
        initial_provenance={"kind": "explicit saved2D contact-density numerical preparation",
                            "seed_sha256": seed_sha256, "accepted_for_contact_model": False})
    record = {"schema": STEADY_SCHEMA, "result": json_value(result), "preparation": json_value(info),
        "compiled_identity": d.identity, "state_shape_Ny_Nx": d.geometry.shape,
        "contact_channels": [dict(asdict(c), kind=c.kind) for c in d.column.channels],
        "newton_calls": result.solver_controls["full_2d_newton_calls"], "auxiliary_calls": 0,
        "accepted": result.accepted, "qualification": False}
    _write(output, record)
    return record


def _write(output: Path, value: dict[str, Any]) -> None:
    raw = (json.dumps(json_value(value), sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(raw)


def assemble() -> dict[str, Any]:
    prepared, spatial = contact_device(), spatial_input()
    d = compile_device_2d(prepared, spatial)
    problem = DC2DProblem(d, VOLTAGE_V, 1.0)
    u = algebra_coordinates(d)
    matrix = problem.jacobian(u)
    # Generation is state independent; retain the existing derivative probe
    # without its large constant source, and check the illuminated J exactly.
    derivative = DC2DProblem(d, VOLTAGE_V, 0.0)
    rng = np.random.default_rng(772)
    direction = np.r_[.02 * rng.normal(size=d.node_count), rng.normal(size=2 * d.node_count)]
    h = DIRECTIONAL_STEP
    numerical = (derivative.residual(u - 2 * h * direction) - 8 * derivative.residual(u - h * direction)
                 + 8 * derivative.residual(u + h * direction) - derivative.residual(u + 2 * h * direction)) / (12 * h)
    scale = np.maximum(abs(matrix) @ abs(direction), 1e-20)
    options = Steady2DOptions()
    diagnostics = _diagnostics(problem, u, options)
    summary = {"prepared_identity": prepared.content_sha256, "compiled_identity": d.identity,
        "shape_Ny_Nx": d.geometry.shape, "unknowns": 3 * d.node_count, "jacobian_nnz": matrix.nnz,
        "contact_channels": [dict(asdict(channel), kind=channel.kind) for channel in d.column.channels],
        "fixed_carrier_rows": int(np.count_nonzero(d.fixed_carrier_mask)),
        "directional_error": float(np.max(abs(numerical - matrix @ direction) / scale)),
        "directional_step": h, "directional_rtol": DIRECTIONAL_RTOL,
        "illuminated_jacobian_identical": bool(np.array_equal(matrix.toarray(), derivative.jacobian(u).toarray())),
        "units": {"face_particle_flux": "m^-2 s^-1 along +y", "contact_row": "particles/(m depth*s)",
                  "terminal_current": "A/m depth", "terminal_mean_current": "A/m2", "fixed_carrier_row": "log(c/c_eq)"},
        "geometry_rule": "face measure wx; contact area wx*wy with physical half endpoint widths",
        "orientation": "1D left/right map to y-min/y-max; n_out=-1/+1; J_n=-q*Gamma_n, J_p=q*Gamma_p",
        "contact_log_max": diagnostics["contact_log_max"],
        "carrier_balance_bound_A_m2": diagnostics["carrier_balance_bound_A_m2"],
        "source_terminal_current_bound_A_m2": diagnostics["source_terminal_current_bound_A_m2"],
        "newton_calls": 0, "accepted_steady_state": False, "qualification": False}
    return {"schema": SCHEMA, "summary": summary, "device_declaration": prepared.to_mapping(),
        "spatial_declaration": spatial.editing_data(), "state": d.state(u, voltage_V=VOLTAGE_V),
        "residual": problem.residual(u),
        "jacobian_csc": {"data": matrix.data, "indices": matrix.indices, "indptr": matrix.indptr, "shape": matrix.shape},
        "direction": direction, "analytic_direction": matrix @ direction, "numerical_direction": numerical,
        "directional_scale": scale, "cell_areas_m2": d.geometry.contract_geometry.volumes,
        "x_widths_m": d.geometry.x_widths_m, "y_widths_m": d.geometry.y_geometry.cell_measures_m,
        "x_m": d.geometry.x_m, "y_m": d.column.nodes_m, "fixed_carrier_mask": d.fixed_carrier_mask,
        "carrier_fluxes": asdict(problem.fluxes(u)), "contact_particle_fluxes_m2_s": problem.contact_particle_fluxes(u),
        "diagnostics": diagnostics,
        "witness_proposal": {"scheduled": False, "independent_review_required": True,
            "solve": "solve_steady_state_2d", "newton_calls": 1,
            "voltage_V": VOLTAGE_V, "generation_scale": 1.0, "whole_wall_cap_s": 30.0,
            "options": asdict(replace(options, wall_s=25.0)),
            "initial_state": "compiler physical reference on the connected 2D graph; no auxiliary solve",
            "retry_or_fallback": False, "qualification": False}}


def read_result(path: Path, expected_sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("contact algebra SHA-256 differs")
    result = json.loads(raw)
    if isinstance(result, dict) and result.get("schema") == STEADY_SCHEMA:
        value = result["result"]
        if (result.get("qualification") is not False or result.get("accepted") is not value["accepted"]
                or value["physical_certificate_status"] != "not_evaluated"
                or tuple(result["state_shape_Ny_Nx"]) != (13, 5)
                or len(value["initial_coordinates"]) != 195 or len(value["candidate_coordinates"]) != 195):
            raise ValueError("invalid or overclaimed saved contact witness")
        if value["accepted"] and (value["termination"] != "converged" or not value["newton"]["accepted"]
                or not value["checks"] or any(not c["passed"] or not math.isfinite(c["value"]) or c["value"] > c["limit"] for c in value["checks"])
                or value["state"]["device_sha256"] != result["compiled_identity"]
                or not np.array_equal(value["state"]["u"], value["candidate_coordinates"])):
            raise ValueError("saved contact acceptance lost original checks/state")
        return result
    if (not isinstance(result, dict) or result.get("schema") != SCHEMA
            or result["summary"].get("newton_calls") != 0
            or result["summary"].get("accepted_steady_state") is not False
            or result["summary"].get("qualification") is not False):
        raise ValueError("not an unqualified no-solve 2D contact algebra result")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--read", type=Path)
    mode.add_argument("--prepare", type=Path)
    mode.add_argument("--solve", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--seed", type=Path)
    parser.add_argument("--seed-sha256")
    args = parser.parse_args()
    if args.read is not None:
        if args.sha256 is None:
            parser.error("--read requires --sha256")
        value = read_result(args.read, args.sha256)
        print(json.dumps(value.get("summary", {"accepted": value.get("accepted"), "qualification": False,
            "termination": value.get("result", {}).get("termination")}), indent=2))
        return 0
    if args.prepare is not None or args.solve is not None:
        if args.seed is None or args.seed_sha256 is None:
            parser.error("--prepare/--solve require --seed and --seed-sha256")
        if args.solve is not None:
            value = solve_from_seed(args.solve, args.seed, args.seed_sha256)
            print(json.dumps({"accepted": value["accepted"], "termination": value["result"]["termination"], "qualification": False}))
            return 0 if value["accepted"] else 2
        if args.prepare.exists():
            raise FileExistsError("preserve the original contact preparation")
        _, state, info = prepare_seed(args.seed, args.seed_sha256)
        _write(args.prepare, {**info, "state": state})
        print(json.dumps({"path": str(args.prepare), "sha256": hashlib.sha256(args.prepare.read_bytes()).hexdigest(),
            "full_step_inf": info["full_step_inf"], "full_step_fraction": info["full_step_fraction"],
            "full_step_can_progress": info["full_step_can_progress"], "nonlinear_solves": 0, "qualification": False}))
        return 0
    if args.output.exists():
        raise FileExistsError("preserve the previous contact evidence")
    value = assemble()
    raw = (json.dumps(json_value(value), indent=2, allow_nan=False) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"path": str(args.output.resolve()), "sha256": hashlib.sha256(raw).hexdigest(), **value["summary"]}, indent=2))
    return 0 if value["summary"]["directional_error"] < DIRECTIONAL_RTOL else 2


if __name__ == "__main__":
    raise SystemExit(main())
