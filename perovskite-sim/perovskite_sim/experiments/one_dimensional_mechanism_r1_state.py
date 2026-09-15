"""Source-bound common D equilibrium for the opt-in R1-1 controls."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from types import SimpleNamespace

import numpy as np
import scipy

from perovskite_sim.constants import Q
from perovskite_sim.experiments.defect_ion_combined_impedance import (
    CombinedDCCertificate, CombinedDCState, _build_ion_layout,
    _component_inventories, _ion_equilibrium_residual, _state_sha256,
)
from perovskite_sim.experiments.interface_defect_ion_transient import (
    InterfaceDefectIonTransientPolicy, InterfaceIonDarkReference,
)
from perovskite_sim.experiments.one_dimensional_mechanism_r1 import (
    build_r1_material, physical_step_record, solve_r1_dc,
)
from perovskite_sim.experiments.one_dimensional_mechanism_r1_binding import (
    study_input_identity, validate_r1_study_binding,
)
from perovskite_sim.experiments.one_dimensional_mechanism_r1_dynamics import (
    ControlledPhysicalInterfaceIonSystem, R1DynamicsControls,
)
from perovskite_sim.experiments.quasi_fermi_steady_state import (
    _QuasiFermiSystem, _research_charge_off_stack,
)
from perovskite_sim.physics.contacts import (
    ContactThermodynamicCertificate, require_contact_thermodynamic_certificate,
    resolved_contact_velocities,
)
from perovskite_sim.physics.interface_plane import FERMI_DIRAC_RICHARDSON
from perovskite_sim.physics.two_sided_interface import TWO_SIDED_TRACE


STATE_SCHEMA = "R1CommonStateV1"
STUDY_SPEC_SHA256 = "f140505c3a7ec7c52fe38c46f36d53b876a87c2ee50512568cb62671b05c6a73"


class R1StateError(ValueError):
    """A supplied common state cannot be used for this physical experiment."""


def json_data(value):
    if is_dataclass(value):
        return {f.name: json_data(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {str(k): json_data(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_data(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def canonical(value):
    return json.dumps(json_data(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def execution_source():
    package = Path(__file__).resolve().parents[1]
    files = {
        str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(package.rglob("*.py"))
    }
    source = {"files": files, "study_input": study_input_identity()}
    return {**source, "sha256": digest(source)}


def _preparation_history():
    return {
        "kind": "independent_D_dark_dc", "voltage_V": 0.0,
        "temperature_K": 300.0, "illuminated": False,
        "duration_s": None,
        "duration_reason": "certified equilibrium, not finite-time preparation",
    }


@dataclass(frozen=True, slots=True)
class R1PreparedState:
    """Immutable JSON storage; exported records and arrays are fresh copies."""

    canonical_json: str

    def __post_init__(self):
        record = json.loads(self.canonical_json)
        if not isinstance(record, dict) or record.get("schema") != STATE_SCHEMA:
            raise R1StateError("unrecognized R1 common-state schema")
        body = {k: v for k, v in record.items() if k != "sha256"}
        if record.get("sha256") != digest(body):
            raise R1StateError("common-state content hash mismatch")
        object.__setattr__(self, "canonical_json", canonical(record))

    @classmethod
    def from_dict(cls, record):
        return cls(canonical(record))

    def to_dict(self):
        return json.loads(self.canonical_json)

    @property
    def sha256(self):
        return self.to_dict()["sha256"]


def snapshot(system, state):
    return json_data({
        "n_m3": state.n, "p_m3": state.p, "positive_m3": state.positive,
        "occupancy": state.occupancy, "phi_V": state.phi,
        "sheet_charge_C_m2": state.sheet_charge,
        "trace_potential_V": [local.trace_potential for local in state.local],
        "trace_state_m3": [local.state_m3 for local in state.local],
        "capture_m2_s": [local.tangent.balance.capture_flux_m2_s for local in state.local],
        "positive_inventory_m2": _component_inventories(
            state.positive, system.ion_layout.positive_components, system.widths),
        "poisson_residual_C_m2": state.poisson_residual,
        "local_residual": state.local_residual,
    })


def _grid_identity(grid, material):
    layout = _build_ion_layout(material)
    return json_data({
        "coordinates_m": grid, "faces_m": material.physical_cell_faces_m,
        "widths_m": material.dx_cell,
        "interface_positions_m": material.iface_qss_interface_positions_m,
        "positive_nodes": layout.positive_nodes,
        "positive_components": layout.positive_components,
        "background_positive_m3": material.P_ion0,
    })


def _make_system(stack, grid, material, dc, reference, controls, policy, *, canonicalize=False):
    charge_off, microscopic = _research_charge_off_stack(stack)
    qf = _QuasiFermiSystem(
        grid, charge_off, material, 0.0, interface_boundary=True,
        interface_topology=TWO_SIDED_TRACE, interface_transmission=1.0,
        interface_transport_model=FERMI_DIRAC_RICHARDSON,
        interface_charge_reference_occupancy=np.asarray(reference["f_ref"]),
        interface_charge_trap_density_m2=microscopic.trap_density_m2,
        poisson_tolerance_V=1e-13, poisson_max_iterations=100,
    )
    value = qf.evaluate_quasi_fermi_increments_defect_ion_combined(
        dc.electron_qf_increment_V, dc.hole_qf_increment_V, 0.0,
        positive_ion_density_m3=dc.positive_ion_density_m3,
        dynamic_interface_occupancy=dc.interface_occupancy, V_app=0.0,
    )
    if canonicalize:
        # Match the existing public D embedding once, during preparation.
        # Import never replaces recorded populations with a new QF evaluation.
        dc = replace(dc, electron_density_m3=value.y[:grid.size].copy(),
                     hole_density_m3=value.y[grid.size:2*grid.size].copy(),
                     potential_V=value.phi.copy())
    # Local algebraic solves may be repeated on import, but the supplied
    # dynamic populations and electrostatic state must never be re-prepared.
    dynamic = SimpleNamespace(
        y=np.r_[dc.electron_density_m3, dc.hole_density_m3],
        phi=dc.potential_V,
        interface_charge_dynamic=value.interface_charge_dynamic,
    )
    dark = InterfaceIonDarkReference(
        np.asarray(reference["f_ref"]), microscopic.trap_density_m2,
        microscopic.capture_velocities_m_s, microscopic.document_sha256, 1.0, dc,
    )
    system = ControlledPhysicalInterfaceIonSystem(
        grid, charge_off, material, dc, qf, dark, dc.interface_occupancy,
        dynamic, _build_ion_layout(material), voltage=0.0, illuminated=False,
        site_occupancy_ceiling=policy.site_occupancy_ceiling, controls=controls,
    )
    system.common_dc_state = dc
    return system


def _decode_dc(record):
    data = dict(record)
    certificate = dict(data.pop("certificate"))
    certificate["reasons"] = tuple(certificate["reasons"])
    contact = dict(certificate["contact_thermodynamics"])
    if "contact_quasi_fermi_levels_eV" in contact:
        contact["contact_quasi_fermi_levels_eV"] = tuple(contact["contact_quasi_fermi_levels_eV"])
    certificate["contact_thermodynamics"] = ContactThermodynamicCertificate(**contact)
    data["certificate"] = CombinedDCCertificate(**certificate)
    for key in data:
        if key not in ("certificate", "state_sha256") and data[key] is not None:
            array = np.asarray(data[key], dtype=float).copy()
            array.setflags(write=False)
            data[key] = array
    return CombinedDCState(**data)


def equilibrium_checks(system, state, policy):
    """Re-evaluate remaining equations; do not trust an imported pass flag."""
    count = system.interior_count
    w = system.widths
    carrier_bounds = [
        float(np.sum(np.abs(state.rate[k*count:(k+1)*count]) * w[1:-1] * Q))
        for k in (0, 1)
    ]
    carrier_normalized = [
        float(np.max(np.abs(state.rate[k*count:(k+1)*count]) * w[1:-1] * Q))
        / system.system.current_scale for k in (0, 1)
    ]
    mu = _ion_equilibrium_residual(
        state.positive, None, state.phi, system.ion_layout.positive_components,
        system.positive_targets, system.material, positive=True,
    )
    inventory = _component_inventories(state.positive, system.ion_layout.positive_components, w)
    nominal_inventory = _component_inventories(
        system.material.P_ion0, system.ion_layout.positive_components, w)
    carrier, gauss = system.local_normalized_residuals(state)
    physical = physical_step_record(system, state, None, 0.0)
    capture = np.asarray([x.tangent.balance.capture_flux_m2_s for x in state.local])
    balance = np.abs(capture[:, [0, 2]].sum(axis=1) - capture[:, [1, 3]].sum(axis=1))
    metrics = {
        "electron_continuity_A_m2": carrier_bounds[0],
        "hole_continuity_A_m2": carrier_bounds[1],
        "electron_normalized_residual": carrier_normalized[0],
        "hole_normalized_residual": carrier_normalized[1],
        "ion_equilibrium_residual": float(np.max(np.abs(mu))),
        "inventory_relative_error": float(np.max(np.abs(inventory/system.positive_targets-1.0))),
        "nominal_inventory_relative_error": float(np.max(np.abs(inventory/nominal_inventory-1.0))),
        "ionic_face_current_A_m2": float(np.max(np.abs(state.positive_current))),
        "dc_current_spread_A_m2": float(np.ptp(state.conduction)),
        "poisson_normalized": float(np.max(np.abs(state.poisson_residual)))/(Q*1e15),
        "full_gauss_normalized": physical["gauss_normalized"],
        "local_carrier_residual": carrier, "local_gauss_residual": gauss,
        "trap_storage_rate_A_m2": Q * float(np.max(balance)),
        "eliminated_operator_error": system.eliminated_operator_error(state, 0.0),
    }
    limits = {
        "electron_continuity_A_m2": policy.maximum_dc_continuity_bound_A_m2,
        "hole_continuity_A_m2": policy.maximum_dc_continuity_bound_A_m2,
        "electron_normalized_residual": policy.maximum_dc_normalized_residual,
        "hole_normalized_residual": policy.maximum_dc_normalized_residual,
        "ion_equilibrium_residual": policy.maximum_dc_normalized_residual,
        "inventory_relative_error": min(policy.maximum_dc_inventory_error, 1e-10),
        "nominal_inventory_relative_error": 1e-12,
        "ionic_face_current_A_m2": policy.maximum_dc_ionic_face_current_A_m2,
        "dc_current_spread_A_m2": policy.maximum_dc_face_current_spread_A_m2,
        "poisson_normalized": 1e-10, "full_gauss_normalized": 1e-10,
        "local_carrier_residual": policy.maximum_local_carrier_normalized_residual,
        "local_gauss_residual": policy.maximum_local_gauss_normalized_residual,
        "trap_storage_rate_A_m2": policy.maximum_dc_continuity_bound_A_m2,
        "eliminated_operator_error": policy.maximum_eliminated_operator_relative_error,
    }
    failures = [k for k in metrics if not np.isfinite(metrics[k]) or metrics[k] > limits[k]]
    return {"metrics": metrics, "limits": limits, "certified": not failures, "reasons": failures}


def prepare_common_state(stack, intervals, binding, *, policy=None):
    """Prepare D once. A-D import copies of this state without another DC solve."""
    policy = policy or InterfaceDefectIonTransientPolicy(maximum_ion_inventory_relative_drift=1e-10)
    validate_r1_study_binding(binding, stack)
    grid, material, dc, _ = solve_r1_dc(stack, intervals, np.asarray(binding["f_ref"]))
    system = _make_system(stack, grid, material, dc, binding, R1DynamicsControls(), policy,
                          canonicalize=True)
    dc = system.common_dc_state
    state = system.evaluate(system.initial_coordinate(), 0.0)
    checks = equilibrium_checks(system, state, policy)
    if not checks["certified"]:
        raise R1StateError(f"common D equilibrium failed: {checks['reasons']}")
    payload = {
        "schema": STATE_SCHEMA, "kind": "equilibrium_D", "state_time": "0-",
        "preparation_controls": {"nu_I": 1, "nu_t": 1},
        "intervals": int(intervals), "grid": _grid_identity(grid, material),
        "physical_stack": json_data(stack), "stack_sha256": digest(json_data(stack)),
        "fixed_reference": binding, "reference_sha256": binding["sha256"],
        "contact_velocities_m_s": json_data(resolved_contact_velocities(stack)),
        "contact_certificate": json_data(require_contact_thermodynamic_certificate(system.stack, material)),
        "dc_state": json_data(dc), "state": snapshot(system, state),
        "qf_references_V": {"electron": system.qfn_reference.tolist(), "hole": system.qfp_reference.tolist()},
        "preparation_checks": checks,
        "history": _preparation_history(),
        "study_spec_sha256": STUDY_SPEC_SHA256,
        "source": execution_source(),
        "environment": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform()},
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    payload["sha256"] = digest(payload)
    return R1PreparedState.from_dict(payload)


def restore_common_state(prepared, stack, intervals, binding, *, controls=None, policy=None):
    """Verify identities and live residuals, then copy the same populations."""
    if not isinstance(prepared, R1PreparedState):
        prepared = R1PreparedState.from_dict(prepared)
    record = prepared.to_dict()
    policy = policy or InterfaceDefectIonTransientPolicy(maximum_ion_inventory_relative_drift=1e-10)
    controls = controls or R1DynamicsControls()
    validate_r1_study_binding(binding, stack)
    grid, material = build_r1_material(stack, intervals)
    expected = {
        "kind": "equilibrium_D", "state_time": "0-", "intervals": int(intervals),
        "preparation_controls": {"nu_I": 1, "nu_t": 1},
        "grid": _grid_identity(grid, material), "physical_stack": json_data(stack),
        "stack_sha256": digest(json_data(stack)), "fixed_reference": binding,
        "reference_sha256": binding["sha256"], "study_spec_sha256": STUDY_SPEC_SHA256,
        "source": execution_source(),
        "contact_velocities_m_s": json_data(resolved_contact_velocities(stack)),
        "history": _preparation_history(),
    }
    mismatches = [k for k, value in expected.items() if record.get(k) != value]
    if mismatches:
        raise R1StateError(f"common-state identity mismatch: {', '.join(mismatches)}")
    try:
        dc = _decode_dc(record["dc_state"])
        dc_arrays = [dc.electron_qf_increment_V, dc.hole_qf_increment_V,
                     dc.positive_ion_density_m3]
        if dc.negative_ion_density_m3 is not None:
            dc_arrays.append(dc.negative_ion_density_m3)
        if dc.state_sha256 != _state_sha256(*dc_arrays):
            raise R1StateError("common-state DC coordinate hash mismatch")
        if not dc.certificate.certified or not dc.certificate.optimizer_success or dc.certificate.reasons:
            raise R1StateError("common-state DC certificate was not accepted")
        for dc_field, physical_field in (
            ("electron_density_m3", "n_m3"), ("hole_density_m3", "p_m3"),
            ("positive_ion_density_m3", "positive_m3"), ("interface_occupancy", "occupancy"),
            ("potential_V", "phi_V"),
        ):
            if not np.array_equal(getattr(dc, dc_field), record["state"][physical_field]):
                raise R1StateError(f"common-state DC physical array mismatch: {dc_field}")
        baseline = _make_system(stack, grid, material, dc, binding, R1DynamicsControls(), policy)
        initial = baseline.evaluate(baseline.initial_coordinate(), 0.0)
        actual = snapshot(baseline, initial)
        if set(actual) != set(record["state"]):
            raise R1StateError("common-state physical record fields mismatch")
        for key in actual:
            if not np.array_equal(actual[key], record["state"][key]):
                raise R1StateError(f"common-state physical array mismatch: {key}")
        if record["qf_references_V"] != {"electron": baseline.qfn_reference.tolist(), "hole": baseline.qfp_reference.tolist()}:
            raise R1StateError("common-state QF reference mismatch")
        if record["contact_certificate"] != json_data(require_contact_thermodynamic_certificate(baseline.stack, material)):
            raise R1StateError("common-state contact certificate mismatch")
        checks = equilibrium_checks(baseline, initial, policy)
        if not checks["certified"]:
            raise R1StateError(f"imported common state fails live D equations: {checks['reasons']}")
        if controls == R1DynamicsControls():
            return baseline, initial
        system = _make_system(stack, grid, material, dc, binding, controls, policy)
        controlled = system.evaluate(system.initial_coordinate(), 0.0)
        for name in ("n", "p", "positive", "occupancy", "phi", "sheet_charge"):
            if not np.array_equal(getattr(controlled, name), getattr(initial, name)):
                raise R1StateError(f"control switch changed a shared physical population: {name}")
        checks = equilibrium_checks(system, controlled, policy)
        if not checks["certified"]:
            raise R1StateError(f"control switch fails zero-excitation equations: {checks['reasons']}")
        return system, controlled
    except (KeyError, TypeError, IndexError) as exc:
        raise R1StateError(f"malformed common-state record: {exc}") from exc


__all__ = ["R1PreparedState", "R1StateError", "prepare_common_state", "restore_common_state", "equilibrium_checks"]
