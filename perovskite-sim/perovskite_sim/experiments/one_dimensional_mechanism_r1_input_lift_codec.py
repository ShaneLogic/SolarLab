"""Exact accepted-state checkpoints for the explicit R1 input-lift operator.

Restoring a checkpoint creates a *new* zero-coordinate reference.  It does not
recover or invent the historical Newton coordinate, initialize a device, solve
DC, or advance time.  One evaluation at that same physical state reconstructs
the local constitutive payload and Jacobians and checks the saved derived
words.  Saved incremental electrostatic residuals remain explicit anchors.
"""
from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass, replace
import hashlib
import json
import math
import re
from types import MappingProxyType
from collections.abc import Mapping

import numpy as np

from perovskite_sim.constants import Q
from perovskite_sim.physics.compensated import DD
from .interface_defect_ion_transient import _InterfaceIonDeviceState
from . import one_dimensional_mechanism_r1_input_lift as lift
from .one_dimensional_mechanism_r1_local_carrier import carrier_data


SCHEMA = "R1InputLiftCheckpointV1"
SNAPSHOT_SCHEMA = "R1InputLiftStateV1"
COORDINATE_ROLE = "accepted_physical_state_imported_as_new_zero_coordinate_reference"
_HASH = re.compile(r"^[0-9a-f]{64}$")
_DERIVED = ("storage", "rate", "sheet_charge_C_m2", "poisson_residual_C_m2",
            "local_residual", "electron_current_A_m2", "hole_current_A_m2",
            "positive_flux_m2_s", "positive_rate_m3_s")
_FIELDS = frozenset(lift.PRIMARY_FIELDS) | frozenset(_DERIVED)
_OPTIONAL_FIELDS = frozenset(("carrier_rate",))
_SNAPSHOT_KEYS = frozenset(("schema", "operator_representation",
    "coordinate_reference_identity", "high_word_state", "represented_fields"))
_HIGH_FIELDS = frozenset(("n_m3", "p_m3", "positive_m3", "occupancy", "phi_V",
    "sheet_charge_C_m2", "trace_potential_V", "trace_state_m3", "capture_m2_s",
    "positive_inventory_m2", "poisson_residual_C_m2", "local_residual"))
_CHECKPOINT_KEYS = frozenset(("schema", "operator_representation", "time_s",
    "voltage_V", "coordinate_role", "source_coordinate", "state_identity",
    "prior_coordinate_reference_identity", "new_reference_identity",
    "physical_model", "snapshot", "sha256"))


class InputLiftCheckpointError(ValueError):
    """Checkpoint data cannot identify the requested accepted physical state."""


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise InputLiftCheckpointError("checkpoint is not finite canonical JSON") from exc


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _number(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise InputLiftCheckpointError(name + " must be a finite real number")
    value = float(value)
    if not math.isfinite(value) or positive and value <= 0 or nonnegative and value < 0:
        raise InputLiftCheckpointError(name + " is outside its finite domain")
    return value


def _array(value, name, shape):
    def has_boolean(item):
        return (isinstance(item, (bool, np.bool_))
                or isinstance(item, (tuple, list)) and any(has_boolean(part) for part in item))
    if has_boolean(value):
        raise InputLiftCheckpointError(name + " must not contain booleans")
    try:
        array = np.asarray(value)
    except (ValueError, TypeError) as exc:
        raise InputLiftCheckpointError(name + " is not a rectangular numeric array") from exc
    if (array.dtype.kind not in "fiu" or array.shape != shape
            or not np.isfinite(array).all()):
        raise InputLiftCheckpointError(name + " has invalid shape, type or finite values")
    converted = array.astype(float)
    if array.dtype.kind in "iu" and any(int(a) != int(b) for a, b in zip(array.flat, converted.flat)):
        raise InputLiftCheckpointError(name + " loses integer precision on import")
    return converted


def _same_words(left, right):
    return (left.hi.shape == right.hi.shape
            and left.hi.tobytes() == right.hi.tobytes()
            and left.lo.tobytes() == right.lo.tobytes())


def _shapes(system):
    nodes, interfaces = int(system.node_count), int(system.interface_count)
    storage = 2 * int(system.interior_count) + interfaces + int(system.positive_nodes.size)
    return {**dict.fromkeys(("n_m3", "p_m3", "phi_V", "dqfn_V", "dqfp_V",
                           "positive_m3", "positive_rate_m3_s"), (nodes,)),
        **dict.fromkeys(("occupancy", "sheet_charge_C_m2"), (interfaces,)),
        **dict.fromkeys(("electron_current_A_m2", "hole_current_A_m2", "positive_flux_m2_s"), (nodes - 1,)),
        "trace_potential_V": (interfaces, 2), "trace_state_m3": (interfaces, 4),
        "storage": (storage,), "rate": (storage,),
        "carrier_rate": (2 * int(system.interior_count) + interfaces,),
        "poisson_residual_C_m2": (int(system.interior_count),),
        "local_residual": (6 * interfaces,), "capture_m2_s": (interfaces, 4),
        "positive_inventory_m2": (len(system.ion_layout.positive_components),)}


def _validate_snapshot(system, record):
    if (not isinstance(record, dict) or set(record) != _SNAPSHOT_KEYS
            or record["schema"] != SNAPSHOT_SCHEMA
            or record["operator_representation"] != lift.REPRESENTATION):
        raise InputLiftCheckpointError("explicit input-lift snapshot schema/representation mismatch")
    identity = record["coordinate_reference_identity"]
    if not isinstance(identity, str) or not _HASH.fullmatch(identity):
        raise InputLiftCheckpointError("snapshot reference identity is invalid")
    fields_record, high = record["represented_fields"], record["high_word_state"]
    if (not isinstance(fields_record, dict) or not _FIELDS <= set(fields_record)
            or set(fields_record) - _FIELDS - _OPTIONAL_FIELDS):
        raise InputLiftCheckpointError("snapshot represented field coverage mismatch")
    if not isinstance(high, dict) or set(high) != _HIGH_FIELDS:
        raise InputLiftCheckpointError("snapshot high-word field coverage mismatch")
    shapes, result = _shapes(system), {}
    for name, pair in fields_record.items():
        if not isinstance(pair, dict) or set(pair) != {"hi", "lo"}:
            raise InputLiftCheckpointError("each represented field needs exactly hi and lo: " + name)
        hi, lo = (_array(pair[part], name + "." + part, shapes[name]) for part in ("hi", "lo"))
        try:
            normalized = DD(hi, lo)
        except (ArithmeticError, ValueError, TypeError) as exc:
            raise InputLiftCheckpointError("invalid DD words: " + name) from exc
        if not np.array_equal(normalized.hi, hi) or not np.array_equal(normalized.lo, lo):
            raise InputLiftCheckpointError("DD words are not normalized: " + name)
        # Validation has established normalization.  Preserve the original
        # signed zero words too; a second DD arithmetic operation may erase it.
        result[name] = DD._trusted_parts(hi, lo)
    for name, raw in high.items():
        value = _array(raw, "high_word_state." + name, shapes[name])
        if name in result and value.tobytes() != result[name].hi.tobytes():
            raise InputLiftCheckpointError("snapshot public high view differs from DD: " + name)
    for name in ("n_m3", "p_m3", "trace_state_m3"):
        value = result[name]
        if np.any(value.hi < 0) or np.any((value.hi == 0) & (value.lo <= 0)):
            raise InputLiftCheckpointError("snapshot density must be positive: " + name)
    occupancy = result["occupancy"]
    if (np.any(occupancy.hi <= 0) or np.any(occupancy.hi >= 1)
            or np.any(result["positive_m3"].hi < 0)):
        raise InputLiftCheckpointError("snapshot populations are outside the supported physical domain")
    return MappingProxyType(result)


def _model_json(value):
    """Stable physical coefficients; inactive infinite lifetimes are explicit."""
    if isinstance(value, np.ndarray):
        return _model_json(value.tolist())
    if isinstance(value, np.generic):
        return _model_json(value.item())
    if is_dataclass(value):
        return {item.name: _model_json(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(name): _model_json(item) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_model_json(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value):
            raise InputLiftCheckpointError("physical model contains NaN")
        return value if math.isfinite(value) else {"infinite_coefficient": "+inf" if value > 0 else "-inf"}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    raise InputLiftCheckpointError("unsupported physical coefficient type: " + type(value).__name__)


def _attributes(owner, names):
    return {name: _model_json(getattr(owner, name)) if hasattr(owner, name) else {"absent": True}
            for name in names}


def _physical_model(system, state):
    """Bind coefficients used by this supported operator and its fixed scales."""
    return {
        "system": _attributes(system, ("grid", "widths", "dimension", "left_nodes", "right_nodes",
            "interface_faces", "positive_nodes", "negative_nodes", "thermal_voltage", "polarity",
            "trap_density", "equilibrium_occupancy", "reference_local_scale", "eps_face",
            "site_occupancy_ceiling", "capture_multiplier", "illuminated", "controls",
            "positive_targets", "negative_targets", "qfn_reference", "qfp_reference")),
        "material": _attributes(system.material, ("chi", "Eg", "D_n_face", "D_p_face", "eps_r",
            "P_lim_node", "D_ion_face", "P_ion0", "V_T_device", "has_dual_ions",
            "ion_steric_diffusion_only", "ion_steric_shared_site",
            "iface_qss_left_distances_m", "iface_qss_right_distances_m",
            "iface_qss_interface_positions_m", "N_A", "N_D", "n_L", "p_L", "n_R", "p_R")),
        "poisson": _attributes(system.material.poisson_factor, ("C", "h_cell")),
        "source": _attributes(system.system.source_mat, ("carrier_params", "ni_sq", "tau_n", "tau_p",
            "n1", "p1", "B_rad", "C_n", "C_p", "neutral_bulk_defects", "monovalent_bulk_defects",
            "multivalent_bulk_defects", "frozen_metastable_defects", "has_selective_contacts",
            "has_radiative_reabsorption")),
        "transport": _attributes(system.system, ("reference_edge_drop_n", "reference_edge_drop_p", "generation")),
        "positive_components": _model_json(system.ion_layout.positive_components),
        "local_coefficient_identities": [carrier_data(item).coefficient_identity for item in state.local],
    }


def save_checkpoint(system, state, *, time_s, voltage_V):
    """Return a self-contained JSON value; no file or system state is changed."""
    snap = lift.snapshot(system, state)
    values = _validate_snapshot(system, snap)
    source_coordinate = _array(state.coordinate, "source_coordinate", (system.dimension,))
    primary = {name: values[name] for name in lift.PRIMARY_FIELDS}
    record = {"schema": SCHEMA, "operator_representation": lift.REPRESENTATION,
        "time_s": _number(time_s, "time_s", nonnegative=True),
        "voltage_V": _number(voltage_V, "voltage_V"), "coordinate_role": COORDINATE_ROLE,
        "source_coordinate": source_coordinate.tolist(), "state_identity": _digest(snap["represented_fields"]),
        "prior_coordinate_reference_identity": state.coordinate_reference_identity,
        "new_reference_identity": lift._identity(primary), "physical_model": _physical_model(system, state),
        "snapshot": snap}
    record["sha256"] = _digest(record)
    return record


def _populate_template(system, template, values):
    if not isinstance(template, _InterfaceIonDeviceState) or len(template.local) != system.interface_count:
        raise InputLiftCheckpointError("restore needs a compatible system state template")
    attributes = {item.name: copy.deepcopy(getattr(template, item.name))
                  for item in fields(_InterfaceIonDeviceState) if item.name != "local"}
    attributes.update(coordinate=np.zeros(system.dimension), negative=None, negative_rate=None,
                      negative_flux=None, negative_current=None)
    for name, attr in (("n_m3", "n"), ("p_m3", "p"), ("phi_V", "phi"), ("dqfn_V", "dqfn"),
                      ("dqfp_V", "dqfp"), ("positive_m3", "positive"), ("occupancy", "occupancy"),
                      ("storage", "storage"), ("rate", "rate"), ("sheet_charge_C_m2", "sheet_charge"),
                      ("poisson_residual_C_m2", "poisson_residual"), ("local_residual", "local_residual"),
                      ("electron_current_A_m2", "current_n"), ("hole_current_A_m2", "current_p"),
                      ("positive_flux_m2_s", "positive_flux"), ("positive_rate_m3_s", "positive_rate")):
        attributes[attr] = values[name].hi.copy()
    attributes["local"] = tuple(replace(item,
        trace_potential=values["trace_potential_V"].hi[index].copy(),
        state_m3=values["trace_state_m3"].hi[index].copy(),
        log_state=values["trace_state_m3"][index].log().hi.copy(),
        sheet_charge_C_m2=float(values["sheet_charge_C_m2"].hi[index]),
        electrostatic_residual=values["local_residual"].hi[6*index:6*index+2].copy())
        for index, item in enumerate(template.local))
    attributes["positive_current"] = system.polarity * Q * attributes["positive_flux"]
    attributes["carrier_conduction"] = system.polarity * (attributes["current_n"] + attributes["current_p"])
    attributes["conduction"] = attributes["carrier_conduction"] + attributes["positive_current"]
    return lift.InputLiftState(**attributes, input_lift=MappingProxyType(dict(values)),
                              coordinate_reference_identity=lift._identity({name: values[name] for name in lift.PRIMARY_FIELDS}))


def _restore(system, template, snap, *, time_s, voltage_V, checkpoint_sha256=None,
             model=None, source_coordinate=None):
    values = _validate_snapshot(system, snap)
    skeleton = _populate_template(system, template, values)
    working, reference = lift.from_accepted_state(system, skeleton)
    # This is an algebraic reconstruction, never a Newton solve or advance.
    evaluated = working.evaluate(np.zeros(system.dimension), voltage_V)
    actual = lift.primary_inputs(evaluated)
    changed = [name for name, value in values.items() if name not in actual or not _same_words(value, actual[name])]
    if changed:
        raise InputLiftCheckpointError("saved words disagree with same-state reconstruction: " + ", ".join(changed))
    if _canonical(lift.snapshot(working, evaluated)["high_word_state"]) != _canonical(snap["high_word_state"]):
        raise InputLiftCheckpointError("saved public diagnostics disagree with same-state reconstruction")
    actual_model = _physical_model(working, evaluated)
    if model is not None and _canonical(model) != _canonical(actual_model):
        raise InputLiftCheckpointError("checkpoint physical model differs from restore system")
    # Preserve the entire saved word set including optional absence, while
    # retaining newly reconstructed local payload/Jacobians for the next step.
    accepted = replace(evaluated, input_lift=MappingProxyType(dict(values)),
                       coordinate=np.zeros(system.dimension),
                       coordinate_reference_identity=reference.coordinate_reference_identity)
    working._step_reference = accepted
    working.input_lift_restore_receipt = MappingProxyType({
        "schema": "R1InputLiftRestoreReceiptV1", "operator_representation": lift.REPRESENTATION,
        "coordinate_role": COORDINATE_ROLE, "time_s": time_s, "voltage_V": voltage_V,
        "checkpoint_sha256": checkpoint_sha256, "snapshot_sha256": _digest(snap),
        "state_identity": _digest(snap["represented_fields"]),
        "prior_coordinate_reference_identity": snap["coordinate_reference_identity"],
        "new_reference_identity": accepted.coordinate_reference_identity,
        "historical_coordinate_available": source_coordinate is not None,
        "all_saved_words_exact": True, "derived_reconstruction_exact": True,
        "algebraic_evaluations": 1, "nonlinear_solves": 0, "time_advances": 0})
    return working, accepted


def import_snapshot(system, template, record, *, time_s, voltage_V):
    """Import a V37 snapshot; its historical coordinate is explicitly absent."""
    return _restore(system, template, record,
        time_s=_number(time_s, "time_s", nonnegative=True),
        voltage_V=_number(voltage_V, "voltage_V"))


def restore_checkpoint(system, template, record):
    """Validate identity and reconstruct a new exact accepted-state reference."""
    if (not isinstance(record, dict) or set(record) != _CHECKPOINT_KEYS
            or record["schema"] != SCHEMA or record["operator_representation"] != lift.REPRESENTATION
            or record["coordinate_role"] != COORDINATE_ROLE):
        raise InputLiftCheckpointError("input-lift checkpoint schema/representation/coordinate role mismatch")
    if record["sha256"] != _digest({name: value for name, value in record.items() if name != "sha256"}):
        raise InputLiftCheckpointError("checkpoint SHA256 mismatch")
    values = _validate_snapshot(system, record["snapshot"])
    if (record["state_identity"] != _digest(record["snapshot"]["represented_fields"])
            or record["prior_coordinate_reference_identity"] != record["snapshot"]["coordinate_reference_identity"]
            or record["new_reference_identity"] != lift._identity({name: values[name] for name in lift.PRIMARY_FIELDS})):
        raise InputLiftCheckpointError("checkpoint state/reference identity mismatch")
    coordinate = _array(record["source_coordinate"], "source_coordinate", (system.dimension,))
    return _restore(system, template, record["snapshot"],
        time_s=_number(record["time_s"], "time_s", nonnegative=True),
        voltage_V=_number(record["voltage_V"], "voltage_V"),
        checkpoint_sha256=record["sha256"], model=record["physical_model"], source_coordinate=coordinate)


def replay_saved_step(system, previous, state, *, dt_s, storage_scale, poisson_scale,
                      local_scale, reported=None):
    """Recompute the saved step residual and physics without advancing time."""
    dt = _number(dt_s, "dt_s", positive=True)
    _validate_snapshot(system, lift.snapshot(system, previous))
    _validate_snapshot(system, lift.snapshot(system, state))
    before, current = lift.primary_inputs(previous), lift.primary_inputs(state)
    scales = []
    for name, raw, shape in (("storage_scale", storage_scale, current["storage"].shape),
                             ("poisson_scale", poisson_scale, current["poisson_residual_C_m2"].shape),
                             ("local_scale", local_scale, current["local_residual"].shape)):
        array = _array(raw, name, shape)
        if np.any(array <= 0):
            raise InputLiftCheckpointError(name + " must be strictly positive")
        scales.append(DD(array))
    residual = lift.cat((current["storage"] - before["storage"] - DD(dt)*current["rate"])/scales[0],
                        current["poisson_residual_C_m2"]/scales[1], current["local_residual"]/scales[2])
    return {"schema": "R1InputLiftSavedStepReplayV1", "operator_representation": lift.REPRESENTATION,
        "scope": "saved_primary_word_physics_and_saved_equation_residual_no_time_advance",
        "dt_s": dt, "scaled_residual_vector": residual.hi.tolist(),
        "maximum_scaled_residual": float(np.max(np.abs(residual.hi))),
        "previous_state_identity": _digest(lift._words(before)),
        "state_identity": _digest(lift._words(current)),
        "independent_physics": lift.independent_physics_row(system, state, previous, dt, reported=reported),
        "nonlinear_solves": 0, "time_advances": 0}


__all__ = ["SCHEMA", "InputLiftCheckpointError", "save_checkpoint", "restore_checkpoint",
           "import_snapshot", "replay_saved_step"]
