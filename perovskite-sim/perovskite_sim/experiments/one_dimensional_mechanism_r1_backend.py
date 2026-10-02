"""Explicit per-run numerical backend for the R1 production protocol.

The backend is immutable and travels with each reconstructed system. Importing
or selecting it never changes module functions, aliases or the legacy default.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy

import numpy as np


@dataclass(frozen=True, slots=True)
class R1Backend:
    representation_id: str

    @property
    def is_pair(self):
        return self.representation_id == "float64-pair-v1"

    def bind(self, system):
        existing = getattr(system, "_r1_backend", None)
        if existing is not None and existing != self:
            raise ValueError("a reconstructed R1 system cannot change numerical backend")
        system._r1_backend = self
        return system

    def decode_prepared(self, prepared):
        from .one_dimensional_mechanism_r1_state import R1PreparedState
        if self.is_pair:
            from .one_dimensional_mechanism_r1_pair_codec import decode_prepared
            return decode_prepared(prepared, representation=self.representation_id)
        # Pair prepared objects subclass the legacy API; the schema, rather
        # than isinstance alone, must still enforce the selected representation.
        value = prepared.to_dict() if hasattr(prepared, "to_dict") else prepared
        return R1PreparedState.from_dict(value)

    def fine_factory(self, baseline, initial):
        from .one_dimensional_mechanism_r1_precision import CompensatedR1System
        system, state = CompensatedR1System.from_baseline(baseline, initial)
        self.bind(system)
        return system, state

    def prepare(self, stack, intervals, binding, *, policy=None):
        from .one_dimensional_mechanism_r1_pair_codec import prepare_pair_state
        return prepare_pair_state(stack, intervals, binding, policy=policy,
            legacy_prepare=_legacy_prepare, legacy_verify=_legacy_verify,
            fine_factory=self.fine_factory, snapshot=self.snapshot)

    def verify(self, prepared, stack, binding, *, policy=None):
        from .one_dimensional_mechanism_r1_pair_codec import verify_pair_state
        return verify_pair_state(prepared, stack, binding, policy=policy,
            legacy_verify=_legacy_verify, fine_factory=self.fine_factory,
            snapshot=self.snapshot)

    def controlled(self, base, initial, stack, binding, controls, policy):
        from .one_dimensional_mechanism_r1_state import _make_system
        system = _make_system(stack, base.grid, base.material, base.common_dc_state,
                              binding, controls, policy)
        if self.is_pair:
            from .one_dimensional_mechanism_r1_precision import CompensatedR1System, REFERENCE_FIELDS
            # Build every control-dependent legacy dependency normally, then
            # transfer only the already verified common physical reference.
            # Derived currents, capture rates and local residuals are evaluated
            # anew with the selected controls; no DC solve or fine init repeats.
            system.__class__ = CompensatedR1System
            for name in ("_prescribed_trace_jump", "_precision_upstream",
                         "precision_initialization_diagnostics", "_fine_anchor"):
                setattr(system, name, copy.deepcopy(getattr(base, name)))
            system._fine_reference = {k: initial.fine[k].copy() for k in REFERENCE_FIELDS}
            system._fine_work = {}
            system._fine_evaluation_cache = {}
            system._fine_constants = {}
            system._reference_quantization_enabled = False
            system.rebase_evidence = None
            system.precision_fault = "none"
            system.precision_constraint_corrections = 0
            system._step_reference = None
            system._maximum_eliminated_operator_components = {}
        self.bind(system)
        return system, system.evaluate(system.initial_coordinate(), 0.0)

    def snapshot(self, system, state):
        from .one_dimensional_mechanism_r1_state import _snapshot_legacy
        record = _snapshot_legacy(system, state)
        if self.is_pair:
            from .one_dimensional_mechanism_r1_precision import PrecisionState
            if not isinstance(state, PrecisionState):
                raise TypeError("pair backend requires an actual compensated state")
            for key, value in state.fine.items():
                record["precision_" + key + "_hi"] = value.hi.tolist()
                record["precision_" + key + "_lo"] = value.lo.tolist()
        return record

    def output_states(self, system, states):
        result = {name: np.asarray([getattr(s, attribute) for s in states])
            for name, attribute in (("n_m3", "n"), ("p_m3", "p"), ("positive_m3", "positive"),
                                    ("occupancy", "occupancy"), ("phi_V", "phi"),
                                    ("sheet_charge_C_m2", "sheet_charge"))}
        if self.is_pair and states:
            snapshots = [self.snapshot(system, state) for state in states]
            result.update({key: np.asarray([row[key] for row in snapshots])
                           for key in snapshots[0] if key.startswith("precision_")})
        return result

    def solve_step(self, *args, **kwargs):
        from .interface_defect_transient import _solve_step
        if self.is_pair:
            from .one_dimensional_mechanism_r1_precision import _precision_solve_step
            return _precision_solve_step(_solve_step, *args, **kwargs)
        return _solve_step(*args, **kwargs)

    def initial_local_solved(self, system, state, voltage):
        if self.is_pair:
            from .one_dimensional_mechanism_r1_precision import capture_initial_local_context
            capture_initial_local_context(system, state, voltage)

    def finalize_initial(self, system, before, result):
        if self.is_pair:
            from .one_dimensional_mechanism_r1_precision import finalize_initial_precision
            return finalize_initial_precision(system, before, result, snapshot=self.snapshot)
        return result

    def finalize_record(self, record, *, kind):
        if self.is_pair:
            from .one_dimensional_mechanism_r1_pair_codec import finalize_record
            return finalize_record(record, kind=kind)
        return record


LEGACY = R1Backend("float64-baseline")
PAIR = R1Backend("float64-pair-v1")


def get_backend(value=None):
    if isinstance(value, R1Backend):
        if value not in (LEGACY, PAIR):
            raise ValueError("unrecognized R1 numerical backend")
        return value
    if value is None or value in ("legacy", "float64", "float64-baseline"):
        return LEGACY
    if value in ("pair", "float64-pair-v1"):
        return PAIR
    raise ValueError("unrecognized R1 numerical backend")


def backend_for(system, value=None):
    attached = getattr(system, "_r1_backend", None)
    selected = get_backend(attached if value is None else value)
    if attached is not None and attached != selected:
        raise ValueError("R1 system and explicit numerical backend disagree")
    return selected


resolve_backend = get_backend


def _legacy_prepare(*args, **kwargs):
    from .one_dimensional_mechanism_r1_state import prepare_common_state
    return prepare_common_state(*args, **kwargs, backend=LEGACY)


def _legacy_verify(*args, **kwargs):
    from .one_dimensional_mechanism_r1_state import verify_prepared_physics
    return verify_prepared_physics(*args, **kwargs, backend=LEGACY)
