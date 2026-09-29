"""State-owned stable local carrier assembly for the explicit R1 backends.

Only material/table coefficients are cached on a system. Every constitutive
evaluation is immutable data carried by its own local state and aggregate;
later evaluations cannot change the currents or tangents of an earlier state.
The baseline supplies its actual binary64 inputs with zero low words. The
pair adapter supplies its complete physical inputs without changing storage,
coordinate updates, scales, or acceptance rules.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
from types import MappingProxyType
from typing import Mapping

import numpy as np
from scipy import sparse

from perovskite_sim.constants import Q
from perovskite_sim.experiments.interface_defect_transient import _LocalState, _RIGHT_FIRST
from perovskite_sim.physics.compensated import DD
from perovskite_sim.physics.interface_plane import FERMI_DIRAC_RICHARDSON
from perovskite_sim.physics.two_sided_interface import (
    TwoSidedMaterialQSSResult,
    _material_two_sided_interface_problem,
    electrostatic_trace_residual_and_jacobian,
    shared_trap_occupancy,
)
from perovskite_sim.physics.two_sided_interface_compensated import (
    default_fd_table, evaluate_local_carrier_pair, freeze_coefficients,
    to_production_tangent,
)


_CHARGE = DD(Q)


def _cat(*values):
    if not values:
        return DD(np.empty(0))
    return DD._trusted_parts(
        np.concatenate([np.atleast_1d(value.hi) for value in values]),
        np.concatenate([np.atleast_1d(value.lo) for value in values]),
    )


def _put(value, index, part):
    part = part if isinstance(part, DD) else DD(part)
    high, low = value.hi.copy(), value.lo.copy()
    high[index], low[index] = part.hi, part.lo
    return DD._trusted_parts(high, low)


@dataclass(frozen=True, slots=True)
class LocalCarrierInputs:
    state_density: DD
    trace_potential: DD
    bulk_density: DD
    bulk_potential: DD
    occupancy: DD

    def __post_init__(self):
        for name, shape in (("state_density", (4,)), ("trace_potential", (2,)),
                            ("bulk_density", (4,)), ("bulk_potential", (2,)),
                            ("occupancy", ())):
            value = getattr(self, name)
            if not isinstance(value, DD) or value.shape != shape:
                raise TypeError(name + " must be a DD local input of shape " + repr(shape))

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self


@dataclass(frozen=True, slots=True)
class LocalCarrierEvaluation:
    inputs: LocalCarrierInputs
    coefficient_identity: str
    schema: str
    scope: str
    balance: Mapping[str, DD]
    tangent: Mapping[str, DD]
    metadata: Mapping

    def as_evaluation(self):
        return MappingProxyType({"schema": self.schema, "scope": self.scope,
                                 "balance": self.balance, "tangent": self.tangent,
                                 "metadata": self.metadata})

    def production_tangent(self):
        return to_production_tangent(self.as_evaluation())

    def __deepcopy__(self, memo):
        # Every component is immutable; keep MappingProxyType out of pickle-
        # based copying when an otherwise mutable device state is copied.
        memo[id(self)] = self
        return self


@dataclass(slots=True, kw_only=True)
class _StableLocalState(_LocalState):
    carrier_data: LocalCarrierEvaluation


@dataclass(frozen=True, kw_only=True)
class _StableInterfaceQSS(TwoSidedMaterialQSSResult):
    carrier_data: tuple[LocalCarrierEvaluation, ...] = field(default_factory=tuple)


def float_local_carrier_inputs(system, index, n, p, phi, occupancy,
                               trace_potential, trace_log_state, *, trace_density_m3=None):
    """Retain the original float state/coordinate map, including initial exp."""
    left, right = system.left_nodes[index], system.right_nodes[index]
    density = (np.exp(np.asarray(trace_log_state[index], dtype=float))
               if trace_density_m3 is None else np.asarray(trace_density_m3[index], dtype=float))
    return LocalCarrierInputs(
        DD(density), DD(trace_potential[index]),
        DD([n[left], p[left], n[right], p[right]]),
        DD([phi[left], phi[right]]), DD(occupancy[index]),
    )


def fine_local_carrier_inputs(system, index, n, p, phi, occupancy,
                              trace_potential, trace_log_state, *, trace_density_m3=None, fine):
    """Transfer this call's complete fields after checking its high-word view."""
    del trace_log_state
    for name, actual in (("n_m3", n), ("p_m3", p), ("phi_V", phi),
                         ("occupancy", occupancy), ("trace_potential_V", trace_potential)):
        value = fine.get(name)
        if not isinstance(value, DD) or not np.array_equal(value.hi, np.asarray(actual)):
            raise ValueError("complete local carrier inputs disagree with current high words: " + name)
    density = fine.get("trace_state_m3")
    if (not isinstance(density, DD) or trace_density_m3 is None
            or not np.array_equal(density.hi, np.asarray(trace_density_m3))):
        raise ValueError("complete local carrier input requires its actual resolved trace density")
    left, right = system.left_nodes[index], system.right_nodes[index]
    return LocalCarrierInputs(
        density[index], fine["trace_potential_V"][index],
        _cat(fine["n_m3"][left], fine["p_m3"][left], fine["n_m3"][right], fine["p_m3"][right]),
        _cat(fine["phi_V"][left], fine["phi_V"][right]), fine["occupancy"][index],
    )


def _coefficients(system, index, geometry, physics):
    table, table_identity = default_fd_table()
    # Carrier coefficients do not depend on dynamic sheet charge; callers
    # supply the original uncharged geometry before the electrostatic replace.
    descriptor = {"geometry": vars(geometry), "physics": vars(physics),
                  "fd_table_sha256": table_identity, "q_C": Q}
    encoded = json.dumps(descriptor, sort_keys=True, separators=(",", ":"), allow_nan=False)
    identity = hashlib.sha256(encoded.encode()).hexdigest()
    cache = getattr(system, "_r1_local_carrier_coefficients", {})
    cached = cache.get(index)
    if cached is None or cached[0] != identity:
        coefficients = freeze_coefficients(vars(geometry), vars(physics), fd_table=table,
            fd_table_sha256=table_identity, q_C=Q)
        updated = dict(cache)
        updated[index] = (identity, coefficients)
        system._r1_local_carrier_coefficients = MappingProxyType(updated)
    else:
        coefficients = cached[1]
    return identity, coefficients


def evaluate_local_carrier_inputs(system, index, geometry, physics, inputs):
    """Evaluate explicit local inputs; retain no state-dependent system cache."""
    if not isinstance(inputs, LocalCarrierInputs):
        raise TypeError("local carrier assembly requires explicit LocalCarrierInputs")
    identity, coefficients = _coefficients(system, index, geometry, physics)
    evaluated = evaluate_local_carrier_pair(
        state_density=inputs.state_density, trace_potential=inputs.trace_potential,
        bulk_density=inputs.bulk_density, bulk_potential=inputs.bulk_potential,
        occupancy=inputs.occupancy, capture_multiplier=system.capture_multiplier,
        coefficients=coefficients,
    )
    # The pure physics API freezes its nested mappings and array metadata.
    return LocalCarrierEvaluation(inputs, identity, evaluated["schema"], evaluated["scope"],
                                  evaluated["balance"], evaluated["tangent"], evaluated["metadata"])


def evaluate_interface_from_fine(system, fine, index=0):
    """Fresh local evaluation from explicit complete fields for initialization."""
    n, p, phi = (fine[name].hi for name in ("n_m3", "p_m3", "phi_V"))
    geometry, physics, _ = _material_two_sided_interface_problem(
        system.material, system.stack, n, p, phi, index,
        cross_transmission=system.dark_reference.interface_transmission)
    inputs = fine_local_carrier_inputs(system, index, n, p, phi,
        fine["occupancy"].hi, fine["trace_potential_V"].hi, None,
        trace_density_m3=fine["trace_state_m3"].hi, fine=fine)
    return evaluate_local_carrier_inputs(system, index, geometry, physics, inputs)


def independent_interface_evaluation(system, state, index):
    """Reconstruct from physical fields, ignoring every direct local payload."""
    fine = getattr(state, "fine", None)
    if fine is not None:
        # Only primary fields are selected by the fine adapter; stored fine
        # currents, rates, residuals and direct carrier_data are never inputs.
        return evaluate_interface_from_fine(system, fine, index)
    n, p, phi = state.n, state.p, state.phi
    trace_potential = np.asarray([item.trace_potential for item in state.local])
    trace_density = np.asarray([item.state_m3 for item in state.local])
    geometry, physics, _ = _material_two_sided_interface_problem(
        system.material, system.stack, n, p, phi, index,
        cross_transmission=system.dark_reference.interface_transmission)
    inputs = float_local_carrier_inputs(system, index, n, p, phi, state.occupancy,
        trace_potential, None, trace_density_m3=trace_density)
    return evaluate_local_carrier_inputs(system, index, geometry, physics, inputs)


def build_r1_local_states(system, n, p, phi, occupancy, trace_potential,
                          trace_log_state, *, trace_density_m3=None):
    states, evaluations = [], []
    size = 4 * system.interface_count
    aggregate = {name: np.empty(size) for name in ("state_m3", "bulk_flux_m2_s",
        "cross_flux_m2_s", "capture_flux_m2_s", "state_flux_m2_s")}
    maximum_residual = 0.0
    for index in range(system.interface_count):
        geometry, physics, bulk = _material_two_sided_interface_problem(
            system.material, system.stack, n, p, phi, index,
            cross_transmission=system.dark_reference.interface_transmission)
        sheet_charge = -Q * system.trap_density[index] * (occupancy[index] - system.equilibrium_occupancy[index])
        charged_geometry = replace(geometry,
            fixed_sheet_charge_C_m2=float(geometry.fixed_sheet_charge_C_m2) + sheet_charge)
        electrostatic, _ = electrostatic_trace_residual_and_jacobian(
            trace_potential[index], charged_geometry, bulk)
        inputs = system._local_carrier_inputs(index, n, p, phi, occupancy,
            trace_potential, trace_log_state, trace_density_m3=trace_density_m3)
        evaluated = evaluate_local_carrier_inputs(system, index, geometry, physics, inputs)
        tangent = evaluated.production_tangent()
        balance = tangent.balance
        block = slice(4*index, 4*index+4)
        for name in ("state_m3", "bulk_flux_m2_s", "cross_flux_m2_s", "capture_flux_m2_s"):
            aggregate[name][block] = getattr(balance, name)[_RIGHT_FIRST]
        aggregate["state_flux_m2_s"][block] = balance.residual_m2_s[_RIGHT_FIRST]
        maximum_residual = max(maximum_residual, float(np.max(
            np.abs(balance.residual_m2_s) / system.reference_local_scale[index, 2:])))
        states.append(_StableLocalState(
            trace_potential=np.asarray(trace_potential[index]).copy(),
            log_state=np.asarray(trace_log_state[index]).copy(),
            state_m3=np.asarray(balance.state_m3).copy(),
            quasi_steady_occupancy=shared_trap_occupancy(balance.state_m3, physics),
            sheet_charge_C_m2=float(sheet_charge), electrostatic_residual=np.asarray(electrostatic).copy(),
            tangent=tangent, carrier_data=evaluated))
        evaluations.append(evaluated)
    return tuple(states), _StableInterfaceQSS(**aggregate,
        normalized_residual=maximum_residual, evaluations=0,
        transport_model=FERMI_DIRAC_RICHARDSON, occupancy=np.asarray(occupancy).copy(),
        carrier_data=tuple(evaluations))


def carrier_data(item):
    value = getattr(item, "carrier_data", None)
    if not isinstance(value, LocalCarrierEvaluation):
        raise TypeError("R1 local consumers require the state-owned carrier evaluation")
    return value


def without_local_exchange(interface_qss):
    if not isinstance(interface_qss, _StableInterfaceQSS):
        raise TypeError("R1 source requires an explicit stable interface aggregate")
    return replace(interface_qss, bulk_flux_m2_s=np.zeros_like(interface_qss.bulk_flux_m2_s))


def assemble_r1_carrier_source(system, source_without_exchange, interface_qss):
    if not isinstance(interface_qss, _StableInterfaceQSS):
        raise TypeError("R1 source requires the same call's stable aggregate")
    if len(interface_qss.carrier_data) != system.interface_count:
        raise ValueError("R1 interface source count differs from the material topology")
    source = DD(source_without_exchange)
    if source.shape != (2*system.node_count,):
        raise ValueError("R1 bulk source shape differs from the carrier layout")
    for index, (left, right) in enumerate(zip(system.left_nodes, system.right_nodes)):
        flux = interface_qss.carrier_data[index].balance["bulk_flux_m2_s"]
        for local, node, offset in ((0, left, 0), (1, left, system.node_count),
                                    (2, right, 0), (3, right, system.node_count)):
            row = offset + node
            source = _put(source, row, source[row] - flux[local]/DD(system.widths[node]))
    return source


def _divergence(flux):
    return _cat(flux[0], flux[1:]-flux[:-1], -flux[-1])


def r1_carrier_rate_fields(system, source, transport_n, transport_p, local):
    if not isinstance(source, DD) or source.shape != (2*system.node_count,):
        raise TypeError("R1 rates require their complete assembled carrier source")
    electron = transport_n if isinstance(transport_n, DD) else DD(transport_n)
    hole = transport_p if isinstance(transport_p, DD) else DD(transport_p)
    expected = (system.node_count-1,)
    if electron.shape != expected or hole.shape != expected:
        raise ValueError("R1 transport arrays differ from the face topology")
    widths = DD(system.widths)
    rate_n = source[:system.node_count] + _divergence(electron)/(_CHARGE*widths)
    rate_p = source[system.node_count:] - _divergence(hole)/(_CHARGE*widths)
    captures = [carrier_data(item).balance["capture_flux_m2_s"] for item in local]
    trap_rate = _cat(*(value[0]+value[2]-value[1]-value[3] for value in captures))
    return rate_n.to_float(), rate_p.to_float(), trap_rate.to_float()


def r1_reported_interface_currents(system, local, current_n, current_p):
    electron, hole = np.asarray(current_n).copy(), np.asarray(current_p).copy()
    for face, item in zip(system.interface_faces, local):
        flux = carrier_data(item).balance["bulk_flux_m2_s"]
        electron[face] = float((-_CHARGE*flux[0]).to_float())
        hole[face] = float((_CHARGE*flux[1]).to_float())
    return electron, hole


def interface_carrier_conduction_pair(system, item):
    flux = carrier_data(item).balance["bulk_flux_m2_s"]
    return DD(system.polarity)*_CHARGE*_cat(-flux[0]+flux[1], flux[2]-flux[3])


def _global_chain(system, index, left, right, bulk, trace, log_state, occupancy, f):
    """Map only the thirteen active columns in DD before sparse rounding."""
    columns = {}
    def add(column, value):
        columns[column] = columns[column]+value if column in columns else value
    vt = DD(system.thermal_voltage)
    for side, node in enumerate((left, right)):
        electron, hole = bulk[:, 2+2*side], bulk[:, 3+2*side]
        add(system.electron_slice.start+node-1, electron)
        add(system.hole_slice.start+node-1, hole)
        add(system.potential_slice.start+node-1, vt*bulk[:, side]+electron-hole)
    block = system._local_block_slice(index)
    for side in range(2):
        add(block.start+side, vt*trace[:, side])
    for component in range(4):
        add(block.start+2+component, log_state[:, component])
    add(system.trap_slice.start+index, occupancy*f*(1-f))
    ordered = sorted(columns)
    values = _cat(*(columns[column] for column in ordered)).reshape(len(ordered), 4)
    return ordered, DD._trusted_parts(values.hi.T.copy(), values.lo.T.copy())


def _csr(columns, values, dimension):
    array = values.to_float()
    rows = np.repeat(np.arange(array.shape[0]), len(columns))
    matrix = sparse.csr_matrix((array.reshape(-1), (rows, np.tile(columns, array.shape[0]))),
                               shape=(array.shape[0], dimension))
    matrix.eliminate_zeros()
    return matrix


def r1_local_carrier_jacobians(system, index, left, right, item):
    evaluated = carrier_data(item)
    tangent, balance, f = evaluated.tangent, evaluated.balance, evaluated.inputs.occupancy
    columns, bulk = _global_chain(system, index, left, right,
        tangent["bulk_flux_jacobian_bulk_coordinates"], tangent["bulk_flux_jacobian_trace_potential_m2_s_V"],
        tangent["bulk_flux_jacobian_log_state_m2_s"], DD(np.zeros(4)), f)
    # The base loop subtracts these positive receiving-node loss rows.
    receivers = DD(np.asarray(system.widths)[[left, left, right, right]])
    loss = bulk/receivers.reshape(4, 1)
    capture_columns, captures = _global_chain(system, index, left, right,
        DD(np.zeros((4, 6))), DD(np.zeros((4, 2))), tangent["capture_flux_jacobian_log_state_m2_s"],
        tangent["capture_flux_occupancy_derivative_m2_s"], f)
    trap_rate = (captures[0]+captures[2]-captures[1]-captures[3]).reshape(1, len(capture_columns))
    carrier_columns, carrier = _global_chain(system, index, left, right,
        balance["jacobian_bulk_coordinates"], balance["jacobian_trace_potential_m2_s_V"],
        balance["jacobian_log_state_m2_s"], tangent["residual_occupancy_derivative_m2_s"], f)
    return (_csr(columns, loss, system.dimension), _csr(capture_columns, trap_rate, system.dimension),
            _csr(carrier_columns, carrier, system.dimension))


__all__ = ["LocalCarrierInputs", "LocalCarrierEvaluation", "build_r1_local_states",
           "float_local_carrier_inputs", "fine_local_carrier_inputs", "evaluate_local_carrier_inputs",
           "evaluate_interface_from_fine", "independent_interface_evaluation", "carrier_data",
           "without_local_exchange", "assemble_r1_carrier_source", "r1_carrier_rate_fields",
           "r1_reported_interface_currents", "interface_carrier_conduction_pair", "r1_local_carrier_jacobians"]
