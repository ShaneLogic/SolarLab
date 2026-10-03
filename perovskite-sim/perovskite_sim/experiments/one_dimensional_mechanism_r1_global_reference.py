"""Independent ion elimination using the original global quasi-Fermi map.

Only QF, positive-ion and trap-occupancy inputs cross the live-state boundary.
The carrier anchors are the original ``phi0/log_n0/log_p0`` preparation, not a
rebased accepted population. DD is evaluation arithmetic here; using this
reference does not add low words to a baseline state.
"""
from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np

from perovskite_sim.constants import Q
from perovskite_sim.physics.compensated import DD
from perovskite_sim.solver.mol import poisson_right_boundary


FIXED_FIELDS = ("dqfn_V", "dqfp_V", "positive_m3", "occupancy")
_COEFFICIENTS = (
    "trap_density_m2", "equilibrium_occupancy", "static_sheet_charge_C_m2",
    "trace_jump_V", "grid_spacing_m", "control_volume_width_m",
    "ion_capacity_m3", "ion_diffusion_m2_s", "ion_thermal_voltage_V",
    "positive_nodes", "site_occupancy_ceiling", "ion_flux_enabled",
    "boundary_flux_m2_s", "model",
)
_MODEL = "one_positive_ion_steric_diffusion_only"


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json(text):
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("global reference JSON contains duplicate keys")
            result[key] = value
        return result
    if not isinstance(text, str):
        raise TypeError("global reference DTO owns JSON strings")
    return json.loads(text, object_pairs_hook=unique)


def _words(value):
    return {"hi": value.hi.tolist(), "lo": value.lo.tolist()}


def _keys(value, names, label):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError(f"global reference {label} field coverage mismatch")


def _array(value, shape, label):
    result = np.asarray(value)
    if result.dtype.kind not in "iuf" or not np.isfinite(result).all():
        raise ValueError(f"global reference {label} must be finite real numbers")
    if result.size == 0 and np.prod(shape) == 0:
        result = result.reshape(shape)
    if result.shape != shape:
        raise ValueError(f"global reference {label} must have shape {shape}")
    return result.astype(float)


def _pair(value, shape, label):
    _keys(value, ("hi", "lo"), label)
    return DD(_array(value["hi"], shape, label), _array(value["lo"], shape, label))


def _put(value, index, part):
    part = part if isinstance(part, DD) else DD(part)
    hi, lo = value.hi.copy(), value.lo.copy()
    hi[index], lo[index] = part.hi, part.lo
    return DD(hi, lo)


def _decode(inputs):
    # The original Poisson DTO has its own field allowlist. Validate physical
    # shapes/domains here before invoking that unchanged isolated solver.
    fixed = _json(inputs.fixed_inputs_json)
    a = _json(inputs.coefficients_json)
    p = _json(inputs.poisson_inputs_json)
    _keys(fixed, FIXED_FIELDS, "fixed inputs")
    _keys(a, _COEFFICIENTS, "coefficients")
    _keys(p, ("schema", "fixed_inputs", "preparation", "seed_phi_V", "seed_provenance"), "Poisson input")
    if p["schema"] != "R1IndependentPoissonInputsV1":
        raise ValueError("global reference requires the original global-QF Poisson DTO")
    prep = p["preparation"]
    shape = np.asarray(prep["phi0_V"]).shape
    if len(shape) != 1 or shape[0] < 3:
        raise ValueError("global reference requires at least three nodes")
    n = shape[0]
    count = len(a["trap_density_m2"])
    f = {key: _pair(value, (count,) if key == "occupancy" else shape, key)
         for key, value in fixed.items()}
    for key in ("phi0_V", "log_n0", "log_p0", "N_D_m3", "N_A_m3", "ion_background_m3"):
        _array(prep[key], shape, key)
    for key in ("contact_n_m3", "contact_p_m3", "boundary_phi_V"):
        _array(prep[key], (2,), key)
    for key, item_shape in (("poisson_capacitance_F_m2", (n-1,)),
                            ("poisson_width_m", (n-2,)), ("thermal_voltage_V", ())):
        if np.any(_array(prep[key], item_shape, key) <= 0):
            raise ValueError(f"global reference {key} must be positive")
    if any(np.any(np.asarray(prep[key]) <= 0) for key in ("contact_n_m3", "contact_p_m3")):
        raise ValueError("global reference contact populations must be positive")
    _pair(p["seed_phi_V"], shape, "seed potential")
    shapes = {"trap_density_m2": (count,), "equilibrium_occupancy": (count,),
              "static_sheet_charge_C_m2": (count,), "trace_jump_V": (count,),
              "grid_spacing_m": (n-1,), "control_volume_width_m": shape,
              "ion_capacity_m3": shape, "ion_diffusion_m2_s": (n-1,),
              "ion_thermal_voltage_V": (), "site_occupancy_ceiling": ()}
    c = {key: DD(_array(a[key], item_shape, key)) for key, item_shape in shapes.items()}
    c["boundary_flux_m2_s"] = _pair(a["boundary_flux_m2_s"], (2,), "boundary flux")
    if any(np.any(c[key].hi != 0) for key in ("static_sheet_charge_C_m2", "trace_jump_V")):
        raise ValueError("global reference requires zero static sheet charge and trace jump")
    for key in ("grid_spacing_m", "control_volume_width_m", "ion_capacity_m3", "ion_thermal_voltage_V"):
        if np.any(c[key].hi <= 0):
            raise ValueError(f"global reference {key} must be positive")
    for key in ("trap_density_m2", "ion_diffusion_m2_s"):
        if np.any(c[key].hi < 0):
            raise ValueError(f"global reference {key} must be nonnegative")
    for key in ("occupancy", "equilibrium_occupancy"):
        value = f[key] if key == "occupancy" else c[key]
        if np.any(value <= 0) or np.any(value >= 1):
            raise ValueError(f"global reference {key} must lie inside (0, 1)")
    active = np.asarray(a["positive_nodes"])
    active = _array(active, (active.size,), "positive nodes")
    if (np.any(active != np.floor(active)) or np.any(active < 0) or np.any(active >= n)
            or len(np.unique(active)) != len(active)):
        raise ValueError("global reference positive nodes are invalid")
    if np.any(f["positive_m3"] < 0) or np.any(f["positive_m3"][active.astype(int)] <= 0):
        raise ValueError("global reference ion population is invalid")
    ceiling = c["site_occupancy_ceiling"]
    if ceiling <= 0 or ceiling > .999 or np.any(f["positive_m3"]/c["ion_capacity_m3"] >= ceiling):
        raise ValueError("global reference reached the original ion site bound")
    nodes = _array(prep["interface_nodes"], (count, 2), "interface nodes")
    if (np.any(nodes != np.floor(nodes)) or np.any(nodes < 1) or np.any(nodes >= n-1)
            or np.any(nodes[:, 1] != nodes[:, 0]+1)):
        raise ValueError("global reference interfaces must join adjacent interior nodes")
    weights = _array(prep["sheet_weights"], (count, 2), "sheet weights")
    if np.any(weights < 0):
        raise ValueError("global reference sheet weights must be nonnegative")
    # Preserve the pair implementation's association, while independently
    # rebuilding the value from occupancy rather than accepting direct sheet.
    sheet = DD(Q)*c["trap_density_m2"]*(c["equilibrium_occupancy"]-f["occupancy"])
    expected = {**fixed, "sheet_charge_C_m2": _words(sheet)}
    if p["fixed_inputs"] != expected:
        raise ValueError("global reference Poisson inputs differ from independently rebuilt fixed charge")
    if type(a["ion_flux_enabled"]) is not bool or a["model"] != _MODEL:
        raise ValueError("global reference requires the original positive-ion model")
    return f, c, a["ion_flux_enabled"]


@dataclass(frozen=True, slots=True)
class GlobalIonReferenceInputs:
    """Owned immutable input words; no trial potential, carrier or rate cache."""

    fixed_inputs_json: str
    poisson_inputs_json: str
    coefficients_json: str

    def __post_init__(self):
        for key in ("fixed_inputs_json", "poisson_inputs_json", "coefficients_json"):
            object.__setattr__(self, key, _canonical(_json(getattr(self, key))))
        self.poisson_inputs  # Validate the original DTO's field coverage.
        _decode(self)

    @property
    def poisson_inputs(self):
        from .one_dimensional_mechanism_r1_precision import IndependentPoissonInputs
        value = _json(self.poisson_inputs_json)
        return IndependentPoissonInputs(
            _canonical(value["fixed_inputs"]), _canonical(value["preparation"]),
            _canonical(value["seed_phi_V"]), value["seed_provenance"])

    def to_dict(self):
        return {"schema": "R1GlobalIonReferenceInputsV1",
                "fixed_inputs": _json(self.fixed_inputs_json),
                "poisson_inputs": _json(self.poisson_inputs_json),
                "coefficients": _json(self.coefficients_json)}


def make_global_ion_reference_inputs(system, fixed_inputs, seed_phi, voltage, *, boundary_flux=None):
    """Copy four fixed fields and immutable original global preparation data."""
    _keys(fixed_inputs, FIXED_FIELDS, "fixed inputs (rejects direct outputs)")
    if any(not isinstance(value, DD) for value in fixed_inputs.values()):
        raise TypeError("global reference fixed inputs require explicit DD words")
    mat, source = system.material, system.system.source_mat
    if mat.has_dual_ions or not mat.ion_steric_diffusion_only or np.size(system.negative_nodes):
        raise ValueError("global reference supports only the original positive-ion steric model")
    params = source.carrier_params
    if (params.get("carrier_statistics", "maxwell_boltzmann") != "maxwell_boltzmann"
            or getattr(source, "has_selective_contacts", False)):
        raise ValueError("global reference requires Maxwell-Boltzmann pinned contacts")
    if any(getattr(owner, key, None) is not None for owner in (mat, source) for key in
           ("monovalent_bulk_defects", "multivalent_bulk_defects", "frozen_metastable_defects")):
        raise ValueError("global reference does not support additional charged bulk defects")
    from .one_dimensional_mechanism_r1_local_carrier import _material_two_sided_interface_problem
    geometry = [_material_two_sided_interface_problem(
        mat, system.stack, system.reference_n, system.reference_p, system.system.phi0,
        k, cross_transmission=system.dark_reference.interface_transmission)[0]
        for k in range(system.interface_count)]
    voltage = float(_array(voltage, (), "applied voltage"))
    switch = system.controls.nu_I
    if isinstance(switch, (float, np.floating)) or switch not in (0, 1):
        raise ValueError("global reference ion switch must be zero or one")
    boundary = DD(np.zeros(2)) if boundary_flux is None else boundary_flux
    if not isinstance(boundary, DD):
        raise TypeError("global reference boundary flux must retain DD words")
    coefficients = {
        "trap_density_m2": system.trap_density, "equilibrium_occupancy": system.equilibrium_occupancy,
        "static_sheet_charge_C_m2": [g.fixed_sheet_charge_C_m2 for g in geometry],
        "trace_jump_V": [g.potential_jump_right_minus_left_V for g in geometry],
        "grid_spacing_m": np.diff(system.grid), "control_volume_width_m": system.widths,
        "ion_capacity_m3": mat.P_lim_node, "ion_diffusion_m2_s": mat.D_ion_face,
        "ion_thermal_voltage_V": mat.V_T_device, "positive_nodes": system.positive_nodes,
        "site_occupancy_ceiling": system.site_occupancy_ceiling,
    }
    coefficients = {key: np.asarray(value).tolist() for key, value in coefficients.items()}
    coefficients.update(ion_flux_enabled=bool(switch), boundary_flux_m2_s=_words(boundary), model=_MODEL)
    prep = {
        "phi0_V": system.system.phi0, "log_n0": system.system.log_n0, "log_p0": system.system.log_p0,
        "contact_n_m3": system.reference_n[[0, -1]], "contact_p_m3": system.reference_p[[0, -1]],
        "thermal_voltage_V": system.thermal_voltage,
        "poisson_capacitance_F_m2": mat.poisson_factor.C, "poisson_width_m": mat.poisson_factor.h_cell,
        "N_D_m3": mat.N_D, "N_A_m3": mat.N_A, "ion_background_m3": mat.P_ion0,
        "interface_nodes": list(zip(system.left_nodes, system.right_nodes)),
        "sheet_weights": [system._sheet_weights(k) for k in range(system.interface_count)],
        "boundary_phi_V": [0., poisson_right_boundary(mat, voltage)],
    }
    prep = {key: np.asarray(value).tolist() for key, value in prep.items()}
    fixed = {key: _words(value) for key, value in fixed_inputs.items()}
    sheet = DD(Q)*DD(system.trap_density)*(DD(system.equilibrium_occupancy)-fixed_inputs["occupancy"])
    poisson = {"schema": "R1IndependentPoissonInputsV1",
               "fixed_inputs": {**fixed, "sheet_charge_C_m2": _words(sheet)},
               "preparation": prep, "seed_phi_V": _words(DD(seed_phi)),
               "seed_provenance": "same_diagnostics_call_legacy_potential_eliminated_channel"}
    return GlobalIonReferenceInputs(_canonical(fixed), _canonical(poisson), _canonical(coefficients))


def evaluate_independent_ion_transport(inputs, phi, *, fault="none"):
    """Independent DD SG/FV with the original explicit pair fault semantics.

    No production flux function or live system is reachable here. The named
    faults remain actual constitutive changes for historical negative tests.
    """
    if not isinstance(inputs, GlobalIonReferenceInputs) or not isinstance(phi, DD):
        raise TypeError("independent transport requires isolated inputs and DD potential")
    f, a, enabled = _decode(inputs)
    _pair(_words(phi), f["positive_m3"].shape, "transport potential")
    if fault not in {"none", "thermal_voltage", "drift_sign", "diffusion", "omit_ion", "single_face_sign"}:
        raise ValueError("unknown implementation fault")
    diffusion = a["ion_diffusion_m2_s"]*(1.01 if fault == "diffusion" else 1)
    flux = DD(np.zeros(diffusion.shape))
    active = diffusion.hi > 0
    if enabled and np.any(active):
        population = f["positive_m3"]
        chemical = -(DD(1)-population/a["ion_capacity_m3"]).log()
        vt = a["ion_thermal_voltage_V"]*(1.01 if fault == "thermal_voltage" else 1)
        sign = -1 if fault == "drift_sign" else 1
        drive = (sign*(phi[1:][active]-phi[:-1][active])/vt
                 + (chemical[1:][active]-chemical[:-1][active]))
        magnitude = abs(drive)
        forward = DD(np.ones(magnitude.shape))
        nonzero = (magnitude.hi != 0) | (magnitude.lo != 0)
        if np.any(nonzero):
            forward = _put(forward, nonzero, magnitude[nonzero]/magnitude[nonzero].expm1())
        backward = forward+magnitude
        positive = drive.hi >= 0
        plus = DD(np.where(positive, forward.hi, backward.hi), np.where(positive, forward.lo, backward.lo))
        minus = DD(np.where(positive, backward.hi, forward.hi), np.where(positive, backward.lo, forward.lo))
        flux = _put(flux, active, diffusion[active]/a["grid_spacing_m"][active]*(
            plus*population[:-1][active]-minus*population[1:][active]))
        if fault == "omit_ion":
            flux = DD(np.zeros(flux.shape))
        if fault == "single_face_sign":
            first = int(np.flatnonzero(active)[0])
            flux = _put(flux, first, -flux[first])
    bounded = _put(DD(np.zeros(phi.size+1)), slice(1, -1), flux)
    bounded = _put(bounded, [0, -1], a["boundary_flux_m2_s"])
    return flux, -(bounded[1:]-bounded[:-1])/a["control_volume_width_m"]


def solve_global_ion_reference(inputs):
    """Original isolated global Poisson, followed by independent DD SG/FV."""
    if not isinstance(inputs, GlobalIonReferenceInputs):
        raise TypeError("global reference solve requires GlobalIonReferenceInputs")
    from .one_dimensional_mechanism_r1_precision import solve_independent_poisson
    from .one_dimensional_mechanism_r1_state import digest
    fields, poisson = solve_independent_poisson(inputs.poisson_inputs)
    flux, rate = evaluate_independent_ion_transport(inputs, fields["phi_V"])
    fields = {**fields, "positive_flux_m2_s": flux, "positive_rate_m3_s": rate}
    return fields, {
        "schema": "R1GlobalIonReferenceSolveV1", "poisson": poisson,
        "inputs": inputs.to_dict(), "inputs_sha256": digest(inputs.to_dict()),
        "fields": {key: _words(value) for key, value in fields.items()},
        "fields_sha256": digest({key: _words(value) for key, value in fields.items()}),
        "carrier_mapping": "original_global_phi0_log_n0_log_p0",
        "used_direct_phi": False, "used_direct_carriers": False,
        "used_direct_flux_or_rate": False, "used_direct_sheet_charge": False,
        "used_rebased_population_origin": False, "used_historical_residual": False,
        "ion_constitutive_implementation": "independent_DD_SG_and_explicit_boundary_FV_divergence",
    }


def baseline_eliminated_operator_diagnostics(system, state, voltage, legacy_values):
    """Replace only baseline ion references, keeping its high-array reduction."""
    if hasattr(state, "fine") or hasattr(state, "input_lift"):
        raise TypeError("baseline reference adapter requires the actual binary64 state")
    fixed = {key: DD(np.asarray(getattr(state, attribute), dtype=float)) for key, attribute in
             (("dqfn_V", "dqfn"), ("dqfp_V", "dqfp"), ("positive_m3", "positive"), ("occupancy", "occupancy"))}
    inputs = make_global_ion_reference_inputs(system, fixed, legacy_values["potential"]["eliminated"], voltage)
    fields, receipt = solve_global_ion_reference(inputs)
    values = dict(legacy_values)
    for name, attribute, field in (("positive_ion_flux", "positive_flux", "positive_flux_m2_s"),
                                   ("positive_ion_rate", "positive_rate", "positive_rate_m3_s")):
        old = legacy_values[name]
        left = np.asarray(getattr(state, attribute), dtype=float).copy()
        right = fields[field].hi.copy()
        if not np.array_equal(left, old["direct"]):
            raise ValueError("baseline ion diagnostic differs from its actual direct state")
        floor = float(old["normalization_floor"])
        left_max = float(np.max(np.abs(left), initial=0.))
        right_max = float(np.max(np.abs(right), initial=0.))
        scale = max(left_max, right_max, floor)
        delta = left-right
        absolute = float(np.max(np.abs(delta), initial=0.))
        values[name] = {**old, "direct": left, "eliminated": right, "difference": delta,
                        "direct_maximum_absolute": left_max, "eliminated_maximum_absolute": right_max,
                        "maximum_absolute_difference": absolute, "normalization_scale": scale,
                        "floor_active": max(left_max, right_max) < floor, "relative_error": absolute/scale,
                        "legacy_binary64": old}
    values["positive_ion_rate"]["independent_reference"] = receipt
    return values
