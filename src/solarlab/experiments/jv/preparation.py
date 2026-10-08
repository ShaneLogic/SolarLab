"""Declared 1D J-V history, with no physical state or executor construction."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, NoReturn

import numpy as np

from solarlab.config.resolve_device import resolve_device
from solarlab.device.defaults import DefaultCatalog
from solarlab.experiments.jv.inputs import DarkJVInput, JVExperimentInput, JVInput, JVProtocolInput
from solarlab.experiments.two_dimensional.inputs import invalid
from solarlab.experiments.two_dimensional.preparation import _document
from solarlab.materials.resources import ResourceLibrary
from solarlab.materials.source import SourceDocument


def _bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _fail(path: tuple[str | int, ...], message: str, value: Any) -> NoReturn:
    invalid("JVPreparation", path, message, value)


def _canonical(value: Any) -> Any:
    # The old declaration hash normalizes floating zero only. Editable input
    # independently keeps its sign and original numeric/string representation.
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    return 0.0 if type(value) is float and value == 0 else value


def _phase(name: str, light: str, duration: float, source: str | None = None, suns: float | None = None) -> dict[str, Any]:
    return dict(phase=name, condition=light, duration_s=duration, intensity_suns=suns,
                photon_flux_m2_s=None, relative_generation_change=None, source_reference=source)


def _settle(duration: float | None) -> dict[str, Any]:
    return dict(kind="not_applicable" if duration is None else "finite_time", duration_s=duration,
                max_carrier_area_rate_A_m2=None, max_ion_area_rate_A_m2=None,
                max_ionic_face_current_A_m2=None, max_face_current_spread_A_m2=None)


def _expected_protocol(values: dict[str, Any], temperature: float, waveform: dict[str, Any] | None,
                       soak_s: float) -> dict[str, Any] | None:
    maximum = values["V_max"]
    if maximum is None:
        return None
    count = values["n_points"]
    # This is a bounded metadata expansion, not an executor point limit.
    if count > 8192:
        _fail(("experiment", "n_points"), "preview expansion is limited to 8192 declared samples per branch; no trajectory was started", count)
    start = 0.0 if waveform is None else waveform["start_voltage_V"]
    forward = np.linspace(start, maximum, count).tolist()
    scan_s = abs(maximum - start) / values["v_rate"]
    if not math.isfinite(scan_s):
        _fail(("experiment", "v_rate"), "declared scan duration is not finite", values["v_rate"])
    illuminated = values["illuminated"]
    if waveform is not None:
        rate = waveform["uniform_generation_rate_m3_s"]
        source = "device_optics" if rate is None else f"uniform_absorber_G_m3_s:{rate:.17g}"
        light = "baseline" if illuminated else "dark"
        history = [
            _phase("dark_seed_at_0V", "dark", waveform["dark_seed_s"]),
            _phase("dark_prebias_at_scan_start", "dark", waveform["dark_prep_s"]),
            _phase("forward_start_dwell", light, waveform["branch_dwell_s"], source),
            _phase("forward_continuous_ramp", light, scan_s, source),
            _phase("turnaround_at_scan_stop", "dark" if waveform["turnaround_dark"] else light, waveform["turnaround_s"], source),
            _phase("reverse_start_dwell", light, waveform["branch_dwell_s"], source),
            _phase("reverse_continuous_ramp", light, scan_s, source),
        ]
        state, bias, soak, dwell = "finite_time_dc_preconditioned", start, waveform["dark_prep_s"], waveform["branch_dwell_s"]
        settle, mode, implicit = _settle(soak if soak else None), "piecewise_linear", False
        reverse = list(reversed(forward))
    else:
        dwell = abs(maximum) / (values["v_rate"] * (count - 1))
        reverse = np.linspace(maximum, 0.0, count).tolist()
        if illuminated:
            state, bias, soak = "finite_time_illuminated_preconditioned", 0.0, soak_s
            history = [_phase("short_circuit_preconditioning", "baseline", soak, "stack_baseline_generation", 1.0),
                       _phase("forward_reverse_scan", "baseline", 2.0 * count * dwell, "stack_baseline_generation", 1.0)]
            settle = _settle(soak)
        else:
            state, bias, soak = "dark_equilibrium", None, 0.0
            history, settle = [_phase("forward_reverse_scan", "dark", 2.0 * count * dwell)], _settle(None)
        mode, implicit = "linear", True
    declaration = dict(experiment="jv_hysteresis", initial_state_source=state, pre_bias_V=bias,
        soak_duration_s=soak, dwell_duration_s=dwell, illumination_history=history, temperature_K=temperature,
        scan=dict(axis="voltage_V", direction="ascending_then_descending", start=start, stop=maximum, rate_V_s=values["v_rate"]),
        ac_excitation=None, dc_settle=settle, sampling=dict(axis="voltage_V", mode=mode, values=forward + reverse),
        voc_search=None, implicit_legacy_protocol=implicit, schema_version=1)
    return _canonical(JVProtocolInput.model_validate(declaration).normalized_data())


def _compare(actual: Any, expected: Any, path: tuple[str | int, ...]) -> None:
    if isinstance(actual, dict) and isinstance(expected, dict):
        for key in expected:
            if key == "implicit_legacy_protocol":
                continue
            _compare(actual.get(key), expected[key], (*path, key))
    elif isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            _fail(path, "supplied protocol array length disagrees with the declared request", actual)
        for index, (left, right) in enumerate(zip(actual, expected)):
            _compare(left, right, (*path, index))
    elif actual != expected:
        _fail(path, "supplied protocol disagrees with the declared J-V history; no reordering or normalization repair is applied", actual)


def _resolve(experiment: JVInput | DarkJVInput, defaults: DefaultCatalog, device: dict[str, Any]) -> dict[str, Any]:
    section = "dark_jv" if isinstance(experiment, DarkJVInput) else "jv_endpoint" if experiment.request_api == "jv_endpoint" else "jv_jobs"
    values = defaults.experiment_defaults_for(section)
    origins = {key: f"default_catalog.experiment_defaults.{section}.{key}" for key in values}
    declared = experiment.normalized_data()
    for key in values:
        if key in declared:
            values[key], origins[key] = declared[key], f"input.experiment.{key}"
    is_jv = isinstance(experiment, JVInput)
    if section == "jv_endpoint":
        if "illuminated" in declared and declared["illuminated"] is not True:
            _fail(("experiment", "illuminated"), "the existing /api/jv endpoint fixes illuminated=True; use the jobs request contract for a dark J-V history", declared["illuminated"])
        values["illuminated"], origins["illuminated"] = True, "existing_run_jv_endpoint_fixed_illumination"
    if not is_jv:
        values.update(illuminated=False, solver="transient", protocol_mode="compatibility")
        origins.update(illuminated="existing_dark_jv_driver", solver="existing_dark_jv_driver", protocol_mode="existing_dark_jv_driver")
    waveform = experiment.waveform.normalized_data() if is_jv and experiment.waveform is not None else None
    supplied = experiment.experiment_protocol if is_jv else None
    if is_jv and experiment.waveform_controls is not None and waveform is None:
        _fail(("experiment", "waveform_controls"), "waveform_controls requires an explicit waveform", experiment.waveform_controls.editing_data())
    if waveform is not None:
        for key, required in (("solver", "transient"), ("iface_states", False), ("interface_boundary", False)):
            if values[key] != required:
                _fail(("experiment", key), "continuous waveform requires the transient driver without algebraic interface options", values[key])
        maximum = values["V_max"]
        if not isinstance(maximum, (int, float)) or maximum <= 0:
            _fail(("experiment", "V_max"), "continuous waveform requires an explicit positive V_max", values["V_max"])
        if waveform["uniform_generation_rate_m3_s"] is not None and not any(layer["role"] == "absorber" for layer in device["layers"]):
            _fail(("experiment", "waveform", "uniform_generation_rate_m3_s"), "uniform generation requires a supplied absorber layer", waveform["uniform_generation_rate_m3_s"])
    if device["settings"]["interface_charge_closure"] != "off":
        _fail(("device", "settings", "interface_charge_closure"), "charged-interface J-V requires its dedicated protocol and qualified binding, outside this preparation route", device["settings"]["interface_charge_closure"])
    if device["settings"]["jv_solver_policy"] == "cancellation_safe_qf_required" and values["solver"] != "quasi_fermi":
        _fail(("experiment", "solver"), "the supplied device requires the quasi_fermi driver; no substitution is made", values["solver"])
    if is_jv and waveform is None:
        if values["interface_boundary"] and values["solver"] != "quasi_fermi":
            _fail(("experiment", "interface_boundary"), "interface_boundary requires quasi_fermi", True)
        if not values["interface_boundary"] and values["interface_transport_model"] != "fermi_richardson":
            _fail(("experiment", "interface_transport_model"), "interface transport choice requires interface_boundary=True", values["interface_transport_model"])
    transient = values["solver"] == "transient"
    if not transient and (supplied is not None or values["protocol_mode"] != "compatibility"):
        _fail(("experiment", "experiment_protocol"), "this physical-history protocol belongs to the transient branch, not a steady-state curve", None if supplied is None else supplied.editing_data())
    if waveform is not None:
        numeric = experiment.waveform_controls.normalized_data() if is_jv and experiment.waveform_controls is not None else defaults.experiment_defaults_for("jv_waveform_controls")
        numeric_origin = "input.experiment.waveform_controls" if is_jv and experiment.waveform_controls is not None else "default_catalog.experiment_defaults.jv_waveform_controls"
    elif transient:
        key = "jv_numerical" if is_jv else "dark_jv_numerical"
        numeric, numeric_origin = defaults.experiment_defaults_for(key), f"default_catalog.experiment_defaults.{key}"
    else:
        numeric, numeric_origin = None, "separate_steady_driver_controls_not_prepared"
    soak = defaults.experiment_defaults_for("jv_history")["illuminated_soak_s"]
    assert isinstance(soak, (int, float))  # The catalog validates JVHistoryDefaults.
    expected = _expected_protocol(values, device["settings"]["T"], waveform, soak) if transient else None
    protocol = _canonical(supplied.normalized_data()) if supplied is not None else expected
    if supplied is not None and expected is not None:
        _compare(protocol, expected, ("experiment", "experiment_protocol"))
    if values["protocol_mode"] == "research_strict" and (protocol is None or protocol["implicit_legacy_protocol"]):
        _fail(("experiment", "experiment_protocol"), "research_strict requires an explicit history; no state is created by declaration", None if supplied is None else supplied.editing_data())
    return dict(kind=experiment.kind, request_api=section, controls=values, field_origins=origins,
        waveform=waveform, numerical_controls=numeric, numerical_controls_origin=numeric_origin,
        history_kind="continuous_forward_reverse" if waveform is not None else "staircase_forward_reverse" if transient else "single_steady_curve",
        state_history=dict(required=transient, generated=False, continuation="one_state_through_forward_turnaround_reverse" if transient else "separate_steady_driver",
                           initial_state="not_created", reset_between_branches=False if transient else None),
        protocol=dict(status="declared_history" if expected is not None else "pending_operating_contact_potential" if transient else "separate_steady_driver_declaration",
            declaration=protocol, declaration_sha256=hashlib.sha256(_bytes(protocol)).hexdigest() if protocol is not None else None,
            request_history_compared=supplied is not None and expected is not None,
            execution_bound=False, executor_id=None),
        mesh=dict(requested_N_grid=values["N_grid"], actual_nodes=None, generated=False, retained_electrical_grid=device["electrical_grid"]),
        legacy_result_scope="forward_curve_and_diode_fit_only_from_transient_dark_sweep" if not is_jv else "forward_reverse_history" if transient else "forward_equals_reverse_DC_wrapper",
        pending=["qualified_material_and_initial_state_binding", "actual_mesh_and_numerical_admission", "execution_identity_and_executor_unavailable"],
        compatibility=["jv_jobs_and_jv_endpoint_defaults_are_distinct", "dark_jv_does_not_accept_waveforms_or_illumination_overrides",
                       "null_or_omitted_jv_V_max_requires_operating_contact_potential_not_evaluated_here",
                       "waveform_complete_input_defines_history_even_without_a_duplicate_experiment_protocol",
                       "unknown_server_fields_reject_instead_of_legacy_params_silent_ignore",
                       "declarations_do_not_reproduce_historical_currents_or_hysteresis_metrics"])


@dataclass(frozen=True, slots=True)
class PreparedJVExperiment:
    input_json: bytes
    defaults: DefaultCatalog
    resources: ResourceLibrary
    sources: tuple[SourceDocument, ...] = ()
    can_execute: bool = field(default=False, init=False)
    _resolved_json: bytes = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.input_json) is not bytes or not self.input_json or len(self.input_json) > 2**23:
            raise ValueError("J-V declaration requires bounded immutable JSON bytes")
        model = JVExperimentInput.model_validate(_document(self.input_json))
        sources = tuple(self.sources)
        device = resolve_device(model.device, self.defaults, self.resources, sources=sources)
        data = device.to_mapping()
        result = dict(schema="solarlab.resolved-jv-experiment-preparation.v1", id=model.id,
            status="prepared_pending_dependencies", can_execute=False, device=data,
            experiment=_resolve(model.experiment, self.defaults, data),
            identity=dict(scope="experiment_declaration_content", execution_identity=None,
                device_content_sha256=device.content_sha256, default_catalog_sha256=self.defaults.content_sha256,
                resource_library_sha256=self.resources.content_sha256, default_source_bindings=[list(pair) for pair in self.defaults.evidence],
                input_source_bindings=[dict(id=source.id, sha256=source.sha256) for source in sources]),
            capability_gaps=[*data["capability_gaps"], "one_dimensional_experiment_execution_binding_not_prepared",
                             "initial_state_history_not_generated", "qualified_executor_not_registered"])
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "input_json", _bytes(model.model_dump(mode="json", exclude_unset=True)))
        object.__setattr__(self, "_resolved_json", _bytes(result))

    def to_input(self) -> JVExperimentInput:
        return JVExperimentInput.model_validate(_document(self.input_json))

    def to_mapping(self) -> dict[str, Any]:
        return json.loads(self._resolved_json)

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(_bytes([_document(self.input_json), self.to_mapping()])).hexdigest()


def prepare_jv_experiment(input: JVExperimentInput, defaults: DefaultCatalog, resources: ResourceLibrary,
                          *, sources: tuple[SourceDocument, ...] = ()) -> PreparedJVExperiment:
    model = JVExperimentInput.model_validate(input)
    return PreparedJVExperiment(_bytes(model.model_dump(mode="json", exclude_unset=True)), defaults, resources, sources)
