"""Historical preparation inspection using real config, selections and captures.

The optional reviewed capture archive is supplied explicitly by the engineering
check. Repository-only checks still exercise the real SCAPS and saved Calado
inputs; they cannot claim the archive-specific acceptance when it is absent.
No legacy import, material builder, numerical solve or trajectory is executed.
"""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import base64
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

from solarlab.config.behavior import (
    CapturedPreparation, _verify_words, behavior_source_requirements, inspect_historical_behavior,
)
from solarlab.config.device_import import import_standard_device
from solarlab.config.legacy_defaults import read_legacy_default_catalog
from solarlab.config.model_parameters import device_topology, layer_parameter_source
from solarlab.config.resolve_device import resolve_device
from solarlab.config.scaps_input import import_scaps_device
from solarlab.device.inputs import DeviceInput
from solarlab.materials.full_parameters import FullParameterInput
from solarlab.materials.resources import ResourceLibrary, ResourceTable
from solarlab.materials.source import SourceDocument
from solarlab.physics.registry import CapabilityContext, CapabilityRule, EvidenceRef, ModelDefinition, ModelRef, ModelRegistry, Scope
from solarlab.physics.selection import ModelChoice, select_models

ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT / "perovskite-sim/perovskite_sim"
CATALOG_PATHS = ("models/parameters.py", "models/device.py", "models/defects.py", "physics/statistics.py",
                 "physics/band_gap_narrowing.py", "scaps_compat/loader.py", "constants.py",
                 "twod/microstructure.py", "models/tandem_config.py")


def document(name, data):
    return SourceDocument(name, json.dumps(data, allow_nan=False).encode())


@pytest.fixture(scope="module")
def sources():
    return tuple(SourceDocument(name, (LEGACY / name).read_bytes()) for name, _ in behavior_source_requirements())


@pytest.fixture(scope="module")
def defaults(sources):
    by_id = {source.id: source for source in sources}
    return read_legacy_default_catalog(tuple(by_id[name] for name in CATALOG_PATHS))


@pytest.fixture(scope="module")
def resources():
    return ResourceLibrary(tuple(ResourceTable(path.stem, "nk", SourceDocument(str(path.relative_to(ROOT)), path.read_bytes()))
                                 for path in sorted((LEGACY / "data/nk").glob("*.csv"))))


@pytest.fixture(scope="module")
def scaps(defaults, resources):
    path = ROOT / "perovskite-sim/configs/scaps_mirror_v2.yaml"
    source = SourceDocument(str(path.relative_to(ROOT)), path.read_bytes())
    data = import_scaps_device(source, id="behavior_scaps", defaults=defaults)
    return resolve_device(data, defaults, resources, sources=(source,))


@pytest.fixture(scope="module")
def ion(defaults, resources):
    path = ROOT / "perovskite-sim/tests/fixtures/IonDiffusivityFigureReference.json"
    original = SourceDocument(str(path.relative_to(ROOT)), path.read_bytes())
    request = json.loads(original.content)["request"]
    source = document(original.id + "#request.device", request["device"])
    data = import_standard_device(source, id="behavior_ion", defaults=defaults)
    prepared = resolve_device(data, defaults, resources, sources=(original, source))
    return prepared, document(original.id + "#request.params", request["params"])


def selection(device, problem="transient"):
    # Real source-owned mobility declaration, as in the independently accepted
    # identity-preparation fixture. No fake executor is registered or invoked.
    source = SourceDocument("physics/temperature.py", (LEGACY / "physics/temperature.py").read_bytes())
    model = ModelRef("transport.temperature_mobility", 1)
    rule = CapabilityRule("source_data_only", (1,), ("dc", "transient"),
                          (EvidenceRef(source.id + "#mu_at_T", source.sha256, "source"),), backends=("metadata_only",))
    definition = ModelDefinition(model, "Temperature-dependent mobility", "transport", ("layer",), "node",
                                 ("mu_n", "mu_p", "mu_T_gamma"), (), (), rule, parameter_schema="full_layer")
    registry = ModelRegistry((definition,))
    return select_models(registry, device_topology(device),
                         CapabilityContext(1, problem, "metadata_only", "maxwell_boltzmann", temperature_K=device.values["settings"]["T"]),
                         {"layer": (ModelChoice("mobility", model, parameter_schema="full_layer"),)},
                         {Scope("layer", (layer.id,)): layer_parameter_source(layer)
                          for layer in device.layers if layer.role != "substrate"})


def context(device, sources, *, driver="scaps_transient", arguments=None, environment=(), capture=None):
    if arguments is None:
        arguments = {"N_grid": 30, "iface_states": True} if driver == "scaps_steady_state" else {
            "N_grid": 30, "n_points": 40, "v_rate": 5.0, "V_max": 1.6}
    source = arguments if isinstance(arguments, SourceDocument) else document("explicit_driver_call", arguments)
    return inspect_historical_behavior(device, selection(device, "dc" if driver == "scaps_steady_state" else "transient"),
                                       driver=driver, sources=sources, environment=environment, arguments=source, capture=capture)


def receipt(name, value):
    target = os.environ.get("SOLARLAB_BEHAVIOR_EXAMPLES")
    if target:
        with Path(target).open("a") as stream:
            stream.write(json.dumps({"case": name, "context_sha256": value.content_sha256, "context": value.export()}, allow_nan=False) + "\n")


def edit(device, change):
    data = deepcopy(device.to_input().editing_data())
    change(data)
    return resolve_device(DeviceInput.model_validate(data), device.defaults, device.resources, sources=device.sources)


def test_actual_scaps_driver_preparation_and_glass_semantics_are_distinct(scaps, sources):
    transient_context = context(scaps, sources)
    steady_context = context(scaps, sources, driver="scaps_steady_state")
    transient = transient_context.export()
    steady = steady_context.export()
    a, b = transient["source_inspection"], steady["source_inspection"]
    assert a["te"]["softness"] == 0.0 and b["te"]["softness"] == 0.02
    assert [row["calibration_if_plane_exists"] for row in a["interfaces"]] == [1.0, 1.0]
    assert [row["calibration_if_plane_exists"] for row in b["interfaces"]] == [0.02, 0.1]
    assert not a["interface_state_requested"] and b["interface_state_requested"]
    assert b["interface_state_count"]["status"] == "unobserved"
    glass = a["optics"]["incoherence"][0]
    assert glass["normalized_configured"] is True and glass["historical_loader_value"] is False
    assert a["preparation"]["preconditioner"]["t_settle"] == 0.001
    assert a["preparation"]["preconditioner"]["mat"] is None
    assert not a["preparation"]["solver_executed"]
    assert transient["selection_sha256"] != steady["selection_sha256"]  # transient vs dc metadata context
    assert transient["status"] == "prepared_pending_dependencies" and transient["can_execute"] is False
    receipt("scaps_transient_source_inspection", transient_context)
    receipt("scaps_steady_source_inspection", steady_context)


def test_explicit_environment_zero_does_not_turn_off_a_true_default_and_legacy_has_exact_ceiling(scaps, sources):
    original = context(scaps, sources)
    before = original.content_sha256
    candidate = edit(scaps, lambda d: d["settings"].update(dos_band_potentials=False))
    unset = context(candidate, sources).export()
    zero = context(candidate, sources, environment=(("SOLARLAB_DOS_BAND", "0"),)).export()
    one = context(candidate, sources, environment=(("SOLARLAB_DOS_BAND", "1"),)).export()
    assert unset["source_inspection"]["switches"]["dos_band_potentials"] is False
    assert zero["source_inspection"]["switches"]["dos_band_potentials"] is False
    assert one["source_inspection"]["switches"]["dos_band_potentials"] is True
    assert unset["environment_supplied"] == {} and zero["environment_supplied"] == {"SOLARLAB_DOS_BAND": "0"}
    enabled = edit(candidate, lambda d: d["settings"].update(dos_band_potentials=True))
    assert context(enabled, sources, environment=(("SOLARLAB_DOS_BAND", "0"),)).export()["source_inspection"]["switches"]["dos_band_potentials"] is True
    legacy = edit(candidate, lambda d: d["settings"].update(mode="legacy", interface_plane_projection=True))
    old = context(legacy, sources, environment=(("SOLARLAB_DOS_BAND", "1"), ("SOLARLAB_ION_STERIC_DIFF", "1"))).export()["source_inspection"]
    assert old["switches"]["dos_band_potentials"] is False
    assert old["ion_flux_label"] == "whole_flux"
    assert old["switches"]["interface_plane_projection"] is True  # no blanket mode suppression
    assert original.content_sha256 == before


def test_omitted_false_null_and_zero_remain_separate_with_provenance(scaps, sources):
    omitted = edit(scaps, lambda d: d["settings"].pop("te_physical_norm", None))
    explicit = edit(omitted, lambda d: d["settings"].update(te_physical_norm=False, built_in_potential_mode=None))
    a, b = context(omitted, sources).export(), context(explicit, sources).export()
    assert "te_physical_norm" not in a["normalized_supplied"]["settings"]
    assert b["normalized_supplied"]["settings"]["te_physical_norm"] is False
    assert b["normalized_supplied"]["settings"]["built_in_potential_mode"] is None
    assert b["input_sources"] == a["input_sources"] and b["default_catalog"] == a["default_catalog"]
    assert b["layer_parameter_origins"] and b["sources"]
    cleared = edit(scaps, lambda d: d["contacts"][0].update(S_n=None))
    blocking = edit(scaps, lambda d: d["contacts"][0].update(S_n=0))
    absent = edit(scaps, lambda d: d["contacts"][0].pop("S_n", None))
    # This actual SCAPS input selects FAST, which suppresses selective contacts.
    gated = context(blocking, sources).export()["source_inspection"]["contacts"][0]
    assert gated["configured_S_m_s"] == 0 and gated["effective_S_m_s"] is None
    cleared, blocking, absent = (edit(v, lambda d: d["settings"].update(mode="full")) for v in (cleared, blocking, absent))
    c, z, m = (context(v, sources).export() for v in (cleared, blocking, absent))
    assert c["source_inspection"]["contacts"][0]["condition"] == "density_dirichlet"
    assert z["source_inspection"]["contacts"][0]["condition"] == "blocking_robin"
    assert z["source_inspection"]["contacts"][0]["effective_S_m_s"] == 0.0
    assert "S_n" not in m["normalized_supplied"]["contacts"][0]
    assert c["normalized_supplied"]["contacts"][0]["S_n"] is None


def test_flat_contacts_override_mode_without_erasing_signed_electrostatic_uncertainty(scaps, sources):
    candidate = edit(scaps, lambda d: d["settings"].update(mode="legacy", flat_band_contacts=True, V_bi=-1.3))
    body = context(candidate, sources).export()["source_inspection"]
    assert all(row["effective_S_m_s"] == 1e5 for row in body["contacts"])
    assert body["electrostatics"]["configured_V_bi"] == -1.3
    assert body["electrostatics"]["signed_V_bi_bc"] == {"status": "unobserved"}
    assert "junction_polarity" in body["electrostatics"]["rule"]


def test_import_time_flags_preserve_exact_strings_and_do_not_leak(scaps, sources):
    a = context(scaps, sources, environment=(("SOLARLAB_SS_JAC_REUSE", "0"), ("PEROVSKITE_RHS_FINITE_CHECK", "1")))
    b = context(scaps, sources, environment=(("SOLARLAB_SS_JAC_REUSE", "1"), ("PEROVSKITE_RHS_FINITE_CHECK", "0")))
    assert a.export()["source_inspection"]["import_time_policy"]["ss_jacobian_reuse"] is False
    assert b.export()["source_inspection"]["import_time_policy"]["ss_jacobian_reuse"] is True
    assert a.export()["source_inspection"]["import_time_policy"]["rhs_finite_check"] is True
    assert b.export()["source_inspection"]["import_time_policy"]["rhs_finite_check"] is False
    with pytest.raises(ValueError, match="non-string"):
        context(scaps, sources, environment=(("SOLARLAB_DOS_BAND", False),))
    with pytest.raises(ValueError, match="unsupported environment"):
        context(scaps, sources, environment=(("UNREVIEWED_PHYSICS_OVERRIDE", "1"),))


def test_original_ion_request_preserves_protocol_and_distinguishes_frozen_from_absent_inventory(ion, sources):
    prepared, args = ion
    original_context = context(prepared, sources, driver="ion_hysteresis", arguments=args)
    base = original_context.export()
    observed = base["source_inspection"]
    assert observed["optics"]["generation"] == "uniform_absorber_override"
    assert observed["optics"]["uniform_generation_m3_s"] == 2.5e27
    # The saved request omits this flag. The actually supplied current source
    # catalog defaults to diffusion-only; it cannot prove the old run's flags.
    assert observed["ion_flux_label"] == "diffusion_only"
    explicit_whole = edit(prepared, lambda d: d["settings"].update(ion_steric_diffusion_only=False))
    assert context(explicit_whole, sources, driver="ion_hysteresis", arguments=args).export()["source_inspection"]["ion_flux_label"] == "whole_flux"
    call = observed["preparation"]["actual_driver_arguments"]
    assert (call["N_grid"], call["n_points"], call["v_rate"], call["V_max"], call["atol"]) == (60, 111, 0.04, 1.2, 1)
    assert call["waveform"]["dark_seed_s"] == 120 and call["waveform"]["dark_prep_s"] == 30
    assert call["waveform"]["turnaround_s"] == 3 and call["waveform"]["branch_dwell_s"] == 0.5
    assert not observed["preparation"]["converged_state_observed"]
    frozen = edit(prepared, lambda d: d["layers"][1]["parameters"].update(D_ion=0.0))
    empty = edit(prepared, lambda d: d["layers"][1]["parameters"].update(P0=0.0))
    def absorber(device):
        rows = context(device, sources, driver="ion_hysteresis", arguments=args).export()["source_inspection"]["ions"]
        return next(row for row in rows if row["layer"] == "layer_1" and row["species"] == "positive")
    f, e = absorber(frozen), absorber(empty)
    assert f["seed_population_and_fixed_background_m3"] == 1e25
    assert f["diffusion_status"] == "frozen_zero_diffusion" and f["inventory_status"] == "population_and_equal_opposite_background_retained"
    assert e["seed_population_and_fixed_background_m3"] == 0 and e["D_reference_m2_s"] == 2.585e-18
    assert e["inventory_status"] == "zero_population_and_background"
    # P0 is required by the real source catalog; absence must not invent zero
    # inventory (or an arbitrary default population).
    with pytest.raises(ValueError, match="missing full parameter values.*P0"):
        edit(prepared, lambda d: d["layers"][1]["parameters"].pop("P0"))
    assert len(base["input_sources"]) == 2  # original saved request and derived device fragment
    receipt("original_ion_request_source_inspection", original_context)


def test_interface_calibration_changes_remain_input_scoped(scaps, sources):
    zero = edit(scaps, lambda d: d["interfaces"][1]["defect"].update(iface_state_calibration_factor=0.0))
    unity = edit(scaps, lambda d: d["interfaces"][1]["defect"].update(iface_state_calibration_factor=1.0))
    a = context(zero, sources, driver="scaps_steady_state").export()["source_inspection"]["interfaces"]
    b = context(unity, sources, driver="scaps_steady_state").export()["source_inspection"]["interfaces"]
    assert a[0]["calibration_if_plane_exists"] == 0.0 and b[0]["calibration_if_plane_exists"] == 1.0
    assert context(scaps, sources, driver="scaps_steady_state").export()["source_inspection"]["interfaces"][0]["calibration_if_plane_exists"] == 0.02


def test_original_zero_ct_exponents_import_and_roundtrip_without_coercing_omission_or_invalid_values(ion):
    prepared, _ = ion
    original = json.loads(prepared.sources[0].content)["request"]
    assert hashlib.sha256(json.dumps(original, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == "f85b35f9926a85b31bb8c865c91d1ab093abdca5dbcf06a906b940aa0ee13d6d"
    for layer in prepared.layers:
        values = dict(layer.parameters)
        assert values["ct_beta_n"] == values["ct_beta_p"] == values["v_sat_n"] == values["v_sat_p"] == 0.0
    same = resolve_device(prepared.to_input(), prepared.defaults, prepared.resources, sources=prepared.sources)
    assert same.content_sha256 == prepared.content_sha256
    assert "ct_beta_n" not in FullParameterInput().editing_data()
    assert FullParameterInput(ct_beta_n=0, ct_beta_p=0).editing_data() == {"ct_beta_n": 0, "ct_beta_p": 0}
    for invalid in (-1, float("inf"), float("nan"), True, None):
        with pytest.raises(ValueError):
            FullParameterInput(ct_beta_n=invalid)


def test_context_is_deeply_independent_and_unsupported_calls_fail_closed(scaps, sources, ion):
    original = context(scaps, sources)
    with pytest.raises(FrozenInstanceError):
        original.driver = "ion_hysteresis"
    with pytest.raises(FrozenInstanceError):
        original.sources[0].content = b"changed"
    detached = original.export()
    detached["source_inspection"]["te"]["softness"] = 55
    assert original.export()["source_inspection"]["te"]["softness"] == 0
    with pytest.raises(ValueError, match="exact audited"):
        context(scaps, (replace(sources[0], content=sources[0].content + b"\n"), *sources[1:]))
    with pytest.raises(ValueError, match="unknown driver"):
        context(scaps, sources, arguments={"magic_physics": True})
    with pytest.raises(ValueError, match="positive integer"):
        context(scaps, sources, arguments={"N_grid": False})
    with pytest.raises(ValueError, match="boolean"):
        context(scaps, sources, driver="scaps_steady_state", arguments={"iface_states": 0})
    with pytest.raises(ValueError, match="outside this inspection"):
        context(scaps, sources, arguments={"rhs_regularization": {"te_cap_relative_width": 0.0}})
    with pytest.raises(ValueError, match="explicit saved waveform"):
        context(ion[0], sources, driver="ion_hysteresis", arguments={})
    with pytest.raises(ValueError, match="topology"):
        inspect_historical_behavior(scaps, selection(ion[0]), driver="scaps_transient", sources=sources,
                                    environment=(), arguments=document("call", {}))
    for value in ({"N_grid": None}, {"N_grid": 0}, {"V_max": False}):
        with pytest.raises(ValueError):
            context(scaps, sources, arguments=value)


def test_driver_argument_null_and_omission_and_signed_zero_are_not_conflated(scaps, sources):
    omitted = context(scaps, sources, arguments={}).export()
    cleared = context(scaps, sources, arguments={"V_max": None}).export()
    assert "V_max" not in omitted["arguments_supplied"] and cleared["arguments_supplied"]["V_max"] is None
    signed = document("call_with_exact_zero", {"V_lo": -0.0})
    stored = context(scaps, sources, driver="scaps_steady_state", arguments=signed)
    assert stored.arguments.content == signed.content
    assert stored.export()["arguments_supplied"]["V_lo"].hex() == "-0x0.0p+0"


@pytest.fixture(scope="module")
def archive():
    name = os.environ.get("SOLARLAB_BEHAVIOR_CAPTURE_ROOT")
    if not name:
        pytest.skip("explicit independently reviewed SCAPS archive not supplied; no capture acceptance")
    root = Path(name)
    assert root.is_absolute() and (root / "ConditioningCapture02/case.yaml").is_file()
    return root


def captured(archive, folder):
    def read(name):
        return SourceDocument(name, (archive / name).read_bytes())
    return CapturedPreparation(read(folder + "/Result.json"), read(folder + "/Freeze.json"), read("ConditioningCapture02/case.yaml"))


def captured_device(record, defaults, resources):
    data = import_scaps_device(record.case, id="captured_scaps", defaults=defaults)
    return resolve_device(data, defaults, resources, sources=(record.case,))


@pytest.mark.parametrize("driver", ["scaps_transient", "scaps_steady_state"])
def test_real_reviewed_outer_driver_capture_words_survive(archive, defaults, resources, sources, driver):
    record = captured(archive, "DriverCapture01")
    device = captured_device(record, defaults, resources)
    captured_context = context(device, sources, driver=driver, capture=record)
    result = captured_context.export()
    data = result["captured_preparation"]["payload"]["record"]
    material = data["boundary_calls"][0]["kwargs"].get("mat", data["preparation"][0]["material"])["fields"]
    assert material["N_iface_state"] == (2 if driver == "scaps_steady_state" else 0)
    assert material["iface_state_shared_occ"] is (driver == "scaps_steady_state")
    assert float.fromhex(material["te_softness"]["binary64_hex"]) == (0.02 if driver == "scaps_steady_state" else 0.0)
    assert float.fromhex(material["V_bi_eff"]["binary64_hex"]) == 1.2939750419068696
    assert float.fromhex(material["V_bi_bc"]["binary64_hex"]) == 1.3
    assert all(material[key] is None for key in ("S_n_L", "S_p_L", "S_n_R", "S_p_R"))
    array = material["P_ion0"]
    raw = base64.b64decode(array["bytes_b64"])
    assert array["shape"] == [31] and array["dtype"] == "<f8"
    assert raw == bytes(31 * 8) and hashlib.sha256(raw).hexdigest() == array["sha256"]
    assert record.result.sha256 == result["captured_preparation"]["result"]["sha256"]
    receipt(driver + "_outer_capture", captured_context)


def test_actual_conditioning_retains_three_materials_and_analytic_seed_without_convergence(archive, defaults, resources, sources):
    record = captured(archive, "ConditioningCapture02")
    device = captured_device(record, defaults, resources)
    context_value = context(device, sources, capture=record)
    result = context_value.export()["captured_preparation"]["payload"]
    objects = result["objects"]
    assert len({objects[name]["process_local_id"] for name in ("outer_material", "inner_material", "seed_material")}) == 3
    assert result["boundary_identity_checks"]["y0_is_actual_seed"] is True
    assert result["boundary_identity_checks"]["mat_is_inner"] is True
    assert result["boundary_identity_checks"]["mat_is_outer"] is False
    assert result["seed_has_independent_phi_block"] is False
    assert result["boundary_call"]["effective_arguments"]["max_step"] == {"binary64_hex": "inf"}
    schema = "solarlab.scaps-conditioning-capture.v1"
    for word in ("nan", "-inf", "inf"):
        with pytest.raises(ValueError, match="outside the reviewed"):
            _verify_words({"material_parameter": {"binary64_hex": word}}, schema=schema)
    for word in ("nan", "-inf"):
        with pytest.raises(ValueError, match="outside the reviewed"):
            _verify_words({"boundary_call": {"effective_arguments": {"max_step": {"binary64_hex": word}}}}, schema=schema)
    with pytest.raises(ValueError, match="outside the reviewed"):
        _verify_words({"boundary_call": {"effective_arguments": {"max_step": {"binary64_hex": "inf"}}}}, schema="solarlab.scaps-driver-preparation.v1")
    assert [item["slice_half_open"] for item in result["seed_layout"]] == [[0, 31], [31, 62], [62, 93]]
    assert result["scope"]["ODE_steps"] == result["scope"]["Newton_steps"] == result["scope"]["Poisson_state_solves"] == 0
    assert result["scope"]["finite_time_conditioned_state_observed"] is False
    raw = base64.b64decode(objects["seed"]["value"]["bytes_b64"])
    assert hashlib.sha256(raw).hexdigest() == objects["seed"]["value"]["sha256"]
    receipt("transient_inner_and_seed_capture", context_value)
    with pytest.raises(ValueError, match="independently reviewed"):
        replace(record, result=replace(record.result, content=record.result.content + b"\n"))
    with pytest.raises(ValueError, match="captured input"):
        context(edit(device, lambda d: d["layers"][2]["parameters"].update(D_ion=0.1)), sources, capture=record)
    with pytest.raises(ValueError, match="driver arguments"):
        context(device, sources, arguments={"N_grid": 31}, capture=record)
    with pytest.raises(ValueError, match="effective gates conflict"):
        context(device, sources, environment=(("SOLARLAB_TE_PHYSICAL", "1"),), capture=record)
    with pytest.raises(ValueError, match="only to the transient"):
        context(device, sources, driver="scaps_steady_state", capture=record)


def test_import_boundary_has_no_legacy_or_scientific_solver():
    forbidden = {"perovskite_sim", "scipy", "sksundae", "flint", "backend", "solarlab_server", "solarlab_research"}
    assert not {name.split(".")[0] for name in sys.modules}.intersection(forbidden)
