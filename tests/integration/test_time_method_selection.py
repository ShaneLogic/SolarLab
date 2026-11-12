"""The time-method axis at the transient production entry points (CND2).

The ten allowlisted envelopes are declared as ReadinessV1 declared them, on
this checkout's research device, from cold steady initials.
CommanderBdfDefaultFreezeV1 freezes their source words and controls, and
allowlist.json holds them with each envelope's CND2 scope key. A key digests
numerical-stack words: the compiled device content and the initial state.
Only the tests that compare live keys or histories with frozen ones therefore
need the capture stack. Elsewhere the same envelopes are allowlisted under
this platform's own keys.
"""

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import date
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from examples.research_1d.ionic_residual import prepare_single_ion_device
from examples.research_1d.ionic_steady import controls as ionic_steady_controls
from examples.research_1d.ionic_transient import controls as ionic_controls, json_value
from examples.research_1d.steady import load_device
from examples.research_1d.transient import controls
from solarlab.device.compile_1d import compile_device_1d
from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
from solarlab.equations import time_method
from solarlab.equations.implementation import ASSEMBLED, LEGACY, equation_implementation
from solarlab.equations.time_method import (
    ADAPTIVE_MAX_STEPS, ALLOWLIST, BACKWARD_EULER, BDF, ENVIRONMENT, PROFILE, PROFILE_SHA256, AdmittedBDFCall,
    electronic_components, electronic_scope, time_method as time_method_scope,
)
from solarlab.experiments import bdf as bdf_backend, bdf_results, transient
from solarlab.experiments.bdf_results import _decode, _digest, _loads, _validate_body, read_bdf_result
from solarlab.experiments.ionic_transient import load_ionic_steady_initial, solve_ionic_transient
from solarlab.experiments.transient import TransientOptions, TransientProtocol, TransientResult1D, solve_transient
from solarlab.experiments.transient_results import (
    publish_transient_result, read_transient_result, run_transient_for_publication,
)
from solarlab.io.run_store import RunStore
from solarlab.solvers.nonlinear.ionic_1d import solve_ionic_steady_state
from solarlab.solvers.nonlinear.problem import SolveBudget
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions, solve_steady_state
from solarlab.solvers.time.bdf import BDFOptions, BDFResult
from solarlab.solvers.time.bdf_1d import BDFDrive
from support.equation_provenance import counted

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/integration/fixtures/bdf_readiness"
ALLOWLIST_FIXTURE = FIXTURES / "allowlist.json"
SAVED_PACKET = FIXTURES / "readinessv2_e_saved_packet.json.gz"


@pytest.fixture(scope="module", autouse=True)
def blas1():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(autouse=True)
def no_process_settings(monkeypatch):
    for name in (ENVIRONMENT, "NEWCORE_EQUATIONS"):
        monkeypatch.delenv(name, raising=False)


def frozen_table():
    return json.loads(ALLOWLIST_FIXTURE.read_text())


def sources():
    """The ten envelopes' source knots, as ReadinessV1 declared them."""
    t = np.linspace(0, 4e-7, 5)
    declared = {"E_dark_hold": (t, 0.0, 0.0), "E_light_hold": (t, 0.0, 1.0),
                "E_dark_voltage_ramp": (t, .002 * t / t[-1], 0.0)}
    for n in (4, 8, 16):
        grid = np.linspace(0, 4e-7, n + 1)
        declared[f"E_light_ramp_{n}"] = (grid, 0.0, 1 + .01 * (grid / grid[-1]) ** 2)
        declared[f"E_voltage_ramp_{n}"] = (grid, .1 + .001 * (grid / grid[-1]) ** 2, 1.0)
    declared["E_saved_packet"] = (np.asarray([0, 1e-7, 2e-7], float), 0.0, [1, 1.0025, 1.01])
    return {name: TransientProtocol(times, *(np.broadcast_to(value, times.shape).astype(float).copy()
                                             for value in (voltage, light)))
            for name, (times, voltage, light) in declared.items()}


INITIALS = {}


def envelope(device, protocol):
    """One envelope's call: its cold steady initial, source knots and the research physical controls."""
    key = (device.identity, protocol.voltage_V[0], protocol.generation_scale[0])
    if key not in INITIALS:
        INITIALS[key] = solve_steady_state(device, voltage_V=float(protocol.voltage_V[0]),
                                           generation_scale=float(protocol.generation_scale[0]), options=SteadyOptions())
    return INITIALS[key], protocol, controls()


@pytest.fixture(scope="module")
def electronic37():
    """The research device, compiled from this checkout (its own path-bound identity)."""
    return compile_device_1d(load_device(), intervals_per_layer=12)


@pytest.fixture(scope="module")
def envelopes(electronic37):
    return {name: envelope(electronic37, protocol) for name, protocol in sources().items()}


@pytest.fixture
def unflipped(monkeypatch):
    """The scope-based default before the route decision: backward Euler for every scope."""
    monkeypatch.setattr(time_method, "ALLOWLISTED_DEFAULT", (BACKWARD_EULER, None))


@pytest.fixture
def allowlisted(monkeypatch, electronic37, envelopes):
    """This platform's own keys for the ten frozen envelopes; on the capture stack they are ALLOWLIST itself."""
    keys = {electronic_scope(electronic37, initial, protocol, options=options, initial_options=SteadyOptions()): name
            for name, (initial, protocol, options) in envelopes.items()}
    assert len(keys) == 10
    monkeypatch.setattr(time_method, "ALLOWLIST", keys)
    # Only tests that explicitly flip the default use this prospective date.
    monkeypatch.setattr(time_method, "_SWITCH_DATE", "2026-11-08")
    monkeypatch.setattr(time_method, "_route_today", lambda: date(2026, 11, 8))
    return {name: key for key, name in keys.items()}


def capture_guard():
    import importlib.util

    spec = importlib.util.spec_from_file_location("frozen_oracle", ROOT / "tests/unit/assembly_parity/test_frozen_baselines.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.require_capture_environment(frozen_table()["capture"])


def words(value):
    """Every result leaf as binary64 words; wall clocks dropped."""
    if isinstance(value, dict):
        return tuple((key, words(item)) for key, item in value.items()
                     if key not in {"wall_s", "initial_admission_wall_s", "time_method"})
    if hasattr(value, "__dataclass_fields__") and not isinstance(value, type):
        return (type(value).__name__, tuple((name, words(getattr(value, name))) for name in value.__dataclass_fields__
                                            if name != "wall_s"))
    if isinstance(value, (list, tuple)):
        return tuple(words(item) for item in value)
    if isinstance(value, np.ndarray):
        return (value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, float):
        return float(value).hex()
    return value


@contextmanager
def no_numerical_work(monkeypatch):
    """Refusals come before any solve: neither integrator nor its Newton may start."""
    def forbidden(*args, **kwargs):
        raise AssertionError("numerical work ran before the refusal")

    with monkeypatch.context() as patch:
        for module, name in ((transient, "solve_newton"), (bdf_backend, "solve_bdf"),
                             (bdf_backend, "solve_electronic_bdf"), (bdf_backend, "solve_ionic_bdf")):
            patch.setattr(module, name, forbidden)
        yield


@pytest.mark.platform_bound
def test_the_allowlist_holds_exactly_the_ten_frozen_envelopes():
    table = frozen_table()
    assert dict(ALLOWLIST) == {case["scope_sha256"]: case["case_id"] for case in table["envelopes"]}
    assert table["allowlist_sha256"] == time_method.ALLOWLIST_SHA256 and table["qualification"] is False
    assert table["profile"] == {"sha256": PROFILE_SHA256, "options": dict(PROFILE), "adaptive_max_steps": 10000}
    assert (hashlib.sha256((ROOT / "examples/research_1d/device.json").read_bytes()).hexdigest()
            == table["device"]["config_sha256"])
    # The declared sources and controls are the frozen ones, word for word.
    declared = sources()
    for case in table["envelopes"]:
        protocol = declared[case["case_id"]]
        for name, values in case["frozen_source"].items():
            assert np.asarray(values, dtype=float).tobytes() == getattr(protocol, name).tobytes(), case["case_id"]
        physical = case["physical_controls"]
        assert TransientOptions(**{**physical, "newton_budget": SolveBudget(**physical["newton_budget"])}) == controls()
        assert case["initial_controls"] == asdict(SteadyOptions())


def test_scope_keys_are_path_independent(tmp_path, envelopes):
    """Two absolute config paths compile different identities but the same scope key for every envelope."""
    devices = []
    for name in ("first", "second/deeper"):
        path = tmp_path / name / "device.json"
        path.parent.mkdir(parents=True)
        shutil.copyfile(ROOT / "examples/research_1d/device.json", path)
        devices.append(compile_device_1d(load_device(path), intervals_per_layer=12))
    assert devices[0].identity != devices[1].identity
    keys = set()
    for name, declared in sources().items():
        found = set()
        for device in devices:
            initial, protocol, options = envelope(device, declared)
            found.add(electronic_scope(device, initial, protocol, options=options, initial_options=SteadyOptions()))
        assert len(found) == 1, name
        keys |= found
    assert len(keys) == 10


def test_an_ionic_manifest_missing_any_path_bound_identity_has_no_scope(tmp_path):
    """G1 (Science recheck): Grammar V2 R1 refuses a binding manifest without each identity it sets aside.

    Removing the top-level original_prepared_sha256, electronic_component.compiled_sha256,
    electronic_component.prepared_sha256, or both nested ones, leaves no scope key. An explicit bdf request is then
    refused as an uncomputable scope, while the complete manifest keeps its key.
    """
    from solarlab.equations.time_method import ionic_scope

    device, initial, protocol = ionic_case(tmp_path)
    physical = ionic_controls()
    key = ionic_scope(device, initial, protocol, options=physical)
    for removed in (("original_prepared_sha256",), ("electronic_component.compiled_sha256",),
                    ("electronic_component.prepared_sha256",),
                    ("electronic_component.compiled_sha256", "electronic_component.prepared_sha256")):
        manifest = json.loads(device.binding_manifest_json)
        for name in removed:
            section, _, field = name.rpartition(".")
            (manifest[section] if section else manifest).pop(field)
        changed = replace(device, binding_manifest_json=json.dumps(manifest, sort_keys=True))
        with pytest.raises(TypeError, match="must name its path-bound identities"):
            ionic_scope(changed, initial, protocol, options=physical)
        with pytest.raises(ValueError, match="scope cannot be computed.*path-bound identities"):
            time_method.resolve_time_method(BDF, compiled=changed, initial=initial, protocol=protocol,
                                            options=physical, family=time_method.IONIC)
    assert ionic_scope(replace(device), initial, protocol, options=physical) == key


def test_ionic_scope_keys_are_path_independent(tmp_path, monkeypatch):
    """The ionic device carries three path-bound identities; its scope sets them aside too."""
    import examples.research_1d.ionic_residual as residual
    from solarlab.equations.time_method import ionic_scope

    original = residual.load_device
    found, identities = set(), set()
    for name in ("first", "second/deeper"):
        path = tmp_path / name / "device.json"
        path.parent.mkdir(parents=True)
        shutil.copyfile(ROOT / "examples/research_1d/device.json", path)
        monkeypatch.setattr(residual, "load_device", lambda selected=None, path=path: original(path))
        device, initial, protocol = ionic_case(tmp_path / name, device=compile_ionic_device_1d(
            residual.prepare_single_ion_device(diffusion_only=True), intervals_per_layer=12))
        identities.add(device.identity)
        found.add(ionic_scope(device, initial, protocol, options=ionic_controls()))
    assert len(identities) == 2 and len(found) == 1


@pytest.mark.platform_bound
def test_live_scope_keys_are_the_frozen_allowlist(electronic37, envelopes):
    """On the capture stack the live envelopes compute exactly the frozen keys and components."""
    capture_guard()
    for case in frozen_table()["envelopes"]:
        initial, protocol, options = envelopes[case["case_id"]]
        components = electronic_components(electronic37, initial, protocol, options=options,
                                           initial_options=SteadyOptions())
        assert {name: time_method._sha256(value) for name, value in components.items()} == case["components"]
        key = electronic_scope(electronic37, initial, protocol, options=options, initial_options=SteadyOptions())
        # The cold steady initial on today's code has the frozen artifact's words.
        assert ALLOWLIST[key] == case["case_id"] == ALLOWLIST[case["scope_sha256"]]


def test_the_default_is_bitwise_the_backward_euler_history_on_every_envelope(monkeypatch, electronic37, envelopes,
                                                                             allowlisted, unflipped):
    for name, (initial, protocol, options) in envelopes.items():
        call = dict(options=options, initial_options=SteadyOptions())
        default = solve_transient(electronic37, initial, protocol, **call)
        explicit = solve_transient(electronic37, initial, protocol, method=BACKWARD_EULER, **call)
        with time_method_scope(BACKWARD_EULER):
            scoped = solve_transient(electronic37, initial, protocol, **call)
        monkeypatch.setenv(ENVIRONMENT, BACKWARD_EULER)
        process = solve_transient(electronic37, initial, protocol, **call)
        monkeypatch.delenv(ENVIRONMENT)
        assert type(default) is TransientResult1D and default.completed, name
        assert words(default) == words(explicit) == words(scoped) == words(process), name


def assert_frozen_bdf(result, name, source):
    assert type(result) is BDFResult and result.completed, (name, result.termination, result.error)
    controls_ = result.controls
    assert {key: controls_[key] for key in PROFILE if key not in {"max_steps", "wall_s"}} == {
        key: value for key, value in PROFILE.items() if key not in {"max_steps", "wall_s"}}
    # The forwarded adaptive work cap and the separate sample-interval admission (RV2-C2).
    assert controls_["max_steps"] == controls_["adapter_adaptive_max_steps"] == ADAPTIVE_MAX_STEPS == 10000
    assert controls_["adapter_sample_interval_limit"] == 32 and controls_["max_attempts"] == 20000
    assert controls_["max_consecutive_rejections"] == 12 and 0 < controls_["wall_s"] <= 30.0
    record = controls_["time_method"]
    assert (record["requested"], record["source"], record["effective"], record["scope"]) == (
        None if source == "default" else BDF, source, BDF, name)
    assert record["profile"] == dict(PROFILE) and record["profile_sha256"] == PROFILE_SHA256
    assert record["adaptive_max_steps"] == 10000 and record["qualification"] is False


def test_explicit_bdf_runs_the_frozen_profile_with_the_forwarded_cap(monkeypatch, electronic37, envelopes,
                                                                    allowlisted):
    for name, (initial, protocol, options) in envelopes.items():
        call = dict(options=options, initial_options=SteadyOptions())
        argument = solve_transient(electronic37, initial, protocol, method=BDF, **call)
        assert_frozen_bdf(argument, name, "argument")
        if name in {"E_saved_packet", "E_dark_voltage_ramp"}:
            with time_method_scope(BDF):
                scoped = solve_transient(electronic37, initial, protocol, **call)
            monkeypatch.setenv(ENVIRONMENT, BDF)
            process = solve_transient(electronic37, initial, protocol, **call)
            monkeypatch.delenv(ENVIRONMENT)
            assert_frozen_bdf(scoped, name, "scope")
            assert_frozen_bdf(process, name, "environment")
            exact = solve_transient(electronic37, initial, protocol, method=BDF, bdf_options=BDFOptions(**PROFILE),
                                    bdf_drive=BDFDrive.from_protocol(protocol), **call)
            # BDF runs the assembled equations by default (A3) and the original ones on the LEGACY opt-out.
            with equation_implementation(LEGACY), counted() as original:
                legacy = solve_transient(electronic37, initial, protocol, method=BDF, **call)
            with equation_implementation(ASSEMBLED), counted() as assembled:
                forced = solve_transient(electronic37, initial, protocol, method=BDF, **call)
            assert original.original > 0 and original.assembled == 0, name
            assert assembled.original == 0 and assembled.assembled > 0, name
            assert words(argument) == words(scoped) == words(process) == words(exact) == words(legacy) == words(
                forced), name


@pytest.mark.platform_bound
def test_production_bdf_reproduces_the_reviewed_readiness_history(electronic37, envelopes):
    """E_saved_packet through solve_transient is the ReadinessV2 history (cap 10000), word for word."""
    capture_guard()
    packet = _decode(_loads(gzip.decompress(SAVED_PACKET.read_bytes())))
    initial, protocol, options = envelopes["E_saved_packet"]
    result = solve_transient(electronic37, initial, protocol, options=options, initial_options=SteadyOptions(),
                             method=BDF)
    history = _decode(_loads(json.dumps(bdf_results._encode(result)).encode()))
    saved = packet["result"]
    assert packet["BDF_options"] == dict(PROFILE)
    for name in ("accepted_times_s", "accepted_coordinates", "attempts", "samples", "restarts", "nonlinear_calls",
                 "evaluation_counts", "completed", "termination", "requested_output_times_s"):
        assert words(history[name]) == words(saved[name]), name
    path_bound = {"compiled_identity", "time_method", "initial_admission_wall_s", "wall_s"}
    assert ({key: value for key, value in history["controls"].items() if key not in path_bound | {"initial_controls"}}
            .keys() == {key for key in saved["controls"] if key not in path_bound | {"initial_controls"}})
    for key in saved["controls"]:
        if key not in path_bound:
            assert words(history["controls"][key]) == words(saved["controls"][key]), key


def ionic_case(tmp_path, device=None):
    if device is None:
        device = compile_ionic_device_1d(prepare_single_ion_device(diffusion_only=True), intervals_per_layer=12)
    admission = ionic_steady_controls(device.ion_node_count)
    point = solve_ionic_steady_state(device, voltage_V=.002, generation_scale=0.0, options=admission)
    raw = (json.dumps(json_value({"summary": {"accepted": True, "compiled_identity": device.identity,
        "voltage_V": .002, "generation_scale": 0.0}, "result": point}), allow_nan=False) + "\n").encode()
    path = tmp_path / "Steady.json"
    path.write_bytes(raw)
    initial = load_ionic_steady_initial(device, path, hashlib.sha256(raw).hexdigest(), controls=admission)
    times = np.linspace(0.0, 1.0, 5)
    return device, initial, TransientProtocol(times, .002 + .002 * times, np.zeros(5))


def test_unlisted_scopes_custom_profiles_and_assembled_bdf_are_refused(monkeypatch, tmp_path, electronic37,
                                                                       envelopes, allowlisted):
    initial, protocol, options = envelopes["E_light_ramp_4"]
    call = dict(options=options, initial_options=SteadyOptions())
    nudged = protocol.generation_scale.copy()
    nudged[-1] = np.nextafter(nudged[-1], 2.0)
    jump = BDFDrive(protocol.times_s, np.column_stack((protocol.voltage_V, protocol.generation_scale)),
                    np.column_stack((protocol.voltage_V, nudged)))
    other_device = compile_device_1d(load_device(), intervals_per_layer=8)
    other_initial = solve_steady_state(other_device, voltage_V=0.0, generation_scale=1.0, options=SteadyOptions())
    moved = initial.state.u.copy()
    moved[electronic37.node_count] = np.nextafter(moved[electronic37.node_count], np.inf)
    other_state = replace(initial, state=electronic37.state(moved, voltage_V=initial.voltage_V))
    unlisted = {
        "one-ulp source word": lambda: solve_transient(electronic37, initial, replace(protocol, generation_scale=nudged),
                                                       method=BDF, **call),
        "physical control": lambda: solve_transient(electronic37, initial, protocol, method=BDF,
            options=replace(options, max_steps=31), initial_options=SteadyOptions()),
        "initial control": lambda: solve_transient(electronic37, initial, protocol, method=BDF, options=options,
            initial_options=replace(SteadyOptions(), max_iterations=99)),
        "other device": lambda: solve_transient(other_device, other_initial, protocol, method=BDF, **call),
        "one-ulp initial word": lambda: solve_transient(electronic37, other_state, protocol, method=BDF, **call),
    }
    with no_numerical_work(monkeypatch):
        for reason, run in unlisted.items():
            with pytest.raises(ValueError, match="only the 10 allowlisted electronic envelopes"):
                run()
            with time_method_scope(BDF), pytest.raises(ValueError, match="only the 10 allowlisted"):
                run()
        for custom in (BDFOptions(1e-7), replace(BDFOptions(**PROFILE), relative_tolerance=1e-4)):
            with pytest.raises(ValueError, match="only the frozen profile 92396791"):
                solve_transient(electronic37, initial, protocol, method=BDF, bdf_options=custom, **call)
        # An unknown equation selection is refused before the time-method scope is computed.
        for request in (lambda: solve_transient(electronic37, initial, protocol, method=BDF, **call),
                        lambda: run_transient_for_publication(electronic37, initial, protocol, method=BDF, **call)):
            monkeypatch.setenv("NEWCORE_EQUATIONS", "unknown")
            with pytest.raises(ValueError, match="unknown equation implementation"):
                request()
            monkeypatch.delenv("NEWCORE_EQUATIONS")
        with pytest.raises(ValueError, match="protocol's own continuous drive"):
            solve_transient(electronic37, initial, protocol, method=BDF, bdf_drive=jump, **call)
        with pytest.raises(ValueError, match="explicit bdf time-method request"):
            solve_transient(electronic37, initial, protocol, bdf_options=BDFOptions(**PROFILE), **call)
        with pytest.raises(ValueError, match="explicit bdf time-method request"):
            solve_transient(electronic37, initial, protocol, method=BACKWARD_EULER, bdf_drive=jump, **call)
    device, ionic_initial, ionic_protocol = ionic_case(tmp_path)
    with no_numerical_work(monkeypatch):
        with pytest.raises(ValueError, match="only the 10 allowlisted"):
            solve_ionic_transient(device, ionic_initial, ionic_protocol, options=ionic_controls(), method=BDF)
        for implementation in (ASSEMBLED, LEGACY):
            with time_method_scope(BDF), equation_implementation(implementation), pytest.raises(
                    ValueError, match="only the 10 allowlisted"):
                solve_ionic_transient(device, ionic_initial, ionic_protocol, options=ionic_controls())
    from solarlab.experiments.multiple_ionic_transient import solve_multiple_ionic_transient

    with time_method_scope(BDF), pytest.raises(ValueError, match="no BDF adapter"):
        solve_multiple_ionic_transient(None, None, None, options=None)


class InteriorDrive(BDFDrive):
    """Science's D1 reproducer: the protocol's own arrays, another waveform between the knots."""

    def values(self, t, side="right"):
        voltage, light = super().values(t, side)
        return voltage, light + 0.1


@dataclass(frozen=True)
class LookAlikeDrive:
    times_s: np.ndarray
    left_values: np.ndarray
    right_values: np.ndarray


def test_admission_trusts_no_caller_drive_family_or_digest(monkeypatch, tmp_path, electronic37, envelopes, allowlisted):
    """D1 and D2 at every production entry point: each is refused explicitly before numerical work, never run as BE.

    The drive is the protocol's own: a subclass with equal arrays and its own values(), an equal drive carrying
    a method of its own, an equal look-alike of another type and a replaced from_protocol are refused. The
    family is the compiled device's exact class, and no entry point takes a scope digest.
    """
    from solarlab.equations.dc_1d import CompiledDevice1D

    initial, protocol, options = envelopes["E_saved_packet"]
    call = dict(options=options, initial_options=SteadyOptions())
    values = np.column_stack((protocol.voltage_V, protocol.generation_scale))
    carried = BDFDrive.from_protocol(protocol)
    object.__setattr__(carried, "values", lambda t, side="right": (0.0, 1.1))
    drives = {"subclass with its own values()": InteriorDrive(protocol.times_s, values, values),
              "equal drive carrying a method": carried,
              "equal arrays, another type": LookAlikeDrive(protocol.times_s, values, values)}
    entries = {
        "solve_transient": lambda **kw: solve_transient(electronic37, initial, protocol, **call, **kw),
        "run_transient_for_publication": lambda **kw: run_transient_for_publication(electronic37, initial, protocol,
                                                                                    **call, **kw),
        "run_production_bdf_for_publication": lambda **kw: bdf_results.run_production_bdf_for_publication(
            electronic37, initial, protocol, **call, **kw),
    }
    with no_numerical_work(monkeypatch):
        for entry, run in entries.items():
            for reason, drive in drives.items():
                with pytest.raises(ValueError, match="protocol's own continuous drive"):
                    run(method=BDF, bdf_drive=drive)
                with time_method_scope(BDF), pytest.raises(ValueError, match="protocol's own continuous drive"):
                    run(bdf_drive=drive)
                with pytest.raises(ValueError, match="explicit bdf time-method request"):
                    run(bdf_drive=drive)
            for mode in ("explicit", "flipped default"):
                with monkeypatch.context() as patch:
                    patch.setattr(BDFDrive, "from_protocol",
                                  classmethod(lambda cls, p: InteriorDrive(p.times_s, values, values)))
                    if mode == "explicit":
                        with pytest.raises(ValueError, match="not its own continuous knots"):
                            run(method=BDF)
                    else:
                        patch.setattr(time_method, "ALLOWLISTED_DEFAULT", (BDF, "CommanderBdfDefaultFreezeV2"))
                        with pytest.raises(ValueError, match="not its own continuous knots"):
                            run()
        # D2: the family is the device's exact class; a device of another family or a subclass is refused.
        device, ionic_initial, ionic_protocol = ionic_case(tmp_path)
        with pytest.raises(ValueError, match="not of the ionic_1d family"):
            solve_ionic_transient(electronic37, initial, protocol, options=ionic_controls(), method=BDF)
        with pytest.raises(ValueError, match="not of the electronic_1d family"):
            solve_transient(device, ionic_initial, ionic_protocol, options=options, initial_options=SteadyOptions(),
                            method=BDF)

        class WidenedDevice(CompiledDevice1D):
            pass

        widened = object.__new__(WidenedDevice)
        widened.__dict__.update(vars(electronic37))
        for run in (lambda: solve_transient(widened, initial, protocol, method=BDF, **call),
                    lambda: bdf_results.run_production_bdf_for_publication(widened, initial, protocol, method=BDF,
                                                                           **call)):
            with pytest.raises(ValueError, match="not of a supported family|not of the electronic_1d family"):
                run()
        # The scope words come from the actual call: a device whose derived geometry is not its own has none.
        stale = replace(electronic37.geometry)
        object.__setattr__(stale, "extent_m", stale.extent_m * 2)
        bent = replace(electronic37)
        object.__setattr__(bent, "geometry", stale)
        with pytest.raises(ValueError, match="scope cannot be computed.*not derived"):
            solve_transient(bent, initial, protocol, method=BDF, **call)
    with pytest.raises(TypeError):
        time_method.resolve_time_method(time_method.IONIC, method=BDF, scope=lambda: allowlisted["E_saved_packet"])


class IntegratorReached(BaseException):
    """Raised by the integrator spy: the gate admitted the call; no step was taken."""


def production_routes(electronic37, initial, protocol, call):
    """The three production BDF routes, each with an explicit bdf request."""
    from support.provenance import checked_provenance_owner

    def owned(run):
        def route():
            with checked_provenance_owner():
                return run()
        return route

    return {
        "solve_transient": lambda: solve_transient(electronic37, initial, protocol, method=BDF, **call),
        "run_production_bdf_for_publication": owned(lambda: bdf_results.run_production_bdf_for_publication(
            electronic37, initial, protocol, method=BDF, **call)),
        "run_transient_for_publication": owned(lambda: run_transient_for_publication(
            electronic37, initial, protocol, method=BDF, **call)),
    }


def test_admission_keys_what_runs_and_runs_what_it_keyed(monkeypatch, electronic37, allowlisted):
    """D1b: the integrator receives exactly the inputs admission keyed, or the call is refused before it.

    Science's reproducer (probe_constructor_mutation.py): a replaced BDFDrive.from_protocol changes the generation
    of the protocol it is handed after keying, and returns the exact drive of the changed protocol. Here it is
    refused at every production route. A variant changes the caller's protocol, initial state and options while
    admission runs, which cannot reach the integrator. A change to the inputs that the adapter prepares, which lasts
    until its last check, is refused there. The integrator is a spy: no step runs.
    """
    from solarlab.experiments import bdf as bdf_backend
    from solarlab.solvers.time.bdf_1d import BDFDrive

    initial, protocol, options = owned_envelope(electronic37)
    call = dict(options=options, initial_options=SteadyOptions())
    keyed = {"u": initial.state.u.tobytes(), "generation": protocol.generation_scale.tobytes(),
             "max_steps": options.max_steps, "budget": options.newton_budget}
    midpoint = float(protocol.times_s[1] / 2)
    seen = []

    def spy(factory, start, span, **kwargs):
        problem = factory(midpoint, "right").problem
        seen.append({"generation": problem.generation_scale, "start": start.tobytes(),
                     "budget": kwargs["settings"](None).budget})
        raise IntegratorReached()

    mutate_caller, restore_caller = caller_changes(electronic37, initial, protocol, options)
    monkeypatch.setattr(bdf_backend, "solve_bdf", spy)
    for route, run in production_routes(electronic37, initial, protocol, call).items():
        # Science's attack: the replaced constructor changes the protocol it is handed, after keying.
        def changes_its_argument(cls, p):
            changed = p.generation_scale.copy()
            changed[1] += 0.1
            object.__setattr__(p, "generation_scale", TransientProtocol(p.times_s, p.voltage_V, changed).generation_scale)
            values = np.column_stack((p.voltage_V, p.generation_scale))
            return cls(p.times_s, values, values)

        with monkeypatch.context() as patch:
            patch.setattr(BDFDrive, "from_protocol", classmethod(changes_its_argument))
            with pytest.raises(ValueError, match="not its own continuous knots"):
                run()
        assert protocol.generation_scale.tobytes() == keyed["generation"] and not seen, route

        # The caller's own objects change while admission runs: the integrator still receives the keyed words.
        def mutates_the_caller(cls, p):
            mutate_caller()
            values = np.column_stack((p.voltage_V, p.generation_scale))
            return cls(p.times_s, values, values)

        with monkeypatch.context() as patch:
            patch.setattr(BDFDrive, "from_protocol", classmethod(mutates_the_caller))
            with pytest.raises(IntegratorReached):
                run()
        restore_caller()
        observed = seen.pop()
        assert observed["generation"] == 0.0 and observed["start"] == keyed["u"], route
        assert observed["budget"] == keyed["budget"] and observed["budget"].max_iterations > 1, route

        # A change to the inputs the adapter prepares that lasts until its last check: refused there.
        verify_point = bdf_backend._verify_point

        def swaps_the_built_initial(compiled, point, generation, controls):
            verify_point(compiled, point, generation, controls)
            moved = point.state.u.copy()
            moved[compiled.node_count] = np.nextafter(moved[compiled.node_count], np.inf)
            object.__setattr__(point, "state", compiled.state(moved, voltage_V=point.voltage_V))

        with monkeypatch.context() as patch:
            patch.setattr(bdf_backend, "_verify_point", swaps_the_built_initial)
            with pytest.raises(ValueError, match="inputs handed to the integrator are not the admitted ones"):
                run()
        assert not seen, route

    # Science's attack under the flipped default (a default request on the allowlisted envelope): refused too.
    def changes_its_argument(cls, p):
        changed = p.generation_scale.copy()
        changed[1] += 0.1
        object.__setattr__(p, "generation_scale", TransientProtocol(p.times_s, p.voltage_V, changed).generation_scale)
        values = np.column_stack((p.voltage_V, p.generation_scale))
        return cls(p.times_s, values, values)

    with monkeypatch.context() as patch:
        patch.setattr(time_method, "ALLOWLISTED_DEFAULT", (BDF, "CommanderBdfDefaultFreezeV2"))
        patch.setattr(BDFDrive, "from_protocol", classmethod(changes_its_argument))
        for run in (lambda: solve_transient(electronic37, initial, protocol, **call),
                    lambda: bdf_results.run_production_bdf_for_publication(electronic37, initial, protocol, **call)):
            with pytest.raises(ValueError, match="not its own continuous knots"):
                run()
    assert protocol.generation_scale.tobytes() == keyed["generation"] and not seen
    # A caller object that changes between keying and its record is a divergence, refused under either request.
    record = time_method._record

    def changes_before_its_record(value):
        if value is protocol:
            mutate_caller()
        return record(value)

    for request in (BDF, None):
        with monkeypatch.context() as patch:
            patch.setattr(time_method, "_record", changes_before_its_record)
            if request is None:
                patch.setattr(time_method, "ALLOWLISTED_DEFAULT", (BDF, "CommanderBdfDefaultFreezeV2"))
            with pytest.raises(ValueError, match="changed while they were admitted"):
                solve_transient(electronic37, initial, protocol, method=request, **call)
        restore_caller()
    assert not seen
    # An admission holds records with the admitted key, never an object of the caller's.
    selection, admitted = time_method.admit_time_method(BDF, compiled=electronic37, initial=initial, protocol=protocol,
                                                        family=time_method.ELECTRONIC, **call)
    assert admitted is not None and admitted.scope_sha256 == selection.scope_sha256 == allowlisted["E_dark_hold"]
    built = [time_method._build(node) for node in admitted.inputs]
    for copy, value in zip(built, (electronic37, initial, protocol, options, SteadyOptions())):
        assert copy is not value and type(copy) is type(value) and time_method._same(copy, value)
    assert time_method._scope_of(time_method.ELECTRONIC, *built, time_method._canonical_drive(built[2])) == (
        admitted.scope_sha256)


def test_a_production_capture_takes_one_admission_and_its_own_selection(monkeypatch, electronic37, allowlisted):
    """D1-R1: capture_admitted_bdf, the single-admission capture, refuses a selection that is not its admission's."""
    from dataclasses import replace as rebuilt

    initial, protocol, options = owned_envelope(electronic37)
    call = dict(options=options, initial_options=SteadyOptions())
    selection, admitted = time_method.admit_time_method(BDF, compiled=electronic37, initial=initial, protocol=protocol,
                                                        family=time_method.ELECTRONIC, **call)
    light_initial, light_protocol, light_options = envelope(electronic37, sources()["E_light_hold"])
    other, _ = time_method.admit_time_method(BDF, compiled=electronic37, initial=light_initial, protocol=light_protocol,
                                             options=light_options, initial_options=SteadyOptions(),
                                             family=time_method.ELECTRONIC)
    assert other.scope == "E_light_hold" and other.scope_sha256 != selection.scope_sha256
    backward = time_method.resolve_time_method(BACKWARD_EULER, compiled=electronic37, family=time_method.ELECTRONIC)
    with no_numerical_work(monkeypatch):
        for wrong in (other, rebuilt(selection, scope_sha256=other.scope_sha256), rebuilt(selection, scope="E_light_hold"),
                      rebuilt(selection, family=time_method.IONIC), backward, selection.record()):
            with pytest.raises(ValueError, match="selection"):
                bdf_results.capture_admitted_bdf(wrong, admitted)
        with pytest.raises(ValueError, match="not an admission of a production BDF call"):
            bdf_results.capture_admitted_bdf(selection, tuple(admitted))


def owned_envelope(electronic37):
    """E_dark_hold with protocol, initial state and controls of the caller's own, which the tests change."""
    initial, protocol, options = envelope(electronic37, sources()["E_dark_hold"])
    return replace(initial), replace(protocol), replace(options, newton_budget=replace(options.newton_budget))


def caller_changes(electronic37, initial, protocol, options, drive=None):
    """Change, and restore, every field of the caller's objects that the recheck changed (Cnd2D1D2RecheckV2)."""
    saved = [(protocol, "generation_scale", protocol.generation_scale), (initial, "state", initial.state),
             (options, "max_steps", options.max_steps),
             (options.newton_budget, "max_iterations", options.newton_budget.max_iterations)]
    if drive is not None:
        saved += [(drive, "left_values", drive.left_values), (drive, "right_values", drive.right_values)]

    def mutate():
        changed = protocol.generation_scale.copy()
        changed[1] += 0.1
        object.__setattr__(protocol, "generation_scale", TransientProtocol(
            protocol.times_s, protocol.voltage_V, changed).generation_scale)
        moved = initial.state.u.copy()
        moved[electronic37.node_count] = np.nextafter(moved[electronic37.node_count], np.inf)
        object.__setattr__(initial, "state", electronic37.state(moved, voltage_V=initial.voltage_V))
        object.__setattr__(options, "max_steps", 31)
        object.__setattr__(options.newton_budget, "max_iterations", options.newton_budget.max_iterations + 1)
        if drive is not None:
            values = np.column_stack((protocol.voltage_V, protocol.generation_scale))
            object.__setattr__(drive, "left_values", BDFDrive(protocol.times_s, values, values).left_values)
            object.__setattr__(drive, "right_values", BDFDrive(protocol.times_s, values, values).right_values)

    def restore():
        for target, name, value in saved:
            object.__setattr__(target, name, value)

    return mutate, restore


class Recast(tuple):
    """A layout-compatible tuple class: what rebinding an admission's class would give it."""

    __slots__ = ()


@pytest.mark.parametrize("request_source", ["argument", "scope", "environment", "default"])
@pytest.mark.parametrize("route", ["solve_transient", "run_production_bdf_for_publication",
                                   "run_transient_for_publication"])
@pytest.mark.parametrize("condition", ["profile", "trace", "monitoring", "gc", "numpy", "signal", "thread"])
def test_interface_trust_admits_monitoring_profilers_callbacks_and_worker_threads(
        monkeypatch, electronic37, allowlisted, route, request_source, condition):
    """2026-11-02 interface_level_trust_standard_v1: monitoring, profilers, callbacks and worker threads are trusted."""
    import gc
    import signal
    import sys
    import threading

    from support.provenance import checked_provenance_owner

    initial, protocol, options = owned_envelope(electronic37)
    call = dict(options=options, initial_options=SteadyOptions())
    expected_u = initial.state.u.tobytes()
    entry = {"solve_transient": solve_transient, "run_transient_for_publication": run_transient_for_publication,
             "run_production_bdf_for_publication": bdf_results.run_production_bdf_for_publication}[route]
    if request_source == "environment":
        monkeypatch.setenv(ENVIRONMENT, BDF)
    elif request_source == "default":
        monkeypatch.setattr(time_method, "ALLOWLISTED_DEFAULT", (BDF, "CommanderBdfDefaultFreezeV2"))
    observed = []
    collecting = gc.isenabled()

    def spy(factory, start, interval, **kwargs):
        observed.append((start.tobytes(), kwargs["settings"](None).budget.max_iterations,
                         kwargs["output_times_s"].tobytes(), kwargs["options"].max_steps))
        assert gc.isenabled() is collecting
        raise IntegratorReached()

    monkeypatch.setattr(bdf_backend, "solve_bdf", spy)

    def run():
        def invoke():
            if request_source == "scope":
                with time_method_scope(BDF):
                    return entry(electronic37, initial, protocol, **call)
            return entry(electronic37, initial, protocol, method=BDF if request_source == "argument" else None, **call)

        if route == "solve_transient":
            return invoke()
        with checked_provenance_owner():
            return invoke()

    def observer(frame, event, value):
        return observer

    def callback(*args):
        return None

    ready, finish = threading.Event(), threading.Event()
    worker = None
    previous_profile, previous_trace = sys.getprofile(), sys.gettrace()
    previous_signal, previous_error = signal.getsignal(signal.SIGUSR1), np.geterrcall()
    tool = 5
    if condition == "monitoring" and not hasattr(sys, "monitoring"):
        pytest.skip("sys.monitoring is unavailable on Python 3.11; profile and trace routes are covered")
    try:
        if condition == "profile":
            sys.setprofile(observer)
        elif condition == "trace":
            sys.settrace(observer)
        elif condition == "monitoring":
            sys.monitoring.use_tool_id(tool, "cnd2_input_observer")
        elif condition == "gc":
            gc.callbacks.append(callback)
        elif condition == "numpy":
            np.seterrcall(callback)
        elif condition == "signal":
            signal.signal(signal.SIGUSR1, callback)
        elif condition == "thread":
            def caller():
                ready.set()
                finish.wait(10)
            worker = threading.Thread(target=caller)
            worker.start()
            assert ready.wait(5)
        with pytest.raises(IntegratorReached):
            run()
        assert observed == [(expected_u, options.newton_budget.max_iterations, protocol.times_s.tobytes(), 10000)]
    finally:
        sys.setprofile(previous_profile)
        sys.settrace(previous_trace)
        signal.signal(signal.SIGUSR1, previous_signal)
        np.seterrcall(previous_error)
        if callback in gc.callbacks:
            gc.callbacks.remove(callback)
        if condition == "monitoring" and hasattr(sys, "monitoring"):
            sys.monitoring.free_tool_id(tool)
        finish.set()
        if worker is not None:
            worker.join(5)
            assert not worker.is_alive()


def test_admitted_records_are_immutable_and_a_recast_admission_refuses(monkeypatch, electronic37, allowlisted):
    initial, protocol, options = owned_envelope(electronic37)
    selection, admitted = time_method.admit_time_method(
        BDF, compiled=electronic37, initial=initial, protocol=protocol, options=options,
        initial_options=SteadyOptions())
    assert admitted is not None and admitted.scope_sha256 == selection.scope_sha256
    for name in AdmittedBDFCall._fields + ("cache",):
        with pytest.raises(AttributeError):
            object.__setattr__(admitted, name, None)
    admitted.__class__ = Recast
    with no_numerical_work(monkeypatch), pytest.raises(ValueError, match="not an admission"):
        time_method.integrate_admitted(admitted, bdf_backend.solve_electronic_bdf)


@pytest.mark.parametrize("name", list(sources()))
def test_canonical_bdf_admission_with_real_supervisor_helper(
        name, monkeypatch, tmp_path, electronic37, envelopes, allowlisted):
    """Ordinary product helper threads and SQLite trace callbacks remain admitted without pausing GC."""
    import gc
    import sys

    from solarlab.experiments.execution import ProcessSupervisor
    from solarlab.experiments.transient_worker import transient_worker_binding

    initial, protocol, options = envelopes[name]
    expected = initial.state.u.tobytes()
    observed = []
    collecting = gc.isenabled()

    def recording(factory, start, interval, **kwargs):
        observed.append((start.tobytes(), kwargs["settings"](None).budget.max_iterations,
                         kwargs["output_times_s"].tobytes(), kwargs["options"].max_steps))
        assert gc.isenabled() is collecting
        raise IntegratorReached()

    monkeypatch.setattr(bdf_backend, "solve_bdf", recording)
    with RunStore(tmp_path / "store") as store, ProcessSupervisor(
            store, tmp_path / "work", (transient_worker_binding(),), max_active=1, max_pending=1,
            shared_rss_bytes=3 * 1024**3, shared_output_bytes=384 * 1024**2) as supervisor:
        assert supervisor._thread is not None and supervisor._thread.is_alive()
        frames = sys._current_frames()
        assert supervisor._thread.ident in frames
        store._db.set_trace_callback(lambda statement: None)
        try:
            with pytest.raises(IntegratorReached):
                solve_transient(electronic37, initial, protocol, options=options,
                                initial_options=SteadyOptions(), method=BDF)
        finally:
            store._db.set_trace_callback(None)
    assert gc.isenabled() is collecting
    assert observed == [(expected, options.newton_budget.max_iterations, protocol.times_s.tobytes(), 10000)]


def test_the_flip_point_defaults_only_allowlisted_envelopes_to_bdf(monkeypatch, tmp_path, electronic37, envelopes,
                                                                    allowlisted):
    monkeypatch.setattr(time_method, "ALLOWLISTED_DEFAULT", (BDF, "CommanderBdfDefaultFreezeV2"))
    initial, protocol, options = envelopes["E_saved_packet"]
    call = dict(options=options, initial_options=SteadyOptions())
    default = solve_transient(electronic37, initial, protocol, **call)
    assert_frozen_bdf(default, "E_saved_packet", "default")
    assert words(default) == words(solve_transient(electronic37, initial, protocol, method=BDF, **call))
    for opt_out in (lambda: solve_transient(electronic37, initial, protocol, method=BACKWARD_EULER, **call),
                    lambda: run_transient_for_publication(electronic37, initial, protocol, method=BACKWARD_EULER,
                                                          **call).result):
        assert type(opt_out()) is TransientResult1D
    light = protocol.generation_scale.copy()
    light[-1] *= 1.0001
    unlisted = replace(protocol, generation_scale=light)
    kept = solve_transient(electronic37, initial, unlisted, **call)
    assert type(kept) is TransientResult1D and kept.completed
    assert words(kept) == words(solve_transient(electronic37, initial, unlisted, method=BACKWARD_EULER, **call))
    device, ionic_initial, ionic_protocol = ionic_case(tmp_path)
    assert solve_ionic_transient(device, ionic_initial, ionic_protocol, options=ionic_controls()).completed
    # The flipped default on either equation implementation: the same BDF history.
    for implementation in (ASSEMBLED, LEGACY):
        with equation_implementation(implementation):
            assert words(solve_transient(electronic37, initial, protocol, **call)) == words(default), implementation


def test_the_ionic_production_branch_forwards_the_work_cap_past_the_old_clip(monkeypatch, tmp_path):
    """RV2-C2 on ReadinessV2's ionic_single13 envelope, admitted by an injected test allowlist entry.

    No ionic scope is allowlisted: without the entry the request is refused. With it, the production caller
    runs the frozen profile with the forwarded cap of 10000 beside the sample-interval admission of 16. The
    adapter with the cap omitted keeps its old clipping and exhausts it at 16 accepted steps, as ReadinessV1
    recorded (G3).
    """
    from solarlab.equations.time_method import ionic_scope
    from solarlab.experiments.bdf import solve_ionic_bdf

    device, initial, protocol = ionic_case(tmp_path)
    physical = ionic_controls()
    with pytest.raises(ValueError, match="only the 10 allowlisted"):
        solve_ionic_transient(device, initial, protocol, options=physical, method=BDF)
    monkeypatch.setattr(time_method, "ALLOWLIST",
                        {ionic_scope(device, initial, protocol, options=physical): "test:ionic_single13"})
    result = solve_ionic_transient(device, initial, protocol, options=physical, method=BDF)
    assert type(result) is BDFResult and result.completed, (result.termination, result.error)
    assert len(result.accepted_steps) > physical.electronic.max_steps == 16
    assert result.controls["max_steps"] == result.controls["adapter_adaptive_max_steps"] == ADAPTIVE_MAX_STEPS
    assert result.controls["adapter_sample_interval_limit"] == 16
    assert {key: result.controls[key] for key in PROFILE if key not in {"max_steps", "wall_s"}} == {
        key: value for key, value in PROFILE.items() if key not in {"max_steps", "wall_s"}}
    record = result.controls["time_method"]
    assert (record["family"], record["scope"], record["adaptive_max_steps"]) == ("ionic_1d", "test:ionic_single13", 10000)
    with equation_implementation(LEGACY):
        clipped = solve_ionic_bdf(device, initial, protocol, options=physical, bdf_options=BDFOptions(**PROFILE))
    assert (clipped.termination, len(clipped.accepted_steps), clipped.controls["max_steps"]) == (
        "step_budget_exhausted", 16, 16)
    assert words(result.accepted_coordinates[:17]) == words(clipped.accepted_coordinates)


def published(store, name, capture, nodes_m):
    row = store.enqueue(name, opaque_identity=capture.identity, input_metadata={"fixture": name})
    token = store.claim(name, expected_attempt_id=row["current_attempt_id"], expected_generation=row["generation"],
                        owner_id="test")
    document = publish_transient_result(store, token, capture.result, nodes_m=nodes_m, capture=capture)
    store.complete(token, state="succeeded" if capture.result.completed else "failed", result=document,
                   artifacts=document["artifact_ids"])


def test_both_methods_publish_their_selection_and_read_back_without_a_solver(tmp_path, electronic37, envelopes,
                                                                             allowlisted, unflipped):
    from support.provenance import checked_provenance_owner

    initial, protocol, options = envelopes["E_saved_packet"]
    call = dict(options=options, initial_options=SteadyOptions())
    with checked_provenance_owner():
        production = run_transient_for_publication(electronic37, initial, protocol, method=BDF, **call)
    with checked_provenance_owner():
        research = bdf_results.run_bdf_for_publication(electronic37, initial, protocol,
                                                       bdf_options=BDFOptions(1e-7, adaptive=False, max_order=1), **call)
    with checked_provenance_owner():
        default = run_transient_for_publication(electronic37, initial, protocol, **call)
    assert type(default.result) is TransientResult1D
    with RunStore(tmp_path / "store") as store:
        for name, capture in (("production", production), ("research", research), ("default", default)):
            published(store, name, capture, electronic37.nodes_m)
        loaded = read_bdf_result(store, "production")
        record = loaded["record"]["request"]["time_method"]
        assert record == production.result.controls["time_method"] == loaded["history"]["controls"]["time_method"]
        assert (record["requested"], record["source"], record["effective"], record["scope"]) == (
            BDF, "argument", BDF, "E_saved_packet")
        assert record["scope_sha256"] == allowlisted["E_saved_packet"]
        assert loaded["record"]["caps"] == {"requested_max_steps": 10000, "physical_max_steps": 32,
                                            "effective_max_steps": 10000, "effective_max_attempts": 20000}
        assert loaded["summary"]["completed"] and loaded["requested_end_state"] is not None
        assert read_transient_result(store, "production")["summary"] == loaded["summary"]
        explicit = read_bdf_result(store, "research")["record"]["request"]["time_method"]
        assert (explicit["source"], explicit["scope"], explicit["profile"], explicit["adaptive_max_steps"]) == (
            "research_capture", None, None, None)
        saved = read_transient_result(store, "default")
        assert saved["provenance"]["description"]["time_method"] == {
            "schema": "solarlab.time-method-selection.v1", "family": "electronic_1d", "requested": None,
            "source": "default", "effective": BACKWARD_EULER, "allowlisted_default": [BACKWARD_EULER, None]}
    body = _decode(_loads(production.history_bytes))
    assert _validate_body(body) == loaded["summary"]
    for change in ("history_record", "scope", "profile", "cap", "research_claims_scope"):
        changed = _decode(_loads(production.history_bytes))
        request, history = changed["request"], changed["history"]
        if change == "history_record":
            del history["controls"]["time_method"]
        elif change == "scope":
            request["time_method"] = history["controls"]["time_method"] = {**request["time_method"], "scope": None}
        elif change == "profile":
            request["time_method"] = history["controls"]["time_method"] = {
                **request["time_method"], "profile": {**PROFILE, "relative_tolerance": 1e-4}}
        elif change == "cap":
            history["controls"]["adapter_adaptive_max_steps"] = 32
        else:
            request["time_method"] = history["controls"]["time_method"] = {
                **request["time_method"], "source": "research_capture"}
        with pytest.raises(ValueError, match="time-method|frozen profile|research BDF capture"):
            _validate_body(changed)
    assert _digest(body) == _digest(_decode(_loads(production.history_bytes)))
