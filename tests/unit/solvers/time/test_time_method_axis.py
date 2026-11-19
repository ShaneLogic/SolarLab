"""The time-method axis (CND2): requests, scope-based defaults, refusals and records.

Every scope here is a stand-in digest; the production entry points compute
real ones (tests/integration/test_time_method_selection.py).
"""

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest

from solarlab.equations import time_method
from solarlab.equations.time_method import (
    ADAPTIVE_MAX_STEPS, ALLOWLIST, BACKWARD_EULER, BDF, ELECTRONIC, ENVIRONMENT, IONIC, MULTIPLE_IONIC, PROFILE,
    PROFILE_SHA256, TRUST_STANDARD, TimeMethodSelection, requested_time_method, resolve_time_method, scope_digest,
    time_method_setting, verify_time_method,
)

LISTED = next(iter(ALLOWLIST))
UNLISTED = "0" * 64


@pytest.fixture(autouse=True)
def no_process_setting(monkeypatch):
    monkeypatch.delenv(ENVIRONMENT, raising=False)


def never():
    raise AssertionError("the scope was computed")


def flipped(monkeypatch):
    monkeypatch.setattr(time_method, "ALLOWLISTED_DEFAULT", (BDF, "CommanderBdfDefaultFreezeV2"))
    monkeypatch.setattr(time_method, "_SWITCH_DATE", "2026-11-08")
    monkeypatch.setattr(time_method, "_route_today", lambda: date(2026, 11, 8))


def test_no_default_is_flipped_before_the_route_decision():
    """No flip until the commander sets a post-recheck date, never before 2026-11-08 Asia/Shanghai."""
    method, authority = time_method.ALLOWLISTED_DEFAULT
    chosen = time_method._SWITCH_DATE
    today = datetime.now(timezone(timedelta(hours=8))).date()
    if chosen is None or today < date.fromisoformat(chosen):
        assert (method, authority) == (BACKWARD_EULER, None)
    if chosen is not None:
        assert date.fromisoformat(chosen) >= date(2026, 11, 8)
    assert method in time_method.TIME_METHODS and (method == BDF) == (authority is not None)
    assert authority is None or authority.startswith("CommanderBdfDefaultFreeze")


@pytest.mark.parametrize("chosen,today,allowed", [
    (None, date(2026, 11, 2), False), (None, date(2026, 11, 8), False),
    (None, date(2026, 11, 9), False), ("2026-11-07", date(2026, 11, 8), False),
    ("2026-11-08", date(2026, 11, 7), False), ("2026-11-08", date(2026, 11, 8), True),
    ("2026-11-09", date(2026, 11, 8), False), ("2026-11-09", date(2026, 11, 9), True),
])
def test_the_default_switch_requires_a_commander_date(monkeypatch, chosen, today, allowed):
    monkeypatch.setattr(time_method, "_SWITCH_DATE", chosen)
    monkeypatch.setattr(time_method, "_route_today", lambda: today)
    if allowed:
        time_method._check_default_switch()
    else:
        with pytest.raises(ValueError, match="BDF is not selectable"):
            time_method._check_default_switch()


def test_requests_resolve_by_argument_then_scope_then_process_setting(monkeypatch):
    assert requested_time_method() == (None, "default")
    monkeypatch.setenv(ENVIRONMENT, BACKWARD_EULER)
    assert requested_time_method() == (BACKWARD_EULER, "environment")
    with time_method.time_method(BDF):
        assert requested_time_method() == (BDF, "scope")
        assert requested_time_method(BACKWARD_EULER) == (BACKWARD_EULER, "argument")
        with time_method.time_method(BACKWARD_EULER):
            assert requested_time_method() == (BACKWARD_EULER, "scope")
        assert requested_time_method() == (BDF, "scope")
    for refused in ("Bdf", "radau", ""):
        monkeypatch.setenv(ENVIRONMENT, refused)
        with pytest.raises(ValueError, match="unknown time method"):
            requested_time_method()
    monkeypatch.delenv(ENVIRONMENT)
    with pytest.raises(ValueError, match="unknown time method"):
        requested_time_method("ida")
    with pytest.raises(ValueError, match="unknown time method"), time_method.time_method("backward-euler"):
        pass


@pytest.fixture(scope="module")
def device():
    """A genuine compiled electronic device; the scope digest itself is stood in for below."""
    from examples.research_1d.steady import load_device
    from solarlab.device.compile_1d import compile_device_1d

    return compile_device_1d(load_device(), intervals_per_layer=12)


@pytest.fixture
def protocol():
    from solarlab.experiments.transient import TransientProtocol

    return TransientProtocol(np.linspace(0, 4e-7, 5), np.zeros(5), np.zeros(5))


@pytest.fixture
def stand_in(monkeypatch):
    """The resolver's own scope computation, replaced by a stand-in digest the test sets."""
    digest = {"value": LISTED, "calls": 0}

    def scope(*args, **kwargs):
        digest["calls"] += 1
        if isinstance(digest["value"], Exception):
            raise digest["value"]
        return digest["value"]

    monkeypatch.setattr(time_method, "electronic_scope", scope)
    return digest


def call(device, protocol, **kwargs):
    return resolve_time_method(kwargs.pop("method", None), compiled=device, protocol=protocol, family=ELECTRONIC, **kwargs)


@pytest.mark.parametrize("flip", [False, True], ids=["before_flip", "after_flip"])
def test_backward_euler_is_the_explicit_opt_out_and_computes_no_scope(monkeypatch, device, protocol, stand_in, flip):
    if flip:
        flipped(monkeypatch)
    selection = call(device, protocol, method=BACKWARD_EULER)
    assert selection == TimeMethodSelection(ELECTRONIC, BACKWARD_EULER, "argument", BACKWARD_EULER)
    with time_method.time_method(BACKWARD_EULER):
        assert call(device, protocol).effective == BACKWARD_EULER
    monkeypatch.setenv(ENVIRONMENT, BACKWARD_EULER)
    assert call(device, protocol).source == "environment"
    assert stand_in["calls"] == 0


def test_the_default_is_backward_euler_for_every_scope_before_the_flip(monkeypatch, device, protocol, stand_in):
    monkeypatch.setattr(time_method, "ALLOWLISTED_DEFAULT", (BACKWARD_EULER, None))
    assert call(device, protocol) == TimeMethodSelection(ELECTRONIC, None, "default", BACKWARD_EULER)
    assert resolve_time_method(compiled=None, family=MULTIPLE_IONIC).effective == BACKWARD_EULER
    assert stand_in["calls"] == 0


def test_the_flip_point_defaults_to_bdf_on_allowlisted_scopes_only(monkeypatch, device, protocol, stand_in):
    flipped(monkeypatch)
    listed = call(device, protocol)
    assert (listed.requested, listed.source, listed.effective) == (None, "default", BDF)
    assert (listed.scope, listed.scope_sha256) == (ALLOWLIST[LISTED], LISTED)
    stand_in["value"] = UNLISTED
    assert call(device, protocol).effective == BACKWARD_EULER
    # A scope that cannot be computed is not allowlisted: the default keeps backward Euler.
    stand_in["value"] = TypeError("no scope content is defined for object")
    assert call(device, protocol).effective == BACKWARD_EULER
    assert resolve_time_method(compiled=object(), family=ELECTRONIC).effective == BACKWARD_EULER
    stand_in["value"] = LISTED
    from solarlab.equations.implementation import ASSEMBLED, LEGACY, equation_implementation

    # The equation selection is its own axis: either implementation keeps the time-method selection.
    for implementation in (ASSEMBLED, LEGACY):
        with equation_implementation(implementation):
            assert call(device, protocol).effective == BDF
    monkeypatch.setenv("NEWCORE_EQUATIONS", "unknown")
    with pytest.raises(ValueError, match="unknown equation implementation"):
        call(device, protocol)


@pytest.mark.parametrize("flip", [False, True], ids=["before_flip", "after_flip"])
def test_explicit_bdf_needs_an_allowlisted_scope_and_the_frozen_profile(monkeypatch, device, protocol, stand_in, flip):
    from dataclasses import replace

    from solarlab.solvers.time.bdf import BDFOptions

    if flip:
        flipped(monkeypatch)
    assert call(device, protocol, method=BDF) == TimeMethodSelection(ELECTRONIC, BDF, "argument", BDF,
                                                                     ALLOWLIST[LISTED], LISTED)
    assert call(device, protocol, method=BDF, bdf_options=BDFOptions(**PROFILE)).effective == BDF
    for options in (replace(BDFOptions(**PROFILE), relative_tolerance=1e-4), replace(BDFOptions(**PROFILE), max_steps=32),
                    BDFOptions(**{**PROFILE, "wall_s": 30.5}), dict(PROFILE), BDFOptions(**{**PROFILE, "adaptive": 1}),
                    BDFOptions(**{**PROFILE, "max_order": 2.0})):
        with pytest.raises(ValueError, match="only the frozen profile 92396791"):
            call(device, protocol, method=BDF, bdf_options=options)
    stand_in["value"] = UNLISTED
    with pytest.raises(ValueError, match="only the 10 allowlisted electronic envelopes"):
        call(device, protocol, method=BDF)
    stand_in["value"] = TypeError("no scope content is defined for object")
    with pytest.raises(ValueError, match="scope cannot be computed"):
        call(device, protocol, method=BDF)
    with time_method.time_method(BDF), pytest.raises(ValueError, match="no BDF adapter"):
        resolve_time_method(compiled=_multiple_device(), family=MULTIPLE_IONIC)
    stand_in["value"] = LISTED
    with time_method.time_method(BDF):
        assert call(device, protocol).source == "scope"
    monkeypatch.setenv(ENVIRONMENT, BDF)
    assert call(device, protocol).source == "environment"


def _multiple_device():
    from support.assembly_references import REFERENCE_FACTORIES

    return REFERENCE_FACTORIES["multiple_constrained:positive"]().device


def test_the_family_comes_from_the_device_and_is_never_taken_from_the_caller(device, protocol, stand_in):
    """D2: a stated family or a supplied hash that disagrees with the actual device is refused, never admitted."""
    with pytest.raises(ValueError, match=f"not of the {IONIC} family"):
        resolve_time_method(BDF, compiled=device, protocol=protocol, family=IONIC)
    # The multiple-domain entry point has no BDF adapter for any device.
    with pytest.raises(ValueError, match="no BDF adapter"):
        resolve_time_method(BDF, compiled=device, protocol=protocol, family=MULTIPLE_IONIC)
    with pytest.raises(ValueError, match="not of a supported family"):
        resolve_time_method(BDF, compiled=object(), protocol=protocol)
    # A subclass of the compiled device has no family, whatever it states.
    from solarlab.equations.dc_1d import CompiledDevice1D

    class Device(CompiledDevice1D):
        pass

    widened = object.__new__(Device)
    widened.__dict__.update(vars(device))
    with pytest.raises(ValueError, match="not of the electronic_1d family"):
        resolve_time_method(BDF, compiled=widened, protocol=protocol, family=ELECTRONIC)
    assert stand_in["calls"] == 0
    # The resolver takes no family or scope digest of its own: the old form is refused outright.
    with pytest.raises(TypeError):
        resolve_time_method("ionic_1d", method=BDF, scope=lambda: LISTED)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="unknown time method"):
        resolve_time_method("ionic_1d", compiled=device, protocol=protocol)
    with pytest.raises(ValueError, match="unknown time-method family"):
        resolve_time_method(BDF, compiled=device, protocol=protocol, family="electronic_2d")


def test_only_the_protocols_own_bdf_drive_is_admitted(monkeypatch, device, protocol, stand_in):
    """D1: the drive is not a free input; a subclass, look-alike or replaced construction is refused."""
    from dataclasses import dataclass

    from solarlab.solvers.time.bdf_1d import BDFDrive

    values = np.column_stack((protocol.voltage_V, protocol.generation_scale))
    assert call(device, protocol, method=BDF, bdf_drive=BDFDrive.from_protocol(protocol)).effective == BDF

    class Interior(BDFDrive):
        def values(self, t, side="right"):
            voltage, light = super().values(t, side)
            return voltage, light + 0.1

    @dataclass(frozen=True)
    class LookAlike:
        times_s: np.ndarray
        left_values: np.ndarray
        right_values: np.ndarray

    jump = values.copy()
    jump[1, 1] = np.nextafter(jump[1, 1], np.inf)
    for drive in (Interior(protocol.times_s, values, values), LookAlike(protocol.times_s, values, values),
                  BDFDrive(protocol.times_s, values, jump)):
        with pytest.raises(ValueError, match="protocol's own continuous drive"):
            call(device, protocol, method=BDF, bdf_drive=drive)
    for replacement in (lambda cls, p: Interior(p.times_s, values, values), lambda cls, p: BDFDrive(p.times_s, values, jump)):
        with monkeypatch.context() as patch:
            patch.setattr(BDFDrive, "from_protocol", classmethod(replacement))
            with pytest.raises(ValueError, match="not its own continuous knots"):
                call(device, protocol, method=BDF)
            flipped(patch)
            with pytest.raises(ValueError, match="not its own continuous knots"):
                call(device, protocol)
    genuine = BDFDrive.from_protocol(protocol)
    carried = BDFDrive.from_protocol(protocol)
    object.__setattr__(carried, "values", lambda t, side="right": (0.0, 1.0))
    for drive in (Interior(protocol.times_s, values, values), LookAlike(protocol.times_s, values, values),
                  BDFDrive(protocol.times_s, values, jump), carried):
        with pytest.raises(TypeError, match="exactly the protocol's own continuous BDFDrive"):
            time_method._source(protocol, drive)
    assert time_method._source(protocol, genuine) == time_method._source(protocol, None)
    assert time_method._source(protocol, None)["drive"] == "continuous_protocol_knots"
    # The same drive carrying a method of its own is refused at admission, and from a replaced constructor too.
    with pytest.raises(ValueError, match="protocol's own continuous drive"):
        call(device, protocol, method=BDF, bdf_drive=carried)
    with monkeypatch.context() as patch:
        patch.setattr(BDFDrive, "from_protocol", classmethod(lambda cls, p: carried))
        with pytest.raises(ValueError, match="not its own continuous knots"):
            call(device, protocol, method=BDF)


def test_bdf_options_or_drive_need_an_explicit_bdf_request(device, protocol, stand_in):
    from solarlab.solvers.time.bdf import BDFOptions
    from solarlab.solvers.time.bdf_1d import BDFDrive

    for method in (None, BACKWARD_EULER):
        with pytest.raises(ValueError, match="explicit bdf time-method request"):
            call(device, protocol, method=method, bdf_options=BDFOptions(**PROFILE))
        with pytest.raises(ValueError, match="explicit bdf time-method request"):
            call(device, protocol, method=method, bdf_drive=BDFDrive.from_protocol(protocol))


def test_workers_forward_the_request_and_refuse_bdf_while_prepared(monkeypatch):
    assert time_method_setting(ELECTRONIC) == (ENVIRONMENT, None)
    with time_method.time_method(BACKWARD_EULER):
        assert time_method_setting(IONIC) == (ENVIRONMENT, BACKWARD_EULER)
    flipped(monkeypatch)
    assert time_method_setting(ELECTRONIC) == (ENVIRONMENT, None)
    with time_method.time_method(BDF), pytest.raises(ValueError, match="worker jobs"):
        time_method_setting(ELECTRONIC)
    monkeypatch.setenv(ENVIRONMENT, BDF)
    with pytest.raises(ValueError, match="worker jobs"):
        time_method_setting(IONIC)
    monkeypatch.delenv(ENVIRONMENT)
    threads = (("OMP_NUM_THREADS", "1"),)
    default = verify_time_method((*threads, (ENVIRONMENT, None)), ELECTRONIC)
    assert (default.requested, default.effective) == (None, BACKWARD_EULER)
    monkeypatch.setenv(ENVIRONMENT, BACKWARD_EULER)
    explicit = verify_time_method((*threads, (ENVIRONMENT, BACKWARD_EULER)), ELECTRONIC)
    assert (explicit.requested, explicit.source, explicit.effective) == (BACKWARD_EULER, "environment", BACKWARD_EULER)
    for settings in (threads, (*threads, (ENVIRONMENT, None))):
        with pytest.raises(ValueError, match="differs from its execution manifest"):
            verify_time_method(settings, ELECTRONIC)
    monkeypatch.setenv(ENVIRONMENT, BDF)
    with pytest.raises(ValueError, match="worker jobs"):
        verify_time_method((*threads, (ENVIRONMENT, BDF)), IONIC)


def test_records_state_the_request_effective_method_and_production_scope():
    default = TimeMethodSelection(ELECTRONIC, None, "default", BACKWARD_EULER).record()
    assert default == {"schema": "solarlab.time-method-selection.v1", "family": ELECTRONIC, "requested": None,
                       "source": "default", "effective": BACKWARD_EULER,
                       "allowlisted_default": list(time_method.ALLOWLISTED_DEFAULT)}
    production = TimeMethodSelection(ELECTRONIC, BDF, "argument", BDF, ALLOWLIST[LISTED], LISTED).record()
    assert production["profile"] == dict(PROFILE) and production["profile_sha256"] == PROFILE_SHA256
    assert production["adaptive_max_steps"] == ADAPTIVE_MAX_STEPS == 10000
    assert (production["scope"], production["scope_sha256"]) == (ALLOWLIST[LISTED], LISTED)
    assert production["allowlist_sha256"] == time_method.ALLOWLIST_SHA256
    assert production["trust_standard"] == TRUST_STANDARD == "interface_level_trust_standard_v1"
    assert production["qualification"] is False and production["label"] == "experimental_unqualified"
    research = TimeMethodSelection(IONIC, BDF, "research_capture", BDF).record()
    assert all(research[name] is None for name in ("scope", "scope_sha256", "allowlist_sha256", "profile_sha256",
                                                   "profile", "adaptive_max_steps", "trust_standard"))


def test_the_frozen_profile_and_allowlist_are_the_reviewed_ones():
    assert dict(PROFILE) == {
        "initial_step_s": 0.001, "min_step_s": 1e-16, "max_step_s": 1e+30, "relative_tolerance": 1e-05,
        "absolute_tolerance": 1e-12, "max_order": 2, "adaptive": True, "safety": 0.8, "min_factor": 0.2,
        "max_factor": 2.0, "max_steps": 10000, "max_attempts": 20000, "max_consecutive_rejections": 12, "wall_s": 30.0}
    assert PROFILE_SHA256 == "9239679194ed68a8c72780877a80a84ecca3dba24a895a17fe1ac1a94fd76bea"
    assert sorted(ALLOWLIST.values()) == sorted([
        "E_dark_hold", "E_light_hold", "E_dark_voltage_ramp", "E_light_ramp_4", "E_light_ramp_8", "E_light_ramp_16",
        "E_voltage_ramp_4", "E_voltage_ramp_8", "E_voltage_ramp_16", "E_saved_packet"])
    assert time_method.ALLOWLIST_SHA256 == "e6835baf36e3b98aa1af2108053e39570f6e4133f2f8ee1dd242127b9b0b2173"


def test_scope_content_refuses_a_value_it_cannot_encode():
    """The grammar of Cnd2ScopeKeyGrammarV1: tagged hex floats, typed envelopes, raw array words."""
    assert scope_digest(ELECTRONIC, {"a": time_method._content(np.zeros(2))}) != scope_digest(
        ELECTRONIC, {"a": time_method._content(np.array([0.0, -0.0]))})
    assert time_method._content(-0.0) == {"float": "-0x0.0p+0"} != time_method._content(0.0)
    assert time_method._content(1.0) != time_method._content(1) and time_method._content(True) is True
    nan = np.array([np.nan])
    assert time_method._content(nan) == {"array": {"dtype": "<f8", "shape": [1], "sha256": __import__(
        "hashlib").sha256(nan.tobytes()).hexdigest()}}
    for value in (object(), {1: "key"}, np.array(["text"]), np.zeros(1, dtype=complex), float("nan"), float("inf"),
                  np.float64("-inf")):
        with pytest.raises(TypeError):
            time_method._content(value)
    from dataclasses import dataclass

    from solarlab.solvers.nonlinear.problem import SolveBudget

    class Budget(SolveBudget):
        pass

    @dataclass(frozen=True)
    class SolveBudgetLike:
        max_iterations: int

    # Rule R1 applies to the module-level solarlab class only: a subclass or a local look-alike is refused.
    assert time_method._content(SolveBudget(1, 2, 3, 4))["type"] == "SolveBudget"
    for value in (Budget(1, 2, 3, 4), SolveBudgetLike(1)):
        with pytest.raises(TypeError, match="not a module-level solarlab class"):
            time_method._content(value)


def test_every_scoped_value_is_exactly_its_type_with_only_its_declared_state(device):
    """Grammar V2: a subclass or numpy scalar with equal words, extra state or a stale derived field has no scope."""
    from dataclasses import replace

    class Float(float):
        pass

    class Text(str):
        pass

    class Items(tuple):
        pass

    class Words(np.ndarray):
        pass

    class Table(dict):
        pass

    plain = {"float": (1.5).hex()}
    # Grammar V2 refuses what V1 encoded ambiguously, and keeps V1's bytes for everything else.
    from types import MappingProxyType

    for value in ([1], MappingProxyType({"a": 1}), np.array(1.0)):
        with pytest.raises(TypeError):
            time_method._content(value)
    encodings = [time_method._content(value) for value in (
        (1,), {"a": 1}, 1, 1.0, True, "1", None, np.ones(1), np.ones(1, dtype=int))]
    assert len({time_method._sha256(encoded) for encoded in encodings}) == len(encodings)
    # A derived field's comparison also reads a mapping proxy, never as a dict.
    proxy = MappingProxyType({"a": 1})
    assert time_method._same(proxy, MappingProxyType({"a": 1})) and not time_method._same(proxy, {"a": 1})
    grid = np.arange(6.0).reshape(2, 3)
    assert time_method._content(grid[:, ::-1]) == time_method._content(grid[:, ::-1].copy())
    assert time_method._content(1.5) == plain and time_method._content(np.zeros(1))["array"]["dtype"] == "<f8"
    for value in (Float(1.5), Text("x"), Items((1,)), np.zeros(2).view(Words), np.ma.masked_array(np.zeros(2)),
                  Table(a=1), np.float64(1.5), np.int64(1), np.bool_(True), {"a": np.float32(1)}):
        with pytest.raises(TypeError, match="no scope content|exactly a numpy ndarray"):
            time_method._content(value)
    assert time_method._content(device)["type"] == "CompiledDevice1D"
    # A frozen instance given state beyond its fields, or a derived field that is not its re-derivation.
    extra = replace(device)
    object.__setattr__(extra, "cache", 1)
    with pytest.raises(TypeError, match="beyond its declared fields"):
        time_method._content(extra)
    stale = replace(device.geometry)
    object.__setattr__(stale, "extent_m", stale.extent_m * 2)
    with pytest.raises(TypeError, match="ControlVolumeGeometry1D.extent_m is not derived"):
        time_method._content(stale)
    with pytest.raises(TypeError, match="not derived"):
        time_method._content(replace(device, geometry=stale))


def test_a_record_is_immutable_and_builds_a_fresh_frozen_copy_with_the_same_key(device):
    """D1b, D1-R1: admission records each input as immutable values, and the inputs a history integrates are built
    from them; what the scope grammar refuses, the record refuses."""
    from dataclasses import replace
    from types import MappingProxyType

    record = time_method._record(device)

    def leaves(node):
        """Every leaf of a record: tuples all the way down, over immutable values, classes and enum members."""
        assert type(node) is tuple
        for item in node:
            if type(item) is tuple:
                yield from leaves(item)
            else:
                yield item

    kinds = {type(leaf) for leaf in leaves(record)}
    assert kinds <= {str, bytes, int, float, bool, type(None), type}, kinds
    copy, again = time_method._build(record), time_method._build(record)
    assert copy is not device and copy is not again and type(copy) is type(device) and copy.identity == device.identity
    assert time_method._same(copy, device) and time_method._sha256(time_method._content(copy)) == time_method._sha256(
        time_method._content(device))
    arrays = [copy.nodes_m, copy.geometry.nodes_m, copy.parameters.acceptor_density_m3]
    assert all(not array.flags.writeable and not array.flags.owndata for array in arrays)
    for array in arrays:
        with pytest.raises(ValueError):
            array.setflags(write=True)
    # Built by its own constructors: no part of a copy is the device's or another copy's.
    assert copy.geometry is not device.geometry and copy.parameters.layers is not device.parameters.layers
    assert copy.parameters.layers[0] is not device.parameters.layers[0] and copy.geometry is not again.geometry
    # A record refuses exactly what the scope grammar refuses.
    extra = replace(device)
    object.__setattr__(extra, "cache", 1)
    stale = replace(device.geometry)
    object.__setattr__(stale, "extent_m", stale.extent_m * 2)
    for value in ([1], MappingProxyType({"a": 1}), np.float64(1.0), np.zeros(1).view(np.ma.MaskedArray), {1: "a"},
                  extra, stale, object()):
        with pytest.raises(TypeError):
            time_method._record(value)
        with pytest.raises(TypeError):
            time_method._content(value)
    assert time_method._build(time_method._record((1, 2.0, "x", None, True))) == (1, 2.0, "x", None, True)
    for node in ((), ["value", 1], ("dataclass", object, ()), ("bogus", 1)):
        with pytest.raises(TypeError):
            time_method._build(node)


def test_an_admission_is_an_immutable_tuple_read_only_through_tuple(device, protocol):
    """D1-R1: no field of an admission can be rebound, and the gate refuses anything but exactly an admission."""
    from solarlab.equations.time_method import AdmittedBDFCall, integrate_admitted, verify_integration

    admission = AdmittedBDFCall(ELECTRONIC, ALLOWLIST[LISTED], LISTED, (time_method._record(protocol),) * 5)
    for name, value in (("scope_sha256", UNLISTED), ("inputs", ()), ("case", "E_light_hold"), ("cache", 1)):
        with pytest.raises(AttributeError):
            object.__setattr__(admission, name, value)
    with pytest.raises(TypeError):
        vars(admission)

    class Recast(tuple):
        __slots__ = ()

    def never_called(*args, **kwargs):
        raise AssertionError("an adapter ran on a refused admission")

    recast = AdmittedBDFCall(*admission)
    recast.__class__ = Recast
    for forged in (recast, tuple(admission), (ELECTRONIC, ALLOWLIST[LISTED], LISTED), None):
        with pytest.raises(ValueError, match="not an admission of a production BDF call"):
            integrate_admitted(forged, never_called)
        with pytest.raises(ValueError, match="not an admission of a production BDF call"):
            verify_integration(forged, compiled=device, initial=None, protocol=protocol, options=None, drive=None,
                               start=None, run_options=None)


def test_the_bdf_adapters_take_an_admission_only_as_a_keyword_that_defaults_to_none(monkeypatch):
    """D1-R1: research calls are unchanged; forged admissions are refused at adapter entry; runtime hooks are absent."""
    import importlib.util
    import inspect

    from solarlab.experiments import bdf
    from solarlab.experiments.ionic_transient import solve_ionic_transient
    from solarlab.experiments.transient import solve_transient

    assert importlib.util.find_spec("solarlab.equations.time_method_runtime") is None
    assert time_method.ALLOWLISTED_DEFAULT == (BACKWARD_EULER, None)
    assert time_method._SWITCH_DATE is None
    for entry in (solve_transient, solve_ionic_transient):
        params = inspect.signature(entry).parameters
        assert params["method"].default is None
        assert params["bdf_options"].default is None
        assert params["bdf_drive"].default is None
    for adapter in (bdf.solve_electronic_bdf, bdf.solve_ionic_bdf):
        parameter = inspect.signature(adapter).parameters["admission"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY and parameter.default is None

    def forbidden(*args, **kwargs):
        raise AssertionError("numerical admission ran before forged-admission refusal")

    monkeypatch.setattr(bdf, "_verify_point", forbidden)
    monkeypatch.setattr(bdf, "_admit_evidence", forbidden)
    with pytest.raises(ValueError, match="not an admission of a production BDF call"):
        bdf.solve_electronic_bdf(None, None, None, options=None, initial_options=None, bdf_options=None,
                                 admission="forged")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="not an admission of a production BDF call"):
        bdf.solve_ionic_bdf(None, None, None, options=None, bdf_options=None,
                            admission="forged")  # type: ignore[arg-type]
