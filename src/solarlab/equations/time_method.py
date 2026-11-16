"""Time-method selection of the transient production entry points (CND2).

The time integrator is an axis of its own, separate from the equation
implementation (:mod:`solarlab.equations.implementation`). One names the
integrator a history runs, the other the equation and storage classes it
evaluates. Each is resolved, recorded and refused on its own.

The values are ``backward_euler`` and ``bdf``. A history requests one

* per call: the entry point's ``method`` argument;
* in code: ``with time_method(BDF): ...`` (this context and what it calls);
* per process: ``NEWCORE_TIME_METHOD=<value>``, read outside any scope;

and otherwise takes the scope-based default. An explicit ``backward_euler``
is the opt-out: it computes no scope and always integrates with backward
Euler. Unknown values are refused, never mapped to a default.

``bdf`` is selectable only for an allowlisted scope, and runs the frozen
profile ``PROFILE`` (ReadinessV1 ProposedDefault ``92396791``) with the
adaptive work cap ``ADAPTIVE_MAX_STEPS`` forwarded to the adapter. The
allowlist holds exactly the ten electronic envelopes of
CommanderBdfDefaultFreezeV2 (those of V1, keyed by content). Each envelope is
one 37-node device content, one initial state, one set of source knots,
physical and initial controls and a drive without jumps. These envelopes are
all it covers, not other jobs on the same device. A scope is a
path-independent content digest (:func:`scope_digest`): compiled identities
bind the configuration path, so the device enters by the content they digest.
The exact byte grammar of the digest, with worked vectors, is the archive
record Cnd2ScopeKeyGrammarV2 (Core1D/AdaptiveBDFV1).

Admission keys what runs, and runs what it keyed (D1, D1b, D1-R1, D2):

* The resolver computes the scope digest itself from the call's actual
  inputs. It takes the family from the compiled device's exact class. No
  caller supplies a digest or a family.
* Every scoped value must be exactly its type, with no state beyond its
  declared fields.
* A BDF selection records the inputs as immutable values (``_record``):
  nested tuples of their declared words, as bytes and Python scalars. It
  builds objects from the records and keys them again. If that key differs
  from the key of the caller's objects, an input changed while it was
  admitted, and the call is refused. No caller-owned object is read after
  it is recorded.
* Admission returns a :class:`TimeMethodSelection` and an
  :class:`AdmittedBDFCall`. The admission is a tuple of immutable values:
  the family, the case, the key and the records. Records also name the
  solarlab classes and enum members their inputs are built from, which
  belong to solarlab's code, not to the call. Neither the selection nor
  the admission holds an object that a history integrates.
* A production route admits its call once, and integrates it with
  :func:`integrate_admitted`. That builds the history's inputs from the
  admission's records after admission has returned, and keys them again:
  they must be the admitted, allowlisted scope. It hands them to the BDF
  adapter, with the protocol's own continuous drive, the frozen profile,
  the work cap and the admission.
* The adapter first computes the integrator's start, interval, options,
  breakpoints and events. Its last step before calling the integrator
  re-keys the inputs they come from (:func:`verify_integration`).
* No function of this module returns an object that a production history
  integrates. :func:`integrate_admitted` builds the inputs in place
  (``_build_into``), and keys them without ``_declared_only``'s rebuild.

Interface-level trust boundary (``interface_level_trust_standard_v1``,
archive record ``Cnd2ReviewScopeDecisionV1`` of 2026-11-02; the earlier V4
strict-scope outcome remains preserved in history as ``not_passed`` and is
not relabelled):

* Inputs passed to ``solve_transient``, ``solve_ionic_transient``,
  ``run_production_bdf_for_publication`` and ``run_transient_for_publication``
  are copied and frozen at admission (:class:`AdmittedBDFCall`); nothing a
  caller later does to its own objects, before or after admission, changes
  what is integrated.
* Data handed to caller-supplied code or inspected via the public surface
  cannot change a run's admitted inputs: ``_genuine`` refuses subclasses,
  look-alikes and custom callbacks, ``_declared_only`` refuses state beyond
  declared fields or stale derived fields, and ``_probe_drive_constructor``
  probes only a throwaway copy.
* The integrator runs only on inputs whose scope digest equals one of the
  ten allowlisted keys with the frozen profile ``PROFILE`` and
  ``ADAPTIVE_MAX_STEPS``; anything else raises ``ValueError`` before any
  integration step, with no fallback.
* Stored records state the requested/effective method, admitted scope key,
  ``trust_standard`` and ``qualification=False`` truthfully.
* Outside this interface boundary, ``solarlab``'s internal code and runtime
  objects created during a run, other threads in the process, profiling,
  tracing, debugging and ``sys.monitoring`` tools, low-level runtime
  introspection, third-party in-process code and source changes are trusted
  rather than policed by interpreter/OS hooks.

Every other BDF request is refused before numerical work, with no fallback
to backward Euler. This covers:

* a different device, initial state, drive or control;
* a BDF profile other than the frozen one;
* an unsupported population or entry point;
* every worker job;
* a scope whose digest cannot be computed.

The scope-based default is backward Euler for every scope.
``ALLOWLISTED_DEFAULT`` is the separate, unapplied flip point. An allowlisted
default also requires the commander's switch date after an accepted
independent recheck; ``_SWITCH_DATE`` is unset. The earliest permitted date
is 2026-11-08 Asia/Shanghai. A missing, invalid or future switch date refuses
a selected BDF default with no fallback. Every unlisted scope keeps backward
Euler; explicit BDF requests retain their existing date-independent behavior.

A scope is a ContextVar, as for the equation selection. Worker jobs therefore
carry the parent's request themselves. Preparation records it among the
execution manifest's child settings (:func:`time_method_setting`):
``NEWCORE_TIME_METHOD`` is unset for the default and ``backward_euler`` for
the opt-out. The child installs the setting and refuses a request that
differs (:func:`verify_time_method`). No worker scope is allowlisted, because
a worker's initial state is a cold recipe, not a frozen initial. Workers
therefore integrate with backward Euler, and refuse ``bdf`` when the job is
prepared.

Each BDF history, captured backward-Euler publication and worker result
records its selection (:meth:`TimeMethodSelection.record`). The record holds
the request and its source and the effective method. A BDF record also holds
the scope, the profile and the work cap.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import date, datetime, timedelta, timezone
import enum
import hashlib
import json
import math
import os
import sys
from types import MappingProxyType
from typing import Any, Final, NamedTuple, TypeVar, cast

import numpy as np

__all__ = [
    "ADAPTIVE_MAX_STEPS", "ALLOWLIST", "ALLOWLIST_SHA256", "ALLOWLISTED_DEFAULT", "BACKWARD_EULER", "BDF", "ELECTRONIC",
    "ENVIRONMENT", "IONIC", "MULTIPLE_IONIC", "PROFILE", "PROFILE_SHA256", "SCOPE_SCHEMA", "SELECTION_SCHEMA",
    "TIME_METHODS", "TRUST_STANDARD", "AdmittedBDFCall", "TimeMethodSelection", "admit_time_method",
    "electronic_components", "electronic_scope", "integrate_admitted", "ionic_scope", "requested_time_method",
    "resolve_time_method", "scope_digest", "time_method", "time_method_setting", "verify_integration",
    "verify_time_method",
]

BACKWARD_EULER: Final = "backward_euler"
BDF: Final = "bdf"
TIME_METHODS = (BACKWARD_EULER, BDF)
ENVIRONMENT = "NEWCORE_TIME_METHOD"
TRUST_STANDARD: Final = "interface_level_trust_standard_v1"

# The scope-based default of the allowlisted scopes and its authority: the 2026-11-08 flip point.
ALLOWLISTED_DEFAULT: tuple[str, str | None] = (BACKWARD_EULER, None)
# A date is set only by the commander after an accepted independent recheck.
# FreezeV2's 2026-11-08 is the earliest permitted date, not an automatic switch.
_SWITCH_DATE: str | None = None

# Entry-point families.
ELECTRONIC = "electronic_1d"
IONIC = "ionic_1d"
MULTIPLE_IONIC = "multiple_ionic_1d"
_FAMILIES = (ELECTRONIC, IONIC, MULTIPLE_IONIC)

SELECTION_SCHEMA = "solarlab.time-method-selection.v1"
SCOPE_SCHEMA = "solarlab.time-method-scope.v1"

# ReadinessV1/ProposedDefault.json (file sha256), frozen by CommanderBdfDefaultFreezeV1.
PROFILE_SHA256 = "9239679194ed68a8c72780877a80a84ecca3dba24a895a17fe1ac1a94fd76bea"
PROFILE: Mapping[str, Any] = MappingProxyType({
    "initial_step_s": 0.001, "min_step_s": 1e-16, "max_step_s": 1e30, "relative_tolerance": 1e-05,
    "absolute_tolerance": 1e-12, "max_order": 2, "adaptive": True, "safety": 0.8, "min_factor": 0.2,
    "max_factor": 2.0, "max_steps": 10000, "max_attempts": 20000, "max_consecutive_rejections": 12, "wall_s": 30.0,
})
# The accepted-step work cap forwarded to the BDF adapter (ReadinessReviewV2 RV2-C2). The sample
# intervals stay admitted by the physical controls' max_steps.
ADAPTIVE_MAX_STEPS = 10000

# CommanderBdfDefaultFreezeV2's allowlist: the ten electronic envelopes of V1 by scope digest, derived
# from the frozen ReadinessV1 Cases.json and initial artifacts (the CND2 key table, Cnd2ScopeKeyTableV1).
ALLOWLIST: Mapping[str, str] = MappingProxyType({
    "0ac50dbd36e58217773463a00250520d17aed4e0ba27058b2d0d4019a0a08cd4": "E_dark_hold",
    "8a9602518d2ebdc3f12d13f627b6c4561c5ac8748d8391355ea2314cde84b588": "E_dark_voltage_ramp",
    "67ab993b820dc4e8f30dad963354af04ab3440b164926e92177418d7bd56f7a6": "E_light_hold",
    "cd1e484022fc874cc405043a056d89f59fdce6c8de594fdb1d52dc9a5f2e7395": "E_light_ramp_16",
    "b8deddb08c4e9a836930ffacf8587a43e31d6057fe7c089c1a23a6795469361b": "E_light_ramp_4",
    "6697fade77e85c744e1f37cf119e5f72f81066b7c03fe1e1f3f9a806559e0d54": "E_light_ramp_8",
    "d37348bef4d23387aff7bf769e65908c18c8bd77c0c579c2228b19751c196e99": "E_saved_packet",
    "b4d010a6708ab44e74dd63af184647f59204c9793618cd06c147f83d9a782beb": "E_voltage_ramp_16",
    "fe20d57689805abedd663af99790ed24f267af9d5ffff0b31d4d1c65574e62b8": "E_voltage_ramp_4",
    "858a988f48596ef322590f62f5f1fd6eb3d48a1e516c6c5543cfda1277a13965": "E_voltage_ramp_8",
})

_SCOPE: ContextVar[str | None] = ContextVar("solarlab_time_method", default=None)
_R = TypeVar("_R")


def _checked(name: object) -> str:
    if not isinstance(name, str) or name not in TIME_METHODS:
        raise ValueError(f"unknown time method {name!r}; expected one of {TIME_METHODS}")
    return name


def _route_today() -> date:
    return datetime.now(timezone(timedelta(hours=8))).date()


def _check_default_switch() -> None:
    if type(_SWITCH_DATE) is not str:
        raise ValueError("BDF is not selectable: the allowlisted default has no commander switch date")
    try:
        chosen = date.fromisoformat(_SWITCH_DATE)
    except ValueError as exc:
        raise ValueError("BDF is not selectable: the commander switch date is invalid") from exc
    if chosen.isoformat() != _SWITCH_DATE or chosen < date(2026, 11, 8):
        raise ValueError("BDF is not selectable: the commander switch date must be on or after 2026-11-08")
    if _route_today() < chosen:
        raise ValueError("BDF is not selectable: the allowlisted default starts on " + _SWITCH_DATE + " Asia/Shanghai")


@contextmanager
def time_method(name: str) -> Iterator[None]:
    """Request ``name`` for the transient entry points called inside this block."""
    token = _SCOPE.set(_checked(name))
    try:
        yield
    finally:
        _SCOPE.reset(token)


def requested_time_method(method: str | None = None) -> tuple[str | None, str]:
    """The explicit request and its source; ``(None, "default")`` takes the scope-based default."""
    if method is not None:
        return _checked(method), "argument"
    scoped = _SCOPE.get()
    if scoped is not None:
        return scoped, "scope"
    declared = os.environ.get(ENVIRONMENT)
    if declared is not None:
        return _checked(declared), "environment"
    return None, "default"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


# Registered with the frozen allowlist; the CND2 key table lists every case it holds.
ALLOWLIST_SHA256 = _sha256(dict(ALLOWLIST))


@dataclass(frozen=True)
class TimeMethodSelection:
    """One resolved request: the method a history runs, and the scope that admitted BDF.

    A BDF selection without a scope is an explicit research capture of
    supplied options (``source="research_capture"``), never a production
    selection. A selection, and its record, describe a request. Nothing that
    a history integrates is read from them.
    """

    family: str
    requested: str | None
    source: str
    effective: str
    scope: str | None = None
    scope_sha256: str | None = None

    def record(self) -> dict[str, Any]:
        """The provenance record of this selection; a production BDF one adds its scope, profile and work cap."""
        record: dict[str, Any] = {
            "schema": SELECTION_SCHEMA, "family": self.family, "requested": self.requested, "source": self.source,
            "effective": self.effective, "allowlisted_default": list(ALLOWLISTED_DEFAULT),
        }
        if self.effective == BDF:
            production = self.scope is not None
            record.update(scope=self.scope, scope_sha256=self.scope_sha256,
                          allowlist_sha256=ALLOWLIST_SHA256 if production else None,
                          profile_sha256=PROFILE_SHA256 if production else None,
                          profile=dict(PROFILE) if production else None,
                          adaptive_max_steps=ADAPTIVE_MAX_STEPS if production else None,
                          trust_standard=TRUST_STANDARD if production else None,
                          qualification=False, label="experimental_unqualified")
        return record


def _genuine(value: object, qualname: str) -> bool:
    """Whether ``value`` is exactly the module-level solarlab class ``qualname``: no subclass, no look-alike."""
    cls = type(value)
    module = sys.modules.get(cls.__module__)
    return (cls.__qualname__ == qualname and cls.__module__.startswith("solarlab.") and module is not None
            and getattr(module, qualname, None) is cls)


def _family(compiled: object) -> str | None:
    """The entry-point family of the actual compiled device, by its exact class."""
    for family, qualname in ((ELECTRONIC, "CompiledDevice1D"), (IONIC, "CompiledIonicDevice1D"),
                             (MULTIPLE_IONIC, "CompiledMultipleIonicDevice1D")):
        if _genuine(compiled, qualname):
            return family
    return None


def _canonical_drive(protocol: Any) -> Any:
    """The protocol's own continuous drive, built by the genuine constructor from its knots."""
    from solarlab.solvers.time.bdf_1d import BDFDrive

    values = np.column_stack((protocol.voltage_V, protocol.generation_scale))
    return BDFDrive(protocol.times_s, values, values)


def _probe_drive_constructor(record: tuple[Any, ...]) -> None:
    """Refuse a replaced ``BDFDrive.from_protocol`` (D1); it sees a throwaway build and never builds a run's drive."""
    from solarlab.solvers.time.bdf_1d import BDFDrive

    refusal = "BDF is not selectable: the protocol's drive is not its own continuous knots"
    try:
        drive = BDFDrive.from_protocol(_build(record))
    except Exception as exc:
        raise ValueError(refusal) from exc
    if not _continuous(drive, _build(record)):
        raise ValueError(refusal)


def _continuous(drive: object, protocol: Any) -> bool:
    """A drive that is exactly a BDFDrive with the protocol's knots on both sides of every knot."""
    if not _genuine(drive, "BDFDrive"):
        return False
    try:
        return _same(drive, _canonical_drive(protocol))
    except (TypeError, ValueError):
        return False


_OPERATORS = {ELECTRONIC: "BDFStep:DC1DProblem", IONIC: "BDFStep:Ionic1DProblem"}
_UNSUPPORTED = {MULTIPLE_IONIC: "BDF is not selectable for multiple-domain ionic histories: this entry point has no "
                                "BDF adapter; request backward_euler"}


class AdmittedBDFCall(NamedTuple):
    """One production BDF call as its admission keyed it, as immutable values only (D1b, D1-R1).

    ``inputs`` holds the records (``_record``) of the compiled device, the
    initial state, the protocol, the physical controls and the initial
    controls, in that order (a record of ``None`` for an ionic call's
    initial controls).
    A record is a tuple of the declared words of one input, as bytes and
    Python scalars, with the solarlab classes and enum members it names.
    ``scope_sha256`` is the admitted key, and ``case`` the allowlisted
    envelope.

    No field can be rebound, and no object of a history is reachable from an
    admission: :func:`integrate_admitted` builds the history's inputs from
    the records when it runs. Both it and :func:`verify_integration` take
    exactly this class, and read an admission through ``tuple``'s own
    accessors, never its attributes.
    """

    family: str
    case: str
    scope_sha256: str
    inputs: tuple[Any, ...]

    def __repr__(self) -> str:
        values = tuple.__getitem__(self, slice(None))
        return f"AdmittedBDFCall(family={values[0]!r}, case={values[1]!r}, scope_sha256={values[2]!r}, inputs=<records>)"


def _handle(admission: Any) -> tuple[str, str, str, tuple[Any, ...]]:
    """An admission's values, read through ``tuple``'s own accessors: its attributes are never read."""
    refusal = "BDF is not selectable: this is not an admission of a production BDF call"
    if type(admission) is not AdmittedBDFCall:
        raise ValueError(refusal)
    values = tuple.__getitem__(admission, slice(None))
    if len(values) != 4:
        raise ValueError(refusal)
    family, case, digest, inputs = values
    if (type(family) is not str or family not in _OPERATORS or type(case) is not str or type(digest) is not str
            or type(inputs) is not tuple or len(inputs) != 5):
        raise ValueError(refusal)
    return family, case, digest, inputs


def _scope_of(family: str, compiled: Any, initial: Any, protocol: Any, options: Any, initial_options: Any,
              drive: Any, rederive: bool = True) -> str:
    """The scope digest of one call's inputs, its drive included.

    ``rederive=False`` keys the inputs that a production history integrates,
    which this module built from records itself. It gives the same digest. It
    skips only ``_declared_only``, which refuses extra state and stale derived
    fields without changing any key byte. ``_declared_only`` rebuilds a value
    with ``dataclasses.replace``, whose result shares the value's fields: it
    would hand those inputs on to code that observes return values.
    """
    if rederive:
        return (electronic_scope(compiled, initial, protocol, options=options, initial_options=initial_options,
                                 drive=drive) if family == ELECTRONIC
                else ionic_scope(compiled, initial, protocol, options=options, drive=drive))
    if family == ELECTRONIC:
        return scope_digest(ELECTRONIC, _electronic_components(compiled, initial, protocol, options, initial_options,
                                                               drive, False))
    return scope_digest(IONIC, _ionic_components(compiled, initial, protocol, options, drive, False))


def _built(inputs: tuple[Any, ...]) -> tuple[Any, ...]:
    """The objects that records name, built anew, and the protocol's own continuous drive (never integrated)."""
    compiled, initial, protocol, options, initial_options = [_build(node) for node in inputs]
    return compiled, initial, protocol, options, initial_options, _canonical_drive(protocol)


def integrate_admitted(admission: AdmittedBDFCall, adapter: Callable[..., _R]) -> _R:
    """Start one admitted production BDF history, the only way a production route starts one (D1b, D1-R1).

    The history's inputs are built here from the admission's records, after
    admission has returned. They are keyed again and must be the admitted,
    allowlisted scope. ``adapter`` is handed exactly these inputs, the
    protocol's own continuous drive, the frozen profile, the forwarded work
    cap and the admission, which it re-keys before the integrator
    (:func:`verify_integration`). Any divergence is refused, with no fallback.
    """
    from solarlab.solvers.time.bdf import BDFOptions
    from solarlab.solvers.time.bdf_1d import BDFDrive

    family, case, digest, inputs = _handle(admission)
    # The inputs are built in place, never returned by a function of this module (module docstring).
    built: list[Any] = []
    try:
        for node in inputs:
            _build_into(node, built)
        compiled, initial, protocol, options, initial_options = built
        knots = np.column_stack((protocol.voltage_V, protocol.generation_scale))
        drive = BDFDrive(protocol.times_s, knots, knots)
        key = _scope_of(family, compiled, initial, protocol, options, initial_options, drive, rederive=False)
    except Exception as exc:
        raise ValueError(f"BDF is not selectable: the admitted inputs cannot be built and keyed ({exc})") from exc
    if key != digest or ALLOWLIST.get(key) != case:
        raise ValueError(f"BDF is not selectable: the inputs built for integration are not the admitted, allowlisted "
                         f"ones (admitted {digest[:12]}, built {key[:12]})")
    controls = {"initial_options": initial_options} if family == ELECTRONIC else {}
    return adapter(compiled, initial, protocol, options=options, bdf_options=BDFOptions(**PROFILE), drive=drive,
                   adaptive_max_steps=ADAPTIVE_MAX_STEPS, admission=admission, **controls)


def verify_integration(
    admission: AdmittedBDFCall, *, compiled: Any, initial: Any, protocol: Any, options: Any, drive: Any, start: Any,
    run_options: Any, initial_options: Any = None,
) -> None:
    """A BDF adapter's last step before the integrator: refuse unless it hands over the admitted inputs.

    The adapter has computed the integrator's start, interval, options,
    breakpoints and events before this call, and calls the integrator right
    after it. ``start`` is the first state vector and must hold the
    initial's words.
    ``run_options`` are the integrator's options: the frozen profile with
    the forwarded work cap and the remaining wall budget. The device,
    initial, protocol, controls and drive whose objects the integrator reads
    are keyed again, with declared and re-derived state checked, and must
    be the admitted, allowlisted scope.
    """
    from solarlab.solvers.time.bdf import BDFOptions

    family, case, digest, _ = _handle(admission)
    try:
        wall = run_options.wall_s
        profile = (type(run_options) is BDFOptions and type(run_options.max_steps) is int
                   and run_options.max_steps == ADAPTIVE_MAX_STEPS and type(wall) is float
                   and 0.0 < wall <= PROFILE["wall_s"] and _same_profile(replace(run_options, wall_s=PROFILE["wall_s"])))
    except (AttributeError, TypeError, ValueError):
        profile = False
    if not profile or type(start) is not np.ndarray or _words(start) != _words(initial.state.u):
        raise ValueError("BDF is not selectable: the integrator's start, profile or work cap is not the admitted one")
    try:
        handed = _scope_of(family, compiled, initial, protocol, options, initial_options, drive)
    except Exception as exc:
        raise ValueError(f"BDF is not selectable: the inputs handed to the integrator cannot be keyed ({exc})") from exc
    if handed != digest or ALLOWLIST.get(handed) != case:
        raise ValueError(f"BDF is not selectable: the inputs handed to the integrator are not the admitted ones "
                         f"(admitted {digest[:12]}, handed {handed[:12]})")


def resolve_time_method(
    method: str | None = None, *, compiled: Any, initial: Any = None, protocol: Any = None, options: Any = None,
    initial_options: Any = None, bdf_options: Any = None, bdf_drive: Any = None, family: str | None = None,
) -> TimeMethodSelection:
    """Resolve one production call's request from its actual inputs, before numerical work.

    This is :func:`admit_time_method`'s selection: the selection a call's
    record states.
    """
    return admit_time_method(method, compiled=compiled, initial=initial, protocol=protocol, options=options,
                             initial_options=initial_options, bdf_options=bdf_options, bdf_drive=bdf_drive,
                             family=family)[0]


def admit_time_method(
    method: str | None = None, *, compiled: Any, initial: Any = None, protocol: Any = None, options: Any = None,
    initial_options: Any = None, bdf_options: Any = None, bdf_drive: Any = None, family: str | None = None,
) -> tuple[TimeMethodSelection, AdmittedBDFCall | None]:
    """Admit one production call before numerical work: its selection and, for BDF, the inputs BDF integrates.

    The family comes from the exact class of ``compiled``. ``family`` names
    the entry point's own family, and a device of another family is refused.

    The scope digest is computed here from the actual device, initial state,
    protocol and controls, never supplied. An explicit ``bdf`` request runs
    exactly ``PROFILE`` with the protocol's own continuous drive.
    BDFOptions or a drive must be exactly those, and are refused otherwise.
    Under the scope-based default neither may be supplied, and only an
    allowlisted scope can select BDF, after the flip. Every other BDF request
    is refused, with no fallback.

    A BDF selection returns its :class:`AdmittedBDFCall`:
    - Admission records the caller's inputs as immutable values. It builds
      objects from the records, with the protocol's own drive, and keys them.
    - That key must equal the key of the caller's objects. A divergence means
      an input changed while it was admitted, and is refused.
    - No caller-owned object is read after it is recorded. The admission
      holds only the records and the key.
    - The entry point integrates it with :func:`integrate_admitted`, which
      builds the history's inputs from the records.
    """
    requested, source = requested_time_method(method)
    if family is not None and family not in _FAMILIES:
        raise ValueError(f"unknown time-method family {family!r}")
    if requested != BDF and (bdf_options is not None or bdf_drive is not None):
        raise ValueError("BDF options or a BDF drive require an explicit bdf time-method request")
    if requested == BDF and family in _UNSUPPORTED:
        raise ValueError(_UNSUPPORTED[family])
    actual = _family(compiled)
    label = actual or family or "unsupported"
    if requested == BACKWARD_EULER:
        return TimeMethodSelection(label, requested, source, BACKWARD_EULER), None

    def unselectable(reason: str) -> tuple[TimeMethodSelection, None]:
        # The default keeps backward Euler wherever BDF is not selectable; a bdf request is refused.
        if requested == BDF:
            raise ValueError(reason)
        return TimeMethodSelection(label, None, source, BACKWARD_EULER), None

    if actual is None or family not in (None, actual):
        return unselectable("BDF is not selectable: the compiled device is not of "
                            + (f"the {family} family" if family else "a supported family"))
    if requested is None and (ALLOWLISTED_DEFAULT[0] == BACKWARD_EULER or actual in _UNSUPPORTED):
        return TimeMethodSelection(actual, None, source, BACKWARD_EULER), None
    from solarlab.equations.implementation import selected_implementation

    if requested == BDF:
        if actual in _UNSUPPORTED:
            raise ValueError(_UNSUPPORTED[actual])
        selected_implementation(operator=_OPERATORS[actual])
        if bdf_options is not None and not _same_profile(bdf_options):
            raise ValueError(f"production BDF runs only the frozen profile {PROFILE_SHA256[:8]}: request it without "
                             "BDFOptions or with exactly that profile")
    if not _genuine(protocol, "TransientProtocol"):
        return unselectable("BDF is not selectable: the protocol is not a TransientProtocol")
    if bdf_drive is not None and not _continuous(bdf_drive, protocol):
        raise ValueError("production BDF runs only the protocol's own continuous drive: request it without a drive "
                         "or with exactly BDFDrive.from_protocol(protocol)")
    try:
        digest = (electronic_scope(compiled, initial, protocol, options=options, initial_options=initial_options)
                  if actual == ELECTRONIC else ionic_scope(compiled, initial, protocol, options=options))
    except Exception as exc:  # no scope: the default keeps backward Euler, whatever the call's inputs raise
        return unselectable(f"BDF is not selectable: this call's scope cannot be computed ({exc})")
    case = ALLOWLIST.get(digest)
    if case is None:
        return unselectable(f"BDF is not selectable for scope {digest[:12]}: only the {len(ALLOWLIST)} allowlisted "
                            "electronic envelopes run BDF; request backward_euler")
    # The scope selects BDF. From here every failure is a refusal under either request, never run as BE.
    if requested is None:
        _check_default_switch()
    try:
        inputs = tuple(_record(value) for value in (
            compiled, initial, protocol, options, initial_options if actual == ELECTRONIC else None))
        recorded = _scope_of(actual, *_built(inputs))
    except Exception as exc:
        raise ValueError(f"BDF is not selectable: this call's inputs cannot be admitted ({exc})") from exc
    if recorded != digest:
        raise ValueError(f"BDF is not selectable: this call's inputs changed while they were admitted (keyed "
                         f"{digest[:12]}, recorded {recorded[:12]})")
    _probe_drive_constructor(inputs[2])
    if requested is None:
        selected_implementation(operator=_OPERATORS[actual])
    return TimeMethodSelection(actual, requested, source, BDF, case, digest), AdmittedBDFCall(actual, case, digest,
                                                                                              inputs)


def _same_profile(options: Any) -> bool:
    """Whether ``options`` is exactly a BDFOptions of the frozen profile, by its scope content."""
    from solarlab.solvers.time.bdf import BDFOptions

    try:
        return _same(options, BDFOptions(**PROFILE))
    except TypeError:
        return False


_WORKER_REFUSAL = ("BDF is not selectable for process or HTTP worker jobs: no worker scope is allowlisted "
                   "(a worker's initial state is a cold recipe); request backward_euler")


def _worker_selection(family: str) -> TimeMethodSelection:
    """A worker job's selection: backward Euler, or a refusal of ``bdf``; no worker scope is allowlisted."""
    if family not in _FAMILIES:
        raise ValueError(f"unknown time-method family {family!r}")
    requested, source = requested_time_method()
    if requested == BDF:
        raise ValueError(_WORKER_REFUSAL)
    return TimeMethodSelection(family, requested, source, BACKWARD_EULER)


def time_method_setting(family: str) -> tuple[str, str | None]:
    """The parent's request as the child setting a worker's execution manifest records.

    The setting is unset (``None``) for the default and ``backward_euler``
    for the opt-out. A worker integrates with backward Euler, and a ``bdf``
    request is refused while the job is prepared.
    """
    return ENVIRONMENT, _worker_selection(family).requested


def verify_time_method(settings: Iterable[tuple[str, str | None]], family: str) -> TimeMethodSelection:
    """Refuse a worker whose request is not the one its job's manifest bound; return the worker's selection."""
    declared = dict(settings)
    selection = _worker_selection(family)
    if ENVIRONMENT not in declared or selection.requested != declared[ENVIRONMENT]:
        raise ValueError("the worker's time-method selection differs from its execution manifest")
    return selection


def _words(value: np.ndarray) -> dict[str, Any]:
    """An array's dtype, shape and C-order element bytes; a 0-d array has none (V1 gave it shape [1])."""
    if type(value) is not np.ndarray:
        raise TypeError("a scope array must be exactly a numpy ndarray")
    if value.ndim == 0:
        raise TypeError("a scope array has at least one dimension")
    if value.dtype.hasobject or value.dtype.fields is not None or value.dtype.kind not in "biuf":
        raise TypeError("scope arrays hold plain numerical words")
    return {"dtype": value.dtype.str, "shape": list(value.shape),
            "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}


# Each identity digests a prepared device, whose catalog evidence names the configuration path.
_PATH_BOUND = {"CompiledDevice1D": ("sha256",), "CompiledIonicDevice1D": ("sha256", "original_prepared_sha256")}
_MANIFEST_PATH_BOUND = {"electronic_component": ("compiled_sha256", "prepared_sha256")}


def _manifest(text: str) -> Any:
    """An ionic binding manifest with its path-bound identities set aside; each must be present (Grammar V2, R1)."""
    if type(text) is not str:
        raise TypeError("an ionic binding manifest is a JSON text")
    try:
        manifest = json.loads(text)
        manifest.pop("original_prepared_sha256")
        for section, names in _MANIFEST_PATH_BOUND.items():
            identities = manifest[section]
            if type(identities) is not dict or not set(names) <= set(identities):
                raise KeyError(section)
            manifest[section] = {key: value for key, value in identities.items() if key not in names}
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise TypeError("an ionic binding manifest must name its path-bound identities") from exc
    return _content(_tuples(manifest))


def _tuples(value: Any) -> Any:
    """Parsed JSON with its arrays as tuples: JSON has one sequence kind, encoded as every sequence is."""
    if type(value) is list:
        return tuple(_tuples(item) for item in value)
    if type(value) is dict:
        return {key: _tuples(item) for key, item in value.items()}
    return value


def _declared_only(value: Any) -> None:
    """Refuse a dataclass whose state is not its declared fields: no other attribute, each derived field re-derived."""
    declared = [item.name for item in fields(value)]
    if set(getattr(value, "__dict__", ())) - set(declared):
        raise TypeError(f"{type(value).__qualname__} carries state beyond its declared fields")
    derived = [item.name for item in fields(value) if not item.init]
    if derived:
        try:
            rebuilt = replace(value)
        except Exception as exc:  # whatever its own class refuses to rebuild, it did not construct
            raise TypeError(f"{type(value).__qualname__} is not constructed from its declared fields") from exc
        for name in derived:
            if not _same(getattr(value, name), getattr(rebuilt, name)):
                raise TypeError(f"{type(value).__qualname__}.{name} is not derived from its declared fields")


def _content(value: Any, compare: bool = False, rederive: bool = True) -> Any:
    """Every declared word of ``value``, typed; a value of any other kind is refused, never skipped.

    Each value must be exactly its type. A subclass or look-alike could act
    differently on equal words, so it has no scope. ``compare`` encodes a
    derived field for comparison only, never for a key: it also accepts the
    mapping proxies some derived fields hold. ``rederive=False`` skips
    ``_declared_only`` (``_scope_of``), which never changes the content.
    """
    kind = type(value)
    if is_dataclass(value) and not isinstance(value, type):
        name = kind.__qualname__
        if not _genuine(value, name):
            raise TypeError(f"{name} is not a module-level solarlab class")
        if not kind.__dataclass_params__.frozen:  # type: ignore[attr-defined]
            raise TypeError(f"{name} is not a frozen dataclass")
        # Fields derived in __post_init__ repeat the declared ones, and must be their re-derivation.
        if rederive:
            _declared_only(value)
        skip = _PATH_BOUND.get(name, ())
        return {"type": name, "fields": {
            item.name: _manifest(getattr(value, item.name)) if item.name == "binding_manifest_json"
            else _content(getattr(value, item.name), compare, rederive) for item in fields(value)
            if item.init and item.name not in skip}}
    if kind is np.ndarray:
        return {"array": _words(value)}
    if isinstance(value, enum.Enum):
        if not _genuine(value, kind.__qualname__):
            raise TypeError(f"{kind.__qualname__} is not a module-level solarlab enum")
        return {"enum": kind.__qualname__, "value": _content(value.value, compare, rederive)}
    if kind is bool or kind is int:
        return value
    if kind is float:
        # A non-finite scalar has no single word in float.hex(); arrays keep their raw words.
        if not math.isfinite(value):
            raise TypeError("a scope scalar must be finite")
        return {"float": value.hex()}
    if value is None or kind is str:
        return value
    if kind is tuple:
        return [_content(item, compare, rederive) for item in value]
    if (kind is dict or (compare and kind is MappingProxyType)) and all(type(key) is str for key in value):
        pairs = [[key, _content(value[key], compare, rederive)] for key in sorted(value)]
        return {"mapping": pairs} if kind is dict else {"mapping_proxy": pairs}
    raise TypeError(f"no scope content is defined for {kind.__module__}.{kind.__qualname__}")


def _record(value: Any) -> tuple[Any, ...]:
    """One input as an immutable record of its declared words, from which a history's inputs are built (D1b, D1-R1).

    A record is a tuple of one of these kinds:
    - a frozen solarlab dataclass: its class and its init fields in order;
    - an array: its dtype, shape and C-order bytes;
    - a tuple, or a dict with str keys: its items;
    - an immutable scalar or a genuine solarlab enum member.

    It admits what the scope grammar admits, with the same exact-type rules,
    and refuses everything else.
    """
    kind = type(value)
    if is_dataclass(value) and not isinstance(value, type):
        if not (_genuine(value, kind.__qualname__) and kind.__dataclass_params__.frozen):  # type: ignore[attr-defined]
            raise TypeError(f"{kind.__qualname__} is not a frozen module-level solarlab class")
        _declared_only(value)
        return ("dataclass", kind, tuple((item.name, _record(getattr(value, item.name)))
                                         for item in fields(value) if item.init))
    if kind is np.ndarray:
        if value.dtype.hasobject or value.dtype.fields is not None or value.dtype.kind not in "biuf":
            raise TypeError("scope arrays hold plain numerical words")
        return ("array", value.dtype.str, tuple(value.shape), value.tobytes(order="C"))
    if kind is tuple:
        return ("tuple", tuple(_record(item) for item in value))
    if kind is dict and all(type(key) is str for key in value):
        return ("dict", tuple((key, _record(item)) for key, item in value.items()))
    if value is None or kind in (bool, int, float, str) or (
            isinstance(value, enum.Enum) and _genuine(value, kind.__qualname__)):
        return ("value", value)
    raise TypeError(f"no scope content is defined for {kind.__module__}.{kind.__qualname__}")


def _build(node: Any) -> Any:
    """The object a record names, built anew (``_build_into``); for admission's own checks, never integrated."""
    built: list[Any] = []
    _build_into(node, built)
    return built[0]


def _build_into(node: Any, built: list[Any]) -> None:
    """Build the object a record names, by its own constructors over the record's immutable words, onto ``built``.

    A dataclass's derived fields are derived again. An array is a read-only
    view of the record's immutable bytes. Nothing is returned, so no function
    of this module returns the inputs a production history integrates
    (:func:`integrate_admitted`).
    """
    if type(node) is not tuple or not node or type(node[0]) is not str:
        raise TypeError("a record is a tagged tuple")
    if node[0] == "dataclass":
        _, kind, items = node
        module = sys.modules.get(getattr(kind, "__module__", ""))
        if not (isinstance(kind, type) and is_dataclass(kind) and kind.__module__.startswith("solarlab.")
                and module is not None and getattr(module, kind.__qualname__, None) is kind):
            raise TypeError("a record names a module-level solarlab class")
        arguments = {}
        for name, item in items:
            _build_into(item, built)
            arguments[name] = built.pop()
        built.append(cast(Any, kind)(**arguments))
    elif node[0] == "array":
        _, dtype, shape, words = node
        built.append(np.frombuffer(words, dtype=np.dtype(dtype)).reshape(shape))
    elif node[0] in ("tuple", "dict"):
        values = []
        for item in node[1]:
            _build_into(item if node[0] == "tuple" else item[1], built)
            values.append(built.pop())
        built.append(tuple(values) if node[0] == "tuple" else dict(zip([key for key, _ in node[1]], values)))
    elif node[0] == "value":
        built.append(node[1])
    else:
        raise TypeError(f"no record kind {node[0]!r}")


def _same(first: Any, second: Any) -> bool:
    """Equal canonical bytes of scope content (``True`` and ``1`` differ, as do ``0.0`` and ``-0.0``)."""
    return _canonical(_content(first, True)) == _canonical(_content(second, True))


def scope_digest(family: str, components: Mapping[str, Any]) -> str:
    """The digest of each component's canonical content, with the family."""
    return _sha256({"schema": SCOPE_SCHEMA, "family": family,
                    "components": {name: _sha256(value) for name, value in components.items()}})


def _source(protocol: Any, drive: Any, rederive: bool = True) -> dict[str, Any]:
    """The source knots, and the drive: absent, or exactly the protocol's own continuous BDFDrive.

    Any other drive has no scope: a jump drive, and a BDFDrive subclass or
    look-alike with equal arrays, whose waveform its words do not bind.
    """
    if not _genuine(protocol, "TransientProtocol"):
        raise TypeError("a scope protocol must be a TransientProtocol")
    if rederive:
        _declared_only(protocol)
    if drive is not None and not _continuous(drive, protocol):
        raise TypeError("a scope drive must be exactly the protocol's own continuous BDFDrive")
    return {"protocol": {name: _words(getattr(protocol, name)) for name in ("times_s", "voltage_V", "generation_scale")},
            "drive": "continuous_protocol_knots"}


def _electronic_state(state: Any) -> dict[str, Any]:
    return {name: _words(getattr(state, name)) for name in ("u", "potential_V", "electron_density_m3", "hole_density_m3")}


def electronic_components(compiled: Any, initial: Any, protocol: Any, *, options: Any, initial_options: Any,
                          drive: Any = None) -> dict[str, Any]:
    """The scope components of one ``solve_transient`` call, as the CND2 key table lists them.

    The initial enters by its state words, voltage and generation.
    """
    return _electronic_components(compiled, initial, protocol, options, initial_options, drive, True)


def _electronic_components(compiled: Any, initial: Any, protocol: Any, options: Any, initial_options: Any, drive: Any,
                           rederive: bool) -> dict[str, Any]:
    if not (_genuine(initial, "SteadyState1D") and _genuine(initial.state, "State1D")):
        raise TypeError("a scope initial must be a SteadyState1D with its State1D")
    if rederive:
        _declared_only(initial)
        _declared_only(initial.state)
    return {
        "device": _content(compiled, rederive=rederive),
        "initial_state": {**_electronic_state(initial.state), "voltage_V": _content(initial.voltage_V),
                          "generation_scale": _content(initial.generation_scale)},
        **_source(protocol, drive, rederive), "physical_controls": _content(options, rederive=rederive),
        "initial_controls": _content(initial_options, rederive=rederive),
    }


def electronic_scope(compiled: Any, initial: Any, protocol: Any, *, options: Any, initial_options: Any,
                     drive: Any = None) -> str:
    """The scope digest of one ``solve_transient`` call."""
    return scope_digest(ELECTRONIC, electronic_components(compiled, initial, protocol, options=options,
                                                          initial_options=initial_options, drive=drive))


def ionic_scope(compiled: Any, initial: Any, protocol: Any, *, options: Any, drive: Any = None) -> str:
    """The scope digest of one ``solve_ionic_transient`` call from its pinned steady initial.

    The initial enters by its state words, generation and steady controls. Its
    evidence bytes and path name the path-bound compiled identity.
    """
    return scope_digest(IONIC, _ionic_components(compiled, initial, protocol, options, drive, True))


def _ionic_components(compiled: Any, initial: Any, protocol: Any, options: Any, drive: Any,
                      rederive: bool) -> dict[str, Any]:
    if not (_genuine(initial, "IonicTransientInitial") and _genuine(initial.state, "IonicState1D")
            and _genuine(initial.state.electronic_state, "State1D")):
        raise TypeError("a scope initial must be an IonicTransientInitial with its IonicState1D")
    if rederive:
        for part in (initial, initial.state, initial.state.electronic_state):
            _declared_only(part)
    state = initial.state
    return {
        "device": _content(compiled, rederive=rederive),
        "initial_state": {**{name: _words(getattr(state, name))
                             for name in ("u", "positive_ion_density_m3", "negative_ion_density_m3")},
                          "electronic_state": _electronic_state(state.electronic_state),
                          "domain_inventory_m2": _content(state.domain_inventory_m2),
                          "negative_domain_inventory_m2": _content(state.negative_domain_inventory_m2),
                          "voltage_V": _content(state.voltage_V), "generation_scale": _content(initial.generation_scale)},
        **_source(protocol, drive, rederive), "physical_controls": _content(options, rederive=rederive),
        "initial_controls": _content(initial.steady_controls, rederive=rederive),
    }
