"""Equation implementation of the 1D production entry points (milestone A switch).

The assembled systems of :mod:`solarlab.equations.assembly` are the default
for the steady, transient and small-signal 1D entry points. The original
Problem, closure and small-signal classes stay selectable for one release,
by an explicit opt-out:

* in code: ``with equation_implementation(LEGACY): ...`` (this context and
  everything it calls);
* per process: ``NEWCORE_EQUATIONS=legacy``, read outside any such scope.
  The name avoids the ``SOLARLAB_`` prefix, which the legacy engine refuses
  as a physics override.

A scope is a ContextVar: it holds in this execution context and what it
calls, not in a new thread (unless that thread runs a copied context) or a
spawned process. Worker jobs therefore carry the resolved choice themselves:
preparation records ``NEWCORE_EQUATIONS=<selection>`` among the execution
manifest's environment settings (:func:`selection_setting`), and the child
installs those settings before any engine import, so queued and spawned
execution run the selection of the parent that prepared them.

Both implementations give the same binary64 words for every production
quantity, Newton history and saved array (tests/unit/assembly), so the
selection changes which code runs, never a result. Unknown values are
refused, never mapped to a default.

Each entry point branches on :func:`use_assembled` where it constructs a
problem; the original branch keeps the names it always used. Closures,
domain problems and steppers follow the problem they are given, so one
history never mixes implementations. :func:`implementation_source_modules`
names this module and the assembly modules a producer can run, for source
maps that must not depend on the selection.

Every production job admission (Q6 ``job_provenance``) states the
implementation it runs, before numerical work (:func:`provenance_binding`):
the original classes record the job's original operator label and no
Assembler; the assembled system records the assembled operator and the
``Assembler.identity`` of the device's bound system. Each analysis identity
digests the analysis kind (steady, transient, ac or initialization), the
implementation, the operator and every Assembler the analysis evaluates. A
stage inside an admitted job, such as the initialization of a J-V job,
re-derives the job's binding from its own device and controls
(:func:`substage_binding`), so Q6 admits it only when both run the same
implementation on the same Assemblers.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
import importlib
import os
from typing import TYPE_CHECKING, Any, TypeAlias, TypedDict

from solarlab.equations.assembly.classical_1d import AssembledDC1DProblem, classical_dc_1d
from solarlab.equations.dc_1d import CompiledDevice1D, DC1DProblem
from solarlab.reproducibility.job_provenance import current_provenance, digest

if TYPE_CHECKING:
    from solarlab.equations.assembly.ionic_1d import IonicSystem

__all__ = [
    "ASSEMBLED", "BINDING_SCHEMA", "ENVIRONMENT", "IMPLEMENTATIONS", "LEGACY", "DCProblem", "ProvenanceBinding",
    "equation_implementation", "implementation_source_modules", "observation_binding",
    "provenance_binding", "selected_implementation", "selection_setting", "substage_binding", "use_assembled",
    "verify_selection",
]

ASSEMBLED = "assembled"
LEGACY = "legacy"
IMPLEMENTATIONS = (ASSEMBLED, LEGACY)
ENVIRONMENT = "NEWCORE_EQUATIONS"

DCProblem: TypeAlias = DC1DProblem | AssembledDC1DProblem

_DC = ("solarlab.equations.assembly.assembler", "solarlab.equations.assembly.classical_1d",
       "solarlab.equations.assembly.layout", "solarlab.equations.assembly.quasi_fermi",
       "solarlab.equations.assembly.terms", "solarlab.equations.assembly.weighted")
_IONIC = _DC + ("solarlab.equations.assembly.ionic_1d",)
_SMALL_SIGNAL = _IONIC + ("solarlab.equations.assembly.small_signal",)
_FAMILIES = {"dc": _DC, "ionic": _IONIC, "small_signal": _SMALL_SIGNAL}

_SCOPE: ContextVar[str | None] = ContextVar("solarlab_equation_implementation", default=None)

BINDING_SCHEMA = "solarlab.equation-binding.v1"
# Each production job's original operator label (Q6): its analysis kind, its assembled
# operator and the Assemblers it evaluates besides the device's bound ``system``.
_OPERATORS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "DC1DProblem.steady_jv": ("steady", "AssembledDC1DProblem.steady_jv", ()),
    "BackwardEuler1DStep": ("transient", "BackwardEuler1DStep(AssembledStorage1D)", ()),
    "DC1DProblem.small_signal": ("ac", "AssembledDC1DProblem.small_signal", ()),
    "Ionic1DProblem.steady_jv": ("steady", "AssembledIonic1DProblem.steady_jv", ("closure", "electronic")),
    "Ionic1DProblem.initialization": ("initialization", "AssembledIonic1DProblem.initialization",
                                      ("closure", "electronic")),
    "IonicBackwardEuler1DStep": ("transient", "AssembledIonicBackwardEuler1DStep", ("closure", "electronic")),
    "IonicSmallSignal1D": ("ac", "AssembledIonicSmallSignal1D", ("closure", "electronic")),
    "MultipleIonicBackwardEuler1DStep": ("transient", "AssembledMultipleIonicBackwardEuler1DStep",
                                         ("closure", "electronic")),
    "MultipleIonicSmallSignal1D": ("ac", "AssembledMultipleIonicSmallSignal1D", ("closure", "electronic", "domains")),
    "DC2DProblem.steady_jv": ("steady", "AssembledDC2DProblem.steady_jv", ()),
    "DC2DProblem.steady": ("steady", "AssembledDC2DProblem.steady", ("column",)),
}
_ORIGINAL = {**{name: name for name in _OPERATORS}, **{entry[1]: name for name, entry in _OPERATORS.items()}}


class ProvenanceBinding(TypedDict):
    """The keywords of Q6 ``job_provenance`` (and ``WorkerContext.admit_provenance``) for one job."""

    operator: str
    assembler_identity: str | None
    analysis_identity: str


def _checked(name: object) -> str:
    if not isinstance(name, str) or name not in IMPLEMENTATIONS:
        raise ValueError(f"unknown equation implementation {name!r}; expected one of {IMPLEMENTATIONS}")
    return name


def selected_implementation() -> str:
    """The innermost scope's choice, else the process setting, else the assembled default."""
    scoped = _SCOPE.get()
    if scoped is not None:
        return scoped
    return _checked(os.environ.get(ENVIRONMENT, ASSEMBLED))


def use_assembled() -> bool:
    """Whether an entry point constructs the assembled system now."""
    return selected_implementation() == ASSEMBLED


@contextmanager
def equation_implementation(name: str) -> Iterator[None]:
    """Select ``name`` for the production entry points called inside this block."""
    token = _SCOPE.set(_checked(name))
    try:
        yield
    finally:
        _SCOPE.reset(token)


def implementation_source_modules(family: str) -> tuple[str, ...]:
    """This module and the assembly modules a ``dc``, ``ionic`` or ``small_signal`` producer can run.

    The names do not depend on the selection, so a source map is the same
    under either implementation. Each module is imported, so a pin hashes a
    loaded file.
    """
    names = (__name__,) + _FAMILIES[family]
    for name in names:
        importlib.import_module(name)
    return names


def selection_setting() -> tuple[str, str]:
    """The resolved selection as the child setting a worker's execution manifest records."""
    return ENVIRONMENT, selected_implementation()


def verify_selection(settings: Iterable[tuple[str, str | None]]) -> None:
    """Refuse a process whose resolved selection is not the one its job's manifest bound."""
    declared = dict(settings).get(ENVIRONMENT)
    if declared is None or selected_implementation() != declared:
        raise ValueError("the worker's equation selection differs from its execution manifest")


def _assemblers(device: object, roles: tuple[str, ...], inventory_row_node: int | None) -> dict[str, str]:
    """``Assembler.identity`` of the device's bound system and of each further role an analysis evaluates.

    The identity digests the layout, mesh, field map and ordered Terms, so it
    is the same for every binding production builds for the device: the
    ``system``; its steady inventory/profile ``closure`` at the controls' node;
    the ``electronic`` sub-system of the carrier rows (density preparation and
    electronic observations); the per-domain systems of two-domain AC; and the
    1D ``column`` system a 2D job's seed solve runs.
    """
    if isinstance(device, CompiledDevice1D):
        return {"system": classical_dc_1d(device).identity}
    from solarlab.equations.dc_2d import CompiledDevice2D

    if isinstance(device, CompiledDevice2D):
        from solarlab.equations.assembly.two_dimensional import classical_dc_2d

        result = {"system": classical_dc_2d(device).identity}
        if "column" in roles:
            result["column"] = classical_dc_1d(device.column).identity
        return result
    from solarlab.equations.assembly.ionic_1d import ionic_system, multiple_ionic_system
    from solarlab.equations.ionic_1d import CompiledIonicDevice1D
    from solarlab.equations.multiple_ionic_1d import CompiledMultipleIonicDevice1D

    def bind(node: int | None) -> IonicSystem:
        if isinstance(device, CompiledMultipleIonicDevice1D):
            return multiple_ionic_system(device, inventory_row_node=node)
        if isinstance(device, CompiledIonicDevice1D):
            return ionic_system(device, inventory_row_node=node)
        raise TypeError(f"no assembled equation system is bound for {type(device).__name__}")

    system = bind(None)
    result = {"system": system.identity}
    if "closure" in roles:
        if inventory_row_node is None:
            raise ValueError("the steady closure of an ionic job needs its controls' inventory_row_node")
        result["closure"] = bind(inventory_row_node).identity
    if "electronic" in roles:
        result["electronic"] = system.electronic.identity
    if "domains" in roles:
        result.update(("domain:" + domain.domain_layer_id, ionic_system(domain).identity)
                      for domain in getattr(device, "domains", ()))
    return result


def provenance_binding(operator: str, device: object, *, inventory_row_node: int | None = None,
                       assembled: bool | None = None) -> ProvenanceBinding:
    """Q6 ``job_provenance`` keywords stating the implementation that runs ``operator`` on ``device``.

    ``operator`` is the job's original operator label and ``inventory_row_node``
    the steady closure node of an ionic job's controls. The original classes
    record that label and no Assembler. The assembled system records its
    operator and the identity of the device's bound system; the analysis
    identity digests the analysis kind, implementation, operator and every
    Assembler the analysis evaluates (``BINDING_SCHEMA``), so the two
    bindings are explicit and distinct. A producer that does not follow the
    selection states its implementation with ``assembled``.
    """
    if operator not in _OPERATORS:
        raise ValueError(f"no equation binding is defined for operator {operator!r}")
    analysis, label_assembled, roles = _OPERATORS[operator]
    if use_assembled() if assembled is None else assembled:
        implementation, label = ASSEMBLED, label_assembled
        assemblers = _assemblers(device, roles, inventory_row_node)
    else:
        implementation, label, assemblers = LEGACY, operator, {}
    return ProvenanceBinding(operator=label, assembler_identity=assemblers.get("system"), analysis_identity=digest({
        "schema": BINDING_SCHEMA, "implementation": implementation, "analysis": analysis, "operator": label,
        "assemblers": assemblers}))


def substage_binding(device: object, *, inventory_row_node: int | None = None) -> ProvenanceBinding:
    """The admitted job's binding, re-derived for a stage inside it from the stage's own device and controls.

    Q6 refuses an inherited binding under an assembled job and any
    declaration that differs from the job's, so the stage is admitted only
    when it runs the job's implementation on the same Assemblers.
    """
    active = current_provenance()
    if active is None:
        raise ValueError("a stage binding requires an admitted job")
    operator = active["record"]["assembler"]["operator"]
    if operator not in _ORIGINAL:
        raise ValueError(f"no equation binding is defined for the admitted operator {operator!r}")
    return provenance_binding(_ORIGINAL[operator], device, inventory_row_node=inventory_row_node)


def observation_binding(operator: object, system: object) -> dict[str, Any]:
    """``observe`` keywords recording the solved system's identities (assembled operators only)."""
    from solarlab.equations.assembly.small_signal import (
        AssembledIonicSmallSignal1D, AssembledMultipleIonicSmallSignal1D,
    )

    if isinstance(operator, (AssembledIonicSmallSignal1D, AssembledMultipleIonicSmallSignal1D)):
        return {"system": system}
    return {}
