"""The reviewed equation-binding specification and the one mask it permits (PW-C1, PW-C5).

A legacy and an assembled run of one job admit the same Q6 record except its
``assembler`` binding; the envelope digest covers that binding too. ``split``
verifies an envelope, then separates exactly that binding from the rest of
the record, which stays compared. ``expected`` states the binding from this
specification and Core's binding functions, independently of the production
helper that writes it.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
import copy
from typing import Any
from unittest import mock

from solarlab.equations.assembly.classical_1d import classical_dc_1d
from solarlab.equations.dc_1d import CompiledDevice1D
from solarlab.reproducibility.job_provenance import digest, read_provenance

SCHEMA = "solarlab.equation-binding.v1"
# Original operator label: analysis kind, assembled operator, Assemblers evaluated besides the system.
OPERATORS: dict[str, tuple[str, str, tuple[str, ...]]] = {
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
BINDING_KEYS = frozenset({"operator", "unified_assembler", "identity", "analysis_identity"})


def assemblers(device: Any, roles: tuple[str, ...], closure: int | None) -> dict[str, str]:
    """Each role's Assembler identity, from Core's binding functions."""
    if isinstance(device, CompiledDevice1D):
        return {"system": classical_dc_1d(device).identity}
    from solarlab.equations.dc_2d import CompiledDevice2D

    if isinstance(device, CompiledDevice2D):
        from solarlab.equations.assembly.two_dimensional import classical_dc_2d

        two = {"system": classical_dc_2d(device).identity}
        return {**two, "column": classical_dc_1d(device.column).identity} if "column" in roles else two
    from solarlab.equations.assembly.ionic_1d import ionic_system, multiple_ionic_system
    from solarlab.equations.multiple_ionic_1d import CompiledMultipleIonicDevice1D

    bind = multiple_ionic_system if isinstance(device, CompiledMultipleIonicDevice1D) else ionic_system
    system = bind(device)
    result = {"system": system.identity}
    if "closure" in roles:
        result["closure"] = bind(device, inventory_row_node=closure).identity
    if "electronic" in roles:
        result["electronic"] = classical_dc_1d(device.electronic_component).identity
    if "domains" in roles:
        result.update(("domain:" + domain.domain_layer_id, ionic_system(domain).identity) for domain in device.domains)
    return result


def expected(operator: str, device: Any, implementation: str, closure: int | None = None) -> dict[str, Any]:
    """The Q6 ``assembler`` record the job must admit under ``implementation``."""
    analysis, assembled, roles = OPERATORS[operator]
    label, bound = (operator, {}) if implementation == "legacy" else (assembled, assemblers(device, roles, closure))
    return {"operator": label, "unified_assembler": bool(bound), "identity": bound.get("system"),
            "analysis_identity": digest({"schema": SCHEMA, "implementation": implementation, "analysis": analysis,
                                         "operator": label, "assemblers": bound})}


def split(envelope: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The verified record without its ``assembler`` binding, and that binding."""
    record = read_provenance(envelope).export()
    binding = record.pop("assembler")
    assert set(binding) == BINDING_KEYS
    return record, binding


def allowed(operator: str, device: Any, closure: int | None = None) -> set[str]:
    """Every Assembler identity the assembled analysis may evaluate."""
    return set(assemblers(device, OPERATORS[operator][2], closure).values())


def checked(result: Any, operator: str, device: Any, implementation: str, closure: int | None = None) -> Any:
    """``result`` with its admitted binding asserted, then set aside; the rest of its Q6 record stays compared."""
    record, binding = split(result.job_provenance)
    assert binding == expected(operator, device, implementation, closure), binding
    other = copy.copy(result)
    object.__setattr__(other, "job_provenance", record)
    return other


def checked_document(document: Any, operator: str, device: Any, implementation: str,
                     closure: int | None = None) -> tuple[Any, int]:
    """A saved document with each Q6 envelope's binding asserted and set aside, and how many there were."""
    found = 0

    def walk(value: Any) -> Any:
        nonlocal found
        if isinstance(value, dict):
            if set(value) == {"sha256", "record"} and isinstance(value["record"], dict) \
                    and value["record"].get("schema") == "solarlab.job-provenance.v1":
                record, binding = split(value)
                assert binding == expected(operator, device, implementation, closure), binding
                found += 1
                return record
            return {key: walk(item) for key, item in value.items()}
        if isinstance(value, list):
            return [walk(item) for item in value]
        return value

    return walk(document), found


class Evaluations:
    """Original-class and Assembler evaluations inside one ``counted`` block."""

    def __init__(self) -> None:
        self.original = 0
        self.assembled = 0
        self.identities: set[str] = set()


@contextmanager
def counted() -> Iterator[Evaluations]:
    """Count the original equation residual/Jacobian calls and the Assembler evaluations (with identities)."""
    from solarlab.equations.assembly.assembler import Assembler
    from solarlab.equations.dc_1d import DC1DProblem
    from solarlab.equations.ionic_1d import Ionic1DProblem
    from solarlab.equations.multiple_ionic_1d import MultipleIonic1DProblem

    observed = Evaluations()
    with ExitStack() as stack:
        for cls in (DC1DProblem, Ionic1DProblem, MultipleIonic1DProblem):
            for name in ("residual", "jacobian"):
                def original(self: Any, *args: Any, __method: Any = getattr(cls, name), **kwargs: Any) -> Any:
                    observed.original += 1
                    return __method(self, *args, **kwargs)
                stack.enter_context(mock.patch.object(cls, name, original))
        evaluate = Assembler.evaluate

        def assembled(self: Any, *args: Any, **kwargs: Any) -> Any:
            observed.assembled += 1
            observed.identities.add(self.identity)
            return evaluate(self, *args, **kwargs)

        stack.enter_context(mock.patch.object(Assembler, "evaluate", assembled))
        yield observed


@contextmanager
def one_ulp_residual() -> Iterator[list[int]]:
    """Move every nonzero entry of each assembled residual one ulp toward +inf (assembled side only).

    The original classes never call ``Assembler.evaluate``, so a legacy
    reference is unchanged; yields the count of perturbed systems.
    """
    import dataclasses

    import numpy as np

    from solarlab.equations.assembly.assembler import Assembler

    evaluate = Assembler.evaluate
    perturbed: list[int] = []

    def mutated(self: Any, *args: Any, **kwargs: Any) -> Any:
        system = evaluate(self, *args, **kwargs)
        if not system.admissible:
            return system
        values = np.array(system.F, dtype=float, copy=True)
        nonzero = values != 0.0
        values[nonzero] = np.nextafter(values[nonzero], np.inf)
        perturbed.append(1)
        return dataclasses.replace(system, F=values)

    with mock.patch.object(Assembler, "evaluate", mutated):
        yield perturbed
