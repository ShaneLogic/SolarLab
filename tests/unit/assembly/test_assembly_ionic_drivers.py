"""The ionic drivers on the assembled ionic system, word for word.

The ionic steady drivers (single-domain and per-domain closures with their
row weights, step limits, diagnostics and numerical checks) and the ionic
backward-Euler transient drivers (evidence admission, voltage-lift seeds,
Newton steps, step accounting) run twice: under the legacy opt-out, and
under the assembled production selection. During the assembled run every
evaluation method of the original classes raises, so the second history
comes from the Assembler alone. Every result field must then have the same
binary64 words, except the wall-clock ``wall_s``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack
import dataclasses
import enum
import hashlib
import json
from pathlib import Path
from typing import Any, NoReturn
from unittest import mock

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from examples.research_1d.ionic_steady import controls as steady_controls
from examples.research_1d.ionic_transient import controls as transient_controls, protocol
from examples.research_1d.transient import json_value
import solarlab.experiments.ionic_transient as ionic_transient
import solarlab.experiments.multiple_ionic_transient as multiple_transient
from solarlab.equations.assembly.assembler import Assembler
from solarlab.equations.dc_1d import DC1DProblem
from solarlab.equations.implementation import ASSEMBLED, LEGACY, equation_implementation
from solarlab.equations.ionic_1d import Ionic1DProblem
from solarlab.equations.multiple_ionic_1d import CompiledMultipleIonicDevice1D, MultipleIonic1DProblem
from solarlab.experiments.transient_results import _encode
import solarlab.solvers.nonlinear.ionic_1d as ionic_steady
import solarlab.solvers.nonlinear.multiple_ionic_1d as multiple_steady
from support.assembly_references import REFERENCE_FACTORIES
from support import equation_provenance

SINGLE = ("ionic_constrained:confined", "ionic_constrained:dual", "ionic_constrained:shared",
          "ionic_constrained:immobile", "ionic_constrained:immobile_single")
MULTIPLE = ("multiple_constrained:positive", "multiple_constrained:independent",
            "multiple_constrained:shared", "multiple_constrained:immobile")
WALL_CLOCK = frozenset({"wall_s"})
ORIGINAL_EVALUATIONS = (
    (DC1DProblem, ("residual", "jacobian", "storage_jacobian", "fluxes", "recombination_rate", "_contact_exchange")),
    (Ionic1DProblem, ("residual", "jacobian", "storage_jacobian", "ion_fluxes")),
    (MultipleIonic1DProblem, ("residual", "jacobian", "storage_jacobian", "ion_fluxes", "physical_storage")),
    (ionic_steady.ConstrainedIonic1DProblem, ("residual", "jacobian")),
    (multiple_steady.ConstrainedMultipleIonic1DProblem, ("residual", "jacobian")),
)


@pytest.fixture(scope="module", autouse=True)
def blas1() -> Iterator[None]:
    with threadpool_limits(limits=1):
        yield


class Substituted:
    """The assembled production selection, with original evaluations forbidden."""

    def __init__(self) -> None:
        self.evaluations = 0
        self._stack = ExitStack()

    def __enter__(self) -> Substituted:
        def forbidden(*args: object, **kwargs: object) -> NoReturn:
            raise AssertionError("an original evaluation ran during the substituted history")

        self._stack.enter_context(equation_implementation(ASSEMBLED))
        for cls, methods in ORIGINAL_EVALUATIONS:
            for method in methods:
                self._stack.enter_context(mock.patch.object(cls, method, forbidden))
        original = Assembler.evaluate

        def counted(assembler: Assembler, *args: Any, **kwargs: Any) -> Any:
            self.evaluations += 1
            return original(assembler, *args, **kwargs)

        self._stack.enter_context(mock.patch.object(Assembler, "evaluate", counted))
        return self

    def __exit__(self, *exc: object) -> None:
        self._stack.close()


def words(value: Any) -> Any:
    """Every result leaf; floats and arrays as their binary64 words, wall clock dropped."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return (type(value).__name__, tuple((item.name, words(getattr(value, item.name)))
                                            for item in dataclasses.fields(value) if item.name not in WALL_CLOCK))
    if isinstance(value, dict):
        return ("dict", tuple((key, words(item)) for key, item in value.items() if key not in WALL_CLOCK))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(words(item) for item in value))
    if isinstance(value, np.ndarray):
        return ("array", value.dtype.str, value.shape, np.ascontiguousarray(value).tobytes())
    if isinstance(value, (float, np.floating)):
        return ("float", np.float64(value).tobytes())
    if isinstance(value, enum.Enum):
        return ("enum", type(value).__name__, value.value)
    return ("value", value)


def steady(device: Any) -> Any:
    if isinstance(device, CompiledMultipleIonicDevice1D):
        return multiple_steady.solve_multiple_ionic_steady_state(
            device, voltage_V=0.002, generation_scale=0.0, options=steady_controls(device.domains[0].ion_node_count))
    return ionic_steady.solve_ionic_steady_state(
        device, voltage_V=0.002, generation_scale=0.0, options=steady_controls(device.ion_node_count))


def evidence(device: Any, point: Any, directory: Path) -> tuple[Path, str]:
    """The steady evidence document each transient driver admits."""
    if isinstance(device, CompiledMultipleIonicDevice1D):
        document: Any = _encode({"schema": "solarlab.multiple_ionic_steady.v1", "result": point,
                                 "source_sha256": {}, "qualification": False})
        raw = (json.dumps(document, indent=2, allow_nan=False) + "\n").encode()
    else:
        document = json_value({"summary": {"accepted": point.accepted, "compiled_identity": device.identity,
                                           "voltage_V": 0.002, "generation_scale": 0.0}, "result": point})
        raw = (json.dumps(document, allow_nan=False) + "\n").encode()
    path = directory / "Steady.json"
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def bound(result: Any, device: Any, implementation: str) -> Any:
    """A transient history with its Q6 equation binding asserted and set aside (two-domain results record one)."""
    if not isinstance(device, CompiledMultipleIonicDevice1D):
        return result
    closure = steady_controls(device.domains[0].ion_node_count).inventory_row_node
    return equation_provenance.checked(result, "MultipleIonicBackwardEuler1DStep", device, implementation, closure)


def transient(device: Any, path: Path, digest: str) -> Any:
    if isinstance(device, CompiledMultipleIonicDevice1D):
        controls = steady_controls(device.domains[0].ion_node_count)
        initial = multiple_transient.load_multiple_ionic_steady_initial(device, path, digest, controls=controls)
        return multiple_transient.solve_multiple_ionic_transient(device, initial, protocol(4),
                                                                 options=transient_controls())
    initial = ionic_transient.load_ionic_steady_initial(device, path, digest,
                                                        controls=steady_controls(device.ion_node_count))
    return ionic_transient.solve_ionic_transient(device, initial, protocol(4), options=transient_controls())


@pytest.mark.parametrize("name", SINGLE + MULTIPLE)
def test_steady_and_transient_histories_are_identical(name: str, tmp_path: Path,
                                                     request: pytest.FixtureRequest) -> None:
    device = REFERENCE_FACTORIES[name]().device
    with equation_implementation(LEGACY):
        legacy = steady(device)
    with Substituted() as run:
        candidate = steady(device)
    assert run.evaluations > 0
    assert words(candidate) == words(legacy)
    record: dict[str, Any] = {"factory": name, "steady_accepted": legacy.accepted,
                              "steady_termination": legacy.termination,
                              "steady_iterations": None if legacy.newton is None else len(legacy.newton.iterations),
                              "steady_assembled_evaluations": run.evaluations}
    if legacy.accepted:
        path, digest = evidence(device, legacy, tmp_path)
        with equation_implementation(LEGACY):
            before = transient(device, path, digest)
        with Substituted() as run:
            after = transient(device, path, digest)
        assert run.evaluations > 0
        assert len(before.states) == protocol(4).times_s.size
        assert words(bound(after, device, ASSEMBLED)) == words(bound(before, device, LEGACY))
        record.update(transient_states=len(before.states), transient_assembled_evaluations=run.evaluations,
                      transient_iterations=[None if attempt.newton is None else len(attempt.newton.iterations)
                                            for attempt in before.attempts])
    request.node.user_properties.append(("ionic_driver_parity", record))


def test_a_changed_history_is_detected() -> None:
    """The word comparison distinguishes signed zeros and one-ulp changes."""
    assert words(0.0) != words(-0.0)
    assert words(np.array([1.0])) != words(np.array([np.nextafter(1.0, 2.0)]))
    assert words({"wall_s": 1.0, "value": 2.0}) == words({"wall_s": 3.0, "value": 2.0})
