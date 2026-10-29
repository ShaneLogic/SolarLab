"""Unmodified ionic BE drivers on the assembled physical-storage step adapters.

One current-source accepted steady preparation is shared by each paired
history. The production driver source, waveform and controls stay untouched.
The substituted run forbids every original equation and step evaluation;
all accepted states, Newton records, gates and accounting must be bitwise.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict
import json
import hashlib
import os
from pathlib import Path
import runpy
from typing import Any, NoReturn
from unittest import mock

import pytest
from threadpoolctl import threadpool_limits

from solarlab.equations.multiple_ionic_1d import CompiledMultipleIonicDevice1D
from solarlab.experiments import ionic_transient, multiple_ionic_transient
from solarlab.experiments.transient_results import _decode, _encode
from solarlab.solvers.time.assembled_ionic_backward_euler import (
    AssembledIonicBackwardEuler1DStep, AssembledMultipleIonicBackwardEuler1DStep,
)
from solarlab.solvers.time.ionic_backward_euler import IonicBackwardEuler1DStep
from solarlab.solvers.time.multiple_ionic_backward_euler import MultipleIonicBackwardEuler1DStep
from support.assembly_references import REFERENCE_FACTORIES

FACTORIES = ("ionic_be:confined", "ionic_be:dual", "ionic_be:shared", "ionic_be:immobile",
             "multiple_be:positive", "multiple_be:independent", "multiple_be:shared", "multiple_be:immobile")
ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture(scope="module", autouse=True)
def blas1() -> Iterator[None]:
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(scope="module")
def helpers() -> dict[str, Any]:
    return runpy.run_path(str(ROOT / "tests/unit/assembly/test_assembly_ionic_drivers.py"))


@pytest.mark.parametrize("name", FACTORIES)
def test_ionic_BE_driver_states_histories_gates_and_accounting_are_bitwise(
    name: str, helpers: dict[str, Any], tmp_path: Path, request: pytest.FixtureRequest,
) -> None:
    device = REFERENCE_FACTORIES[name]().device
    admitted_path = os.environ.get("SOLARLAB_IMMOBILE_ASSEMBLY_INITIAL") if name == "multiple_be:immobile" else None
    if admitted_path:
        path = Path(admitted_path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        initial_record = _decode(json.loads(path.read_bytes()))["result"]
        assert initial_record["accepted"] is True and initial_record["compiled_identity"] == device.identity
        assert all(item["passed"] for item in initial_record["checks"])
        preparation_calls = 0
    else:
        initial = helpers["steady"](device)
        initial_record = asdict(initial)
        if not initial.accepted and name == "multiple_be:immobile":
            pytest.xfail("original immobile cold initializer failed; use an explicitly admitted current-source warm artifact")
        assert initial.accepted, (initial.termination, initial.error)
        path, digest = helpers["evidence"](device, initial, tmp_path)
        preparation_calls = 1
    reference = helpers["transient"](device, path, digest)

    def forbidden(*args: Any, **kwargs: Any) -> NoReturn:
        raise AssertionError("an original BE step was evaluated during the assembled-storage history")

    with helpers["Substituted"]() as run, mock.patch.object(
        ionic_transient, "IonicBackwardEuler1DStep", AssembledIonicBackwardEuler1DStep,
    ), mock.patch.object(
        multiple_ionic_transient, "MultipleIonicBackwardEuler1DStep", AssembledMultipleIonicBackwardEuler1DStep,
    ), mock.patch.multiple(
        IonicBackwardEuler1DStep, residual=forbidden, jacobian=forbidden, inventory_rates_m2_s=forbidden,
    ), mock.patch.multiple(
        MultipleIonicBackwardEuler1DStep, residual=forbidden, jacobian=forbidden, inventory_rates_m2_s=forbidden,
    ):
        candidate = helpers["transient"](device, path, digest)
    output = os.environ.get("SOLARLAB_IONIC_ASSEMBLY_EVIDENCE")
    if output:
        saved = Path(output) / (name.replace(":", "_") + ".json")
        with saved.open("x") as stream:
            json.dump(_encode({"factory": name, "steady_initial": initial_record,
                "reference": asdict(reference), "candidate": asdict(candidate),
                "cold_preparation_calls": preparation_calls, "qualification": False}), stream, indent=2, allow_nan=False)
            stream.write("\n")
    assert run.evaluations > 0
    assert helpers["words"](reference) == helpers["words"](candidate)
    assert reference.completed and candidate.completed
    assert len(reference.states) == helpers["protocol"](4).times_s.size
    record = {"factory": name, "multiple_domains": isinstance(device, CompiledMultipleIonicDevice1D),
        "completed": True, "states": len(reference.states), "attempts": len(reference.attempts),
        "steady_initial_iterations": len(initial_record["newton"]["iterations"]),
        "Newton_iterations": [len(item.newton.iterations) for item in reference.attempts],
        "assembled_evaluations": run.evaluations, "all_histories_and_readouts_changed_words": 0,
        "steady_preparation_calls": preparation_calls, "steady_wall_s": initial_record["wall_s"], "reference_wall_s": reference.wall_s,
        "candidate_wall_s": candidate.wall_s, "qualification": False}
    request.node.user_properties.append(("ionic_driver_parity", record))
