"""Local support-package import and a hard budget for the no-solve harness."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
import sys
import time
from typing import NoReturn

import pytest
from threadpoolctl import threadpool_limits

# Avoid depending on an installed package named ``tests`` or changing its
# namespace outside this owned directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@pytest.fixture(scope="session", autouse=True)
def assembly_budget() -> Iterator[list[float]]:
    elapsed = [0.0]
    with threadpool_limits(limits=1):
        yield elapsed
    assert elapsed[0] < 60.0, f"assembly parity self-test exceeded 60 s: {elapsed[0]:.3f} s"


@pytest.fixture(autouse=True)
def assembly_case_time(assembly_budget: list[float]) -> Iterator[None]:
    """Charge only this harness; unrelated tests may follow it in the session."""
    started = time.perf_counter()
    try:
        yield
    finally:
        assembly_budget[0] += time.perf_counter() - started


@pytest.fixture(autouse=True)
def forbid_numerical_solves(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("the live assembly parity harness must not run a solver")

    for name, module in tuple(sys.modules.items()):
        if name.startswith(("solarlab.solvers", "examples.research_")):
            for function in ("solve_newton", "solve_frequency_systems", "solve_steady_state", "solve_ionic_steady_state",
                             "solve_multiple_ionic_steady_state", "solve_steady_state_2d"):
                if hasattr(module, function):
                    monkeypatch.setattr(module, function, forbidden)
