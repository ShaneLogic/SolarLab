"""Analyses run only against the assembled system, with unchanged results.

The existing steady driver (Poisson-Boltzmann preparation, light and bias
continuation, density-basin seeds, physical gates), unmodified, is run twice:
as shipped, and with the assembled problem substituted for the DC1DProblem
class it constructs. The substitution is test-only, so no source file pinned
by saved artifacts changes. Every accepted state, every Newton iteration
record, every gate value and every current must be identical. The frequency operator J + i omega M and a sparse
solve against it are checked the same way.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_limits

from examples.research_1d.steady import load_device
from solarlab.device.compile_1d import compile_device_1d
from solarlab.equations.assembly.classical_1d import AssembledDC1DProblem, assembled_problem_factory
from solarlab.equations.assembly.quasi_fermi import quasi_fermi_potential_tangents
from solarlab.equations.dc_1d import CompiledDevice1D, DC1DProblem, _quasi_fermi_potential_tangents
from solarlab.solvers.nonlinear import steady_1d
from solarlab.solvers.nonlinear.newton import NewtonResult
from solarlab.solvers.nonlinear.steady_1d import SteadyState1D, solve_equilibrium, solve_steady_state


@pytest.fixture(scope="module", autouse=True)
def blas1() -> Iterator[None]:
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(scope="module")
def device() -> CompiledDevice1D:
    return compile_device_1d(load_device())


@pytest.fixture
def assembled(monkeypatch: pytest.MonkeyPatch, device: CompiledDevice1D) -> Iterator[None]:
    """Substitute the assembled problem for the class the driver constructs."""
    with monkeypatch.context() as patch:
        patch.setattr(steady_1d, "DC1DProblem", assembled_problem_factory(device))
        yield


def assert_same_newton(reference: NewtonResult, candidate: NewtonResult) -> None:
    assert candidate.state.tobytes() == reference.state.tobytes()
    assert candidate.termination == reference.termination and candidate.accepted == reference.accepted
    assert candidate.iterations == reference.iterations
    assert candidate.gates == reference.gates
    assert candidate.scaled_residual == reference.scaled_residual
    assert (candidate.residual_evaluations, candidate.jacobian_evaluations) == (
        reference.residual_evaluations, reference.jacobian_evaluations)
    assert candidate.start_digest == reference.start_digest


def assert_same_point(reference: SteadyState1D, candidate: SteadyState1D) -> None:
    assert_same_newton(reference.newton, candidate.newton)
    assert len(candidate.preparation) == len(reference.preparation)
    for left, right in zip(reference.preparation, candidate.preparation):
        assert_same_newton(left, right)
    assert candidate.state.u.tobytes() == reference.state.u.tobytes()
    for name in ("current_density_A_m2", "contact_current_density_A_m2"):
        assert getattr(candidate, name).tobytes() == getattr(reference, name).tobytes()
    assert candidate.density_basin_stages == reference.density_basin_stages


CASES = [(0.0, 0.0), (0.0, 1.0), (0.6, 1.0), (0.9, 0.0), (-0.2, 1.0)]


@pytest.fixture(scope="module")
def references(device: CompiledDevice1D) -> dict[tuple[float, float], SteadyState1D]:
    return {case: solve_steady_state(device, voltage_V=case[0], generation_scale=case[1]) for case in CASES}


@pytest.mark.parametrize("voltage, light", CASES)
def test_steady_driver_history_is_identical(device: CompiledDevice1D, references: dict[tuple[float, float], SteadyState1D],
                                            assembled: None, voltage: float, light: float) -> None:
    candidate = solve_steady_state(device, voltage_V=voltage, generation_scale=light)
    assert isinstance(candidate.newton, NewtonResult)
    assert_same_point(references[(voltage, light)], candidate)
    assert all(gate.passed for gate in candidate.physical_gates)


def test_substitution_really_routes_through_the_assembler(device: CompiledDevice1D, assembled: None) -> None:
    calls = []
    original = AssembledDC1DProblem.assemble

    def counted(self: AssembledDC1DProblem, *args: object, **kwargs: object) -> object:
        calls.append(1)
        return original(self, *args, **kwargs)  # type: ignore[arg-type]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(AssembledDC1DProblem, "assemble", counted)
        solve_equilibrium(device)
    assert calls


@pytest.fixture(scope="module")
def reference_sweep(device: CompiledDevice1D) -> list[SteadyState1D]:
    points = [solve_steady_state(device)]
    for voltage in np.round(np.arange(0.05, 1.01, 0.05), 2):
        points.append(solve_steady_state(device, initial_state=points[-1].state, voltage_V=float(voltage)))
    return points


def test_continuation_from_an_accepted_state_is_identical(device: CompiledDevice1D,
                                                           reference_sweep: list[SteadyState1D],
                                                           assembled: None) -> None:
    candidate = solve_steady_state(device)
    assert_same_point(reference_sweep[0], candidate)
    for reference, voltage in zip(reference_sweep[1:], np.round(np.arange(0.05, 1.01, 0.05), 2)):
        candidate = solve_steady_state(device, initial_state=candidate.state, voltage_V=float(voltage))
        assert_same_point(reference, candidate)


def test_equilibrium_entry_point_is_identical(device: CompiledDevice1D) -> None:
    reference = solve_equilibrium(device)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(steady_1d, "DC1DProblem", assembled_problem_factory(device))
        candidate = solve_equilibrium(device)
    assert_same_point(reference, candidate)


def test_frequency_operator_and_solve_are_identical(device: CompiledDevice1D) -> None:
    """(J + i omega M) and its sparse solve, for a consistent right-hand side."""
    point = solve_steady_state(device, voltage_V=0.6, generation_scale=1.0)
    u = point.state.u
    reference, candidate = DC1DProblem(device, 0.6, 1.0), AssembledDC1DProblem(device, 0.6, 1.0)
    J, M = reference.jacobian(u), reference.storage_jacobian(u)
    J_new, M_new = candidate.jacobian(u), candidate.storage_jacobian(u)
    rng = np.random.default_rng(17)
    count = device.node_count
    direction = np.r_[1.0e-3 * rng.standard_normal(count), 1.0e-2 * rng.standard_normal(2 * count)]
    for frequency in (0.0, 1.0e2, 1.0e5, 1.0e8):
        omega = 2.0 * np.pi * frequency
        A = (J + 1j * omega * M).tocsc()
        A_new = (J_new + 1j * omega * M_new).tocsc()
        A.sort_indices()
        A_new.sort_indices()
        assert A_new.indptr.tobytes() == A.indptr.tobytes()
        assert A_new.indices.tobytes() == A.indices.tobytes()
        assert A_new.data.tobytes() == A.data.tobytes()
        b = A @ direction
        assert (A_new @ direction).tobytes() == b.tobytes()
        assert splu(A_new).solve(b).tobytes() == splu(A).solve(b).tobytes()


def test_public_quasi_fermi_chain_equals_the_private_one(device: CompiledDevice1D) -> None:
    point = solve_steady_state(device, voltage_V=0.3, generation_scale=1.0)
    rng = np.random.default_rng(5)
    for scale in (0.0, 1.0e-3, 1.0):
        u = point.state.u + scale * rng.standard_normal(point.state.u.size)
        problem = DC1DProblem(device, 0.3, 1.0)
        fluxes = problem.fluxes(u)
        count = device.node_count
        for flux, offset, sign in ((fluxes.electron, count, 1.0), (fluxes.hole, 2 * count, -1.0)):
            drive = sign * np.diff(u[offset:offset + count])
            public = quasi_fermi_potential_tangents(flux, drive, sign, device.thermal_voltage_V)
            private = _quasi_fermi_potential_tangents(flux, drive, sign, device.thermal_voltage_V)
            for left, right in zip(public, private):
                assert left.tobytes() == right.tobytes()
