"""Assembled classical 1D DC versus the existing DC1DProblem, bit for bit.

States are real accepted operating points (dark equilibrium, dark and
illuminated forward bias reached by the production continuation) and seeded
perturbations of them, which are not solutions. Every comparison is on the
represented entries: identical sparsity and identical binary64 words,
including the sign of every zero. There is no tolerance.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
import pytest
from scipy.sparse import csc_matrix
from threadpoolctl import threadpool_limits

from examples.research_1d.field_transport import prepare_case as field_transport_case
from examples.research_1d.steady import SaturatingRecombination, load_device
from solarlab.config.resolve_device import resolve_device
from solarlab.device.compile_1d import compile_device_1d
from solarlab.device.inputs import DeviceInput
from solarlab.device.resolved import PreparedDevice
from solarlab.equations.assembly.classical_1d import AssembledDC1DProblem, classical_dc_1d
from solarlab.equations.dc_1d import CapabilityError, CompiledDevice1D, DC1DProblem, State1D
from solarlab.solvers.nonlinear.steady_1d import solve_equilibrium, solve_steady_state
from solarlab.physics.contacts.exchange import CarrierExchangeChannel
from solarlab.solvers.time.backward_euler import BackwardEuler1DStep, inventory_m2, step_accounting

Vector: TypeAlias = NDArray[np.float64]
BLOCKS = ("potential", "electron", "hole")


@pytest.fixture(scope="module", autouse=True)
def blas1() -> Iterator[None]:
    with threadpool_limits(limits=1):
        yield


def edited(prepared: PreparedDevice, change: dict[str, object]) -> PreparedDevice:
    document = prepared.to_input().editing_data()
    document.update(change)
    return resolve_device(DeviceInput.model_validate(document), prepared.defaults,
                          prepared.resources, sources=prepared.sources)


def finite_contacts(kind: str) -> CompiledDevice1D:
    prepared = load_device()
    if kind == "robin":
        settings = dict(prepared.to_input().editing_data()["settings"])
        settings["flat_band_contacts"] = True
        return compile_device_1d(edited(prepared, {"settings": settings}), intervals_per_layer=12)
    contacts = [dict(id="left", side="left", layer="htl", S_n=0.0, S_p=200.0),
                dict(id="right", side="right", layer="etl", S_n=200.0, S_p=0.0)]
    return compile_device_1d(edited(prepared, {"contacts": contacts}), intervals_per_layer=12)


def saturating(device: CompiledDevice1D) -> SaturatingRecombination:
    return SaturatingRecombination(
        np.full(device.node_count, 1.0e-17), device.recombination.coefficients.ni_sq_m6, 1.0e22)


@pytest.fixture(scope="module")
def research() -> CompiledDevice1D:
    return compile_device_1d(load_device())


def continuation(device: CompiledDevice1D, voltages: tuple[float, ...],
                 generation_scale: float) -> list[tuple[str, State1D, float, float]]:
    """Accepted points of the production driver along a 50 mV bias continuation."""
    start = solve_steady_state(device, generation_scale=generation_scale) if generation_scale else \
        solve_equilibrium(device)
    label = "light" if generation_scale else "dark"
    states = [(f"{label}_V0", start.state, 0.0, generation_scale)]
    state = start.state
    for step in range(1, int(round(max(voltages) / 0.05)) + 1):
        voltage = round(0.05 * step, 10)
        point = solve_steady_state(device, initial_state=state, voltage_V=voltage,
                                   generation_scale=generation_scale)
        state = point.state
        if any(abs(voltage - target) < 1e-9 for target in voltages):
            states.append((f"{label}_V{voltage:g}", state, voltage, generation_scale))
    return states


def perturbed(device: CompiledDevice1D, state: State1D, seed: int, size: float = 1.0e-2) -> Vector:
    """Seeded off-solution coordinates around an accepted state (not a solution)."""
    rng = np.random.default_rng(seed)
    count = device.node_count
    scale = np.r_[np.full(count, device.thermal_voltage_V), np.ones(2 * count)]
    result: Vector = state.u + size * scale * rng.standard_normal(3 * count)
    return result


@pytest.fixture(scope="module")
def research_states(research: CompiledDevice1D) -> list[tuple[str, Vector, float, float]]:
    states = continuation(research, (0.3, 0.6, 0.9), 0.0)
    states += continuation(research, (0.3, 0.6, 0.9), 1.0)
    cases = [(label, state.u, voltage, light) for label, state, voltage, light in states]
    for seed, (label, state, voltage, light) in enumerate(states):
        if label in ("dark_V0", "light_V0.6", "light_V0.9"):
            cases.append((label + "_perturbed", perturbed(research, state, seed), voltage, light))
    return cases


def assert_bits(reference: object, candidate: object) -> None:
    """Identical binary64 words (signed zeros and NaN payloads included)."""
    left = np.ascontiguousarray(reference, dtype=np.float64)
    right = np.ascontiguousarray(candidate, dtype=np.float64)
    assert left.shape == right.shape
    assert left.tobytes() == right.tobytes()


def assert_same_matrix(reference: csc_matrix, candidate: csc_matrix) -> None:
    """Identical represented entries and identical binary64 values."""
    left, right = csc_matrix(reference), csc_matrix(candidate)
    left.sum_duplicates()
    right.sum_duplicates()
    left.sort_indices()
    right.sort_indices()
    assert left.shape == right.shape
    np.testing.assert_array_equal(left.indptr, right.indptr)
    np.testing.assert_array_equal(left.indices, right.indices)
    assert_bits(left.data, right.data)


def storage_mask(device: CompiledDevice1D) -> NDArray[np.bool_]:
    mask: NDArray[np.bool_] = np.ones((2, device.node_count), dtype=bool)
    for index, carrier in enumerate(("n", "p")):
        for side, node in (("left", 0), ("right", device.node_count - 1)):
            if device.channel(carrier, side).kind == "fixed_reservoir":
                mask[index, node] = False
    return mask


def legacy_storage(device: CompiledDevice1D, u: Vector, voltage_V: float) -> Vector:
    """The BE/BDF storage vector: zero Gauss rows, masked carrier inventories."""
    inventory = inventory_m2(device, device.state(u, voltage_V=voltage_V))
    result: Vector = np.r_[np.zeros(device.node_count), np.where(storage_mask(device), inventory, 0.0).ravel()]
    return result


def check_point(device: CompiledDevice1D, u: Vector, voltage_V: float, light: float) -> None:
    reference = DC1DProblem(device, voltage_V, light)
    candidate = AssembledDC1DProblem(device, voltage_V, light)
    assert_bits(reference.residual(u), candidate.residual(u))
    assert_same_matrix(reference.jacobian(u), candidate.jacobian(u))
    assert_same_matrix(reference.storage_jacobian(u), candidate.storage_jacobian(u))
    assert_same_matrix(-reference.storage_jacobian(u), candidate.storage_tangent(u))
    assert_bits(legacy_storage(device, u, voltage_V), candidate.storage(u))
    for alpha in (0.25, -3.0e7):
        assert_same_matrix(reference.jacobian(u, alpha=alpha), candidate.jacobian(u, alpha=alpha))
    assert_bits(reference.current_density(u), candidate.current_density(u))
    assert_bits(reference.contact_current_density(u), candidate.contact_current_density(u))
    ref_fluxes, new_fluxes = reference.fluxes(u), candidate.fluxes(u)
    for carrier in ("electron", "hole"):
        for name in ("flux", "left_density_derivative", "right_density_derivative",
                     "left_potential_derivative", "right_potential_derivative"):
            assert_bits(getattr(getattr(ref_fluxes, carrier), name),
                        getattr(getattr(new_fluxes, carrier), name))


def test_research_device_layout_and_identity(research: CompiledDevice1D) -> None:
    system = classical_dc_1d(research)
    count = research.node_count
    assert system.layout.block_ranges() == {
        "potential": (0, count), "electron": (count, 2 * count), "hole": (2 * count, 3 * count)}
    mask = system.layout.constraint_mask()
    assert np.flatnonzero(mask).tolist() == [0, count - 1, count, 2 * count - 1, 2 * count, 3 * count - 1]
    assert classical_dc_1d(research).identity == system.identity
    assert len(system.identity) == 64 and system.identity != research.identity


def test_research_states_are_bitwise_equal(research: CompiledDevice1D,
                                           research_states: list[tuple[str, Vector, float, float]]) -> None:
    assert len(research_states) == 11
    for _label, u, voltage, light in research_states:
        check_point(research, u, voltage, light)


@pytest.mark.parametrize("kind", ["robin", "selective"])
def test_finite_and_blocking_contacts_are_bitwise_equal(kind: str) -> None:
    device = finite_contacts(kind)
    kinds = sorted(channel.kind for channel in device.channels)
    assert kinds == (["robin"] * 4 if kind == "robin" else ["blocking", "blocking", "robin", "robin"])
    system = classical_dc_1d(device)
    # No carrier reservoir rows: all four channels are port exchange Terms.
    assert len(system.exchanges) == 4 and len(system.layout.constraints) == 1
    dark = solve_equilibrium(device)
    light = solve_steady_state(device, voltage_V=0.2, generation_scale=1.0)
    for seed, (state, voltage, scale) in enumerate(((dark.state, 0.0, 0.0), (light.state, 0.2, 1.0))):
        check_point(device, state.u, voltage, scale)
        check_point(device, perturbed(device, state, 10 + seed), voltage, scale)


def test_field_dependent_transport_is_bitwise_equal() -> None:
    device = compile_device_1d(field_transport_case(), intervals_per_layer=(12, 12, 12))
    assert device.field_transport is not None
    point = solve_steady_state(device, voltage_V=0.002, generation_scale=1.0)
    check_point(device, point.state.u, 0.002, 1.0)
    check_point(device, perturbed(device, point.state, 21), 0.002, 1.0)


def test_injected_local_recombination_is_bitwise_equal() -> None:
    base = compile_device_1d(load_device(), intervals_per_layer=12)
    device = compile_device_1d(load_device(), intervals_per_layer=12,
                               additional_recombination=saturating(base))
    point = solve_steady_state(device, voltage_V=0.4, generation_scale=1.0)
    check_point(device, point.state.u, 0.4, 1.0)
    check_point(device, perturbed(device, point.state, 31), 0.4, 1.0)


def test_inadmissible_state_matches_the_nan_residual(research: CompiledDevice1D) -> None:
    u = np.zeros(3 * research.node_count)
    u[research.node_count:2 * research.node_count] = 800.0  # n overflows
    reference = DC1DProblem(research, 0.0, 1.0).residual(u)
    candidate = AssembledDC1DProblem(research, 0.0, 1.0).residual(u)
    assert np.all(np.isnan(reference)) and np.all(np.isnan(candidate))


def test_backward_euler_step_in_storage_difference_form(research: CompiledDevice1D) -> None:
    """R = F(u) - (S(u) - S(u_old))/dt and J - S_u/dt equal BackwardEuler1DStep."""
    old = solve_steady_state(research, voltage_V=0.5, generation_scale=1.0)
    system = classical_dc_1d(research)
    old_problem = AssembledDC1DProblem(research, 0.5, 1.0, system)
    rows = system.layout.constraint_mask()
    for voltage, dt in ((0.5, 1.0e-9), (0.55, 1.0e-7), (0.45, 1.0e-6)):
        step = BackwardEuler1DStep(DC1DProblem(research, voltage, 1.0), old.state, dt)
        problem = old_problem.at(voltage, 1.0)
        for seed in range(2):
            u = perturbed(research, State1D(step.initial_coordinates(), old.state.potential_V,
                                            old.state.electron_density_m3, old.state.hole_density_m3,
                                            research.sha256, voltage), 40 + seed, 1.0e-3)
            increment = problem.storage_increment(u, old_problem, old.state.u)
            rate = step.inventory_rate_m2_s(u)
            # Masked rows are x * 0, as the original storage adapters' ``*= storage_mask``.
            assert_bits(np.r_[np.zeros(research.node_count), (rate * storage_mask(research)).ravel()],
                        increment / dt)
            # The original step never writes its algebraic rows: R = F there.
            residual = problem.residual(u)
            assert_bits(step.residual(u), np.where(rows, residual, residual - increment / dt))
            assert_same_matrix(step.jacobian(u), csc_matrix(problem.jacobian(u) - problem.storage_tangent(u) / dt))


def test_udot_form_is_refused(research: CompiledDevice1D) -> None:
    problem = AssembledDC1DProblem(research, 0.0, 0.0)
    reference = DC1DProblem(research, 0.0, 0.0)
    u = solve_equilibrium(research).state.u
    assert_bits(problem.residual(u), problem.residual(u, np.zeros_like(u)))
    assert_same_matrix(problem.jacobian(u), problem.jacobian(u, np.zeros_like(u)))
    with pytest.raises(CapabilityError):
        problem.residual(u, np.ones_like(u))
    with pytest.raises(CapabilityError):
        problem.jacobian(u, np.ones_like(u))
    # R02: shape and finiteness are admitted exactly as DC1DProblem admits them.
    for bad in (0.0, np.zeros(1), np.zeros(u.size + 1), np.full(u.size, np.nan)):
        with pytest.raises(ValueError):
            reference.residual(u, bad)
        with pytest.raises(ValueError):
            problem.residual(u, bad)
        with pytest.raises(ValueError):
            problem.jacobian(u, bad)


def test_problem_rejects_invalid_drive(research: CompiledDevice1D) -> None:
    with pytest.raises(ValueError):
        AssembledDC1DProblem(research, 0.0, -1.0)
    with pytest.raises(ValueError):
        AssembledDC1DProblem(research, float("nan"), 0.0)
    other = compile_device_1d(load_device(), intervals_per_layer=6)
    with pytest.raises(ValueError):
        AssembledDC1DProblem(research, 0.0, 0.0, classical_dc_1d(other))


class _LegacyNamedEndpoint(AssembledDC1DProblem):
    """Test-only alias of the private name the unchanged BE accounting calls."""

    def _contact_exchange(self, u: Vector, channel: CarrierExchangeChannel, node: int) -> tuple[float, float]:
        return self.contact_exchange(u, channel, node)


def observation_states(device: CompiledDevice1D) -> list[tuple[Vector, float, float]]:
    dark = solve_equilibrium(device)
    light = solve_steady_state(device, voltage_V=0.4, generation_scale=1.0)
    return [(dark.state.u, 0.0, 0.0), (light.state.u, 0.4, 1.0),
            (perturbed(device, dark.state, 5), 0.0, 0.0), (perturbed(device, light.state, 6), 0.4, 1.0)]


@pytest.mark.parametrize("kind", ["fixed", "robin", "selective", "ct", "injected"])
def test_observation_kernels_equal_dc1d(kind: str) -> None:
    """``recombination_rate`` and ``contact_exchange`` return the DC1DProblem words."""
    if kind == "ct":
        device = compile_device_1d(field_transport_case(), intervals_per_layer=(4, 4, 4))
    elif kind in ("robin", "selective"):
        device = finite_contacts(kind)
    elif kind == "injected":
        base = compile_device_1d(load_device(), intervals_per_layer=12)
        device = compile_device_1d(load_device(), intervals_per_layer=12, additional_recombination=saturating(base))
    else:
        device = compile_device_1d(load_device())
    exchanged = 0
    for u, voltage, light in observation_states(device):
        reference = DC1DProblem(device, voltage, light)
        candidate = AssembledDC1DProblem(device, voltage, light)
        _, n, p = device.fields(u, voltage_V=voltage)
        for derivatives in (True, False):
            old = reference.recombination_rate(n, p, derivatives=derivatives)
            new = candidate.recombination_rate(n, p, derivatives=derivatives)
            assert new.has_derivatives == old.has_derivatives == derivatives
            for name in ("rate", "d_rate_dn", "d_rate_dp"):
                if derivatives:
                    assert_bits(getattr(old, name), getattr(new, name))
        for channel in device.channels:
            node = 0 if channel.side == "left" else device.node_count - 1
            if channel.kind == "fixed_reservoir":
                with pytest.raises(ValueError, match="no bound exchange"):
                    candidate.contact_exchange(u, channel, node)
                continue
            exchanged += 1
            assert_bits(reference.contact_exchange(u, channel, node), candidate.contact_exchange(u, channel, node))
            with pytest.raises(ValueError, match="no bound exchange"):
                candidate.contact_exchange(u, channel, device.node_count - 1 - node)
        with pytest.raises(ValueError, match="coordinate vector"):
            candidate.contact_exchange(u[:-1], device.channels[0], 0)
    assert (exchanged > 0) == (kind in ("robin", "selective"))


@pytest.mark.parametrize("kind", ["fixed", "robin", "selective"])
def test_backward_euler_accounting_on_the_assembled_endpoint(kind: str) -> None:
    """The unchanged BE step accounting reads only the public observation surface."""
    device = finite_contacts(kind) if kind != "fixed" else compile_device_1d(load_device())
    old = solve_steady_state(device, voltage_V=0.4, generation_scale=1.0)
    system = classical_dc_1d(device)
    for voltage, dt in ((0.4, 1.0e-9), (0.45, 1.0e-7)):
        legacy = BackwardEuler1DStep(DC1DProblem(device, voltage, 1.0), old.state, dt)
        assembled = BackwardEuler1DStep(_LegacyNamedEndpoint(device, voltage, 1.0, system), old.state, dt)
        u = perturbed(device, State1D(legacy.initial_coordinates(), old.state.potential_V,
                                      old.state.electron_density_m3, old.state.hole_density_m3,
                                      device.sha256, voltage), 70, 1.0e-3)
        reference, candidate = step_accounting(legacy, u), step_accounting(assembled, u)
        assert reference.keys() == candidate.keys()
        for name, value in reference.items():
            if isinstance(value, (np.ndarray, float)):
                assert_bits(value, candidate[name])
            else:
                assert value == candidate[name], name


def test_masked_storage_increment_keeps_the_original_signed_zeros(research: CompiledDevice1D) -> None:
    """Constraint rows of the increment are ``x * 0``, byte for byte as ``*= storage_mask``.

    A falling boundary inventory gives a negative zero there; a positive zero
    would be value-equal but not byte-identical to the original storage path.
    """
    old = solve_steady_state(research, voltage_V=0.5, generation_scale=1.0)
    system = classical_dc_1d(research)
    old_problem = AssembledDC1DProblem(research, 0.5, 1.0, system)
    rows = system.layout.constraint_mask()
    negative = 0
    for voltage in (0.45, 0.5, 0.55):
        step = BackwardEuler1DStep(DC1DProblem(research, voltage, 1.0), old.state, 1.0)
        for seed in range(3):
            u = perturbed(research, State1D(step.initial_coordinates(), old.state.potential_V,
                                            old.state.electron_density_m3, old.state.hole_density_m3,
                                            research.sha256, voltage), 80 + seed, 1.0e-3)
            change = np.r_[np.zeros(research.node_count), step.inventory_rate_m2_s(u).ravel()]
            change[research.node_count:] *= storage_mask(research).ravel()
            increment = old_problem.at(voltage, 1.0).storage_increment(u, old_problem, old.state.u)
            assert_bits(change, increment)
            negative += int(np.count_nonzero(np.signbit(increment[rows])))
    assert negative > 0
