"""P02-06 port and native-interval prototype, independent of device engines.

Solver return and physical acceptance are separate. An observer must
consume the current interpolation interval before the solver advances; only
immutable endpoints and observations survive that boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike

from scripts.benchmarks.contract_prototype import (
    AcceptedStep, BalanceTerms, COULOMB, ContractError, EquationSpec,
    ImmutableArrays, ImplicitSystem, Layout, LinearCoordinates, LinearStorage,
    Point, SECOND, StateIncrement, Support, TerminalPort, VariableSpec, Vector,
    VOLT, frozen_array,
)
from scripts.benchmarks.storage_prototype import conservative_be_step, consistent_initial_state


def finite_scalar(value: float) -> float:
    array = np.asarray(value)
    if array.ndim != 0 or not np.isfinite(array):
        raise ContractError("invalid_port_scalar")
    return float(array)


@dataclass(frozen=True)
class PortSample(ImmutableArrays):
    time: float
    voltage: float
    voltage_rate: float
    charge: Vector
    conduction: Vector
    current: Vector
    side: str = "continuous"

    def __post_init__(self) -> None:
        for name in ("time", "voltage", "voltage_rate"):
            object.__setattr__(self, name, finite_scalar(getattr(self, name)))
        for name in ("charge", "conduction", "current"):
            value = frozen_array(getattr(self, name))
            if value.shape != (2,):
                raise ContractError("two_port_shape_mismatch")
            object.__setattr__(self, name, value)
        if self.side not in {"continuous", "left", "right"}:
            raise ContractError("invalid_event_side")


@dataclass(frozen=True)
class PortReading(ImmutableArrays):
    point: Point
    derivative: Vector
    sample: PortSample
    origin: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "derivative", frozen_array(self.derivative))
        if self.derivative.shape != self.point.y.shape or not self.origin:
            raise ContractError("invalid_port_reading")


def consistent_physical_rate(system: ImplicitSystem, coordinates: LinearCoordinates,
                             point: Point, input_rate: ArrayLike, *,
                             history: tuple[Point, ...] = (),
                             gauge_nullspace: bool = False) -> Vector:
    """Strict prototype tangent estimate at the unchanged physical state.

    This estimates the physical derivative from the equations. It is not the
    derivative of the approximate interpolation curve, so its interval charge
    balance and analytical error require separate validation.
    """
    _, terms = system.evaluate(point)
    if np.any(terms.algebraic != 0):
        raise ContractError("nonzero_algebraic_state")
    result = consistent_initial_state(
        system, coordinates, point, system.storage.value(point), input_rate,
        history=history, max_iterations=0, tolerance=0,
        gauge_nullspace=gauge_nullspace,
    )
    if (result.point.identity != point.identity or result.iterations != 0
            or tuple(p.identity for p in result.history) != tuple(p.identity for p in history)
            or np.any(result.storage_error != 0)):
        raise ContractError("reconstruction_changed_physical_history")
    if result.rank.rank != point.y.size:
        raise ContractError("reconstruction_rank_deficiency")
    if np.any(result.residual != 0) or np.any(result.tangent != 0):
        raise ContractError("reconstructed_rate_not_exactly_consistent")
    return frozen_array(result.derivative)


@dataclass(frozen=True)
class VoltageImpulse(ImmutableArrays):
    before: PortSample
    after: PortSample
    charge: Vector

    def __post_init__(self) -> None:
        if (self.before.time != self.after.time or self.before.side != "left"
                or self.after.side != "right"):
            raise ContractError("event_limits_mismatch")
        value = frozen_array(self.charge)
        if value.shape != (2,):
            raise ContractError("two_port_shape_mismatch")
        object.__setattr__(self, "charge", value)


@dataclass(frozen=True)
class Capacitor:
    epsilon: float
    area: float
    length: float

    def __post_init__(self) -> None:
        for name in ("epsilon", "area", "length"):
            value = finite_scalar(getattr(self, name))
            if value <= 0:
                raise ContractError("invalid_capacitor_geometry")
            object.__setattr__(self, name, value)

    @property
    def capacitance(self) -> float:
        return self.epsilon * self.area / self.length

    @property
    def ports(self) -> tuple[TerminalPort, TerminalPort]:
        return (TerminalPort("left", -1, self.area),
                TerminalPort("right", 1, self.area))

    def sample(self, time: float, voltage: float, voltage_rate: float,
               conduction: float = 0.0, side: str = "continuous") -> PortSample:
        voltage, voltage_rate, conduction = map(
            finite_scalar, (voltage, voltage_rate, conduction))
        displacement = self.epsilon * voltage / self.length
        displacement_rate = self.epsilon * voltage_rate / self.length
        jc = conduction / self.area
        return PortSample(
            time, voltage, voltage_rate,
            [float(port.charge(displacement)) for port in self.ports],
            [float(port.current(jc, 0.0)) for port in self.ports],
            [float(port.current(jc, displacement_rate)) for port in self.ports],
            side,
        )

    def admittance(self, frequency: float) -> np.ndarray:
        omega = 2 * np.pi * finite_scalar(frequency)
        d_displacement_d_voltage = self.epsilon / self.length
        return np.asarray([1j * omega * float(port.charge(d_displacement_d_voltage))
                           for port in self.ports])

    def event(self, time: float, before: float, after: float) -> VoltageImpulse:
        left = self.sample(time, before, 0.0, side="left")
        right = self.sample(time, after, 0.0, side="right")
        # An ideal voltage jump has a charge impulse, no finite current sample.
        return VoltageImpulse(left, right, right.charge - left.charge)

    def ramp_step(self, start: float, end: float, voltage: float,
                  slope: float) -> NativeStep:
        coordinates = LinearCoordinates(Layout((), (), ()))
        left = coordinates.point([], start, [voltage])
        right, increment = coordinates.advance(
            left, [], end, [voltage + slope * (end - start)])
        delta_d = self.epsilon * slope * (end - start) / self.length
        # The storage layout is empty; port charge comes from the input.
        q0 = self.sample(start, voltage, slope).charge
        sample = self.sample(end, right.inputs[0], slope)
        defect = sample.charge - q0 - (end - start) * sample.current
        return NativeStep(left, right, increment, [], defect, [],
                          [delta_d, delta_d], [slope], "prescribed-ramp")


@dataclass(frozen=True)
class RCModel:
    capacitance: float
    resistance: float

    def __post_init__(self) -> None:
        for name in ("capacitance", "resistance"):
            value = finite_scalar(getattr(self, name))
            if value <= 0:
                raise ContractError("invalid_rc_parameter")
            object.__setattr__(self, name, value)

    @property
    def capacitor(self) -> Capacitor:
        return Capacitor(self.capacitance, 1.0, 1.0)

    def system(self) -> tuple[LinearCoordinates, ImplicitSystem]:
        support = Support("ports", "global", (1,))
        layout = Layout((support,), (
            VariableSpec("potential", "rc", "ports", (1,), VOLT),
            VariableSpec("terminal_current", "rc", "ports", (1,),
                         COULOMB / SECOND, role="constraint"),
        ), (
            EquationSpec("charge_balance", "rc", "ports", (1,), COULOMB / SECOND,
                         derivative_support=("potential", "terminal_current")),
            EquationSpec("open_circuit", "rc", "ports", (1,), COULOMB / SECOND,
                         role="constraint", derivative_support=("terminal_current",)),
        ))
        coordinates = LinearCoordinates(layout)

        def terms(point: Point) -> BalanceTerms:
            voltage, external_current = point.y
            return BalanceTerms(
                np.array([external_current - voltage / self.resistance]),
                np.array([external_current]),
                np.array([[-1 / self.resistance, 1.0]]), np.zeros((1, 0)),
                np.zeros(1), np.array([[0.0, 1.0]]), np.zeros((1, 0)), np.zeros(1),
            )

        return coordinates, ImplicitSystem(
            LinearStorage([[self.capacitance, 0.0]]), terms)

    def sample(self, point: Point, derivative: ArrayLike) -> PortSample:
        derivative = frozen_array(derivative)
        if point.y.shape != (2,) or derivative.shape != (2,):
            raise ContractError("rc_state_shape_mismatch")
        return self.capacitor.sample(
            point.time, point.y[0], derivative[0], point.y[0] / self.resistance)


@dataclass(frozen=True)
class NativeStep(ImmutableArrays):
    """Retained step output; an IDA stop output may itself be interpolated."""
    left: Point
    right: Point
    increment: StateIncrement
    derivative: Vector
    residual: Vector
    storage_delta: Vector
    displacement_delta: Vector
    input_rate: Vector
    method: str

    def __post_init__(self) -> None:
        self.increment.validate(self.left, self.right)
        if self.right.time <= self.left.time:
            raise ContractError("nonpositive_step")
        for name in ("derivative", "residual", "storage_delta", "displacement_delta", "input_rate"):
            object.__setattr__(self, name, frozen_array(getattr(self, name)))
        if self.derivative.shape != self.right.y.shape or self.input_rate.shape != self.right.inputs.shape:
            raise ContractError("step_derivative_shape_mismatch")

    def accept(self, metrics: tuple[tuple[str, float], ...], passed: bool) -> AcceptedStep:
        return AcceptedStep(
            self.left, self.right, self.increment, self.storage_delta,
            self.displacement_delta, self.derivative, self.input_rate, self.method, "continuous",
            metrics, passed,
        )


class NativeInterval:
    """A temporary interpolation capability, revoked before the next step."""

    def __init__(self, solver, model: RCModel, coordinates: LinearCoordinates,
                 step: NativeStep, stop: float, derivative_policy: str = "interpolant"):
        self.step = step
        self._solver, self._model, self._coordinates = solver, model, coordinates
        self._stop = stop
        self._active = True
        self.derivative_policy = derivative_policy
        self._raw: dict[tuple[float, str], PortReading] = {}
        self._physical: dict[tuple[str, str], PortReading] = {}
        _, self._system = model.system()

    def reading(self, time: float) -> PortReading:
        if not self._active:
            raise ContractError("interpolation_interval_expired")
        time = finite_scalar(time)
        if not self.step.left.time <= time <= self.step.right.time:
            raise ContractError("outside_native_interval")
        kind = self.step.method if time == self.step.right.time else "IDA-normal-output"
        key = (time, kind)
        if key in self._raw:
            return self._raw[key]
        if time == self.step.right.time:
            point, derivative = self.step.right, self.step.derivative
        else:
            result = self._solver.step(time, method="normal", tstop=self._stop)
            if not result.success:
                raise ContractError("native_interpolation_failed")
            if abs(float(result.t) - time) > 8 * np.spacing(max(abs(time), 1.0)):
                raise ContractError("native_interpolation_time_mismatch")
            point, derivative = self._coordinates.point(result.y, time), result.yp
        reading = PortReading(point, derivative, self._model.sample(point, derivative), kind)
        self._raw[key] = reading
        return reading

    def physical_reading(self, reading: PortReading) -> PortReading:
        if not self._active:
            raise ContractError("interpolation_interval_expired")
        if not self.step.left.time <= reading.point.time <= self.step.right.time:
            raise ContractError("outside_native_interval")
        if not any(reading is owned for owned in self._raw.values()):
            raise ContractError("foreign_interval_reading")
        key = (reading.point.identity, reading.origin)
        if key not in self._physical:
            derivative = consistent_physical_rate(
                self._system, self._coordinates, reading.point, [], history=(self.step.left,))
            self._physical[key] = PortReading(
                reading.point, derivative, self._model.sample(reading.point, derivative),
                "physical-tangent-v1 from " + reading.origin,
            )
        return self._physical[key]

    def sample(self, time: float, derivative_policy: str | None = None) -> PortSample:
        reading = self.reading(time)
        policy = derivative_policy or self.derivative_policy
        if policy == "physical-tangent-v1":
            return self.physical_reading(reading).sample
        if policy != "interpolant":
            raise ContractError("unknown_derivative_policy")
        return reading.sample

    @property
    def raw_readings(self) -> tuple[PortReading, ...]:
        return tuple(self._raw.values())

    @property
    def physical_readings(self) -> tuple[PortReading, ...]:
        return tuple(self._physical.values())

    def close(self) -> None:
        self._active = False
        self._solver = None

    def restore_endpoint(self) -> PortSample:
        """Finish normal-output queries at the already accepted native endpoint.

        IDA remembers its last returned output time. After an interior normal
        query, ONE_STEP may return that same native endpoint again. Restoring
        the output cursor here avoids mistaking that return for a new step.
        """
        if not self._active:
            raise ContractError("interpolation_interval_expired")
        result = self._solver.step(self.step.right.time, method="normal", tstop=self._stop)
        if not result.success:
            raise ContractError("native_endpoint_restore_failed")
        if float(result.t) != self.step.right.time:
            raise ContractError("native_endpoint_time_changed_by_observation")
        # The interpolation derivative has its own numerical error; it is not
        # bitwise identical to the native method derivative. Preserve both.
        point = self._coordinates.point(result.y, self.step.right.time)
        reading = PortReading(point, result.yp, self._model.sample(point, result.yp),
                              "IDA-normal-endpoint-restore")
        self._raw[(point.time, reading.origin)] = reading
        if self.derivative_policy == "physical-tangent-v1":
            self.physical_reading(reading)
        return reading.sample


@dataclass(frozen=True)
class RCResult:
    observations: tuple[PortSample, ...]
    native_steps: tuple[NativeStep, ...]
    endpoint_outputs: tuple[PortSample, ...]
    raw_readings: tuple[PortReading, ...]
    physical_readings: tuple[PortReading, ...]


def rc_ida(model: RCModel, initial_voltage: float, end_time: float,
           observation_times: ArrayLike, *, rtol: float, atol: float,
           max_step: float, max_order: int,
           observer: Callable[[NativeInterval], None], step_budget: int = 10000,
           derivative_policy: str = "interpolant") -> RCResult:
    from sksundae.ida import IDA

    end_time = finite_scalar(end_time)
    times = frozen_array(observation_times)
    if (end_time <= 0 or times.ndim != 1 or np.any(np.diff(times) <= 0)
            or np.any(times < 0) or np.any(times > end_time)):
        raise ContractError("invalid_observation_times")
    coordinates, system = model.system()

    def residual(time, y, ydot, output):
        output[:] = system.residual(coordinates.point(y, time), ydot, np.empty(0))

    def jacobian(time, y, ydot, residual_value, cj, values):
        values[:] = system.linearize(
            coordinates.point(y, time), ydot, np.empty(0)).ida_matrix(cj)

    solver = IDA(residual, jacfn=jacobian, algebraic_idx=[1],
                 rtol=rtol, atol=atol, max_step=max_step, max_order=max_order)
    derivative = np.array([-initial_voltage / (model.resistance * model.capacitance), 0.0])
    initial = solver.init_step(0.0, [initial_voltage, 0.0], derivative)
    if not initial.success:
        raise ContractError("native_initialization_failed")
    left = coordinates.point(initial.y, 0.0)
    observations, steps, endpoint_outputs = [], [], []
    initial_reading = PortReading(left, initial.yp, model.sample(left, initial.yp), "IDA-initial-output")
    raw_readings, physical_readings = [initial_reading], []
    if derivative_policy == "physical-tangent-v1":
        initial_rate = consistent_physical_rate(system, coordinates, left, [])
        initial_estimate = PortReading(left, initial_rate, model.sample(left, initial_rate), "physical-tangent-v1 from IDA-initial-output")
        physical_readings.append(initial_estimate)
    elif derivative_policy != "interpolant":
        raise ContractError("unknown_derivative_policy")
    cursor = 0
    if times.size and times[0] == 0:
        observations.append(physical_readings[0].sample if physical_readings else initial_reading.sample)
        cursor = 1
    for _ in range(step_budget):
        result = solver.step(end_time, method="onestep", tstop=end_time)
        if not result.success:
            raise ContractError("native_step_failed")
        time = float(result.t)
        if not left.time < time <= end_time:
            raise ContractError("invalid_native_interval")
        right = coordinates.point(result.y, time)
        increment = StateIncrement(left.identity, right.identity, {
            spec.id: right.state.field(spec.id).difference(left.state.field(spec.id))
            for spec in coordinates.layout.variables
        })
        delta_q = system.storage.delta(left, right, increment)
        delta_d = model.capacitance * increment.field("potential").values[0]
        native = NativeStep(
            left, right, increment, result.yp,
            system.residual(right, np.asarray(result.yp), np.empty(0)),
            delta_q, [delta_d, delta_d], [],
            "IDA-stop-output" if result.status == 1 else "IDA-onestep-output",
        )
        interval = NativeInterval(solver, model, coordinates, native, end_time, derivative_policy)
        try:
            while cursor < times.size and times[cursor] <= time:
                observations.append(interval.sample(float(times[cursor])))
                cursor += 1
            observer(interval)
            endpoint_outputs.append(interval.restore_endpoint())
            if derivative_policy == "physical-tangent-v1":
                for reading in interval.raw_readings:
                    interval.physical_reading(reading)
            raw_readings.extend(interval.raw_readings)
            physical_readings.extend(interval.physical_readings)
        finally:
            interval.close()
        steps.append(native)
        left = right
        if time == end_time:
            break
    else:
        raise ContractError("native_step_budget_exhausted")
    if cursor != times.size:
        raise ContractError("missing_observation")
    return RCResult(tuple(observations), tuple(steps), tuple(endpoint_outputs),
                    tuple(raw_readings), tuple(physical_readings))


def rc_be(model: RCModel, initial_voltage: float, end_time: float,
          step_size: float) -> tuple[tuple[PortSample, ...], tuple[NativeStep, ...]]:
    end_time, step_size = map(finite_scalar, (end_time, step_size))
    if step_size <= 0 or end_time <= 0:
        raise ContractError("nonpositive_step")
    count = round(end_time / step_size)
    if count < 1 or abs(count * step_size - end_time) > 8 * np.spacing(end_time):
        raise ContractError("incomplete_be_grid")
    coordinates, system = model.system()
    left = coordinates.point([initial_voltage, 0.0])
    initial_rate = [-initial_voltage / (model.resistance * model.capacitance), 0.0]
    samples, history = [model.sample(left, initial_rate)], []
    for _ in range(count):
        trial = conservative_be_step(system, coordinates, left, step_size)
        rate = (trial.right.y - left.y) / step_size
        delta_q = system.storage.delta(left, trial.right, trial.increment)
        delta_d = model.capacitance * trial.increment.field("potential").values[0]
        history.append(NativeStep(
            left, trial.right, trial.increment, rate, trial.residual,
            delta_q, [delta_d, delta_d], [], "conservative-BE",
        ))
        samples.append(model.sample(trial.right, rate))
        left = trial.right
    return tuple(samples), tuple(history)
