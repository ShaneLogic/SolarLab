"""Unchanged electronic BE/BDF controls applied to the assembled storage system.

The original diagnostics receive only an assembled endpoint, with the public
observation names mapped by AssembledStorage1D. Missing phase-2a observations
are strict expected failures before any nonlinear solve, never legacy fallbacks.
"""

from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from examples.research_1d.steady import load_device
from examples.research_1d.transient import controls, json_value
from solarlab.device.compile_1d import compile_device_1d
from solarlab.equations.assembly.classical_1d import AssembledDC1DProblem, classical_dc_1d
from solarlab.equations.dc_1d import DC1DProblem
from solarlab.experiments import transient
from solarlab.experiments.ionic_transient import _finite_accounting
from solarlab.experiments.jv.operating_points import _verify_point
from solarlab.experiments.transient import TransientProtocol, solve_transient
from solarlab.solvers.nonlinear.newton import NewtonSettings
from solarlab.solvers.nonlinear.problem import GateOutcome
from solarlab.solvers.nonlinear.production import (
    production_convergence_norm, production_coordinates, production_residual_scaling,
)
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions, solve_steady_state
from solarlab.solvers.time.assembled_storage import AssembledStorage1D
from solarlab.solvers.time.backward_euler import BackwardEuler1DStep, step_accounting
from solarlab.solvers.time.bdf import Assessment, BDFOptions, BDFStep, HistoryPoint, solve_bdf
from solarlab.solvers.time.bdf_1d import BDFDrive, ElectronicStorage1D, bdf_accounting


INITIALS = {}
RECORDS = {}


@pytest.fixture(scope="module", autouse=True)
def blas1():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(scope="module")
def device():
    return compile_device_1d(load_device(), intervals_per_layer=12)


def require_observations():
    for name in ("recombination_rate", "contact_exchange"):
        assert callable(getattr(AssembledDC1DProblem, name, None)), name


def record(name, value):
    RECORDS[name] = json_value(value)
    output = os.environ.get("SOLARLAB_BDF_ASSEMBLY_OUTPUT")
    if output:
        Path(output).write_text(json.dumps(RECORDS, indent=2, allow_nan=False) + "\n")


def initial(device, generation):
    key = (device.identity, generation)
    if key not in INITIALS:
        point = solve_steady_state(device, voltage_V=0.0, generation_scale=generation, options=SteadyOptions())
        record("initial_" + str(generation), point)
        assert point.converged
        INITIALS[key] = point
    return INITIALS[key]


def protocol(generation, output=None):
    times = np.linspace(0.0, 4e-7, 5)
    voltage = np.zeros(5) if generation else .002 * (times / times[-1])**2
    light = 1.0 + .01 * (times / times[-1])**2 if generation else np.zeros(5)
    drive = BDFDrive.from_protocol(TransientProtocol(times, voltage, light))
    if output is None:
        return TransientProtocol(times, voltage, light), drive
    values = np.array([drive.values(t) for t in output])
    return TransientProtocol(output, values[:, 0], values[:, 1]), drive


def electronic_view(step):
    """Only the unchanged diagnostic helper's endpoint interface is adapted."""
    endpoint = ElectronicStorage1D(step.endpoint)
    history = tuple(HistoryPoint(point.time_s, point.u, point.storage_value,
                                 ElectronicStorage1D(point.problem)) for point in step.history)
    return BDFStep(endpoint, history, step.end_s, step.order)


def run_assembled(device, point, requested, drive, options):
    """Bind the existing controls, predictor and accounting to solve_bdf."""
    physical = controls()
    admission = SteadyOptions()
    _verify_point(device, point, point.generation_scale, admission)
    base = NewtonSettings(production_residual_scaling(device,
        poisson_tolerance=physical.poisson_tolerance,
        continuity_tolerance_A_m2=physical.continuity_tolerance_A_m2,
        contact_log_tolerance=physical.contact_log_tolerance), 1.0,
        production_coordinates(device, potential_step_V=physical.potential_step_V,
            quasi_fermi_step=physical.quasi_fermi_step), physical.newton_budget, (), False)
    norm = production_convergence_norm(device, convergence_policy=physical.convergence_policy)
    binding = classical_dc_1d(device)
    def factory(t, side):
        voltage, generation = drive.values(t, side)
        return AssembledStorage1D(AssembledDC1DProblem(device, voltage, generation, binding))
    def assessment(step, u):
        data = bdf_accounting(electronic_view(step), u)
        data["assembled_full_residual"] = step.residual(u)
        finite = _finite_accounting(data)
        return Assessment(data, (GateOutcome("finite_original_accounting", finite,
                                            0.0 if finite else math.inf, 0.0),))
    def seed(step):
        old = step.history[0]
        previous = device.state(old.u, voltage_V=old.problem.voltage_V)
        return BackwardEuler1DStep(step.endpoint, previous, step.dt_s).initial_coordinates(
            physical.predictor, voltage_lift=True)
    return solve_bdf(factory, point.state.u,
        (float(requested.times_s[0]), float(requested.times_s[-1])),
        options=replace(options, wall_s=min(options.wall_s, physical.wall_s),
                        max_steps=min(options.max_steps, physical.max_steps)),
        settings=lambda step: base, convergence_norm=norm, assess=assessment, seed=seed,
        breakpoints_s=tuple(float(t) for t in drive.times_s[1:-1]),
        output_times_s=requested.times_s)


def assert_bits(left, right):
    left, right = np.asarray(left), np.asarray(right)
    assert left.shape == right.shape and left.dtype == right.dtype
    assert left.tobytes() == right.tobytes()


def assert_newton(left, right):
    if left is None or right is None:
        assert left is right
        return
    assert_bits(left.state, right.state)
    assert left.accepted == right.accepted and left.termination == right.termination
    assert left.iterations == right.iterations and left.gates == right.gates
    for a, b in zip(left.iterations, right.iterations):
        assert_bits(np.array([a.scaled_residual, a.step_inf, a.damping]),
                    np.array([b.scaled_residual, b.step_inf, b.damping]))
    assert left.start_digest == right.start_digest and left.message == right.message
    assert_bits(np.array([left.scaled_residual]), np.array([right.scaled_residual]))
    assert left.residual_evaluations == right.residual_evaluations
    assert left.jacobian_evaluations == right.jacobian_evaluations


def assert_accounting(left, right):
    for key, value in left.items():
        if isinstance(value, np.ndarray):
            assert_bits(value, right[key])
        elif isinstance(value, dict):
            assert_accounting(value, right[key])
        else:
            assert value == right[key]


def assert_bdf(left, right, device, drive):
    assert left.completed == right.completed, (left.termination, right.termination, right.error)
    assert left.termination == right.termination and left.error == right.error
    assert_bits(left.accepted_times_s, right.accepted_times_s)
    assert left.nonlinear_calls == right.nonlinear_calls
    assert left.evaluation_counts == right.evaluation_counts
    assert len(left.attempts) == len(right.attempts)
    for a, b in zip(left.attempts, right.attempts):
        assert (a.start_s, a.end_s, a.order, a.accepted, a.termination, a.estimator) == (
            b.start_s, b.end_s, b.order, b.accepted, b.termination, b.estimator)
        assert_bits(a.initial_coordinates, b.initial_coordinates)
        assert_bits(a.candidate_coordinates, b.candidate_coordinates)
        assert_bits(a.coefficients, b.coefficients)
        assert_bits(np.array([a.error_norm]), np.array([b.error_norm]))
        assert_newton(a.newton, b.newton)
        assert len(a.auxiliary_newton) == len(b.auxiliary_newton)
        for original, assembled in zip(a.auxiliary_newton, b.auxiliary_newton):
            assert_newton(original, assembled)
        if a.assessment is not None:
            assert a.assessment.checks == b.assessment.checks
            assert_accounting(a.assessment.accounting, b.assessment.accounting)
    for t, a, b in zip(left.accepted_times_s, left.accepted_coordinates, right.accepted_coordinates):
        assert_bits(a, b)
        voltage, _ = drive.values(t)
        old, new = device.state(a, voltage_V=voltage), device.state(b, voltage_V=voltage)
        for field in ("potential_V", "electron_density_m3", "hole_density_m3"):
            assert_bits(getattr(old, field), getattr(new, field))
    assert len(left.samples) == len(right.samples)
    for a, b in zip(left.samples, right.samples):
        assert (a.time_s, a.interpolant_order, a.accepted_step_endpoint) == (
            b.time_s, b.interpolant_order, b.accepted_step_endpoint)
        assert_bits(a.u, b.u)
        assert_bits(a.storage_value, b.storage_value)
        assert_newton(a.projection, b.projection)


def test_original_accounting_uses_assembled_public_observations(device):
    require_observations()
    n = device.node_count
    old_u = np.r_[device.reference_potential_V, np.zeros(2 * n)]
    u = old_u.copy()
    u[n + 1:2 * n - 1] += .01
    u[2 * n + 1:-1] -= .02
    previous = device.state(old_u, voltage_V=0.0)
    original = BackwardEuler1DStep(DC1DProblem(device, 0.0, 1.0), previous, 1e-7)
    assembled = BackwardEuler1DStep(AssembledStorage1D(AssembledDC1DProblem(device, 0.0, 1.0)), previous, 1e-7)
    assert_accounting(step_accounting(original, u), step_accounting(assembled, u))


@pytest.mark.parametrize("generation", [0.0, 1.0])
def test_fixed_step_assembled_BE_and_BDF1_are_bitwise(device, generation, monkeypatch):
    require_observations()
    point = initial(device, generation)
    requested, drive = protocol(generation)
    settings = BDFOptions(1e-7, max_step_s=1e-7, max_order=1, adaptive=False)
    original_BE = solve_transient(device, point, requested, options=controls(), initial_options=SteadyOptions())
    record("BE_" + str(generation), original_BE)
    original_BDF = solve_transient(device, point, requested, options=controls(), initial_options=SteadyOptions(),
                                  method="bdf", bdf_options=settings)
    record("BDF1_legacy_" + str(generation), original_BDF)
    assembled_BDF = run_assembled(device, point, requested, drive, settings)
    record("BDF1_assembled_" + str(generation), assembled_BDF)
    assert_bdf(original_BDF, assembled_BDF, device, drive)
    binding = classical_dc_1d(device)
    with monkeypatch.context() as patch:
        patch.setattr(transient, "DC1DProblem", lambda bound, voltage, light:
            AssembledStorage1D(AssembledDC1DProblem(bound, voltage, light, binding)))
        assembled_BE = solve_transient(device, point, requested, options=controls(), initial_options=SteadyOptions())
    record("BE_assembled_" + str(generation), assembled_BE)
    assert original_BE.completed and assembled_BE.completed
    assert_bits(original_BE.accepted_times_s, assembled_BE.accepted_times_s)
    for a, b, u in zip(original_BE.states, assembled_BE.states, assembled_BDF.accepted_coordinates):
        assert_bits(a.u, b.u)
        assert_bits(a.u, u)
        for field in ("potential_V", "electron_density_m3", "hole_density_m3"):
            assert_bits(getattr(a, field), getattr(b, field))
    for a, b, c in zip(original_BE.attempts, assembled_BE.attempts, assembled_BDF.accepted_steps):
        assert_newton(a.newton, b.newton)
        assert_newton(a.newton, c.newton)
        assert_accounting(a.accounting, b.accounting)
        assert_accounting(a.accounting, c.assessment.accounting)


@pytest.mark.parametrize("generation", [0.0, 1.0])
def test_adaptive_assembled_history_orders_rejections_and_dense_samples_are_bitwise(device, generation):
    require_observations()
    point = initial(device, generation)
    requested, drive = protocol(generation, np.array([0.0, .13, .31, 1.0]) * 4e-7)
    settings = BDFOptions(1e-7, max_step_s=2e-7, relative_tolerance=1e-5)
    original = solve_transient(device, point, requested, options=controls(), initial_options=SteadyOptions(),
                               method="bdf", bdf_options=settings, bdf_drive=drive)
    record("adaptive_legacy_" + str(generation), original)
    assembled = run_assembled(device, point, requested, drive, settings)
    record("adaptive_assembled_" + str(generation), assembled)
    assert_bdf(original, assembled, device, drive)
    record("parity_" + str(generation), {
        "completed": assembled.completed, "termination": assembled.termination,
        "accepted_steps": len(assembled.accepted_steps), "rejected_steps": len(assembled.rejected_steps),
        "samples": len(assembled.samples), "identical_step_sequence": True,
        "all_compared_arrays_bitwise": True, "maximum_ulp_difference": 0,
        "physical_controls": asdict(controls()), "LTE_controls": asdict(settings),
        "qualification": False})
