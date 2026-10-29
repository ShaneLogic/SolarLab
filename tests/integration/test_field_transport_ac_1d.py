"""CT AC derivative/drive/accounting checks without DC or AC solves.

Synthetic states are mathematical points only. Optional checksum-bound native
records are read without solving and retain their original acceptance identity.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from examples.research_1d import field_transport_ac as example
from examples.research_1d.field_transport import prepare_case
from examples.research_1d.ionic_multilayer import prepare_confined_dual_ion_device, prepare_confined_ion_device
from examples.research_1d.shared_site_ionic_fixture import prepare_confined_shared_site_ion_device
from solarlab.config.resolve_device import resolve_device
from solarlab.device.compile_1d import compile_device_1d
from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
from solarlab.device.inputs import DeviceInput
from solarlab.equations.dc_1d import DC1DProblem, _quasi_fermi_potential_tangents
from solarlab.equations.ionic_1d import Ionic1DProblem
from solarlab.equations.ionic_small_signal import build_ionic_small_signal
from solarlab.experiments.impedance import (
    _lifted_storage_drive, _moving_voltage_partial, _particle_perturbations, voltage_reference_direction,
)
from solarlab.solvers.linear import FrequencyOptions

Q = 1.602176634e-19


@pytest.fixture(scope="module", autouse=True)
def blas1():
    with threadpool_limits(limits=1):
        yield


def all_layer_ct(base, *, active=("n", "p")):
    document = base.to_input().editing_data()
    for layer in document["layers"]:
        for carrier in ("n", "p"):
            layer["parameters"]["v_sat_" + carrier] = 1e4 if carrier in active else 0.0
            layer["parameters"]["ct_beta_" + carrier] = 2.0
    return resolve_device(DeviceInput.model_validate(document), base.defaults, base.resources, sources=base.sources)


def electronic_coordinates(device):
    N = device.node_count
    phi = device.reference_potential_V + .012 * np.sin(np.linspace(0.0, 2 * np.pi, N))
    return np.r_[phi, np.linspace(-.15, .2, N), np.linspace(.1, -.25, N)]


def derivative(function, u, direction, step=1e-5):
    # Pair opposite samples first so invariant large residuals cancel exactly.
    return (8 * (function(u + step * direction) - function(u - step * direction))
            - (function(u + 2 * step * direction) - function(u - 2 * step * direction))) / (12 * step)


def unchanged_sg_particle_tangent(problem, u, x, voltage):
    """Exact pre-CT AC expression, retained solely for CT-off identity checks."""
    device, N = problem.device, problem.device.node_count
    fluxes = problem.fluxes(u)
    _, n, p = device.fields(u, voltage_V=problem.voltage_V)
    before, after = device.carrier_reference_energies(0.0), device.carrier_reference_energies(1.0)
    result = []
    for offset, density, flux, sign, reference in (
        (N, n, fluxes.electron, 1.0, after[0] - before[0]),
        (2 * N, p, fluxes.hole, -1.0, after[1] - before[1]),
    ):
        left, _ = _quasi_fermi_potential_tangents(flux, sign * np.diff(u[offset:offset + N]), sign, device.thermal_voltage_V)
        left_z = sign * flux.left_density_derivative * density[:-1]
        result.append(left * (x[:N - 1] - x[1:N])
            + left_z * (x[offset:offset + N - 1] - x[offset + 1:offset + N])
            + sign * flux.flux * (x[offset + 1:offset + N] + (x[1:N] + reference * voltage) / device.thermal_voltage_V))
    return tuple(result)


@pytest.mark.parametrize("active", [("n",), ("p",), ("n", "p")])
def test_complex_carrier_direction_requires_ct_chain_once(active):
    device = compile_device_1d(all_layer_ct(prepare_case(), active=active), intervals_per_layer=(12, 12, 12))
    problem = DC1DProblem(device, voltage_V=.002)
    u, N = electronic_coordinates(device), device.node_count
    rng = np.random.default_rng(1146)
    real = np.r_[.02 * rng.normal(size=N), rng.normal(size=2 * N)]
    imag = np.r_[.02 * rng.normal(size=N), rng.normal(size=2 * N)]
    actual = _particle_perturbations(problem, u, real + 1j * imag, 0.0)
    omitted = unchanged_sg_particle_tangent(problem, u, real + 1j * imag, 0.0)
    for index, name in enumerate(("electron", "hole")):
        def primal(value):
            return getattr(problem.fluxes(value), name).flux
        numerical = derivative(primal, u, real) + 1j * derivative(primal, u, imag)
        scale = np.maximum(abs(actual[index]), np.finfo(float).tiny)
        assert np.max(abs(numerical - actual[index]) / scale) < 2e-7
        if ("n", "p")[index] in active:
            assert np.max(abs(actual[index] - omitted[index]) / scale) > 1e-3
        else:
            np.testing.assert_array_equal(actual[index], omitted[index])


@pytest.mark.parametrize("zero_field", [False, True])
def test_ct_off_and_smooth_zero_field_ac_tangents(zero_field):
    device = compile_device_1d(all_layer_ct(prepare_case()), intervals_per_layer=(12, 12, 12))
    u = electronic_coordinates(device)
    if zero_field:
        u[:device.node_count] = 0.0
    else:
        device = replace(device, field_transport=None)
    problem = DC1DProblem(device, voltage_V=.002)
    x = np.random.default_rng(89).normal(size=u.size) * (1e-4 + 3e-4j)
    actual = _particle_perturbations(problem, u, x, 2e-6 + 1e-6j)
    expected = unchanged_sg_particle_tangent(problem, u, x, 2e-6 + 1e-6j)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)


def test_actual_moving_voltage_drive_and_lift_includes_boundary_field_chain():
    device = compile_device_1d(all_layer_ct(prepare_case()), intervals_per_layer=(12, 12, 12))
    N, voltage = device.node_count, .002
    problem = DC1DProblem(device, voltage_V=voltage)
    u = electronic_coordinates(device)
    c = voltage_reference_direction(device)
    lift = np.zeros(3 * N)
    lift[N - 1] = -device.voltage_polarity
    for offset in (N, 2 * N):
        for node in (0, N - 1):
            lift[offset + node] = -lift[node] / device.thermal_voltage_V + c[offset + node]
    direction = np.r_[.02 * np.cos(np.linspace(0, 2 * np.pi, N)), np.zeros(2 * N)]
    for delta in (np.zeros(3 * N), c, lift, direction):
        def primal(parameter):
            return DC1DProblem(device, voltage_V=voltage + parameter[0]).residual(u + parameter[0] * delta)
        numerical = derivative(primal, np.zeros(1), np.ones(1))
        analytic = _moving_voltage_partial(problem, u, delta)
        scale = np.maximum(abs(problem.jacobian(u)) @
            (abs(delta) + abs(c) + np.r_[np.full(N, .02), np.ones(2 * N)]), 1e-20)
        scale[N - 1] = max(scale[N - 1], 1.0)
        assert np.max(abs(numerical - analytic) / scale) < 2e-7
        merged = problem.jacobian(u) @ (delta - c)
        merged[N - 1] += device.voltage_polarity
        # Summing the nodal SG matrix loses some grouped uniform-mode digits;
        # the actual AC system is checked under its existing linear criterion.
        assert np.max(abs(analytic - merged) / scale) < FrequencyOptions().relative_tolerance
    np.testing.assert_allclose(_lifted_storage_drive(problem, u, lift),
                               problem.storage_jacobian(u) @ (c - lift), rtol=2e-14, atol=1e-12)
    flux = _particle_perturbations(problem, u, lift, 1.0)
    omitted = unchanged_sg_particle_tangent(problem, u, lift, 1.0)
    assert any(np.any(a != b) for a, b in zip(flux, omitted))


@pytest.fixture(scope="module", params=[(kind, all_layers) for kind in ("positive", "independent", "shared")
                                       for all_layers in (False, True)])
def ionic_case(request):
    kind, all_layers = request.param
    prepared = example.prepare_ionic_case(kind)
    if all_layers:
        prepared = all_layer_ct(prepared)
    device = compile_ionic_device_1d(prepared,
        intervals_per_layer=(12, 12, 12), ion_domain_layer="absorber",
        ion_domain_endpoints=("blocking", "blocking"))
    u = device.initial_state(voltage_V=.002).u.copy()
    u[:3 * device.node_count] = electronic_coordinates(device.electronic_component)
    for index, block in enumerate(device.ion_species_slices):
        u[block] += .07 * np.sin(np.linspace(.3 + index, 2.2 + index, device.ion_node_count))
    state = device.state(u, voltage_V=.002)
    return device, state, build_ionic_small_signal(device, state)


def test_ionic_ct_full_matrices_drive_and_shared_storage(ionic_case):
    device, state, operator = ionic_case
    N = device.node_count
    problem = Ionic1DProblem(device, voltage_V=state.voltage_V)
    np.testing.assert_array_equal(operator.jacobian.toarray(), problem.jacobian(state.u).toarray())
    np.testing.assert_array_equal(operator.storage_jacobian.toarray(), problem.storage_jacobian(state.u).toarray())
    if device.ion_site_contract == "shared_finite_sites":
        assert np.count_nonzero(operator.storage_jacobian[device.ion_species_slices[0], device.ion_species_slices[1]].data)
    V = 2e-6 + 1e-6j
    system = operator.system(1.0, voltage_phasor_V=V)
    lift = operator.coordinate_lift_per_V
    expected = _moving_voltage_partial(problem.electronic_problem, state.u[:3 * N], lift[:3 * N])
    np.testing.assert_array_equal(system.rhs[:3 * N], -expected * V)
    np.testing.assert_array_equal(system.rhs_rate[:3 * N], _lifted_storage_drive(problem.electronic_problem, state.u[:3 * N], lift[:3 * N]) * V)
    rng = np.random.default_rng(1148)
    direction = rng.normal(size=device.state_size) * .01
    direction[:N] *= device.electronic_component.thermal_voltage_V
    numeric = derivative(problem.residual, state.u, direction)
    scale = np.maximum(abs(operator.jacobian) @ abs(direction), 1e-20)
    assert np.max(abs(numeric - operator.jacobian @ direction) / scale) < 2e-7
    zero = operator.system(0.0)
    for name, block in zip(device.ion_species_names, device.ion_species_slices):
        expected = -np.asarray(operator.storage_jacobian[block, :].sum(axis=0)).ravel()
        np.testing.assert_array_equal(zero.jacobian[zero.constraint_rows[name], :].toarray()[0], expected)


def test_ionic_ct_complex_ports_signed_charge_and_maxwell_identity(ionic_case):
    device, state, operator = ionic_case
    N, V, f = device.node_count, 2e-6 + 1e-6j, 1.0
    rng = np.random.default_rng(1149)
    real = rng.normal(size=device.state_size) * .01
    imag = rng.normal(size=device.state_size) * .01
    real[:N] *= .02
    imag[:N] *= .02
    x = real + 1j * imag
    out = operator.observe(x, f, voltage_phasor_V=V, coordinate_form="fixed_reference_v1")
    problem = Ionic1DProblem(device, state.voltage_V)
    carriers = np.asarray(out["carrier_particle_fluxes_m2_s"])
    for index, name in enumerate(("electron", "hole")):
        def primal(value):
            return getattr(problem.electronic_problem.fluxes(value[:3 * N]), name).flux
        numerical = derivative(primal, state.u, real) + 1j * derivative(primal, state.u, imag)
        np.testing.assert_allclose(carriers[index, 1:-1], numerical, rtol=2e-7, atol=1e-12)
    expected_current = Q * (carriers[1] - carriers[0])
    expected_charge = Q * device.electronic_component.geometry.cell_measures_m * (
        out["hole_density_perturbation_m3"] - out["electron_density_perturbation_m3"])
    for index, name in enumerate(device.ion_species_names):
        sign = 1.0 if index == 0 else -1.0
        particles = out["ion_particle_fluxes_m2_s"][name]
        expected_current += sign * Q * particles
        expected_charge[device.ion_node_indices] += sign * Q * device.ion_geometry.cell_measures_m * out["ion_density_perturbation_m3"][name]
        assert particles[device.ion_node_indices[0]] == particles[device.ion_node_indices[-1] + 1] == 0.0
    np.testing.assert_array_equal(out["conduction_current_density_A_m2"], expected_current)
    np.testing.assert_array_equal(out["charge_inventory_perturbation_C_m2"], expected_charge)
    assert np.asarray(out["total_current_density_A_m2"]).shape == (N + 1,)
    scale = max(np.max(abs(out["total_current_density_A_m2"])),
                2 * np.pi * np.max(abs(expected_charge)), np.finfo(float).tiny)
    assert np.max(abs(out["maxwell_identity_error_A_m2"])) < 256 * np.finfo(float).eps * scale
    lifted = operator.observe(x, f, voltage_phasor_V=V)
    known_n, known_p = _particle_perturbations(problem.electronic_problem, state.u[:3 * N],
        operator.coordinate_lift_per_V[:3 * N], 1.0)
    unknown_n, unknown_p = _particle_perturbations(problem.electronic_problem, state.u[:3 * N], x[:3 * N], 0.0)
    np.testing.assert_array_equal(lifted["carrier_particle_fluxes_m2_s"][0, 1:-1], unknown_n + known_n * V)
    np.testing.assert_array_equal(lifted["carrier_particle_fluxes_m2_s"][1, 1:-1], unknown_p + known_p * V)
    fixed = operator.observe(x + (operator.coordinate_lift_per_V - operator.reference_direction_per_V) * V,
        f, voltage_phasor_V=V, coordinate_form="fixed_reference_v1")
    np.testing.assert_allclose(lifted["total_current_density_A_m2"], fixed["total_current_density_A_m2"], rtol=3e-12, atol=1e-12)
    frozen_x = x.copy()
    frozen_x[3 * N:] = 0.0
    frozen = operator.observe(frozen_x, f, voltage_phasor_V=V, ion_mode="frozen")
    assert np.count_nonzero(frozen["ionic_conduction_current_density_A_m2"]) == 0
    for name in device.ion_species_names:
        assert np.count_nonzero(frozen["ion_density_perturbation_m3"][name]) == 0
    assert frozen["physical_certificate_status"] == "not_evaluated"


@pytest.mark.parametrize("kind", ["positive", "independent", "shared"])
def test_no_ct_ionic_flux_tangent_keeps_exact_pre_ct_expression(kind):
    factory = {"positive": prepare_confined_ion_device, "independent": prepare_confined_dual_ion_device,
               "shared": prepare_confined_shared_site_ion_device}[kind]
    device = compile_ionic_device_1d(factory(), intervals_per_layer=(12, 12, 12),
        ion_domain_layer="absorber", ion_domain_endpoints=("blocking", "blocking"))
    state = device.initial_state(voltage_V=.002)
    operator = build_ionic_small_signal(device, state)
    x = np.random.default_rng(1172).normal(size=device.state_size) * (1e-4 + 3e-4j)
    expected = unchanged_sg_particle_tangent(operator.problem.electronic_problem,
        state.u[:3 * device.node_count], x[:3 * device.node_count], 2e-6 + 1e-6j)
    for actual, previous in zip(operator._carrier_particles(x, 2e-6 + 1e-6j), expected):
        np.testing.assert_array_equal(actual, previous)


def test_bound_pre_edit_no_ct_arrays_remain_bitwise_exact():
    directory = os.environ.get("SOLARLAB_CT_AC_BASELINE")
    if directory is None:
        pytest.skip("bind pre-edit CT-off arrays explicitly; no physics solve is needed")
    archive = Path(directory)
    metadata = json.loads((archive / "NoCTBaseline.json").read_text())
    assert hashlib.sha256((archive / "NoCTBaseline.npz").read_bytes()).hexdigest() == metadata["arrays_sha256"]
    with np.load(archive / "NoCTBaseline.npz") as arrays:
        from examples.research_1d.steady import load_device
        device = compile_device_1d(load_device(), intervals_per_layer=(12, 12, 12))
        assert device.identity == metadata["identities"]["ion_free"]
        problem = DC1DProblem(device, .002)
        u, x = arrays["ion_free_u"], arrays["ion_free_x"]
        n, p = _particle_perturbations(problem, u, x, 2e-6 + 1e-6j)
        for key, actual in (("fn", n), ("fp", p), ("partial", _moving_voltage_partial(problem, u, x.real)),
                            ("rate", _lifted_storage_drive(problem, u, x.real))):
            np.testing.assert_array_equal(actual, arrays["ion_free_" + key])
            assert actual.tobytes() == arrays["ion_free_" + key].tobytes()
        factories = {"positive": prepare_confined_ion_device, "independent": prepare_confined_dual_ion_device,
                     "shared": prepare_confined_shared_site_ion_device}
        for kind, factory in factories.items():
            ionic = compile_ionic_device_1d(factory(), intervals_per_layer=(12, 12, 12),
                ion_domain_layer="absorber", ion_domain_endpoints=("blocking", "blocking"))
            assert ionic.identity == metadata["identities"][kind]
            state = ionic.state(arrays[kind + "_u"], voltage_V=.002)
            operator = build_ionic_small_signal(ionic, state)
            system = operator.system(1.0, voltage_phasor_V=2e-6 + 1e-6j)
            out = operator.observe(arrays[kind + "_x"], 1.0, voltage_phasor_V=2e-6 + 1e-6j)
            for key, actual in (("J", operator.jacobian.toarray()), ("K", operator.storage_jacobian.toarray()),
                ("rhs", system.rhs), ("rhs_rate", system.rhs_rate),
                ("conduction", out["conduction_current_density_A_m2"]),
                ("displacement", out["displacement_current_density_A_m2"]),
                ("charge", out["charge_inventory_perturbation_C_m2"]), ("carrier", out["carrier_particle_fluxes_m2_s"])):
                np.testing.assert_array_equal(actual, arrays[kind + "_" + key])
                assert actual.tobytes() == arrays[kind + "_" + key].tobytes()


def test_ct_operator_pins_actual_field_law_and_rejects_changed_source(ionic_case, monkeypatch, tmp_path):
    from solarlab.equations import field_transport_1d
    device, _state, operator = ionic_case
    assert "solarlab.equations.field_transport_1d" in operator.source_sha256
    assert "solarlab.physics.transport.field_mobility" in operator.source_sha256
    assert "solarlab.physics.transport.field_mobility_tangents" in operator.source_sha256
    path = tmp_path / "changed_field_transport.py"
    path.write_text("changed source\n")
    monkeypatch.setattr(field_transport_1d, "__file__", str(path))
    with pytest.raises(RuntimeError, match="physical source changed"):
        operator.observe(np.zeros(device.state_size, dtype=complex), 1.0)


@pytest.mark.parametrize("kind", ["ion-free", "shared"])
def test_saved_native_ct_ac_reads_without_model_or_solver(kind, monkeypatch):
    key = "SOLARLAB_CT_AC_" + kind.upper().replace("-", "_")
    path, sha = os.environ.get(key + "_RESULT"), os.environ.get(key + "_SHA256")
    if path is None or sha is None:
        pytest.skip("bind one saved native witness explicitly; never repeat its solve")
    def forbidden(*args, **kwargs):
        raise AssertionError("saved reading called a compiler or a solver")
    for name in ("compile_device_1d", "compile_ionic_device_1d", "solve_impedance", "solve_ionic_impedance"):
        monkeypatch.setattr(example, name, forbidden)
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha
    record = example.read_result(Path(path), sha)
    assert record["summary"]["kind"] == kind and record["summary"]["accepted"]
    assert not record["summary"]["qualification"]
    assert record["summary"]["physical_certificate_status"] == "not_evaluated"
    if kind == "ion-free":
        result = record["result"]
        scale = np.max(abs(result["total_current_density_A_m2"] / result["voltage_phasor_V"]), axis=1)
        assert np.all(result["maxwell_spread_A_m2_V"] <= 1e-8 + 1e-10 * scale)
        assert "solarlab.equations.field_transport_1d" in result["source_sha256"]
    else:
        assert len(record["result"]["attempts"]) == len(record["summary"]["requested_frequencies_Hz"])
        assert all(item["accepted"] and all(check["passed"] for check in item["checks"])
                   for item in record["result"]["attempts"])
        for attempt in record["result"]["attempts"]:
            observed = attempt["observation"]
            particles = observed["carrier_particle_fluxes_m2_s"]
            expected = Q * (particles[1] - particles[0])
            expected += Q * observed["ion_particle_fluxes_m2_s"]["positive"]
            expected -= Q * observed["ion_particle_fluxes_m2_s"]["negative"]
            np.testing.assert_array_equal(observed["conduction_current_density_A_m2"], expected)
            total = observed["total_current_density_A_m2"]
            np.testing.assert_array_equal(total, expected + observed["displacement_current_density_A_m2"])
            np.testing.assert_array_equal(observed["terminal_current_density_A_m2"], total[[0, -1]])
            assert total.shape == (38,) and particles.shape == (2, 38)
            assert observed["input_current_density_A_m2"] == total[0]
            assert observed["ion_mode"] == "mobile"
            for name in ("positive", "negative"):
                assert np.any(observed["ion_density_perturbation_m3"][name] != 0.0)
                assert np.all(observed["base_ion_density_m3"][name] > 0.0)
            voltage = record["result"]["voltage_phasor_V"]
            normalized = total / voltage
            spread, scale = np.max(abs(normalized - normalized[0])), np.max(abs(normalized))
            assert spread <= 1e-8 + 1e-10 * scale

