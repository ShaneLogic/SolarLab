"""Ionic AC operators and physical port derivatives at recorded point coordinates.

The seeds are copied from the listed accepted artifacts. These hermetic tests
use them as mathematical points in the current compiled environment; they do
not relabel/rebind historical acceptance. Main independently admits the full
saved artifact and rechecks its original physical gates before a response.
"""

from __future__ import annotations

from dataclasses import replace
import ast
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from examples.research_1d.ionic_multilayer import (
    prepare_confined_ion_device,
    prepare_confined_dual_ion_device,
)
from examples.research_1d.shared_site_ionic_fixture import (
    prepare_confined_shared_site_ion_device,
)
from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
from solarlab.equations.ionic_1d import Ionic1DProblem
from solarlab.equations.ionic_small_signal import build_ionic_small_signal
from solarlab.experiments.impedance import (
    _moving_voltage_partial,
    _lifted_storage_drive,
    _particle_perturbations,
)
from examples.research_1d.ionic_residual import prepare_single_ion_device

Q = 1.602176634e-19


@pytest.fixture(scope="module", autouse=True)
def blas1():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(scope="module", params=["positive", "independent", "shared"])
def case(request):
    kind = request.param
    prepared = {
        "positive": prepare_confined_ion_device,
        "independent": prepare_confined_dual_ion_device,
        "shared": prepare_confined_shared_site_ion_device,
    }[kind]()
    device = compile_ionic_device_1d(
        prepared,
        intervals_per_layer=(12, 12, 12),
        ion_domain_layer="absorber",
        ion_domain_endpoints=("blocking", "blocking"),
    )
    sample = RECORDED[kind]
    state = device.state(np.asarray(sample["u"]), voltage_V=sample["voltage_V"])
    return device, state, build_ionic_small_signal(device, state)


def test_production_matrices_and_grouped_lift_match_existing_core_math(case):
    device, state, operator = case
    problem = Ionic1DProblem(device, voltage_V=state.voltage_V)
    N = device.node_count
    np.testing.assert_array_equal(
        operator.jacobian.toarray(), problem.jacobian(state.u).toarray()
    )
    np.testing.assert_array_equal(
        operator.storage_jacobian.toarray(), problem.storage_jacobian(state.u).toarray()
    )
    V = 2e-6 + 1e-6j
    system = operator.system(1.0, voltage_phasor_V=V, ion_mode="mobile")
    assert system.coordinate_form == "moving_boundary_lift_v1"
    assert tuple(system.original_rows_enforced) == tuple(range(device.state_size))
    assert len(system.fixed_coordinates) == 6 and all(
        value == 0.0 for value in system.fixed_coordinates.values()
    )
    np.testing.assert_array_equal(
        operator.reference_direction_per_V[3 * N :], np.zeros(device.state_size - 3 * N)
    )
    lift = operator.coordinate_lift_per_V
    expected = _moving_voltage_partial(
        problem.electronic_problem, state.u[: 3 * N], lift[: 3 * N]
    )
    expected_rate = _lifted_storage_drive(
        problem.electronic_problem, state.u[: 3 * N], lift[: 3 * N]
    )
    np.testing.assert_array_equal(system.rhs[: 3 * N], -expected * V)
    np.testing.assert_array_equal(system.rhs_rate[: 3 * N], expected_rate * V)
    np.testing.assert_array_equal(
        system.rhs[3 * N :], -(operator.jacobian[3 * N :, :N] @ lift[:N]) * V
    )
    np.testing.assert_array_equal(
        system.rhs_rate[3 * N :], np.zeros(device.state_size - 3 * N)
    )


def test_zero_frequency_full_inventory_rows_and_frozen_clamps(case):
    device, state, operator = case
    N = device.node_count
    system = operator.system(0.0, voltage_phasor_V=2e-6 + 1e-6j)
    for species, block in zip(device.ion_species_names, device.ion_species_slices):
        row = system.constraint_rows[species]
        expected = -np.asarray(operator.storage_jacobian[block, :].sum(axis=0)).ravel()
        np.testing.assert_array_equal(system.jacobian[row, :].toarray()[0], expected)
        assert system.rhs[row] == 0.0 and system.rhs_rate[row] == 0.0
        assert system.storage_jacobian[row, :].nnz == 0
    frozen = operator.system(1.0, ion_mode="frozen")
    assert frozen.original_rows_enforced == tuple(range(3 * N))
    for row in range(3 * N, device.state_size):
        assert frozen.jacobian[row, row] == 1.0 and frozen.jacobian[row, :].nnz == 1
        assert frozen.storage_jacobian[row, :].nnz == 0
        assert (
            frozen.fixed_coordinates[row]
            == frozen.rhs[row]
            == frozen.rhs_rate[row]
            == 0.0
        )
    x = np.zeros(device.state_size, dtype=complex)
    x[:N] = 0.1 * np.sin(np.linspace(0.0, np.pi, N))
    out = operator.observe(x, 1.0, voltage_phasor_V=0.001 + 0.002j, ion_mode="frozen")
    assert np.count_nonzero(out["ionic_conduction_current_density_A_m2"]) == 0
    for species, density in zip(
        device.ion_species_names,
        (state.positive_ion_density_m3, state.negative_ion_density_m3),
    ):
        assert np.count_nonzero(out["ion_density_perturbation_m3"][species]) == 0
        np.testing.assert_array_equal(out["base_ion_density_m3"][species], density)
    assert out["physical_certificate_status"] == "not_evaluated"


def test_complex_particle_tangents_have_independent_directional_check(case):
    device, state, operator = case
    N = device.node_count
    problem = Ionic1DProblem(device, voltage_V=state.voltage_V)
    rng = np.random.default_rng(40619)
    real = rng.uniform(-0.02, 0.02, device.state_size)
    imag = rng.uniform(-0.02, 0.02, device.state_size)
    real[:N] *= device.electronic_component.thermal_voltage_V
    imag[:N] *= device.electronic_component.thermal_voltage_V
    h = 5e-5
    rp, rm = state.u + h * real, state.u - h * real
    ip, im = state.u + h * imag, state.u - h * imag
    effective = (rp - rm) / (2 * h) + 1j * (ip - im) / (2 * h)
    out = operator.observe(effective, 1.0, coordinate_form="fixed_reference_v1")
    erp, erm = (
        problem.electronic_problem.fluxes(rp[: 3 * N]),
        problem.electronic_problem.fluxes(rm[: 3 * N]),
    )
    eip, eim = (
        problem.electronic_problem.fluxes(ip[: 3 * N]),
        problem.electronic_problem.fluxes(im[: 3 * N]),
    )
    for index, name in enumerate(("electron", "hole")):
        finite = (getattr(erp, name).flux - getattr(erm, name).flux) / (2 * h) + 1j * (
            getattr(eip, name).flux - getattr(eim, name).flux
        ) / (2 * h)
        analytic = np.asarray(out["carrier_particle_fluxes_m2_s"])[index, 1:-1]
        error = float(
            np.max(
                abs(finite - analytic) / np.maximum(abs(analytic), np.finfo(float).tiny)
            )
        )
        print(f"{name}: complex physical tangent error {error:.9g}")
        assert error < 2e-7
    for species in device.ion_species_names:
        finite = (
            problem.ion_fluxes(rp, species=species)
            - problem.ion_fluxes(rm, species=species)
        ) / (2 * h)
        finite = finite + 1j * (
            problem.ion_fluxes(ip, species=species)
            - problem.ion_fluxes(im, species=species)
        ) / (2 * h)
        analytic = out["ion_particle_fluxes_m2_s"][species][
            device.ion_node_indices[:-1] + 1
        ]
        error = float(
            np.max(
                abs(finite - analytic) / np.maximum(abs(analytic), np.finfo(float).tiny)
            )
        )
        print(f"{species}: complex physical tangent error {error:.9g}")
        assert error < 2e-7


def test_lifted_current_observation_preserves_small_coordinates_and_linear_amplitude(
    case,
):
    device, state, operator = case
    N = device.node_count
    rng = np.random.default_rng(40704)
    x = (
        rng.uniform(-1, 1, device.state_size) * 1e-9
        + 1j * rng.uniform(-1, 1, device.state_size) * 1e-9
    )
    for index in operator.system(1.0).fixed_coordinates:
        x[index] = 0.0
    V = 2e-6 + 1e-6j
    out = operator.observe(x, 1.0, voltage_phasor_V=V)
    np.testing.assert_array_equal(out["solve_coordinate_perturbation"], x)
    assert out["coordinate_form"] == "moving_boundary_lift_v1"
    np.testing.assert_array_equal(
        out["moving_coordinate_perturbation"], x + operator.coordinate_lift_per_V * V
    )
    expected_n, expected_p = _particle_perturbations(
        Ionic1DProblem(device, state.voltage_V).electronic_problem,
        state.u[: 3 * N],
        x[: 3 * N],
        0.0,
    )
    known_n, known_p = _particle_perturbations(
        Ionic1DProblem(device, state.voltage_V).electronic_problem,
        state.u[: 3 * N],
        operator.coordinate_lift_per_V[: 3 * N],
        1.0,
    )
    np.testing.assert_array_equal(
        out["carrier_particle_fluxes_m2_s"][0, 1:-1], expected_n + known_n * V
    )
    np.testing.assert_array_equal(
        out["carrier_particle_fluxes_m2_s"][1, 1:-1], expected_p + known_p * V
    )
    assert np.asarray(out["total_current_density_A_m2"]).shape == (N + 1,)
    scale = max(
        float(np.max(abs(out["total_current_density_A_m2"]))),
        float(np.max(abs(out["charge_inventory_perturbation_C_m2"]))) * 2 * np.pi,
        np.finfo(float).tiny,
    )
    assert (
        np.max(abs(out["maxwell_identity_error_A_m2"]))
        <= 256 * np.finfo(float).eps * scale
    )
    factor = 0.4 - 0.7j
    scaled = operator.observe(x * factor, 1.0, voltage_phasor_V=V * factor)
    np.testing.assert_allclose(
        scaled["total_current_density_A_m2"],
        out["total_current_density_A_m2"] * factor,
        rtol=2e-13,
        atol=1e-16,
    )


def test_series_dielectric_displacement_has_correct_units_and_passive_sign(case):
    device, _state, operator = case
    N, frequency, V = device.node_count, 1000.0, 2e-6 + 1e-6j
    inverse_capacitances = (
        1.0 / device.electronic_component.poisson.face_capacitance_F_m2
    )
    capacitance = 1.0 / float(np.sum(inverse_capacitances))
    phi = (
        -device.electronic_component.voltage_polarity
        * V
        * np.r_[0.0, np.cumsum(inverse_capacitances)]
        / np.sum(inverse_capacitances)
    )
    x = np.zeros(device.state_size, dtype=complex)
    x[:N], x[N : 2 * N], x[2 * N : 3 * N] = (
        phi,
        -phi / device.electronic_component.thermal_voltage_V,
        -phi / device.electronic_component.thermal_voltage_V,
    )
    out = operator.observe(
        x,
        frequency,
        voltage_phasor_V=V,
        ion_mode="frozen",
        coordinate_form="fixed_reference_v1",
    )
    expected = (
        2j
        * np.pi
        * frequency
        * device.electronic_component.voltage_polarity
        * capacitance
        * V
    )
    np.testing.assert_allclose(
        out["displacement_current_density_A_m2"],
        np.full(N + 1, expected),
        rtol=2e-13,
        atol=1e-20,
    )


def test_domain_endpoint_lift_retains_ion_force_and_frozen_rhs_is_zero():
    device = compile_ionic_device_1d(
        prepare_single_ion_device(diffusion_only=True), intervals_per_layer=12
    )
    # This is an explicitly mathematical trial point, not an accepted baseline.
    operator = build_ionic_small_signal(device, device.initial_state(voltage_V=0.002))
    N, V = device.node_count, 2e-6 + 1e-6j
    mobile = operator.system(1.0, voltage_phasor_V=V)
    expected = (
        -(operator.jacobian[3 * N :, :N] @ operator.coordinate_lift_per_V[:N]) * V
    )
    assert np.count_nonzero(expected) > 0
    np.testing.assert_array_equal(mobile.rhs[3 * N :], expected)
    frozen = operator.system(1.0, voltage_phasor_V=V, ion_mode="frozen")
    assert (
        np.count_nonzero(frozen.rhs[3 * N :])
        == np.count_nonzero(frozen.rhs_rate[3 * N :])
        == 0
    )


def test_bound_state_refusals_and_equation_layer_has_no_experiment_import(case):
    device, state, _operator = case
    with pytest.raises(ValueError, match="complete compiled"):
        build_ionic_small_signal(device, None)
    with pytest.raises(ValueError, match="same complete"):
        build_ionic_small_signal(device, replace(state, device_sha256="foreign"))
    with pytest.raises(ValueError, match="operating voltage"):
        from solarlab.equations.ionic_small_signal import IonicSmallSignal1D

        IonicSmallSignal1D(
            Ionic1DProblem(device, voltage_V=state.voltage_V + 0.1), state
        )
    path = (
        Path(__file__).resolve().parents[2]
        / "src/solarlab/equations/ionic_small_signal.py"
    )
    tree = ast.parse(path.read_text())
    names = [
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    ]
    assert not any(name.startswith("solarlab.experiments") for name in names)


# Source-pinned coordinate samples; no DC/history solver is called here.
RECORDED = {
    "positive": {
        "artifact": "IonicMultilayerV1/Solver/FirstRun.json",
        "artifact_sha256": "03362c7ed68790818da8df0897b00feb89093815fe4d1133335f22f959a59c76",
        "voltage_V": 0.002,
        "u": [
            1.726341228922086e-33,
            1.35763438129151e-10,
            5.170031915191985e-10,
            1.9783166159875546e-09,
            1.6378825607028234e-08,
            4.5607970379197223e-07,
            3.1393596890898953e-05,
            0.00258845058298586,
            0.11014864838343513,
            0.4672797886711301,
            0.6718459911531932,
            0.7592870933261524,
            0.77830034633792,
            0.790699109043159,
            0.8084904256822595,
            0.8214392733770689,
            0.8242451689762729,
            0.8243786504201266,
            0.8243805347239643,
            0.8243905182781228,
            0.8251248142700244,
            0.8403070127646914,
            0.891593634679238,
            0.9252751153912254,
            0.9530354886886763,
            0.963803205009102,
            0.9914793176164602,
            1.055835021617463,
            1.165634572287429,
            1.2114559566503063,
            1.2145941999837604,
            1.214702720372403,
            1.2147069141022844,
            1.2147072416723543,
            1.2147073060102498,
            1.2147073273209972,
            1.2147073351557045,
            -0.07736345414366722,
            -0.07691129203351975,
            -0.07570230693833753,
            -0.07255754526768153,
            -0.06493114278361511,
            -0.04920226667756155,
            -0.02539774589261314,
            -0.003211742610037348,
            -7.721089062204673e-06,
            -1.2261881389542458e-07,
            -3.204314378554467e-08,
            -6.295741639015783e-10,
            -4.938626237058874e-12,
            -4.928283390579807e-12,
            -4.904801537530673e-12,
            -4.856485436500645e-12,
            -4.730676567609592e-12,
            -4.328876819293277e-12,
            -3.305315005264167e-12,
            -1.7905517964074103e-12,
            -5.643434236740368e-13,
            -7.383269728644053e-14,
            -7.81298398563634e-15,
            -3.327899275279077e-15,
            -4.9800978296520695e-17,
            -3.393624686995816e-17,
            -1.292673114785592e-17,
            -1.6597662011618738e-18,
            -2.240352685863172e-19,
            -1.3758210789692726e-19,
            -9.20033219759228e-20,
            -4.9224893189422746e-20,
            -2.156253930628746e-20,
            -8.303768086873162e-21,
            -2.86566764981295e-21,
            -7.795595807157868e-22,
            3.737807963910957e-32,
            4.891945003028334e-31,
            7.795594650061691e-22,
            2.8656660679787163e-21,
            8.303753868613343e-21,
            2.1562397847525512e-20,
            4.9222310332488057e-20,
            9.192969784413905e-20,
            1.3684746918796194e-19,
            6.00590861081696e-19,
            6.793971705834936e-14,
            1.329486292921504e-10,
            3.5128841566837637e-09,
            1.1742400213148171e-07,
            2.384567621269184e-07,
            8.080529929845579e-07,
            3.3689218623660863e-06,
            1.1234090885176946e-05,
            2.6344688418935814e-05,
            4.377370381784362e-05,
            5.398759214186359e-05,
            5.679373253856814e-05,
            5.718787779703405e-05,
            5.7226777939753717e-05,
            5.72334000488287e-05,
            5.723905868912047e-05,
            5.7286208610626305e-05,
            5.7487018202913894e-05,
            5.999308238180836e-05,
            0.0003007362876126939,
            0.006258035410632555,
            0.027355867560817537,
            0.05025559006028201,
            0.06539912909964189,
            0.07273909465433637,
            0.07576515202120172,
            0.0769284081586817,
            0.07736345414366727,
            1.7824589638597863,
            1.3028534088673012,
            0.6146545543850441,
            0.11377076195968513,
            0.005233874199219112,
            7.058141891614883e-05,
            -2.30670785859216e-06,
            -0.00038848782502085184,
            -0.028792324973700448,
            -0.6160659834948153,
            -2.599921094829622,
            -3.9027789391059504,
            -4.976598122410182,
        ],
    },
    "independent": {
        "artifact": "DualIonsV1/Consumers/TestEvidence/dual_wave0/Steady.json",
        "artifact_sha256": "8a0670519a462a17f48850264d333b341156f14547eb4c62aae34a039446c18e",
        "voltage_V": 0.002,
        "u": [
            -1.8262864531690207e-27,
            1.7483854681843299e-10,
            6.658058002064729e-10,
            2.547710845450957e-09,
            2.109293838266946e-08,
            5.873473789751932e-07,
            4.042912143775069e-05,
            0.0033328752185633683,
            0.13991452125185158,
            0.5139479774173015,
            0.7254467376490491,
            0.8155472406156137,
            0.8351170184966139,
            0.8479290763431812,
            0.8662197693410394,
            0.8779799390574241,
            0.8797808892586558,
            0.8798369773313023,
            0.8798374738634578,
            0.8798380584227686,
            0.879904192004084,
            0.8820264175768906,
            0.8956306064361608,
            0.91491725371838,
            0.9382174337350392,
            0.9493164073200034,
            0.977878936915973,
            1.0445447965423516,
            1.1599409527161961,
            1.2110540345633964,
            1.214580182582754,
            1.2147021485508993,
            1.214706861928852,
            1.2147072300886665,
            1.2147073023987844,
            1.2147073263501846,
            1.2147073351557045,
            -0.07736345414366722,
            -0.0769048565523072,
            -0.07567867884378898,
            -0.07248929510506778,
            -0.0647552689459054,
            -0.04880707827031363,
            -0.024681384357987602,
            -0.0025148002515792095,
            -2.623208039805906e-06,
            -1.1695598753523824e-07,
            -2.9347172719830132e-08,
            -2.333002126483853e-10,
            -2.1352469776147556e-13,
            -2.107891625866937e-13,
            -2.0664177313169892e-13,
            -2.0049444642148362e-13,
            -1.8854942644697286e-13,
            -1.624135271465421e-13,
            -1.16574955927206e-13,
            -6.400056259516793e-14,
            -2.638115558683348e-14,
            -8.132830779938804e-15,
            -2.425363029712769e-15,
            -1.2469017843554613e-15,
            -2.076353711493025e-17,
            -1.394277414843881e-17,
            -5.097392234967338e-18,
            -5.750100383108518e-19,
            -5.80827000958965e-20,
            -3.363873535172188e-20,
            -2.243452565528009e-20,
            -1.2001693536298751e-20,
            -5.257203210495746e-21,
            -2.0245540429571433e-21,
            -6.986804722661844e-22,
            -1.9006261595855892e-22,
            3.230104456207033e-27,
            -1.5091824440004506e-28,
            1.9006566325938673e-22,
            6.986831185578038e-22,
            2.024553236257664e-21,
            5.257167723100546e-21,
            1.2000999141212022e-20,
            2.241541179292412e-20,
            3.3530635167573076e-20,
            3.1814300433404083e-19,
            9.617996987780014e-14,
            2.4926671637626254e-10,
            6.50792942201467e-09,
            1.2917239428098061e-07,
            2.612776948908769e-07,
            8.960348581219018e-07,
            3.716447266434374e-06,
            1.2071269348483931e-05,
            2.7888188900414033e-05,
            4.6370408313256e-05,
            5.7632261181900665e-05,
            6.107956344635342e-05,
            6.168354672638855e-05,
            6.175797978160718e-05,
            6.17663283727639e-05,
            6.177126885517121e-05,
            6.180893895231241e-05,
            6.195852274407256e-05,
            6.36395990883718e-05,
            0.0002498994390202662,
            0.0057591153536556096,
            0.026885692028135826,
            0.04999433286664253,
            0.06528286291523827,
            0.07269398682308748,
            0.0757495374886833,
            0.07692415547992137,
            0.07736345414366727,
            1.704699769481798,
            1.2091072446323405,
            0.50159165013449,
            0.04668797484685288,
            -0.022975889307157448,
            -0.02514547282525004,
            -0.025164679546572637,
            -0.02518729131029198,
            -0.027745452453015088,
            -0.10983680284518839,
            -0.6360703233331564,
            -1.382111149634668,
            -2.283402353760783,
            -1.7721346857142914,
            -1.2765421608696599,
            -0.5690265663720515,
            -0.11412289108441555,
            -0.04445902693040522,
            -0.04228944341231246,
            -0.04227023669098994,
            -0.04224762492727078,
            -0.03968946378454746,
            0.04240188660762597,
            0.568635407095594,
            1.3146762333971174,
            2.21596743752612,
        ],
    },
    "shared": {
        "artifact": "SharedSiteIonsV1/Consumers/TestEvidence/shared_wave0/Steady.json",
        "artifact_sha256": "71a51353822f5469bd827604336b1a78a4a0043f8c07bd849cf42c6e515e342c",
        "voltage_V": 0.002,
        "u": [
            2.3141431697201916e-32,
            1.4744430144611004e-10,
            5.614852924455838e-10,
            2.1485276900782377e-09,
            1.7788032390719166e-08,
            4.953200270462432e-07,
            3.4094623718807324e-05,
            0.0028110111341600077,
            0.11913405010846251,
            0.4817498254461791,
            0.6885655747761281,
            0.7768696263666892,
            0.7960634662282992,
            0.8105757070510617,
            0.8406709253529183,
            0.8696005886150102,
            0.8743576706026527,
            0.8745099568352801,
            0.8745113376914929,
            0.8745121746112723,
            0.8746040082399621,
            0.8774773500026521,
            0.8959074316424414,
            0.9217195601330112,
            0.9483050726295317,
            0.9591802790764008,
            0.9871440246517139,
            1.0522493198105154,
            1.163863350492509,
            1.211331514981567,
            1.2145898607233618,
            1.2147025433591816,
            1.2147068979514708,
            1.214707238086507,
            1.2147073048922843,
            1.2147073270204725,
            1.2147073351557045,
            -0.07736345414366722,
            -0.07690915711800649,
            -0.07569446847614306,
            -0.072534903394289,
            -0.06487279481875548,
            -0.049071147783548326,
            -0.025159890821373043,
            -0.0029697190292101074,
            -5.511537396250848e-06,
            -1.2105811286326811e-07,
            -3.146732983022008e-08,
            -5.292652193280678e-10,
            -2.3449539302753506e-13,
            -2.277589047991301e-13,
            -2.1843253917952231e-13,
            -2.097591929304446e-13,
            -1.983518027932077e-13,
            -1.735504942343674e-13,
            -1.2700879274048146e-13,
            -7.029704814646546e-14,
            -2.818924959844184e-14,
            -7.697035912354135e-15,
            -1.8533211109081857e-15,
            -8.759616843423336e-16,
            -1.3450870218594359e-17,
            -9.121965168787189e-18,
            -3.428130342141287e-18,
            -4.2173079875460993e-19,
            -5.19564019555041e-20,
            -3.13514451145258e-20,
            -2.09478563958103e-20,
            -1.1207368664997364e-20,
            -4.909281457372797e-21,
            -1.8905717099347537e-21,
            -6.524447559136005e-22,
            -1.7748727855068435e-22,
            -1.1474451923396768e-32,
            -2.8644835890260836e-29,
            1.7748722231697178e-22,
            6.52444353038017e-22,
            1.8905683216968884e-21,
            4.9092480373104136e-21,
            1.1206760813204654e-20,
            2.093073823208067e-20,
            3.120285565277799e-20,
            1.700392941843308e-19,
            2.6662913704670368e-14,
            5.716266038857495e-11,
            1.5209076169500929e-09,
            3.9753460041695395e-08,
            8.091316196821761e-08,
            3.308972336765008e-07,
            2.2137098425272606e-06,
            1.0049340144946106e-05,
            2.574806397960089e-05,
            4.409306036269149e-05,
            5.521794388185516e-05,
            5.8578008416836286e-05,
            5.9147514142568677e-05,
            5.921363251125166e-05,
            5.9221125996256927e-05,
            5.922651002669932e-05,
            5.927008617713658e-05,
            5.945163842476554e-05,
            6.165397765986442e-05,
            0.0002837618004808212,
            0.0060966567796281925,
            0.027205489455603648,
            0.050172079159770365,
            0.06536196734638952,
            0.07272467738047048,
            0.07576016138881073,
            0.07692704894603192,
            0.07736345414366727,
            2.99941188890219,
            2.438053350195366,
            1.2739183296757008,
            0.15486899109141214,
            -0.029143156012909045,
            -0.0350338505002003,
            -0.03508726440335179,
            -0.03511963790583874,
            -0.038671921266832394,
            -0.14981774311530763,
            -0.862725131017246,
            -1.8611828404325679,
            -2.8895563788877987,
            -3.0735623288824425,
            -2.5122037901758265,
            -1.3480687696561517,
            -0.22901943107186254,
            -0.04500728396754061,
            -0.03911658948024743,
            -0.039063175577098604,
            -0.0390308020746126,
            -0.03547851871361622,
            0.07566730313485945,
            0.7885746910367966,
            1.7870324004521159,
            2.8154059389073476,
        ],
    },
}
