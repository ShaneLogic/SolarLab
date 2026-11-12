"""Integration and M-class parity tests for solarlab.experiments.identifiability."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from perovskite_sim.experiments import identifiability as legacy_id
from solarlab.contracts.units import ELEMENTARY_CHARGE_C
from solarlab.device.compile_1d import compile_device_1d
from solarlab.experiments import identifiability as new_id
from solarlab.experiments.solarscale import (
    BOLTZMANN_CONSTANT_J_K,
    prepare_solarscale_device,
)
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions, solve_steady_state


@pytest.mark.parametrize(
    ("family", "estimated", "expect_identifiable", "expect_rank"),
    [
        (
            "recombination_only",
            ("trap_density_cm2", "capture_cross_section_scale", "calibration_factor"),
            False,
            1,
        ),
        (
            "recombination_plus_charge",
            ("trap_density_cm2", "capture_cross_section_scale", "calibration_factor"),
            False,
            2,
        ),
        (
            "recombination_plus_charge",
            ("trap_density_cm2", "capture_cross_section_scale"),
            True,
            2,
        ),
    ],
)
def test_interface_srh_identifiability_m_class_parity(
    family: new_id.ObservableFamily,
    estimated: tuple[new_id.ParameterName, ...],
    expect_identifiable: bool,
    expect_rank: int,
) -> None:
    proto_old = legacy_id.build_interface_srh_identifiability_protocol(
        observable_family=family,
        estimated_parameters=estimated,
        carrier_condition_count=5,
        synthetic_noise_sigma_multiplier=0.1,
        noise_seed=1729,
    )
    proto_new = new_id.build_interface_srh_identifiability_protocol(
        observable_family=family,
        estimated_parameters=estimated,
        carrier_condition_count=5,
        synthetic_noise_sigma_multiplier=0.1,
        noise_seed=1729,
    )
    assert proto_new.sha256 == proto_old.sha256

    res_old = legacy_id.run_interface_srh_identifiability(proto_old)
    res_new = new_id.run_interface_srh_identifiability(proto_new)

    assert res_new.protocol_sha256 == res_old.protocol_sha256
    assert res_new.mapping_sha256 == res_old.mapping_sha256
    assert res_new.singular_values == res_old.singular_values
    assert res_new.fisher_information == res_old.fisher_information
    assert res_new.numerical_rank == expect_rank == res_old.numerical_rank
    assert res_new.parameters_identifiable is expect_identifiable
    assert res_new.parameters_identifiable == res_old.parameters_identifiable
    assert res_new.analysis_certified == res_old.analysis_certified
    assert len(res_new.profiles) == len(res_old.profiles)
    for p_new, p_old in zip(res_new.profiles, res_old.profiles, strict=True):
        assert p_new.chi_square == p_old.chi_square
        assert p_new.parameter_values_log10 == p_old.parameter_values_log10

    # JSON round-trip preserves mapping_sha256 bit-identically.
    reloaded = new_id.InterfaceSRHIdentifiabilityResult.from_json(res_new.to_json())
    assert reloaded.mapping_sha256 == res_new.mapping_sha256

    if expect_identifiable:
        assert res_new.collinearity_index is not None and res_new.collinearity_index < 50.0
        assert res_new.confounded_parameter_groups == ()
    else:
        assert len(res_new.confounded_parameter_groups) >= 1


_OBS_CACHE: dict[tuple[float, float], np.ndarray] = {}


def _solve_1d_observables(*, alpha_m_inv: float, tau_s: float) -> np.ndarray:
    key = (float(alpha_m_inv), float(tau_s))
    cached = _OBS_CACHE.get(key)
    if cached is not None:
        return cached.copy()
    vt = BOLTZMANN_CONSTANT_J_K * 300.0 / ELEMENTARY_CHARGE_C

    def _ni(eg: float, nc: float, nv: float) -> float:
        return math.sqrt(nc * nv) * math.exp(-eg / (2.0 * vt))

    doc: dict[str, Any] = {
        "device": {
            "T": 300.0,
            "Phi": 2.0e21,
            "built_in_potential_mode": "semiconductor_work_function",
        },
        "layers": [
            {
                "name": "HTL",
                "role": "HTL",
                "thickness": 80.0e-9,
                "eps_r": 3.0,
                "mu_n": 3.868e-5,
                "mu_p": 3.868e-5,
                "chi": 3.6,
                "Eg": 1.8,
                "Nc300": 2.0e25,
                "Nv300": 3.0e25,
                "ni": _ni(1.8, 2.0e25, 3.0e25),
                "n1": _ni(1.8, 2.0e25, 3.0e25),
                "p1": _ni(1.8, 2.0e25, 3.0e25),
                "N_A": 1.0e24,
                "N_D": 0.0,
                "tau_n": 1.0e-6,
                "tau_p": 1.0e-6,
                "B_rad": 1.0e-18,
                "alpha": 0.0,
            },
            {
                "name": "absorber",
                "role": "absorber",
                "thickness": 300.0e-9,
                "eps_r": 25.0,
                "mu_n": 1.934e-3,
                "mu_p": 1.934e-3,
                "chi": 3.9,
                "Eg": 1.6,
                "Nc300": 1.0e25,
                "Nv300": 1.0e25,
                "ni": _ni(1.6, 1.0e25, 1.0e25),
                "n1": _ni(1.6, 1.0e25, 1.0e25),
                "p1": _ni(1.6, 1.0e25, 1.0e25),
                "N_A": 0.0,
                "N_D": 0.0,
                "tau_n": tau_s,
                "tau_p": tau_s,
                "B_rad": 1.0e-18,
                "alpha": alpha_m_inv,
            },
            {
                "name": "ETL",
                "role": "ETL",
                "thickness": 80.0e-9,
                "eps_r": 9.0,
                "mu_n": 3.868e-5,
                "mu_p": 3.868e-5,
                "chi": 4.0,
                "Eg": 2.0,
                "Nc300": 4.0e25,
                "Nv300": 2.0e25,
                "ni": _ni(2.0, 4.0e25, 2.0e25),
                "n1": _ni(2.0, 4.0e25, 2.0e25),
                "p1": _ni(2.0, 4.0e25, 2.0e25),
                "N_A": 0.0,
                "N_D": 1.0e24,
                "tau_n": 1.0e-6,
                "tau_p": 1.0e-6,
                "B_rad": 1.0e-18,
                "alpha": 0.0,
            },
        ],
    }
    prepared, _ = prepare_solarscale_device(doc)
    dev = compile_device_1d(prepared, intervals_per_layer=2)
    pol = float(dev.voltage_polarity)
    opts = SteadyOptions(max_iterations=200, max_bias_step_V=0.10)
    sol_sc = solve_steady_state(dev, voltage_V=0.0, generation_scale=1.0, options=opts)
    sol_mid = solve_steady_state(
        dev,
        initial_state=sol_sc.state,
        voltage_V=0.80,
        generation_scale=1.0,
        options=opts,
    )
    sol_hi = solve_steady_state(
        dev,
        initial_state=sol_mid.state,
        voltage_V=0.96,
        generation_scale=1.0,
        options=opts,
    )
    out = np.array(
        [
            -pol * float(sol_sc.current_density_A_m2[0]),
            -pol * float(sol_mid.current_density_A_m2[0]),
            -pol * float(sol_hi.current_density_A_m2[0]),
        ],
        dtype=float,
    )
    _OBS_CACHE[key] = out
    return out.copy()


def test_live_1d_compiled_device_protocol_identifiability_and_confounding() -> None:
    identifiable_proto = new_id.GenericIdentifiabilityProtocol(
        protocol_id="live_1d_alpha_tau_jv",
        parameters=(
            new_id.GenericIdentifiabilityParameter("log_alpha_m_inv", 5.5, 6.8, 6.2),
            new_id.GenericIdentifiabilityParameter("log_tau_s", -8.0, -6.0, -7.0),
        ),
        observable_labels=("J_0V", "J_0p80V", "J_0p96V"),
        observable_units=("A/m^2", "A/m^2", "A/m^2"),
        standard_deviations=(1.0, 1.0, 1.0),
        finite_difference_step_log10=1.0e-2,
        profile_grid_count=5,
        least_squares_max_nfev=10,
    )

    res_id = new_id.run_protocol_identifiability(
        identifiable_proto,
        lambda coords: _solve_1d_observables(
            alpha_m_inv=10.0 ** float(coords[0]),
            tau_s=10.0 ** float(coords[1]),
        ),
        multistart_fractions=((0.5, 0.5),),
    )
    assert res_id.parameters_identifiable is True
    assert res_id.analysis_certified is True
    assert res_id.failed_samples == ()
    assert res_id.collinearity_index is not None and res_id.collinearity_index < 25.0

    # Confounded product tau_s = 10**(c0 + c1 - 7.0) with explicit out-of-domain failure probe.
    confounded_proto = new_id.GenericIdentifiabilityProtocol(
        protocol_id="live_1d_confounded_tau_factors",
        parameters=(
            new_id.GenericIdentifiabilityParameter("tau_factor_a", -0.5, 0.5, 0.0),
            new_id.GenericIdentifiabilityParameter("tau_factor_b", -0.5, 0.5, 0.0),
        ),
        observable_labels=("J_0V", "J_0p80V", "J_0p96V"),
        observable_units=("A/m^2", "A/m^2", "A/m^2"),
        standard_deviations=(1.0, 1.0, 1.0),
        finite_difference_step_log10=1.0e-2,
        profile_grid_count=5,
        least_squares_max_nfev=10,
    )

    def _confounded_forward(coords: np.ndarray) -> np.ndarray:
        if float(coords[0]) > 0.42 and float(coords[1]) > 0.42:
            raise ValueError("declared_domain_exclusion")
        tau_eff = 10.0 ** (-7.0 + float(coords[0]) + float(coords[1]))
        return _solve_1d_observables(alpha_m_inv=2.0e6, tau_s=tau_eff)

    res_conf = new_id.run_protocol_identifiability(
        confounded_proto,
        _confounded_forward,
        multistart_fractions=((0.5, 0.5), (0.95, 0.95)),
    )
    assert res_conf.parameters_identifiable is False
    assert res_conf.confounded_parameter_groups == (("tau_factor_a", "tau_factor_b"),)
    assert len(res_conf.failed_samples) >= 1
    assert res_conf.analysis_certified is False
