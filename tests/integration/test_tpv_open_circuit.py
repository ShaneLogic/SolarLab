"""Integration and legacy-parity tests for P05-04a charge-conserving open-circuit & TPV."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.research_1d.ionic_multilayer import (  # noqa: E402
    prepare_confined_ion_device,
)
from examples.research_1d.steady import load_device  # noqa: E402
from solarlab.config.resolve_device import resolve_device  # noqa: E402
from solarlab.device.compile_1d import compile_device_1d  # noqa: E402
from solarlab.device.inputs import DeviceInput  # noqa: E402
from solarlab.device.resolved import PreparedDevice  # noqa: E402
from solarlab.experiments.open_circuit import (  # noqa: E402
    OpenCircuitOptions,
    integrate_open_circuit,
)
from solarlab.experiments.tpv import (  # noqa: E402
    TPVError,
    TPVProtocol,
    fit_tpv_decay,
    run_tpv,
    solve_illuminated_voc_state,
)


@pytest.fixture(scope="module", autouse=True)
def single_thread_blas() -> object:
    with threadpool_limits(limits=1):
        yield


def test_open_circuit_integrator_enforces_maxwell_total_current_and_charge_conservation() -> None:
    prepared = load_device()
    compiled = compile_device_1d(prepared, intervals_per_layer=(8, 8, 8))

    state_half, v_half = solve_illuminated_voc_state(
        compiled, generation_scale=0.5, upper_bound_V=1.3
    )
    _, v_full = solve_illuminated_voc_state(
        compiled, generation_scale=1.0, upper_bound_V=1.3
    )
    assert v_full > v_half > 1.0

    # Step illumination from 0.5 sun to 1.0 sun under genuine floating-terminal open circuit
    times_s = np.r_[0.0, np.geomspace(1e-8, 8e-5, 22)]
    gen_scale = np.ones_like(times_s)
    gen_scale[0] = 0.5

    traj = integrate_open_circuit(
        compiled,
        state_half,
        times_s,
        gen_scale,
        initial_voltage_V=v_half,
        options=OpenCircuitOptions(
            max_current_error_A_m2=1e-7,
            max_voltage_error_V=1e-6,
        ),
    )
    assert traj.certified

    # 1. Total Maxwell current (I_cond + dD/dt) vanishes on ALL faces at EVERY step
    assert float(np.max(traj.max_face_total_current_A_m2)) <= 1e-7
    np.testing.assert_allclose(
        traj.left_conduction_current_A_m2 + traj.left_displacement_current_A_m2,
        traj.left_total_current_A_m2,
        rtol=0.0,
        atol=1e-12,
    )

    # 2. Dynamic capacitive charging (NOT the unphysical steady I_cond == 0 clamp,
    #    which would jump instantaneously to v_full at step 1!)
    assert v_half < float(traj.terminal_voltage_V[1]) < v_full - 0.010
    assert float(np.max(np.abs(traj.left_conduction_current_A_m2))) > 1e-9
    assert float(np.max(np.abs(traj.left_displacement_current_A_m2))) > 1e-9

    # 3. Telescoping port charge error stays below 1 uV and V_oc(t) relaxes to 1-sun steady V_oc
    assert traj.charge_voltage_error_V < 1e-6
    assert traj.terminal_voltage_V[-1] == pytest.approx(v_full, abs=1e-4)


def test_ion_free_tpv_single_exponential_extraction_truncation_refusal_and_legacy_parity() -> None:
    prepared = load_device()
    prot = TPVProtocol(
        delta_G_frac=0.01,
        t_pulse_s=1e-6,
        t_decay_s=2e-5,
        pulse_points=5,
        decay_points=25,
        decay_grid="linear",
        voc_upper_bound_V=1.3,
        intervals_per_layer=8,
    )
    res = run_tpv(prepared, prot)

    assert res.initial_state_source == "steady_illuminated_open_circuit"
    assert res.equilibrium_claimed is True
    assert res.preconditioning_trajectory is None
    assert res.max_baseline_drift_V < 1e-8
    assert res.max_current_error_A_m2 <= 1e-7
    assert res.charge_voltage_error_V < 1e-6
    assert res.delta_V_0 > 0.0
    assert res.decay_fit.status == "single_exponential"
    assert res.tau is not None and res.tau > 0.0

    # Standalone fit_tpv_decay consistency on the solved decay trace
    pulse_end = prot.pulse_points
    t_rel = res.t[pulse_end:] - float(res.t[pulse_end])
    dv_rel = res.delta_V[pulse_end:]
    direct_fit = fit_tpv_decay(t_rel, dv_rel)
    assert direct_fit.status == "single_exponential"
    assert direct_fit.tau_s is not None
    assert res.tau == pytest.approx(direct_fit.tau_s, rel=5e-3)

    # Also verify non-identifiable refusal on truncated decay window
    short_prot = TPVProtocol(
        delta_G_frac=0.05,
        t_pulse_s=5e-7,
        t_decay_s=1e-7,
        pulse_points=4,
        decay_points=8,
        voc_upper_bound_V=1.3,
        intervals_per_layer=8,
    )
    short_res = run_tpv(prepared, short_prot)
    assert short_res.tau is None
    assert short_res.decay_fit.status == "insufficient_decay_window"
    assert short_res.decay_fit.non_identifiable_reason is not None

    t_short = short_res.t[short_prot.pulse_points :] - float(
        short_res.t[short_prot.pulse_points]
    )
    direct_short = fit_tpv_decay(
        t_short, short_res.delta_V[short_prot.pulse_points :]
    )
    assert direct_short.tau_s is None
    assert direct_short.status == "insufficient_decay_window"


def test_mobile_ion_tpv_paired_control_subtraction_under_active_baseline_drift() -> None:
    base_ion = prepare_confined_ion_device()
    editing = base_ion.to_input().editing_data()
    absorber = next(lyr for lyr in editing["layers"] if lyr["role"] == "absorber")
    # Fast diffusivity so ionic open-circuit baseline drift is actively evolving on sub-ms timescales
    absorber["parameters"]["D_ion"] = 2e-12
    fast_ion_prepared: PreparedDevice = resolve_device(
        DeviceInput.model_validate(editing),
        base_ion.defaults,
        base_ion.resources,
        sources=base_ion.sources,
    )

    prot = TPVProtocol(
        delta_G_frac=0.05,
        t_pulse_s=2e-5,
        t_decay_s=3e-4,
        pulse_points=2,
        decay_points=14,
        decay_grid="linear",
        soak_duration_s=5e-4,
        soak_steps=6,
        voc_upper_bound_V=1.3,
        intervals_per_layer=8,
    )
    res = run_tpv(fast_ion_prepared, prot)

    # 1. Explicit finite-time preconditioning provenance (never claims ionic equilibrium)
    assert res.initial_state_source == "finite_time_illuminated_preconditioned"
    assert res.equilibrium_claimed is False
    assert res.preconditioning_trajectory is not None
    assert res.preconditioning_trajectory.certified

    # 2. Pulsed and unpulsed control start from bit-identical (state_oc, V_oc)
    s0_pulsed = res.pulsed_trajectory.states[0]
    s0_ctrl = res.control_trajectory.states[0]
    np.testing.assert_array_equal(s0_pulsed.u, s0_ctrl.u)
    assert res.V[0] == res.V_control[0] == res.V_oc

    # 3. Unpulsed control captures nonzero ionic baseline drift while delta_V isolates pulse
    assert res.max_baseline_drift_V > 1e-6
    np.testing.assert_allclose(res.delta_V, res.V - res.V_control, rtol=0.0, atol=1e-15)
    assert res.delta_V_0 > 0.0
    assert res.max_current_error_A_m2 <= 1e-7
    assert res.charge_voltage_error_V < 1e-6

    # 4. Zero soak_duration_s on mobile-ion device is refused
    with pytest.raises(TPVError, match="soak_duration_s"):
        run_tpv(
            fast_ion_prepared,
            TPVProtocol(soak_duration_s=0.0, intervals_per_layer=8),
        )
