import numpy as np
import pytest
from types import SimpleNamespace
from perovskite_sim.experiments.impedance import extract_impedance


def test_extract_impedance_shape():
    freqs = np.logspace(0, 6, 10)
    Z = extract_impedance(freqs, delta_V=0.01, t_settle=1e-3, n_cycles=5,
                          dummy_mode=True)
    assert Z.shape == (len(freqs),)
    assert np.iscomplexobj(Z)


def test_extract_impedance_high_freq_real():
    """High-frequency Z should be real-dominated (resistive)."""
    freqs = np.array([1e6])
    Z = extract_impedance(freqs, delta_V=0.01, t_settle=1e-3, n_cycles=5,
                          dummy_mode=True)
    assert abs(Z[0].real) > 0


def test_dummy_rc_phase():
    """Dummy RC circuit: Z must have negative imaginary part (capacitive)
    and |angle| must decrease as frequency increases."""
    freqs = np.array([1e2, 1e4, 1e6])
    Z = extract_impedance(freqs, dummy_mode=True)
    # Capacitive: imaginary part must be negative
    assert np.all(Z.imag < 0), f"Expected negative Im(Z), got {Z.imag}"
    # Phase angle |θ| must decrease with frequency (more resistive at high f)
    angles = np.abs(np.angle(Z, deg=True))
    assert angles[0] > angles[1] > angles[2], (
        f"Phase angle should decrease with frequency: {angles}"
    )


def test_impedance_rejects_empty_frequencies():
    from perovskite_sim.experiments.impedance import run_impedance
    from perovskite_sim.models.config_loader import load_device_from_yaml
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    with pytest.raises(ValueError, match="frequenc"):
        run_impedance(stack, np.array([]))


def test_impedance_rejects_small_n_grid():
    from perovskite_sim.experiments.impedance import run_impedance
    from perovskite_sim.models.config_loader import load_device_from_yaml
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    with pytest.raises(ValueError, match="N_grid"):
        run_impedance(stack, np.array([1e3]), N_grid=2)


def test_impedance_rejects_underresolved_csi_grid_before_integration():
    from perovskite_sim.discretization.grid import GridResolutionError
    from perovskite_sim.experiments.impedance import run_impedance
    from perovskite_sim.models.config_loader import load_device_from_yaml

    stack = load_device_from_yaml("configs/cSi_homojunction.yaml")
    with pytest.raises(GridResolutionError, match="under-resolved"):
        run_impedance(stack, np.array([1e5]), N_grid=30)


def test_impedance_rejects_failed_dark_dc_preconditioning(monkeypatch):
    import perovskite_sim.experiments.impedance as impedance_module
    from perovskite_sim.experiments.impedance import run_impedance
    from perovskite_sim.models.config_loader import load_device_from_yaml

    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    monkeypatch.setattr(
        impedance_module,
        "run_transient",
        lambda *args, **kwargs: SimpleNamespace(
            success=False,
            message="deliberate DC failure",
        ),
    )

    with pytest.raises(RuntimeError, match="dark DC preconditioning failed"):
        run_impedance(
            stack,
            np.array([1e3]),
            V_dc=0.1,
            N_grid=12,
            illuminated=False,
        )


def test_impedance_rejects_zero_delta_v():
    from perovskite_sim.experiments.impedance import run_impedance
    from perovskite_sim.models.config_loader import load_device_from_yaml
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    with pytest.raises(ValueError, match="delta_V"):
        run_impedance(stack, np.array([1e3]), delta_V=0.0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_cycles", 1.5),
        ("n_extract", 1.5),
        ("points_per_cycle", 40.5),
    ],
)
def test_impedance_rejects_noninteger_protocol_counts(field, value):
    from perovskite_sim.experiments.impedance import run_impedance
    from perovskite_sim.models.config_loader import load_device_from_yaml

    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    with pytest.raises(ValueError, match=field):
        run_impedance(stack, np.array([1.0e3]), **{field: value})


def test_transient_impedance_enforces_strict_small_signal_amplitude():
    from perovskite_sim.experiments.impedance import run_impedance
    from perovskite_sim.models.config_loader import load_device_from_yaml

    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    with pytest.raises(ValueError, match="below the 20 mV"):
        run_impedance(
            stack,
            np.array([1.0e3]),
            delta_V=0.02,
            method="transient",
        )


def test_qf_frequency_impedance_enforces_strict_small_signal_amplitude():
    from perovskite_sim.experiments.impedance import run_impedance
    from perovskite_sim.models.config_loader import load_device_from_yaml

    stack = load_device_from_yaml("configs/cSi_homojunction.yaml")
    with pytest.raises(ValueError, match="below the 20 mV"):
        run_impedance(
            stack,
            np.array([1.0e5]),
            V_dc=-0.2,
            delta_V=0.02,
            N_grid=200,
            illuminated=False,
            method="quasi_fermi_frequency",
        )


def test_qf_frequency_impedance_rejects_mobile_ion_model():
    from perovskite_sim.experiments.impedance import run_impedance
    from perovskite_sim.experiments.quasi_fermi_steady_state import (
        QuasiFermiSteadyStateError,
    )
    from perovskite_sim.models.config_loader import load_device_from_yaml

    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    with pytest.raises(QuasiFermiSteadyStateError, match="mobile ions"):
        run_impedance(
            stack,
            np.array([1.0e5]),
            V_dc=0.0,
            N_grid=12,
            illuminated=False,
            method="quasi_fermi_frequency",
        )


def test_impedance_rejects_nonpositive_frequency():
    from perovskite_sim.experiments.impedance import run_impedance
    from perovskite_sim.models.config_loader import load_device_from_yaml
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    with pytest.raises(ValueError, match="positive"):
        run_impedance(stack, np.array([0.0, 1e3]))


def test_frequency_window_reports_omitted_ionmonger_branch():
    from perovskite_sim.experiments.impedance import (
        assess_impedance_frequency_window,
    )
    from perovskite_sim.experiments.jv_sweep import build_electrical_grid
    from perovskite_sim.models.config_loader import load_device_from_yaml
    from perovskite_sim.solver.mol import build_material_arrays

    stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    x = build_electrical_grid(stack, 30)
    mat = build_material_arrays(x, stack)

    assessment = assess_impedance_frequency_window(
        x, mat, np.logspace(1, 5, 5),
    )

    assert assessment.has_mobile_ions
    assert assessment.ionic_branch_covered is False
    assert len(assessment.ionic_timescales) == 1
    scale = assessment.ionic_timescales[0]
    assert scale.debye_length_m == pytest.approx(1.467e-9, rel=5e-3)
    assert scale.blocking_charge_frequency_Hz == pytest.approx(
        5.49e-3, rel=1e-2,
    )
    assert "ionic_blocking_charge_frequency_not_bracketed" in (
        assessment.warnings[0]
    )


def _ionmonger_frequency_fixture():
    from perovskite_sim.experiments.jv_sweep import build_electrical_grid
    from perovskite_sim.models.config_loader import load_device_from_yaml
    from perovskite_sim.solver.mol import build_material_arrays

    stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    x = build_electrical_grid(stack, 30)
    return x, build_material_arrays(x, stack)


def test_frequency_window_does_not_promote_one_characteristic_point():
    from perovskite_sim.experiments.impedance import (
        assess_impedance_frequency_window,
    )

    x, mat = _ionmonger_frequency_fixture()
    seed = assess_impedance_frequency_window(x, mat, np.array([1.0]))
    blocking = seed.ionic_timescales[0].blocking_charge_frequency_Hz
    assessment = assess_impedance_frequency_window(
        x, mat, np.array([blocking]),
    )

    assert assessment.characteristic_frequency_bracketed
    assert not assessment.ionic_branch_covered
    assert "ionic_branch_sampling_inadequate" in assessment.warnings[0]


def test_frequency_window_does_not_promote_sparse_endpoint_envelope():
    from perovskite_sim.experiments.impedance import (
        assess_impedance_frequency_window,
    )

    x, mat = _ionmonger_frequency_fixture()
    assessment = assess_impedance_frequency_window(
        x, mat, np.array([1.0e-6, 1.0e5]),
    )

    assert assessment.characteristic_frequency_bracketed
    assert not assessment.ionic_branch_covered
    assert "ionic_branch_sampling_inadequate" in assessment.warnings[0]


def test_frequency_window_certifies_dense_margin_coverage():
    from perovskite_sim.experiments.impedance import (
        assess_impedance_frequency_window,
    )

    x, mat = _ionmonger_frequency_fixture()
    seed = assess_impedance_frequency_window(x, mat, np.array([1.0]))
    scale = seed.ionic_timescales[0]
    low = min(
        scale.blocking_charge_frequency_Hz,
        scale.dielectric_frequency_Hz,
    ) / 10.0
    high = max(
        scale.blocking_charge_frequency_Hz,
        scale.dielectric_frequency_Hz,
    ) * 10.0
    decades = np.log10(high / low)
    n_points = int(np.ceil(decades / 0.25)) + 3
    frequencies = np.logspace(
        np.log10(low) - 0.01,
        np.log10(high) + 0.01,
        n_points,
    )
    assessment = assess_impedance_frequency_window(x, mat, frequencies)

    assert assessment.characteristic_frequency_bracketed
    assert assessment.ionic_branch_covered
    assert assessment.warnings == ()


def test_strict_transient_impedance_rejects_uncertified_dc_state():
    from perovskite_sim.experiments.impedance import (
        ImpedanceCertificationError,
        run_impedance,
    )
    from perovskite_sim.models.config_loader import load_device_from_yaml

    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    with pytest.raises(ImpedanceCertificationError, match="uncertified"):
        run_impedance(
            stack,
            np.array([1.0e3]),
            V_dc=0.9,
            N_grid=20,
            n_cycles=1,
            require_operating_point_certificate=True,
        )


def test_public_qf_result_preserves_frequency_domain_diagnostics(monkeypatch):
    import perovskite_sim.experiments.quasi_fermi_impedance as qf_module
    from perovskite_sim.experiments.impedance import run_impedance
    from perovskite_sim.models.config_loader import load_device_from_yaml

    frequencies = np.array([1.0e3, 1.0e4])
    fake = SimpleNamespace(
        frequencies=frequencies,
        Z=np.array([1.0 - 2.0j, 2.0 - 3.0j]),
        Y=np.array([0.2 + 0.4j, 0.1 + 0.2j]),
        Y_faces=np.ones((2, 3), dtype=complex),
        max_relative_face_spread=np.array([1e-8, 2e-8]),
        reciprocal_condition=np.array([1e-3, 2e-3]),
        backward_error=np.array([1e-12, 2e-12]),
        electron_storage_response_F_m2=np.array([1e-4, 2e-4]),
        hole_storage_response_F_m2=np.array([3e-4, 4e-4]),
        dc_state=SimpleNamespace(
            certified=True,
            electron_continuity_bound_A_m2=1e-4,
            hole_continuity_bound_A_m2=2e-4,
            face_current_spread_A_m2=1e-5,
        ),
    )
    monkeypatch.setattr(
        qf_module, "run_quasi_fermi_impedance", lambda *args, **kwargs: fake,
    )

    result = run_impedance(
        load_device_from_yaml("configs/cSi_homojunction.yaml"),
        frequencies,
        V_dc=-0.2,
        N_grid=200,
        illuminated=False,
        method="qf_frequency_ion_free",
    )

    assert result.protocol is not None
    assert result.protocol.method == "qf_frequency_ion_free"
    assert result.operating_point is not None
    assert result.operating_point.numerically_certified
    assert result.diagnostics is not None
    assert np.array_equal(result.diagnostics.backward_error, fake.backward_error)
    assert np.array_equal(result.diagnostics.admittance_faces_S_m2, fake.Y_faces)


def test_run_impedance_uses_passive_capacitive_sign_convention():
    from perovskite_sim.experiments.impedance import run_impedance
    from perovskite_sim.models.config_loader import load_device_from_yaml

    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    result = run_impedance(
        stack, np.array([1e3, 1e4]), V_dc=0.9, delta_V=0.01, N_grid=20, n_cycles=2
    )

    assert np.all(np.isfinite(result.Z.real))
    assert np.all(np.isfinite(result.Z.imag))
    assert np.all(result.Z.real > 0.0)
    assert np.all(result.Z.imag < 0.0)
    assert result.protocol is not None
    assert result.protocol.method == "transient_ion_aware"
    assert result.protocol.dc_settle_time == pytest.approx(1e-3)
    assert result.protocol.experiment_protocol is not None
    assert result.protocol.experiment_protocol.implicit_legacy_protocol
    assert result.operating_point is not None
    assert result.operating_point.source == "finite_time_preconditioned"
    assert result.frequency_window is not None
    assert result.frequency_window.has_mobile_ions
    assert result.frequency_window.ionic_branch_covered is False


def test_transient_production_loop_is_copointed_for_a_pure_capacitor(
    monkeypatch,
):
    import perovskite_sim.experiments.impedance as impedance_module
    from perovskite_sim.experiments.impedance import (
        FrequencyWindowAssessment,
        run_impedance,
    )

    x = np.array([0.0, 0.5, 1.0])
    y_dc = np.zeros(3 * len(x))
    capacitance = 2.5e-3
    frequency = 100.0
    saw_callable_voltage = []

    monkeypatch.setattr(
        impedance_module, "build_electrical_grid", lambda *_args: x,
    )
    monkeypatch.setattr(
        impedance_module,
        "require_thick_layer_interface_resolution",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        impedance_module,
        "build_material_arrays",
        lambda *_args: SimpleNamespace(N_iface_state=0),
    )
    monkeypatch.setattr(
        impedance_module,
        "assess_impedance_frequency_window",
        lambda *_args: FrequencyWindowAssessment(
            f_min_Hz=frequency,
            f_max_Hz=frequency,
            has_mobile_ions=False,
        ),
    )
    monkeypatch.setattr(
        impedance_module, "solve_illuminated_ss", lambda *_args, **_kwargs: y_dc,
    )
    monkeypatch.setattr(
        impedance_module,
        "_transient_operating_point_certificate",
        lambda *_args, **_kwargs: SimpleNamespace(certified=True, reasons=()),
    )

    def fake_run_transient(
        _x, y0, _span, t_eval, _stack, *, V_app, **_kwargs,
    ):
        saw_callable_voltage.append(callable(V_app))
        assert np.all(np.isfinite([V_app(t) for t in t_eval]))
        return SimpleNamespace(
            success=True,
            y=np.zeros((len(y0), len(t_eval))),
        )

    def fake_current(_x, _y, _stack, _voltage, **_kwargs):
        return SimpleNamespace(J_total=np.zeros(len(x) - 1))

    def fake_total_current(
        _x,
        _y,
        _stack,
        voltage,
        *,
        dt,
        V_app_prev,
        **_kwargs,
    ):
        # Solver current uses the solar-cell sign. The public lock-in flips it
        # to passive convention, hence the leading minus sign here.
        passive_displacement = capacitance * (voltage - V_app_prev) / dt
        return np.full(len(x) - 1, -passive_displacement)

    monkeypatch.setattr(impedance_module, "run_transient", fake_run_transient)
    monkeypatch.setattr(
        impedance_module, "compute_current_components", fake_current,
    )
    monkeypatch.setattr(
        impedance_module, "_total_current_faces", fake_total_current,
    )

    result = run_impedance(
        object(),
        np.array([frequency]),
        delta_V=0.01,
        N_grid=3,
        n_cycles=3,
        n_extract=2,
        points_per_cycle=40,
    )

    ideal = -1j / (2.0 * np.pi * frequency * capacitance)
    assert saw_callable_voltage == [True]
    assert np.angle(result.Z[0], deg=True) == pytest.approx(-90.0, abs=1.0e-8)
    assert result.Z[0].real == pytest.approx(0.0, abs=1.0e-10)
    assert abs(result.Z[0]) == pytest.approx(abs(ideal), rel=2.0e-3)
    assert result.protocol.points_per_cycle == 40


def test_transient_certificate_fails_closed_on_nonfinite_rhs(monkeypatch):
    import perovskite_sim.experiments.impedance as impedance_module
    from perovskite_sim.experiments.jv_sweep import build_electrical_grid
    from perovskite_sim.models.config_loader import load_device_from_yaml
    from perovskite_sim.solver.mol import build_material_arrays
    from perovskite_sim.solver.newton import solve_equilibrium

    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    x = build_electrical_grid(stack, 12)
    mat = build_material_arrays(x, stack)
    y = solve_equilibrium(x, stack)
    monkeypatch.setattr(
        impedance_module,
        "assemble_rhs",
        lambda *_args, **_kwargs: np.full_like(y, np.nan),
    )

    certificate = impedance_module._transient_operating_point_certificate(
        x,
        y,
        stack,
        mat,
        V_dc=0.0,
        illuminated=False,
        source="dark_equilibrium",
        max_carrier_area_rate_A_m2=1.0,
        max_ion_area_rate_A_m2=1.0,
        max_ionic_face_current_A_m2=1.0,
        max_dc_face_spread_A_m2=1.0,
    )

    assert not certificate.numerically_certified
    assert "state_rate_nonfinite" in certificate.reasons
    assert "carrier_area_rate_nonfinite" in certificate.reasons
    assert "ion_area_rate_nonfinite" in certificate.reasons


def test_qf_certificate_fails_closed_on_nonfinite_diagnostics():
    import perovskite_sim.experiments.impedance as impedance_module
    from perovskite_sim.experiments.jv_sweep import build_electrical_grid
    from perovskite_sim.models.config_loader import load_device_from_yaml
    from perovskite_sim.solver.mol import build_material_arrays

    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    x = build_electrical_grid(stack, 12)
    mat = build_material_arrays(x, stack)
    dc_state = SimpleNamespace(
        certified=True,
        electron_continuity_bound_A_m2=np.nan,
        hole_continuity_bound_A_m2=0.0,
        face_current_spread_A_m2=0.0,
    )

    certificate = impedance_module._qf_operating_point_certificate(
        stack, mat, dc_state,
    )

    assert not certificate.numerically_certified
    assert "qf_electron_continuity_bound_nonfinite" in certificate.reasons


def test_public_impedance_stops_before_ac_on_nonfinite_dc_evidence(
    monkeypatch,
):
    import perovskite_sim.experiments.impedance as impedance_module
    from perovskite_sim.experiments.impedance import (
        ImpedanceCertificationError,
        run_impedance,
    )
    from perovskite_sim.models.config_loader import load_device_from_yaml

    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    monkeypatch.setattr(
        impedance_module,
        "solve_illuminated_ss",
        lambda x, *_args, **_kwargs: np.zeros(3 * len(x)),
    )
    monkeypatch.setattr(
        impedance_module,
        "_transient_operating_point_certificate",
        lambda *_args, **_kwargs: SimpleNamespace(
            certified=False,
            reasons=("state_rate_nonfinite",),
        ),
    )

    with pytest.raises(ImpedanceCertificationError, match="non-finite evidence"):
        run_impedance(stack, np.array([1.0e3]), N_grid=12)


def test_grid_assessment_records_underresolution_override():
    import perovskite_sim.experiments.impedance as impedance_module

    assessment = impedance_module._assess_impedance_grid(
        (
            SimpleNamespace(
                layer_debye_span=2.0e3,
                cell_debye_ratio=2.0,
            ),
        ),
        allow_underresolved_grid=True,
    )

    assert not assessment.certified
    assert assessment.override_used
    assert assessment.guarded_cell_count == 1
    assert assessment.offender_count == 1
    assert assessment.max_guarded_cell_debye_ratio == pytest.approx(2.0)
    assert "underresolved_grid_override_used" in assessment.warnings[0]


def test_strict_impedance_rejects_an_underresolved_grid_override():
    from perovskite_sim.experiments.impedance import (
        ImpedanceCertificationError,
        run_impedance,
    )
    from perovskite_sim.models.config_loader import load_device_from_yaml

    stack = load_device_from_yaml("configs/cSi_homojunction.yaml")
    with pytest.raises(ImpedanceCertificationError, match="grid is uncertified"):
        run_impedance(
            stack,
            np.array([1.0e5]),
            N_grid=30,
            allow_underresolved_grid=True,
            require_operating_point_certificate=True,
        )


def test_impedance_rejects_dynamic_interface_state_blocks(monkeypatch):
    import perovskite_sim.experiments.impedance as impedance_module
    from perovskite_sim.experiments.impedance import run_impedance

    monkeypatch.setattr(
        impedance_module,
        "build_electrical_grid",
        lambda *_args: np.array([0.0, 0.5, 1.0]),
    )
    monkeypatch.setattr(
        impedance_module,
        "require_thick_layer_interface_resolution",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        impedance_module,
        "build_material_arrays",
        lambda *_args: SimpleNamespace(N_iface_state=1),
    )

    with pytest.raises(
        impedance_module.ImpedanceCapabilityError,
        match="interface-state",
    ):
        run_impedance(object(), np.array([1.0e3]), N_grid=3)
