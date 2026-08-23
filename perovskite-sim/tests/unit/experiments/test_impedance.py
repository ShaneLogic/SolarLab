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
    assert assessment.full_timescale_envelope_bracketed is False
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
    assert not assessment.full_timescale_envelope_bracketed
    assert "ionic_timescale_envelope_not_bracketed" in (
        assessment.warnings[0]
    )


def test_frequency_window_does_not_promote_sparse_endpoint_envelope():
    from perovskite_sim.experiments.impedance import (
        assess_impedance_frequency_window,
    )

    x, mat = _ionmonger_frequency_fixture()
    assessment = assess_impedance_frequency_window(
        x, mat, np.array([1.0e-6, 1.0e5]),
    )

    assert assessment.characteristic_frequency_bracketed
    assert assessment.full_timescale_envelope_bracketed
    assert not assessment.ionic_branch_covered
    assert "ionic_branch_sampling_inadequate" in assessment.warnings[0]


def test_frequency_window_certifies_dense_margin_coverage():
    from perovskite_sim.experiments.impedance import (
        assess_impedance_frequency_window,
    )

    x, mat = _ionmonger_frequency_fixture()
    seed = assess_impedance_frequency_window(x, mat, np.array([1.0]))
    low = seed.recommended_f_min_Hz
    high = seed.recommended_f_max_Hz
    decades = np.log10(high / low)
    n_points = int(np.ceil(decades / 0.25)) + 3
    frequencies = np.logspace(
        np.log10(low) - 0.01,
        np.log10(high) + 0.01,
        n_points,
    )
    assessment = assess_impedance_frequency_window(x, mat, frequencies)

    assert assessment.characteristic_frequency_bracketed
    assert assessment.full_timescale_envelope_bracketed
    assert assessment.ionic_branch_covered
    assert assessment.ionic_branch_assessments[0].covered
    assert assessment.max_observed_sampling_gap_decades <= 0.5
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


def test_public_ion_aware_frequency_route_preserves_certification_evidence(
    monkeypatch,
):
    import perovskite_sim.experiments.impedance as impedance_module
    import perovskite_sim.experiments.ion_aware_dc as dc_module
    import perovskite_sim.experiments.ion_aware_impedance as ion_module
    from perovskite_sim.models.config_loader import load_device_from_yaml

    frequencies = np.array([1.0e-3, 1.0])
    stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    contact = SimpleNamespace(status="compatible_unverified", certified=False)
    dc_certificate = SimpleNamespace(
        certified=False,
        numerically_certified=True,
        thermodynamically_certified=False,
        carrier_area_rate_A_m2=1.0e-4,
        ion_area_rate_A_m2=1.0e-8,
        max_ionic_face_current_A_m2=2.0e-9,
        dc_face_current_spread_A_m2=3.0e-4,
        contact_thermodynamics=contact,
        reasons=("contact_thermodynamics_compatible_unverified",),
    )
    captured: dict[str, object] = {}

    def fake_solve(x, _stack, protocol, **kwargs):
        captured["dc_kwargs"] = kwargs
        return SimpleNamespace(
            x=x,
            protocol=protocol,
            protocol_hash="a" * 64,
            state_certificate=dc_certificate,
            total_settle_time_s=32.0,
            consecutive_certified_steps=2,
        )

    frequency_protocol = SimpleNamespace(
        dc_state_sha256="b" * 64,
        protocol_hash="c" * 64,
    )

    def fake_build_frequency(dc_state, values, **kwargs):
        captured["frequency_values"] = np.asarray(values)
        captured["frequency_build_kwargs"] = kwargs
        return frequency_protocol

    face_values = np.ones((2, 3), dtype=complex)
    reference = SimpleNamespace(
        max_relative_face_spread=np.array([1.0e-8, 2.0e-8]),
        reciprocal_condition=np.array([1.0e-3, 2.0e-3]),
        backward_error=np.array([1.0e-13, 2.0e-13]),
    )
    certificate = SimpleNamespace(
        numerically_certified=True,
        thermodynamically_certified=False,
        frequency_window_certified=True,
        certified=False,
        max_relative_face_spread=2.0e-8,
        max_backward_error=2.0e-13,
        minimum_reciprocal_condition=1.0e-3,
        max_mass_diagonal_relative_error=2.0e-11,
        max_mass_off_diagonal_relative=0.0,
        max_ion_inventory_response_relative=3.0e-13,
        max_current_decomposition_relative_error=4.0e-15,
        perturbation_assessments=(SimpleNamespace(passed=True),),
        frequency_point_certificates=(
            SimpleNamespace(frequency_Hz=1.0e-3, numerically_certified=True),
            SimpleNamespace(frequency_Hz=1.0, numerically_certified=True),
        ),
        reasons=(),
    )
    frequency_window = SimpleNamespace(
        has_mobile_ions=True,
        ionic_branch_covered=True,
        warnings=(),
    )

    def fake_run_frequency(*args, **kwargs):
        captured["frequency_run_kwargs"] = kwargs
        return SimpleNamespace(
            frequencies=frequencies,
            Z=np.array([1.0 - 2.0j, 2.0 - 3.0j]),
            Y=np.array([0.2 + 0.4j, 0.1 + 0.2j]),
            Y_faces=face_values,
            reference_linearization=reference,
            frequency_window=frequency_window,
            certificate=certificate,
            electron_storage_response_F_m2=np.array([1.0e-5, 2.0e-5]),
            hole_storage_response_F_m2=np.array([2.0e-5, 3.0e-5]),
            conduction_admittance_faces_S_m2=face_values,
            displacement_admittance_faces_S_m2=2.0 * face_values,
            electron_admittance_faces_S_m2=3.0 * face_values,
            hole_admittance_faces_S_m2=4.0 * face_values,
            positive_ion_admittance_faces_S_m2=5.0 * face_values,
            negative_ion_admittance_faces_S_m2=None,
            positive_ion_storage_response_F_m2=np.array([3.0e-5, 4.0e-5]),
            negative_ion_storage_response_F_m2=None,
            net_charge_storage_response_F_m2=np.array([4.0e-5, 5.0e-5]),
        )

    monkeypatch.setattr(dc_module, "solve_ion_aware_dc", fake_solve)
    monkeypatch.setattr(
        ion_module,
        "build_ion_aware_impedance_protocol",
        fake_build_frequency,
    )
    monkeypatch.setattr(
        ion_module,
        "run_ion_aware_impedance",
        fake_run_frequency,
    )
    monkeypatch.setattr(
        impedance_module,
        "assess_impedance_frequency_window",
        lambda *_args, **_kwargs: frequency_window,
    )

    result = impedance_module.run_impedance(
        stack,
        frequencies,
        N_grid=12,
        method="ion_aware_frequency",
        require_frequency_window_certificate=True,
    )

    assert result.protocol.method == "ion_aware_frequency_certified"
    assert result.protocol.dc_settle_time is None
    assert result.protocol.n_cycles is None
    assert result.operating_point.source == "ion_aware_residual_certified"
    assert result.operating_point.numerically_certified
    assert not result.operating_point.thermodynamically_certified
    assert result.ion_aware_evidence.dc_protocol_sha256 == "a" * 64
    assert result.ion_aware_evidence.dc_state_sha256 == "b" * 64
    assert result.ion_aware_evidence.frequency_protocol_sha256 == "c" * 64
    assert result.ion_aware_evidence.numerically_certified
    assert result.ion_aware_evidence.frequency_window_certified
    assert not result.ion_aware_evidence.certified
    assert len(result.ion_aware_evidence.frequency_point_certificates) == 2
    assert isinstance(
        captured["dc_kwargs"]["atol"],
        impedance_module.ComponentwiseAtol,
    )
    assert captured["frequency_run_kwargs"][
        "require_frequency_window_certificate"
    ] is True
    np.testing.assert_array_equal(
        result.diagnostics.positive_ion_admittance_faces_S_m2,
        5.0 * face_values,
    )


def test_ion_aware_frequency_protocol_records_residual_dc_history():
    from perovskite_sim.experiments.impedance import (
        build_impedance_experiment_protocol,
    )
    from perovskite_sim.models.config_loader import load_device_from_yaml

    protocol = build_impedance_experiment_protocol(
        load_device_from_yaml("configs/ionmonger_benchmark.yaml"),
        np.array([1.0e-3, 1.0]),
        method="ion_aware_frequency_certified",
    )

    assert protocol.initial_state_source == "dark_equilibrium"
    assert protocol.dc_settle.kind == "residual_certified"
    assert protocol.ac_excitation.cycles is None
    assert [step.phase for step in protocol.illumination_history] == [
        "residual_certified_ion_aware_dc",
        "frequency_domain_ion_aware_linear_response",
    ]


def test_public_ion_aware_frequency_route_rejects_ion_free_device():
    from perovskite_sim.experiments.impedance import (
        ImpedanceCapabilityError,
        run_impedance,
    )
    from perovskite_sim.models.config_loader import load_device_from_yaml

    with pytest.raises(ImpedanceCapabilityError, match="mobile-ion"):
        run_impedance(
            load_device_from_yaml("configs/cSi_homojunction.yaml"),
            np.array([1.0e3]),
            N_grid=200,
            illuminated=False,
            method="ion_aware_frequency_certified",
        )


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
