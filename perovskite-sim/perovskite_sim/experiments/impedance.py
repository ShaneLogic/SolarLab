from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Literal

import numpy as np

from perovskite_sim.constants import Q
from perovskite_sim.discretization.grid import (
    MAX_INTERFACE_CELL_DEBYE_RATIO,
    MIN_GUARDED_LAYER_DEBYE_SPAN,
    require_thick_layer_interface_resolution,
)
from perovskite_sim.models.device import DeviceStack
from perovskite_sim.experiments import impedance_frequency as _frequency
from perovskite_sim.physics.contacts import (
    ContactThermodynamicCertificate,
    assess_contact_thermodynamics,
)
from perovskite_sim.solver.illuminated_ss import solve_illuminated_ss
from perovskite_sim.solver.mol import (
    StateVec,
    assemble_rhs,
    build_material_arrays,
    run_transient,
)
from perovskite_sim.solver.newton import solve_equilibrium
from perovskite_sim.solver.tolerances import AbsoluteTolerance, ComponentwiseAtol
from perovskite_sim.experiments.protocol import (
    ACExcitation,
    DCSettleCriterion,
    ExperimentProtocol,
    IlluminationStep,
    ProtocolMode,
    SamplingProtocol,
    ScanProtocol,
    resolve_experiment_protocol,
)
from perovskite_sim.experiments.jv_sweep import (
    _total_current_faces,
    build_electrical_grid,
    compute_current_components,
)
from perovskite_sim.experiments.dynamic_defect_impedance import (
    DYNAMIC_DEFECT_IMPEDANCE_METHOD,
    DEFAULT_REFINEMENT_FACTORS as DEFAULT_DYNAMIC_DEFECT_REFINEMENT_FACTORS,
    DynamicDefectImpedanceCapabilityError,
    DynamicDefectImpedanceCertificationError,
    DynamicDefectImpedanceEvidence,
    DynamicDefectImpedanceGates,
    DynamicDefectImpedanceProtocol,
    DynamicDefectImpedanceProtocolError,
    build_dynamic_defect_impedance_protocol,
    classify_dynamic_defect_capability,
    resolve_dynamic_defect_impedance_protocol,
    run_dynamic_defect_impedance,
)
from perovskite_sim.experiments.quasi_fermi_steady_state import (
    build_two_sided_trace_grid,
)
from perovskite_sim.physics.defect_distributions import (
    DEFAULT_DEFECT_ENERGY_QUADRATURE_ORDER,
)


# Callback signature: stage, current, total, message.
ProgressCallback = Callable[[str, int, int, str], None]


MAX_LINEAR_PERTURBATION_V = 0.02
DEFAULT_MAX_CARRIER_AREA_RATE_A_M2 = 1.0e-1
DEFAULT_MAX_ION_AREA_RATE_A_M2 = 1.0e-6
DEFAULT_MAX_DC_FACE_SPREAD_A_M2 = 1.0e-1
_IMPEDANCE_METHOD_ALIASES = {
    "transient": "transient_ion_aware",
    "transient_ion_aware": "transient_ion_aware",
    "quasi_fermi_frequency": "qf_frequency_ion_free",
    "qf_frequency_ion_free": "qf_frequency_ion_free",
    "ion_aware_frequency": "ion_aware_frequency_certified",
    "ion_aware_frequency_certified": "ion_aware_frequency_certified",
    "dynamic_defect_frequency": DYNAMIC_DEFECT_IMPEDANCE_METHOD,
    DYNAMIC_DEFECT_IMPEDANCE_METHOD: DYNAMIC_DEFECT_IMPEDANCE_METHOD,
}

# Backward-compatible public exports; implementation lives in the shared
# frequency-assessment module used by both impedance engines.
FrequencyWindowAssessment = _frequency.FrequencyWindowAssessment
IonicBranchCoverage = _frequency.IonicBranchCoverage
IonicTimescale = _frequency.IonicTimescale
assess_impedance_frequency_window = _frequency.assess_impedance_frequency_window


@dataclass(frozen=True)
class ImpedanceProtocol:
    """The bias history and excitation that define an impedance result."""

    method: Literal[
        "transient_ion_aware",
        "qf_frequency_ion_free",
        "ion_aware_frequency_certified",
        "dynamic_defect_frequency_certified",
    ]
    V_dc: float
    delta_V: float
    illuminated: bool
    dc_settle_time: float | None
    n_cycles: int | None
    n_extract: int | None
    points_per_cycle: int | None = None
    experiment_protocol: ExperimentProtocol | None = None
    dynamic_defect_protocol: DynamicDefectImpedanceProtocol | None = None


@dataclass(frozen=True)
class GridAssessment:
    """Necessary interface-resolution evidence for the electrical mesh."""

    certified: bool
    override_used: bool
    guarded_cell_count: int
    offender_count: int
    max_guarded_cell_debye_ratio: float | None
    max_cell_debye_ratio_limit: float
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class OperatingPointCertificate:
    """Residual, conservation, and contact evidence for the DC state."""

    certified: bool
    numerically_certified: bool
    thermodynamically_certified: bool
    source: Literal[
        "finite_time_preconditioned",
        "dark_equilibrium",
        "qf_residual_certified",
        "ion_aware_residual_certified",
    ]
    carrier_area_rate_A_m2: float
    ion_area_rate_A_m2: float
    max_ionic_face_current_A_m2: float
    dc_face_current_spread_A_m2: float
    carrier_area_rate_limit_A_m2: float | None
    ion_area_rate_limit_A_m2: float | None
    ionic_face_current_limit_A_m2: float | None
    dc_face_current_spread_limit_A_m2: float | None
    contact_thermodynamics: ContactThermodynamicCertificate
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class ImpedanceDiagnostics:
    """Frequency-domain diagnostics retained from the selected engine."""

    admittance_S_m2: np.ndarray | None = None
    admittance_faces_S_m2: np.ndarray | None = None
    max_relative_face_spread: np.ndarray | None = None
    reciprocal_condition: np.ndarray | None = None
    backward_error: np.ndarray | None = None
    electron_storage_response_F_m2: np.ndarray | None = None
    hole_storage_response_F_m2: np.ndarray | None = None
    conduction_admittance_faces_S_m2: np.ndarray | None = None
    displacement_admittance_faces_S_m2: np.ndarray | None = None
    electron_admittance_faces_S_m2: np.ndarray | None = None
    hole_admittance_faces_S_m2: np.ndarray | None = None
    positive_ion_admittance_faces_S_m2: np.ndarray | None = None
    negative_ion_admittance_faces_S_m2: np.ndarray | None = None
    positive_ion_storage_response_F_m2: np.ndarray | None = None
    negative_ion_storage_response_F_m2: np.ndarray | None = None
    net_charge_storage_response_F_m2: np.ndarray | None = None
    bulk_trap_charge_storage_response_F_m2: np.ndarray | None = None
    interface_sheet_charge_storage_response_F_m2: np.ndarray | None = None
    bulk_trap_occupancy_response_per_V: np.ndarray | None = None
    interface_occupancy_response_per_V: np.ndarray | None = None


@dataclass(frozen=True)
class IonAwareImpedanceEvidence:
    """Exact DC identity and graded evidence for the ion-aware engine."""

    dc_protocol: Any
    frequency_protocol: Any
    dc_state_certificate: Any
    dc_protocol_sha256: str
    dc_state_sha256: str
    frequency_protocol_sha256: str
    dc_total_settle_time_s: float
    dc_consecutive_certified_steps: int
    dc_solver_rtol: float
    dc_solver_atol: Any
    numerically_certified: bool
    thermodynamically_certified: bool
    frequency_window_certified: bool
    certified: bool
    max_relative_face_spread: float
    max_backward_error: float
    minimum_reciprocal_condition: float
    max_mass_diagonal_relative_error: float
    max_mass_off_diagonal_relative: float
    max_ion_inventory_response_relative: float
    max_current_decomposition_relative_error: float
    perturbation_assessments: tuple[Any, ...]
    frequency_point_certificates: tuple[Any, ...]
    reasons: tuple[str, ...]


class ImpedanceCertificationError(RuntimeError):
    """The requested strict impedance protocol lacks a DC certificate."""


class ImpedanceCapabilityError(RuntimeError):
    """The requested model state is unsupported by the impedance engine."""


@dataclass(frozen=True)
class ImpedanceResult:
    frequencies: np.ndarray
    Z: np.ndarray  # complex impedance [Ω m²]
    protocol: ImpedanceProtocol | None = None
    operating_point: OperatingPointCertificate | None = None
    frequency_window: FrequencyWindowAssessment | None = None
    grid_assessment: GridAssessment | None = None
    diagnostics: ImpedanceDiagnostics | None = None
    ion_aware_evidence: IonAwareImpedanceEvidence | None = None
    dynamic_defect_evidence: DynamicDefectImpedanceEvidence | None = None


def extract_impedance(
    frequencies: np.ndarray,
    delta_V: float = 0.01,
    t_settle: float = 1e-3,
    n_cycles: int = 5,
    dummy_mode: bool = False,
) -> np.ndarray:
    """
    Returns complex impedance array Z [Ω m²] for each frequency.
    dummy_mode=True returns synthetic RC response for testing.
    """
    if dummy_mode:
        # RC circuit: Z = R + 1/(jωC)
        R = 10.0
        C = 1e-6
        omega = 2 * np.pi * frequencies
        return R + 1.0 / (1j * omega * C)

    raise NotImplementedError("Full IS requires a DeviceStack argument.")


def _linear_detrend(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Remove a linear trend from *y* sampled at times *t*.

    Retained as a utility for external callers and diagnostics. The primary
    lock-in path (``_lockin_extract``) no longer uses this helper, because
    detrending *before* projecting onto sin/cos leaks a fraction of the AC
    component into the subtracted slope whenever the sampling window is not
    a strict integer number of periods. Instead ``_lockin_extract`` does a
    joint least-squares fit to ``[1, t, sin(ωt), cos(ωt)]``, which absorbs
    any linear drift into the first two basis vectors while leaving the
    sin/cos amplitudes untouched.
    """
    coeffs = np.polyfit(t, y, 1)
    return y - np.polyval(coeffs, t)


def _lockin_extract(
    I_t: np.ndarray,
    t: np.ndarray,
    freq: float,
    delta_V: float,
) -> complex:
    """Extract complex impedance Z = V̂ / Î from a passive-convention current.

    Implements the lock-in step used inside ``run_impedance`` via a single
    least-squares projection onto the basis ``[1, t, sin(ωt), cos(ωt)]``:

    1. Build the design matrix from the four basis vectors evaluated at the
       sample times.
    2. Solve ``A·c = I_t`` in the least-squares sense. The sin/cos coefficients
       ``c[2], c[3]`` are the in-phase and quadrature amplitudes of the AC
       response; any DC offset or linear drift is absorbed into ``c[0], c[1]``
       and cannot contaminate them.
    3. Form the phasor Î = I_in + j·I_quad under the imag-part convention
       (V(t) = δV·sin(ωt) ⇒ V̂ = δV), and return Z = δV / Î.

    Why joint LS instead of detrend → project
    -----------------------------------------
    The previous implementation detrended *y* first and then projected onto
    sin/cos. When the sampling window was not an exact integer number of
    periods (which is the case in ``run_impedance``: the 80-sample midpoint
    grid for ``n_extract=2, pts_per_cycle=40`` spans 1.975·T, not 2·T), the
    linear-fit slope picks up a non-zero projection onto the sinusoid itself,
    and ~15–20% of the AC amplitude gets subtracted out — biasing |Z| high
    by the same factor. A joint fit to the combined basis is exact for any
    finite window and any sampling pattern, and trivially reduces to the
    textbook lock-in in the orthogonal limit (integer-period window).

    This helper is called by ``run_impedance`` and drives the analytic
    Randles-circuit regression test
    (``tests/unit/experiments/test_impedance_randles.py``).

    Parameters
    ----------
    I_t     : (M,) current samples in passive convention [A m⁻²].
    t       : (M,) sample times [s]; need not be uniform or an integer number
              of periods — the joint fit is unbiased on any window.
    freq    : excitation frequency [Hz].
    delta_V : voltage-excitation amplitude [V].

    Returns
    -------
    Z       : complex impedance [Ω m²].
    """
    y = np.asarray(I_t, dtype=float)
    t_arr = np.asarray(t, dtype=float)
    omega = 2.0 * np.pi * freq
    # Design matrix: [1, t, sin(ωt), cos(ωt)]. Scale t to O(1) so the
    # least-squares system is well-conditioned regardless of the absolute
    # time units (which span many decades across the freq sweep).
    t_scale = t_arr[-1] - t_arr[0]
    if t_scale <= 0.0:
        t_scale = 1.0
    A = np.column_stack(
        [
            np.ones_like(t_arr),
            (t_arr - t_arr[0]) / t_scale,
            np.sin(omega * t_arr),
            np.cos(omega * t_arr),
        ]
    )
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    I_in = float(coeffs[2])
    I_quad = float(coeffs[3])
    delta_I = I_in + 1j * I_quad
    if abs(delta_I) == 0.0:
        return complex(np.inf, 0.0)
    return delta_V / delta_I


def _assess_impedance_grid(
    diagnostics,
    *,
    allow_underresolved_grid: bool,
) -> GridAssessment:
    """Classify the same guarded cells used by the pre-integration mesh gate."""
    guarded = tuple(
        item
        for item in diagnostics
        if item.layer_debye_span >= MIN_GUARDED_LAYER_DEBYE_SPAN
    )
    offenders = tuple(
        item
        for item in guarded
        if (
            not np.isfinite(item.cell_debye_ratio)
            or item.cell_debye_ratio > MAX_INTERFACE_CELL_DEBYE_RATIO
        )
    )
    guarded_ratios = tuple(float(item.cell_debye_ratio) for item in guarded)
    max_ratio = (
        float(max(guarded_ratios))
        if guarded_ratios and all(np.isfinite(guarded_ratios))
        else None
    )
    warnings = (
        (
            (
                "underresolved_grid_override_used; this impedance result lacks "
                "the necessary interface Debye-resolution certificate"
            ),
        )
        if offenders
        else ()
    )
    return GridAssessment(
        certified=not offenders,
        override_used=bool(offenders and allow_underresolved_grid),
        guarded_cell_count=len(guarded),
        offender_count=len(offenders),
        max_guarded_cell_debye_ratio=max_ratio,
        max_cell_debye_ratio_limit=MAX_INTERFACE_CELL_DEBYE_RATIO,
        warnings=warnings,
    )


def _transient_operating_point_certificate(
    x: np.ndarray,
    y: np.ndarray,
    stack: DeviceStack,
    mat,
    *,
    V_dc: float,
    illuminated: bool,
    source: Literal["finite_time_preconditioned", "dark_equilibrium"],
    max_carrier_area_rate_A_m2: float,
    max_ion_area_rate_A_m2: float,
    max_ionic_face_current_A_m2: float,
    max_dc_face_spread_A_m2: float,
) -> OperatingPointCertificate:
    """Evaluate an independent DC residual and all-face current certificate."""
    state = np.asarray(y, dtype=float)
    rate = assemble_rhs(
        0.0,
        state,
        np.asarray(x, dtype=float),
        stack,
        mat,
        illuminated=illuminated,
        V_app=V_dc,
    )
    state_rate = StateVec.unpack(
        rate,
        len(x),
        N_iface_state=mat.N_iface_state,
    )
    widths = np.asarray(mat.dx_cell, dtype=float)
    carrier_area_rate = float(
        Q
        * max(
            np.sum(np.abs(state_rate.n) * widths),
            np.sum(np.abs(state_rate.p) * widths),
        )
    )
    ion_area_rate = float(Q * np.sum(np.abs(state_rate.P) * widths))
    if state_rate.P_neg is not None:
        ion_area_rate += float(Q * np.sum(np.abs(state_rate.P_neg) * widths))
    current = compute_current_components(
        x,
        y,
        stack,
        V_dc,
        mat=mat,
    )
    max_ionic_current = float(np.max(np.abs(current.J_ion)))
    face_spread = float(np.ptp(current.J_total))
    contact = assess_contact_thermodynamics(stack, mat)

    reasons: list[str] = []
    if np.any(~np.isfinite(state)):
        reasons.append("dc_state_nonfinite")
    if np.any(~np.isfinite(rate)):
        reasons.append("state_rate_nonfinite")
    for name, value, limit in (
        (
            "carrier_area_rate",
            carrier_area_rate,
            max_carrier_area_rate_A_m2,
        ),
        ("ion_area_rate", ion_area_rate, max_ion_area_rate_A_m2),
        (
            "ionic_face_current",
            max_ionic_current,
            max_ionic_face_current_A_m2,
        ),
        (
            "dc_face_current_spread",
            face_spread,
            max_dc_face_spread_A_m2,
        ),
    ):
        if not np.isfinite(value):
            reasons.append(f"{name}_nonfinite")
        elif value > limit:
            reasons.append(f"{name}_exceeds_limit")
    numerical = not reasons
    if not contact.certified:
        reasons.append(f"contact_thermodynamics_{contact.status}")
    return OperatingPointCertificate(
        certified=numerical and contact.certified,
        numerically_certified=numerical,
        thermodynamically_certified=contact.certified,
        source=source,
        carrier_area_rate_A_m2=carrier_area_rate,
        ion_area_rate_A_m2=ion_area_rate,
        max_ionic_face_current_A_m2=max_ionic_current,
        dc_face_current_spread_A_m2=face_spread,
        carrier_area_rate_limit_A_m2=max_carrier_area_rate_A_m2,
        ion_area_rate_limit_A_m2=max_ion_area_rate_A_m2,
        ionic_face_current_limit_A_m2=max_ionic_face_current_A_m2,
        dc_face_current_spread_limit_A_m2=max_dc_face_spread_A_m2,
        contact_thermodynamics=contact,
        reasons=tuple(reasons),
    )


def _qf_operating_point_certificate(
    stack: DeviceStack,
    mat,
    dc_state,
) -> OperatingPointCertificate:
    """Promote the QF solver's existing DC certificate without discarding it."""
    contact = assess_contact_thermodynamics(stack, mat)
    electron_bound = float(dc_state.electron_continuity_bound_A_m2)
    hole_bound = float(dc_state.hole_continuity_bound_A_m2)
    face_spread = float(dc_state.face_current_spread_A_m2)
    reasons: list[str] = []
    if not bool(dc_state.certified):
        reasons.append("qf_dc_state_not_certified")
    for name, value in (
        ("qf_electron_continuity_bound", electron_bound),
        ("qf_hole_continuity_bound", hole_bound),
        ("qf_face_current_spread", face_spread),
    ):
        if not np.isfinite(value):
            reasons.append(f"{name}_nonfinite")
        elif value < 0.0:
            reasons.append(f"{name}_negative")
    numerical = not reasons
    if not contact.certified:
        reasons.append(f"contact_thermodynamics_{contact.status}")
    return OperatingPointCertificate(
        certified=numerical and contact.certified,
        numerically_certified=numerical,
        thermodynamically_certified=contact.certified,
        source="qf_residual_certified",
        carrier_area_rate_A_m2=float(max(electron_bound, hole_bound)),
        ion_area_rate_A_m2=0.0,
        max_ionic_face_current_A_m2=0.0,
        dc_face_current_spread_A_m2=face_spread,
        carrier_area_rate_limit_A_m2=None,
        ion_area_rate_limit_A_m2=None,
        ionic_face_current_limit_A_m2=None,
        dc_face_current_spread_limit_A_m2=None,
        contact_thermodynamics=contact,
        reasons=tuple(reasons),
    )


def _ion_aware_operating_point_certificate(
    dc_state: Any,
) -> OperatingPointCertificate:
    """Project the dedicated ion-aware DC certificate onto the public schema."""

    state_certificate = dc_state.state_certificate
    dc_protocol = dc_state.protocol
    return OperatingPointCertificate(
        certified=bool(state_certificate.certified),
        numerically_certified=bool(state_certificate.numerically_certified),
        thermodynamically_certified=bool(state_certificate.thermodynamically_certified),
        source="ion_aware_residual_certified",
        carrier_area_rate_A_m2=float(state_certificate.carrier_area_rate_A_m2),
        ion_area_rate_A_m2=float(state_certificate.ion_area_rate_A_m2),
        max_ionic_face_current_A_m2=float(
            state_certificate.max_ionic_face_current_A_m2
        ),
        dc_face_current_spread_A_m2=float(
            state_certificate.dc_face_current_spread_A_m2
        ),
        carrier_area_rate_limit_A_m2=float(dc_protocol.max_carrier_area_rate_A_m2),
        ion_area_rate_limit_A_m2=float(dc_protocol.max_ion_area_rate_A_m2),
        ionic_face_current_limit_A_m2=float(dc_protocol.max_ionic_face_current_A_m2),
        dc_face_current_spread_limit_A_m2=float(
            dc_protocol.max_dc_face_current_spread_A_m2
        ),
        contact_thermodynamics=state_certificate.contact_thermodynamics,
        reasons=tuple(state_certificate.reasons),
    )


def build_impedance_experiment_protocol(
    stack: DeviceStack,
    frequencies: np.ndarray,
    *,
    V_dc: float = 0.9,
    delta_V: float = 0.01,
    n_cycles: int = 5,
    n_extract: int = 2,
    points_per_cycle: int = 40,
    illuminated: bool = True,
    method: str = "transient",
    dc_settle_time: float = 1e-3,
    max_carrier_area_rate_A_m2: float = DEFAULT_MAX_CARRIER_AREA_RATE_A_M2,
    max_ion_area_rate_A_m2: float = DEFAULT_MAX_ION_AREA_RATE_A_M2,
    max_ionic_face_current_A_m2: float = DEFAULT_MAX_ION_AREA_RATE_A_M2,
    max_dc_face_spread_A_m2: float = DEFAULT_MAX_DC_FACE_SPREAD_A_M2,
    implicit_legacy_protocol: bool = False,
) -> ExperimentProtocol:
    """Describe the actual DC preparation and AC sampling of an IS run."""

    freq = np.asarray(frequencies, dtype=float)
    if freq.ndim != 1 or freq.size == 0:
        raise ValueError("frequencies must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(freq)) or np.any(freq <= 0.0):
        raise ValueError("frequencies must be finite and positive")
    if method not in _IMPEDANCE_METHOD_ALIASES:
        raise ValueError(f"unknown impedance method {method!r}")
    temperature = getattr(stack, "T", None)
    if temperature is None:
        if not implicit_legacy_protocol:
            raise TypeError("an explicit impedance protocol requires stack.T")
        # Preserve legacy pure-circuit unit-test stubs that use object().
        temperature = 300.0
    canonical_method = _IMPEDANCE_METHOD_ALIASES[method]
    if canonical_method == "transient_ion_aware":
        cycles = int(n_cycles)
        extraction_cycles = min(max(int(n_extract), 1), cycles)
        ppc = int(points_per_cycle)
        if illuminated or abs(V_dc) >= 1.0e-12:
            initial_state = "finite_time_dc_preconditioned"
            soak = float(dc_settle_time)
            condition = "baseline" if illuminated else "dark"
            history = (
                IlluminationStep(
                    phase="dc_preconditioning",
                    condition=condition,
                    duration_s=soak,
                    intensity_suns=1.0 if illuminated else None,
                    source_reference=(
                        "stack_baseline_generation" if illuminated else None
                    ),
                ),
                IlluminationStep(
                    phase="sinusoidal_ac_measurement",
                    condition=condition,
                    duration_s=float(np.sum(cycles / freq)),
                    intensity_suns=1.0 if illuminated else None,
                    source_reference=(
                        "stack_baseline_generation" if illuminated else None
                    ),
                ),
            )
            settle = DCSettleCriterion(
                kind="finite_time_with_certificate",
                duration_s=soak,
                max_carrier_area_rate_A_m2=max_carrier_area_rate_A_m2,
                max_ion_area_rate_A_m2=max_ion_area_rate_A_m2,
                max_ionic_face_current_A_m2=max_ionic_face_current_A_m2,
                max_face_current_spread_A_m2=max_dc_face_spread_A_m2,
            )
        else:
            initial_state = "dark_equilibrium"
            soak = 0.0
            history = (
                IlluminationStep(
                    phase="sinusoidal_ac_measurement",
                    condition="dark",
                    duration_s=float(np.sum(cycles / freq)),
                ),
            )
            settle = DCSettleCriterion(
                kind="residual_certified",
                max_carrier_area_rate_A_m2=max_carrier_area_rate_A_m2,
                max_ion_area_rate_A_m2=max_ion_area_rate_A_m2,
                max_ionic_face_current_A_m2=max_ionic_face_current_A_m2,
                max_face_current_spread_A_m2=max_dc_face_spread_A_m2,
            )
        ac = ACExcitation(
            dc_bias_V=float(V_dc),
            amplitude_V=float(delta_V),
            cycles=cycles,
            extraction_cycles=extraction_cycles,
            points_per_cycle=ppc,
        )
    elif canonical_method == "qf_frequency_ion_free":
        initial_state = "qf_dc_candidate"
        soak = None
        condition = "baseline" if illuminated else "dark"
        history = (
            IlluminationStep(
                phase="residual_dc_operating_point",
                condition=condition,
                intensity_suns=1.0 if illuminated else None,
                source_reference=("stack_baseline_generation" if illuminated else None),
            ),
            IlluminationStep(
                phase="frequency_domain_linear_response",
                condition=condition,
                intensity_suns=1.0 if illuminated else None,
                source_reference=("stack_baseline_generation" if illuminated else None),
            ),
        )
        settle = DCSettleCriterion(kind="residual_certified")
        ac = ACExcitation(
            dc_bias_V=float(V_dc),
            amplitude_V=float(delta_V),
        )
    elif canonical_method == DYNAMIC_DEFECT_IMPEDANCE_METHOD:
        initial_state = "qf_dc_candidate"
        soak = None
        condition = "baseline" if illuminated else "dark"
        history = (
            IlluminationStep(
                phase="residual_certified_dynamic_defect_dc",
                condition=condition,
                intensity_suns=1.0 if illuminated else None,
                source_reference=("stack_baseline_generation" if illuminated else None),
            ),
            IlluminationStep(
                phase="frequency_domain_dynamic_defect_linear_response",
                condition=condition,
                intensity_suns=1.0 if illuminated else None,
                source_reference=("stack_baseline_generation" if illuminated else None),
            ),
        )
        settle = DCSettleCriterion(kind="residual_certified")
        ac = ACExcitation(
            dc_bias_V=float(V_dc),
            amplitude_V=float(delta_V),
        )
    else:
        initial_state = "dark_equilibrium"
        soak = None
        condition = "baseline" if illuminated else "dark"
        history = (
            IlluminationStep(
                phase="residual_certified_ion_aware_dc",
                condition=condition,
                intensity_suns=1.0 if illuminated else None,
                source_reference=("stack_baseline_generation" if illuminated else None),
            ),
            IlluminationStep(
                phase="frequency_domain_ion_aware_linear_response",
                condition=condition,
                intensity_suns=1.0 if illuminated else None,
                source_reference=("stack_baseline_generation" if illuminated else None),
            ),
        )
        settle = DCSettleCriterion(
            kind="residual_certified",
            max_carrier_area_rate_A_m2=max_carrier_area_rate_A_m2,
            max_ion_area_rate_A_m2=max_ion_area_rate_A_m2,
            max_ionic_face_current_A_m2=max_ionic_face_current_A_m2,
            max_face_current_spread_A_m2=max_dc_face_spread_A_m2,
        )
        ac = ACExcitation(
            dc_bias_V=float(V_dc),
            amplitude_V=float(delta_V),
        )
    return ExperimentProtocol(
        experiment="impedance",
        initial_state_source=initial_state,
        pre_bias_V=float(V_dc),
        soak_duration_s=soak,
        dwell_duration_s=None,
        illumination_history=history,
        temperature_K=float(temperature),
        scan=ScanProtocol(
            axis="frequency_Hz",
            direction="declared_order",
            start=float(freq[0]),
            stop=float(freq[-1]),
        ),
        ac_excitation=ac,
        dc_settle=settle,
        sampling=SamplingProtocol(
            axis="frequency_Hz",
            mode="declared",
            values=tuple(freq),
        ),
        implicit_legacy_protocol=implicit_legacy_protocol,
    )


def run_impedance(
    stack: DeviceStack,
    frequencies: np.ndarray,
    V_dc: float = 0.9,
    delta_V: float = 0.01,
    N_grid: int = 60,
    n_cycles: int = 5,
    n_extract: int = 2,
    rtol: float = 1e-4,
    atol: AbsoluteTolerance = 1e-6,
    illuminated: bool = True,
    progress: ProgressCallback | None = None,
    allow_underresolved_grid: bool = False,
    method: str = "transient",
    dc_settle_time: float = 1e-3,
    require_operating_point_certificate: bool = False,
    require_frequency_window_certificate: bool = False,
    max_carrier_area_rate_A_m2: float = DEFAULT_MAX_CARRIER_AREA_RATE_A_M2,
    max_ion_area_rate_A_m2: float = DEFAULT_MAX_ION_AREA_RATE_A_M2,
    max_ionic_face_current_A_m2: float = DEFAULT_MAX_ION_AREA_RATE_A_M2,
    max_dc_face_spread_A_m2: float = DEFAULT_MAX_DC_FACE_SPREAD_A_M2,
    points_per_cycle: int = 40,
    experiment_protocol: ExperimentProtocol | None = None,
    protocol_mode: ProtocolMode = "compatibility",
    ion_aware_dc_atol: AbsoluteTolerance | None = None,
    dynamic_defect_protocol: DynamicDefectImpedanceProtocol | None = None,
    defect_energy_quadrature_order: int = DEFAULT_DEFECT_ENERGY_QUADRATURE_ORDER,
    dynamic_defect_state_step: float = 1.0e-5,
    dynamic_defect_voltage_step: float = 1.0e-5,
    dynamic_defect_refinement_factors: object = (
        DEFAULT_DYNAMIC_DEFECT_REFINEMENT_FACTORS
    ),
    dynamic_defect_gates: DynamicDefectImpedanceGates | None = None,
) -> ImpedanceResult:
    """Run small-signal impedance at each frequency.

    Parameters
    ----------
    n_cycles : int
        Total AC cycles to simulate (includes settling + extraction).
    n_extract : int
        Number of *final* cycles used for lock-in extraction. The preceding
        ``n_cycles - n_extract`` cycles serve as ionic-settling warm-up.
        Using >1 cycle for extraction reduces noise via averaging.
    points_per_cycle : int, default 40
        Uniform AC time resolution. The returned protocol records this value
        so 40/80/160-point time-resolution ladders can be compared directly.
    illuminated : bool, default True
        If True (default), the DC steady-state and every AC cycle run
        under AM1.5G illumination — appropriate for operating-point
        impedance spectroscopy of solar cells. If False, both legs run
        in the dark (G = 0 everywhere) — required for Mott-Schottky C-V
        analysis, where photogenerated carriers would mask the depletion
        capacitance.
    allow_underresolved_grid : bool, default False
        Diagnostic-only escape hatch for an electrical mesh that fails the
        thick-layer Debye-resolution guard. Such a run is not physically
        certifiable.
    method : {"transient", "transient_ion_aware",
              "quasi_fermi_frequency", "qf_frequency_ion_free",
              "ion_aware_frequency", "ion_aware_frequency_certified",
              "dynamic_defect_frequency",
              "dynamic_defect_frequency_certified"}
        The aliases resolve to an explicit ion-aware transient protocol, an
        ion-free residual-certified QF frequency-domain protocol, or the
        residual-certified mobile-ion frequency-domain reference engine, or
        the protocol-bound dynamic explicit-defect frequency engine.
    dc_settle_time : float, default 1e-3
        Finite DC preconditioning interval used by the transient engine. It is
        part of the returned protocol and is not itself a steady-state proof.
    require_operating_point_certificate : bool, default False
        Fail closed unless the DC residual/current and contact-thermodynamic
        gates all pass. The default preserves compatibility while publishing
        the complete certificate in ``ImpedanceResult``.
    require_frequency_window_certificate : bool, default False
        For the certified ion-aware frequency engine, fail closed unless the
        requested samples densely cover its model-derived ionic timescales.
    ion_aware_dc_atol : float or ComponentwiseAtol, optional
        Absolute-tolerance policy for the certified ion-aware DC ladder. The
        componentwise density policy is used when omitted. The legacy
        transient engine continues to use ``atol`` unchanged.
    """
    if len(frequencies) == 0:
        raise ValueError("frequencies must be non-empty")
    if np.any(~np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
        raise ValueError("frequencies must be finite and positive")
    if N_grid < 3:
        raise ValueError(f"N_grid must be >= 3, got {N_grid}")
    if not np.isfinite(V_dc):
        raise ValueError("V_dc must be finite")
    if not np.isfinite(delta_V) or not 0.0 < delta_V < MAX_LINEAR_PERTURBATION_V:
        raise ValueError(
            "delta_V must be finite, positive, and below the 20 mV small-signal limit"
        )
    if isinstance(n_cycles, bool) or int(n_cycles) != n_cycles or n_cycles < 1:
        raise ValueError(f"n_cycles must be >= 1, got {n_cycles}")
    n_cycles = int(n_cycles)
    if (
        isinstance(points_per_cycle, bool)
        or int(points_per_cycle) != points_per_cycle
        or points_per_cycle < 8
    ):
        raise ValueError(
            f"points_per_cycle must be an integer >= 8, got {points_per_cycle}"
        )
    points_per_cycle = int(points_per_cycle)
    if isinstance(n_extract, bool) or int(n_extract) != n_extract:
        raise ValueError(f"n_extract must be an integer, got {n_extract}")
    n_extract = int(n_extract)
    if method not in _IMPEDANCE_METHOD_ALIASES:
        raise ValueError(
            "method must select the transient ion-aware, QF frequency "
            "ion-free, or certified ion-aware frequency engine, got "
            f"{method!r}"
        )
    canonical_method = _IMPEDANCE_METHOD_ALIASES[method]
    if (
        dynamic_defect_protocol is not None
        and canonical_method != DYNAMIC_DEFECT_IMPEDANCE_METHOD
    ):
        raise ValueError(
            "dynamic_defect_protocol is valid only with "
            f"method={DYNAMIC_DEFECT_IMPEDANCE_METHOD!r}"
        )
    if not np.isfinite(dc_settle_time) or dc_settle_time <= 0.0:
        raise ValueError("dc_settle_time must be finite and positive")
    for name, value in (
        ("max_carrier_area_rate_A_m2", max_carrier_area_rate_A_m2),
        ("max_ion_area_rate_A_m2", max_ion_area_rate_A_m2),
        ("max_ionic_face_current_A_m2", max_ionic_face_current_A_m2),
        ("max_dc_face_spread_A_m2", max_dc_face_spread_A_m2),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    n_extract = min(max(n_extract, 1), n_cycles)
    resolved_experiment_protocol = resolve_experiment_protocol(
        experiment_protocol,
        build_impedance_experiment_protocol(
            stack,
            np.asarray(frequencies, dtype=float),
            V_dc=V_dc,
            delta_V=delta_V,
            n_cycles=n_cycles,
            n_extract=n_extract,
            points_per_cycle=points_per_cycle,
            illuminated=illuminated,
            method=canonical_method,
            dc_settle_time=dc_settle_time,
            max_carrier_area_rate_A_m2=max_carrier_area_rate_A_m2,
            max_ion_area_rate_A_m2=max_ion_area_rate_A_m2,
            max_ionic_face_current_A_m2=max_ionic_face_current_A_m2,
            max_dc_face_spread_A_m2=max_dc_face_spread_A_m2,
            implicit_legacy_protocol=True,
        ),
        mode=protocol_mode,
    )
    protocol = ImpedanceProtocol(
        method=canonical_method,
        V_dc=float(V_dc),
        delta_V=float(delta_V),
        illuminated=bool(illuminated),
        dc_settle_time=(
            float(dc_settle_time) if canonical_method == "transient_ion_aware" else None
        ),
        n_cycles=(int(n_cycles) if canonical_method == "transient_ion_aware" else None),
        n_extract=(
            int(n_extract) if canonical_method == "transient_ion_aware" else None
        ),
        points_per_cycle=(
            int(points_per_cycle) if canonical_method == "transient_ion_aware" else None
        ),
        experiment_protocol=resolved_experiment_protocol,
        dynamic_defect_protocol=None,
    )

    # Use the same executable electrical-grid contract as J-V and steady-state
    # experiments, including configured interval weights and per-layer
    # clustering. A different impedance mesh would make C-V results
    # incomparable with the rest of the solver and can collapse a resolved
    # depletion region into the geometric-capacitance limit.
    x = build_electrical_grid(stack, N_grid)
    dynamic_capability = None
    if canonical_method == DYNAMIC_DEFECT_IMPEDANCE_METHOD:
        try:
            dynamic_capability = classify_dynamic_defect_capability(stack)
        except DynamicDefectImpedanceCapabilityError as exc:
            raise ImpedanceCapabilityError(str(exc)) from exc
    grid_diagnostics = require_thick_layer_interface_resolution(
        x,
        stack,
        N_grid=N_grid,
        allow_underresolved_grid=allow_underresolved_grid,
    )
    grid_assessment = _assess_impedance_grid(
        grid_diagnostics,
        allow_underresolved_grid=allow_underresolved_grid,
    )
    if (
        require_operating_point_certificate
        or canonical_method == DYNAMIC_DEFECT_IMPEDANCE_METHOD
    ) and not grid_assessment.certified:
        raise ImpedanceCertificationError(
            "impedance electrical grid is uncertified: "
            + ", ".join(grid_assessment.warnings)
        )
    if dynamic_capability is not None and "interface" in dynamic_capability:
        x = build_two_sided_trace_grid(x, stack)
    if canonical_method == DYNAMIC_DEFECT_IMPEDANCE_METHOD:
        try:
            expected_dynamic_protocol = build_dynamic_defect_impedance_protocol(
                stack,
                x,
                np.asarray(frequencies, dtype=float),
                requested_grid_intervals=N_grid,
                V_dc=V_dc,
                delta_V=delta_V,
                illuminated=illuminated,
                defect_energy_quadrature_order=(defect_energy_quadrature_order),
                state_step=dynamic_defect_state_step,
                voltage_step=dynamic_defect_voltage_step,
                refinement_factors=dynamic_defect_refinement_factors,
                gates=dynamic_defect_gates,
            )
            resolved_dynamic_protocol = resolve_dynamic_defect_impedance_protocol(
                dynamic_defect_protocol,
                expected_dynamic_protocol,
            )
            protocol = replace(
                protocol,
                dynamic_defect_protocol=resolved_dynamic_protocol,
            )
            dynamic_result = run_dynamic_defect_impedance(
                x,
                stack,
                resolved_dynamic_protocol,
                progress=progress,
            )
        except DynamicDefectImpedanceCapabilityError as exc:
            raise ImpedanceCapabilityError(str(exc)) from exc
        except (
            DynamicDefectImpedanceCertificationError,
            DynamicDefectImpedanceProtocolError,
        ) as exc:
            raise ImpedanceCertificationError(str(exc)) from exc
        return ImpedanceResult(
            frequencies=dynamic_result.frequencies_Hz,
            Z=dynamic_result.impedance_ohm_m2,
            protocol=protocol,
            operating_point=None,
            frequency_window=None,
            grid_assessment=grid_assessment,
            diagnostics=ImpedanceDiagnostics(
                admittance_S_m2=dynamic_result.admittance_S_m2,
                admittance_faces_S_m2=dynamic_result.admittance_faces_S_m2,
                electron_storage_response_F_m2=(
                    dynamic_result.electron_storage_response_F_m2
                ),
                hole_storage_response_F_m2=(dynamic_result.hole_storage_response_F_m2),
                conduction_admittance_faces_S_m2=(
                    dynamic_result.electron_admittance_faces_S_m2
                    + dynamic_result.hole_admittance_faces_S_m2
                ),
                displacement_admittance_faces_S_m2=(
                    dynamic_result.displacement_admittance_faces_S_m2
                ),
                electron_admittance_faces_S_m2=(
                    dynamic_result.electron_admittance_faces_S_m2
                ),
                hole_admittance_faces_S_m2=(dynamic_result.hole_admittance_faces_S_m2),
                positive_ion_admittance_faces_S_m2=(
                    dynamic_result.positive_ion_admittance_faces_S_m2
                ),
                negative_ion_admittance_faces_S_m2=(
                    dynamic_result.negative_ion_admittance_faces_S_m2
                ),
                positive_ion_storage_response_F_m2=(
                    dynamic_result.positive_ion_storage_response_F_m2
                ),
                negative_ion_storage_response_F_m2=(
                    dynamic_result.negative_ion_storage_response_F_m2
                ),
                bulk_trap_charge_storage_response_F_m2=(
                    dynamic_result.bulk_trap_charge_storage_response_F_m2
                ),
                interface_sheet_charge_storage_response_F_m2=(
                    dynamic_result.interface_sheet_charge_storage_response_F_m2
                ),
                bulk_trap_occupancy_response_per_V=(
                    dynamic_result.bulk_trap_occupancy_response_per_V
                ),
                interface_occupancy_response_per_V=(
                    dynamic_result.interface_occupancy_response_per_V
                ),
            ),
            dynamic_defect_evidence=dynamic_result.evidence,
        )
    dx_faces = np.diff(x)
    L_total = float(x[-1] - x[0])
    # Build the material cache once — reused across every frequency and
    # every RHS call inside each frequency's transient.
    mat = build_material_arrays(x, stack)
    if mat.N_iface_state > 0:
        raise ImpedanceCapabilityError(
            "impedance does not support dynamic interface-state blocks until "
            "their residual and electrostatic charge are jointly certified"
        )
    frequency_window = assess_impedance_frequency_window(
        x,
        mat,
        np.asarray(frequencies, dtype=float),
    )
    if canonical_method == "ion_aware_frequency_certified":
        from perovskite_sim.experiments.ion_aware_dc import (
            IonAwareDCCapabilityError,
            IonAwareDCCertificationError,
            IonAwareDCSolverError,
            build_ion_aware_dc_protocol,
            solve_ion_aware_dc,
        )
        from perovskite_sim.experiments.ion_aware_impedance import (
            IonAwareImpedanceCapabilityError,
            IonAwareImpedanceCertificationError,
            IonAwareImpedanceError,
            build_ion_aware_impedance_protocol,
            run_ion_aware_impedance,
        )

        dc_protocol = build_ion_aware_dc_protocol(
            stack,
            V_dc=V_dc,
            illuminated=illuminated,
            max_carrier_area_rate_A_m2=max_carrier_area_rate_A_m2,
            max_ion_area_rate_A_m2=max_ion_area_rate_A_m2,
            max_ionic_face_current_A_m2=max_ionic_face_current_A_m2,
            max_dc_face_current_spread_A_m2=max_dc_face_spread_A_m2,
        )
        effective_dc_atol = (
            ComponentwiseAtol() if ion_aware_dc_atol is None else ion_aware_dc_atol
        )
        try:
            dc_state = solve_ion_aware_dc(
                x,
                stack,
                dc_protocol,
                mat=mat,
                rtol=rtol,
                atol=effective_dc_atol,
                require_numerical_certificate=True,
                require_contact_certificate=False,
                progress=progress,
            )
        except IonAwareDCCapabilityError as exc:
            raise ImpedanceCapabilityError(str(exc)) from exc
        except (IonAwareDCCertificationError, IonAwareDCSolverError) as exc:
            raise ImpedanceCertificationError(str(exc)) from exc

        operating_point = _ion_aware_operating_point_certificate(dc_state)
        if require_operating_point_certificate and not operating_point.certified:
            raise ImpedanceCertificationError(
                "impedance DC operating point is uncertified: "
                + ", ".join(operating_point.reasons)
            )

        frequency_protocol = build_ion_aware_impedance_protocol(
            dc_state,
            np.asarray(frequencies, dtype=float),
            delta_V=delta_V,
        )
        try:
            ion_result = run_ion_aware_impedance(
                x,
                stack,
                frequency_protocol,
                dc_state=dc_state,
                mat=mat,
                require_numerical_certificate=True,
                require_contact_certificate=False,
                require_frequency_window_certificate=(
                    require_frequency_window_certificate
                ),
                progress=progress,
            )
        except IonAwareImpedanceCapabilityError as exc:
            raise ImpedanceCapabilityError(str(exc)) from exc
        except (
            IonAwareImpedanceCertificationError,
            IonAwareImpedanceError,
        ) as exc:
            raise ImpedanceCertificationError(str(exc)) from exc

        reference = ion_result.reference_linearization
        certificate = ion_result.certificate
        return ImpedanceResult(
            frequencies=ion_result.frequencies,
            Z=ion_result.Z,
            protocol=protocol,
            operating_point=operating_point,
            frequency_window=ion_result.frequency_window,
            grid_assessment=grid_assessment,
            diagnostics=ImpedanceDiagnostics(
                admittance_S_m2=ion_result.Y,
                admittance_faces_S_m2=ion_result.Y_faces,
                max_relative_face_spread=reference.max_relative_face_spread,
                reciprocal_condition=reference.reciprocal_condition,
                backward_error=reference.backward_error,
                electron_storage_response_F_m2=(
                    ion_result.electron_storage_response_F_m2
                ),
                hole_storage_response_F_m2=(ion_result.hole_storage_response_F_m2),
                conduction_admittance_faces_S_m2=(
                    ion_result.conduction_admittance_faces_S_m2
                ),
                displacement_admittance_faces_S_m2=(
                    ion_result.displacement_admittance_faces_S_m2
                ),
                electron_admittance_faces_S_m2=(
                    ion_result.electron_admittance_faces_S_m2
                ),
                hole_admittance_faces_S_m2=(ion_result.hole_admittance_faces_S_m2),
                positive_ion_admittance_faces_S_m2=(
                    ion_result.positive_ion_admittance_faces_S_m2
                ),
                negative_ion_admittance_faces_S_m2=(
                    ion_result.negative_ion_admittance_faces_S_m2
                ),
                positive_ion_storage_response_F_m2=(
                    ion_result.positive_ion_storage_response_F_m2
                ),
                negative_ion_storage_response_F_m2=(
                    ion_result.negative_ion_storage_response_F_m2
                ),
                net_charge_storage_response_F_m2=(
                    ion_result.net_charge_storage_response_F_m2
                ),
            ),
            ion_aware_evidence=IonAwareImpedanceEvidence(
                dc_protocol=dc_protocol,
                frequency_protocol=frequency_protocol,
                dc_state_certificate=dc_state.state_certificate,
                dc_protocol_sha256=dc_state.protocol_hash,
                dc_state_sha256=frequency_protocol.dc_state_sha256,
                frequency_protocol_sha256=frequency_protocol.protocol_hash,
                dc_total_settle_time_s=float(dc_state.total_settle_time_s),
                dc_consecutive_certified_steps=int(
                    dc_state.consecutive_certified_steps
                ),
                dc_solver_rtol=float(rtol),
                dc_solver_atol=effective_dc_atol,
                numerically_certified=bool(certificate.numerically_certified),
                thermodynamically_certified=bool(
                    certificate.thermodynamically_certified
                ),
                frequency_window_certified=bool(certificate.frequency_window_certified),
                certified=bool(certificate.certified),
                max_relative_face_spread=float(certificate.max_relative_face_spread),
                max_backward_error=float(certificate.max_backward_error),
                minimum_reciprocal_condition=float(
                    certificate.minimum_reciprocal_condition
                ),
                max_mass_diagonal_relative_error=float(
                    certificate.max_mass_diagonal_relative_error
                ),
                max_mass_off_diagonal_relative=float(
                    certificate.max_mass_off_diagonal_relative
                ),
                max_ion_inventory_response_relative=float(
                    certificate.max_ion_inventory_response_relative
                ),
                max_current_decomposition_relative_error=float(
                    certificate.max_current_decomposition_relative_error
                ),
                perturbation_assessments=certificate.perturbation_assessments,
                frequency_point_certificates=(certificate.frequency_point_certificates),
                reasons=tuple(certificate.reasons),
            ),
        )
    if canonical_method == "qf_frequency_ion_free":
        from perovskite_sim.experiments.quasi_fermi_impedance import (
            run_quasi_fermi_impedance,
        )

        result = run_quasi_fermi_impedance(
            x,
            stack,
            np.asarray(frequencies, dtype=float),
            V_dc=V_dc,
            delta_V=delta_V,
            illuminated=illuminated,
            mat=mat,
            progress=progress,
        )
        operating_point = _qf_operating_point_certificate(
            stack,
            mat,
            result.dc_state,
        )
        if any(reason.endswith("_nonfinite") for reason in operating_point.reasons):
            raise ImpedanceCertificationError(
                "impedance DC operating point contains non-finite evidence: "
                + ", ".join(operating_point.reasons)
            )
        if require_operating_point_certificate and not operating_point.certified:
            raise ImpedanceCertificationError(
                "impedance DC operating point is uncertified: "
                + ", ".join(operating_point.reasons)
            )
        return ImpedanceResult(
            frequencies=result.frequencies,
            Z=result.Z,
            protocol=protocol,
            operating_point=operating_point,
            frequency_window=frequency_window,
            grid_assessment=grid_assessment,
            diagnostics=ImpedanceDiagnostics(
                admittance_S_m2=result.Y,
                admittance_faces_S_m2=result.Y_faces,
                max_relative_face_spread=result.max_relative_face_spread,
                reciprocal_condition=result.reciprocal_condition,
                backward_error=result.backward_error,
                electron_storage_response_F_m2=(result.electron_storage_response_F_m2),
                hole_storage_response_F_m2=result.hole_storage_response_F_m2,
            ),
        )

    # Pre-condition: DC steady state at V_dc. Illuminated path uses the
    # dark→light solver; dark path starts from equilibrium and (if
    # V_dc ≠ 0) drives to V_dc via a short dark transient so the AC
    # cycles begin at the correct operating point.
    if illuminated:
        y_dc = solve_illuminated_ss(
            x,
            stack,
            V_app=V_dc,
            t_settle=dc_settle_time,
            rtol=rtol,
            atol=atol,
            mat=mat,
        )
        dc_source = "finite_time_preconditioned"
    else:
        y_eq = solve_equilibrium(x, stack)
        if abs(V_dc) < 1e-12:
            y_dc = y_eq
            dc_source = "dark_equilibrium"
        else:
            sol_dc = run_transient(
                x,
                y_eq,
                (0.0, dc_settle_time),
                np.array([dc_settle_time]),
                stack,
                illuminated=False,
                V_app=V_dc,
                rtol=rtol,
                atol=atol,
                mat=mat,
            )
            if not sol_dc.success:
                detail = getattr(sol_dc, "message", "no solver diagnostic")
                raise RuntimeError(
                    "dark DC preconditioning failed before impedance "
                    f"integration at V_dc={V_dc:.6g} V: {detail}"
                )
            y_dc = sol_dc.y[:, -1]
            dc_source = "finite_time_preconditioned"

    operating_point = _transient_operating_point_certificate(
        x,
        y_dc,
        stack,
        mat,
        V_dc=V_dc,
        illuminated=illuminated,
        source=dc_source,
        max_carrier_area_rate_A_m2=max_carrier_area_rate_A_m2,
        max_ion_area_rate_A_m2=max_ion_area_rate_A_m2,
        max_ionic_face_current_A_m2=max_ionic_face_current_A_m2,
        max_dc_face_spread_A_m2=max_dc_face_spread_A_m2,
    )
    if any(reason.endswith("_nonfinite") for reason in operating_point.reasons):
        raise ImpedanceCertificationError(
            "impedance DC operating point contains non-finite evidence: "
            + ", ".join(operating_point.reasons)
        )
    if require_operating_point_certificate and not operating_point.certified:
        raise ImpedanceCertificationError(
            "impedance DC operating point is uncertified: "
            + ", ".join(operating_point.reasons)
        )

    Z_arr = np.zeros(len(frequencies), dtype=complex)
    for k, f in enumerate(frequencies):
        T_period = 1.0 / f
        dt = T_period / points_per_cycle
        n_intervals = n_cycles * points_per_cycle
        t_edges = np.arange(n_intervals + 1, dtype=float) * dt
        t_mid = t_edges[:-1] + 0.5 * dt
        t_samples = np.empty(2 * n_intervals + 1, dtype=float)
        t_samples[0::2] = t_edges
        t_samples[1::2] = t_mid

        def V_ac(t):
            return V_dc + delta_V * np.sin(2 * np.pi * f * t)

        # One continuous solve per frequency lets Radau see the sinusoidal
        # boundary waveform. Edge and midpoint states are both retained so the
        # centered displacement current and conduction current are co-located
        # at the same lock-in timestamp.
        sol = run_transient(
            x,
            y_dc.copy(),
            (0.0, float(t_edges[-1])),
            t_samples,
            stack,
            illuminated=illuminated,
            V_app=V_ac,
            rtol=rtol,
            atol=atol,
            max_step=dt / 5.0,
            mat=mat,
        )
        if not sol.success:
            detail = getattr(sol, "message", "no solver diagnostic")
            raise RuntimeError(f"impedance transient failed at f={f:.3e} Hz: {detail}")
        y_edges = sol.y[:, 0::2]
        y_mid = sol.y[:, 1::2]
        J_t = np.zeros(n_intervals, dtype=float)
        for i in range(n_intervals):
            V_lo = V_ac(t_edges[i])
            V_hi = V_ac(t_edges[i + 1])
            V_mid = V_ac(t_mid[i])
            total_at_edge = _total_current_faces(
                x,
                y_edges[:, i + 1],
                stack,
                V_hi,
                y_prev=y_edges[:, i],
                dt=dt,
                mat=mat,
                V_app_prev=V_lo,
            )
            conduction_at_edge = compute_current_components(
                x,
                y_edges[:, i + 1],
                stack,
                V_hi,
                mat=mat,
            ).J_total
            conduction_at_mid = compute_current_components(
                x,
                y_mid[:, i],
                stack,
                V_mid,
                mat=mat,
            ).J_total
            J_face_mid = conduction_at_mid + total_at_edge - conduction_at_edge
            J_t[i] = float(np.sum(J_face_mid * dx_faces) / L_total)

        # Lock-in over the last n_extract cycles. Using multiple cycles
        # averages residual transient noise; the helper jointly fits offset,
        # linear drift, sine, and cosine instead of subtracting drift first.
        n_extract_pts = n_extract * points_per_cycle
        J_ext = J_t[-n_extract_pts:]
        t_ext = t_mid[-n_extract_pts:]
        # Passive-convention current is -J (so that a positive δV drives a
        # positive Î through a resistive device). The lock-in helper assumes
        # passive convention so the sign flip happens here.
        Z_arr[k] = _lockin_extract(-J_ext, t_ext, f, delta_V)

        if progress is not None:
            progress("impedance", k + 1, len(frequencies), f"f={f:.3e} Hz")

    return ImpedanceResult(
        frequencies=np.asarray(frequencies, dtype=float),
        Z=Z_arr,
        protocol=protocol,
        operating_point=operating_point,
        frequency_window=frequency_window,
        grid_assessment=grid_assessment,
    )
