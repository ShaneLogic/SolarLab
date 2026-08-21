from __future__ import annotations
import dataclasses
import hashlib
import warnings
from dataclasses import dataclass
from typing import Callable, Literal, Sequence
import numpy as np

from perovskite_sim._compat.numpy_compat import trapezoid
from perovskite_sim.discretization.fe_operators import bernoulli
from perovskite_sim.discretization.grid import (
    Layer,
    multilayer_grid,
    require_thick_layer_interface_resolution,
)
from perovskite_sim.physics.ion_migration import ion_face_flux
from perovskite_sim.physics.regularization import RHSRegularization
from perovskite_sim.solver.numerical_diagnostics import (
    NumericalDiagnosticsReport,
)
from perovskite_sim.physics.poisson import solve_poisson_prefactored
from perovskite_sim.solver.illuminated_ss import (
    IlluminatedPreconditionerDiagnostics,
    solve_illuminated_ss,
)
from perovskite_sim.solver.newton import solve_equilibrium
from perovskite_sim.solver.mol import (
    StateVec, run_transient,
    MaterialArrays, build_material_arrays,
    _charge_density,
    _harmonic_face_average,
    poisson_right_boundary,
)
from perovskite_sim.models.device import DeviceStack, electrical_layers
from perovskite_sim.physics.grading import has_grading_params
from perovskite_sim.models.current import CurrentComponents, IonicCurrentComponents
from perovskite_sim.models.spatial import SpatialSnapshot
from perovskite_sim.constants import EPS_0, Q
from perovskite_sim.experiments.protocol import (
    DCSettleCriterion,
    ExperimentProtocol,
    IlluminationStep,
    ProtocolMismatchError,
    ProtocolMode,
    SamplingProtocol,
    ScanProtocol,
    resolve_experiment_protocol,
)


# Callback signature: stage, current, total, message.
ProgressCallback = Callable[[str, int, int, str], None]


@dataclass(frozen=True)
class JVMetrics:
    V_oc: float
    J_sc: float
    FF: float
    PCE: float
    voc_bracketed: bool = True
    """``True`` iff an open-circuit point was resolved inside the sampled
    voltage range, in which case V_oc / FF / PCE are physical. ``False``
    carries sentinel zeros for those three and has exactly two causes: the
    current never reached zero in the window (V_max stopped short), or the
    extracted V_oc exceeded the ``V_oc_max`` ceiling. ``J_sc`` is still
    meaningful (interpolated at V=0) and is returned in both cases. See
    ``compute_metrics`` for the full extraction contract — in particular
    that "reached zero" means reached the current-resolution floor, so a
    device whose photocurrent collapses before the diode turns on is
    ``True``, not ``False``."""


@dataclass(frozen=True)
class JVCurrentDecomp:
    """Per-voltage-point current decomposition (contact-face values, A/m²)."""
    J_n: np.ndarray
    J_p: np.ndarray
    J_ion: np.ndarray
    J_disp: np.ndarray
    J_total: np.ndarray


@dataclass(frozen=True)
class JVAcceptedSolveDiagnostics:
    """One solver segment on the accepted integration path."""

    report: NumericalDiagnosticsReport
    nfev: int | None = None
    njev: int | None = None
    nlu: int | None = None


@dataclass(frozen=True)
class JVPointStatus:
    """Numerical certificate for one voltage point in a J-V branch.

    The valid flag certifies both the local solve and the complete warm-start
    state chain leading to this point. A locally successful point is therefore
    still invalid when upstream_valid is false: its initial state came from an
    uncertified predecessor and cannot silently restart a trusted branch.
    candidate_current preserves the rejected diagnostic value; JVResult stores
    NaN in the corresponding J array instead.
    """

    branch: str
    index: int
    voltage: float
    valid: bool = True
    upstream_valid: bool = True
    refinement_required: bool = False
    refinement_converged: bool | None = None
    attempted_legs: tuple[int, ...] = (1,)
    attempted_currents: tuple[float | None, ...] = (None,)
    failed_legs: tuple[int, ...] = ()
    accepted_legs: int = 1
    reason_code: str = "certified"
    message: str = ""
    candidate_current: float | None = None
    solver: str = "transient"
    max_normalized_residual: float | None = None
    electron_continuity_bound_A_m2: float | None = None
    hole_continuity_bound_A_m2: float | None = None
    face_current_spread_A_m2: float | None = None
    poisson_residual: float | None = None
    numerical_diagnostics: tuple[JVAcceptedSolveDiagnostics, ...] = ()
    nfev: int | None = None
    njev: int | None = None
    nlu: int | None = None


class JVCertificationError(RuntimeError):
    """Raised when strict J-V execution cannot certify a voltage point."""

    def __init__(self, status: JVPointStatus):
        self.status = status
        super().__init__(
            f"J-V {status.branch} point {status.index} at "
            f"V={status.voltage:.6g} V is uncertified "
            f"({status.reason_code}): {status.message}"
        )


class JVDriverCapabilityError(RuntimeError):
    """Raised when a stack requires a different certified J-V driver."""

    def __init__(self, *, policy: str, requested_driver: str):
        self.policy = policy
        self.requested_driver = requested_driver
        super().__init__(
            f"J-V driver {requested_driver!r} is not production-certified "
            f"for stack policy {policy!r}; select solver='quasi_fermi'. "
            "For numerical diagnosis only, pass "
            "allow_unvalidated_driver=True explicitly."
        )


def require_jv_driver_capability(
    stack: DeviceStack,
    *,
    requested_driver: str,
    allow_unvalidated_driver: bool = False,
) -> None:
    """Fail closed when the stack's certified J-V variable set is required."""

    requires_qf = (
        stack.jv_solver_policy == "cancellation_safe_qf_required"
        and requested_driver != "quasi_fermi"
    )
    if requires_qf and not allow_unvalidated_driver:
        raise JVDriverCapabilityError(
            policy=stack.jv_solver_policy,
            requested_driver=requested_driver,
        )
    if requires_qf:
        warnings.warn(
            f"J-V driver {requested_driver!r} is running under the explicit "
            "allow_unvalidated_driver override; output is diagnostic and is "
            "not a production-certified physical J-V curve.",
            JVCertificationWarning,
            stacklevel=2,
        )


class JVCertificationWarning(UserWarning):
    """Diagnostic-mode notice that is distinct from solver numeric warnings."""


class _JVCandidateFailure(RuntimeError):
    """Internal bridge from numeric failures to a point certificate."""

    def __init__(
        self,
        reason_code: str,
        message: str,
        *,
        state: np.ndarray | None = None,
    ):
        self.reason_code = reason_code
        self.state = None if state is None else np.asarray(state).copy()
        super().__init__(message)


@dataclass(frozen=True)
class JVResult:
    V_fwd: np.ndarray
    J_fwd: np.ndarray
    V_rev: np.ndarray
    J_rev: np.ndarray
    metrics_fwd: JVMetrics
    metrics_rev: JVMetrics
    hysteresis_index: float
    snapshots_fwd: tuple[SpatialSnapshot, ...] | None = None
    snapshots_rev: tuple[SpatialSnapshot, ...] | None = None
    decomp_fwd: JVCurrentDecomp | None = None
    decomp_rev: JVCurrentDecomp | None = None
    status_fwd: tuple[JVPointStatus, ...] | None = None
    status_rev: tuple[JVPointStatus, ...] | None = None
    initial_numerical_diagnostics: IlluminatedPreconditionerDiagnostics | None = None
    protocol: ExperimentProtocol | None = None
    rhs_regularization: RHSRegularization | None = None

    @property
    def certified(self) -> bool:
        """Whether both branches carry complete, valid point certificates."""
        def _branch_is_complete(
            V: np.ndarray,
            J: np.ndarray,
            statuses: tuple[JVPointStatus, ...] | None,
            branch: str,
        ) -> bool:
            try:
                V_arr = np.asarray(V, dtype=float)
                J_arr = np.asarray(J, dtype=float)
            except (TypeError, ValueError):
                return False
            if (
                statuses is None
                or V_arr.ndim != 1
                or J_arr.ndim != 1
                or V_arr.shape != J_arr.shape
                or len(statuses) != len(V_arr)
                or len(statuses) == 0
                or not np.all(np.isfinite(V_arr))
                or not np.all(np.isfinite(J_arr))
            ):
                return False
            return all(
                isinstance(status, JVPointStatus)
                and status.valid
                and status.upstream_valid
                and status.branch == branch
                and status.index == index
                and np.isfinite(status.voltage)
                and float(status.voltage) == float(V_arr[index])
                for index, status in enumerate(statuses)
            )

        return _branch_is_complete(
            self.V_fwd, self.J_fwd, self.status_fwd, "jv_forward",
        ) and _branch_is_complete(
            self.V_rev, self.J_rev, self.status_rev, "jv_reverse",
        )


# --- V_oc extraction: terminal-current resolution floor --------------------
#
# ``_J_ZERO_FRACTION_OF_JSC`` is the fraction of |J_sc| below which the
# terminal current is treated as having REACHED zero, instead of waiting for
# a sign change.  Waiting for the sign change is what let a numerical-noise
# flip volts past the true open-circuit point be reported as V_oc on devices
# whose photocurrent collapses before the diode turns on (see the module
# docstring of ``tests/unit/experiments/test_voc_collapsed_current.py``).
#
# Provenance — metrology and diode physics, NOT any observed residual:
#   * Lower bound, from what a real J-V trace can resolve.  A 0.1 cm² lab cell
#     at J_sc ≈ 200-250 A/m² carries I_sc ≈ 2 mA.  A Keithley 2400/2450-class
#     SMU on its 10 mA range has an accuracy floor of ~0.5 µA + 0.05 % of
#     reading ≈ 1.5 µA ≈ 7e-4 of I_sc, and a class-AAA solar simulator's
#     temporal irradiance instability is ±0.5-1 %.  A current under ~1e-3 of
#     J_sc is therefore not a current any real measurement can distinguish
#     from zero, so a simulator must not either.
#   * Upper bound, from what the truncation costs at a GENUINE V_oc.  There
#     the curve is diode-limited, |dJ/dV| ≈ J_sc / (m·V_T), so stopping the
#     search at |J| = ε·J_sc displaces V_oc by at most ε·m·V_T — 52 µV for
#     ε=1e-3, m=2, V_T=25.85 mV.  Keeping that displacement an order of
#     magnitude under the 1 mV at which V_oc is quoted requires
#     ε ≪ 1e-3/(10·m·V_T) ≈ 2e-3.
# 1e-3 sits at the metrology floor and just under the cost bound.
#
# The cost bound holds only where the diode slope does.  On a COLLAPSED
# curve there is no such slope — the device this guard exists for drops
# 148.8 → 0.127 A/m² across one 37 mV sample (measured, scaps_mirror +
# Robin contacts, N_D_ETL = 1e18 m^-3) — so V_oc is then resolved only to
# the sample spacing and is reported at the bracket edge.  That is the same
# resolution the sign-change rule has; the difference is that this bracket
# is anchored at the collapse instead of at a noise flip 1 V further on.
_J_ZERO_FRACTION_OF_JSC = 1e-3


def thermodynamic_voc_ceiling(stack: DeviceStack) -> float | None:
    """Hard upper bound on any physical V_oc for ``stack``, in volts.

    The open-circuit voltage of a single-junction cell cannot exceed the
    smallest band gap the carriers must traverse: q·V_oc is the splitting of
    the quasi-Fermi levels, and that splitting is bounded by the band gap
    (the radiative / detailed-balance limit is well below it).  Returns
    ``min(Eg)/q`` over the ELECTRICAL layers — optical-only substrates are
    excluded, since they carry no carriers.

    Returns ``None`` when no electrical layer declares a band gap (the legacy
    ``chi = Eg = 0`` presets such as ``nip_MAPbI3``/``pin_MAPbI3``).  There is
    no information from which to build a ceiling in that case, so the caller
    must not apply one — a ``min()`` over all-zero gaps would reject every
    V_oc.

    Why ``min`` over the electrical layers and not the absorber's gap.  The
    bound that is actually a theorem is ``V_oc ≤ Eg_absorber/q`` (the
    splitting is generated in the absorber), and ``min`` is only equal to it
    while no transport layer is narrower than the absorber.  On every shipped
    preset it is — measured 2026-07-27 over all of ``configs/*.yaml`` +
    ``configs/twod/*.yaml``: the ``role: absorber`` layer has the narrowest
    electrical gap in every config that declares gaps at all (the closest
    call is ``cSi_homojunction``, where the n-emitter ties it at 1.12 eV).
    ``min`` is preferred because it needs no ``role`` tag, so a config that
    omits or mistags roles still gets a ceiling.  A future stack with a
    genuinely narrow-gap transport layer would make this ceiling TIGHTER
    than the thermodynamic bound and could refuse a valid V_oc — the
    equality is therefore pinned by
    ``tests/unit/experiments/test_voc_collapsed_current.py::
    test_ceiling_equals_the_absorber_gap_on_every_shipped_preset``, which is
    the signal to revisit this choice rather than to relax the test.

    Known plumbing gap (2026-07-27): only :func:`run_jv_sweep` passes this
    ceiling into :func:`compute_metrics`.  The other call sites that hold a
    ``DeviceStack`` — ``degradation._measure_snapshot_metrics``,
    ``steady_state.run_jv_sweep_ss``, ``voc_t``,
    ``twod.experiments.jv_sweep_2d`` and ``twod.experiments.voc_grain_sweep``
    — still call it with the default ``V_oc_max=None``, so they keep the
    resolution floor but not the ceiling.
    """
    gaps = [
        float(layer.params.Eg)
        for layer in electrical_layers(stack)
        if float(layer.params.Eg) > 0.0
    ]
    return min(gaps) if gaps else None


def compute_metrics(
    V: np.ndarray,
    J: np.ndarray,
    *,
    assume_jsc_positive: bool = True,
    P_in: float = 1000.0,
    V_oc_max: float | None = None,
    validity: Sequence[bool] | Sequence[JVPointStatus] | np.ndarray | None = None,
    mpp_interpolation: Literal["sampled", "local_quadratic"] = "sampled",
) -> JVMetrics:
    """Compute V_oc, J_sc, FF, PCE from a J-V array (J in A/m²).

    By default, reports metrics directly from the simulated J(V) samples and
    does not smooth. With ``mpp_interpolation='local_quadratic'``, only the
    three certified power samples surrounding the discrete maximum are fitted;
    a concave vertex inside that local voltage bracket may replace the sampled
    maximum. Boundary maxima, non-concave fits, and out-of-bracket vertices
    fall back to the sampled value. The caller remains responsible for voltage
    refinement and convergence checks.

    Sign convention. The 1D solver follows the IonMonger / DriftFusion
    convention where J(V=0) > 0 (the photocurrent flows out of the device
    and powers an external load). The 2D solver currently emits the
    opposite sign — J(V=0) < 0 — so 2D callers must pass
    ``assume_jsc_positive=False`` to flip J internally before extraction.
    The returned :class:`JVMetrics` is always reported in the
    "J_sc positive" convention, regardless of which sign the input used.

    V_oc extraction
    ---------------
    Open circuit is the first voltage at which the terminal current REACHES
    zero — not the first voltage at which it changes sign.  The two coincide
    on a well-resolved curve, but they diverge badly on a device whose
    photocurrent collapses before the diode turns on: J then sits on a
    residual plateau whose SIGN is numerical noise, and the first sign flip
    can land volts past the true open-circuit point.  Past flat band the
    J-V curve is in the near-singular-Jacobian region the solver notes
    already document, so its sign there carries no information.  Measured
    on scaps_mirror + Robin contacts at N_D_ETL = 1e18 m^-3 (Eg = 1.53 eV,
    J_sc = 219 A/m²), forward branch at dV = 75 mV: J falls to 2.1e-2 A/m²
    at V = 1.425 and then plateaus at ~1.5e-3 (7e-6 of J_sc) all the way to
    2.700, with an isolated negative excursion at 2.475 — which the
    sign-change rule reports as V_oc = 2.4056 V, above the band gap.

    The bracket is therefore the first sample pair with
    ``J[i] > J_tol`` and ``J[i+1] <= J_tol``, where
    ``J_tol = 1e-3 · |J_sc|`` is the current-resolution floor documented on
    ``_J_ZERO_FRACTION_OF_JSC`` above.  V_oc is linearly interpolated to
    J = 0 inside that pair and clamped to it.  When ``J_sc = 0`` (dark
    sweeps) ``J_tol`` is zero and this reduces exactly to the sign-change
    rule.

    Cost on healthy curves: none measured.  On a curve with a genuine
    crossing no sample lands in ``(0, J_tol]`` before it, so the bracket,
    the interpolation and the clamp all reproduce the sign-change rule
    exactly.  Verified 2026-07-27 by extracting metrics TWICE from one
    sweep per config (N_grid=60, n_points=40, v_rate=0.5) — once with
    ``_J_ZERO_FRACTION_OF_JSC`` monkeypatched to 0.0 (which is the
    sign-change rule identically) and once as shipped: V_oc, J_sc, FF and
    PCE compared exactly equal as floats on ionmonger_benchmark,
    nip_MAPbI3, pin_MAPbI3, driftfusion_benchmark and scaps_mirror_v2,
    forward and reverse (10/10 branches).

    Cost on collapsed curves: V_oc is resolved only to the sample spacing,
    because the interpolation inside the collapse bracket is meaningless and
    the clamp returns its right edge.  On the fixture above at dV = 37 mV
    this reports 1.3756 V where the (noise-signed) crossing sits at 1.4373 V
    — 62 mV lower, and the current across that gap is ≤ 5.8e-4 of J_sc
    (1.3 µA on a 0.1 cm² cell), i.e. inside the measurement floor the
    threshold is derived from.  Both numbers are honest; neither is
    resolvable.  What the guard buys is that the answer stops depending on
    where the sweep happened to stop.

    ``V_oc_max`` is an optional hard ceiling in volts.  A V_oc above it is
    impossible regardless of the numerics, so it is refused rather than
    reported: the result carries ``voc_bracketed=False`` and sentinel zeros.
    ``jv_sweep`` callers get it from :func:`thermodynamic_voc_ceiling`
    (``min(Eg)/q`` over the electrical layers).  ``None`` (the default)
    disables the check, which is what a stack with no declared band gaps
    must do.

    validity may be either a boolean mask or the per-point JVPointStatus
    sequence returned by run_jv_sweep. Metrics are undefined for a partially
    certified curve, so any false status raises ValueError instead of dropping,
    interpolating across, or otherwise hiding that point. Non-finite V/J
    samples are always rejected.

    Bracketing flag
    ---------------
    ``voc_bracketed`` means exactly one thing: **an open-circuit point was
    resolved inside the sampled window, and V_oc / FF / PCE below are
    physical.**  It is ``False``, with sentinel zeros for V_oc / FF / PCE,
    when the current never reached zero in the window (V_max stopped short)
    and when the extracted V_oc was refused by ``V_oc_max``.  It is ``True``
    for a collapsed-current device: a cell delivering less than the
    measurement resolution HAS reached open circuit, and reporting the
    voltage at which it got there is the honest answer — the alternative,
    calling it "unbracketed" and inviting the caller to widen V_max, walks
    further into the region whose sign is noise.  ``J_sc`` is interpolated
    at V=0 and stays meaningful in every case, including the two ``False``
    ones.  Callers should surface ``False`` as an "increase V_max" warning
    rather than reading 0 V as a physical V_oc.
    """
    V = np.asarray(V, dtype=float)
    J = np.asarray(J, dtype=float)
    if mpp_interpolation not in ("sampled", "local_quadratic"):
        raise ValueError(
            "mpp_interpolation must be 'sampled' or 'local_quadratic'"
        )
    if V.ndim != 1 or J.ndim != 1 or V.shape != J.shape:
        raise ValueError(
            f"V and J must be one-dimensional arrays of equal shape, got "
            f"{V.shape} and {J.shape}"
        )
    if validity is not None:
        validity_items = tuple(validity)
        if len(validity_items) != len(V):
            raise ValueError(
                f"validity length {len(validity_items)} != J-V length {len(V)}"
            )
        status_items = tuple(
            isinstance(item, JVPointStatus) for item in validity_items
        )
        if any(status_items):
            if not all(status_items):
                raise TypeError("validity cannot mix JVPointStatus and boolean values")
            valid_mask = np.array(
                [item.valid for item in validity_items], dtype=bool,
            )
        else:
            if not all(isinstance(item, (bool, np.bool_)) for item in validity_items):
                raise TypeError("validity values must be boolean or JVPointStatus")
            valid_mask = np.asarray(validity_items, dtype=bool)
        if valid_mask.shape != V.shape:
            raise ValueError(
                f"validity must have shape {V.shape}, got {valid_mask.shape}"
            )
        invalid = np.flatnonzero(~valid_mask)
        if invalid.size:
            raise ValueError(
                "J-V metrics require every point to be certified; invalid "
                f"indices: {invalid.tolist()}"
            )
    finite = np.isfinite(V) & np.isfinite(J)
    if not np.all(finite):
        invalid = np.flatnonzero(~finite)
        raise ValueError(
            "J-V metrics require finite voltage and current samples; "
            f"non-finite indices: {invalid.tolist()}"
        )
    if not assume_jsc_positive:
        J = -J
    order = np.argsort(V)
    V_s = V[order]
    J_s = J[order]

    J_sc = float(np.interp(0.0, V_s, J_s))
    # Current-resolution floor. Anchored on |J_sc| so it scales with the
    # device; zero for a dark sweep, where it degenerates to the sign rule.
    J_tol = _J_ZERO_FRACTION_OF_JSC * abs(J_sc)
    crossings = np.where((J_s[:-1] > J_tol) & (J_s[1:] <= J_tol))[0]
    if len(crossings) == 0:
        return JVMetrics(
            V_oc=0.0, J_sc=J_sc, FF=0.0, PCE=0.0, voc_bracketed=False,
        )
    idx = int(crossings[0])
    dV = V_s[idx + 1] - V_s[idx]
    dJ = J_s[idx + 1] - J_s[idx]
    V_oc = float(V_s[idx] - J_s[idx] * dV / dJ) if dJ != 0.0 else float(V_s[idx])
    # When J[idx+1] is a small POSITIVE residual rather than a sign flip the
    # interpolation extrapolates past the bracket, and the overshoot is
    # unbounded as J[idx] approaches J_tol from above. Clamp to the bracket.
    # No-op whenever J[idx+1] <= 0, i.e. for every genuine sign crossing.
    V_oc = min(max(V_oc, float(V_s[idx])), float(V_s[idx + 1]))

    if V_oc_max is not None and V_oc > V_oc_max:
        # Above the thermodynamic ceiling: impossible whatever the numerics
        # produced, so refuse it rather than report it.
        return JVMetrics(
            V_oc=0.0, J_sc=J_sc, FF=0.0, PCE=0.0, voc_bracketed=False,
        )

    mask = (V_s >= 0.0) & (V_s <= V_oc)
    operating_voltage = V_s[mask]
    operating_power = operating_voltage * J_s[mask]
    if operating_power.size:
        maximum_index = int(np.argmax(operating_power))
        P_mpp = float(operating_power[maximum_index])
        if (
            mpp_interpolation == "local_quadratic"
            and 0 < maximum_index < operating_power.size - 1
        ):
            local_voltage = operating_voltage[
                maximum_index - 1 : maximum_index + 2
            ]
            local_power = operating_power[
                maximum_index - 1 : maximum_index + 2
            ]
            centered_voltage = local_voltage - local_voltage[1]
            coefficients = np.polynomial.polynomial.polyfit(
                centered_voltage,
                local_power,
                2,
            )
            curvature = float(coefficients[2])
            if np.isfinite(curvature) and curvature < 0.0:
                vertex = -float(coefficients[1]) / (2.0 * curvature)
                if centered_voltage[0] <= vertex <= centered_voltage[-1]:
                    vertex_power = float(
                        np.polynomial.polynomial.polyval(vertex, coefficients)
                    )
                    if np.isfinite(vertex_power) and vertex_power >= P_mpp:
                        P_mpp = vertex_power
    else:
        P_mpp = 0.0
    FF = P_mpp / (V_oc * J_sc) if (V_oc * J_sc) > 0 else 0.0
    # PCE is defined against the incident optical power density P_in
    # [W/m^2]. The default 1000 W/m^2 is the AM1.5G 1-sun convention;
    # callers sweeping illumination intensity or spectra must pass the
    # actual P_in or the reported efficiency is wrong (2026-07 review F18).
    PCE = P_mpp / P_in if P_in > 0.0 else 0.0
    return JVMetrics(
        V_oc=V_oc, J_sc=J_sc, FF=FF, PCE=PCE, voc_bracketed=True,
    )


def hysteresis_index(
    V_fwd: np.ndarray, J_fwd: np.ndarray,
    V_rev: np.ndarray, J_rev: np.ndarray,
    *,
    V_oc_max: float | None = None,
    validity_fwd: Sequence[bool] | Sequence[JVPointStatus] | np.ndarray | None = None,
    validity_rev: Sequence[bool] | Sequence[JVPointStatus] | np.ndarray | None = None,
) -> float:
    """Forward/reverse PCE asymmetry. ``V_oc_max`` is forwarded to
    :func:`compute_metrics` on both branches (see its docstring); ``None``
    keeps the pre-existing no-ceiling behaviour."""
    m_fwd = compute_metrics(
        V_fwd, J_fwd, V_oc_max=V_oc_max, validity=validity_fwd,
    )
    m_rev = compute_metrics(
        V_rev, J_rev, V_oc_max=V_oc_max, validity=validity_rev,
    )
    if m_rev.PCE == 0:
        return 0.0
    return (m_rev.PCE - m_fwd.PCE) / m_rev.PCE


def _unavailable_metrics() -> JVMetrics:
    """Return an unmistakable sentinel for an uncertified J-V branch."""
    return JVMetrics(
        V_oc=np.nan,
        J_sc=np.nan,
        FF=np.nan,
        PCE=np.nan,
        voc_bracketed=False,
    )


def _state_fields(
    x: np.ndarray,
    y_state: np.ndarray,
    stack: DeviceStack,
    V_bc: float,
    mat: MaterialArrays,
    *,
    phi_frozen: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StateVec]:
    """Unpack state vector, apply BCs, solve Poisson. Returns (n, p, phi, sv).

    With selective / Schottky contacts active (Phase 3.3) the Robin sides
    are left free — boundary densities come straight from the state vector
    — while the Dirichlet sides are still pinned so that post-processing
    matches what the solver saw.
    """
    N = len(x)
    sv = StateVec.unpack(y_state, N, N_iface_state=mat.N_iface_state)
    n = sv.n.copy()
    p = sv.p.copy()
    if not mat.has_selective_contacts:
        n[0] = mat.n_L
        n[-1] = mat.n_R
        p[0] = mat.p_L
        p[-1] = mat.p_R
    else:
        if mat.S_n_L is None:
            n[0] = mat.n_L
        if mat.S_n_R is None:
            n[-1] = mat.n_R
        if mat.S_p_L is None:
            p[0] = mat.p_L
        if mat.S_p_R is None:
            p[-1] = mat.p_R
    if phi_frozen is None:
        rho = _charge_density(
            p, n, sv.P, mat.P_ion0, mat.N_A, mat.N_D,
            P_neg=sv.P_neg, P_neg0=mat.P_ion0_neg,
        )
        phi = solve_poisson_prefactored(
            mat.poisson_factor, rho, phi_left=0.0,
            phi_right=poisson_right_boundary(mat, V_bc),
        )
    else:
        phi = np.asarray(phi_frozen, dtype=float)
        if phi.shape != x.shape or not np.all(np.isfinite(phi)):
            raise ValueError("phi_frozen must be finite and match the spatial grid")
    return n, p, phi, sv


def extract_spatial_snapshot(
    x: np.ndarray,
    y: np.ndarray,
    stack: DeviceStack,
    V_app: float,
    mat: MaterialArrays | None = None,
    *,
    phi_frozen: np.ndarray | None = None,
) -> SpatialSnapshot:
    """Extract spatial profiles from a state vector at a given voltage.

    Returns a SpatialSnapshot with all node/face quantities in SI units.
    """
    if mat is None:
        mat = build_material_arrays(x, stack)
    n, p, phi, sv = _state_fields(
        x,
        y,
        stack,
        V_app,
        mat,
        phi_frozen=phi_frozen,
    )
    dx = np.diff(x)
    E = -(phi[1:] - phi[:-1]) / dx
    rho = _charge_density(
        p, n, sv.P, mat.P_ion0, mat.N_A, mat.N_D,
        P_neg=sv.P_neg, P_neg0=mat.P_ion0_neg,
    )
    return SpatialSnapshot(
        x=x.copy(), phi=phi, E=E, n=n, p=p, P=sv.P.copy(), rho=rho, V_app=V_app,
    )


def compute_current_components(
    x: np.ndarray,
    y: np.ndarray,
    stack: DeviceStack,
    V_app: float,
    y_prev: np.ndarray | None = None,
    dt: float | None = None,
    mat: MaterialArrays | None = None,
    V_app_prev: float | None = None,
    phi_frozen: np.ndarray | None = None,
) -> CurrentComponents:
    """Decompose the total current into electron, hole, ion, and displacement.

    All arrays have shape (N-1,). Sign convention: positive when the device
    delivers power (solar convention, consistent with _compute_current).
    """
    if mat is None:
        mat = build_material_arrays(x, stack)

    dx = np.diff(x)
    n, p, phi, sv = _state_fields(
        x,
        y,
        stack,
        V_app,
        mat,
        phi_frozen=phi_frozen,
    )
    V_T_dev = mat.V_T_device

    # Electron and hole conduction currents (SG fluxes). When the device
    # enables field-dependent mobility, the terminal J must use the same
    # field-corrected face diffusivities that assemble_rhs saw — otherwise
    # the reported current would be inconsistent with the state the solver
    # integrated, which would break charge conservation at the contact.
    phi_n = phi + mat.chi
    phi_p = phi + mat.chi + mat.Eg
    xi_n = (phi_n[1:] - phi_n[:-1]) / V_T_dev
    xi_p = (phi_p[1:] - phi_p[:-1]) / V_T_dev
    B_pos_n = bernoulli(xi_n)
    B_neg_n = bernoulli(-xi_n)
    B_pos_p = bernoulli(xi_p)
    B_neg_p = bernoulli(-xi_p)
    if mat.has_field_mobility:
        from perovskite_sim.physics.field_mobility import apply_field_mobility
        E_face = -(phi[1:] - phi[:-1]) / dx
        mu_n_face_base = mat.D_n_face / V_T_dev
        mu_p_face_base = mat.D_p_face / V_T_dev
        D_n_face_eff = apply_field_mobility(
            mu_n_face_base, E_face,
            mat.v_sat_n_face, mat.ct_beta_n_face, mat.pf_gamma_n_face,
        ) * V_T_dev
        D_p_face_eff = apply_field_mobility(
            mu_p_face_base, E_face,
            mat.v_sat_p_face, mat.ct_beta_p_face, mat.pf_gamma_p_face,
        ) * V_T_dev
    else:
        D_n_face_eff = mat.D_n_face
        D_p_face_eff = mat.D_p_face
    J_n = Q * D_n_face_eff / dx * (B_pos_n * n[1:] - B_neg_n * n[:-1])
    J_p = Q * D_p_face_eff / dx * (B_pos_p * p[:-1] - B_neg_p * p[1:])

    ionic = _ionic_current_components_from_fields(x, phi, sv, mat)

    # Displacement current
    J_disp = np.zeros_like(J_n)
    if y_prev is not None and dt is not None and dt > 0.0:
        V_prev_bc = V_app_prev if V_app_prev is not None else V_app
        _, _, phi_prev, _ = _state_fields(x, y_prev, stack, V_prev_bc, mat)
        eps_face = _harmonic_face_average(mat.eps_r)
        E_prev = -(phi_prev[1:] - phi_prev[:-1]) / dx
        E_now = -(phi[1:] - phi[:-1]) / dx
        J_disp = EPS_0 * eps_face * (E_now - E_prev) / dt

    polarity = float(mat.junction_polarity)
    J_ion = -polarity * ionic.J_total
    return CurrentComponents(
        J_n=-polarity * J_n,
        J_p=-polarity * J_p,
        J_ion=ionic.J_total,
        J_disp=-polarity * J_disp,
        J_total=-polarity * (J_n + J_p + J_ion + J_disp),
    )


def _ionic_current_components_from_fields(
    x: np.ndarray,
    phi: np.ndarray,
    state: StateVec,
    mat: MaterialArrays,
) -> IonicCurrentComponents:
    """Evaluate each ionic charge-current channel from an unpacked state."""
    dx = np.diff(x)
    D_ion_face = np.broadcast_to(
        np.asarray(mat.D_ion_face, dtype=float), dx.shape,
    )
    P_lim_face = np.broadcast_to(
        np.asarray(mat.P_lim_face, dtype=float), dx.shape,
    )
    shared = (
        mat.ion_steric_diffusion_only
        and mat.ion_steric_shared_site
        and mat.has_dual_ions
        and state.P_neg is not None
    )
    positive_flux = ion_face_flux(
        phi,
        state.P,
        dx,
        D_ion_face,
        mat.V_T_device,
        P_lim_face,
        steric_diffusion_only=mat.ion_steric_diffusion_only,
        P_lim_node=mat.P_lim_node,
        P_other_node=(state.P_neg if shared else None),
        drift_sign=+1.0,
    )
    positive = Q * positive_flux
    negative = None
    if mat.has_dual_ions and state.P_neg is not None:
        negative_flux = ion_face_flux(
            phi,
            state.P_neg,
            dx,
            np.broadcast_to(np.asarray(mat.D_ion_neg_face, dtype=float), dx.shape),
            mat.V_T_device,
            np.broadcast_to(np.asarray(mat.P_lim_neg_face, dtype=float), dx.shape),
            steric_diffusion_only=mat.ion_steric_diffusion_only,
            P_lim_node=mat.P_lim_neg_node,
            P_other_node=(state.P if shared else None),
            drift_sign=-1.0,
        )
        negative = -Q * negative_flux

    polarity = float(mat.junction_polarity)
    positive_solar = -polarity * positive
    negative_solar = None if negative is None else -polarity * negative
    total = positive if negative is None else positive + negative
    return IonicCurrentComponents(
        J_positive=positive_solar,
        J_negative=negative_solar,
        J_total=-polarity * total,
    )


def compute_ionic_current_components(
    x: np.ndarray,
    y: np.ndarray,
    stack: DeviceStack,
    V_app: float,
    *,
    mat: MaterialArrays | None = None,
    phi_frozen: np.ndarray | None = None,
) -> IonicCurrentComponents:
    """Return positive, negative, and net ionic current at every face.

    This uses the same :func:`ion_face_flux` implementation as the transient
    RHS, including the selected steric model.  The per-species channels are
    needed for DC certification because their net electrical current can
    cancel in a dual-ion model even when neither species is stationary.
    """
    if mat is None:
        mat = build_material_arrays(x, stack)
    _, _, phi, state = _state_fields(
        x,
        y,
        stack,
        V_app,
        mat,
        phi_frozen=phi_frozen,
    )
    return _ionic_current_components_from_fields(x, phi, state, mat)


def _total_current_faces(
    x: np.ndarray,
    y: np.ndarray,
    stack: DeviceStack,
    V_app: float,
    y_prev: np.ndarray | None = None,
    dt: float | None = None,
    mat: MaterialArrays | None = None,
    V_app_prev: float | None = None,
) -> np.ndarray:
    """Return total current density at every face (N-1,), solar sign convention.

    Thin wrapper around compute_current_components for backward compatibility.
    """
    return compute_current_components(
        x, y, stack, V_app, y_prev=y_prev, dt=dt, mat=mat, V_app_prev=V_app_prev,
    ).J_total


def _compute_current(
    x: np.ndarray,
    y: np.ndarray,
    stack: DeviceStack,
    V_app: float,
    y_prev: np.ndarray | None = None,
    dt: float | None = None,
    mat: MaterialArrays | None = None,
    V_app_prev: float | None = None,
) -> float:
    """Extract terminal current density J [A/m²] at the contact-adjacent face.

    The terminal current combines carrier conduction and, when a previous state
    is available, the displacement current over the last time step. When the
    applied voltage changed between `y_prev` and `y`, pass `V_app_prev` so the
    Poisson solve for `y_prev` uses the right Dirichlet condition; otherwise
    the ∂V_boundary/∂t contribution to the displacement current is lost.

    Convention: J > 0 when the device delivers power (J_sc > 0 at V=0).
    """
    J_faces = _total_current_faces(
        x, y, stack, V_app, y_prev=y_prev, dt=dt, mat=mat, V_app_prev=V_app_prev,
    )
    return float(J_faces[0])


def _compute_current_ss(
    x: np.ndarray,
    y: np.ndarray,
    stack: DeviceStack,
    V_app: float,
    mat: MaterialArrays | None = None,
) -> float:
    """Extract J [A/m²] using the **median** across all interior faces.

    Use this when the caller has settled the device with a finite ``t_settle``
    and is reading J at fixed V — Suns-Voc, EQE, and similar V=0 SS probes.
    At true steady state every face carries the same total current (charge
    conservation), so any face is correct. In practice on ionic-rich presets
    the contact-adjacent faces still see residual ionic / displacement
    transients that haven't fully damped, while the interior faces agree on
    the photo-current to many digits. Median is the robust mid-summary that
    is insensitive to those boundary outliers — verified on
    ``ionmonger_benchmark_tmm`` where ``J[0]`` swings ±1700 A/m² across
    t_settle but median stays pinned at the linear-with-suns photo value.

    Use ``_compute_current`` (face[0]) on transient J-V sweeps where the
    displacement current at the contact is the physical terminal-current
    quantity flowing through the external circuit.

    Convention: J > 0 when the device delivers power (J_sc > 0 at V=0).
    """
    return _compute_current_ss_with_spread(x, y, stack, V_app, mat=mat)[0]


def _compute_current_ss_with_spread(
    x: np.ndarray,
    y: np.ndarray,
    stack: DeviceStack,
    V_app: float,
    mat: MaterialArrays | None = None,
) -> tuple[float, float]:
    """Return ``(J_median, dJ_spread)`` — the SS current and its certificate.

    Charge conservation makes ``J(x)`` uniform at true steady state, so the
    face-to-face disagreement is a direct measure of how far the state is
    from that limit. Taking the median (see ``_compute_current_ss``) makes
    the reported number robust to that disagreement, but robustness is not
    convergence: without the spread alongside it, a median can look clean
    while the underlying state still carries a continuity residual. This
    helper returns both so callers can gate on the certificate rather than
    trusting the summary (2026-07 review finding F-13).

    ``dJ_spread`` is ``max - min`` over the **interior** faces only. The two
    contact-adjacent faces are excluded on purpose: there the displacement
    term is part of the physical external-circuit current and, on ionic-rich
    presets, legitimately swings by ~1e3 A/m2 while the interior faces agree
    to many digits — including them would report a physical boundary
    transient as a conservation error. Grids with fewer than three faces
    fall back to the full set.

    IMPORTANT — report it, do NOT gate on it. The review that requested
    this certificate also proposed refusing to extract V_oc / J_sc / FF
    until the spread falls below a fixed tolerance. Measured on the TMM
    presets at V=0, that gate is not defensible, because the spread is
    neither monotone in settling time nor stable under mesh refinement
    while the reported median is both:

        preset            N/layer  t_settle   median J   interior spread
        nip_MAPbI3_tmm      20      1e-3       213.02      7.99e+01
        nip_MAPbI3_tmm      20      1e-1       213.02      1.47e+01
        ionmonger_bm_tmm    20      1e-3       212.99      1.63e+02
        ionmonger_bm_tmm    20      1e-1       213.00      7.06e+03   (!)
        ionmonger_bm_tmm    12      1e-1       218.65      8.80e+01

    The median moves by <0.02 % across settling times while the spread
    grows 43x with a *longer* settle and swings ~80x with the mesh, so a
    threshold would fire on converged runs and its natural remedy
    ("settle longer") is backwards. Normalising by the largest current
    component does not rescue it either (0.33 at N=12 vs ~26 at N=20).
    The cause is the documented terminal-current flux cancellation: J is
    a sum of large, nearly-cancelling electron, hole, ionic, and
    displacement contributions, so a small relative error in the parts
    becomes a large absolute error in their sum — which is exactly the
    condition under which the median is the right summary and the raw
    spread is a poor convergence proxy. Treat ``dJ_spread`` as a
    magnitude diagnostic to report, and use mesh convergence of the
    extracted metrics for the convergence question it cannot answer.
    """
    J_faces = _total_current_faces(x, y, stack, V_app, mat=mat)
    J = float(np.median(J_faces))
    interior = J_faces[1:-1] if J_faces.size > 2 else J_faces
    return J, float(np.max(interior) - np.min(interior))


# Upper bound on RHS evaluations per run_transient call inside a JV sweep.
# Calibration: well-conditioned sub-intervals at N_grid=60 complete in a few
# hundred nfev; degenerate calls (reverse sweep reheating from a high-V_app
# carrier-injection state on ionmonger_benchmark) spin indefinitely without
# a bound. Well-conditioned TMM presets at moderate forward bias can legit
# consume ~30-40k nfev on a single sub-interval when the corrected generation
# profile drives high carrier densities, so the cap must stay well above
# that or the bisection fallback wastes budget on already-good calls.
# 100k aborts long before a wall-time user would notice (~10-15 s) and
# leaves headroom for the bisection fallback to retry on halved intervals.
_JV_RADAU_MAX_NFEV = 100_000

# Wrong-branch detector for the forward power quadrant (see
# _integrate_step's docstring for why step control alone cannot fix this).
#
# For run_jv_sweep's built-in short-circuit initialization, the expected
# forward power-quadrant response is J(V) = J_sc - J_dark(V), with J_dark >= 0.
# This is a protocol guard, not a universal theorem for arbitrary precharged
# transient states (this API does not accept one): stored ionic/capacitive
# energy can add terminal current. A step far above the reference on the
# controlled protocol has landed on the carrier-injection branch.
#
# The margin absorbs the one legitimate way J(V) can exceed J(0) on this
# code path — a fast scan over a mobile-ion stack, where the ionic
# configuration lags the bias and the early forward points carry a small
# transient excess. Measured across the shipped presets, the largest
# genuine excess is ~1e-4 of J_sc, four orders below this bound, while the
# defect it must catch is 2.3x (509.8 vs 221.9) and the smallest wrong
# branch seen while sweeping step control is 1.15x (255.9). 5 % therefore
# sits ~500x above the noise and ~3x below the smallest real defect.
_J_BRANCH_EXCESS = 0.05

# Leg counts tried, in order, when the bound above rejects a step. Every
# one of these recovered the identical physical value at the measured
# failure (2/4/8 legs and BDF all give 161.986 vs the single-leg 509.796),
# so the ladder is about robustness on unmeasured configs, not tuning.
_J_BRANCH_RETRY_LEGS = (2, 4, 8)

# A rejected single-leg step is not recovered merely because one retry falls
# below the current ceiling: the same wrong-branch defect can undershoot as
# well as overshoot. Require two successful subdivisions to agree before a
# retry is allowed to replace the original state. An intermediate numeric
# failure does not erase the preceding successful result when a finer retry
# later completes. One percent is deliberately
# much wider than the measured 4-vs-8-leg agreement (<= 0.2 %) and narrower
# than the smallest measured wrong landing (3.3 % at V=0).
_J_BRANCH_REFINEMENT_RTOL = 0.01

# Absolute current resolution paired with the relative branch ceiling above.
# A purely relative test degenerates when an optical override has J_sc near
# zero: round-off-scale positive currents then exceed 1.05*J_sc despite being
# numerically indistinguishable from zero. The same 1 A/m^2 scale floor is
# used by _currents_agree, so 1 % of that scale gives 0.01 A/m^2 = 1 uA/cm^2.
# This remains more than three orders below the smallest measured wrong-branch
# excess on the calibrated one-sun gate.
_J_BRANCH_CURRENT_ATOL = 0.01

# V=0 has no universal lower current bound, so an under-read cannot be found
# by a one-sided protocol ceiling.  Probe the same interval with four legs;
# only when that materially disagrees with the original do we pay for an
# eight-leg confirmation and replace the state.
_J_ZERO_BIAS_PROBE_LEGS = 4
_J_ZERO_BIAS_CONFIRM_LEGS = 8

# Numeric failures that can arise from scipy integration or terminal-current
# reconstruction. Programming errors remain loud instead of being mislabeled
# as solver failures.
_JV_NUMERIC_FAILURES = (
    RuntimeError,
    ValueError,
    FloatingPointError,
    OverflowError,
    RuntimeWarning,
    np.linalg.LinAlgError,
)


def _branch_current_ceiling(J_sc: float) -> float:
    """Return the forward-branch trigger with relative and absolute margin."""
    J_ref = float(J_sc)
    margin = max(
        _J_BRANCH_EXCESS * abs(J_ref),
        _J_BRANCH_CURRENT_ATOL,
    )
    return J_ref + margin


def _currents_agree(J_a: float, J_b: float) -> bool:
    """Return whether two independently integrated currents are consistent."""
    if not (np.isfinite(J_a) and np.isfinite(J_b)):
        return False
    scale = max(abs(float(J_a)), abs(float(J_b)), 1.0)
    return abs(float(J_a) - float(J_b)) <= _J_BRANCH_REFINEMENT_RTOL * scale


def _states_agree(
    y_a: np.ndarray,
    y_b: np.ndarray,
    N: int,
    N_iface_state: int = 0,
) -> bool:
    """Compare each physical state block on its own density scale."""
    if y_a.shape != y_b.shape:
        return False
    a = StateVec.unpack(y_a, N, N_iface_state)
    b = StateVec.unpack(y_b, N, N_iface_state)
    for block_a, block_b in (
        (a.n, b.n),
        (a.p, b.p),
        (a.P, b.P),
        (a.P_neg, b.P_neg),
        (a.iface_state, b.iface_state),
    ):
        if block_a is None or block_b is None:
            if block_a is not None or block_b is not None:
                return False
            continue
        if not (np.all(np.isfinite(block_a)) and np.all(np.isfinite(block_b))):
            return False
        scale = max(
            float(np.max(np.abs(block_a), initial=0.0)),
            float(np.max(np.abs(block_b), initial=0.0)),
            1.0,
        )
        gap = float(np.max(np.abs(block_a - block_b), initial=0.0))
        if gap > _J_BRANCH_REFINEMENT_RTOL * scale:
            return False
    return True


def _refinements_agree(
    y_a: np.ndarray,
    J_a: float,
    y_b: np.ndarray,
    J_b: float,
    N: int,
    N_iface_state: int = 0,
) -> bool:
    """Require both the reported current and the state that drives later steps."""
    return (
        _currents_agree(J_a, J_b)
        and _states_agree(y_a, y_b, N, N_iface_state)
    )


def _bake_radiative_reabsorption_step(
    y: np.ndarray, x: np.ndarray, mat: MaterialArrays, illuminated: bool,
) -> MaterialArrays:
    """Freeze the Phase 3.1b G_rad source for one ``_integrate_step`` call.

    Phase 3.1b's per-RHS hook recomputes ``R_tot = ∫ B·n·p dx`` inside every
    Radau Newton iteration, which couples every absorber node to every other
    through a non-local integral. At low forward bias (V ≈ 0.21 V on TMM
    presets) the diode-injection knee makes ``d(n·p)/dV`` steep and the
    Newton iteration cannot contract on the resulting low-rank dense block —
    bisection-and-retry runs out and ``_integrate_step`` raises.

    Fix: evaluate ``R_tot`` once at the entry state ``y`` of each voltage
    step, fold ``G_rad`` into a step-local ``G_optical`` copy, and clear
    ``has_radiative_reabsorption`` on the returned ``mat``. Inside the call,
    the SG flux sees a static G and Newton converges. Across voltage steps
    the warm-start chain refreshes ``R_tot`` from the freshly-settled state,
    so the only error is bounded by how much ``n·p`` drifts inside one
    settle interval — sub-percent for the typical ``v_rate=1 V/s`` sweep,
    well below the ~5 mV equivalence window the 3.1b regression tests use
    for V_oc parity with Phase 3.1.

    Beer-Lambert / non-TMM stacks have ``has_radiative_reabsorption=False``
    and skip this path entirely (returned mat is the original).
    """
    if not (mat.has_radiative_reabsorption and mat.absorber_masks):
        return mat
    sv = StateVec.unpack(y, len(x))
    # Under dark forward injection (LED-mode), ``mat.G_optical`` still holds
    # the solar absorption profile that ``build_material_arrays`` cached, but
    # the runtime branch in ``assemble_rhs`` zeros G when ``illuminated=False``.
    # The bake mirrors that decision by zeroing G_base here, so the returned
    # mat carries only the lagged R_rad source. The companion
    # ``force_use_g_optical=True`` flag tells ``assemble_rhs`` to skip the
    # dark zeroing on this baked mat.
    if illuminated:
        G_base = mat.G_optical if mat.G_optical is not None else np.zeros_like(x)
    else:
        G_base = np.zeros_like(x)
    G_with_rad = G_base.copy()
    for mask, P_esc, thickness in zip(
        mat.absorber_masks, mat.absorber_p_esc, mat.absorber_thicknesses
    ):
        if thickness <= 0.0 or P_esc >= 1.0:
            continue
        # NET emission B(np - ni^2) — must match the per-RHS form in
        # assemble_rhs so the baked step preserves detailed balance (F08).
        emission = mat.B_rad[mask] * (sv.n[mask] * sv.p[mask] - mat.ni_sq[mask])
        x_abs = x[mask]
        if x_abs.size < 2:
            continue
        R_tot = float(trapezoid(emission, x_abs))
        if R_tot <= 0.0:
            continue
        G_with_rad[mask] = G_with_rad[mask] + R_tot * (1.0 - P_esc) / thickness
    return dataclasses.replace(
        mat,
        G_optical=G_with_rad,
        has_radiative_reabsorption=False,
        absorber_masks=(),
        absorber_p_esc=(),
        absorber_thicknesses=(),
        force_use_g_optical=True,
    )


def _integrate_step(
    x: np.ndarray,
    y: np.ndarray,
    stack: DeviceStack,
    mat: MaterialArrays,
    V_app: float,
    t_lo: float,
    t_hi: float,
    rtol: float,
    atol: float,
    max_bisect: int = 10,
    illuminated: bool = True,
    n_legs: int = 1,
    regularization: RHSRegularization | None = None,
    _accepted_diagnostics: list[JVAcceptedSolveDiagnostics] | None = None,
) -> np.ndarray:
    """Advance the coupled MOL state from t_lo to t_hi at fixed V_app.

    Radau is adaptive but its error estimator can underreport truncation
    error near V_bi, where the Jacobian becomes nearly singular (flat-band
    region). Without an explicit `max_step` cap it sometimes takes a single
    huge step across the whole [t_lo, t_hi] interval and accepts a state
    that landed on the wrong (carrier-injection) branch of the implicit
    system — producing isolated non-physical spikes in the J-V curve. We
    cap max_step to (t_hi - t_lo)/20 so the solver must resolve the
    transient with at least ~20 internal steps, which suppresses most of
    those spikes without materially slowing well-conditioned regions.

    **The cap is necessary but NOT sufficient, and it cannot be made
    sufficient by tightening it** (measured 2026-07-28 on
    ``ionmonger_benchmark`` N_grid=40/n_points=20, whose V=1.10526 sample
    sits essentially exactly on ``V_bi`` = 1.1). Sweeping the divisor gives
    J = 509.8 / 162.0 / 255.9 / 162.0 / 162.0 at dt/20, /100, /200, /1000,
    /2000 — non-monotone, because a global change to max_step perturbs
    EVERY step's trajectory and merely relocates which one lands on the
    wrong branch. Same lesson as the E9.3 clamp-shape variants; do not
    re-attempt a divisor fix.

    What does work is local subdivision, which is what ``n_legs`` is for.
    The caller uses it either to probe a V=0 step (where a one-sided bound
    cannot detect an under-read) or to recover a positive-bias protocol
    violation. A candidate replaces the original only after two successive
    refinements agree in both terminal current and internal state. Measured
    at the positive-bias failure, 2, 4 and 8 legs and a BDF single leg all
    return J = 161.986 against the single-leg 509.796; at V=0, 4/8 legs close
    both the under-read and over-read counterexamples. A healthy V=0 probe is
    discarded, preserving the original state/current exactly.

    If the implicit solver fails to converge on the full step, subdivide
    the interval (halving up to max_bisect levels) and chain sub-steps.
    Raises RuntimeError if bisection is exhausted.

    Phase 3.1b fallback: when the device has ``has_radiative_reabsorption``
    on, the per-RHS ``G_rad`` source can prevent the Radau Newton iteration
    from contracting on TMM presets at the diode-injection knee
    (V ≈ 0.21 V — see saved memory `project_tmm_jv_regression_021.md`). If
    the standard call fails, retry once with ``G_rad`` frozen at the entry
    state — the warm-started chain refreshes it on the next voltage step,
    so the lag stays sub-percent on the typical ``v_rate=1 V/s`` sweep.
    Steps where the per-RHS hook converges (the vast majority) keep the
    fully self-consistent semantics; only the pathological steps fall back.
    """
    dt = t_hi - t_lo
    accepted_here: list[JVAcceptedSolveDiagnostics] = []

    def _commit(
        state: np.ndarray,
        record=None,
    ) -> np.ndarray:
        if record is not None and _accepted_diagnostics is not None:
            report = getattr(record, "numerical_diagnostics", None)
            if not isinstance(report, NumericalDiagnosticsReport):
                raise RuntimeError(
                    "accepted transient segment omitted numerical diagnostics"
                )
            accepted_here.append(
                JVAcceptedSolveDiagnostics(
                    report=report,
                    nfev=_optional_solver_count(record, "nfev"),
                    njev=_optional_solver_count(record, "njev"),
                    nlu=_optional_solver_count(record, "nlu"),
                )
            )
        if _accepted_diagnostics is not None:
            _accepted_diagnostics.extend(accepted_here)
        return state
    if n_legs > 1 and dt > 0.0:
        # Forced subdivision for a local convergence probe/recovery. Chain
        # `n_legs` equal sub-intervals, each with its own max_step cap.
        edges = np.linspace(t_lo, t_hi, n_legs + 1)
        y_k = y
        for t_a, t_b in zip(edges[:-1], edges[1:]):
            y_k = _integrate_step(
                x, y_k, stack, mat, V_app, float(t_a), float(t_b),
                rtol, atol, max_bisect, illuminated,
                regularization=regularization,
                _accepted_diagnostics=(
                    accepted_here
                    if _accepted_diagnostics is not None
                    else None
                ),
            )
        return _commit(y_k)

    last_failure: str | None = None

    def _solver_attempt(
        mat_attempt: MaterialArrays,
        *,
        method: str = "Radau",
    ):
        """Return a solver result, or None for a recoverable numeric failure."""
        nonlocal last_failure
        try:
            # Overflow/invalid warnings inside scipy's Jacobian construction
            # mean this attempt is numerically unusable even when the global
            # warning policy would otherwise only print them. Normalize them
            # with the explicit solver exceptions so bisection/BDF can recover.
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                solver_kwargs = dict(
                    illuminated=illuminated,
                    V_app=V_app,
                    rtol=rtol,
                    atol=atol,
                    max_step=dt / 20.0 if dt > 0.0 else np.inf,
                    mat=mat_attempt,
                    max_nfev=_JV_RADAU_MAX_NFEV,
                    method=method,
                )
                if regularization is not None:
                    solver_kwargs["regularization"] = regularization
                sol_attempt = run_transient(
                    x,
                    y,
                    (t_lo, t_hi),
                    np.array([t_hi]),
                    stack,
                    **solver_kwargs,
                )
        except _JV_NUMERIC_FAILURES as exc:
            last_failure = f"{type(exc).__name__}: {exc}"
            return None
        if not sol_attempt.success:
            last_failure = str(
                getattr(sol_attempt, "message", "solver returned success=False")
            )
        return sol_attempt

    sol = _solver_attempt(mat)
    if sol is not None and sol.success:
        return _commit(sol.y[:, -1], sol)
    if mat.has_radiative_reabsorption:
        mat_step = _bake_radiative_reabsorption_step(y, x, mat, illuminated)
        sol = _solver_attempt(mat_step)
        if sol is not None and sol.success:
            return _commit(sol.y[:, -1], sol)
    if max_bisect == 0:
        # Last-chance BDF fallback. When Radau bisection is exhausted on
        # a near-flat-band / deep-injection state, scipy's BDF (variable-
        # order BDF) sometimes converges where Radau cannot. Cost is one
        # extra solve attempt per pathological step; healthy steps never
        # reach this branch.
        sol_bdf = _solver_attempt(mat, method="BDF")
        if sol_bdf is not None and sol_bdf.success:
            return _commit(sol_bdf.y[:, -1], sol_bdf)
        detail = f"; last failure: {last_failure}" if last_failure else ""
        raise RuntimeError(
            f"JV sweep: coupled solver failed to converge on [{t_lo:.3e},{t_hi:.3e}] "
            f"at V_app={V_app:.4f} V after bisection{detail}"
        )
    t_mid = 0.5 * (t_lo + t_hi)
    y_mid = _integrate_step(x, y, stack, mat, V_app, t_lo, t_mid, rtol, atol,
                             max_bisect - 1, illuminated,
                             regularization=regularization,
                             _accepted_diagnostics=(
                                 accepted_here
                                 if _accepted_diagnostics is not None
                                 else None
                             ))
    y_final = _integrate_step(
        x, y_mid, stack, mat, V_app, t_mid, t_hi, rtol, atol,
        max_bisect - 1, illuminated,
        regularization=regularization,
        _accepted_diagnostics=(
            accepted_here if _accepted_diagnostics is not None else None
        ),
    )
    return _commit(y_final)


def _optional_solver_count(result, name: str) -> int | None:
    value = getattr(result, name, None)
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        normalized = int(value)
        return normalized if normalized >= 0 else None
    return None


def _sum_accepted_solver_count(
    diagnostics: tuple[JVAcceptedSolveDiagnostics, ...],
    name: str,
) -> int | None:
    values = [
        value
        for item in diagnostics
        for value in (getattr(item, name),)
        if value is not None
    ]
    return sum(values) if values else None


def quasi_static_sweep(
    x: np.ndarray,
    y_init: np.ndarray,
    stack: DeviceStack,
    voltages: np.ndarray,
    sweep_time: float,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    mat: MaterialArrays | None = None,
    regularization: RHSRegularization | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Quasi-static illuminated J-V from an existing state, carrying state forward.

    Integrates the full coupled MOL system at piecewise-constant V_app, stepping
    through `voltages` over a total wall time of `sweep_time` seconds. Each step
    advances the carrier+ion state to quasi-steady for the applied voltage, and
    the terminal current (conduction + displacement) is read from the same
    coupled solve. Used for snapshot J-V measurements inside degradation where
    we want ions effectively frozen (sweep_time ≪ τ_ion) but carriers relaxed.
    """
    n = len(voltages)
    if n < 2:
        raise ValueError(f"voltages must have at least 2 points, got {n}")
    if mat is None:
        mat = build_material_arrays(x, stack)
    dt = sweep_time / (n - 1)
    J_arr = np.zeros(n, dtype=float)
    y = y_init.copy()
    V_prev = float(voltages[0])
    t = 0.0
    for k in range(n):
        V_k = float(voltages[k])
        y_prev = y.copy()
        if regularization is None:
            y = _integrate_step(
                x, y, stack, mat, V_k, t, t + dt, rtol, atol,
            )
        else:
            y = _integrate_step(
                x,
                y,
                stack,
                mat,
                V_k,
                t,
                t + dt,
                rtol,
                atol,
                regularization=regularization,
            )
        J_arr[k] = _compute_current(x, y, stack, V_k, y_prev=y_prev, dt=dt,
                                     mat=mat, V_app_prev=V_prev)
        V_prev = V_k
        t += dt
    return np.asarray(voltages, dtype=float), J_arr


def _custom_grid_parameters(
    stack: DeviceStack,
    elec: Sequence,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Validate a programmatically constructed custom electrical-grid spec."""
    weights = tuple(getattr(stack, "grid_interval_weights", ()) or ())
    alphas = tuple(getattr(stack, "grid_alphas", ()) or ())
    if not weights and not alphas:
        return (), ()
    if not weights or not alphas:
        raise ValueError(
            "custom electrical grid requires both grid_interval_weights and "
            "grid_alphas"
        )
    if len(weights) != len(elec) or len(alphas) != len(elec):
        raise ValueError(
            "custom electrical-grid tuples must contain one value per "
            "electrical layer"
        )
    for label, values in (
        ("grid_interval_weights", weights),
        ("grid_alphas", alphas),
    ):
        try:
            numeric = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{label} values must be finite and positive"
            ) from exc
        if not np.all(np.isfinite(numeric)) or np.any(numeric <= 0.0):
            raise ValueError(f"{label} values must be finite and positive")
    return tuple(float(value) for value in weights), tuple(
        float(value) for value in alphas
    )


def _largest_remainder_interval_counts(
    N_grid: int,
    weights: Sequence[float],
) -> list[int]:
    """Allocate exactly ``N_grid`` intervals with deterministic tie-breaking."""
    total = int(N_grid)
    if total != N_grid:
        raise ValueError(f"N_grid must be an integer, got {N_grid}")
    n_layers = len(weights)
    if total < n_layers:
        raise ValueError(
            f"custom electrical grid needs at least {n_layers} intervals, "
            f"got N_grid={N_grid}"
        )

    weight_array = np.asarray(weights, dtype=float)
    # Normalize first so a set of individually finite, very large weights
    # cannot overflow its sum and silently collapse every quota to zero.
    scaled_weights = weight_array / float(np.max(weight_array))
    quotas = total * scaled_weights / float(np.sum(scaled_weights))
    counts = np.floor(quotas).astype(int)
    shortfall = total - int(np.sum(counts))
    order = sorted(
        range(n_layers),
        key=lambda index: (-(quotas[index] - counts[index]), index),
    )
    for index in order[:shortfall]:
        counts[index] += 1

    # Positive weights can still receive zero intervals when N_grid is small
    # or highly skewed. Repair those slots by taking one interval from the
    # most over-allocated donor; electrical-layer order resolves exact ties.
    for recipient in np.flatnonzero(counts == 0):
        donors = [index for index in range(n_layers) if counts[index] > 1]
        if not donors:  # N_grid >= n_layers makes this unreachable.
            raise RuntimeError("cannot satisfy one-interval-per-layer constraint")
        donor = max(
            donors,
            key=lambda index: (
                counts[index] - quotas[index], counts[index], -index,
            ),
        )
        counts[donor] -= 1
        counts[recipient] = 1

    assert int(np.sum(counts)) == total
    assert np.all(counts >= 1)
    return counts.tolist()


def _layer_node_counts(stack: DeviceStack, N_grid: int) -> list[int]:
    """Per-electrical-layer interval count for the multilayer grid.

    With a configured ``grid_interval_weights`` tuple, deterministic largest
    remainder allocation returns exactly ``N_grid`` total intervals and at
    least one per electrical layer. Without it, the historical
    ``n_per = N_grid // n_elec`` path is retained exactly, including optional
    ``grading_N_mult`` refinement of graded layers.
    """
    elec = electrical_layers(stack)
    weights, _ = _custom_grid_parameters(stack, elec)
    if weights:
        if bool(getattr(stack, "band_grading", False)) and any(
            layer.params is not None
            and int(getattr(layer.params, "grading_N_mult", 1)) > 1
            for layer in elec
        ):
            raise ValueError(
                "custom electrical-grid interval weights cannot be combined "
                "with band_grading and grading_N_mult > 1"
            )
        return _largest_remainder_interval_counts(N_grid, weights)

    n_per = N_grid // len(elec)
    band_grading = bool(getattr(stack, "band_grading", False))
    counts: list[int] = []
    for layer in elec:
        mult = 1
        if (
            band_grading
            and layer.params is not None
            and has_grading_params(layer.params)
        ):
            mult = max(1, int(getattr(layer.params, "grading_N_mult", 1)))
        counts.append(n_per * mult)
    return counts


def _grid_node_count(stack: DeviceStack, N_grid: int) -> int:
    """Return the number of electrical grid nodes run_jv_sweep will build.

    This is the single source of truth for the electrical-grid sizing formula.
    Both run_jv_sweep (internally) and tandem callers (to pre-size generation
    profiles) must use this helper so the two sites can never silently diverge.

    Formula: ``1 + sum(_layer_node_counts(stack, N_grid))``. multilayer_grid
    deduplicates shared boundary points, so the total node count is one more
    than the sum of per-layer intervals. With no graded layer this reduces to
    ``1 + n_elec * (N_grid // n_elec)`` — byte-identical to the legacy formula.
    A custom weighted grid instead has exactly ``N_grid + 1`` nodes.
    """
    return 1 + sum(_layer_node_counts(stack, N_grid))


def build_electrical_grid(stack: DeviceStack, N_grid: int) -> np.ndarray:
    """Return the electrical grid POSITIONS run_jv_sweep will integrate on.

    Companion to :func:`_grid_node_count`, which single-sources the node COUNT.
    Sizing alone was not enough: ``compute_tandem_generation`` used the count to
    pre-size a generation profile and then sampled it on a UNIFORM grid, while
    ``run_jv_sweep`` integrated that profile on this tanh-clustered one.  The
    shapes agreed and the positions did not, so every value landed at the wrong
    depth — measured at 105 nm of mismatch on a 500 nm sub-cell.  A profile is a
    spatial density, so any caller that builds one for this solver needs the
    positions, not just how many of them there are.

    Substrate layers are excluded for the same reason ``run_jv_sweep`` excludes
    them: they are optical-only and have no drift-diffusion counterpart.
    """
    elec = electrical_layers(stack)
    _, alphas = _custom_grid_parameters(stack, elec)
    return multilayer_grid([
        Layer(lyr.thickness, n)
        for lyr, n in zip(elec, _layer_node_counts(stack, N_grid))
    ], alpha=alphas if alphas else 3.0)


def _default_V_max(stack: DeviceStack) -> float:
    """Default upper voltage for a J-V sweep when the caller passes V_max=None.

    Rationale
    ---------
    V_oc on heterostacks is bounded above by the active operating built-in
    potential. Explicit contact-potential modes use their selected work-function
    difference; compatibility stacks retain the historical band-derived
    ``compute_V_bi()`` value, which can exceed the manual ``stack.V_bi`` field.
    If we opened the sweep
    only to the manual V_bi, forward sweeps on high-V_oc stacks (MAPbI3 etc.)
    would never cross J = 0 and ``compute_metrics`` would return V_oc = V_max.

    Formula:
        V_upper = max(V_bi_eff * 1.3, 1.4)

    The 1.3 headroom captures the minority-quasi-Fermi-level rise beyond V_bi
    under strong illumination; the 1.4 V floor is a backstop for legacy configs
    where chi/Eg are not set (so the compatibility estimate falls back to V_bi,
    which for a MAPbI3-like stack can be ~1.05 V — 1.3× that is only 1.37 V,
    uncomfortably close to the observed 1.05-1.15 V V_oc range).

    This is the single source of truth for the default V_max and is unit-tested
    directly so the formula can be audited without running a full sweep.
    """
    # The operating value is signed (negative for n-contact-left devices); the
    # sweep range needs its magnitude.
    resolve_operating = getattr(
        stack, "operating_built_in_potential", stack.compute_V_bi
    )
    V_bi_eff = abs(resolve_operating())
    return max(V_bi_eff * 1.3, 1.4)


def _fixed_generation_reference(generation: np.ndarray | None) -> str:
    if generation is None:
        return "stack_baseline_generation"
    values = np.ascontiguousarray(np.asarray(generation, dtype="<f8"))
    if np.any(~np.isfinite(values)):
        raise ValueError("fixed_generation must contain only finite values")
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode("ascii"))
    digest.update(values.tobytes())
    return f"sha256:{digest.hexdigest()}"


def build_jv_experiment_protocol(
    stack: DeviceStack,
    *,
    v_rate: float = 0.1,
    n_points: int = 50,
    V_max: float | None = None,
    illuminated: bool = True,
    fixed_generation: np.ndarray | None = None,
    implicit_legacy_protocol: bool = False,
) -> ExperimentProtocol:
    """Describe the exact built-in forward/reverse J-V history."""

    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")
    if not np.isfinite(v_rate) or v_rate <= 0.0:
        raise ValueError(f"v_rate must be finite and positive, got {v_rate}")
    V_upper = _default_V_max(stack) if V_max is None else float(V_max)
    if not np.isfinite(V_upper):
        raise ValueError("V_max must be finite")
    forward = np.linspace(0.0, V_upper, n_points)
    reverse = np.linspace(V_upper, 0.0, n_points)
    dwell = abs(V_upper) / (v_rate * (n_points - 1))
    if illuminated:
        initial_state = "finite_time_illuminated_preconditioned"
        pre_bias = 0.0
        soak = 1.0e-3
        source = _fixed_generation_reference(fixed_generation)
        history = (
            IlluminationStep(
                phase="short_circuit_preconditioning",
                condition="baseline",
                duration_s=soak,
                intensity_suns=1.0,
                source_reference=source,
            ),
            IlluminationStep(
                phase="forward_reverse_scan",
                condition="baseline",
                duration_s=2.0 * n_points * dwell,
                intensity_suns=1.0,
                source_reference=source,
            ),
        )
        settle = DCSettleCriterion(kind="finite_time", duration_s=soak)
    else:
        initial_state = "dark_equilibrium"
        pre_bias = None
        soak = 0.0
        history = (
            IlluminationStep(
                phase="forward_reverse_scan",
                condition="dark",
                duration_s=2.0 * n_points * dwell,
            ),
        )
        settle = DCSettleCriterion(kind="not_applicable")
    return ExperimentProtocol(
        experiment="jv_hysteresis",
        initial_state_source=initial_state,
        pre_bias_V=pre_bias,
        soak_duration_s=soak,
        dwell_duration_s=dwell,
        illumination_history=history,
        temperature_K=float(stack.T),
        scan=ScanProtocol(
            axis="voltage_V",
            direction="ascending_then_descending",
            start=0.0,
            stop=V_upper,
            rate_V_s=float(v_rate),
        ),
        ac_excitation=None,
        dc_settle=settle,
        sampling=SamplingProtocol(
            axis="voltage_V",
            mode="linear",
            values=tuple(np.concatenate((forward, reverse))),
        ),
        implicit_legacy_protocol=implicit_legacy_protocol,
    )


def run_jv_sweep(
    stack: DeviceStack,
    N_grid: int = 100,
    v_rate: float = 0.1,      # V/s
    n_points: int = 50,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    V_max: float | None = None,
    progress: ProgressCallback | None = None,
    fixed_generation: np.ndarray | None = None,
    illuminated: bool = True,
    save_snapshots: bool = False,
    decompose_currents: bool = False,
    v_max_max_attempts: int = 1,
    certification_mode: Literal["strict", "diagnostic"] = "strict",
    allow_underresolved_grid: bool = False,
    allow_unvalidated_driver: bool = False,
    experiment_protocol: ExperimentProtocol | None = None,
    protocol_mode: ProtocolMode = "compatibility",
    rhs_regularization: RHSRegularization | None = None,
    collect_numerical_diagnostics: bool = False,
) -> JVResult:
    """Run forward and reverse J-V sweeps.

    V_max : upper voltage limit. If None, defaults to max(V_bi_eff*1.3, 1.4). With
      heterojunction band offsets, V_oc can exceed V_bi, so pass a larger
      value (e.g. 1.4 V for MAPbI3) to capture the full forward curve.
      That default is derived from the CONTACTS and knows nothing about the
      point being swept, so it misses in both directions. Too low is loud —
      ``voc_bracketed=False``, and ``v_max_max_attempts`` below recovers it.
      Too high is silent and expensive: a device whose V_oc has collapsed
      (deep cliff/spike offsets, short lifetime, strong interface
      recombination) still gets swept to 1.4 V, grinding the deep
      forward-injection region above its own V_oc. Measured on two
      spike-side screening points: 1567 s and 630 s at the default versus
      23 s and 16 s starting low and climbing, agreeing on V_oc to 1.6 mV
      and 0.1 mV. For a sweep over devices whose V_oc varies, prefer a low
      ``V_max`` plus ``v_max_max_attempts`` over one range that fits all.

    v_max_max_attempts : Phase E1.9 — opt-in adaptive V_max bump when the
      first sweep fails to bracket V_oc (forward J never crosses zero).
      Default 1 = legacy no-retry behaviour, bit-identical to pre-E1.9.
      Values > 1 trigger up to ``v_max_max_attempts - 1`` retries with
      V_max bumped by 0.5 V per attempt (cap at V_initial + 2.0 V). The
      cost is bounded — at worst, ``v_max_max_attempts`` full forward/
      reverse sweeps, and fewer once V_max saturates at the cap, since a
      further rung would repeat that sweep verbatim. Each rung holds the
      VOLTAGE STEP of the first attempt and grows ``n_points`` to match, so
      a retry never answers more coarsely than the attempt it is correcting
      (V_oc is interpolated between the bracketing samples, so its error
      scales with that step). Useful for stacks where V_oc may exceed the
      default V_max (e.g. Robin contacts at low ETL doping; SCAPS
      validation script's extreme-N_D sweep points), and — passed together
      with a deliberately low ``V_max`` — as the cheap way to sweep devices
      whose V_oc is unknown or collapses: start below every candidate and
      let the ladder climb only where it must. On exhaustion the
      result returns ``voc_bracketed=False`` and the standard sentinel
      zeros for V_oc/FF/PCE — no exception is raised, so callers can
      inspect the flag and decide how to handle the residual gap.

    fixed_generation : optional pre-computed generation profile G(x) [m⁻³ s⁻¹].
      Must be a 1-D array of shape (N,) where N is the number of electrical-grid
      nodes (determined by N_grid and the number of electrical layers). When
      provided, this profile is used verbatim in place of Beer-Lambert or TMM
      optics for both the initial finite-time illuminated preconditioning and
      every subsequent solver call. When None (default), the existing
      single-junction optics path is used unchanged.

    illuminated : when False, run a dark J-V (G=0 everywhere). The initial
      state is dark equilibrium instead of a light-conditioned state. Cannot
      be combined with fixed_generation.

    certification_mode : "strict" (default) raises
      JVCertificationError as soon as a voltage point fails its required
      local-refinement certificate. "diagnostic" keeps running from the
      rejected candidate state for debugging, but stores NaN at the failed
      point and every downstream point in that warm-start chain, records
      immutable per-point statuses, and returns unavailable NaN metrics.
      Diagnostic output must not be used as a physical J-V curve.

    allow_underresolved_grid : explicit diagnostic-only override for the
      thick-layer Debye-resolution guard. The default False rejects a known
      under-resolved wafer-scale grid before material construction or time
      integration. True permits execution but does not certify mesh or J-V
      convergence and must not be used for physical results.

    allow_unvalidated_driver : explicit diagnostic-only override for a stack
      whose production policy requires the cancellation-safe quasi-Fermi
      driver. This never certifies the transient result and never switches
      solvers implicitly.
    """
    if certification_mode not in ("strict", "diagnostic"):
        raise ValueError(
            "certification_mode must be 'strict' or 'diagnostic', got "
            f"{certification_mode!r}"
        )
    if protocol_mode not in ("compatibility", "research_strict"):
        raise ValueError(
            "protocol_mode must be 'compatibility' or 'research_strict', got "
            f"{protocol_mode!r}"
        )
    if v_max_max_attempts < 1:
        raise ValueError(
            f"v_max_max_attempts must be >= 1, got {v_max_max_attempts}"
        )
    if n_points < 2:
        # Checked before the ladder below, which divides by ``n_points - 1``.
        raise ValueError(f"n_points must be >= 2, got {n_points}")
    if v_max_max_attempts > 1:
        if experiment_protocol is not None or protocol_mode == "research_strict":
            expected = build_jv_experiment_protocol(
                stack,
                v_rate=v_rate,
                n_points=n_points,
                V_max=V_max,
                illuminated=illuminated,
                fixed_generation=fixed_generation,
                implicit_legacy_protocol=True,
            )
            resolve_experiment_protocol(
                experiment_protocol, expected, mode=protocol_mode,
            )
            raise ProtocolMismatchError(
                "an explicit research protocol requires a fixed J-V voltage "
                "window; v_max_max_attempts must be 1"
            )
        # Adaptive V_max bump: run the legacy single-attempt path first;
        # if it brackets V_oc successfully, return that result unchanged
        # (so attempts > 1 with successful first try is bit-identical to
        # attempts = 1). On failed bracket, retry with V_max += 0.5 per
        # attempt until success or attempts exhaust.
        V_max_initial = (
            V_max
            if V_max is not None
            else max(abs(stack.operating_built_in_potential()) * 1.3, 1.4)
        )
        V_max_cap = V_max_initial + 2.0
        V_max_attempt = V_max_initial
        # Hold the VOLTAGE STEP fixed and let ``n_points`` grow with the rung,
        # instead of stretching a fixed sample count over a wider range. V_oc
        # is interpolated between the two samples that bracket it, so its
        # error scales with this step; passing ``n_points`` through unchanged
        # made every retry answer more coarsely than the attempt it was
        # correcting, and did so systematically in V_max.
        dV = V_max_initial / (n_points - 1)
        attempts_used = 0
        last_result: JVResult | None = None
        while True:
            last_result = run_jv_sweep(
                stack=stack, N_grid=N_grid, v_rate=v_rate,
                n_points=int(round(V_max_attempt / dV)) + 1,
                rtol=rtol, atol=atol,
                V_max=V_max_attempt, progress=progress,
                fixed_generation=fixed_generation,
                illuminated=illuminated, save_snapshots=save_snapshots,
                decompose_currents=decompose_currents,
                v_max_max_attempts=1,  # disable recursion on inner call
                certification_mode=certification_mode,
                allow_underresolved_grid=allow_underresolved_grid,
                allow_unvalidated_driver=allow_unvalidated_driver,
                rhs_regularization=rhs_regularization,
                collect_numerical_diagnostics=collect_numerical_diagnostics,
            )
            if (
                last_result.status_fwd is not None
                and not all(status.valid for status in last_result.status_fwd)
            ):
                # A wider voltage window cannot repair an uncertified
                # warm-start chain. Diagnostic mode returns it immediately.
                return last_result
            if last_result.metrics_fwd.voc_bracketed:
                return last_result
            attempts_used += 1
            V_max_next = min(V_max_attempt + 0.5, V_max_cap)
            if attempts_used >= v_max_max_attempts or V_max_next <= V_max_attempt:
                # Budget spent, or V_max has saturated at the cap so a further
                # rung would repeat this exact sweep. (The old bail-out here
                # compared a float with a JVResult and could never fire, so an
                # exhausted ladder re-ran the capped sweep to the last attempt.)
                break
            V_max_attempt = V_max_next
        # ``last_result`` is set: the loop body always runs at least once.
        assert last_result is not None
        return last_result
    if N_grid < 3:
        raise ValueError(f"N_grid must be >= 3, got {N_grid}")
    if v_rate <= 0:
        raise ValueError(f"v_rate must be positive, got {v_rate}")
    if not illuminated and fixed_generation is not None:
        raise ValueError("Cannot combine illuminated=False with fixed_generation")
    resolved_protocol = resolve_experiment_protocol(
        experiment_protocol,
        build_jv_experiment_protocol(
            stack,
            v_rate=v_rate,
            n_points=n_points,
            V_max=V_max,
            illuminated=illuminated,
            fixed_generation=fixed_generation,
            implicit_legacy_protocol=True,
        ),
        mode=protocol_mode,
    )
    for i, layer in enumerate(stack.layers):
        if layer.thickness <= 0:
            raise ValueError(
                f"layer {i} ({layer.name!r}) has non-positive thickness {layer.thickness}"
            )

    # Grid construction uses electrical layers only — substrate layers are
    # optical-only and have no drift-diffusion counterpart, so allocating
    # grid nodes inside them would desync MaterialArrays masks with the
    # solver state vector. TMM/optics paths still see the full stack.
    elec = electrical_layers(stack)
    x = build_electrical_grid(stack, N_grid)
    require_thick_layer_interface_resolution(
        x,
        stack,
        N_grid=N_grid,
        allow_underresolved_grid=allow_underresolved_grid,
    )
    require_jv_driver_capability(
        stack,
        requested_driver="transient",
        allow_unvalidated_driver=allow_unvalidated_driver,
    )
    N = _grid_node_count(stack, N_grid)
    assert N == len(x), "grid node count mismatch — _grid_node_count is out of sync"

    # Build the material cache once — shared across forward and reverse sweeps
    # and every RHS call inside them. See solver/mol.py:MaterialArrays.
    mat = build_material_arrays(x, stack)

    # Optional generation override: tandem callers inject a pre-computed G(x)
    # profile (e.g. from combined-TMM) instead of letting the sweep recompute
    # optics internally. Validation is done here so the error surfaces early
    # before any time-consuming ODE work begins.
    if fixed_generation is not None:
        expected_shape = (N,)
        if np.asarray(fixed_generation).shape != expected_shape:
            raise ValueError(
                f"fixed_generation shape {np.asarray(fixed_generation).shape} "
                f"!= expected {expected_shape} "
                f"(N={N} electrical-grid nodes for N_grid={N_grid} "
                f"and {len(elec)} electrical layers)"
            )
        mat = dataclasses.replace(
            mat, G_optical=np.asarray(fixed_generation, dtype=float).copy()
        )

    # Kick off progress BEFORE the initial equilibration solve. That solve
    # (solve_illuminated_ss / dark settle) can be slow on stiff, high-mobility
    # stacks and emits nothing itself, so without this frame the UI sits at a
    # frozen "Idle 0%" for the whole settle. current=0 marks the indeterminate
    # "equilibrating" phase; the sweep loop then reports "jv_forward" from 1.
    if progress is not None:
        progress("jv_init", 0, n_points, "equilibrating")

    # Start from the appropriate equilibrium state:
    # - Dark mode: use dark equilibrium directly (no illumination settle)
    # - Fixed generation: precondition with the overridden material cache
    # - Default: preserve the independently built preconditioning cache; this
    #   is load-bearing for the calibrated stiff-branch regression protocol
    initial_numerical_diagnostics = None
    if not illuminated:
        y_eq = solve_equilibrium(x, stack)
    elif fixed_generation is not None:
        if rhs_regularization is None:
            if collect_numerical_diagnostics:
                y_eq, initial_numerical_diagnostics = solve_illuminated_ss(
                    x,
                    stack,
                    V_app=0.0,
                    rtol=rtol,
                    atol=atol,
                    mat=mat,
                    return_diagnostics=True,
                )
            else:
                y_eq = solve_illuminated_ss(
                    x, stack, V_app=0.0, rtol=rtol, atol=atol, mat=mat,
                )
        else:
            initial = solve_illuminated_ss(
                x,
                stack,
                V_app=0.0,
                rtol=rtol,
                atol=atol,
                mat=mat,
                regularization=rhs_regularization,
                return_diagnostics=collect_numerical_diagnostics,
            )
            if collect_numerical_diagnostics:
                y_eq, initial_numerical_diagnostics = initial
            else:
                y_eq = initial
    else:
        if rhs_regularization is None:
            if collect_numerical_diagnostics:
                y_eq, initial_numerical_diagnostics = solve_illuminated_ss(
                    x,
                    stack,
                    V_app=0.0,
                    rtol=rtol,
                    atol=atol,
                    return_diagnostics=True,
                )
            else:
                y_eq = solve_illuminated_ss(
                    x, stack, V_app=0.0, rtol=rtol, atol=atol,
                )
        else:
            initial = solve_illuminated_ss(
                x,
                stack,
                V_app=0.0,
                rtol=rtol,
                atol=atol,
                regularization=rhs_regularization,
                return_diagnostics=collect_numerical_diagnostics,
            )
            if collect_numerical_diagnostics:
                y_eq, initial_numerical_diagnostics = initial
            else:
                y_eq = initial

    def _sweep(
        V_start: float,
        V_end: float,
        y_init: np.ndarray,
        stage: str,
        initial_state_certified: bool = True,
    ):
        """Sweep from V_start to V_end, starting from carrier state y_init.

        Returns (V_arr, J_arr, y_final, snapshots, decomp, statuses) so sweeps
        can be chained. snapshots and decomp are populated only when the
        corresponding flags (save_snapshots, decompose_currents) are True.

        The wrong-branch detector applies only to ``jv_forward``. That branch
        starts from the built-in short-circuit initialization and therefore
        satisfies the detector's J(V) <= J_sc protocol premise. The reverse
        branch starts from a high-bias, precharged transient state; stored
        electronic, ionic, or displacement energy makes the same ceiling
        physically invalid there.
        """
        V_arr = np.linspace(V_start, V_end, n_points)
        dt = abs(V_end - V_start) / (v_rate * (n_points - 1))
        t_points = np.arange(n_points) * dt
        J_arr = np.zeros(n_points)
        snaps: list[SpatialSnapshot] = []
        d_Jn: list[float] = []
        d_Jp: list[float] = []
        d_Jion: list[float] = []
        d_Jdisp: list[float] = []
        d_Jtot: list[float] = []
        statuses: list[JVPointStatus] = []
        y = y_init.copy()
        V_prev = float(V_arr[0])
        J_ref: float | None = None
        chain_certified = initial_state_certified

        def _candidate(
            y_entry: np.ndarray,
            V_k: float,
            t_lo: float,
            t_hi: float,
            n_legs: int,
        ) -> tuple[
            np.ndarray,
            float,
            tuple[JVAcceptedSolveDiagnostics, ...],
        ]:
            accepted_diagnostics: list[JVAcceptedSolveDiagnostics] | None = (
                [] if collect_numerical_diagnostics else None
            )
            try:
                if rhs_regularization is None:
                    y_candidate = _integrate_step(
                        x,
                        y_entry,
                        stack,
                        mat,
                        V_k,
                        t_lo,
                        t_hi,
                        rtol,
                        atol,
                        illuminated=illuminated,
                        n_legs=n_legs,
                        _accepted_diagnostics=accepted_diagnostics,
                    )
                else:
                    y_candidate = _integrate_step(
                        x,
                        y_entry,
                        stack,
                        mat,
                        V_k,
                        t_lo,
                        t_hi,
                        rtol,
                        atol,
                        illuminated=illuminated,
                        n_legs=n_legs,
                        regularization=rhs_regularization,
                        _accepted_diagnostics=accepted_diagnostics,
                    )
            except _JV_NUMERIC_FAILURES as exc:
                raise _JVCandidateFailure(
                    "integration_failed",
                    f"integration failed at V={V_k:.6g} V with "
                    f"{n_legs} leg(s): {type(exc).__name__}: {exc}",
                ) from exc
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    J_candidate = _compute_current(
                        x, y_candidate, stack, V_k, y_prev=y_entry, dt=dt,
                        mat=mat, V_app_prev=V_prev,
                    )
            except _JV_NUMERIC_FAILURES as exc:
                raise _JVCandidateFailure(
                    "current_extraction_failed",
                    f"terminal-current extraction failed at V={V_k:.6g} V "
                    f"with {n_legs} leg(s): {type(exc).__name__}: {exc}",
                    state=y_candidate,
                ) from exc
            return y_candidate, J_candidate, tuple(accepted_diagnostics or ())

        for k, V_k in enumerate(V_arr):
            y_prev = y.copy()
            t_lo = t_points[k]
            t_hi = t_lo + dt
            candidate_error: _JVCandidateFailure | None = None
            accepted_diagnostics: tuple[JVAcceptedSolveDiagnostics, ...] = ()
            try:
                y, J_k, accepted_diagnostics = _candidate(
                    y_prev, float(V_k), t_lo, t_hi, 1,
                )
            except _JVCandidateFailure as exc:
                candidate_error = exc
                y = y_prev if exc.state is None else exc.state.copy()
                J_k = np.nan

            initial_candidate_current = (
                None if candidate_error is not None else float(J_k)
            )

            upstream_valid = chain_certified
            local_valid = candidate_error is None
            refinement_required = False
            refinement_converged: bool | None = (
                False if candidate_error is not None else None
            )
            attempted_legs = [1]
            attempted_currents: list[float | None] = [
                None if candidate_error is not None else float(J_k)
            ]
            failed_legs: list[int] = [1] if candidate_error is not None else []
            accepted_legs = 0 if candidate_error is not None else 1
            reason_code = (
                candidate_error.reason_code
                if candidate_error is not None else "certified"
            )
            message = (
                f"single-leg candidate failed: {candidate_error}"
                if candidate_error is not None else ""
            )

            if (candidate_error is None
                    and (not np.isfinite(J_k) or not np.all(np.isfinite(y)))):
                local_valid = False
                reason_code = "nonfinite_candidate"
                message = (
                    "the integrator returned a non-finite state or terminal "
                    "current"
                )

            # The first/last short-circuit sample has no trustworthy lower
            # current bound. A wrong Radau branch can under-read it, so the
            # four-leg probe is a required certificate, not an advisory
            # warning. A disagreement additionally requires the independent
            # eight-leg result to agree with the probe.
            if local_valid and illuminated and float(V_k) == 0.0:
                refinement_required = True
                attempted_legs.append(_J_ZERO_BIAS_PROBE_LEGS)
                try:
                    y_probe, J_probe, _probe_diagnostics = _candidate(
                        y_prev, float(V_k), t_lo, t_hi,
                        _J_ZERO_BIAS_PROBE_LEGS,
                    )
                except RuntimeError as exc:
                    attempted_currents.append(None)
                    failed_legs.append(_J_ZERO_BIAS_PROBE_LEGS)
                    local_valid = False
                    refinement_converged = False
                    reason_code = "zero_bias_probe_failed"
                    message = (
                        f"{_J_ZERO_BIAS_PROBE_LEGS}-leg V=0 subdivision "
                        f"probe failed: {exc}"
                    )
                else:
                    attempted_currents.append(float(J_probe))
                    if _refinements_agree(
                        y, J_k, y_probe, J_probe, N, mat.N_iface_state,
                    ):
                        refinement_converged = True
                        reason_code = "zero_bias_refinement_agreed"
                    else:
                        attempted_legs.append(_J_ZERO_BIAS_CONFIRM_LEGS)
                        try:
                            y_confirm, J_confirm, confirm_diagnostics = _candidate(
                                y_prev, float(V_k), t_lo, t_hi,
                                _J_ZERO_BIAS_CONFIRM_LEGS,
                            )
                        except RuntimeError as exc:
                            attempted_currents.append(None)
                            failed_legs.append(_J_ZERO_BIAS_CONFIRM_LEGS)
                            local_valid = False
                            refinement_converged = False
                            reason_code = "zero_bias_confirmation_failed"
                            message = (
                                f"{_J_ZERO_BIAS_CONFIRM_LEGS}-leg V=0 "
                                f"subdivision confirmation failed: {exc}"
                            )
                        else:
                            attempted_currents.append(float(J_confirm))
                            if _refinements_agree(
                                y_probe, J_probe, y_confirm, J_confirm,
                                N, mat.N_iface_state,
                            ):
                                y, J_k = y_confirm, J_confirm
                                accepted_diagnostics = confirm_diagnostics
                                accepted_legs = _J_ZERO_BIAS_CONFIRM_LEGS
                                refinement_converged = True
                                reason_code = "zero_bias_refinement_recovered"
                            else:
                                local_valid = False
                                refinement_converged = False
                                reason_code = "zero_bias_refinement_disagreed"
                                message = (
                                    "V=0 local subdivision did not converge: "
                                    f"single/{_J_ZERO_BIAS_PROBE_LEGS}/"
                                    f"{_J_ZERO_BIAS_CONFIRM_LEGS}-leg currents "
                                    f"were {J_k:.6g}/{J_probe:.6g}/"
                                    f"{J_confirm:.6g} A/m^2"
                                )

            # Wrong-branch rejection. Only fires in the illuminated forward
            # power quadrant after the built-in V=0 initialization. A current
            # above the J_sc ceiling is unphysical for this protocol. Each
            # positive-to-nonpositive transition is also certified by
            # local subdivision because it determines V_oc and can otherwise
            # hide a wrong low-current landing that the one-sided ceiling
            # cannot see. Healthy non-crossing steps remain bit-identical. See
            # _J_BRANCH_EXCESS for the protocol boundary.
            #
            # The V=0 refinement above makes J_ref numerically self-consistent
            # before it becomes this ceiling.  Do not replace it with a photon
            # budget: this routine reports transient total current, including
            # displacement/ionic contributions, which stored energy can drive
            # above the instantaneous optical generation.
            _ceiling = (
                _branch_current_ceiling(J_ref)
                if stage == "jv_forward" and J_ref is not None and J_ref > 0.0
                else None
            )
            _ceiling_violation = bool(
                _ceiling is not None and float(V_k) > 0.0 and J_k > _ceiling
            )
            _zero_crossing = bool(
                stage == "jv_forward"
                and _ceiling is not None
                and k > 0
                and J_arr[k - 1] > 0.0
                and J_k <= 0.0
            )
            if (local_valid and illuminated and _ceiling is not None
                    and (_ceiling_violation or _zero_crossing)):
                refinement_required = True
                refinement_converged = False
                previous_retry: tuple[np.ndarray, float] | None = None
                for n_legs in _J_BRANCH_RETRY_LEGS:
                    attempted_legs.append(n_legs)
                    try:
                        y_retry, J_retry, retry_diagnostics = _candidate(
                            y_prev, float(V_k), t_lo, t_hi, n_legs,
                        )
                    except RuntimeError:
                        attempted_currents.append(None)
                        failed_legs.append(n_legs)
                        continue
                    attempted_currents.append(float(J_retry))
                    if (previous_retry is not None
                            and previous_retry[1] <= _ceiling
                            and J_retry <= _ceiling
                            and _refinements_agree(
                                previous_retry[0], previous_retry[1],
                                y_retry, J_retry, N, mat.N_iface_state,
                            )):
                        y, J_k = y_retry, J_retry
                        accepted_diagnostics = retry_diagnostics
                        accepted_legs = n_legs
                        refinement_converged = True
                        reason_code = (
                            "zero_crossing_refinement_recovered"
                            if _zero_crossing and not _ceiling_violation
                            else "positive_bias_refinement_recovered"
                        )
                        break
                    previous_retry = (y_retry, J_retry)
                else:
                    failure_note = (
                        f" Failed leg counts: {failed_legs}."
                        if failed_legs else ""
                    )
                    local_valid = False
                    if _zero_crossing and not _ceiling_violation:
                        reason_code = "zero_crossing_refinement_exhausted"
                        message = (
                            f"terminal current changed sign from "
                            f"{J_arr[k - 1]:.6g} to {J_k:.6g} A/m^2; "
                            f"refinements {_J_BRANCH_RETRY_LEGS} did not "
                            "produce two converged values within the forward "
                            f"protocol ceiling {_ceiling:.3f} A/m^2."
                            f"{failure_note}"
                        )
                    else:
                        reason_code = "positive_bias_refinement_exhausted"
                        message = (
                            f"terminal current {J_k:.6g} A/m^2 exceeded the "
                            "protocol current ceiling "
                            f"J_sc + max({_J_BRANCH_EXCESS:g}*|J_sc|, "
                            f"{_J_BRANCH_CURRENT_ATOL:g} A/m^2) = "
                            f"{_ceiling:.3f}; refinements "
                            f"{_J_BRANCH_RETRY_LEGS} did not produce two "
                            f"converged values within it.{failure_note}"
                        )

            if local_valid and not upstream_valid:
                reason_code = "upstream_state_uncertified"
                message = (
                    "the local step completed, but its warm-start state chain "
                    "contains an uncertified predecessor"
                )
            diagnostics_for_status = (
                accepted_diagnostics
                if local_valid and collect_numerical_diagnostics
                else ()
            )
            point_status = JVPointStatus(
                branch=stage,
                index=k,
                voltage=float(V_k),
                valid=bool(local_valid and upstream_valid),
                upstream_valid=bool(upstream_valid),
                refinement_required=refinement_required,
                refinement_converged=refinement_converged,
                attempted_legs=tuple(attempted_legs),
                attempted_currents=tuple(attempted_currents),
                failed_legs=tuple(failed_legs),
                accepted_legs=accepted_legs,
                reason_code=reason_code,
                message=message,
                candidate_current=initial_candidate_current,
                numerical_diagnostics=diagnostics_for_status,
                nfev=_sum_accepted_solver_count(
                    diagnostics_for_status, "nfev",
                ),
                njev=_sum_accepted_solver_count(
                    diagnostics_for_status, "njev",
                ),
                nlu=_sum_accepted_solver_count(
                    diagnostics_for_status, "nlu",
                ),
            )
            if not point_status.valid and certification_mode == "strict":
                raise JVCertificationError(point_status)
            if not local_valid:
                warnings.warn(
                    f"{JVCertificationError(point_status)}; diagnostic mode "
                    "will return NaN for this point and its downstream chain.",
                    JVCertificationWarning,
                    stacklevel=2,
                )
            statuses.append(point_status)
            chain_certified = point_status.valid
            J_arr[k] = J_k if point_status.valid else np.nan
            if J_ref is None and float(V_k) == 0.0 and point_status.valid:
                J_ref = J_k
            if save_snapshots:
                snaps.append(extract_spatial_snapshot(x, y, stack, float(V_k), mat=mat))
            if decompose_currents and point_status.valid:
                cc = compute_current_components(
                    x, y, stack, float(V_k),
                    y_prev=y_prev, dt=dt, mat=mat, V_app_prev=V_prev,
                )
                d_Jn.append(float(cc.J_n[0]))
                d_Jp.append(float(cc.J_p[0]))
                d_Jion.append(float(cc.J_ion[0]))
                d_Jdisp.append(float(cc.J_disp[0]))
                d_Jtot.append(float(cc.J_total[0]))
            elif decompose_currents:
                d_Jn.append(np.nan)
                d_Jp.append(np.nan)
                d_Jion.append(np.nan)
                d_Jdisp.append(np.nan)
                d_Jtot.append(np.nan)
            V_prev = float(V_k)
            if progress is not None:
                progress(stage, k + 1, n_points, "")
        decomp = None
        if decompose_currents:
            decomp = JVCurrentDecomp(
                J_n=np.array(d_Jn), J_p=np.array(d_Jp),
                J_ion=np.array(d_Jion), J_disp=np.array(d_Jdisp),
                J_total=np.array(d_Jtot),
            )
        return V_arr, J_arr, y, snaps, decomp, tuple(statuses)

    V_upper = _default_V_max(stack) if V_max is None else V_max
    # Forward sweep: dark equilibrium → short circuit → open circuit
    (
        V_fwd, J_fwd, y_oc, snaps_fwd, decomp_fwd, status_fwd,
    ) = _sweep(0.0, V_upper, y_eq, "jv_forward")
    # Reverse sweep: continue from the high-bias, light-soaked state to short
    # circuit. It retains the mandatory V=0 subdivision certificate, but does
    # not use the forward-only J_sc ceiling: a precharged transient can release
    # stored electronic, ionic, or displacement energy above its terminal J_sc.
    reverse_seed_certified = all(status.valid for status in status_fwd)
    (
        V_rev, J_rev, _, snaps_rev, decomp_rev, status_rev,
    ) = _sweep(
        V_upper, 0.0, y_oc, "jv_reverse",
        initial_state_certified=reverse_seed_certified,
    )
    # Refuse a V_oc above min(Eg)/q over the electrical layers — impossible
    # regardless of what the numerics produced past flat band. None when the
    # stack declares no band gaps (legacy chi=Eg=0 presets), which disables
    # the check rather than rejecting everything.
    V_oc_ceiling = thermodynamic_voc_ceiling(stack)
    fwd_certified = all(status.valid for status in status_fwd)
    rev_certified = all(status.valid for status in status_rev)
    m_fwd = (
        compute_metrics(
            V_fwd, J_fwd, V_oc_max=V_oc_ceiling, validity=status_fwd,
        )
        if fwd_certified else _unavailable_metrics()
    )
    status_rev_ordered = status_rev[::-1]
    m_rev = (
        compute_metrics(
            V_rev[::-1], J_rev[::-1], V_oc_max=V_oc_ceiling,
            validity=status_rev_ordered,
        )
        if rev_certified else _unavailable_metrics()
    )
    HI = (
        hysteresis_index(
            V_fwd, J_fwd, V_rev[::-1], J_rev[::-1],
            V_oc_max=V_oc_ceiling,
            validity_fwd=status_fwd,
            validity_rev=status_rev_ordered,
        )
        if fwd_certified and rev_certified else np.nan
    )

    return JVResult(
        V_fwd=V_fwd, J_fwd=J_fwd, V_rev=V_rev, J_rev=J_rev,
        metrics_fwd=m_fwd, metrics_rev=m_rev, hysteresis_index=HI,
        snapshots_fwd=tuple(snaps_fwd) if save_snapshots else None,
        snapshots_rev=tuple(snaps_rev) if save_snapshots else None,
        decomp_fwd=decomp_fwd,
        decomp_rev=decomp_rev,
        status_fwd=status_fwd,
        status_rev=status_rev,
        initial_numerical_diagnostics=initial_numerical_diagnostics,
        protocol=resolved_protocol,
        rhs_regularization=rhs_regularization or RHSRegularization(),
    )
