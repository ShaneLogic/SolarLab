"""Opt-in quasi-Fermi-potential steady-state solver.

This solver is deliberately narrower than the general transient and steady-
state drivers.  It targets high-conductivity homojunctions, where evaluating
Scharfetter-Gummel currents as the difference of two large density terms loses
the small terminal current to floating-point cancellation.

The nonlinear carrier unknowns are electron and hole quasi-Fermi-potential
increments.  Electrostatic potential is eliminated by an accurately converged
Poisson-Boltzmann solve at every residual evaluation.  Face currents are then
evaluated with ``expm1`` identities in terms of quasi-Fermi differences.  The
result is returned only after independent physical certificates pass; a Newton
termination condition alone is never treated as a steady-state certificate.

The module is opt-in and is not wired into the shipped J-V drivers.  Unsupported
non-local or contact/interface models fail before Newton starts.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.linalg import solve_banded

from perovskite_sim.constants import Q
from perovskite_sim.discretization.fe_operators import (
    bernoulli,
)
from perovskite_sim.experiments.jv_sweep import (
    JVMetrics,
    compute_metrics,
    thermodynamic_voc_ceiling,
)
from perovskite_sim.models.device import DeviceStack
from perovskite_sim.solver.mol import (
    MaterialArrays,
    StateVec,
    _charge_density,
    assemble_rhs,
    build_material_arrays,
    poisson_right_boundary,
)
from perovskite_sim.solver.newton import solve_equilibrium


DEFAULT_ILLUMINATION_STEPS = (
    0.0,
    1.0e-14,
    1.0e-12,
    1.0e-10,
    1.0e-8,
    1.0e-6,
    1.0e-5,
    1.0e-4,
    1.0e-3,
    1.0e-2,
    1.0e-1,
    1.0,
)

_MAX_ABS_LOG_DENSITY = 100.0


class QuasiFermiSteadyStateError(RuntimeError):
    """The guarded QF solve is unsupported or lacks a physical certificate."""


@dataclass(frozen=True)
class QuasiFermiSteadyStateResult:
    """Certified state and cancellation-safe current/QF diagnostics.

    The optional reference/increment arrays preserve QF differences that can
    be smaller than the resolution of the corresponding absolute potential.
    """

    y: np.ndarray
    phi: np.ndarray
    electron_quasi_fermi_potential_V: np.ndarray
    hole_quasi_fermi_potential_V: np.ndarray
    electron_face_current_A_m2: np.ndarray
    hole_face_current_A_m2: np.ndarray
    total_face_current_A_m2: np.ndarray
    electron_rate_per_s: np.ndarray
    hole_rate_per_s: np.ndarray
    current_A_m2: float
    face_current_spread_A_m2: float
    electron_continuity_bound_A_m2: float
    hole_continuity_bound_A_m2: float
    max_normalized_cell_residual: float
    poisson_residual: float
    poisson_residual_C_m2: float
    illumination_steps: tuple[float, ...]
    newton_iterations: int
    residual_evaluations: int
    V_app: float = 0.0
    illuminated: bool = True
    certified: bool = True
    electron_quasi_fermi_reference_V: np.ndarray | None = None
    hole_quasi_fermi_reference_V: np.ndarray | None = None
    electron_quasi_fermi_increment_V: np.ndarray | None = None
    hole_quasi_fermi_increment_V: np.ndarray | None = None


@dataclass(frozen=True)
class QuasiFermiJVSweepResult:
    """Illuminated QF states and extracted metrics on one voltage grid."""

    voltages_V: np.ndarray
    currents_A_m2: np.ndarray
    points: tuple[QuasiFermiSteadyStateResult, ...]
    metrics: JVMetrics

    @property
    def certified(self) -> bool:
        """Whether every retained voltage point has a physical certificate."""
        voltages = np.asarray(self.voltages_V, dtype=float)
        currents = np.asarray(self.currents_A_m2, dtype=float)
        return bool(
            voltages.ndim == 1
            and currents.ndim == 1
            and voltages.shape == currents.shape
            and len(self.points) == voltages.size
            and voltages.size > 0
            and np.all(np.isfinite(voltages))
            and np.all(np.isfinite(currents))
            and all(point.certified for point in self.points)
        )

    @property
    def metrics_certified(self) -> bool:
        """Whether point certificates also span a resolved open circuit."""
        return self.certified and self.metrics.voc_bracketed


@dataclass(frozen=True)
class _Evaluation:
    residual: np.ndarray
    y: np.ndarray
    phi: np.ndarray
    rate_n: np.ndarray
    rate_p: np.ndarray
    current_n: np.ndarray
    current_p: np.ndarray
    poisson_residual: float
    poisson_residual_C_m2: float


def _pin_mask(node_count: int) -> np.ndarray:
    pin = np.zeros(2 * node_count, dtype=bool)
    pin[[0, node_count - 1, node_count, 2 * node_count - 1]] = True
    return pin


def _density_from_log(log_density: np.ndarray, *, context: str) -> np.ndarray:
    """Exponentiate only inside the audited, unclipped density domain."""
    values = np.asarray(log_density, dtype=float)
    if (
        not np.all(np.isfinite(values))
        or np.any(np.abs(values) > _MAX_ABS_LOG_DENSITY)
    ):
        raise QuasiFermiSteadyStateError(
            f"{context} log-density is outside the audited exponential range "
            f"[-{_MAX_ABS_LOG_DENSITY:g}, {_MAX_ABS_LOG_DENSITY:g}]"
        )
    return np.exp(values)


def _require_supported(mat: MaterialArrays) -> None:
    unsupported: list[str] = []
    if getattr(mat, "has_dual_ions", False):
        unsupported.append("dual ions")
    if np.any(np.asarray(mat.D_ion_face, dtype=float) != 0.0):
        unsupported.append("mobile ions")
    if np.any(np.asarray(mat.P_ion0, dtype=float) != 0.0):
        unsupported.append("nonzero ionic background")
    if mat.N_iface_state != 0:
        unsupported.append("interface-plane states/charge")
    if mat.has_selective_contacts:
        unsupported.append("selective contacts")
    if mat.has_field_mobility:
        unsupported.append("field-dependent mobility")
    if mat.has_radiative_reabsorption:
        unsupported.append("non-local photon recycling")
    if mat.interface_faces:
        unsupported.append("thermionic interface flux")
    if any(
        (mat.interface_eval_node_n[k] != node)
        or (mat.interface_eval_node_p[k] != node)
        for k, node in enumerate(mat.interface_nodes)
        if k < len(mat.interface_eval_node_n)
        and k < len(mat.interface_eval_node_p)
    ):
        unsupported.append("cross-node interface recombination")
    if unsupported:
        raise QuasiFermiSteadyStateError(
            "quasi-Fermi steady-state solver does not support "
            + ", ".join(unsupported)
        )


def _validate_illumination_steps(
    illuminated: bool,
    values: tuple[float, ...],
) -> tuple[float, ...]:
    if not illuminated:
        return (0.0,)
    stages = tuple(float(value) for value in values)
    if not stages or stages[-1] != 1.0:
        raise ValueError("illuminated continuation must end at 1.0")
    if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in stages):
        raise ValueError("illumination continuation values must lie in [0, 1]")
    if any(right <= left for left, right in zip(stages[:-1], stages[1:])):
        raise ValueError("illumination continuation values must strictly increase")
    return stages


def _transport_balanced_seed(
    x: np.ndarray,
    stack: DeviceStack,
    mat: MaterialArrays,
    V_app: float,
    *,
    poisson_tolerance_V: float,
    poisson_max_iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a Poisson-consistent seed with resistance-weighted QF drops."""
    node_count = len(x)
    y_neutral = solve_equilibrium(x, stack)
    n_neutral = np.maximum(y_neutral[:node_count], 1.0)
    p_neutral = np.maximum(y_neutral[node_count : 2 * node_count], 1.0)
    dx = np.diff(x)
    thermal_voltage = mat.V_T_device
    phi_right = poisson_right_boundary(mat, V_app)

    def invariant_profile(
        density: np.ndarray,
        diffusivity: np.ndarray,
        left: float,
        right: float,
    ) -> np.ndarray:
        conductance = diffusivity * np.sqrt(density[:-1] * density[1:])
        resistance = dx / np.maximum(conductance, np.finfo(float).tiny)
        cumulative = np.concatenate(([0.0], np.cumsum(resistance)))
        if not np.isfinite(cumulative[-1]) or cumulative[-1] <= 0.0:
            raise QuasiFermiSteadyStateError(
                "cannot construct a finite transport-balanced seed"
            )
        return left + (right - left) * cumulative / cumulative[-1]

    u_n = invariant_profile(
        n_neutral,
        mat.D_n_face,
        np.log(mat.n_L) - mat.chi[0] / thermal_voltage,
        np.log(mat.n_R)
        - (phi_right + mat.chi[-1]) / thermal_voltage,
    )
    u_p = invariant_profile(
        p_neutral,
        mat.D_p_face,
        np.log(mat.p_L) + (mat.chi[0] + mat.Eg[0]) / thermal_voltage,
        np.log(mat.p_R)
        + (phi_right + mat.chi[-1] + mat.Eg[-1]) / thermal_voltage,
    )

    phi = np.linspace(0.0, phi_right, node_count)
    factor = mat.poisson_factor
    for _ in range(poisson_max_iterations):
        n = _density_from_log(
            u_n + (phi + mat.chi) / thermal_voltage,
            context="transport-balanced electron seed",
        )
        p = _density_from_log(
            u_p - (phi + mat.chi + mat.Eg) / thermal_voltage,
            context="transport-balanced hole seed",
        )
        rho = Q * (p - n + mat.N_D - mat.N_A)
        residual = (
            factor.C[:-1] * (phi[:-2] - phi[1:-1])
            + factor.C[1:] * (phi[2:] - phi[1:-1])
            + rho[1:-1] * factor.h_cell
        )
        banded = np.zeros((3, node_count - 2), dtype=float)
        banded[0, 1:] = factor.C[1:-1]
        banded[1] = -(
            factor.C[:-1] + factor.C[1:]
        ) - Q * (n[1:-1] + p[1:-1]) / thermal_voltage * factor.h_cell
        banded[2, :-1] = factor.C[1:-1]
        step = solve_banded((1, 1), banded, -residual)
        damping = min(1.0, 0.05 / max(float(np.max(np.abs(step))), np.finfo(float).tiny))
        phi[1:-1] += damping * step
        if float(np.max(np.abs(damping * step))) < poisson_tolerance_V:
            break
    else:
        raise QuasiFermiSteadyStateError(
            "transport-balanced seed Poisson iteration did not converge"
        )

    n = _density_from_log(
        u_n + (phi + mat.chi) / thermal_voltage,
        context="transport-balanced electron seed",
    )
    p = _density_from_log(
        u_p - (phi + mat.chi + mat.Eg) / thermal_voltage,
        context="transport-balanced hole seed",
    )
    n[[0, -1]] = (mat.n_L, mat.n_R)
    p[[0, -1]] = (mat.p_L, mat.p_R)
    return StateVec.pack(n, p, mat.P_ion0.copy()), phi


class _QuasiFermiSystem:
    def __init__(
        self,
        x: np.ndarray,
        stack: DeviceStack,
        mat: MaterialArrays,
        V_app: float,
        *,
        poisson_tolerance_V: float,
        poisson_max_iterations: int,
    ) -> None:
        self.x = x
        self.stack = stack
        self.mat = mat
        self.V_app = V_app
        self.node_count = len(x)
        self.poisson_tolerance_V = poisson_tolerance_V
        self.poisson_max_iterations = poisson_max_iterations
        self.base, self.phi0 = _transport_balanced_seed(
            x,
            stack,
            mat,
            V_app,
            poisson_tolerance_V=poisson_tolerance_V,
            poisson_max_iterations=poisson_max_iterations,
        )
        n0 = np.maximum(self.base[: self.node_count], 1.0)
        p0 = np.maximum(self.base[self.node_count : 2 * self.node_count], 1.0)
        self.log_n0 = np.log(n0)
        self.log_p0 = np.log(p0)
        self.thermal_voltage = mat.V_T_device
        self.qfn0 = self.thermal_voltage * self.log_n0 - (self.phi0 + mat.chi)
        self.qfp0 = self.thermal_voltage * self.log_p0 + (
            self.phi0 + mat.chi + mat.Eg
        )
        self.dx = np.diff(x)
        self.current_scale = max(abs(Q * float(stack.Phi)), 1.0)
        self.pin = _pin_mask(self.node_count)
        self.evaluation_count = 0

        # Source-only assembly retains generation, bulk recombination, and
        # local interface recombination without first forming the ordinary SG
        # transport divergence.  This avoids subtracting two O(1e12 A/m2)
        # fluxes before inserting the cancellation-safe QF current.
        self.source_mat = replace(
            mat,
            D_n_face=np.zeros_like(mat.D_n_face),
            D_p_face=np.zeros_like(mat.D_p_face),
        )

        dark = assemble_rhs(
            0.0,
            self.base,
            x,
            stack,
            self.source_mat,
            illuminated=False,
            V_app=V_app,
            phi_frozen=self.phi0,
        )[: 2 * self.node_count]
        light = assemble_rhs(
            0.0,
            self.base,
            x,
            stack,
            self.source_mat,
            illuminated=True,
            V_app=V_app,
            phi_frozen=self.phi0,
        )[: 2 * self.node_count]
        self.generation = light - dark

    @staticmethod
    def _stable_difference(
        a: np.ndarray,
        b: np.ndarray,
        delta: np.ndarray,
    ) -> np.ndarray:
        """Evaluate ``a - b`` from their known logarithmic ratio.

        Direct subtraction loses the terminal current when both SG legs are
        around 1e12 A/m2 in a highly doped emitter.  ``delta`` is the relevant
        quasi-Fermi-potential difference divided by thermal voltage.
        """
        out = np.empty_like(delta)
        positive = delta >= 0.0
        out[positive] = a[positive] * (-np.expm1(-delta[positive]))
        out[~positive] = b[~positive] * np.expm1(delta[~positive])
        return out

    def _solve_poisson(
        self,
        dqfn: np.ndarray,
        dqfp: np.ndarray,
        *,
        V_app: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        phi = self.phi0.copy()
        voltage = self.V_app if V_app is None else float(V_app)
        phi[0] = 0.0
        phi[-1] = poisson_right_boundary(self.mat, voltage)
        factor = self.mat.poisson_factor
        for _ in range(self.poisson_max_iterations):
            dphi = phi - self.phi0
            n = _density_from_log(
                self.log_n0 + (dqfn + dphi) / self.thermal_voltage,
                context="electron Poisson-Boltzmann iterate",
            )
            p = _density_from_log(
                self.log_p0 + (dqfp - dphi) / self.thermal_voltage,
                context="hole Poisson-Boltzmann iterate",
            )
            rho = _charge_density(
                p,
                n,
                self.base[2 * self.node_count : 3 * self.node_count],
                self.mat.P_ion0,
                self.mat.N_A,
                self.mat.N_D,
            )
            raw = (
                factor.C[:-1] * (phi[:-2] - phi[1:-1])
                + factor.C[1:] * (phi[2:] - phi[1:-1])
                + rho[1:-1] * factor.h_cell
            )
            banded = np.zeros((3, self.node_count - 2), dtype=float)
            banded[0, 1:] = factor.C[1:-1]
            banded[1] = -(
                factor.C[:-1] + factor.C[1:]
            ) - Q * (n[1:-1] + p[1:-1]) / self.thermal_voltage * factor.h_cell
            banded[2, :-1] = factor.C[1:-1]
            step = solve_banded((1, 1), banded, -raw)
            damping = min(
                1.0,
                0.05 / max(float(np.max(np.abs(step))), np.finfo(float).tiny),
            )
            phi[1:-1] += damping * step
            if float(np.max(np.abs(damping * step))) < self.poisson_tolerance_V:
                break
        else:
            raise QuasiFermiSteadyStateError(
                "eliminated Poisson-Boltzmann solve did not converge"
            )

        dphi = phi - self.phi0
        n = _density_from_log(
            self.log_n0 + (dqfn + dphi) / self.thermal_voltage,
            context="electron Poisson-Boltzmann solution",
        )
        p = _density_from_log(
            self.log_p0 + (dqfp - dphi) / self.thermal_voltage,
            context="hole Poisson-Boltzmann solution",
        )
        rho = _charge_density(
            p,
            n,
            self.base[2 * self.node_count : 3 * self.node_count],
            self.mat.P_ion0,
            self.mat.N_A,
            self.mat.N_D,
        )
        raw = (
            factor.C[:-1] * (phi[:-2] - phi[1:-1])
            + factor.C[1:] * (phi[2:] - phi[1:-1])
            + rho[1:-1] * factor.h_cell
        )
        scale = (factor.C[:-1] + factor.C[1:]) * self.thermal_voltage
        return (
            phi,
            n,
            p,
            float(np.max(np.abs(raw / scale))),
            float(np.max(np.abs(raw))),
        )

    def _evaluate_increments(
        self,
        dqfn: np.ndarray,
        dqfp: np.ndarray,
        illumination_fraction: float,
        *,
        V_app: float | None = None,
    ) -> _Evaluation:
        self.evaluation_count += 1
        dqfn_arr = np.asarray(dqfn, dtype=float)
        dqfp_arr = np.asarray(dqfp, dtype=float)
        if dqfn_arr.shape != (self.node_count,) or dqfp_arr.shape != (
            self.node_count,
        ):
            raise ValueError(
                "quasi-Fermi increment arrays must match the electrical grid"
            )
        voltage = self.V_app if V_app is None else float(V_app)
        phi, n, p, poisson_scaled, poisson_raw = self._solve_poisson(
            dqfn_arr,
            dqfp_arr,
            V_app=voltage,
        )

        y = self.base.copy()
        y[: self.node_count] = n
        y[self.node_count : 2 * self.node_count] = p
        source = assemble_rhs(
            0.0,
            y,
            self.x,
            self.stack,
            self.source_mat,
            illuminated=False,
            V_app=voltage,
            phi_frozen=phi,
        )[: 2 * self.node_count]
        source += float(illumination_fraction) * self.generation

        psi_n = phi + self.mat.chi
        psi_p = phi + self.mat.chi + self.mat.Eg
        xi_n = np.diff(psi_n) / self.thermal_voltage
        xi_p = np.diff(psi_p) / self.thermal_voltage
        # Keep the reference and increment differences separate. Near the DC
        # root, forming an absolute QF potential and subtracting it again loses
        # Newton-scale increments at highly doped contacts.
        delta_n = (
            np.diff(self.qfn0) + np.diff(dqfn_arr)
        ) / self.thermal_voltage
        delta_p = -(
            np.diff(self.qfp0) + np.diff(dqfp_arr)
        ) / self.thermal_voltage
        current_n = Q * self.mat.D_n_face / self.dx * self._stable_difference(
            bernoulli(xi_n) * n[1:],
            bernoulli(-xi_n) * n[:-1],
            delta_n,
        )
        current_p = Q * self.mat.D_p_face / self.dx * self._stable_difference(
            bernoulli(xi_p) * p[:-1],
            bernoulli(-xi_p) * p[1:],
            delta_p,
        )

        rate_n = source[: self.node_count] + np.diff(
            np.r_[0.0, current_n, 0.0]
        ) / (Q * self.mat.dx_cell)
        rate_p = source[self.node_count :] - np.diff(
            np.r_[0.0, current_p, 0.0]
        ) / (Q * self.mat.dx_cell)
        residual = np.r_[
            Q * rate_n * self.mat.dx_cell / self.current_scale,
            Q * rate_p * self.mat.dx_cell / self.current_scale,
        ]
        residual[self.pin] = 0.0
        return _Evaluation(
            residual=residual,
            y=y,
            phi=phi,
            rate_n=rate_n,
            rate_p=rate_p,
            current_n=current_n,
            current_p=current_p,
            poisson_residual=poisson_scaled,
            poisson_residual_C_m2=poisson_raw,
        )

    def evaluate_quasi_fermi(
        self,
        qfn: np.ndarray,
        qfp: np.ndarray,
        illumination_fraction: float,
        *,
        V_app: float | None = None,
    ) -> _Evaluation:
        """Evaluate stable rates/currents at absolute quasi-Fermi potentials.

        This compatibility interface is suitable when the requested QF
        differences are resolvable in the absolute representation. Sensitive
        small-signal paths should use ``evaluate_quasi_fermi_increments``.
        """
        qfn_arr = np.asarray(qfn, dtype=float)
        qfp_arr = np.asarray(qfp, dtype=float)
        if qfn_arr.shape != (self.node_count,) or qfp_arr.shape != (
            self.node_count,
        ):
            raise ValueError("quasi-Fermi arrays must match the electrical grid")
        return self._evaluate_increments(
            qfn_arr - self.qfn0,
            qfp_arr - self.qfp0,
            illumination_fraction,
            V_app=V_app,
        )

    def evaluate_quasi_fermi_increments(
        self,
        dqfn: np.ndarray,
        dqfp: np.ndarray,
        illumination_fraction: float,
        *,
        V_app: float | None = None,
    ) -> _Evaluation:
        """Evaluate QF increments without collapsing them into absolute values."""
        return self._evaluate_increments(
            dqfn,
            dqfp,
            illumination_fraction,
            V_app=V_app,
        )

    def evaluate(self, z: np.ndarray, illumination_fraction: float) -> _Evaluation:
        z_arr = np.asarray(z, dtype=float)
        physical = self._evaluate_increments(
            self.thermal_voltage * z_arr[: self.node_count],
            self.thermal_voltage * z_arr[self.node_count :],
            illumination_fraction,
            V_app=self.V_app,
        )
        residual = physical.residual.copy()
        residual[self.pin] = z_arr[self.pin]
        return replace(physical, residual=residual)


def _solve_newton_stage(
    system: _QuasiFermiSystem,
    z0: np.ndarray,
    illumination_fraction: float,
    *,
    finite_difference_step: float,
    residual_tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, int]:
    z = np.asarray(z0, dtype=float).copy()
    size = z.size
    for iteration in range(max_iterations + 1):
        residual = system.evaluate(z, illumination_fraction).residual
        max_residual = float(np.max(np.abs(residual)))
        if max_residual < residual_tolerance:
            return z, iteration
        if iteration == max_iterations:
            break

        jacobian = np.empty((size, size), dtype=float)
        for column in range(size):
            trial = z.copy()
            trial[column] += finite_difference_step
            jacobian[:, column] = (
                system.evaluate(trial, illumination_fraction).residual - residual
            ) / finite_difference_step
        try:
            step = np.linalg.solve(jacobian, -residual)
        except np.linalg.LinAlgError as exc:
            raise QuasiFermiSteadyStateError(
                f"singular QF Newton Jacobian at illumination={illumination_fraction:g}"
            ) from exc
        step = np.clip(step, -5.0, 5.0)
        norm = float(np.linalg.norm(residual))
        for line_search_iteration in range(30):
            damping = 0.5**line_search_iteration
            candidate = z + damping * step
            candidate_norm = float(
                np.linalg.norm(
                    system.evaluate(candidate, illumination_fraction).residual
                )
            )
            if candidate_norm < norm * (1.0 - 1.0e-4 * damping):
                z = candidate
                break
        else:
            raise QuasiFermiSteadyStateError(
                "QF Newton line search failed at "
                f"illumination={illumination_fraction:g}, "
                f"max normalized residual={max_residual:.6g}"
            )
    raise QuasiFermiSteadyStateError(
        "QF Newton iteration limit reached at "
        f"illumination={illumination_fraction:g}, "
        f"max normalized residual={max_residual:.6g}"
    )


def solve_quasi_fermi_steady_state(
    x: np.ndarray,
    stack: DeviceStack,
    V_app: float = 0.0,
    *,
    illuminated: bool = True,
    mat: MaterialArrays | None = None,
    initial_state: QuasiFermiSteadyStateResult | None = None,
    illumination_steps: tuple[float, ...] = DEFAULT_ILLUMINATION_STEPS,
    finite_difference_step: float = 1.0e-5,
    newton_residual_tolerance: float = 1.0e-10,
    max_newton_iterations: int = 30,
    poisson_tolerance_V: float = 1.0e-13,
    poisson_max_iterations: int = 100,
    continuity_tolerance_A_m2: float = 1.0e-4,
    current_spread_tolerance_A_m2: float = 1.0e-4,
    poisson_residual_tolerance: float = 1.0e-8,
) -> QuasiFermiSteadyStateResult:
    """Solve and certify the guarded local QF steady-state problem.

    ``continuity_tolerance_A_m2`` bounds the integrated absolute continuity
    defect of each carrier over unpinned nodes.  ``current_spread_tolerance``
    gates the peak-to-peak cancellation-safe total face current.  These are
    independent physical gates in addition to Newton's normalized cell
    residual tolerance. ``initial_state`` may warm-start a nearby voltage,
    but only when it already carries a physical certificate; the new voltage
    is still solved and certified independently.
    """
    grid = np.asarray(x, dtype=float)
    if grid.ndim != 1 or len(grid) < 3 or np.any(np.diff(grid) <= 0.0):
        raise ValueError("x must be a strictly increasing one-dimensional grid")
    if not np.isfinite(V_app):
        raise ValueError("V_app must be finite")
    positive_controls = {
        "finite_difference_step": finite_difference_step,
        "newton_residual_tolerance": newton_residual_tolerance,
        "poisson_tolerance_V": poisson_tolerance_V,
        "continuity_tolerance_A_m2": continuity_tolerance_A_m2,
        "current_spread_tolerance_A_m2": current_spread_tolerance_A_m2,
        "poisson_residual_tolerance": poisson_residual_tolerance,
    }
    for name, value in positive_controls.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if max_newton_iterations <= 0 or poisson_max_iterations <= 0:
        raise ValueError("iteration limits must be positive")

    stages = _validate_illumination_steps(illuminated, illumination_steps)
    material = build_material_arrays(grid, stack) if mat is None else mat
    _require_supported(material)
    system = _QuasiFermiSystem(
        grid,
        stack,
        material,
        float(V_app),
        poisson_tolerance_V=poisson_tolerance_V,
        poisson_max_iterations=poisson_max_iterations,
    )
    z = np.zeros(2 * len(grid), dtype=float)
    if initial_state is not None:
        if not initial_state.certified:
            raise ValueError("initial_state must carry a physical certificate")
        qfn = np.asarray(initial_state.electron_quasi_fermi_potential_V, dtype=float)
        qfp = np.asarray(initial_state.hole_quasi_fermi_potential_V, dtype=float)
        if qfn.shape != grid.shape or qfp.shape != grid.shape:
            raise ValueError(
                "initial_state quasi-Fermi arrays must match the target grid"
            )
        if not np.all(np.isfinite(qfn)) or not np.all(np.isfinite(qfp)):
            raise ValueError("initial_state quasi-Fermi arrays must be finite")
        initial_qfn_reference = initial_state.electron_quasi_fermi_reference_V
        initial_qfp_reference = initial_state.hole_quasi_fermi_reference_V
        initial_dqfn = initial_state.electron_quasi_fermi_increment_V
        initial_dqfp = initial_state.hole_quasi_fermi_increment_V
        split_qf = (
            initial_qfn_reference,
            initial_qfp_reference,
            initial_dqfn,
            initial_dqfp,
        )
        split_qf_present = tuple(value is not None for value in split_qf)
        if any(split_qf_present) and not all(split_qf_present):
            raise ValueError(
                "initial_state must provide all QF reference/increment arrays "
                "or none of them"
            )
        if all(split_qf_present):
            qfn_reference_arr = np.asarray(initial_qfn_reference, dtype=float)
            qfp_reference_arr = np.asarray(initial_qfp_reference, dtype=float)
            dqfn_arr = np.asarray(initial_dqfn, dtype=float)
            dqfp_arr = np.asarray(initial_dqfp, dtype=float)
            compensated = (
                qfn_reference_arr,
                qfp_reference_arr,
                dqfn_arr,
                dqfp_arr,
            )
            if any(
                value.shape != grid.shape or not np.all(np.isfinite(value))
                for value in compensated
            ):
                raise ValueError(
                    "initial_state QF reference/increment arrays must be finite "
                    "and match the target grid"
                )
            z = np.r_[
                (qfn_reference_arr - system.qfn0 + dqfn_arr)
                / system.thermal_voltage,
                (qfp_reference_arr - system.qfp0 + dqfp_arr)
                / system.thermal_voltage,
            ]
        else:
            z = np.r_[
                (qfn - system.qfn0) / system.thermal_voltage,
                (qfp - system.qfp0) / system.thermal_voltage,
            ]
        z[system.pin] = 0.0
    total_iterations = 0
    for fraction in stages:
        z, iterations = _solve_newton_stage(
            system,
            z,
            fraction,
            finite_difference_step=finite_difference_step,
            residual_tolerance=newton_residual_tolerance,
            max_iterations=max_newton_iterations,
        )
        total_iterations += iterations

    final = system.evaluate(z, stages[-1])
    interior = np.ones(len(grid), dtype=bool)
    interior[[0, -1]] = False
    electron_bound = float(
        Q * np.sum(np.abs(final.rate_n[interior]) * material.dx_cell[interior])
    )
    hole_bound = float(
        Q * np.sum(np.abs(final.rate_p[interior]) * material.dx_cell[interior])
    )
    max_normalized = float(np.max(np.abs(final.residual)))
    total_faces = -float(material.junction_polarity) * (
        final.current_n + final.current_p
    )
    # The first and last faces enter the first and last solved interior
    # control-volume equations, so terminal faces are part of the certificate.
    face_spread = float(np.ptp(total_faces))
    current = float(total_faces[0])

    diagnostics = {
        "max normalized cell residual": (
            max_normalized,
            newton_residual_tolerance,
        ),
        "electron continuity bound [A/m2]": (
            electron_bound,
            continuity_tolerance_A_m2,
        ),
        "hole continuity bound [A/m2]": (
            hole_bound,
            continuity_tolerance_A_m2,
        ),
        "face-current spread [A/m2]": (
            face_spread,
            current_spread_tolerance_A_m2,
        ),
        "normalized Poisson residual": (
            final.poisson_residual,
            poisson_residual_tolerance,
        ),
    }
    failures = [
        f"{name}={value:.6g} > {limit:.6g}"
        for name, (value, limit) in diagnostics.items()
        if not np.isfinite(value) or value > limit
    ]
    arrays = (
        final.y,
        final.phi,
        final.current_n,
        final.current_p,
        total_faces,
        final.rate_n,
        final.rate_p,
        z,
    )
    if any(not np.all(np.isfinite(value)) for value in arrays):
        failures.append("result contains non-finite state or current values")
    if failures:
        raise QuasiFermiSteadyStateError(
            "QF Newton terminated without a physical certificate: "
            + "; ".join(failures)
        )

    dqfn = system.thermal_voltage * z[: len(grid)]
    dqfp = system.thermal_voltage * z[len(grid) :]
    return QuasiFermiSteadyStateResult(
        y=final.y,
        phi=final.phi,
        electron_quasi_fermi_potential_V=system.qfn0 + dqfn,
        hole_quasi_fermi_potential_V=system.qfp0 + dqfp,
        electron_face_current_A_m2=final.current_n,
        hole_face_current_A_m2=final.current_p,
        total_face_current_A_m2=total_faces,
        electron_rate_per_s=final.rate_n,
        hole_rate_per_s=final.rate_p,
        current_A_m2=current,
        face_current_spread_A_m2=face_spread,
        electron_continuity_bound_A_m2=electron_bound,
        hole_continuity_bound_A_m2=hole_bound,
        max_normalized_cell_residual=max_normalized,
        poisson_residual=final.poisson_residual,
        poisson_residual_C_m2=final.poisson_residual_C_m2,
        illumination_steps=stages,
        newton_iterations=total_iterations,
        residual_evaluations=system.evaluation_count,
        V_app=float(V_app),
        illuminated=bool(illuminated),
        electron_quasi_fermi_reference_V=system.qfn0.copy(),
        hole_quasi_fermi_reference_V=system.qfp0.copy(),
        electron_quasi_fermi_increment_V=dqfn.copy(),
        hole_quasi_fermi_increment_V=dqfp.copy(),
    )


def solve_quasi_fermi_jv_sweep(
    x: np.ndarray,
    stack: DeviceStack,
    voltages_V: np.ndarray,
    *,
    mat: MaterialArrays | None = None,
    P_in_W_m2: float = 1000.0,
    illumination_steps: tuple[float, ...] = DEFAULT_ILLUMINATION_STEPS,
    finite_difference_step: float = 1.0e-5,
    newton_residual_tolerance: float = 1.0e-10,
    max_newton_iterations: int = 30,
    poisson_tolerance_V: float = 1.0e-13,
    poisson_max_iterations: int = 100,
    continuity_tolerance_A_m2: float = 1.0e-4,
    current_spread_tolerance_A_m2: float = 1.0e-4,
    poisson_residual_tolerance: float = 1.0e-8,
    stop_after_voc: bool = False,
) -> QuasiFermiJVSweepResult:
    """Solve a strictly increasing illuminated J-V grid by QF continuation.

    The first voltage uses the full illumination continuation. Each later
    voltage maps the preceding certified QF potentials onto its own contact
    boundary problem and solves directly at one sun. No point is retained if
    its local physical certificate fails. With ``stop_after_voc=True``, the
    sweep stops immediately after the first certified current sign change;
    the retained 0-to-Voc arc still determines all J-V figures of merit.
    """
    voltages = np.asarray(voltages_V, dtype=float)
    if (
        voltages.ndim != 1
        or voltages.size < 2
        or not np.all(np.isfinite(voltages))
        or np.any(np.diff(voltages) <= 0.0)
    ):
        raise ValueError("voltages_V must be finite and strictly increasing")
    if voltages[0] != 0.0:
        raise ValueError("voltages_V must start at 0 V for Jsc extraction")
    if not np.isfinite(P_in_W_m2) or P_in_W_m2 <= 0.0:
        raise ValueError("P_in_W_m2 must be finite and positive")

    grid = np.asarray(x, dtype=float)
    material = build_material_arrays(grid, stack) if mat is None else mat
    common = dict(
        illuminated=True,
        mat=material,
        finite_difference_step=finite_difference_step,
        newton_residual_tolerance=newton_residual_tolerance,
        max_newton_iterations=max_newton_iterations,
        poisson_tolerance_V=poisson_tolerance_V,
        poisson_max_iterations=poisson_max_iterations,
        continuity_tolerance_A_m2=continuity_tolerance_A_m2,
        current_spread_tolerance_A_m2=current_spread_tolerance_A_m2,
        poisson_residual_tolerance=poisson_residual_tolerance,
    )
    points: list[QuasiFermiSteadyStateResult] = []
    previous: QuasiFermiSteadyStateResult | None = None
    for index, voltage in enumerate(voltages):
        point = solve_quasi_fermi_steady_state(
            grid,
            stack,
            V_app=float(voltage),
            initial_state=previous,
            illumination_steps=(illumination_steps if index == 0 else (1.0,)),
            **common,
        )
        points.append(point)
        previous = point
        if (
            stop_after_voc
            and len(points) >= 2
            and points[-2].current_A_m2 > 0.0 >= point.current_A_m2
        ):
            break

    retained_voltages = voltages[: len(points)]
    currents = np.asarray([point.current_A_m2 for point in points], dtype=float)
    metrics = compute_metrics(
        retained_voltages,
        currents,
        P_in=P_in_W_m2,
        V_oc_max=thermodynamic_voc_ceiling(stack),
        validity=[point.certified for point in points],
    )
    return QuasiFermiJVSweepResult(
        voltages_V=retained_voltages.copy(),
        currents_A_m2=currents,
        points=tuple(points),
        metrics=metrics,
    )


__all__ = [
    "DEFAULT_ILLUMINATION_STEPS",
    "QuasiFermiSteadyStateError",
    "QuasiFermiJVSweepResult",
    "QuasiFermiSteadyStateResult",
    "solve_quasi_fermi_jv_sweep",
    "solve_quasi_fermi_steady_state",
]
