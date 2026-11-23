"""Assembled 1D electronic DC and small-signal AC with nonlocal reabsorption and WKB tunnelling.

Composes :class:`~solarlab.equations.assembly.classical_1d.ClassicalDC1D` with
the structured sparse nonlocal operators of
:mod:`solarlab.equations.nonlocal_1d`:

* Disabled/empty nonlocal configurations delegate directly to
  :class:`~solarlab.equations.assembly.classical_1d.AssembledDC1DProblem`,
  preserving every residual, sparse Jacobian, storage, and current array bit for
  bit.
* Active radiative reabsorption and WKB intraband electron tunnelling assemble
  their rank-1 and windowed face-divergence Jacobians directly in sparse CSC
  format with zero dense solver fallbacks.
* :func:`solve_nonlocal_small_signal_1d` enforces pre-solve differentiability
  and quasi-static capability gates: WKB tunnelling, lagged baked reabsorption,
  clamped ``R_tot = 0`` dark-equilibrium reabsorption, thermionic ``min``
  crossovers, and Poole-Frenkel cusps raise :class:`CapabilityError` rather than
  assembling an ill-defined linear AC operator.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csc_matrix, diags, eye
from scipy.sparse.linalg import splu

from solarlab.contracts.arrays import frozen_array
from solarlab.contracts.units import ELEMENTARY_CHARGE_C
from solarlab.equations.assembly.assembler import AssembledSystem
from solarlab.equations.assembly.classical_1d import (
    AssembledDC1DProblem,
    ClassicalDC1D,
    classical_dc_1d,
)
from solarlab.equations.assembly.layout import SystemLayout
from solarlab.equations.assembly.quasi_fermi import quasi_fermi_potential_tangents
from solarlab.equations.assembly.terms import OperatingInputs
from solarlab.equations.dc_1d import CapabilityError
from solarlab.equations.nonlocal_1d import (
    CompiledNonlocalDevice1D,
    SparsitySupportChangedError,
    evaluate_device_tunnelling_1d,
    evaluate_nonlocal_couplings_1d,
)
from solarlab.physics.contacts.exchange import CarrierExchangeChannel
from solarlab.physics.generation.photon_recycling import PhotonRecyclingCapabilityError
from solarlab.physics.generation.radiative_reabsorption import (
    LAGGED_BAKED_GENERATION_V1,
    PhotonRecyclingBudget,
    ReabsorptionEvaluation,
    ThermionicCapDirectionalDerivative,
    assess_switching_surfaces,
    require_smooth_ac_surfaces,
)
from solarlab.physics.recombination.bulk import RecombinationRate
from solarlab.physics.transport.electrochemical_flux import CarrierFaceFluxes
from solarlab.physics.transport.field_mobility_tangents import FieldMobilityDirectionalDerivative
from solarlab.physics.transport.tunneling_channels import TunnellingChannelEvaluation

__all__ = [
    "AssembledNonlocalDC1DProblem",
    "NonlocalClassicalDC1D",
    "NonlocalSmallSignal1D",
    "SparsitySupportChangedError",
    "assembled_nonlocal_problem_factory",
    "nonlocal_classical_dc_1d",
    "solve_nonlocal_small_signal_1d",
]

Vector: TypeAlias = NDArray[np.float64]
ComplexVector: TypeAlias = NDArray[np.complex128]
ComplexMatrix: TypeAlias = NDArray[np.complex128]
Q: float = ELEMENTARY_CHARGE_C


@dataclass(frozen=True)
class NonlocalClassicalDC1D:
    """Assembled classical 1D system paired with nonlocal coupling bindings."""

    device: CompiledNonlocalDevice1D
    classical: ClassicalDC1D = field(init=False, repr=False)
    identity: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.device, CompiledNonlocalDevice1D):
            raise TypeError("NonlocalClassicalDC1D requires a CompiledNonlocalDevice1D")
        classical = classical_dc_1d(self.device.base)
        object.__setattr__(self, "classical", classical)
        if not self.device.has_nonlocal_couplings and not self.device.escape_bindings:
            object.__setattr__(self, "identity", classical.identity)
        else:
            digest = hashlib.sha256(
                json.dumps(
                    {
                        "schema": "solarlab.nonlocal_classical_dc_1d.v1",
                        "classical": classical.identity,
                        "device": self.device.identity,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("ascii")
            ).hexdigest()
            object.__setattr__(self, "identity", digest)

    @property
    def layout(self) -> SystemLayout:
        return self.classical.layout


def nonlocal_classical_dc_1d(device: CompiledNonlocalDevice1D) -> NonlocalClassicalDC1D:
    """Bind the assembled local + nonlocal 1D electronic system of ``device``."""
    return NonlocalClassicalDC1D(device)


def assembled_nonlocal_problem_factory(
    device: CompiledNonlocalDevice1D,
) -> Callable[[CompiledNonlocalDevice1D, float, float], AssembledNonlocalDC1DProblem]:
    """Constructor-compatible factory reusing one ``NonlocalClassicalDC1D`` binding."""
    system = nonlocal_classical_dc_1d(device)

    def factory(
        bound: CompiledNonlocalDevice1D,
        voltage_V: float,
        generation_scale: float,
    ) -> AssembledNonlocalDC1DProblem:
        return AssembledNonlocalDC1DProblem(bound, voltage_V, generation_scale, system)

    return factory


@dataclass(frozen=True)
class AssembledNonlocalDC1DProblem:
    """Assembled 1D DC problem supporting nonlocal reabsorption and WKB tunnelling."""

    device: CompiledNonlocalDevice1D
    voltage_V: float = 0.0
    generation_scale: float = 0.0
    system: NonlocalClassicalDC1D | None = field(default=None, repr=False, compare=False)
    bound_wkb_signature: str | None = None

    def __post_init__(self) -> None:
        self.device.contact_potentials(self.voltage_V)
        if not math.isfinite(self.generation_scale) or self.generation_scale < 0.0:
            raise ValueError("generation_scale must be finite and non-negative")
        if self.system is None:
            object.__setattr__(self, "system", nonlocal_classical_dc_1d(self.device))
        elif self.system.device is not self.device:
            raise ValueError("the assembled nonlocal system must be bound to this device")

    @property
    def binding(self) -> NonlocalClassicalDC1D:
        assert self.system is not None
        return self.system

    @property
    def base_problem(self) -> AssembledDC1DProblem:
        return AssembledDC1DProblem(
            self.device.base,
            self.voltage_V,
            self.generation_scale,
            self.binding.classical,
        )

    @property
    def inputs(self) -> OperatingInputs:
        return OperatingInputs(self.voltage_V, self.generation_scale)

    def at(self, voltage_V: float, generation_scale: float) -> AssembledNonlocalDC1DProblem:
        """Same bound system at another operating point."""
        return AssembledNonlocalDC1DProblem(
            self.device,
            voltage_V,
            generation_scale,
            self.system,
            self.bound_wkb_signature,
        )

    def rebind_for_state(self, u: ArrayLike) -> AssembledNonlocalDC1DProblem:
        """Bind active WKB turning-point support signature at ``u``."""
        if not self.device.has_wkb_tunnelling:
            return self
        ext = evaluate_nonlocal_couplings_1d(
            self.device,
            u,
            voltage_V=self.voltage_V,
            derivatives=False,
            bound_wkb_signature=None,
        )
        sig = None if ext.wkb_support is None else ext.wkb_support.support_signature
        return AssembledNonlocalDC1DProblem(
            self.device,
            self.voltage_V,
            self.generation_scale,
            self.system,
            sig,
        )

    def _require_steady_for_wkb(self, udot: ArrayLike | None, *, alpha: float = 0.0) -> None:
        if not self.device.has_wkb_tunnelling:
            return
        if alpha != 0.0:
            raise CapabilityError(
                "wkb_transient_unsupported: WKB tunnelling channels are quasi-static steady-state only"
            )
        if udot is not None and np.any(np.asarray(udot, dtype=float) != 0.0):
            raise CapabilityError(
                "wkb_transient_unsupported: WKB tunnelling channels do not admit nonzero udot"
            )

    def assemble_local(
        self,
        u: ArrayLike,
        *,
        derivatives: bool = True,
        storage: bool = False,
    ) -> AssembledSystem:
        return self.base_problem.assemble(u, derivatives=derivatives, storage=storage)

    def residual(self, u: ArrayLike, udot: ArrayLike | None = None) -> Vector:
        self._require_steady_for_wkb(udot)
        base_f = self.base_problem.residual(u, udot)
        if not self.device.has_nonlocal_couplings:
            return base_f
        ext = evaluate_nonlocal_couplings_1d(
            self.device,
            u,
            voltage_V=self.voltage_V,
            derivatives=False,
            bound_wkb_signature=None,
        )
        return base_f + ext.residual_addition

    def jacobian(
        self,
        u: ArrayLike,
        udot: ArrayLike | None = None,
        *,
        alpha: float = 0.0,
    ) -> csc_matrix:
        self._require_steady_for_wkb(udot, alpha=alpha)
        base_j = self.base_problem.jacobian(u, udot, alpha=alpha)
        if not self.device.has_nonlocal_couplings:
            return base_j
        ext = evaluate_nonlocal_couplings_1d(
            self.device,
            u,
            voltage_V=self.voltage_V,
            derivatives=True,
            bound_wkb_signature=self.bound_wkb_signature,
        )
        assert ext.jacobian_addition is not None
        return csc_matrix(base_j + ext.jacobian_addition)

    def storage(self, u: ArrayLike) -> Vector:
        if self.device.has_wkb_tunnelling:
            raise CapabilityError(
                "wkb_transient_unsupported: storage time-stepping is unavailable with WKB tunnelling"
            )
        return self.base_problem.storage(u)

    def storage_tangent(self, u: ArrayLike) -> csc_matrix:
        if self.device.has_wkb_tunnelling:
            raise CapabilityError(
                "wkb_transient_unsupported: storage_tangent is unavailable with WKB tunnelling"
            )
        return self.base_problem.storage_tangent(u)

    def storage_jacobian(self, u: ArrayLike) -> csc_matrix:
        if self.device.has_wkb_tunnelling:
            raise CapabilityError(
                "wkb_transient_unsupported: storage_jacobian is unavailable with WKB tunnelling"
            )
        return self.base_problem.storage_jacobian(u)

    def storage_increment(
        self,
        u: ArrayLike,
        old_problem: AssembledNonlocalDC1DProblem,
        old_u: ArrayLike,
    ) -> Vector:
        if self.device.has_wkb_tunnelling:
            raise CapabilityError(
                "wkb_transient_unsupported: storage_increment is unavailable with WKB tunnelling"
            )
        if old_problem.device.identity != self.device.identity:
            raise ValueError("a storage increment needs a previous state of the same device")
        return self.base_problem.storage_increment(u, old_problem.base_problem, old_u)

    def fluxes(self, u: ArrayLike) -> CarrierFaceFluxes:
        return self.base_problem.fluxes(u)

    def recombination_rate(self, n: Vector, p: Vector, *, derivatives: bool = True) -> RecombinationRate:
        return self.base_problem.recombination_rate(n, p, derivatives=derivatives)

    def contact_exchange(
        self,
        u: ArrayLike,
        channel: CarrierExchangeChannel,
        node: int,
    ) -> tuple[float, float]:
        return self.base_problem.contact_exchange(u, channel, node)

    def current_density(self, u: ArrayLike) -> Vector:
        base_j = self.base_problem.current_density(u)
        if not self.device.has_wkb_tunnelling:
            return base_j
        ext = evaluate_nonlocal_couplings_1d(
            self.device,
            u,
            voltage_V=self.voltage_V,
            derivatives=False,
            bound_wkb_signature=self.bound_wkb_signature,
        )
        return base_j + ext.electron_face_current_addition_A_m2

    def contact_current_density(self, u: ArrayLike) -> Vector:
        base_c = self.base_problem.contact_current_density(u)
        if not self.device.has_nonlocal_couplings:
            return base_c
        ext = evaluate_nonlocal_couplings_1d(
            self.device,
            u,
            voltage_V=self.voltage_V,
            derivatives=False,
            bound_wkb_signature=self.bound_wkb_signature,
        )
        dl_n, dl_p = ext.left_contact_exchange_addition
        dr_n, dr_p = ext.right_contact_exchange_addition
        delta = np.array([-Q * dl_n + Q * dl_p, -Q * dr_n + Q * dr_p], dtype=float)
        return base_c + delta

    def photon_recycling_budgets(self, u: ArrayLike) -> tuple[PhotonRecyclingBudget, ...]:
        if not self.device.radiative_reabsorption:
            return ()
        ext = evaluate_nonlocal_couplings_1d(
            self.device, u, voltage_V=self.voltage_V, derivatives=False
        )
        assert ext.reabsorption is not None
        return ext.reabsorption.budgets

    def tunnelling_evaluation(self, u: ArrayLike) -> TunnellingChannelEvaluation:
        phi, n, p = self.device.fields(u, voltage_V=self.voltage_V)
        return evaluate_device_tunnelling_1d(self.device, phi, n, p, derivatives=False)


@dataclass(frozen=True)
class NonlocalSmallSignal1D:
    """Small-signal AC admittance and Maxwell face-current phasors for nonlocal 1D DC."""

    frequencies_Hz: Vector
    voltage_phasor_V: complex
    admittance_density_S_m2: ComplexVector
    conduction_current_density_A_m2: ComplexMatrix
    displacement_current_density_A_m2: ComplexMatrix
    total_current_density_A_m2: ComplexMatrix
    maxwell_spread_A_m2_V: Vector
    fixed_coordinate_perturbation: ComplexMatrix
    compiled_identity: str


def _particle_perturbations(
    problem: AssembledNonlocalDC1DProblem,
    u: Vector,
    x_vec: NDArray[np.float64] | ComplexVector,
    voltage_drive: complex,
    references: tuple[float, float],
) -> tuple[NDArray[np.float64] | ComplexVector, NDArray[np.float64] | ComplexVector]:
    """Cancellation-safe quasi-Fermi face-flux tangent grouping uniform energy modes."""
    device = problem.device.base
    count = device.node_count
    fluxes = problem.fluxes(u)
    transport = None if device.field_transport is None else device.field_transport.evaluate(u[:count])
    _, n, p = device.fields(u, voltage_V=problem.voltage_V)

    carrier_df: list[NDArray[np.float64] | ComplexVector] = []
    for offset, density, flux, sign, reference in (
        (count, n, fluxes.electron, 1.0, references[0]),
        (2 * count, p, fluxes.hole, -1.0, references[1]),
    ):
        drive = sign * np.diff(u[offset : offset + count])
        left_phi, _ = quasi_fermi_potential_tangents(flux, drive, sign, device.thermal_voltage_V)
        if transport is not None:
            diffusivity = (
                transport.electron_diffusivity_m2_s
                if offset == count
                else transport.hole_diffusivity_m2_s
            )
            d_diffusivity = (
                transport.electron_d_diffusivity_d_field
                if offset == count
                else transport.hole_d_diffusivity_d_field
            )
            left_phi = left_phi + (flux.flux / diffusivity) * d_diffusivity / np.diff(device.nodes_m)
        left_z = sign * flux.left_density_derivative * density[:-1]
        df = (
            left_phi * (x_vec[: count - 1] - x_vec[1:count])
            + left_z * (x_vec[offset : offset + count - 1] - x_vec[offset + 1 : offset + count])
            + sign
            * flux.flux
            * (
                x_vec[offset + 1 : offset + count]
                + (x_vec[1:count] + reference * voltage_drive) / device.thermal_voltage_V
            )
        )
        carrier_df.append(df)
    return carrier_df[0], carrier_df[1]


def _reabsorption_source_tangent(
    reabsorption: ReabsorptionEvaluation | None,
    count: int,
    dn: Vector,
    dp: Vector,
) -> Vector:
    out: Vector = np.zeros(count, dtype=float)
    if reabsorption is None:
        return out
    for factor in reabsorption.factors:
        if factor.active:
            emit = factor.emitter_nodes
            d_flux = float(
                np.dot(factor.d_emission_flux_dn_m_s, dn[emit])
                + np.dot(factor.d_emission_flux_dp_m_s, dp[emit])
            )
            out[factor.receiver_nodes] += factor.receiver_vector_m1 * d_flux
    return out


def solve_nonlocal_small_signal_1d(
    problem: AssembledNonlocalDC1DProblem,
    u: ArrayLike,
    frequencies_Hz: Sequence[float] | ArrayLike,
    *,
    voltage_phasor_V: complex = 1.0 + 0.0j,
    thermionic_electron: ThermionicCapDirectionalDerivative | None = None,
    thermionic_hole: ThermionicCapDirectionalDerivative | None = None,
    field_mobility_electron: FieldMobilityDirectionalDerivative | None = None,
    field_mobility_hole: FieldMobilityDirectionalDerivative | None = None,
) -> NonlocalSmallSignal1D:
    """Assemble and solve ``(J + i*omega*M) x = -polarity * e_{N-1} * V`` at smooth points."""
    if problem.device.has_wkb_tunnelling:
        raise CapabilityError(
            "wkb_ac_unsupported: WKB tunnelling channels are certified only for quasi-static steady-state DC"
        )
    for spec in problem.device.radiative_reabsorption:
        if spec.mode == LAGGED_BAKED_GENERATION_V1:
            raise CapabilityError(
                "lagged_reabsorption_ac_unsupported: lagged baked reabsorption lacks live AC derivatives"
            )

    freqs = frozen_array(frequencies_Hz)
    if freqs.ndim != 1 or freqs.size == 0 or np.any(freqs < 0.0):
        raise ValueError("frequencies_Hz must be a non-empty 1D array of non-negative frequencies")
    v_phasor = complex(voltage_phasor_V)
    if not np.isfinite(v_phasor) or v_phasor == 0.0:
        raise ValueError("voltage_phasor_V must be finite and nonzero")

    values = np.asarray(u, dtype=float)
    ext = evaluate_nonlocal_couplings_1d(
        problem.device,
        values,
        voltage_V=problem.voltage_V,
        derivatives=True,
        bound_wkb_signature=problem.bound_wkb_signature,
    )
    assessment = assess_switching_surfaces(
        reabsorption=ext.reabsorption,
        thermionic_electron=thermionic_electron,
        thermionic_hole=thermionic_hole,
        field_mobility_electron=field_mobility_electron,
        field_mobility_hole=field_mobility_hole,
    )
    try:
        require_smooth_ac_surfaces(assessment)
    except PhotonRecyclingCapabilityError as exc:
        raise CapabilityError(str(exc)) from exc

    comp = problem.device.base
    count = comp.node_count
    polarity = comp.voltage_polarity
    _, n, p = comp.fields(values, voltage_V=problem.voltage_V)
    before = comp.carrier_reference_energies(0.0)
    after = comp.carrier_reference_energies(1.0)
    references = (after[0] - before[0], after[1] - before[1])
    c_ref: Vector = np.r_[
        np.zeros(count, dtype=float),
        np.full(count, -references[0] / comp.thermal_voltage_V, dtype=float),
        np.full(count, -references[1] / comp.thermal_voltage_V, dtype=float),
    ]
    lift: Vector = np.zeros(3 * count, dtype=float)
    lift[count - 1] = -polarity
    for carrier, offset in (("n", count), ("p", 2 * count)):
        for side, node in (("left", 0), ("right", count - 1)):
            if comp.channel(carrier, side).kind == "fixed_reservoir":
                lift[offset + node] = -lift[node] / comp.thermal_voltage_V + c_ref[offset + node]

    rate = problem.recombination_rate(n, p)
    assert rate.d_rate_dn is not None and rate.d_rate_dp is not None
    dn_lift = n * ((lift[:count] + references[0]) / comp.thermal_voltage_V + lift[count : 2 * count])
    dp_lift = -p * ((lift[:count] + references[1]) / comp.thermal_voltage_V + lift[2 * count :])
    src_lift = -(rate.d_rate_dn * dn_lift + rate.d_rate_dp * dp_lift) + _reabsorption_source_tangent(
        ext.reabsorption, count, dn_lift, dp_lift
    )
    fn_lift, fp_lift = _particle_perturbations(problem, values, lift, 1.0, references)
    partial: Vector = np.zeros(3 * count, dtype=float)
    partial[:count] = (
        comp.poisson.jacobian() @ lift[:count]
        + comp.poisson.charge_jacobian() * Q * (dp_lift - dn_lift)
    )
    partial[count - 1] += polarity
    for carrier, offset, d_der, face_flux, sign, ref in (
        ("n", count, dn_lift, np.asarray(fn_lift, dtype=float), 1.0, references[0]),
        ("p", 2 * count, dp_lift, np.asarray(fp_lift, dtype=float), -1.0, references[1]),
    ):
        rows = -np.diff(np.r_[0.0, face_flux, 0.0]) + comp.geometry.cell_measures_m * src_lift
        for side, node in (("left", 0), ("right", count - 1)):
            ch = comp.channel(carrier, side)
            if ch.kind == "fixed_reservoir":
                rows[node] = sign * (
                    (lift[node] + ref) / comp.thermal_voltage_V + lift[offset + node]
                )
            else:
                _, slope = problem.base_problem.contact_exchange(values, ch, node)
                rows[node] += (1.0 if side == "left" else -1.0) * slope * d_der[node]
        partial[offset : offset + count] = rows

    rhs_rate: Vector = np.zeros(3 * count, dtype=float)
    for carrier, offset, density, sign, ref in (
        ("n", count, n, 1.0, references[0]),
        ("p", 2 * count, p, -1.0, references[1]),
    ):
        rhs_rate[offset : offset + count] = (
            comp.geometry.cell_measures_m
            * sign
            * density
            * ((lift[:count] + ref) / comp.thermal_voltage_V + lift[offset : offset + count])
        )
        for side, node in (("left", 0), ("right", count - 1)):
            if comp.channel(carrier, side).kind == "fixed_reservoir":
                rhs_rate[offset + node] = 0.0

    j_mat = problem.jacobian(values)
    m_mat = problem.storage_jacobian(values)
    p_lil = eye(3 * count, format="lil")
    anchors: list[int] = []
    for carrier, offset, preferred in (
        ("n", count, "right" if polarity > 0.0 else "left"),
        ("p", 2 * count, "left" if polarity > 0.0 else "right"),
    ):
        fixed_sides = [
            side for side in ("left", "right") if comp.channel(carrier, side).kind == "fixed_reservoir"
        ]
        chosen_side = fixed_sides[0] if fixed_sides else preferred
        anc = offset + (0 if chosen_side == "left" else count - 1)
        anchors.append(anc)
        for r_idx in range(offset, offset + count):
            if anc not in p_lil.rows[r_idx]:
                p_lil.rows[r_idx].append(anc)
                p_lil.data[r_idx].append(1.0)
    p_transform = p_lil.tocsc()
    solve_j_lil = (j_mat @ p_transform).tolil()
    solve_m = (m_mat @ p_transform).tocsc()
    fluxes = problem.fluxes(values)
    for carrier, anc in zip(("n", "p"), anchors):
        dn_u = n if carrier == "n" else np.zeros(count, dtype=float)
        dp_u = -p if carrier == "p" else np.zeros(count, dtype=float)
        src_u = -(rate.d_rate_dn * dn_u + rate.d_rate_dp * dp_u) + _reabsorption_source_tangent(
            ext.reabsorption, count, dn_u, dp_u
        )
        unif: Vector = np.zeros(3 * count, dtype=float)
        unif[:count] = comp.poisson.charge_jacobian() * Q * (dp_u - dn_u)
        for cr, offset, dd, flx, sign in (
            ("n", count, dn_u, fluxes.electron, 1.0),
            ("p", 2 * count, dp_u, fluxes.hole, -1.0),
        ):
            fc = sign * flx.flux if cr == carrier else np.zeros(count - 1, dtype=float)
            rws = -np.diff(np.r_[0.0, fc, 0.0]) + comp.geometry.cell_measures_m * src_u
            for side, node in (("left", 0), ("right", count - 1)):
                ch = comp.channel(cr, side)
                if ch.kind == "fixed_reservoir":
                    rws[node] = sign if cr == carrier else 0.0
                else:
                    _, slope = problem.base_problem.contact_exchange(values, ch, node)
                    rws[node] += (1.0 if side == "left" else -1.0) * slope * dd[node]
            unif[offset : offset + count] = rws
        solve_j_lil[:, anc] = unif[:, None]
    solve_j = solve_j_lil.tocsc()

    n_freq = freqs.size
    n_faces = count - 1
    x_all: ComplexMatrix = np.empty((n_freq, 3 * count), dtype=np.complex128)
    j_cond_all: ComplexMatrix = np.empty((n_freq, n_faces), dtype=np.complex128)
    j_disp_all: ComplexMatrix = np.empty((n_freq, n_faces), dtype=np.complex128)
    j_tot_all: ComplexMatrix = np.empty((n_freq, n_faces), dtype=np.complex128)
    admittance: ComplexVector = np.empty(n_freq, dtype=np.complex128)
    spread: Vector = np.empty(n_freq, dtype=float)

    cap = comp.poisson.face_capacitance_F_m2
    for idx, f_hz in enumerate(freqs):
        omega = 2.0 * math.pi * float(f_hz)
        sys_mat = csc_matrix(
            solve_j.astype(np.complex128) + (1j * omega) * solve_m.astype(np.complex128)
        )
        rhs_vec: ComplexVector = (-partial + (1j * omega) * rhs_rate) * v_phasor
        row_max: Vector = np.zeros(sys_mat.shape[0], dtype=float)
        np.maximum.at(row_max, sys_mat.indices, np.abs(sys_mat.data))
        inv_row = 1.0 / np.where(row_max > 0.0, row_max, 1.0)
        lu = splu(diags(inv_row).dot(sys_mat).tocsc())
        eta: ComplexVector = np.asarray(lu.solve(inv_row * rhs_vec), dtype=np.complex128)
        for _ in range(2):
            eta = np.asarray(
                eta - lu.solve(inv_row * (sys_mat.dot(eta) - rhs_vec)),
                dtype=np.complex128,
            )
        moving_x: ComplexVector = np.asarray(p_transform @ eta + lift * v_phasor, dtype=np.complex128)
        fixed_x: ComplexVector = np.asarray(moving_x - c_ref * v_phasor, dtype=np.complex128)
        x_all[idx] = fixed_x
        fn_ac, fp_ac = _particle_perturbations(problem, values, moving_x, v_phasor, references)
        jc = np.asarray(-Q * fn_ac + Q * fp_ac, dtype=np.complex128)
        jd = np.asarray(-(1j * omega) * cap * (fixed_x[1:count] - fixed_x[: count - 1]), dtype=np.complex128)
        jt = jc + jd
        j_cond_all[idx] = jc
        j_disp_all[idx] = jd
        j_tot_all[idx] = jt
        mean_jt = complex(np.mean(jt))
        # Positive applied voltage lowers phi_right by polarity*V, driving +polarity current along +x.
        admittance[idx] = (polarity * mean_jt) / v_phasor
        spread[idx] = float(np.max(np.abs(jt - mean_jt)) / abs(v_phasor))

    return NonlocalSmallSignal1D(
        frequencies_Hz=freqs,
        voltage_phasor_V=v_phasor,
        admittance_density_S_m2=admittance,
        conduction_current_density_A_m2=j_cond_all,
        displacement_current_density_A_m2=j_disp_all,
        total_current_density_A_m2=j_tot_all,
        maxwell_spread_A_m2_V=frozen_array(spread),
        fixed_coordinate_perturbation=x_all,
        compiled_identity=problem.device.identity,
    )
