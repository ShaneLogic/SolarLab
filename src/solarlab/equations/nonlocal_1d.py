"""Ion-free 1D DC equations extended with nonlocal radiative reabsorption and WKB tunnelling.

Wraps :class:`~solarlab.equations.dc_1d.CompiledDevice1D` and
:class:`~solarlab.equations.dc_1d.DC1DProblem` without mutating baseline local
operators:

* When neither ``radiative_reabsorption`` nor ``tunnelling_channels`` is active,
  ``residual``, ``jacobian``, ``storage_jacobian``, ``fluxes``,
  ``current_density``, and ``contact_current_density`` delegate directly to
  :class:`~solarlab.equations.dc_1d.DC1DProblem` and reproduce every array bit
  for bit.
* When nonlocal radiative reabsorption is active, each absorber's rank-1
  outer-product coupling ``w_recv * u_recv (x) v_emit`` is assembled into sparse
  CSC format on unconstrained carrier balance rows (with constrained contact
  reaction fluxes preserved in ``contact_current_density``).
* When WKB intraband electron tunnelling is active, conservative face fluxes
  ``F_tun`` and their exact analytical tangents wrt ``(phi, z_n)`` are assembled
  under an explicit sparsity-support contract
  (``PREALLOCATED_RESERVOIR_SUPERSET_V1`` or ``STRICT_WINDOW_REBIND_V1``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import coo_matrix, csc_matrix

from solarlab.contracts.arrays import frozen_array
from solarlab.contracts.units import ELEMENTARY_CHARGE_C
from solarlab.equations.dc_1d import (
    CapabilityError,
    CompiledDevice1D,
    DC1DProblem,
    State1D,
)
from solarlab.mesh.geometry import ControlVolumeGeometry1D
from solarlab.physics.band_structure.binding import ParameterBinding
from solarlab.physics.contacts import CarrierExchangeChannel
from solarlab.physics.electrostatics import PoissonOperator1D
from solarlab.physics.electrostatics.charge import ChargeLedger
from solarlab.physics.generation.photon_recycling import AbsorberEscapeBinding
from solarlab.physics.generation.radiative_reabsorption import (
    AbsorberReabsorptionSpec,
    PhotonRecyclingBudget,
    ReabsorptionEvaluation,
    evaluate_radiative_reabsorption,
)
from solarlab.physics.recombination.bulk import RecombinationRate
from solarlab.physics.transport.electrochemical_flux import CarrierFaceFluxes
from solarlab.physics.transport.tunneling_channels import (
    PREALLOCATED_RESERVOIR_SUPERSET_V1,
    STRICT_WINDOW_REBIND_V1,
    WKB_SPARSITY_POLICIES,
    CompiledTunnellingChannels,
    TunnellingChannelCapabilityError,
    TunnellingChannelEvaluation,
    WKBWindowSupport,
    evaluate_tunnelling_channels,
    physical_quasi_fermi_levels_eV,
)

__all__ = [
    "CompiledNonlocalDevice1D",
    "NonlocalCouplingEvaluation1D",
    "NonlocalDC1DProblem",
    "SparsitySupportChangedError",
    "evaluate_device_tunnelling_1d",
    "evaluate_nonlocal_couplings_1d",
]

Vector: TypeAlias = NDArray[np.float64]
Q: float = ELEMENTARY_CHARGE_C


class SparsitySupportChangedError(CapabilityError):
    """Active WKB turning-point support shifted under ``STRICT_WINDOW_REBIND_V1``."""

    def __init__(self, message: str, *, state: ArrayLike | None = None) -> None:
        super().__init__(message)
        self.state: Vector | None = None if state is None else frozen_array(state)


@dataclass(frozen=True)
class CompiledNonlocalDevice1D:
    """Cold-compiled 1D device with optional nonlocal photon recycling and WKB tunnelling."""

    base: CompiledDevice1D
    escape_bindings: tuple[AbsorberEscapeBinding, ...] = ()
    radiative_reabsorption: tuple[AbsorberReabsorptionSpec, ...] = ()
    tunnelling_channels: CompiledTunnellingChannels | None = None
    wkb_sparsity_policy: str = PREALLOCATED_RESERVOIR_SUPERSET_V1
    capabilities: tuple[str, ...] = ()
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.base, CompiledDevice1D):
            raise TypeError("base must be a CompiledDevice1D")
        if self.wkb_sparsity_policy not in WKB_SPARSITY_POLICIES:
            raise CapabilityError(
                f"unknown wkb_sparsity_policy {self.wkb_sparsity_policy!r}; expected {WKB_SPARSITY_POLICIES}"
            )
        escapes = tuple(self.escape_bindings)
        reabs = tuple(self.radiative_reabsorption)
        for spec in reabs:
            if spec.node_count != self.base.node_count:
                raise ValueError(
                    f"reabsorption spec {spec.layer_id!r} node_count {spec.node_count} != {self.base.node_count}"
                )
        tun = self.tunnelling_channels
        if tun is not None and tun.node_count != self.base.node_count:
            raise ValueError(
                f"tunnelling_channels node_count {tun.node_count} != {self.base.node_count}"
            )

        caps = list(self.capabilities) if self.capabilities else list(self.base.capabilities)
        if escapes and "photon_recycling_escape" not in caps:
            caps.append("photon_recycling_escape")
        if reabs and "nonlocal_radiative_reabsorption" not in caps:
            caps.append("nonlocal_radiative_reabsorption")
        if tun is not None and tun.any_enabled and "wkb_intraband_electron_steady" not in caps:
            caps.append("wkb_intraband_electron_steady")

        object.__setattr__(self, "escape_bindings", escapes)
        object.__setattr__(self, "radiative_reabsorption", reabs)
        object.__setattr__(self, "capabilities", tuple(caps))

        if not escapes and not reabs and (tun is None or not tun.any_enabled):
            object.__setattr__(self, "sha256", self.base.sha256)
        else:
            payload = {
                "schema": "solarlab.compiled_nonlocal_device_1d.v1",
                "base_sha256": self.base.sha256,
                "escape_sha256": [item.sha256 for item in escapes],
                "reabsorption_sha256": [item.sha256 for item in reabs],
                "tunnelling_sha256": None if tun is None else tun.digest(),
                "wkb_sparsity_policy": self.wkb_sparsity_policy,
            }
            digest = hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
            ).hexdigest()
            object.__setattr__(self, "sha256", digest)

    @property
    def has_nonlocal_couplings(self) -> bool:
        return bool(self.radiative_reabsorption) or (
            self.tunnelling_channels is not None and self.tunnelling_channels.any_enabled
        )

    @property
    def has_wkb_tunnelling(self) -> bool:
        return self.tunnelling_channels is not None and self.tunnelling_channels.any_enabled

    @property
    def device_id(self) -> str:
        return self.base.device_id

    @property
    def identity(self) -> str:
        return self.sha256

    @property
    def node_count(self) -> int:
        return self.base.node_count

    @property
    def nodes_m(self) -> Vector:
        return self.base.nodes_m

    @property
    def geometry(self) -> ControlVolumeGeometry1D:
        return self.base.geometry

    @property
    def parameters(self) -> ParameterBinding:
        return self.base.parameters

    @property
    def poisson(self) -> PoissonOperator1D:
        return self.base.poisson

    @property
    def thermal_voltage_V(self) -> float:
        return self.base.thermal_voltage_V

    @property
    def voltage_polarity(self) -> float:
        return self.base.voltage_polarity

    @property
    def generation_limit_A_m2(self) -> float:
        return self.base.generation_limit_A_m2

    def contact_potentials(self, voltage_V: float) -> tuple[float, float]:
        return self.base.contact_potentials(voltage_V)

    def channel(self, carrier: str, side: str) -> CarrierExchangeChannel:
        return self.base.channel(carrier, side)

    def carrier_reference_energies(self, voltage_V: float) -> tuple[float, float]:
        return self.base.carrier_reference_energies(voltage_V)

    def carrier_reference_potentials(self, voltage_V: float) -> tuple[Vector, Vector]:
        return self.base.carrier_reference_potentials(voltage_V)

    def rebase_coordinates(self, u: ArrayLike, old_voltage_V: float, new_voltage_V: float) -> Vector:
        return self.base.rebase_coordinates(u, old_voltage_V, new_voltage_V)

    def fields(self, u: ArrayLike, *, voltage_V: float = 0.0) -> tuple[Vector, Vector, Vector]:
        return self.base.fields(u, voltage_V=voltage_V)

    def state(self, u: ArrayLike, *, voltage_V: float = 0.0) -> State1D:
        phi, n, p = self.base.fields(u, voltage_V=voltage_V)
        return State1D(np.asarray(u, dtype=float), phi, n, p, self.sha256, voltage_V)

    def charge_ledger(self, n: Vector, p: Vector) -> ChargeLedger:
        return self.base.charge_ledger(n, p)

    def require_driver_capability(self, driver: str) -> None:
        """Enforce quasi-static driver eligibility pre-solve."""
        if self.has_wkb_tunnelling and driver != "quasi_fermi_steady_state":
            raise CapabilityError(
                f"wkb_driver_unsupported: WKB tunnelling channels are certified only for "
                f"'quasi_fermi_steady_state', not {driver!r}"
            )


@dataclass(frozen=True)
class NonlocalCouplingEvaluation1D:
    """Evaluated residual, sparse Jacobian, and terminal flux additions of nonlocal terms."""

    residual_addition: Vector
    left_contact_exchange_addition: tuple[float, float]
    right_contact_exchange_addition: tuple[float, float]
    electron_face_flux_addition_m2_s: Vector
    electron_face_current_addition_A_m2: Vector
    jacobian_addition: csc_matrix | None
    reabsorption: ReabsorptionEvaluation | None
    tunnelling: TunnellingChannelEvaluation | None
    wkb_support: WKBWindowSupport | None


def evaluate_device_tunnelling_1d(
    device: CompiledNonlocalDevice1D,
    phi: ArrayLike,
    n: ArrayLike,
    p: ArrayLike,
    *,
    derivatives: bool = False,
) -> TunnellingChannelEvaluation:
    """Evaluate the compiled WKB tunnelling channels on ``device`` at ``(phi, n, p)``."""
    if device.tunnelling_channels is None or not device.tunnelling_channels.any_enabled:
        raise CapabilityError("no enabled WKB tunnelling channels are bound on this device")
    bands = device.parameters.band_structure
    try:
        efn, efp = physical_quasi_fermi_levels_eV(
            potential_V=phi,
            affinity_eV=bands.physical_electron_affinity_eV,
            band_gap_eV=bands.physical_band_gap_eV,
            electron_density_m3=n,
            hole_density_m3=p,
            conduction_dos_m3=device.parameters.conduction_dos_m3,
            valence_dos_m3=device.parameters.valence_dos_m3,
            thermal_voltage_V=device.thermal_voltage_V,
        )
        return evaluate_tunnelling_channels(
            device.tunnelling_channels,
            positions_m=device.nodes_m,
            potential_V=phi,
            affinity_eV=bands.physical_electron_affinity_eV,
            band_gap_eV=bands.physical_band_gap_eV,
            electron_quasi_fermi_eV=efn,
            hole_quasi_fermi_eV=efp,
            thermal_voltage_V=device.thermal_voltage_V,
            derivatives=derivatives,
        )
    except TunnellingChannelCapabilityError as exc:
        raise CapabilityError(str(exc)) from exc


def _fixed_reservoir_mask(device: CompiledDevice1D, carrier: str) -> tuple[bool, bool]:
    return (
        device.channel(carrier, "left").kind == "fixed_reservoir",
        device.channel(carrier, "right").kind == "fixed_reservoir",
    )


def evaluate_nonlocal_couplings_1d(
    device: CompiledNonlocalDevice1D,
    u: ArrayLike,
    *,
    voltage_V: float,
    derivatives: bool,
    bound_wkb_signature: str | None = None,
) -> NonlocalCouplingEvaluation1D:
    """Evaluate nonlocal radiative reabsorption and WKB path-flux contributions."""
    count = device.node_count
    values = np.asarray(u, dtype=float)
    phi, n, p = device.fields(values, voltage_V=voltage_V)
    widths = device.geometry.cell_measures_m
    vt = device.thermal_voltage_V

    res_add: Vector = np.zeros(3 * count, dtype=float)
    f_tun_n: Vector = np.zeros(count - 1, dtype=float)
    j_tun_n: Vector = np.zeros(count - 1, dtype=float)
    left_ex_n = 0.0
    left_ex_p = 0.0
    right_ex_n = 0.0
    right_ex_p = 0.0

    fixed_n_left, fixed_n_right = _fixed_reservoir_mask(device.base, "n")
    fixed_p_left, fixed_p_right = _fixed_reservoir_mask(device.base, "p")

    rows_list: list[NDArray[np.intp]] = []
    cols_list: list[NDArray[np.intp]] = []
    data_list: list[Vector] = []

    reabs_eval: ReabsorptionEvaluation | None = None
    if device.radiative_reabsorption:
        reabs_eval = evaluate_radiative_reabsorption(
            device.radiative_reabsorption, n, p, derivatives=derivatives
        )
        pair_add = widths * reabs_eval.generation_m3_s
        n_rows = pair_add.copy()
        p_rows = pair_add.copy()
        if fixed_n_left:
            n_rows[0] = 0.0
            left_ex_n -= float(pair_add[0])
        if fixed_n_right:
            n_rows[-1] = 0.0
            right_ex_n += float(pair_add[-1])
        if fixed_p_left:
            p_rows[0] = 0.0
            left_ex_p -= float(pair_add[0])
        if fixed_p_right:
            p_rows[-1] = 0.0
            right_ex_p += float(pair_add[-1])
        res_add[count : 2 * count] += n_rows
        res_add[2 * count : 3 * count] += p_rows

        if derivatives:
            for factor in reabs_eval.factors:
                if not factor.active:
                    continue
                emit = factor.emitter_nodes
                # In quasi-Fermi coordinates: dn/dz_n = +n, dp/dz_p = -p,
                # and d(n*p)/dphi = (p*n - n*p)/V_T == 0 (kept explicit via general formula).
                vn_z = factor.d_emission_flux_dn_m_s * n[emit]
                vp_z = -factor.d_emission_flux_dp_m_s * p[emit]
                v_phi = (
                    factor.d_emission_flux_dn_m_s * n[emit]
                    - factor.d_emission_flux_dp_m_s * p[emit]
                ) / vt

                for carrier_offset, fixed_l, fixed_r in (
                    (count, fixed_n_left, fixed_n_right),
                    (2 * count, fixed_p_left, fixed_p_right),
                ):
                    keep = np.ones(factor.receiver_nodes.size, dtype=bool)
                    if fixed_l:
                        keep &= factor.receiver_nodes != 0
                    if fixed_r:
                        keep &= factor.receiver_nodes != (count - 1)
                    if not np.any(keep):
                        continue
                    recv_kept = factor.receiver_nodes[keep]
                    u_weighted = widths[recv_kept] * factor.receiver_vector_m1[keep]

                    r_grid: NDArray[np.intp] = np.repeat(carrier_offset + recv_kept, emit.size)
                    c_zn = np.tile(count + emit, recv_kept.size)
                    c_zp = np.tile(2 * count + emit, recv_kept.size)
                    rows_list.append(r_grid)
                    cols_list.append(c_zn)
                    data_list.append(np.outer(u_weighted, vn_z).ravel())
                    rows_list.append(r_grid)
                    cols_list.append(c_zp)
                    data_list.append(np.outer(u_weighted, vp_z).ravel())
                    if np.any(v_phi != 0.0):
                        c_phi = np.tile(emit, recv_kept.size)
                        rows_list.append(r_grid)
                        cols_list.append(c_phi)
                        data_list.append(np.outer(u_weighted, v_phi).ravel())

    tun_eval: TunnellingChannelEvaluation | None = None
    wkb_support: WKBWindowSupport | None = None
    if device.has_wkb_tunnelling:
        tun_eval = evaluate_device_tunnelling_1d(device, phi, n, p, derivatives=derivatives)
        assert tun_eval.intraband_path is not None
        wkb_support = tun_eval.intraband_path.support
        if (
            device.wkb_sparsity_policy == STRICT_WINDOW_REBIND_V1
            and bound_wkb_signature is not None
            and wkb_support is not None
            and wkb_support.support_signature != bound_wkb_signature
        ):
            raise SparsitySupportChangedError(
                "wkb_support_shifted: turning-point window changed from bound signature "
                f"{bound_wkb_signature[:12]} to {wkb_support.support_signature[:12]}; "
                "call rebind_for_state(u) or use PREALLOCATED_RESERVOIR_SUPERSET_V1",
                state=values,
            )

        f_tun_n = np.asarray(tun_eval.electron_face_flux_m2_s, dtype=float)
        j_tun_n = np.asarray(tun_eval.electron_face_current_A_m2, dtype=float)
        transfer_n = np.array(tun_eval.electron_nodal_transfer_m2_s, copy=True)
        if fixed_n_left:
            transfer_n[0] = 0.0
            left_ex_n += float(f_tun_n[0])
        if fixed_n_right:
            transfer_n[-1] = 0.0
            right_ex_n += float(f_tun_n[-1])
        res_add[count : 2 * count] += transfer_n

        if derivatives:
            assert tun_eval.linearization is not None
            assert wkb_support is not None
            if device.wkb_sparsity_policy == PREALLOCATED_RESERVOIR_SUPERSET_V1:
                f_lo, f_hi = wkb_support.reservoir_face_slice
                n_lo, n_hi = wkb_support.reservoir_node_slice
            else:
                f_lo, f_hi = wkb_support.active_face_bounds
                n_lo, n_hi = wkb_support.active_node_bounds

            # Verify zero nonzeros lie outside the declared support slice!
            d_bar = tun_eval.linearization.d_face_flux_d_barrier_eV
            d_qf = tun_eval.linearization.d_face_flux_d_quasi_fermi_eV
            if f_hi > f_lo and n_hi > n_lo:
                outside_mask = np.ones(d_bar.shape, dtype=bool)
                outside_mask[f_lo:f_hi, n_lo:n_hi] = False
                if np.any(d_bar[outside_mask] != 0.0) or np.any(d_qf[outside_mask] != 0.0):
                    raise SparsitySupportChangedError(
                        "WKB analytical Jacobian produced nonzero entries outside declared support"
                    )
            elif np.any(d_bar != 0.0) or np.any(d_qf != 0.0):
                raise SparsitySupportChangedError(
                    "WKB analytical Jacobian produced nonzero entries on an empty support window"
                )

            if f_hi > f_lo and n_hi > n_lo:
                # In quasi-Fermi coordinates (phi, z_n):
                # U = -phi - chi => dU/dphi = -1, dU/dz_n = 0
                # E_Fn = -phi - chi + V_T*(shift_n + z_n + ln(n_ref/N_c)) => dE_Fn/dphi = 0, dE_Fn/dz_n = V_T.
                df_dphi = -d_bar[f_lo:f_hi, n_lo:n_hi]
                df_dzn = vt * d_qf[f_lo:f_hi, n_lo:n_hi]

                # Nodal divergence: transfer[node] = sum_{f} (-delta_{node, f} + delta_{node, f+1}) * F[f]
                node_span = f_hi - f_lo + 1
                div_dphi: NDArray[np.float64] = np.zeros((node_span, n_hi - n_lo), dtype=float)
                div_dzn: NDArray[np.float64] = np.zeros((node_span, n_hi - n_lo), dtype=float)
                div_dphi[:-1, :] -= df_dphi
                div_dphi[1:, :] += df_dphi
                div_dzn[:-1, :] -= df_dzn
                div_dzn[1:, :] += df_dzn

                row_nodes: NDArray[np.intp] = np.arange(f_lo, f_hi + 1, dtype=np.intp)
                col_nodes: NDArray[np.intp] = np.arange(n_lo, n_hi, dtype=np.intp)
                keep_rows: NDArray[np.bool_] = np.ones(row_nodes.size, dtype=bool)
                if fixed_n_left:
                    keep_rows &= row_nodes != 0
                if fixed_n_right:
                    keep_rows &= row_nodes != (count - 1)
                if np.any(keep_rows):
                    kept_r = count + row_nodes[keep_rows]
                    r_mat: NDArray[np.intp] = np.repeat(kept_r, col_nodes.size)
                    c_phi_mat = np.tile(col_nodes, kept_r.size)
                    c_zn_mat = np.tile(count + col_nodes, kept_r.size)
                    rows_list.append(r_mat)
                    cols_list.append(c_phi_mat)
                    data_list.append(div_dphi[keep_rows, :].ravel())
                    rows_list.append(r_mat)
                    cols_list.append(c_zn_mat)
                    data_list.append(div_dzn[keep_rows, :].ravel())

    jac_csc: csc_matrix | None = None
    if derivatives:
        if rows_list:
            all_r = np.concatenate(rows_list)
            all_c = np.concatenate(cols_list)
            all_d = np.concatenate(data_list)
            nz = all_d != 0.0
            jac_csc = coo_matrix(
                (all_d[nz], (all_r[nz], all_c[nz])),
                shape=(3 * count, 3 * count),
                dtype=float,
            ).tocsc()
        else:
            jac_csc = csc_matrix((3 * count, 3 * count), dtype=float)

    return NonlocalCouplingEvaluation1D(
        residual_addition=res_add,
        left_contact_exchange_addition=(left_ex_n, left_ex_p),
        right_contact_exchange_addition=(right_ex_n, right_ex_p),
        electron_face_flux_addition_m2_s=f_tun_n,
        electron_face_current_addition_A_m2=j_tun_n,
        jacobian_addition=jac_csc,
        reabsorption=reabs_eval,
        tunnelling=tun_eval,
        wkb_support=wkb_support,
    )


@dataclass(frozen=True)
class NonlocalDC1DProblem:
    """Direct 1D steady-state problem with nonlocal photon recycling and WKB tunnelling."""

    device: CompiledNonlocalDevice1D
    voltage_V: float = 0.0
    generation_scale: float = 0.0
    bound_wkb_signature: str | None = None

    def __post_init__(self) -> None:
        self.device.contact_potentials(self.voltage_V)
        if not math.isfinite(self.generation_scale) or self.generation_scale < 0.0:
            raise ValueError("generation_scale must be finite and non-negative")

    @property
    def base_problem(self) -> DC1DProblem:
        return DC1DProblem(self.device.base, self.voltage_V, self.generation_scale)

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

    def fluxes(self, u: ArrayLike) -> CarrierFaceFluxes:
        return self.base_problem.fluxes(u)

    def recombination_rate(self, n: Vector, p: Vector, *, derivatives: bool = True) -> RecombinationRate:
        return self.base_problem.recombination_rate(n, p, derivatives=derivatives)

    def storage_jacobian(self, u: ArrayLike) -> csc_matrix:
        if self.device.has_wkb_tunnelling:
            raise CapabilityError(
                "wkb_transient_unsupported: storage_jacobian is unavailable with WKB tunnelling"
            )
        return self.base_problem.storage_jacobian(u)

    def residual(self, u: ArrayLike, udot: ArrayLike | None = None) -> Vector:
        self._require_steady_for_wkb(udot)
        base_f = self.base_problem.residual(u, udot)
        if not self.device.has_nonlocal_couplings or not np.all(np.isfinite(base_f)):
            return base_f
        ext = evaluate_nonlocal_couplings_1d(
            self.device,
            u,
            voltage_V=self.voltage_V,
            derivatives=False,
            bound_wkb_signature=self.bound_wkb_signature,
        )
        return base_f + ext.residual_addition

    def jacobian(self, u: ArrayLike, udot: ArrayLike | None = None, *, alpha: float = 0.0) -> csc_matrix:
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

    def rebind_for_state(self, u: ArrayLike) -> NonlocalDC1DProblem:
        """Bind active WKB turning-point support signature at ``u``."""
        if not self.device.has_wkb_tunnelling:
            return self
        ext = evaluate_nonlocal_couplings_1d(
            self.device, u, voltage_V=self.voltage_V, derivatives=False, bound_wkb_signature=None
        )
        sig = None if ext.wkb_support is None else ext.wkb_support.support_signature
        return NonlocalDC1DProblem(self.device, self.voltage_V, self.generation_scale, sig)
