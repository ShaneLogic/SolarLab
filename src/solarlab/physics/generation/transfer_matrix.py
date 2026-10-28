"""Coherent thin-film transfer-matrix optics (normal incidence, front illumination).

Pettersson et al., J. Appl. Phys. 86, 487 (1999); Burkhard et al., Adv. Mater.
22, 3293 (2010). Every layer carries ``n(lambda) + i k(lambda)`` sampled on one
wavelength grid [m]. The primary generation quantity is the cumulative
absorbed fraction ``C(x, lambda)`` -- the normalised Poynting-flux drop from
the stack front to ``x`` in closed form -- whose differences across control-
volume faces give each cell's exact absorbed photon count with no spatial
quadrature. ``C(back) = 1 - R - T`` up to the dropped multi-pass term of the
incoherent first-layer bypass, so generation is bounded by the incident flux.

Kept guards (do not remove): the ``_inv2x2`` determinant floor, the float64
representability check on the backward-wave power, and the ``C <= 1`` bound.
A first layer may be marked incoherent (a thick substrate): it is treated as
one Fresnel reflection plus Beer-Lambert power attenuation feeding the
coherent sub-stack; only the first layer may be incoherent.

The numerical kernels are migrated with their evaluation order unchanged
(perovskite_sim.physics.optics), so results reproduce bit for bit.
``tmm_transmittance`` is new and diagnostic only: it closes the R + T + A
photon budget and never enters generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

__all__ = [
    "TMMLayer",
    "dropped_multipass_fraction",
    "tmm_absorbed_photon_flux_per_cell",
    "tmm_absorption_profile",
    "tmm_cumulative_absorptance",
    "tmm_point_generation",
    "tmm_reflectance",
    "tmm_transmittance",
]


@dataclass(frozen=True)
class TMMLayer:
    """One optical layer: thickness ``d`` [m], ``n`` and ``k`` per wavelength."""

    d: float
    n: np.ndarray
    k: np.ndarray
    incoherent: bool = False

    @property
    def n_complex(self) -> np.ndarray:
        return self.n + 1j * self.k


def _require_single_incoherent_front(layers: Sequence[TMMLayer]) -> None:
    for j, lyr in enumerate(layers):
        if j > 0 and lyr.incoherent:
            raise ValueError(
                "only the first layer may be incoherent "
                f"(layer index {j} has incoherent=True)"
            )


def _transfer_matrix_stack(
    layers: Sequence[TMMLayer],
    wavelengths: np.ndarray,
    n_ambient: float | np.ndarray = 1.0,
    n_substrate: float | np.ndarray = 1.0,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Total transfer matrix ``(n_wl, 2, 2)`` and partial products per layer."""

    _require_single_incoherent_front(layers)

    n_wl = len(wavelengths)
    S: np.ndarray = np.zeros((n_wl, 2, 2), dtype=complex)
    S[:, 0, 0] = 1.0
    S[:, 1, 1] = 1.0

    def interface_matrix(n_a: np.ndarray, n_b: np.ndarray) -> np.ndarray:
        r = (n_a - n_b) / (n_a + n_b)
        t = 2.0 * n_a / (n_a + n_b)
        I_mat: np.ndarray = np.zeros((n_wl, 2, 2), dtype=complex)
        I_mat[:, 0, 0] = 1.0 / t
        I_mat[:, 0, 1] = r / t
        I_mat[:, 1, 0] = r / t
        I_mat[:, 1, 1] = 1.0 / t
        return I_mat

    def propagation_matrix(n_c: np.ndarray, d: float) -> np.ndarray:
        # Maps fields at the right edge to the left edge; for k > 0 the
        # forward wave is stronger at the (illuminated) left face.
        delta = 2.0 * np.pi * n_c * d / wavelengths
        P: np.ndarray = np.zeros((n_wl, 2, 2), dtype=complex)
        P[:, 0, 0] = np.exp(-1j * delta)
        P[:, 1, 1] = np.exp(1j * delta)
        return P

    def mat_mul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return np.einsum("wij,wjk->wik", A, B)

    S_partial: list[np.ndarray] = []
    n_prev = np.broadcast_to(np.asarray(n_ambient, dtype=complex), (n_wl,)).astype(complex, copy=True)
    for layer in layers:
        n_c = layer.n_complex
        S = mat_mul(S, interface_matrix(n_prev, n_c))
        S = mat_mul(S, propagation_matrix(n_c, layer.d))
        S_partial.append(S.copy())
        n_prev = n_c
    n_sub = np.broadcast_to(np.asarray(n_substrate, dtype=complex), (n_wl,)).astype(complex, copy=True)
    S_total = mat_mul(S, interface_matrix(n_prev, n_sub))
    return S_total, S_partial


def _inv2x2(M: np.ndarray) -> np.ndarray:
    """Batched 2x2 inverse; a singular wavelength contributes zero, not NaN."""

    det = M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0]
    safe = np.abs(det) > 1e-30
    inv_det = np.where(safe, 1.0 / np.where(safe, det, 1.0), 0.0)
    inv = np.empty_like(M)
    inv[:, 0, 0] = M[:, 1, 1] * inv_det
    inv[:, 0, 1] = -M[:, 0, 1] * inv_det
    inv[:, 1, 0] = -M[:, 1, 0] * inv_det
    inv[:, 1, 1] = M[:, 0, 0] * inv_det
    return inv


def _layer_field_amplitudes(
    layers: Sequence[TMMLayer],
    wavelengths: np.ndarray,
    n_ambient: float | np.ndarray = 1.0,
    n_substrate: float | np.ndarray = 1.0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Forward/backward amplitudes at the LEFT edge of every layer.

    Normalised to unit incident amplitude; inside layer ``j`` at local depth
    ``u``: ``E(u) = E+ exp(+i 2 pi n_c u / lam) + E- exp(-i 2 pi n_c u / lam)``.
    """

    n_wl = len(wavelengths)
    S_total, _S_partial = _transfer_matrix_stack(layers, wavelengths, n_ambient, n_substrate)
    r = S_total[:, 1, 0] / S_total[:, 0, 0]

    def interface_matrix(n_a: np.ndarray, n_b: np.ndarray) -> np.ndarray:
        r_ab = (n_a - n_b) / (n_a + n_b)
        t_ab = 2.0 * n_a / (n_a + n_b)
        I_mat: np.ndarray = np.zeros((n_wl, 2, 2), dtype=complex)
        I_mat[:, 0, 0] = 1.0 / t_ab
        I_mat[:, 0, 1] = r_ab / t_ab
        I_mat[:, 1, 0] = r_ab / t_ab
        I_mat[:, 1, 1] = 1.0 / t_ab
        return I_mat

    S_before: list[np.ndarray] = []
    M: np.ndarray = np.zeros((n_wl, 2, 2), dtype=complex)
    M[:, 0, 0] = 1.0
    M[:, 1, 1] = 1.0
    n_prev = np.broadcast_to(np.asarray(n_ambient, dtype=complex), (n_wl,)).astype(complex, copy=True)
    for layer in layers:
        n_c = layer.n_complex
        M = np.einsum("wij,wjk->wik", M, interface_matrix(n_prev, n_c))
        S_before.append(M.copy())
        delta = 2.0 * np.pi * n_c * layer.d / wavelengths
        P_full: np.ndarray = np.zeros((n_wl, 2, 2), dtype=complex)
        P_full[:, 0, 0] = np.exp(-1j * delta)
        P_full[:, 1, 1] = np.exp(1j * delta)
        M = np.einsum("wij,wjk->wik", M, P_full)
        n_prev = n_c

    E_in = np.stack([np.ones(n_wl, dtype=complex), r], axis=1)
    amplitudes: list[tuple[np.ndarray, np.ndarray]] = []
    for j in range(len(layers)):
        S_inv = _inv2x2(S_before[j])
        E_j = np.einsum("wij,wj->wi", S_inv, E_in)
        amplitudes.append((E_j[:, 0], E_j[:, 1]))
    return amplitudes


def _electric_field_profile(
    layers: Sequence[TMMLayer],
    wavelengths: np.ndarray,
    x_positions: np.ndarray,
    layer_indices: np.ndarray,
    local_positions: np.ndarray,
    n_ambient: float | np.ndarray = 1.0,
    n_substrate: float | np.ndarray = 1.0,
) -> np.ndarray:
    """``|E(x)|^2 / |E_0|^2`` with shape ``(N, n_wl)``."""

    n_wl = len(wavelengths)
    N = len(x_positions)
    amplitudes = _layer_field_amplitudes(layers, wavelengths, n_ambient, n_substrate)
    E_sq = np.zeros((N, n_wl))
    for j in range(len(layers)):
        mask = layer_indices == j
        if not np.any(mask):
            continue
        x_local = local_positions[mask]
        n_c = layers[j].n_complex
        E_plus, E_minus = amplitudes[j]
        delta_x = 2.0 * np.pi * np.outer(n_c, x_local) / wavelengths[:, None]
        E_field = E_plus[:, None] * np.exp(1j * delta_x) + E_minus[:, None] * np.exp(-1j * delta_x)
        E_sq[mask, :] = (np.abs(E_field) ** 2).T
    return E_sq


def _map_to_layers(
    x: np.ndarray, layer_boundaries: np.ndarray, n_layers: int
) -> tuple[np.ndarray, np.ndarray]:
    """Layer index and local offset of each position.

    A position exactly on an internal boundary belongs to the LATER layer at
    local depth 0, which keeps the cumulative absorptance continuous.
    """

    layer_indices: np.ndarray = np.zeros(len(x), dtype=int)
    local_positions = np.zeros(len(x))
    for j in range(n_layers):
        x_lo = layer_boundaries[j]
        x_hi = layer_boundaries[j + 1]
        mask = (x >= x_lo - 1e-15) & (x <= x_hi + 1e-15)
        layer_indices[mask] = j
        local_positions[mask] = x[mask] - x_lo
    return layer_indices, local_positions


def _layer_absorbed_fraction(
    layer: TMMLayer,
    wavelengths: np.ndarray,
    u: np.ndarray,
    E_plus: np.ndarray,
    E_minus: np.ndarray,
    n_ambient: float | np.ndarray,
) -> np.ndarray:
    """Exact ``integral_0^u A du'`` inside one coherent layer, ``(n_u, n_wl)``.

    With ``a = 4 pi k / lam`` and ``b = 4 pi n / lam``::

        I(u) = (n / n_amb) [ (|E+|^2 - |E+|^2 e^{-a u}) + (|E-|^2 e^{a u} - |E-|^2) ]
               + (2 k / n_amb) Im( E+ E-^* (e^{i b u} - 1) )

    The prefactor collapse removes the ``a -> 0`` / ``b -> 0`` singularities
    analytically, so a transparent layer contributes an exact zero. Past
    ``a u = 300`` the growing backward power is rebuilt from the log of its
    amplitude; past the float64 window it raises instead of returning +inf.
    The front-referenced backward amplitude itself loses meaning beyond
    ``a d ~ 70`` (inherited transfer-matrix limit, guarded by ``C <= 1``).
    """

    n_r = np.asarray(layer.n, dtype=float)
    kappa = np.asarray(layer.k, dtype=float)
    n_amb = np.asarray(n_ambient, dtype=float)
    a = 4.0 * np.pi * kappa / wavelengths
    b = 4.0 * np.pi * n_r / wavelengths
    u2 = np.asarray(u, dtype=float)[:, None]
    fwd0 = np.abs(E_plus) ** 2
    bwd0 = np.abs(E_minus) ** 2
    au = a[None, :] * u2
    fwd_u = fwd0[None, :] * np.exp(-au)
    big = au > 300.0
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log_amp = np.log(np.abs(E_minus))
        bwd_u = np.where(
            big,
            np.exp(np.where(big, 2.0 * log_amp[None, :] + au, 0.0)),
            bwd0[None, :] * np.exp(np.where(big, 0.0, au)),
        )
    if not np.all(np.isfinite(bwd_u)):
        raise ValueError(
            "TMM backward-wave power is not representable in float64 for "
            f"this layer (d={layer.d:.3e} m, max alpha*d={float(au.max()):.1f}); "
            "split the layer or mark it incoherent"
        )
    cross = E_plus * np.conj(E_minus)
    osc = np.exp(1j * b[None, :] * u2) - 1.0
    decay_part = (n_r / n_amb)[None, :] * ((fwd0[None, :] - fwd_u) + (bwd_u - bwd0[None, :]))
    beat_part = (2.0 * kappa / n_amb)[None, :] * np.imag(cross[None, :] * osc)
    return decay_part + beat_part


#: ``C`` is a fraction of the incident flux; the slack only absorbs round-off.
_C_MAX_SLACK = 1e-6


def _assert_absorptance_bounded(C: np.ndarray, layers: Sequence[TMMLayer], wavelengths: np.ndarray) -> None:
    if C.size == 0:
        return
    C_max = float(np.max(C))
    if np.isfinite(C_max) and C_max <= 1.0 + _C_MAX_SLACK:
        return
    worst, au_max = -1, 0.0
    for j, lyr in enumerate(layers):
        au = float(np.max(4.0 * np.pi * np.asarray(lyr.k) / np.asarray(wavelengths)) * lyr.d)
        if au > au_max:
            worst, au_max = j, au
    where = (
        f"layer {worst} (d={layers[worst].d:.3e} m, max alpha*d={au_max:.1f})"
        if worst >= 0
        else "the stack"
    )
    raise ValueError(
        f"TMM cumulative absorptance reached {C_max:.4g}, above the physical "
        f"bound of 1 — the stack cannot absorb more than the incident flux. "
        f"Thickest optical depth is {where}; the transfer matrix loses the "
        "backward amplitude to round-off past alpha*d ~ 70. Split the layer, "
        "thin it, or mark it incoherent."
    )


def _incoherent_front_factors(
    layer0: TMMLayer, wavelengths: np.ndarray, n_ambient: float | np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Air -> thick-slab Fresnel power reflection, bulk power transmission, slab index.

    Power form, no phase: the field form would reintroduce substrate fringes.
    """

    n_real = np.asarray(layer0.n, dtype=float)
    k_slab = np.asarray(layer0.k, dtype=float)
    r0 = (n_ambient - n_real) / (n_ambient + n_real)
    R_front = r0 * r0
    alpha = 4.0 * np.pi * k_slab / wavelengths
    T_bulk = np.exp(-alpha * layer0.d)
    return R_front, T_bulk, n_real


def tmm_cumulative_absorptance(
    layers: Sequence[TMMLayer],
    wavelengths: np.ndarray,
    x: np.ndarray,
    layer_boundaries: np.ndarray,
    n_ambient: float | np.ndarray = 1.0,
    n_substrate: float | np.ndarray = 1.0,
) -> np.ndarray:
    """Cumulative absorbed fraction ``C(x, lambda)``, shape ``(N, n_wl)``.

    ``x`` is measured from the FRONT of ``layers[0]``. ``C(x_b) - C(x_a)`` is
    exactly the absorbed fraction between the two positions on any sampling.
    """

    x = np.asarray(x, dtype=float)
    N = len(x)
    n_wl = len(wavelengths)
    n_layers = len(layers)
    if n_layers == 0:
        return np.zeros((N, n_wl))
    layer_indices, local_positions = _map_to_layers(x, layer_boundaries, n_layers)

    if layers[0].incoherent:
        _require_single_incoherent_front(layers)
        R_front, T_bulk, n_real = _incoherent_front_factors(layers[0], wavelengths, n_ambient)
        C = np.zeros((N, n_wl))
        alpha_slab = 4.0 * np.pi * layers[0].k / wavelengths
        slab_mask = layer_indices == 0
        if np.any(slab_mask):
            u_slab = np.clip(local_positions[slab_mask], 0.0, layers[0].d)[:, None]
            C[slab_mask, :] = (1.0 - R_front)[None, :] * (-np.expm1(-alpha_slab[None, :] * u_slab))
        C_slab_full = (1.0 - R_front) * (1.0 - T_bulk)
        sub_mask = layer_indices >= 1
        if np.any(sub_mask) and n_layers > 1:
            x0_sub = layer_boundaries[1]
            C_sub = tmm_cumulative_absorptance(
                layers[1:], wavelengths, x[sub_mask] - x0_sub,
                layer_boundaries[1:] - x0_sub,
                n_ambient=n_real, n_substrate=n_substrate,
            )
            scale = (1.0 - R_front) * T_bulk
            C[sub_mask, :] = C_slab_full[None, :] + C_sub * scale[None, :]
        _assert_absorptance_bounded(C, layers, wavelengths)
        return C

    amplitudes = _layer_field_amplitudes(layers, wavelengths, n_ambient, n_substrate)
    offsets = np.zeros((n_layers + 1, n_wl))
    for j, layer in enumerate(layers):
        E_plus, E_minus = amplitudes[j]
        full_j = _layer_absorbed_fraction(
            layer, wavelengths, np.array([layer.d]), E_plus, E_minus, n_ambient
        )[0, :]
        offsets[j + 1, :] = offsets[j, :] + full_j
    C = np.zeros((N, n_wl))
    for j in range(n_layers):
        mask = layer_indices == j
        if not np.any(mask):
            continue
        E_plus, E_minus = amplitudes[j]
        u_local = np.clip(local_positions[mask], 0.0, layers[j].d)
        C[mask, :] = offsets[j, :][None, :] + _layer_absorbed_fraction(
            layers[j], wavelengths, u_local, E_plus, E_minus, n_ambient
        )
    _assert_absorptance_bounded(C, layers, wavelengths)
    return C


def tmm_absorbed_photon_flux_per_cell(
    layers: Sequence[TMMLayer],
    wavelengths: np.ndarray,
    spectral_flux: np.ndarray,
    faces: np.ndarray,
    layer_boundaries: np.ndarray,
    n_ambient: float | np.ndarray = 1.0,
    n_substrate: float | np.ndarray = 1.0,
) -> np.ndarray:
    """Photons absorbed per unit area and time in each cell [m^-2 s^-1].

    ``faces`` are the ``n_cells + 1`` cell boundaries (optical coordinate).
    The spatial part is exact; the wavelength integral is the trapezoid on
    ``wavelengths`` [m], which commutes with the spatial telescoping.
    """

    faces = np.asarray(faces, dtype=float)
    if faces.ndim != 1 or faces.shape[0] < 2:
        raise ValueError(f"faces must be 1-D with at least 2 entries, got shape {faces.shape}")
    C = tmm_cumulative_absorptance(layers, wavelengths, faces, layer_boundaries, n_ambient, n_substrate)
    dC = C[1:, :] - C[:-1, :]
    return np.trapezoid(dC * spectral_flux[None, :], x=wavelengths, axis=1)


def tmm_absorption_profile(
    layers: Sequence[TMMLayer],
    wavelengths: np.ndarray,
    x: np.ndarray,
    layer_boundaries: np.ndarray,
    n_ambient: float | np.ndarray = 1.0,
    n_substrate: float | np.ndarray = 1.0,
) -> np.ndarray:
    """Point-sampled spectral absorption rate ``A(x, lambda)`` [m^-1], ``(N, n_wl)``.

    ``A = (4 pi n k) / (lam n_amb) |E|^2``; a pointwise diagnostic (EQE, EL,
    escape probability), never a photon-conserving cell source.
    """

    N = len(x)
    n_wl = len(wavelengths)
    n_layers = len(layers)
    layer_indices, local_positions = _map_to_layers(x, layer_boundaries, n_layers)

    if layers and layers[0].incoherent:
        R_front, T_bulk, n_real = _incoherent_front_factors(layers[0], wavelengths, n_ambient)
        A = np.zeros((N, n_wl))
        slab_mask = layer_indices == 0
        if np.any(slab_mask):
            x_local_slab = local_positions[slab_mask]
            alpha_slab = 4.0 * np.pi * layers[0].k / wavelengths
            decay = np.exp(-alpha_slab[None, :] * x_local_slab[:, None])
            E_sq_slab = (1.0 - R_front)[None, :] * decay
            A[slab_mask, :] = alpha_slab[None, :] * E_sq_slab
        sub_mask = layer_indices >= 1
        if np.any(sub_mask) and n_layers > 1:
            x0_sub = layer_boundaries[1]
            x_sub = x[sub_mask] - x0_sub
            boundaries_sub = layer_boundaries[1:] - x0_sub
            scale = (1.0 - R_front) * T_bulk
            A_sub = tmm_absorption_profile(
                layers[1:], wavelengths, x_sub, boundaries_sub,
                n_ambient=n_real, n_substrate=n_substrate,
            )
            A[sub_mask, :] = A_sub * scale[None, :]
        return A

    E_sq = _electric_field_profile(
        layers, wavelengths, x, layer_indices, local_positions, n_ambient, n_substrate
    )
    A = np.zeros((N, n_wl))
    for j in range(n_layers):
        mask = layer_indices == j
        if not np.any(mask):
            continue
        coeff_j = 4.0 * np.pi * layers[j].n * layers[j].k / (wavelengths * n_ambient)
        A[mask, :] = E_sq[mask, :] * coeff_j[None, :]
    return A


def tmm_point_generation(
    layers: Sequence[TMMLayer],
    wavelengths: np.ndarray,
    spectral_flux: np.ndarray,
    x: np.ndarray,
    layer_boundaries: np.ndarray,
    n_ambient: float = 1.0,
    n_substrate: float = 1.0,
    return_absorbance: bool = False,
) -> Any:
    """POINT-SAMPLED ``G(x) = int A Phi dlam`` [m^-3 s^-1] (historical ``tmm_generation``).

    Diagnostic only: not photon-conserving under the solver's cell rule;
    use ``tmm_absorbed_photon_flux_per_cell`` for a generation source.
    """

    A = tmm_absorption_profile(layers, wavelengths, x, layer_boundaries, n_ambient, n_substrate)
    G = np.trapezoid(A * spectral_flux[None, :], x=wavelengths, axis=1)
    if return_absorbance:
        return G, A
    return G


def tmm_reflectance(
    layers: Sequence[TMMLayer],
    wavelengths: np.ndarray,
    n_ambient: float = 1.0,
    n_substrate: float = 1.0,
) -> np.ndarray:
    """Spectral power reflectance ``R(lambda)`` of the stack.

    With an incoherent first layer: front Fresnel plus one round trip
    through the slab off the coherent sub-stack (further round trips are
    dropped, see ``dropped_multipass_fraction``).
    """

    if layers and layers[0].incoherent:
        _require_single_incoherent_front(layers)
        R_front, T_bulk, n_real = _incoherent_front_factors(layers[0], wavelengths, n_ambient)
        S_sub, _ = _transfer_matrix_stack(layers[1:], wavelengths, n_ambient=n_real, n_substrate=n_substrate)
        r_sub = S_sub[:, 1, 0] / S_sub[:, 0, 0]
        R_sub = np.abs(r_sub) ** 2
        return R_front + (1.0 - R_front) ** 2 * (T_bulk**2) * R_sub
    S_total, _ = _transfer_matrix_stack(layers, wavelengths, n_ambient, n_substrate)
    r = S_total[:, 1, 0] / S_total[:, 0, 0]
    return np.abs(r) ** 2


def tmm_transmittance(
    layers: Sequence[TMMLayer],
    wavelengths: np.ndarray,
    n_ambient: float = 1.0,
    n_substrate: float = 1.0,
) -> np.ndarray:
    """Spectral power transmittance into the exit medium (diagnostic).

    ``T = (n_exit / n_entry) |t|^2`` for real bounding indices, with the
    coherent sub-stack entered from the slab index after an incoherent
    first layer.
    """

    if layers and layers[0].incoherent:
        _require_single_incoherent_front(layers)
        R_front, T_bulk, n_real = _incoherent_front_factors(layers[0], wavelengths, n_ambient)
        S_sub, _ = _transfer_matrix_stack(layers[1:], wavelengths, n_ambient=n_real, n_substrate=n_substrate)
        return (1.0 - R_front) * T_bulk * np.abs(1.0 / S_sub[:, 0, 0]) ** 2 * (n_substrate / n_real)
    S_total, _ = _transfer_matrix_stack(layers, wavelengths, n_ambient, n_substrate)
    return np.abs(1.0 / S_total[:, 0, 0]) ** 2 * (n_substrate / n_ambient)


def dropped_multipass_fraction(
    layers: Sequence[TMMLayer],
    wavelengths: np.ndarray,
    n_ambient: float = 1.0,
    n_substrate: float = 1.0,
) -> np.ndarray:
    """Fraction not booked as R, T or A by the incoherent first-layer bypass.

    The light the sub-stack reflects back into the slab, ``(1-R_f) T_b R_sub``,
    is booked only for its single front escape ``(1-R_f)^2 T_b^2 R_sub``; the
    remainder ``(1-R_f) T_b R_sub (1 - (1-R_f) T_b)`` (slab re-absorption and
    further round trips) is dropped. Zero for a fully coherent stack.
    """

    n_wl = len(wavelengths)
    if not (layers and layers[0].incoherent):
        return np.zeros(n_wl)
    _require_single_incoherent_front(layers)
    R_front, T_bulk, n_real = _incoherent_front_factors(layers[0], wavelengths, n_ambient)
    S_sub, _ = _transfer_matrix_stack(layers[1:], wavelengths, n_ambient=n_real, n_substrate=n_substrate)
    R_sub = np.abs(S_sub[:, 1, 0] / S_sub[:, 0, 0]) ** 2
    entering = (1.0 - R_front) * T_bulk
    return entering * R_sub * (1.0 - entering)
