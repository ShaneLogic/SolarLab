"""Transfer-matrix method (TMM) for coherent thin-film optics.

Computes position-resolved generation rate G(x) by integrating the
absorbed photon flux over the AM1.5G spectrum. Each layer is described
by its complex refractive index n_complex(lambda) = n + i*k and thickness d.

The implementation follows Pettersson et al., J. Appl. Phys. 86, 487 (1999)
and Burkhard et al., Adv. Mater. 22, 3293 (2010).

Usage:
    layers = [TMMLayer(d=400e-9, n_complex=n_arr, k_complex=k_arr), ...]
    wavelengths = np.linspace(300e-9, 800e-9, 200)
    x = ...  # spatial grid from multilayer_grid
    G = tmm_generation(layers, wavelengths, am15g_flux, x)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np

from perovskite_sim._compat.numpy_compat import trapezoid


@dataclass(frozen=True)
class TMMLayer:
    """One layer in the TMM stack.

    d: physical thickness [m]
    n: real part of refractive index, shape (n_wavelengths,)
    k: imaginary part (extinction coefficient), shape (n_wavelengths,)
    incoherent: if True, this layer is treated as a thick (>> lambda)
        incoherent slab — no phase tracking. Only legal on the FIRST
        layer of the stack (typically a ~1 mm glass substrate). The
        bypass applies a single air->layer Fresnel reflection plus
        bulk Beer-Lambert power attenuation and hands the remaining
        intensity to the coherent sub-stack formed by layers[1:].
    """
    d: float
    n: np.ndarray
    k: np.ndarray
    incoherent: bool = False

    @property
    def n_complex(self) -> np.ndarray:
        """Complex refractive index n + ik."""
        return self.n + 1j * self.k


def _transfer_matrix_stack(
    layers: Sequence[TMMLayer],
    wavelengths: np.ndarray,
    n_ambient: float | np.ndarray = 1.0,
    n_substrate: float | np.ndarray = 1.0,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Compute the total transfer matrix and per-layer partial products.

    Returns:
        S_total: shape (n_wl, 2, 2) — total system transfer matrix
        S_partial: list of len(layers) arrays, each (n_wl, 2, 2) —
                   partial product from ambient through layer j (inclusive).
                   Used for computing the electric field inside each layer.
    """
    # Validator: only the first layer may be marked incoherent. Any
    # incoherent layer at index >= 1 is unsupported (would require a
    # full hybrid incoherent-coherent transfer matrix — see module
    # docstring / commit message for rationale).
    for j, lyr in enumerate(layers):
        if j > 0 and lyr.incoherent:
            raise ValueError(
                "only the first layer may be incoherent "
                f"(layer index {j} has incoherent=True)"
            )

    n_wl = len(wavelengths)
    # Identity start
    S = np.zeros((n_wl, 2, 2), dtype=complex)
    S[:, 0, 0] = 1.0
    S[:, 1, 1] = 1.0

    # Interface matrix: from medium with index n_a to n_b
    # I_{ab} = (1 / t_{ab}) * [[1, r_{ab}], [r_{ab}, 1]]
    # r = (n_a - n_b) / (n_a + n_b), t = 2*n_a / (n_a + n_b)

    def interface_matrix(n_a: np.ndarray, n_b: np.ndarray) -> np.ndarray:
        """Interface (Fresnel) matrix for normal incidence, shape (n_wl, 2, 2)."""
        r = (n_a - n_b) / (n_a + n_b)
        t = 2.0 * n_a / (n_a + n_b)
        I_mat = np.zeros((n_wl, 2, 2), dtype=complex)
        I_mat[:, 0, 0] = 1.0 / t
        I_mat[:, 0, 1] = r / t
        I_mat[:, 1, 0] = r / t
        I_mat[:, 1, 1] = 1.0 / t
        return I_mat

    def propagation_matrix(n_c: np.ndarray, d: float) -> np.ndarray:
        """Propagation matrix through a layer, shape (n_wl, 2, 2).

        Maps fields at the RIGHT edge to fields at the LEFT edge.
        With n_c = n + ik: exp(-i*delta) has |.| = exp(+2*pi*k*d/lambda),
        meaning the forward wave is stronger at the left (front) — correct
        for an absorbing layer where light enters from the left.
        """
        # Phase: delta = 2*pi*n_c*d / lambda
        delta = 2.0 * np.pi * n_c * d / wavelengths
        P = np.zeros((n_wl, 2, 2), dtype=complex)
        P[:, 0, 0] = np.exp(-1j * delta)
        P[:, 1, 1] = np.exp(1j * delta)
        return P

    def mat_mul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Batched 2x2 matrix multiply, shapes (n_wl, 2, 2)."""
        return np.einsum("wij,wjk->wik", A, B)

    S_partial: list[np.ndarray] = []
    n_prev = np.broadcast_to(
        np.asarray(n_ambient, dtype=complex), (n_wl,),
    ).astype(complex, copy=True)

    for layer in layers:
        n_c = layer.n_complex  # (n_wl,)
        # Interface into this layer
        I_in = interface_matrix(n_prev, n_c)
        S = mat_mul(S, I_in)
        # Propagation through this layer
        L = propagation_matrix(n_c, layer.d)
        S = mat_mul(S, L)
        S_partial.append(S.copy())
        n_prev = n_c

    # Final interface: last layer → substrate
    n_sub = np.broadcast_to(
        np.asarray(n_substrate, dtype=complex), (n_wl,),
    ).astype(complex, copy=True)
    I_out = interface_matrix(n_prev, n_sub)
    S_total = mat_mul(S, I_out)

    return S_total, S_partial


def _inv2x2(M: np.ndarray) -> np.ndarray:
    """Batched 2x2 matrix inverse, shape (n_wl, 2, 2).

    Guards against det≈0 (a lossless layer at a specific wavelength can
    produce a singular transfer submatrix). Those wavelengths contribute
    zero to E_sq rather than propagating NaN/Inf downstream into G(x).
    """
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
    """Forward/backward field amplitudes at the LEFT edge of every layer.

    Returns a list of ``(E_plus, E_minus)`` pairs, each of shape ``(n_wl,)``,
    normalised to unit incident amplitude in the ambient medium.  Inside
    layer ``j`` at local position ``u`` the field is::

        E(u) = E_plus * exp(+i*2*pi*n_c*u/lam) + E_minus * exp(-i*2*pi*n_c*u/lam)

    Extracted verbatim from ``_electric_field_profile`` so that the
    point-sampled ``|E|^2`` path and the analytic cell-integrated
    absorptance path (``tmm_cumulative_absorptance``) consume exactly the
    same amplitudes — every operation, and its order, is unchanged.
    """
    n_wl = len(wavelengths)

    S_total, _S_partial = _transfer_matrix_stack(
        layers, wavelengths, n_ambient, n_substrate,
    )

    # Reflection coefficient of the full stack
    r = S_total[:, 1, 0] / S_total[:, 0, 0]  # (n_wl,)

    # At the left edge of the stack (ambient side), E+ = 1, E- = r.
    # Inside layer j at position x_local from its left edge:
    #   [E+_j(x), E-_j(x)] = P_j(x) @ I_{j-1,j} @ ... @ I_{0,1} @ P_1 @ ... @ [1, r]^T
    # where P_j(x) propagates only distance x_local, not full d_j.
    #
    # S_before[j] = product from ambient to just INSIDE layer j
    # (after interface j-1,j).

    def interface_matrix(n_a, n_b):
        r_ab = (n_a - n_b) / (n_a + n_b)
        t_ab = 2.0 * n_a / (n_a + n_b)
        I_mat = np.zeros((n_wl, 2, 2), dtype=complex)
        I_mat[:, 0, 0] = 1.0 / t_ab
        I_mat[:, 0, 1] = r_ab / t_ab
        I_mat[:, 1, 0] = r_ab / t_ab
        I_mat[:, 1, 1] = 1.0 / t_ab
        return I_mat

    S_before: list[np.ndarray] = []
    M = np.zeros((n_wl, 2, 2), dtype=complex)
    M[:, 0, 0] = 1.0
    M[:, 1, 1] = 1.0
    n_prev = np.broadcast_to(
        np.asarray(n_ambient, dtype=complex), (n_wl,),
    ).astype(complex, copy=True)

    for j, layer in enumerate(layers):
        n_c = layer.n_complex
        I_in = interface_matrix(n_prev, n_c)
        M = np.einsum("wij,wjk->wik", M, I_in)
        S_before.append(M.copy())
        # After storing S_before, propagate through the full layer for next iteration
        delta = 2.0 * np.pi * n_c * layer.d / wavelengths
        P_full = np.zeros((n_wl, 2, 2), dtype=complex)
        P_full[:, 0, 0] = np.exp(-1j * delta)
        P_full[:, 1, 1] = np.exp(1j * delta)
        M = np.einsum("wij,wjk->wik", M, P_full)
        n_prev = n_c

    # Input field vector: [E+, E-] = [1, r] at the ambient side
    E_in = np.stack([np.ones(n_wl, dtype=complex), r], axis=1)  # (n_wl, 2)

    amplitudes: list[tuple[np.ndarray, np.ndarray]] = []
    for j in range(len(layers)):
        # S_before[j] maps [a_j, b_j] -> [1, r], so invert to get
        # the field amplitudes at the left edge of layer j:
        # [a_j, b_j] = S_before[j]^{-1} @ [1, r]
        S_inv = _inv2x2(S_before[j])
        E_j = np.einsum("wij,wj->wi", S_inv, E_in)  # (n_wl, 2)
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
    """Compute |E(x)|^2 / |E_0|^2 at each spatial position for each wavelength.

    Args:
        layers: TMM layer stack
        wavelengths: shape (n_wl,) in metres
        x_positions: shape (N,) spatial grid positions [m]
        layer_indices: shape (N,) which TMM layer each grid point is in
        local_positions: shape (N,) distance from the left edge of its layer [m]
        n_ambient, n_substrate: real refractive indices of bounding media

    Returns:
        E_sq: shape (N, n_wl) — normalized |E|^2 at each grid point and wavelength
    """
    n_wl = len(wavelengths)
    N = len(x_positions)
    n_layers = len(layers)

    amplitudes = _layer_field_amplitudes(
        layers, wavelengths, n_ambient, n_substrate,
    )

    # Compute |E|^2 at each grid point
    E_sq = np.zeros((N, n_wl))

    for j in range(n_layers):
        mask = layer_indices == j
        if not np.any(mask):
            continue
        x_local = local_positions[mask]  # (n_j,)
        n_c = layers[j].n_complex        # (n_wl,)

        E_plus, E_minus = amplitudes[j]  # (n_wl,) forward / backward

        # Field at position x_local from the left edge of layer j:
        # E+(x) = E+_left * exp(i*2*pi*n_c*x/lambda) — forward wave, decays for k>0
        # E-(x) = E-_left * exp(-i*2*pi*n_c*x/lambda) — backward wave
        delta_x = 2.0 * np.pi * np.outer(n_c, x_local) / wavelengths[:, None]
        # (n_wl, n_j)

        E_field = (E_plus[:, None] * np.exp(1j * delta_x)
                   + E_minus[:, None] * np.exp(-1j * delta_x))
        # (n_wl, n_j)

        E_sq[mask, :] = (np.abs(E_field) ** 2).T  # (n_j, n_wl)

    return E_sq


def _map_to_layers(
    x: np.ndarray, layer_boundaries: np.ndarray, n_layers: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign every position to a TMM layer and its local offset.

    A position sitting exactly on an internal boundary is assigned to the
    LATER layer with local position 0 (the ascending loop overwrites), which
    is what makes the cumulative absorptance continuous across interfaces.
    """
    layer_indices = np.zeros(len(x), dtype=int)
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
    """Exact ``integral_0^u A(u', lambda) du'`` inside one coherent layer.

    This is the layer-resolved **Poynting-flux drop** between the layer's
    front face and local depth ``u``, normalised by the incident flux —
    i.e. the closed form of what ``tmm_absorption_profile`` returns as a
    density.  Derivation (normal incidence, ``E(u) = E+ e^{i k_c u} +
    E- e^{-i k_c u}``, ``k_c = 2 pi (n + i kappa) / lam``)::

        |E(u)|^2 = |E+|^2 e^{-a u} + |E-|^2 e^{+a u}
                   + 2 Re( E+ E-^* e^{i b u} )
        a = 4 pi kappa / lam        (power absorption coefficient)
        b = 4 pi n / lam            (standing-wave beat)

    and ``A(u) = (4 pi n kappa) / (lam * n_amb) * |E(u)|^2``.  Integrating
    term by term, the two prefactor ratios collapse exactly::

        coeff / a = n / n_amb          coeff / b = kappa / n_amb

    which removes the ``a -> 0`` and ``b -> 0`` singularities analytically,
    so a transparent layer contributes an exact floating-point zero::

        I(u) = (n / n_amb) * [ (|E+|^2 - |E+|^2 e^{-a u})
                             + (|E-|^2 e^{+a u} - |E-|^2) ]
             + (2 kappa / n_amb) * Im( E+ E-^* (e^{i b u} - 1) )

    Cross-checked against the Poynting vector: ``S(u)/S_inc =
    (1/n_amb)[ n(|E+|^2 e^{-a u} - |E-|^2 e^{a u}) - 2 kappa
    Im(E+ E-^* e^{i b u}) ]`` gives ``I(u) = S(0)/S_inc - S(u)/S_inc``
    identically.

    See the inline notes below for how ``|E-|^2 e^{+a u}`` is kept
    representable in a strongly absorbing layer, and for the pre-existing
    accuracy limit this closed form inherits from the transfer matrix.

    Args:
        layer: the TMM layer
        wavelengths: (n_wl,) [m]
        u: (n_u,) local depths inside the layer [m]
        E_plus, E_minus: (n_wl,) amplitudes at the layer's LEFT edge
        n_ambient: scalar or (n_wl,) index of the medium the incident
            flux is referenced to

    Returns:
        (n_u, n_wl) cumulative absorbed fraction of the incident flux.
    """
    n_r = np.asarray(layer.n, dtype=float)          # (n_wl,)
    kappa = np.asarray(layer.k, dtype=float)        # (n_wl,)
    n_amb = np.asarray(n_ambient, dtype=float)

    a = 4.0 * np.pi * kappa / wavelengths           # (n_wl,)
    b = 4.0 * np.pi * n_r / wavelengths             # (n_wl,)

    u2 = np.asarray(u, dtype=float)[:, None]        # (n_u, 1)

    fwd0 = np.abs(E_plus) ** 2                      # (n_wl,)
    bwd0 = np.abs(E_minus) ** 2                     # (n_wl,)

    au = a[None, :] * u2                            # (n_u, n_wl)

    # Forward wave power: monotone decreasing, cannot overflow.
    fwd_u = fwd0[None, :] * np.exp(-au)
    # Backward wave power grows as e^{+a u}.  Below a * u = 300 use the
    # direct product — that keeps a transparent layer (a == 0) at an EXACT
    # floating-point zero, because exp(0) == 1.0 exactly, so fwd_u == fwd0
    # and bwd_u == bwd0 bit-for-bit and the whole term cancels.
    #
    # Past that the front-face backward AMPLITUDE has shrunk to ~e^{-a d/2}
    # while the exponential has grown to ~e^{+a d}; forming either factor
    # alone overflows / underflows.  Reconstruct from the log of the
    # AMPLITUDE (not its square): |E-|^2 e^{a u} = exp(2 ln|E-| + a u).
    # Squaring first would underflow |E-|^2 into the denormal range and
    # destroy the mantissa, which is what makes the naive form blow up to
    # +inf.
    #
    # PRE-EXISTING limit, shared with tmm_absorption_profile and NOT
    # introduced here: past a * d ~ 70 the front-referenced backward
    # amplitude is itself meaningless — the transfer matrix builds it as a
    # difference of O(1) numbers whose true value is e^{-a d / 2}, so once
    # that drops below machine epsilon the result is round-off noise
    # amplified by e^{+a d}.  Measured on a single n=3, d=20 um slab at
    # 500 nm, sweeping k so that a * d walks up: the layer absorptance is
    # physical (<= 1) through a * d = 70 and unphysical from 80 on, and the
    # point-sampled path and this closed form go wrong TOGETHER —
    #     a*d          20        60        70        80        90
    #     max A     0.2500    0.2498    0.2497    0.6669   3.76e+6
    #     C(d)-C(0) 0.7499    0.7493    0.7723    2.7496   1.13e+7
    # — so the closed form inherits the limit rather than adding one.  On a
    # single n=3 / k=0.06 / 200 um slab (a * d = 215-377 over 400-700 nm)
    # the existing point-sampled path returns max A 9.26e137 and a
    # 20001-point trapezoid of 4.910e131, and this closed form reproduces
    # 4.9099e131 (the trapezoid converges onto it: 5.140e131 at 501 pts,
    # 4.9244e131 at 2001, 4.9099e131 at 200001).  No shipped preset goes
    # near it (the thickest coherent TMM layer is sub-micron; the 1 mm
    # glass substrate is `incoherent` and never enters this branch), so it
    # is reported, not papered over.
    big = au > 300.0
    # The overflow is detected and raised on explicitly two lines below, so
    # the warning it would emit first is noise.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log_amp = np.log(np.abs(E_minus))           # (n_wl,), -inf if zero
        bwd_u = np.where(
            big,
            np.exp(np.where(big, 2.0 * log_amp[None, :] + au, 0.0)),
            bwd0[None, :] * np.exp(np.where(big, 0.0, au)),
        )
    if not np.all(np.isfinite(bwd_u)):
        # Beyond the float64 window above there is no honest value to
        # return, so fail loudly rather than let +inf reach G(x).
        raise ValueError(
            "TMM backward-wave power is not representable in float64 for "
            f"this layer (d={layer.d:.3e} m, max alpha*d={float(au.max()):.1f}); "
            "split the layer or mark it incoherent"
        )

    cross = E_plus * np.conj(E_minus)                          # (n_wl,)
    osc = np.exp(1j * b[None, :] * u2) - 1.0                   # (n_u, n_wl)

    decay_part = (n_r / n_amb)[None, :] * (
        (fwd0[None, :] - fwd_u) + (bwd_u - bwd0[None, :])
    )
    beat_part = (2.0 * kappa / n_amb)[None, :] * np.imag(cross[None, :] * osc)
    return decay_part + beat_part


# Physical bound on the cumulative absorptance. C is the normalised
# Poynting-flux drop, so C <= 1 exactly; the slack only absorbs float64
# round-off in the layer-by-layer accumulation.
_C_MAX_SLACK = 1e-6


def _assert_absorptance_bounded(
    C: np.ndarray,
    layers: Sequence[TMMLayer],
    wavelengths: np.ndarray,
) -> None:
    """Fail loudly when the cumulative absorptance exceeds unity.

    ``C`` is a fraction of the incident flux, so ``C > 1`` means the stack
    absorbed more light than reached it. ``_layer_absorbed_fraction``
    documents the mechanism and measures it: past ``alpha*d ~ 70`` the
    front-referenced backward amplitude is round-off noise amplified by
    ``e^{+alpha*d}``, and its own table records the layer absorptance going
    unphysical from ``alpha*d = 80`` on. But the only guard there is an
    ``isfinite`` check, which cannot fire until the value overflows float64
    at ``alpha*d ~ 300`` — so between the two there is a wide window in
    which ``build_material_arrays`` accepts a generation profile larger
    than the incident photon flux: finite, unflagged, and wrong by orders
    of magnitude. Measured: raising the PCBM ETL of pin_MAPbI3_tmm from
    200 nm to 800 nm yields a photon budget of 136 mA/cm^2, 3.6x the whole
    incident 300-1000 nm flux, with no exception and no warning.

    The docstring of ``tmm_cumulative_absorptance`` already promises this
    bound; this is what makes the promise true.
    """
    if C.size == 0:
        return
    C_max = float(np.max(C))
    if np.isfinite(C_max) and C_max <= 1.0 + _C_MAX_SLACK:
        return
    worst, au_max = -1, 0.0
    for j, lyr in enumerate(layers):
        au = float(
            np.max(4.0 * np.pi * np.asarray(lyr.k) / np.asarray(wavelengths))
            * lyr.d
        )
        if au > au_max:
            worst, au_max = j, au
    where = (
        f"layer {worst} (d={layers[worst].d:.3e} m, max alpha*d={au_max:.1f})"
        if worst >= 0 else "the stack"
    )
    raise ValueError(
        f"TMM cumulative absorptance reached {C_max:.4g}, above the physical "
        f"bound of 1 — the stack cannot absorb more than the incident flux. "
        f"Thickest optical depth is {where}; the transfer matrix loses the "
        "backward amplitude to round-off past alpha*d ~ 70. Split the layer, "
        "thin it, or mark it incoherent."
    )


def tmm_cumulative_absorptance(
    layers: Sequence[TMMLayer],
    wavelengths: np.ndarray,
    x: np.ndarray,
    layer_boundaries: np.ndarray,
    n_ambient: float | np.ndarray = 1.0,
    n_substrate: float | np.ndarray = 1.0,
) -> np.ndarray:
    """Cumulative absorbed fraction ``C(x, lambda)`` from the stack front to x.

    ``C`` is the exact antiderivative of ``tmm_absorption_profile``: for any
    two positions ``x_a < x_b`` inside the stack,

        ``C(x_b) - C(x_a) == integral_{x_a}^{x_b} A(x, lambda) dx``

    to machine precision, on ANY sampling of ``x`` — no spatial quadrature
    is involved.  Differencing it across dual-cell faces is what makes the
    solver's generation rectangle rule photon-exact on a coarse mesh (see
    ``physics/generation.py``).

    Because ``C`` is the normalised Poynting-flux drop, ``C`` at the back of
    the stack equals ``1 - R - T`` (up to the documented dropped multi-pass
    term in the incoherent-substrate branch), so it is bounded by 1 by
    construction and the generation it feeds can never exceed the incident
    photon flux.

    Args:
        layers: TMM layer sequence
        wavelengths: (n_wl,) [m]
        x: (N,) positions [m], measured from the FRONT of ``layers[0]``
        layer_boundaries: (n_layers + 1,) cumulative boundaries [m]
        n_ambient, n_substrate: bounding media indices

    Returns:
        C: (N, n_wl) cumulative absorbed fraction of the incident flux.
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    n_wl = len(wavelengths)
    n_layers = len(layers)
    if n_layers == 0:
        return np.zeros((N, n_wl))

    layer_indices, local_positions = _map_to_layers(
        x, layer_boundaries, n_layers,
    )

    # Incoherent first-layer bypass — mirrors tmm_absorption_profile so the
    # two stay two views of ONE optical model.
    if layers[0].incoherent:
        for j, lyr in enumerate(layers):
            if j > 0 and lyr.incoherent:
                raise ValueError(
                    "only the first layer may be incoherent "
                    f"(layer index {j} has incoherent=True)"
                )
        R_front, T_bulk, n_real = _incoherent_front_factors(
            layers[0], wavelengths, n_ambient,
        )
        C = np.zeros((N, n_wl))

        # Thick slab: forward-only power decay, integral is analytic.
        alpha_slab = 4.0 * np.pi * layers[0].k / wavelengths       # (n_wl,)
        slab_mask = layer_indices == 0
        if np.any(slab_mask):
            u_slab = np.clip(
                local_positions[slab_mask], 0.0, layers[0].d,
            )[:, None]
            C[slab_mask, :] = (1.0 - R_front)[None, :] * (
                -np.expm1(-alpha_slab[None, :] * u_slab)
            )

        # Everything the slab absorbs on the single forward pass.
        C_slab_full = (1.0 - R_front) * (1.0 - T_bulk)             # (n_wl,)

        sub_mask = layer_indices >= 1
        if np.any(sub_mask) and n_layers > 1:
            x0_sub = layer_boundaries[1]
            C_sub = tmm_cumulative_absorptance(
                layers[1:], wavelengths, x[sub_mask] - x0_sub,
                layer_boundaries[1:] - x0_sub,
                n_ambient=n_real, n_substrate=n_substrate,
            )
            scale = (1.0 - R_front) * T_bulk                        # (n_wl,)
            C[sub_mask, :] = C_slab_full[None, :] + C_sub * scale[None, :]
        _assert_absorptance_bounded(C, layers, wavelengths)
        return C

    amplitudes = _layer_field_amplitudes(
        layers, wavelengths, n_ambient, n_substrate,
    )

    # Absorbed fraction accumulated up to the FRONT face of each layer.
    offsets = np.zeros((n_layers + 1, n_wl))
    for j, layer in enumerate(layers):
        E_plus, E_minus = amplitudes[j]
        full_j = _layer_absorbed_fraction(
            layer, wavelengths, np.array([layer.d]),
            E_plus, E_minus, n_ambient,
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
            layers[j], wavelengths, u_local, E_plus, E_minus, n_ambient,
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
    """Photons absorbed per unit area per unit time inside each cell.

    ``faces`` are the ``n_cells + 1`` cell boundaries; the return value has
    shape ``(n_cells,)`` in ``m^-2 s^-1``::

        out[i] = integral_lambda Phi(lambda)
                 * [C(faces[i+1], lambda) - C(faces[i], lambda)] dlambda

    The spatial part is exact (analytic antiderivative, no quadrature), so
    ``out.sum()`` telescopes to the photons absorbed between ``faces[0]``
    and ``faces[-1]`` on ANY mesh.  The wavelength integral uses the same
    trapezoidal rule as ``tmm_generation``; because it is linear it commutes
    with the spatial telescoping, so the conservation property is unaffected
    by the spectral resolution.
    """
    faces = np.asarray(faces, dtype=float)
    if faces.ndim != 1 or faces.shape[0] < 2:
        raise ValueError(
            f"faces must be 1-D with at least 2 entries, got shape {faces.shape}"
        )
    C = tmm_cumulative_absorptance(
        layers, wavelengths, faces, layer_boundaries, n_ambient, n_substrate,
    )
    dC = C[1:, :] - C[:-1, :]                       # (n_cells, n_wl)
    return trapezoid(dC * spectral_flux[None, :], wavelengths, axis=1)


def tmm_absorption_profile(
    layers: Sequence[TMMLayer],
    wavelengths: np.ndarray,
    x: np.ndarray,
    layer_boundaries: np.ndarray,
    n_ambient: float | np.ndarray = 1.0,
    n_substrate: float | np.ndarray = 1.0,
) -> np.ndarray:
    """Compute spectral absorption rate A(x, lambda) [m^-1].

    A(x, lambda) = alpha(lambda) * n(lambda) / n_ambient * |E(x, lambda)|^2

    where alpha = 4*pi*k / lambda is the absorption coefficient.

    Args:
        layers: TMM layer sequence
        wavelengths: shape (n_wl,) in metres
        x: shape (N,) spatial grid [m]
        layer_boundaries: shape (n_layers + 1,) cumulative boundaries [m]
        n_ambient, n_substrate: bounding media indices

    Returns:
        A: shape (N, n_wl) — spectral absorption rate [m^-1]
    """
    N = len(x)
    n_wl = len(wavelengths)
    n_layers = len(layers)

    # Map each grid point to its TMM layer index and local position
    layer_indices, local_positions = _map_to_layers(
        x, layer_boundaries, n_layers,
    )

    # Incoherent first-layer bypass. The thick slab absorbs Beer-Lambert
    # style (no phase, no interference); everything past it sees the
    # coherent sub-stack with an entering intensity of (1-R_front)*T_bulk
    # and a new ambient index equal to the slab's real part.
    if layers and layers[0].incoherent:
        R_front, T_bulk, n_real = _incoherent_front_factors(
            layers[0], wavelengths, n_ambient,
        )

        A = np.zeros((N, n_wl))
        # Glass / thick-slab absorption: monotonic power decay.
        # |E(x)|^2 / |E_0|^2 = (1 - R_front) * exp(-alpha_glass * x_local)
        # Then A(x, lam) = alpha_glass * |E|^2 in the absorbing slab.
        slab_mask = layer_indices == 0
        if np.any(slab_mask):
            x_local_slab = local_positions[slab_mask]
            k_slab = layers[0].k
            alpha_slab = 4.0 * np.pi * k_slab / wavelengths  # (n_wl,)
            decay = np.exp(-alpha_slab[None, :] * x_local_slab[:, None])
            E_sq_slab = (1.0 - R_front)[None, :] * decay
            A[slab_mask, :] = alpha_slab[None, :] * E_sq_slab

        # Sub-stack: compute absorption on layers[1:] with the slab's
        # real index as ambient (per-lambda array), then scale by
        # (1-R_front)*T_bulk — the power that reaches the sub-stack
        # after the Fresnel bounce and bulk attenuation.
        sub_mask = layer_indices >= 1
        if np.any(sub_mask) and n_layers > 1:
            x0_sub = layer_boundaries[1]
            x_sub = x[sub_mask] - x0_sub
            boundaries_sub = layer_boundaries[1:] - x0_sub
            scale = (1.0 - R_front) * T_bulk  # (n_wl,)
            A_sub = tmm_absorption_profile(
                layers[1:], wavelengths, x_sub, boundaries_sub,
                n_ambient=n_real,
                n_substrate=n_substrate,
            )  # (n_sub, n_wl)
            A[sub_mask, :] = A_sub * scale[None, :]

        return A

    E_sq = _electric_field_profile(
        layers, wavelengths, x, layer_indices, local_positions,
        n_ambient, n_substrate,
    )
    # (N, n_wl)

    # Fractional absorption rate per unit length (Pettersson et al.):
    # a(x, lam) = (4*pi*n*k) / (lam * n_ambient) * |E(x)|^2
    # The n/n_ambient factor converts from electric field energy to
    # Poynting vector ratio relative to the incident beam.
    A = np.zeros((N, n_wl))
    for j in range(n_layers):
        mask = layer_indices == j
        if not np.any(mask):
            continue
        n_j = layers[j].n  # (n_wl,) real part
        k_j = layers[j].k  # (n_wl,)
        coeff_j = 4.0 * np.pi * n_j * k_j / (wavelengths * n_ambient)  # (n_wl,)
        A[mask, :] = E_sq[mask, :] * coeff_j[None, :]

    return A


def tmm_generation(
    layers: Sequence[TMMLayer],
    wavelengths: np.ndarray,
    spectral_flux: np.ndarray,
    x: np.ndarray,
    layer_boundaries: np.ndarray,
    n_ambient: float = 1.0,
    n_substrate: float = 1.0,
    return_absorbance: bool = False,
):
    """Compute POINT-SAMPLED generation rate G(x) [m^-3 s^-1].

    Integrates the absorbed spectral photon flux over wavelength:
        G(x) = integral A(x, lambda) * Phi_AM1.5(lambda) d_lambda

    NOT photon-conserving on the solver's mesh -- do not feed the result to
    ``MaterialArrays.G_optical``.  This samples the volumetric rate AT the
    nodes; the RHS then integrates it with the node-centred rectangle rule
    ``sum_i G[i]*dx_cell[i]``, which over-counts the sharply peaked
    front-of-absorber generation.  Measured on
    ``configs/ionmonger_benchmark_tmm.yaml`` (analytic absorbed budget
    225.001 A/m^2, hard ceiling q*Phi*(1-R-T) = 226.672): this function
    yields 353.353 at N_grid=10, 262.855 at 15, 238.971 at 30, 228.181 at
    60, 226.253 at 100, 225.365 at 200 -- converging to the budget from
    ABOVE and, at N_grid=10, claiming 57 % more photocurrent than the stack
    can physically absorb.

    Use :func:`tmm_absorbed_photon_flux_per_cell` divided by
    ``physics.generation.dual_cell_widths`` instead -- that is exact on any
    mesh (see ``solver/mol.py:_compute_tmm_generation``).  This function is
    kept for the pointwise-diagnostic use it is still correct for.

    Args:
        layers: TMM layer stack
        wavelengths: shape (n_wl,) in metres
        spectral_flux: shape (n_wl,) photon flux [m^-2 s^-1 m^-1]
        x: shape (N,) spatial grid [m]
        layer_boundaries: shape (n_layers + 1,) cumulative layer boundaries [m]
        n_ambient, n_substrate: bounding media refractive indices
        return_absorbance: if True, also return the spectral absorption
            rate ``A(x, lambda)`` [m^-1] of shape (N, n_wl). Used by the
            photon-recycling layer to estimate the escape probability.

    Returns:
        G: shape (N,) — generation rate [m^-3 s^-1]; or ``(G, A)`` if
        ``return_absorbance`` is True.
    """
    A = tmm_absorption_profile(
        layers, wavelengths, x, layer_boundaries,
        n_ambient, n_substrate,
    )
    # A: (N, n_wl), spectral_flux: (n_wl,)
    # Integrate: G(x) = integral A(x, lam) * Phi(lam) d_lam
    # Using trapezoidal rule over wavelength
    integrand = A * spectral_flux[None, :]  # (N, n_wl)
    G = trapezoid(integrand, wavelengths, axis=1)
    if return_absorbance:
        return G, A
    return G


def _incoherent_front_factors(
    layer0: TMMLayer,
    wavelengths: np.ndarray,
    n_ambient: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Air -> thick-slab Fresnel and bulk Beer-Lambert factors.

    Power form — no phase — because a ~1 mm glass substrate has a
    coherence length far shorter than its thickness, so the many
    internal round trips average out interference. Using exp(-2*pi*k*d/
    lambda) (field form) would reintroduce a phase and produce the
    very fringes we are trying to kill.

    Returns:
        R_front: (n_wl,) — |r0|^2 for the air->slab interface
        T_bulk:  (n_wl,) — exp(-4*pi*k*d / lambda), power transmission
                 through the bulk slab (intensity, not amplitude)
        n_real:  (n_wl,) — real part of slab index (ambient for sub-stack)
    """
    n_real = np.asarray(layer0.n, dtype=float)
    k_slab = np.asarray(layer0.k, dtype=float)
    r0 = (n_ambient - n_real) / (n_ambient + n_real)
    R_front = r0 * r0  # already real for a lossless air->dielectric interface
    # Power-form Beer-Lambert: alpha = 4*pi*k/lambda, T = exp(-alpha*d).
    # Do NOT replace with the field form exp(-2*pi*k*d/lambda) — that
    # keeps phase and reintroduces fringes through the thick substrate.
    alpha = 4.0 * np.pi * k_slab / wavelengths
    T_bulk = np.exp(-alpha * layer0.d)
    return R_front, T_bulk, n_real


def tmm_reflectance(
    layers: Sequence[TMMLayer],
    wavelengths: np.ndarray,
    n_ambient: float = 1.0,
    n_substrate: float = 1.0,
) -> np.ndarray:
    """Compute spectral reflectance R(lambda) of the stack.

    Returns:
        R: shape (n_wl,) — reflectance (power) at each wavelength
    """
    # Incoherent first-layer bypass (e.g. 1 mm glass substrate).
    if layers and layers[0].incoherent:
        # Validate the rest of the stack before doing any work.
        for j, lyr in enumerate(layers):
            if j > 0 and lyr.incoherent:
                raise ValueError(
                    "only the first layer may be incoherent "
                    f"(layer index {j} has incoherent=True)"
                )
        R_front, T_bulk, n_real = _incoherent_front_factors(
            layers[0], wavelengths, n_ambient,
        )
        # Sub-stack reflectance uses the real-valued slab index as its
        # ambient (per-wavelength array because n_glass varies with lambda).
        S_sub, _ = _transfer_matrix_stack(
            layers[1:], wavelengths,
            n_ambient=n_real,
            n_substrate=n_substrate,
        )
        r_sub = S_sub[:, 1, 0] / S_sub[:, 0, 0]
        R_sub = np.abs(r_sub) ** 2
        # Simple first-order incoherent sum: front Fresnel + one round
        # trip through the glass reflecting off the sub-stack. The
        # geometric series of further round trips is dropped because
        # for nonzero k_glass (or typical R_sub < 0.5) it contributes
        # < 2% and simplifies the book-keeping for R+T+A. See commit
        # message trailer "Not-tested: multi-pass geometric series".
        return R_front + (1.0 - R_front) ** 2 * (T_bulk ** 2) * R_sub

    S_total, _ = _transfer_matrix_stack(
        layers, wavelengths, n_ambient, n_substrate,
    )
    r = S_total[:, 1, 0] / S_total[:, 0, 0]
    return np.abs(r) ** 2
