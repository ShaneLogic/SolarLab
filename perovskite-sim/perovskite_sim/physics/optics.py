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

    S_total, S_partial = _transfer_matrix_stack(
        layers, wavelengths, n_ambient, n_substrate,
    )

    # Reflection coefficient of the full stack
    r = S_total[:, 1, 0] / S_total[:, 0, 0]  # (n_wl,)
    t = 1.0 / S_total[:, 0, 0]               # (n_wl,)

    # For each layer j, we need the transfer matrix from the left of layer j
    # to the right end of the stack (including the exit interface).
    # We build S_right[j] = (S_partial[j])^{-1} @ S_total
    # which maps the field at the left of layer j to the exit.

    # But it's easier to work with the forward/backward amplitudes.
    # At the left edge of the stack (ambient side), E+ = 1, E- = r.
    # Inside layer j at position x_local from its left edge:
    #   [E+_j(x), E-_j(x)] = P_j(x) @ I_{j-1,j} @ ... @ I_{0,1} @ P_1 @ ... @ [1, r]^T
    # where P_j(x) propagates only distance x_local, not full d_j.

    # We precompute the partial transfer from ambient through each interface.
    # S_before[j] = product from ambient to just INSIDE layer j (after interface j-1,j).

    def interface_matrix(n_a, n_b):
        r_ab = (n_a - n_b) / (n_a + n_b)
        t_ab = 2.0 * n_a / (n_a + n_b)
        I_mat = np.zeros((n_wl, 2, 2), dtype=complex)
        I_mat[:, 0, 0] = 1.0 / t_ab
        I_mat[:, 0, 1] = r_ab / t_ab
        I_mat[:, 1, 0] = r_ab / t_ab
        I_mat[:, 1, 1] = 1.0 / t_ab
        return I_mat

    # Build S_before[j]: transfer matrix from ambient to just inside layer j
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

    def _inv2x2(M: np.ndarray) -> np.ndarray:
        """Batched 2x2 matrix inverse, shape (n_wl, 2, 2)."""
        det = M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0]
        inv = np.empty_like(M)
        inv[:, 0, 0] = M[:, 1, 1] / det
        inv[:, 0, 1] = -M[:, 0, 1] / det
        inv[:, 1, 0] = -M[:, 1, 0] / det
        inv[:, 1, 1] = M[:, 0, 0] / det
        return inv

    # Compute |E|^2 at each grid point
    E_sq = np.zeros((N, n_wl))

    for j in range(n_layers):
        mask = layer_indices == j
        if not np.any(mask):
            continue
        x_local = local_positions[mask]  # (n_j,)
        n_c = layers[j].n_complex        # (n_wl,)

        # S_before[j] maps [a_j, b_j] -> [1, r], so invert to get
        # the field amplitudes at the left edge of layer j:
        # [a_j, b_j] = S_before[j]^{-1} @ [1, r]
        S_inv = _inv2x2(S_before[j])
        E_j = np.einsum("wij,wj->wi", S_inv, E_in)  # (n_wl, 2)
        E_plus = E_j[:, 0]   # (n_wl,) forward amplitude
        E_minus = E_j[:, 1]  # (n_wl,) backward amplitude

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
    layer_indices = np.zeros(N, dtype=int)
    local_positions = np.zeros(N)
    for j in range(n_layers):
        x_lo = layer_boundaries[j]
        x_hi = layer_boundaries[j + 1]
        mask = (x >= x_lo - 1e-15) & (x <= x_hi + 1e-15)
        layer_indices[mask] = j
        local_positions[mask] = x[mask] - x_lo

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
) -> np.ndarray:
    """Compute position-resolved generation rate G(x) [m^-3 s^-1].

    Integrates the absorbed spectral photon flux over wavelength:
        G(x) = integral A(x, lambda) * Phi_AM1.5(lambda) d_lambda

    Args:
        layers: TMM layer stack
        wavelengths: shape (n_wl,) in metres
        spectral_flux: shape (n_wl,) photon flux [m^-2 s^-1 m^-1]
        x: shape (N,) spatial grid [m]
        layer_boundaries: shape (n_layers + 1,) cumulative layer boundaries [m]
        n_ambient, n_substrate: bounding media refractive indices

    Returns:
        G: shape (N,) — generation rate [m^-3 s^-1]
    """
    A = tmm_absorption_profile(
        layers, wavelengths, x, layer_boundaries,
        n_ambient, n_substrate,
    )
    # A: (N, n_wl), spectral_flux: (n_wl,)
    # Integrate: G(x) = integral A(x, lam) * Phi(lam) d_lam
    # Using trapezoidal rule over wavelength
    d_lam = np.diff(wavelengths)
    integrand = A * spectral_flux[None, :]  # (N, n_wl)
    G = np.trapz(integrand, wavelengths, axis=1)
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
