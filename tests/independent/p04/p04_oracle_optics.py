"""Independent optical-generation oracle for P04-07 (Beer-Lambert and TMM).

Beer-Lambert (monochromatic, incidence from x = 0, photon flux Phi in
m^-2 s^-1).  With optical depth tau(x) = int_0^x alpha, the photons absorbed
between a < b are Phi (e^{-tau(a)} - e^{-tau(b)}) (exact, any alpha(x) >= 0).
For layer-wise constant alpha_k::

    A_k = Phi exp(-sum_{j<k} alpha_j d_j) (1 - exp(-alpha_k d_k))
    sum_k A_k = Phi (1 - exp(-tau_total)) <= Phi                     (BL)

Control-volume generation G_i = (photons absorbed in cell i) / w_i makes the
discrete source sum_i G_i w_i telescope to (BL) on any mesh.  The historical
1D channel (``physics/generation.py``) uses nodal alpha with the RIGHT layer
owning an interface node and a mid-face trapezoid for tau, so its discrete
optical depth is sum_i (alpha_i + alpha_{i+1}) h_i / 2: exact inside each
layer, and equal to the true tau of a structure whose planes sit at the dual
face x_I - h_{I-1}/2 (the same half-interval shift as the Poisson rule).

Transfer matrix (normal incidence; Born & Wolf ch. 1.6; Byrnes,
arXiv:1603.02720).  Complex index N = n + i kappa (kappa >= 0 absorbs), field
in medium j at local depth z: E = v_j e^{i k_j z} + w_j e^{-i k_j z},
k_j = 2 pi N_j / lambda.  Continuity of E and H ~ N (F - B) gives::

    [v_j; w_j] = diag(e^{-i delta_j}, e^{i delta_j}) (1/t_{j,j+1}) [[1, r],[r, 1]] [v_{j+1}; w_{j+1}]
    r_{ab} = (N_a - N_b)/(N_a + N_b),  t_{ab} = 2 N_a/(N_a + N_b),  delta_j = k_j d_j

with [v_0; w_0] = [1; r] in the ambient and [t; 0] in the substrate.  The
time-averaged Poynting flux normalised to the incident beam is
S(z) = Re[conj(N) (F + B) conj(F - B)] / n_0, so R = |r|^2,
T = Re(N_sub) |t|^2 / n_0, and the layer absorptance is S(0) - S(d).  The
local absorption density is a(z) = alpha n |E|^2 / n_0 (alpha = 4 pi kappa /
lambda) and its integral is evaluated in closed form; agreement of the two
routes is the R + T + A = 1 identity the tests assert.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

H_PLANCK = 6.62607015e-34  # J s (exact SI)
C_LIGHT = 299792458.0  # m/s (exact SI)
Q_E = 1.602176634e-19  # C (exact SI)


# --------------------------------------------------------------------------
# Beer-Lambert
# --------------------------------------------------------------------------


def absorbed_between(tau_a: np.ndarray, tau_b: np.ndarray, Phi: float) -> np.ndarray:
    """Phi (e^-tau_a - e^-tau_b), evaluated as Phi e^-tau_a (1 - e^-(tau_b - tau_a))."""

    tau_a = np.asarray(tau_a, dtype=float)
    tau_b = np.asarray(tau_b, dtype=float)
    return Phi * np.exp(-tau_a) * -np.expm1(-(tau_b - tau_a))


def bl_layer_absorption(alpha: tuple[float, ...], d: tuple[float, ...], Phi: float) -> np.ndarray:
    """Equation (BL) per layer, photons m^-2 s^-1."""

    tau = np.concatenate(([0.0], np.cumsum(np.asarray(alpha) * np.asarray(d))))
    return absorbed_between(tau[:-1], tau[1:], Phi)


def exact_optical_depth(xq: np.ndarray, alpha: tuple[float, ...], d: tuple[float, ...]) -> np.ndarray:
    """tau(x) of a layer-wise constant alpha with sharp planes at the true bounds."""

    xq = np.asarray(xq, dtype=float)
    b = np.concatenate(([0.0], np.cumsum(d)))
    tau = np.zeros_like(xq)
    for k, a in enumerate(alpha):
        tau += a * np.clip(xq - b[k], 0.0, b[k + 1] - b[k])
    return tau


def nodal_trapezoid_optical_depth(x: np.ndarray, alpha_node: np.ndarray) -> np.ndarray:
    """Historical channel: tau_i = sum_{j<i} (alpha_j + alpha_{j+1}) h_j / 2."""

    x = np.asarray(x, dtype=float)
    alpha_node = np.asarray(alpha_node, dtype=float)
    tau = np.zeros_like(x)
    tau[1:] = np.cumsum(0.5 * (alpha_node[:-1] + alpha_node[1:]) * np.diff(x))
    return tau


def mirror_stack_generation(G_mirrored: np.ndarray) -> np.ndarray:
    """Illumination from x = L equals the mirrored stack read backwards."""

    return np.asarray(G_mirrored)[::-1]


# --------------------------------------------------------------------------
# Spectral units
# --------------------------------------------------------------------------


def photon_flux_per_m_from_irradiance_per_nm(E_W_m2_nm: np.ndarray, wl_nm: np.ndarray) -> np.ndarray:
    """Phi_lambda [m^-2 s^-1 m^-1] = E [W m^-2 nm^-1] * 1e9 * lambda / (h c)."""

    wl_m = np.asarray(wl_nm, dtype=float) * 1e-9
    return np.asarray(E_W_m2_nm, dtype=float) * 1e9 * wl_m / (H_PLANCK * C_LIGHT)


# --------------------------------------------------------------------------
# Transfer matrix
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TmmResult:
    wavelengths_m: np.ndarray
    N: tuple[np.ndarray, ...]  # media 0..M+1 (complex, per wavelength)
    d: tuple[float, ...]  # finite layers 1..M
    r: np.ndarray
    t: np.ndarray
    v_start: tuple[np.ndarray, ...]  # forward amplitude at each layer start
    b_end: tuple[np.ndarray, ...]  # backward amplitude at each layer end

    @property
    def n_ambient(self) -> np.ndarray:
        return self.N[0].real

    @property
    def R(self) -> np.ndarray:
        return np.abs(self.r) ** 2

    @property
    def T(self) -> np.ndarray:
        return self.N[-1].real * np.abs(self.t) ** 2 / self.n_ambient

    def _k(self, j: int) -> np.ndarray:
        return 2.0 * np.pi * self.N[j + 1] / self.wavelengths_m

    def layer_absorptance_poynting(self) -> np.ndarray:
        """S(0) - S(d) per finite layer, shape (M, n_wl)."""

        out = []
        for j, dj in enumerate(self.d):
            N, k = self.N[j + 1], self._k(j)
            v, b = self.v_start[j], self.b_end[j]
            w = b * np.exp(1j * k * dj)  # backward amplitude at the layer start
            a = v * np.exp(1j * k * dj)  # forward amplitude at the layer end
            s0 = np.real(np.conj(N) * (v + w) * np.conj(v - w))
            s1 = np.real(np.conj(N) * (a + b) * np.conj(a - b))
            out.append((s0 - s1) / self.n_ambient)
        return np.array(out)

    def cumulative_absorbed(self, j: int, u: np.ndarray) -> np.ndarray:
        """int_0^u a(z) dz in layer j (0-based), shape (len(u), n_wl), closed form."""

        u = np.atleast_1d(np.asarray(u, dtype=float))[:, None]
        N, k, dj = self.N[j + 1], self._k(j), self.d[j]
        kr, ki = k.real[None, :], k.imag[None, :]
        v, b = self.v_start[j][None, :], self.b_end[j][None, :]
        alpha = 2.0 * ki  # = 4 pi kappa / lambda
        fwd = np.abs(v) ** 2 * u * _one_minus_exp_over(alpha * u)
        bwd = np.abs(b) ** 2 * np.exp(-alpha * (dj - u)) * u * _one_minus_exp_over(alpha * u)
        cross_amp = v * np.conj(b) * np.exp(-1j * np.conj(k[None, :]) * dj)
        s = 2.0 * kr * u
        osc = u * (np.sinc(s / np.pi) + 1j * (s / 2.0) * np.sinc(s / (2.0 * np.pi)) ** 2)
        cross = 2.0 * np.real(cross_amp * osc)
        coeff = alpha * N.real[None, :] / self.n_ambient[None, :]
        return coeff * (fwd + bwd + cross)

    def layer_absorptance_integral(self) -> np.ndarray:
        return np.array([self.cumulative_absorbed(j, np.array([dj]))[0] for j, dj in enumerate(self.d)])


def _one_minus_exp_over(s: np.ndarray) -> np.ndarray:
    """(1 - e^-s)/s with the s -> 0 limit 1."""

    s = np.asarray(s, dtype=float)
    out = np.ones_like(s)
    big = s > 1e-12
    out[big] = -np.expm1(-s[big]) / s[big]
    small = ~big
    out[small] = 1.0 - s[small] / 2.0
    return out


def tmm_normal_incidence(
    wavelengths_m: np.ndarray,
    layers: list[tuple[np.ndarray, float]],
    n_ambient: float = 1.0,
    n_substrate: complex | np.ndarray = 1.0,
) -> TmmResult:
    """Coherent stack; ``layers`` = [(N_complex(lambda), thickness_m), ...]."""

    wl = np.asarray(wavelengths_m, dtype=float)
    nwl = wl.size
    N = [np.full(nwl, complex(n_ambient))]
    N += [np.broadcast_to(np.asarray(n, dtype=complex), (nwl,)).copy() for n, _ in layers]
    N.append(np.broadcast_to(np.asarray(n_substrate, dtype=complex), (nwl,)).copy())
    d = tuple(float(t) for _, t in layers)
    M = len(d)

    def interface(a: np.ndarray, b: np.ndarray, vec: np.ndarray) -> np.ndarray:
        r = (a - b) / (a + b)
        t = 2.0 * a / (a + b)
        return np.stack([(vec[0] + r * vec[1]) / t, (r * vec[0] + vec[1]) / t])

    # Backward recursion with the unknown t set to 1.  Amplitudes grow like
    # e^{Im(delta)} per absorbing layer, so every step is renormalised and the
    # growth is carried in a per-wavelength log scale (no overflow).
    vec = np.stack([np.ones(nwl, dtype=complex), np.zeros(nwl, dtype=complex)])
    log_s = np.zeros(nwl)
    stored: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    def renorm(arr: np.ndarray, log_scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m = np.maximum(np.abs(arr[0]), np.abs(arr[1]))
        m = np.where(m > 0.0, m, 1.0)
        return arr / m, log_scale + np.log(m)

    for j in range(M, 0, -1):
        end, log_s = renorm(interface(N[j], N[j + 1], vec), log_s)
        b_norm, log_end = end[1].copy(), log_s.copy()
        delta = 2.0 * np.pi * N[j] * d[j - 1] / wl
        dr, di = delta.real, delta.imag
        start = np.stack([
            end[0] * np.exp(-1j * dr),
            end[1] * np.exp(1j * dr) * np.exp(-2.0 * di),
        ])
        vec, log_s = renorm(start, log_s + di)
        stored.append((vec[0].copy(), log_s.copy(), b_norm, log_end))
    top, log_s = renorm(interface(N[0], N[1], vec), log_s)
    r = top[1] / top[0]
    t = np.exp(-log_s) / top[0]
    v_start: list[np.ndarray] = [None] * M  # type: ignore[list-item]
    b_end: list[np.ndarray] = [None] * M  # type: ignore[list-item]
    for (v_norm, log_v, b_norm, log_b), j in zip(stored, range(M - 1, -1, -1)):
        v_start[j] = v_norm * np.exp(log_v - log_s) / top[0]
        b_end[j] = b_norm * np.exp(log_b - log_s) / top[0]
    return TmmResult(wl, tuple(N), d, r, t, tuple(v_start), tuple(b_end))


def airy_slab(
    wavelengths_m: np.ndarray, n0: float, N1: complex, d: float, N2: complex
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form (R, T) of one slab between semi-infinite media."""

    wl = np.asarray(wavelengths_m, dtype=float)
    r01 = (n0 - N1) / (n0 + N1)
    r12 = (N1 - N2) / (N1 + N2)
    t01 = 2 * n0 / (n0 + N1)
    t12 = 2 * N1 / (N1 + N2)
    ph = np.exp(2j * (2 * np.pi * N1 * d / wl))
    r = (r01 + r12 * ph) / (1 + r01 * r12 * ph)
    t = t01 * t12 * np.exp(1j * 2 * np.pi * N1 * d / wl) / (1 + r01 * r12 * ph)
    return np.abs(r) ** 2, np.real(N2) * np.abs(t) ** 2 / n0


def fresnel_interface(n0: float, N1: complex) -> tuple[float, float]:
    r = (n0 - N1) / (n0 + N1)
    t = 2 * n0 / (n0 + N1)
    return float(abs(r) ** 2), float(np.real(N1) * abs(t) ** 2 / n0)


def trapezoid_weights(wl: np.ndarray) -> np.ndarray:
    """Composite trapezoid weights on an arbitrary wavelength grid."""

    wl = np.asarray(wl, dtype=float)
    h = np.diff(wl)
    w = np.zeros_like(wl)
    w[:-1] += h / 2.0
    w[1:] += h / 2.0
    return w


def electron_current_ceiling(Phi: float) -> float:
    """q Phi [A/m^2]: the hard photocurrent ceiling of an incident photon flux."""

    return Q_E * Phi
