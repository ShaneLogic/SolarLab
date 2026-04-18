"""Electroluminescence (EL) and non-radiative voltage loss via reciprocity.

Physics (Rau 2007)
------------------
Detailed-balance reciprocity links the photovoltaic external quantum
efficiency at a given wavelength to the electroluminescence that a cell
emits at the same wavelength under forward bias:

    Phi_EL(lambda) = EQE_PV(lambda) . phi_bb(lambda, T) . exp(q V / kT)

For the absorber-absorptance upper bound on EQE_PV (perfect carrier
collection), this becomes

    Phi_EL(lambda) = A_abs(lambda) . phi_bb(lambda, T) . exp(q V_inj / kT)

where ``A_abs(lambda) = int A(x, lambda) dx`` over absorber nodes is the
dimensionless fraction of incident photons absorbed by the active layer,
extracted from the same TMM machinery the main solver uses, and

    phi_bb(lambda, T) = (2 pi c / lambda^4) / (exp(hc / (lambda kT)) - 1)
                                                        [photons/m^2/s/m]

is the spectral photon flux of an ideal Lambertian blackbody at
temperature ``T``. Integrating gives the radiative limit of the
injection current:

    J_em_rad = q . integral Phi_EL(lambda) d_lambda

The actual injection current ``J_inj`` comes out of the drift-diffusion
solver at the same ``V_inj``. Their ratio is the external
electroluminescence efficiency:

    EQE_EL = J_em_rad / |J_inj|                                  [-]

and the corresponding non-radiative V_oc penalty is

    dV_nr = - (kT / q) . ln(EQE_EL)                              [V]

For a radiatively-dominated device ``EQE_EL`` approaches 1 and ``dV_nr``
approaches 0; for a SRH/Auger-limited one ``EQE_EL`` is small and the
penalty climbs into the hundreds of mV.

Implementation notes
--------------------
- Requires a TMM-capable stack (at least one layer with
  ``optical_material``). Beer-Lambert configs have no wavelength-
  resolved absorptance and raise ``ValueError`` - same guard ``compute_eqe``
  uses.
- The dark-current measurement reuses ``run_jv_sweep(illuminated=False)``
  so that ion equilibration up to ``V_inj`` is treated consistently with
  the rest of the simulator (no new solver path).
- ``A_abs(lambda)`` is the absorber integral only. Parasitic absorption
  in transport layers and the substrate is omitted from the emission
  pathway even though it does appear in ``A(x, lambda)`` - only photons
  produced inside the absorber have a chance of radiatively escaping;
  photons absorbed in a contact layer are effectively lost.
- The ``exp(q V_inj / kT)`` term is a flat-quasi-Fermi-level assumption
  that's exact for an ideal absorber and approximate otherwise. In the
  moderate-injection regime relevant to operating photovoltaics the
  error is small; at very high forward bias (near flat-band) it over-
  estimates emission because the actual quasi-Fermi splitting saturates.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from perovskite_sim.constants import K_B, Q
from perovskite_sim.data import load_nk
from perovskite_sim.discretization.grid import multilayer_grid, Layer
from perovskite_sim.experiments.jv_sweep import run_jv_sweep
from perovskite_sim.models.device import DeviceStack, electrical_layers
from perovskite_sim.models.el import ELResult
from perovskite_sim.physics.optics import TMMLayer, tmm_absorption_profile

ProgressCallback = Callable[[str, int, int, str], None]

# Planck's constant [J s] and speed of light in vacuum [m/s]. Defined
# locally because the rest of the package never needed them until
# reciprocity-based EL landed.
H_PLANCK = 6.62607015e-34
C_LIGHT = 299_792_458.0


def _require_tmm_optical_data(stack: DeviceStack) -> None:
    """Raise a helpful error if no layer has wavelength-resolved optics."""
    has_optical = any(
        layer.params is not None and layer.params.optical_material is not None
        for layer in stack.layers
    )
    if not has_optical:
        raise ValueError(
            "run_el_spectrum requires at least one layer with "
            "optical_material set (tabulated n, k). Use one of the "
            "*_tmm.yaml configs or add optical_material to your stack."
        )


def _build_tmm_layers(
    stack: DeviceStack, wavelengths_nm: np.ndarray
) -> tuple[list[TMMLayer], np.ndarray, float]:
    """Build a TMM layer stack for every wavelength in ``wavelengths_nm``.

    Mirrors the layer construction in ``solver.mol._compute_tmm_generation``:
    layers with tabulated ``optical_material`` load n, k from CSV; layers
    with only a constant ``n_optical`` fall back to
    ``k = alpha * lambda / (4 pi)``; layers with neither estimate ``n``
    from ``sqrt(eps_r)``.

    Returns (tmm_layers, layer_boundaries [m], substrate_offset [m]).
    """
    wavelengths_m = wavelengths_nm * 1e-9
    n_wl = len(wavelengths_nm)

    tmm_layers: list[TMMLayer] = []
    for layer in stack.layers:
        p = layer.params
        if p is None:
            raise ValueError(
                f"Layer {layer.name!r} has no MaterialParams; cannot build "
                "TMM stack for EL."
            )
        if p.optical_material is not None:
            _, n_arr, k_arr = load_nk(p.optical_material, wavelengths_nm)
        elif p.n_optical is not None:
            n_arr = np.full(n_wl, p.n_optical, dtype=float)
            k_arr = p.alpha * wavelengths_m / (4.0 * np.pi)
        else:
            n_arr = np.full(n_wl, np.sqrt(p.eps_r), dtype=float)
            k_arr = p.alpha * wavelengths_m / (4.0 * np.pi)
        tmm_layers.append(
            TMMLayer(
                d=layer.thickness, n=n_arr, k=k_arr,
                incoherent=bool(p.incoherent),
            )
        )

    boundaries = np.zeros(len(stack.layers) + 1)
    for i, layer in enumerate(stack.layers):
        boundaries[i + 1] = boundaries[i] + layer.thickness
    substrate_offset = sum(
        l.thickness for l in stack.layers if l.role == "substrate"
    )
    return tmm_layers, boundaries, substrate_offset


def _absorber_mask(x: np.ndarray, stack: DeviceStack) -> np.ndarray:
    """Boolean mask over electrical-grid nodes selecting absorber layers.

    The drift-diffusion grid ``x`` only spans non-substrate layers. We
    walk through ``electrical_layers(stack)`` in order, accumulating the
    thickness bounds, and mark any node whose position falls inside a
    layer tagged ``role == 'absorber'``. If the stack has no absorber
    role, every electrical node is treated as absorber so the experiment
    still produces a well-defined result on legacy configs.
    """
    elec = electrical_layers(stack)
    if not any(l.role == "absorber" for l in elec):
        return np.ones_like(x, dtype=bool)

    mask = np.zeros_like(x, dtype=bool)
    cum = 0.0
    # 1e-15 matches the tolerance tmm_absorption_profile uses to map grid
    # points to TMM layers, so node-to-layer mapping stays consistent.
    eps = 1e-15
    for layer in elec:
        x_lo = cum
        x_hi = cum + layer.thickness
        if layer.role == "absorber":
            mask |= (x >= x_lo - eps) & (x <= x_hi + eps)
        cum = x_hi
    return mask


def _blackbody_photon_flux(
    wavelengths_m: np.ndarray, T: float
) -> np.ndarray:
    """Lambertian-integrated blackbody spectral photon flux [photons/m^2/s/m].

        phi_bb(lambda, T) = (2 pi c / lambda^4) / (exp(hc / (lambda kT)) - 1)

    The exponent at visible wavelengths and T~300 K is huge (~60), so we
    evaluate ``exp(-hc/lambda kT) / (1 - exp(-hc/lambda kT))`` to avoid
    overflow when ``exp(+60)`` would otherwise be taken.
    """
    x = H_PLANCK * C_LIGHT / (wavelengths_m * K_B * T)
    # Clamp to avoid 1/(exp(x)-1) overflow at tiny lambda; the result is
    # vanishingly small there anyway.
    x_safe = np.clip(x, 0.0, 700.0)
    e_neg = np.exp(-x_safe)
    # e_neg / (1 - e_neg) = 1 / (exp(x) - 1), numerically stable for x>>1
    denom = 1.0 - e_neg
    denom = np.where(denom > 1e-300, denom, 1.0)
    return (2.0 * np.pi * C_LIGHT / wavelengths_m ** 4) * (e_neg / denom)


def run_el_spectrum(
    stack: DeviceStack,
    V_inj: float = 1.0,
    wavelengths_nm: np.ndarray | None = None,
    N_grid: int = 60,
    n_points_dark: int = 30,
    v_rate: float = 1.0,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    progress: ProgressCallback | None = None,
) -> ELResult:
    """Compute the reciprocity EL spectrum and EQE_EL / dV_nr.

    Parameters
    ----------
    stack : DeviceStack
        Must carry at least one layer with ``optical_material`` set so the
        wavelength-resolved absorptance is well defined.
    V_inj : float, default 1.0
        Forward bias [V] at which to inject carriers and evaluate the
        reciprocity spectrum. 1.0 V is close to the operating point of
        a perovskite cell at open circuit.
    wavelengths_nm : np.ndarray, optional
        Probe wavelengths [nm]. Default: 400-1000 nm in 25-point linear
        grid, which covers the MAPbI3 absorption edge around 780 nm.
    N_grid : int, default 60
        Total drift-diffusion nodes across electrical layers.
    n_points_dark : int, default 30
        Voltage samples in the dark forward sweep from 0 to ``V_inj``
        used to obtain ``J_inj``.
    v_rate : float, default 1.0
        Quasi-static sweep rate [V/s].
    rtol, atol : float
        scipy solver tolerances forwarded to ``run_jv_sweep``.
    progress : ProgressCallback | None

    Returns
    -------
    ELResult
    """
    _require_tmm_optical_data(stack)

    if V_inj <= 0.0:
        raise ValueError(f"V_inj must be positive, got {V_inj}")
    if N_grid < 3:
        raise ValueError(f"N_grid must be >= 3, got {N_grid}")

    if wavelengths_nm is None:
        wavelengths_nm = np.linspace(400.0, 1000.0, 25)
    wavelengths_nm = np.asarray(sorted(wavelengths_nm), dtype=float)
    if wavelengths_nm.size < 2:
        raise ValueError(
            f"wavelengths_nm must have at least 2 points, got {wavelengths_nm.size}"
        )
    if np.any(wavelengths_nm <= 0.0):
        raise ValueError(f"wavelengths_nm must be positive, got {wavelengths_nm}")

    wavelengths_m = wavelengths_nm * 1e-9
    T = float(stack.T)
    V_T = K_B * T / Q

    if progress is not None:
        progress("el", 0, 4, "building TMM stack")

    elec = electrical_layers(stack)
    n_per = max(N_grid // len(elec), 2)
    layers_grid = [Layer(l.thickness, n_per) for l in elec]
    x = multilayer_grid(layers_grid)

    tmm_layers, boundaries, substrate_offset = _build_tmm_layers(
        stack, wavelengths_nm,
    )
    x_tmm = x + substrate_offset

    if progress is not None:
        progress("el", 1, 4, "computing A(x, lambda) via TMM")

    # A(x, lambda) [m^-1] on electrical nodes at every probe wavelength.
    A_xl = tmm_absorption_profile(
        tmm_layers, wavelengths_m, x_tmm, boundaries,
    )

    # Integrate spatial absorption over the absorber to get the
    # dimensionless spectral absorptance A_abs(lambda). trapz over x
    # with A in m^-1 gives fraction of photons absorbed per unit incident
    # flux - dimensionless.
    abs_mask = _absorber_mask(x, stack)
    x_abs = x[abs_mask]
    if x_abs.size < 2:
        raise ValueError(
            "absorber mask has fewer than 2 grid nodes; increase N_grid."
        )
    A_abs = np.trapezoid(A_xl[abs_mask, :], x_abs, axis=0)
    # Clamp A_abs into [0, 1]; tiny numerical leaks out of this band are
    # unphysical and would push exp(qV/kT) into noise.
    A_abs = np.clip(A_abs, 0.0, 1.0)

    if progress is not None:
        progress("el", 2, 4, "evaluating reciprocity at V_inj")

    # Reciprocity EL photon flux [photons/m^2/s/m]. Note: exp(qV/kT) is
    # applied to every wavelength (flat-quasi-Fermi-level approximation).
    phi_bb = _blackbody_photon_flux(wavelengths_m, T)
    boltzmann_factor = np.exp(min(V_inj / V_T, 700.0))
    Phi_EL = A_abs * phi_bb * boltzmann_factor

    # Radiative emission current [A/m^2] = q . integral Phi_EL d_lambda.
    J_em_rad = float(Q * np.trapezoid(Phi_EL, wavelengths_m))

    # EL_spectrum reported in photons / m^2 / s / nm for plotting
    # convenience; storage wavelength axis is already in nm.
    EL_spectrum_per_nm = Phi_EL * 1e-9

    if progress is not None:
        progress("el", 3, 4, f"dark J-V sweep to V_inj = {V_inj:.3f} V")

    # Dark forward sweep 0 -> V_inj. J_inj is the last forward point.
    jv = run_jv_sweep(
        stack,
        N_grid=N_grid,
        n_points=max(n_points_dark, 2),
        v_rate=v_rate,
        V_max=V_inj,
        rtol=rtol,
        atol=atol,
        illuminated=False,
        progress=None,  # don't double-report inside the EL progress stream
    )
    J_inj = float(jv.J_fwd[-1])

    # |J_inj| guards against the sign of the dark injection current
    # (negative under our solar convention since carriers flow IN) and
    # against a pathological zero-current solver result.
    J_inj_mag = abs(J_inj)
    if J_inj_mag < 1e-30:
        EQE_EL = 0.0
        delta_V_nr_V = 0.0
    else:
        EQE_EL = J_em_rad / J_inj_mag
        # EQE_EL can overshoot 1.0 when the reciprocity-upper-bound A_abs
        # exceeds the actual absorbed fraction in the solver's drift-
        # diffusion model - that is an assumption boundary, not a physics
        # inconsistency. Clamp to (0, 1] before taking the log so
        # delta_V_nr is never negative (which would be unphysical).
        EQE_EL_clamped = min(max(EQE_EL, 1e-30), 1.0)
        delta_V_nr_V = -(V_T) * np.log(EQE_EL_clamped)

    if progress is not None:
        progress(
            "el", 4, 4,
            f"EQE_EL = {EQE_EL:.2e}, dV_nr = {delta_V_nr_V * 1e3:.1f} mV",
        )

    return ELResult(
        wavelengths_nm=wavelengths_nm,
        EL_spectrum=EL_spectrum_per_nm,
        absorber_absorptance=A_abs,
        V_inj=float(V_inj),
        J_inj=float(J_inj),
        J_em_rad=J_em_rad,
        EQE_EL=float(EQE_EL),
        delta_V_nr_mV=float(delta_V_nr_V * 1e3),
        T=T,
    )
