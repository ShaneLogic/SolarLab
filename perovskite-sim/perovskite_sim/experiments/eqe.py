"""External quantum efficiency (EQE / IPCE) characterisation.

Physics
-------
EQE(λ) is the fraction of incident photons at wavelength λ that become
collected electrons at short circuit:

    EQE(λ) = J_sc(λ) / (q · Φ_inc(λ))

where Φ_inc is the monochromatic incident photon flux. Integrated over
the AM1.5G reference spectrum, the collected photocurrent is

    J_sc_integrated = q · ∫ EQE(λ) · Φ_AM15G(λ) dλ

which must match (to within discretisation and TMM grid error) the
J_sc a full-spectrum simulation produces at V = 0. This is the standard
EQE ↔ J_sc cross-check experimental groups run to validate their
EQE measurement against the AM1.5G reference.

Implementation
--------------
For each wavelength we:

1. Call ``tmm_absorption_profile`` with a single-wavelength array to get
   A(x) [m⁻¹] — the position-resolved absorption coefficient. This
   sidesteps the spectral integration in ``tmm_generation`` (which needs
   ≥2 wavelengths for ``np.trapezoid``) while reusing the same TMM
   machinery the main simulation trusts.
2. Scale by the chosen probe flux: G(x) = A(x) · Φ_inc [m⁻³ s⁻¹].
3. Build a MaterialArrays bundle with G_optical = G, settle the device
   to illuminated steady-state at V_app = 0, and read J_sc from
   ``_compute_current``.

Beer-Lambert–only stacks raise a ``ValueError`` — EQE is a
wavelength-resolved quantity, and MaterialParams carries only a scalar
α (absorption at the design wavelength). Without tabulated n(λ), k(λ)
optical data (``optical_material``) the TMM absorption spectrum is not
defined.

Sign convention
---------------
Matches ``run_jv_sweep``: under illumination at V=0, J > 0 (photocurrent
leaves the absorber contact). EQE is reported as its unsigned
magnitude, which is the sign experimentalists plot.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Callable

import numpy as np

from perovskite_sim.constants import Q
from perovskite_sim.data import load_am15g, load_nk
from perovskite_sim.discretization.grid import multilayer_grid, Layer
from perovskite_sim.experiments.jv_sweep import _compute_current
from perovskite_sim.experiments.suns_voc import _solve_illuminated_ss_with_mat
from perovskite_sim.models.device import DeviceStack, electrical_layers
from perovskite_sim.physics.optics import TMMLayer, tmm_absorption_profile
from perovskite_sim.solver.mol import (
    MaterialArrays,
    build_material_arrays,
)

ProgressCallback = Callable[[str, int, int, str], None]


@dataclass(frozen=True)
class EQEResult:
    """EQE curve and integrated photocurrent.

    Attributes
    ----------
    wavelengths_nm : np.ndarray
        Probe wavelengths [nm], sorted ascending.
    EQE : np.ndarray
        External quantum efficiency at each λ, dimensionless in [0, 1]
        physically (numerical noise may nudge edge points slightly
        outside — see test bands).
    J_sc_per_lambda : np.ndarray
        Short-circuit current density [A/m²] under monochromatic
        illumination at ``Phi_incident`` photons/m²/s. Useful for
        debugging individual-λ behaviour.
    J_sc_integrated : float
        J_sc predicted by integrating EQE against AM1.5G
        [A/m²]. Should match a full-spectrum TMM simulation at V=0 to
        within ~10-20 % (wavelength-grid discretisation error).
    Phi_incident : float
        Monochromatic probe flux [m⁻² s⁻¹] used for each wavelength.
        EQE is independent of this choice to first order, but J_sc
        per λ scales linearly with it.
    """
    wavelengths_nm: np.ndarray
    EQE: np.ndarray
    J_sc_per_lambda: np.ndarray
    J_sc_integrated: float
    Phi_incident: float


def _build_tmm_layers_single_wavelength(
    stack: DeviceStack, lam_m: float
) -> tuple[list[TMMLayer], np.ndarray]:
    """Return (tmm_layers, layer_boundaries) for a single-wavelength query.

    Mirrors ``solver.mol._compute_tmm_generation``'s layer construction
    but at a single wavelength. Layers with tabulated n, k data
    (``optical_material``) use ``load_nk``; layers with only a scalar
    ``n_optical`` and α fall back to k = α·λ/(4π) — same fallbacks the
    full-spectrum path uses, so monochromatic EQE is consistent with the
    integrated simulation.
    """
    wavelengths_m = np.array([lam_m], dtype=float)
    wavelengths_nm = wavelengths_m * 1e9

    tmm_layers: list[TMMLayer] = []
    for layer in stack.layers:
        p = layer.params
        if p is None:
            raise ValueError(
                f"Layer {layer.name!r} has no MaterialParams; cannot build "
                "TMM stack for EQE."
            )
        if p.optical_material is not None:
            _, n_arr, k_arr = load_nk(p.optical_material, wavelengths_nm)
        elif p.n_optical is not None:
            n_arr = np.full(1, p.n_optical, dtype=float)
            k_arr = np.array([p.alpha * lam_m / (4.0 * np.pi)], dtype=float)
        else:
            n_arr = np.full(1, np.sqrt(p.eps_r), dtype=float)
            k_arr = np.array([p.alpha * lam_m / (4.0 * np.pi)], dtype=float)
        tmm_layers.append(
            TMMLayer(
                d=layer.thickness, n=n_arr, k=k_arr,
                incoherent=bool(p.incoherent),
            )
        )

    boundaries = np.zeros(len(stack.layers) + 1)
    for i, layer in enumerate(stack.layers):
        boundaries[i + 1] = boundaries[i] + layer.thickness
    return tmm_layers, boundaries


def _single_wavelength_generation(
    x: np.ndarray, stack: DeviceStack, lam_m: float, Phi_inc: float
) -> np.ndarray:
    """Compute G(x) [m⁻³ s⁻¹] under monochromatic illumination.

    A(x, λ) [m⁻¹] from TMM · Φ_inc [m⁻² s⁻¹] yields G in m⁻³ s⁻¹. The
    TMM query runs on the *full* stack (substrate included) so the
    Fresnel chain is correct; the electrical grid ``x`` only covers the
    non-substrate layers so we shift it by the cumulative substrate
    thickness before handing to ``tmm_absorption_profile`` — same
    offset-shift ``solver.mol._compute_tmm_generation`` applies for the
    spectral path.
    """
    tmm_layers, boundaries = _build_tmm_layers_single_wavelength(stack, lam_m)
    substrate_offset = sum(
        l.thickness for l in stack.layers if l.role == "substrate"
    )
    x_tmm = x + substrate_offset
    wavelengths_m = np.array([lam_m], dtype=float)
    A = tmm_absorption_profile(
        tmm_layers, wavelengths_m, x_tmm, boundaries,
    )
    # A shape: (N, 1), units m^-1
    return np.ascontiguousarray(A[:, 0]) * Phi_inc


def _require_tmm_optical_data(stack: DeviceStack) -> None:
    """Raise a helpful error if no layer has wavelength-resolved optics.

    EQE needs n(λ), k(λ) tables so the absorption spectrum has the right
    shape across the probe range. A scalar α per layer (Beer-Lambert)
    cannot produce a meaningful EQE curve, so we bail out early with a
    clear message rather than silently returning a flat-ish EQE that
    integrates to the wrong J_sc.
    """
    has_optical = any(
        layer.params is not None and layer.params.optical_material is not None
        for layer in stack.layers
    )
    if not has_optical:
        raise ValueError(
            "compute_eqe requires at least one layer with optical_material "
            "set (tabulated n, k). Use one of the *_tmm.yaml configs (e.g. "
            "nip_MAPbI3_tmm.yaml) or add optical_material to your stack."
        )


def compute_eqe(
    stack: DeviceStack,
    wavelengths_nm: np.ndarray | None = None,
    Phi_incident: float = 4e21,
    N_grid: int = 60,
    t_settle: float = 1e-1,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    progress: ProgressCallback | None = None,
) -> EQEResult:
    """Compute EQE(λ) and the AM1.5G-integrated J_sc.

    Parameters
    ----------
    stack : DeviceStack
        Must include at least one layer with wavelength-resolved optical
        data (``optical_material``). See ``_require_tmm_optical_data``.
    wavelengths_nm : np.ndarray, optional
        Probe wavelengths [nm]. Default is 300-1000 nm in 25-nm steps
        (29 points), which comfortably spans AM1.5G's peak region and
        the MAPbI3 absorption edge around 780 nm.
    Phi_incident : float, default 4e21
        Monochromatic probe photon flux [m⁻² s⁻¹]. The default matches
        the integrated AM1.5G photon flux above the band edge so the
        photo-signal swamps any residual ionic / contact transient that
        the t_settle settle did not fully damp. EQE is independent of
        this to first order; J_sc per wavelength scales with it.
        On ionic-rich presets (mobile-ion concentrations comparable
        to or above ~1e25 m⁻³) values much below ~1e21 can produce
        unphysical EQE > 1 because the ionic background swamps the small
        monochromatic photo-signal.
    N_grid : int, default 60
        Total drift-diffusion grid nodes across electrical layers.
    t_settle : float, default 1e-1
        Illuminated-SS settling time [s] per wavelength. 100 ms covers
        the slow ionic dynamics on typical perovskite presets
        (D_ion ≈ 1e-17 m²/s, 400 nm absorber → τ_ion ≈ 16 ms; settling
        for ~5τ_ion damps the ionic transient that would otherwise leak
        into the V=0 terminal current and inflate EQE.
    rtol, atol : float
        scipy solver tolerances forwarded to ``run_transient``.
    progress : ProgressCallback | None

    Returns
    -------
    EQEResult
    """
    _require_tmm_optical_data(stack)

    if wavelengths_nm is None:
        # 25-nm step from 300 to 1000 nm → 29 points. Wider than the
        # MAPbI3 absorption edge so the "EQE → 0" tail is visible for
        # sanity checks.
        wavelengths_nm = np.linspace(300.0, 1000.0, 29)
    wavelengths_nm = np.asarray(sorted(wavelengths_nm), dtype=float)
    if wavelengths_nm.size == 0:
        raise ValueError("wavelengths_nm must be non-empty")
    if np.any(wavelengths_nm <= 0.0):
        raise ValueError(
            f"wavelengths_nm must be positive, got {wavelengths_nm}"
        )
    if Phi_incident <= 0.0:
        raise ValueError(
            f"Phi_incident must be positive, got {Phi_incident}"
        )

    wavelengths_m = wavelengths_nm * 1e-9

    elec = electrical_layers(stack)
    n_per = max(N_grid // len(elec), 2)
    layers_grid = [Layer(l.thickness, n_per) for l in elec]
    x = multilayer_grid(layers_grid)

    # Baseline MaterialArrays — we'll swap G_optical per wavelength. The
    # baseline's own G_optical (if any) is ignored; only the other
    # cached fields (mobilities, doping, Poisson factor, …) matter.
    mat_base: MaterialArrays = build_material_arrays(x, stack)

    # Dark baseline at V=0: settle the device with G=0 to capture any
    # ionic-drift / contact-leakage current that flows even without
    # illumination. Subtracting this from J(λ) at V=0 isolates the pure
    # photo-current. Without this correction, ionic-rich presets like
    # ionmonger_benchmark_tmm produce EQE > 1 at low Phi_incident because
    # the ionic background swamps the small monochromatic photo-signal —
    # the EQE definition assumes the dark current at V=0 is exactly zero,
    # which only holds for a perfectly equilibrated device.
    mat_dark = dataclasses.replace(mat_base, G_optical=np.zeros_like(x))
    y_dark = _solve_illuminated_ss_with_mat(
        x, stack, mat_dark, V_app=0.0,
        t_settle=t_settle, rtol=rtol, atol=atol,
    )
    J_dark = float(_compute_current(x, y_dark, stack, 0.0, mat=mat_dark))

    eqe = np.zeros_like(wavelengths_nm)
    J_sc_lambda = np.zeros_like(wavelengths_nm)

    for k, lam_m in enumerate(wavelengths_m):
        G = _single_wavelength_generation(x, stack, lam_m, Phi_incident)
        mat_k = dataclasses.replace(mat_base, G_optical=G)
        y_ss = _solve_illuminated_ss_with_mat(
            x, stack, mat_k, V_app=0.0,
            t_settle=t_settle, rtol=rtol, atol=atol,
        )
        J_total = float(_compute_current(x, y_ss, stack, 0.0, mat=mat_k))
        # Photo-only current = total at V=0 with λ minus dark at V=0.
        J_k = J_total - J_dark
        J_sc_lambda[k] = J_k
        # |J_sc| because EQE is positive by convention regardless of the
        # solar-vs-injection sign carried by J_sc.
        eqe[k] = abs(J_k) / (Q * Phi_incident)

        if progress is not None:
            progress(
                "eqe", k + 1, len(wavelengths_nm),
                f"λ={wavelengths_nm[k]:.0f} nm, EQE={eqe[k]:.3f}",
            )

    # Integrate EQE · Φ_AM15G dλ. load_am15g enforces in-range bounds,
    # so clip the integration range to the AM1.5G file's coverage.
    # The shipped file starts at 280 nm; 300 nm (our default start) is
    # safe, but user-supplied wavelengths could fall outside.
    try:
        _, phi_am15g = load_am15g(wavelengths_nm)
    except ValueError:
        # Graceful fallback: clip to the AM1.5G range, integrate only
        # over the usable sub-band. Keeps the main EQE curve reported
        # untouched — just skips the integrated J_sc on out-of-range
        # probes.
        J_sc_integrated = float("nan")
    else:
        J_sc_integrated = float(
            Q * np.trapezoid(eqe * phi_am15g, wavelengths_m)
        )

    return EQEResult(
        wavelengths_nm=wavelengths_nm,
        EQE=eqe,
        J_sc_per_lambda=J_sc_lambda,
        J_sc_integrated=J_sc_integrated,
        Phi_incident=float(Phi_incident),
    )
