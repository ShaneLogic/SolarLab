# Beer–Lambert Optical Generation & SCAPS Reference Protocol (`V1`)

This protocol specifies the physical scope, photon-accounting contract, opt-in
compilation boundary, historical numerical stress-case disposition, and
reference-study provenance for `P08-04` (`solarlab`).

---

## 1. Optical Generation Models & Scope Boundaries (`P08-04a` / `P08-04b`)

Electrical physics selection never implicitly alters the optical generation
model. `solarlab` exposes three distinct optical generation models plus
node-aligned fixed generation resources:

1. **`beer_lambert` (Monochromatic Scalar Control-Volume Beer–Lambert)**:
   - Model ID: `beer_lambert_monochromatic_v1` (`BeerLambertBinding`).
   - Uses per-layer scalar `alpha` [$\text{m}^{-1}$] and device-level scalar
     photon flux `Phi` [$\text{m}^{-2}\,\text{s}^{-1}$] on electrical control
     volumes.
   - Applicable to prescribed monochromatic illumination and analytical
     verification benchmarks.

2. **`beer_lambert_spectral` (Single-Pass Spectral Control-Volume Beer–Lambert)**:
   - Model ID: `beer_lambert_spectral_front_incidence_v1`
     (`SpectralBeerLambertBinding`).
   - Uses the same tabulated `k(lambda)` (or constant-index `alpha`), metre
     wavelength quadrature, front-prefix optical-only substrate offset, and
     `SpectralPhotonFlux` (e.g. ASTM G-173-03 AM1.5G) as spectral TMM, with
     per-layer attenuation
     $$\alpha_\ell(\lambda) = \frac{4\pi\,k_\ell(\lambda)}{\lambda}$$
     and cancellation-free control-volume cell absorptance
     $$A_{\text{cell}}(\lambda) = e^{-\tau_j(\lambda)}\bigl(-\operatorname{expm1}(-\Delta\tau_j(\lambda))\bigr).$$
   - Zero front/internal Fresnel reflection ($R(\lambda) = 0$), zero coherent
     standing-wave interference, zero dropped multipass fraction, and no
     TMM-derived photon-recycling reabsorption.
   - Per-wavelength and trapezoid-integrated photon accounting close to machine
     precision on arbitrary uniform, stretched (`tanh`), or two-sided trace
     meshes:
     $$\text{transmitted} + \sum_{\ell}\text{absorbed}_\ell = \text{incident}.$$
   - Optical-only front prefix layers (`role: substrate`, `electrical=False`,
     including `incoherent=True` front glass) attenuate the spectrum entering
     the electrical stack (`parasitic_absorbed_m2_s`) without creating
     electrical carriers.
   - **Applicability vs TMM**: `beer_lambert_spectral` matches idealized
     single-pass reference tools (such as SCAPS-1D configured with zero front
     reflection) and thick/textured stacks without thin-film Fabry–Pérot cavity
     effects. On `scaps_mirror_v2.yaml` (`glass` 1 mm + `spiro_OMeTAD` 20 nm +
     `MAPbI3` 800 nm + `TiO2` 25 nm, AM1.5G 300–1000 nm), single-pass spectral
     Beer–Lambert absorbs $q\int G\,\mathrm{d}x \approx 260.14\ \text{A}\,\text{m}^{-2}$
     versus $\approx 235.06\ \text{A}\,\text{m}^{-2}$ for coherent-glass TMM
     (legacy SCAPS loader effective `incoherent=False`) and
     $\approx 232.52\ \text{A}\,\text{m}^{-2}$ for front-incoherent-glass TMM
     (declared `incoherent=True`), where air/glass and internal Fresnel
     reflections reject ~5-10% of incident photons. Conversely, on stacks with a reflective rear contact or thin
     absorber where back-reflection doubles path length, TMM can exceed
     single-pass Beer–Lambert. **Larger $J_{\text{sc}}$ under
     `beer_lambert_spectral` reflects zero-front-reflection idealization, NOT
     physical superiority over TMM.**

3. **`tmm` (Coherent / Front-Incoherent Transfer-Matrix Optics)**:
   - Model ID: `transfer_matrix_front_incidence_v1` (`OpticalStackBinding`).
   - Solves Maxwell multilayer Fresnel reflection, transmission, and coherent
     standing-wave absorption across planar thin-film stacks.

4. **Explicit Opt-In Governance & `not_available` External File Loaders**:
   - In `P08-04`, `optical_model="beer_lambert_spectral"` is strictly **opt-in**
     on `compile_device_1d`, `compile_ionic_device_1d`, and
     `compile_multiple_ionic_device_1d` (`optical_model=None` preserves 100%
     historical dispatch and bit-identical `H_input`, `H_physics`, and compiled
     SHA-256 identities across all certified presets until `P08-05`
     re-certification).
   - Selecting `"beer_lambert"`, `"beer_lambert_spectral"`, or `"tmm"` produces
     three distinct cryptographic `CompiledDevice1D.identity` digests (and
     distinct `H_input`/`H_physics` digests when bound through `ModelRegistry`).
   - Arbitrary filesystem `profile_file` generation loaders remain explicitly
     `not_available`: prescribed spatial generation is admitted solely through
     immutable content-hashed `ResourceTable(..., kind="fixed_generation")`
     bound to exact compiled node coordinates (`FixedGenerationProfile`).

---

## 2. Historical Stress-Case Witness & New-Core Positivity (`scaps_mirror_v2`)

In historical `archive/wip-beer-lambert` (`658baa84`), switching
`scaps_mirror_v2.yaml` (`N=60` electrical intervals, 200 wavelengths over
300–1000 nm) to `beer_lambert_spectral` steepened front-of-absorber ultraviolet/blue
generation ($q\int G\,\mathrm{d}x \approx 260.14\ \text{A}\,\text{m}^{-2}$).
Under the legacy density-variable (`[n, p]` linear concentration) Method-of-Lines
Radau transient (`test_interface_flux_balance.py`), hole concentration at the
absorber front (`node 20` of `N=60`, immediately right of the `HTL/PVK`
interface) crossed below zero ($p < 0$), triggering Newton/step-size collapse
and $>600\ \text{s}$ timeouts without ad-hoc post-step density clipping.

On `solarlab`'s refactored 1D kernel:
- Carrier unknowns use log-activity / quasi-Fermi coordinates (`zn`, `zp`), for
  which $n = n_{\text{ref}}\exp((\phi - \phi_{\text{ref}})/V_T + z_n) > 0$ and
  $p = p_{\text{ref}}\exp(-(\phi - \phi_{\text{ref}})/V_T - z_p) > 0$ hold
  by construction at every Newton iterate without clipping or projection.
- Time integration (`BackwardEuler1DStep` / `solve_transient`) discretizes exact
  physical control-volume carrier inventories via cancellation-free secants
  $S_{\text{old}}\operatorname{expm1}(\log(c_{\text{new}}/c_{\text{old}}))/\Delta t$,
  closing cell-wise and global particle and Maxwell total-current balances on
  the `scaps_mirror_v2` `N=60` spectral Beer–Lambert stack across DC steady
  state, forward-bias $J\text{--}V$ operating points, and transient illumination
  and voltage steps (`tests/integration/test_spectral_beer_lambert_stress.py`).

---

## 3. SCAPS Reference Catalog & Provenance (`ScapsBeerLambertReferenceV1.json`)

The read-only reference catalog is packaged at
`solarlab.reproducibility.data/ScapsBeerLambertReferenceV1.json` and accessed
through `solarlab.reproducibility.scaps_beer_lambert`:

- **Workbook Provenance**: Extracted read-only from
  `reproducibility/ScapsParams.xlsx`
  (`source_sha256 = 862cf0216b08558253268f4ec99fad3e93842c7ce1e9006874ecf03096e9ae3c`).
  All **93** workbook parameter points across **11** sweep families (`CHI_ETL`,
  `Nd_ETL`, `Nd_PVK`, `Nt_C_PVK`, `Nt_V_PVK`, `Nt_HTL PVK`, `Nt_PVK ETL`,
  `Et_C_PVK`, `Et_V_PVK`, `Et_HTL PVK`, `Et_PVK ETL`) are retained, including
  unmeasured `CHI_ETL` rows preserved as `null`/`None`.
- **High-Doping `Nd_PVK` Extension**: `load_scaps_beer_lambert_catalog()`
  appends two SolarLab-only donor-doping points (`1e19` and `1e20 cm^-3`,
  totaling **95** catalog points) marked `origin="solarlab_extension"`,
  `row=None`, `reference={"V_oc": None, "J_sc": None, "FF": None, "PCE": None}`,
  `scope="high_doping_extrapolation"`, and `min_N_grid=90` (to resolve the
  thin Debye length at $10^{20}\ \text{cm}^{-3}$). Because these densities reach
  or exceed `SCAPS_PVK`'s configured $N_C = 10^{19}\ \text{cm}^{-3}$,
  Maxwell–Boltzmann results at those two points are model extrapolations, not
  validated degenerate-Fermi–Dirac predictions.
- **Evidence & Unit Caveats**:
  - Historical 2026-07-02 TMM comparisons and `scaps_mirror_v2.yaml`'s legacy
    SCAPS-emulation settings (`het_recomb_despike=0.53`,
    `flat_band_metal_contacts=True`, `contact_phi_B_eV=0.42`, `V_bi=1.30`)
    remain historical disclosures; historical TMM runs must never be relabelled
    as Beer–Lambert results.
  - In `historical_effective_lifetime` mode, `Nt_C_PVK` and `Nt_V_PVK` scale
    the combined effective bulk SRH lifetime (`1/tau_total`) rather than
    independent microscopic defect populations (`bulk_defect_mode="species_only"`
    must be passed explicitly to edit a single named species).
  - Interface defect density sweeps (`Nt_HTL PVK`, `Nt_PVK ETL`) retain the
    historical areal `cm^-2` interpretation despite the workbook's `cm^-3`
    column header; that upstream workbook header ambiguity remains explicitly
    documented rather than silently converted by an invented thickness.
