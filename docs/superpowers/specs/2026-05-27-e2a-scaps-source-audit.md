# Phase E2a Sprint 1 Day 1 — SCAPS source audit

**Status:** investigation in progress (no code change)
**Branch:** `e2a-scaps-source-audit`
**Goal:** identify exact SCAPS physics formulations cited in the Manual so
the Phase E2 face-density refactor can target the right closure path.

## Bibliography (from SCAPS Manual Feb 2016, p.111)

| Ref | Citation | Role |
|---|---|---|
| [1] | M. Burgelman, P. Nollet, S. Degrave, *Thin Solid Films* 361 (2000) 527-532 | original SCAPS polycrystalline model |
| [2] | K. Decock, S. Khelifi, M. Burgelman, *Thin Solid Films* 519 (2011) 7481-7484 | "Modelling multivalent defects in thin film solar cells" — cited in §3.6.5 as the "more advanced algorithms" replacing the display-only τ=1/(σ·v<sub>th</sub>·N<sub>t</sub>) |
| [4] | J. Verschraegen, M. Burgelman, *Thin Solid Films* 515 (2007) 6276-6279 | intra-band tunneling for heterojunction solar cells in SCAPS |
| [12] | A. Niemegeers, S. Gillis, M. Burgelman, 2nd WCPVE Wien 1998, p. 672-675 | "A user program for realistic simulation of polycrystalline heterojunction solar cells: SCAPS-1D" — main architecture paper |
| [13] | **H.J. Pauwels, G. Vanhoutte**, *J. Phys. D-Appl. Phys.* 11 (1978) 649-667 | **Interface SRH formula source** — cited in §3.8 as the model implemented for interface recombination |
| [14] | **J. Verschraegen**, dissertation, Universiteit Gent 2006 | **Most likely source for SCAPS internals** — graduate thesis on CISCuT modeling |

## What the SCAPS Manual confirms

### §3.6.5 — Bulk SRH (display-only formula)

```
τ = 1 / (σ · v_th · N_t)
L = sqrt(D · τ)
```

> "NONE OF THESE VALUES ARE USED IN THE ACTUAL SIMULATIONS! More advanced
> algorithms are used there, see e.g. [2]."

So SCAPS computes bulk SRH via the algorithm in **Decock 2011** [2] — not
the textbook lifetime form. This matters: SolarLab uses τ=1/(σ·v<sub>th</sub>·N<sub>t</sub>)
directly in `_apply_interface_recombination`. If SCAPS uses a more
elaborate occupation-statistics algorithm at finite trap density, the
factor in calibration_factor=1e-4 is partly absorbing this mismatch.

### §3.6.6 — Radiative + Auger

```
U_rad  = K · (n·p − ni²)
U_Auger = (c_n·n + c_p·p) · (n·p − ni²)
```

These match SolarLab implementation in `physics/recombination.py`. No
hidden gotcha here.

### §3.8 — Interfaces (THERMIONIC EMISSION + Pauwels-Vanhoutte SRH)

Two quoted statements:

> "The model which is implemented for interface transport in SCAPS is
> thermionic emission. The thermal velocity of the interface transport
> equals the smallest thermal velocity of the two neighboring layers. The
> use of this model implies that there will always be a (small) step in
> the quasi-Fermi level energy values at an interface, even when there
> are no band offsets."

> "Recombination in interface states is modeled by the Pauwels-Vanhoutte
> theory [13], which is an extension of the Shockley-Read-Hall theory."

**Two key inferences:**

1. SCAPS' interface-plane carrier densities are **NOT** the
   bulk-interior densities (idx-1/idx+1 in SolarLab's E1.5 scheme).
   They come out of the thermionic-emission boundary condition,
   which couples to the quasi-Fermi-level step at the heterointerface.
2. The Pauwels-Vanhoutte 1978 formula is the actual interface-SRH
   expression — likely involves the interface-plane densities computed
   in (1), not the bulk densities.

### §3.9 — Tunnelling (WKB)

Band-to-band, intraband, interface-defect, and contact tunnelling all
selectable in SCAPS via WKB approximation. SolarLab does not implement
tunnelling at all. Likely contributes to the 99 mV base V<sub>oc</sub> gap (Phase G
diagnosis "structural") but probably NOT the dominant cause — tunnelling
typically adds reverse-bias leakage, not forward-bias V<sub>oc</sub> shift.

## Remaining unknowns (require ref [13] paper or ref [14] dissertation)

1. **Exact form of Pauwels-Vanhoutte interface SRH.** Manual cites the
   paper but does not quote the formula. SolarLab's current form is
   `R = N_t · σ · v_th · (n·p − ni²) / (n + n1 + p + p1)`. The original
   1978 paper may include band-bending corrections or detailed-balance
   factors not in SolarLab's form.

2. **Thermionic-emission boundary at interface plane.** SCAPS computes
   a Q-Fermi step Δφ_n, Δφ_p at the heterointerface from the TE current.
   The interface-plane n, p that feed into the SRH integral are then:
   ```
   n_iface = N_C · exp(-(E_C − E_Fn_iface)/kT)
   p_iface = N_V · exp(-(E_Fp_iface − E_V)/kT)
   ```
   where E_Fn_iface, E_Fp_iface include the TE step. SolarLab's E1.5
   cross-carrier scheme samples bulk-interior n, p without any TE step,
   so the interface-plane n is missing the band-bending depletion factor.
   This is the **5-order density gap** documented in Phase A2 RFC.

3. **SCAPS' v<sub>th</sub> convention.** Manual says "smallest thermal velocity of
   the two neighboring layers." SolarLab's scaps_mirror.yaml sets a
   uniform `v_th_cm_s: 1e7` per InterfaceDefect, identical on both sides.
   This means the SCAPS convention is moot for our setup, but if partner
   ever specifies asymmetric v<sub>th</sub>, SolarLab's loader would need a min()
   reducer.

4. **Whether SCAPS uses Boltzmann-degenerate carrier statistics.**
   Phase G diagnosis suggested this could account for the residual
   74 mV. The Manual does not specify — most SCAPS internals use
   Boltzmann nondegenerate statistics, but the ETL is doped to 1e18 cm⁻³
   which is borderline. Ref [12] (Niemegeers 1998) likely has this in
   the algorithm description.

## Decision: do NOT block on paper acquisition

Pauwels-Vanhoutte 1978 (J. Phys. D 11, 649-667) is a 47-year-old paper.
Acquiring it via HKUST library / Zotero / partner outreach takes 1-2 days
in the worst case. **Sprint 1 Day 3 (face-density probe extension) can
proceed in parallel** using the inferences above:

- Hypothesis: SCAPS' interface-plane n is **depleted** relative to bulk
  by the band-bending factor `exp(-Δφ_n / V_T)` at the heterointerface.
- Probe: extend `scripts/probe_interface_face_densities.py` with a new
  formula that combines (a) the cached `chi`/`Eg`-derived band offsets
  on `MaterialArrays` and (b) the solved `phi(x)` from the Poisson
  factor to compute interface-plane n, p. Compare against the E1.5
  cross-carrier result to see if the 5-order density gap is recovered.

If the band-bending-depletion probe reproduces SCAPS-like sensitivity
to N_D_ETL (closes ETL doping over-sensitivity), it is the right
formulation regardless of whether the Pauwels-Vanhoutte original paper
uses exactly that form.

If it does NOT reproduce SCAPS-like sensitivity, then we need either
(a) the original paper to extract the exact form, or (b) a different
hypothesis (Boltzmann-degenerate statistics, Φ<sub>b</sub> BC).

## Sprint 1 next actions

- **Day 2:** issue Zotero / inter-library search for Pauwels-Vanhoutte
  1978 + Verschraegen 2006 dissertation. Park as background.
- **Day 3:** extend probe script with band-bending-depletion candidate
  formula. Run on scaps_mirror.yaml across N_D_ETL = 1e16, 1e17, 1e18
  cm⁻³. Compare to SCAPS PDF sweep.
- **Day 4-5:** synthesize into Phase E2 design RFC. If probe succeeds,
  RFC scopes the implementation; if it fails, RFC documents the
  blocker and proposes paper-acquisition timeline.

## Conventions reminders

- Branch stays `e2a-scaps-source-audit`. No code change in Sprint 1
  Days 1-2. Commit this audit doc as the Day 1 deliverable.
- Day 3 probe extension is investigation-only — extends a script under
  `scripts/`, does NOT touch `perovskite_sim/`. Same pattern as Phase A1.
- Day 4-5 RFC lives at `docs/superpowers/specs/2026-05-XX-e2-design-rfc.md`.
