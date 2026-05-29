# Phase E2 Sprint 4 Day 1 — Pauwels-Vanhoutte 1978 formula extraction

**Status:** paper acquired + formula extracted; no code change yet
**Branch:** `e2-pauwels-vanhoutte` (cut off main `dfb7144`)
**Source:** H J Pauwels and G Vanhoutte 1978 *J. Phys. D: Appl. Phys.* 11 649
(`docs/manual/pauwels_vanhoutte_1978.pdf`, 20 pages)

## Paper context

Model assumes:
- p-type semiconductor 1 (PVK) much more heavily doped than n-type
  semiconductor 2 (ETL) — opposite of SCAPS-mirror but symmetric model
  inverts trivially.
- Thermionic emission theory at interface (not diffusion theory).
- Recombination/generation in space-charge region neglected.
- Interface recombination from BOTH conduction bands TO the valence
  bands.
- One type of carrier (e.g. holes) majority at interface — minimises
  interface recombination per Shockley-Read.
- Energies/voltages normalised to kT/q (≈ 25 mV at 300 K).

## Critical equations

### Interface-plane carrier densities (eqs 8, 9, 11)

```
n_2s = n_2 · exp(−V_2)         (8)  electron at interface, ETL side
p_1s = p_1 · exp(−V_1)         (9)  hole     at interface, PVK side
p_2s = p_2 · exp(V_2 + V)      (11) hole     at interface, ETL side
```

Where:
- n_1, p_1, n_2, p_2 = thermal-equilibrium MAJORITY/minority bulk
  densities of semiconductor 1, 2
- V_1, V_2 = band-bending in space-charge regions of semiconductor 1, 2
- V = applied forward voltage
- All quantities normalised to kT/q

### Interface SRH rates (eqs 12, 13) — Shockley-Read on interface plane

```
j_s1 = s_1 · (n_1s − n_1s0)
     = s_1 · [n_1s − n_1 · exp(V_1)]              (12)

j_s2 = s_2 · (n_2s − n_2s0)
     = s_2 · [n_2s − n_2 · exp(−V_2 − V)]         (13)
```

Where:
- n_1s, n_2s = "actual" electron densities at interface (under bias)
- n_1s0, n_2s0 = electron densities at interface calculated with the
  **single hole quasi-Fermi level** (Shockley-Read reference)
- s_1, s_2 = interface recombination velocities from conduction bands
  1 and 2

The (n − n_eq) form is the LINEARISED Shockley-Read theory under the
heavy-doping assumption (eq 10a).

### Heavy-doping limit (eq 30) — V_1 and V_2 are STRUCTURALLY DETERMINED

```
V_1 = 0                  if ΔE_v = ΔE_g + ΔE_c > 0     (30a)
V_1 = −ΔE_g − ΔE_c        if ΔE_v < 0 (inversion)        (30b)
```

V_2 then derived from V<sub>bi</sub> = V_1 + V_2 − V.

**Key insight:** V_1 depends ONLY on band offsets (ΔE_c, ΔE<sub>g</sub>) — NOT
on bulk doping levels (in the heavy-doping regime). This is the
structural difference vs the failed BBD prototype, which used local
grid potential differences that scaled with N_D_ETL.

### Thermionic emission boundary (eq 14a/b/c)

```
ΔE_c > 0:  j_n − j_s2 = [n_2·exp(−V_2) − n_2s'] · v_t · exp(−ΔE_c)    (14a)
ΔE_c < 0:  j_n − j_s2 = [n_2·exp(−V_2) − n_2s'] · v_t                  (14b)
v_t = v_t2 − s_2                                                       (14c)
```

This is the thermionic-emission boundary that couples the bulk j_n
into the interface state. SCAPS uses this convention with smallest
v<sub>th</sub> of the two layers (per SCAPS Manual §3.8).

### Relation between equilibrium concentrations (eq 19)

```
n_1 · exp(V_1) = β_c · n_2 · exp(−V_2 − V − ΔE_c)        (19)
β_c = N_c1 / N_c2                                        (eq 15)
```

This is detailed balance across the heterojunction.

## Mapping to SolarLab solver

SolarLab uses opposite convention: ETL ("semiconductor 2") is heavy-
doped, PVK ("semiconductor 1") is light-doped. Pauwels-Vanhoutte heavy-
doping assumption (10a) inverts: electrons are majority at the
interface (from the ETL side). So:
- V_2 ≈ 0 (heavy-doping ETL side does not deplete)
- V_1 = V<sub>bi</sub> − V<sub>app</sub> (light-doping PVK side absorbs all band-bending)

In the SCAPS-mirror configuration (ΔE_c = −0.16 eV between PVK and
ETL), |ΔE_c| ≪ kT is FALSE (0.16 eV ≈ 6.4 kT), so per eq (30):
- If ΔE_v > 0: V_1 = 0 (no inversion in PVK)
- ΔE_v = ΔE<sub>g</sub> − |ΔE_c| ... need to compute for our stack

For ETL Eg=4 eV, PVK Eg=1.53 eV: ΔE<sub>g</sub> = 4 − 1.53 = 2.47 eV.
ΔE_c = chi_ETL − chi_PVK = 4 − 3.84 = 0.16 eV (positive: ETL chi
deeper, electron well attracts).
ΔE_v = ΔE<sub>g</sub> + ΔE_c = 2.47 + 0.16 = 2.63 eV ≫ 0 → no inversion.

Per eq (30a): **V_1 = 0** in heavy-doping limit. ⚠️ This contradicts
the intuition "PVK depletes more because it is lightly doped." Need
to re-check assumption.

Re-reading: Pauwels-Vanhoutte assumes semiconductor 1 = p-type heavy-
doped, semiconductor 2 = n-type light-doped. They map p+n. **Our
SCAPS-mirror is the OPPOSITE — light-doped p-type PVK, heavy-doped
n-type ETL.** So we invert by swapping subscripts. Then:
- "Sem. 1" in our case = ETL (heavy n-doped) → V_1 = 0 ✓
- "Sem. 2" in our case = PVK (light p-doped) → V_2 = V<sub>bi</sub> − V<sub>app</sub>

So PVK absorbs all band-bending V<sub>bi</sub> − V<sub>app</sub>, consistent with intuition.

p_iface (PVK side) = p_2 · exp(V_2 + V) = p_2 · exp(V<sub>bi</sub>)  (since V_2 = V<sub>bi</sub> − V).

Wait this is constant (independent of V<sub>app</sub>) under the substitution
V_2 + V = V<sub>bi</sub>. Let me re-derive in our solver's convention:

**In SolarLab convention (PVK left/idx_L = "semiconductor 2",
ETL right/idx_R = "semiconductor 1"):**

```
V_1 (ETL band-bending) = 0
V_2 (PVK band-bending) = V_bi − V_app

n_iface ETL-side = n_2·exp(−V_2) — wait this is in PV convention
```

Actually I'm tangling subscripts. Let me restart cleanly with our
solver conventions.

**Cleaner derivation in SolarLab convention:**

Let φ_iface = electrostatic potential at the interface node, fixed by
boundary conditions + Poisson at SS. Heavy-doping ETL side: φ at
bulk ETL ≈ φ_iface (negligible band-bending). Light-doping PVK side:
φ_iface − φ_bulk_PVK = V<sub>bi</sub> − V<sub>app</sub> (full band-bending).

Boltzmann within each layer (assuming flat quasi-Fermi within each
layer's bulk under SS):

```
p[bulk PVK] = N_v_PVK · exp(−(E_F − E_v_bulk_PVK)/kT)
p[iface PVK side] = N_v_PVK · exp(−(E_F − E_v_iface_PVK)/kT)
```

E_v_iface_PVK = E_v_bulk_PVK − q·(φ_iface − φ_bulk_PVK)
            = E_v_bulk_PVK − q·(V<sub>bi</sub> − V<sub>app</sub>)

So p[iface PVK side] = p[bulk PVK] · exp(−q·(V<sub>bi</sub> − V<sub>app</sub>)/kT)
                    = p[bulk PVK] · exp(−(V<sub>bi</sub> − V<sub>app</sub>)/V<sub>T</sub>)

This is what we want. The depletion factor for the PVK-side hole at
interface is exp(−(V<sub>bi</sub> − V<sub>app</sub>)/V<sub>T</sub>). DOES NOT depend on N_D_ETL
(except through V<sub>bi</sub> which depends on Fermi-level alignment, weak
log-dependence).

Compare BBD: used exp((φ[idx] − φ[idx_L])/V<sub>T</sub>) where φ[idx] − φ[idx_L]
is the LOCAL grid potential drop ≈ V<sub>bi</sub> − V<sub>app</sub> for grids fine enough
to resolve depletion zone. So BBD ALMOST gets it right but:
- Grid-discretisation noise contaminates the φ difference.
- For coarse grids, the local drop ≠ total V<sub>bi</sub>.
- The "near-equivalence" is what made probe show 2.8× reduction at
  high-N<sub>D</sub> but the formula diverges from PV at low N<sub>D</sub> where grid
  cannot resolve the wider depletion.

**Pauwels-Vanhoutte implementation strategy:**

Use V<sub>bi</sub> − V<sub>app</sub> (a GLOBAL quantity from cached mat.V<sub>bi,eff</sub> and the
RHS call's V<sub>app</sub> kwarg) instead of local grid potential differences:

```python
V_1_norm = (mat.V_bi_eff - V_app) / V_T_local

# Interface densities in heavy-doping ETL limit:
n_iface = float(n[eval_n_idx])   # ETL bulk, no depletion (V_2 = 0)
p_iface = float(p[eval_p_idx]) * math.exp(-V_1_norm)   # PVK depleted
```

Then standard SRH form.

## Hypothesis

This formula should:
1. **Preserve CBO sensitivity:** V<sub>bi</sub> depends on band offsets (ΔE_c
   shifts Fermi-level alignment → V<sub>bi</sub> changes). So CBO sweep is
   reflected through V<sub>bi</sub> → V_1 → p_iface → R.
2. **Reduce ETL doping sensitivity:** N_D_ETL changes V<sub>bi</sub> only via
   log term (Fermi-level shift in ETL). Much weaker than BBD's local
   grid potential which scaled linearly with depletion-zone widening
   into ETL at low N<sub>D</sub>.
3. **Preserve PVK doping direction:** N_A_PVK changes V<sub>bi</sub>
   symmetrically + changes p[bulk PVK] directly. Direction depends on
   how the two effects combine.

## Sprint 4 schedule

| Day | Deliverable |
|---|---|
| 1 (now) | Formula extraction doc (this file) |
| 2 | RED test file `test_e2_pauwels_vanhoutte_prototype.py` |
| 2-3 | GREEN env-var-gated prototype `SOLARLAB_PAUWELS_VANHOUTTE=1` |
| 4 | Local regression + ad-hoc V<sub>oc</sub> probe + N_D_ETL spot check |
| 5 | Full SCAPS validation gate |
| 6-7 | If PASS: promote to InterfaceDefect data-model field, scaps_mirror.yaml update, partner report update |

## Convention

- Env var: `SOLARLAB_PAUWELS_VANHOUTTE=1` activates (parallel to BBD
  and thin-shell prototypes).
- Default unset → legacy E1.5 cross-carrier path bit-identical.
- Test surface mirrors BBD: 5 tests pinning env contract + finite JV.

**Related:** [[project-scaps-validation-parked]], BBD gate
`2026-05-27-e2-sprint2-day2-3-validation-gate.md`, thin-shell gate
`2026-05-27-e2-sprint3-day6-7-validation-gate.md`.
