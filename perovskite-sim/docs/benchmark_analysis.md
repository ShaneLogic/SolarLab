# Quantitative Benchmark Analysis

## Reference Papers

1. **Courtier et al. (2019)** — "How transport layer properties affect perovskite solar cell performance", *Energy Environ. Sci.*, DOI: 10.1039/C8EE01576G (IonMonger)
2. **Calado et al. (2016)** — "Evidence for ion migration in hybrid perovskite solar cells with minimum hysteresis", *Nature Commun.* 7, 13831 (Driftfusion)

## Test Results Summary (26/28 checks passed, 93%)

### PASSED (26 checks)

| Category | Check | Detail |
|----------|-------|--------|
| **Generation** | Beer-Lambert alpha*L > 4 | alpha*L = 5.20 |
| **Generation** | Numerical G matches theory | error = 0.19% |
| **Equilibrium** | Mass action n*p = ni^2 | range [1.0000, 1.0000] |
| **Equilibrium** | Charge neutrality | rho_max = 0 |
| **Poisson** | BCs match V_bi | phi(L) = 1.1000 V |
| **Contacts** | HTL: p = N_A | exact |
| **Contacts** | ETL: n = N_D | exact |
| **Einstein** | D = mu * V_T | exact |
| **J-V** | J_sc in [20,24] mA/cm2 | 22.51 mA/cm2 |
| **J-V** | J_sc within 5% of theory | error = 0.92% |
| **J-V** | FF in [0.55, 0.88] | 0.797 |
| **J-V** | PCE in [12, 22]% | 13.4% |
| **J-V** | V_oc < V_bi | 0.749 < 1.1 |
| **Diode** | Ideality factor n_id in [1,3] | n_id = 1.56 |
| **Diode** | V_oc increases with light | 0.659 → 0.816 V |
| **Sensitivity** | V_oc increases with tau | 0.586 → 0.905 V |
| **Sensitivity** | J_sc linear with Phi | CV = 0.0% |
| **Hysteresis** | HI > 0 at some scan rates | max HI = 0.019 |
| **Ion** | Ion conservation | error = 0.03% |
| **Ion** | Ions redistribute under bias | max/P0 = 1.66 |
| **Ion** | P < P_lim everywhere | satisfied |
| **Ion** | Zero-flux BC at contacts | dP/dt = 0 |
| **Impedance** | Re(Z) > 0 | all points |
| **Impedance** | R_hf > 0 | 0.089 Ohm.cm2 |
| **Cross-config** | Both configs V_oc > 0.7V | nip: 0.912, pin: 0.932 |
| **Cross-config** | Both configs J_sc > 20 | nip: 40.5, pin: 40.5 |

### FAILED (2 checks)

| Check | Expected | Got | Root Cause |
|-------|----------|-----|------------|
| V_oc in [0.90, 1.15] V | ~1.07 V | 0.749 V | Missing band offsets |
| V_oc increases as ni decreases | monotonic | flat at 0.743 V | Same root cause |

## Root Cause Analysis: V_oc Deficit

The V_oc is systematically ~0.25 V below IonMonger because **the model lacks conduction/valence band offsets at transport layer interfaces**.

### Evidence

1. **Quasi-Fermi level splitting at V_oc** (0.75V applied) is 1.198 V — the absorber *can* generate sufficient quasi-Fermi splitting, but excess carrier injection from highly-doped contacts creates high np product and recombination.

2. **Reducing TL doping** (proxy for band-offset selectivity) raises V_oc:

   | N_dop (m^-3) | V_oc (V) | J_sc (mA/cm2) | FF | PCE (%) |
   |:---:|:---:|:---:|:---:|:---:|
   | 1e24 | 0.743 | 22.9 | 0.796 | 13.5 |
   | 1e23 | 0.856 | 22.9 | 0.633 | 12.4 |
   | 1e22 | 0.967 | 22.9 | 0.545 | 12.1 |
   | 1e21 | 1.083 | 22.8 | 0.413 | 10.2 |

3. **V_oc vs ni is flat** because at high injection (n, p >> ni), SRH recombination rate R = np/(tau_p*n + tau_n*p) is independent of ni. The contact injection, not ni, determines the np product.

### Physics Explanation

In IonMonger and Driftfusion, the ETL has a conduction band offset of ~0.3 eV below the perovskite, and the HTL has a valence band offset of ~0.3 eV above. These offsets create energy barriers that:

- Prevent holes from entering the ETL (hole-blocking)
- Prevent electrons from entering the HTL (electron-blocking)
- Confine minority carriers to the absorber

Without these barriers, our ohmic contacts (n = N_D = 1e24 at ETL, p = N_A = 1e24 at HTL) inject majority carriers into the absorber. At forward bias, the reduced built-in field allows these carriers to flood the absorber, raising the np product far above what the intrinsic absorber would produce, and driving excessive recombination.

## What IS Correct (Validated Quantitatively)

| Physics Module | Validation Method | Accuracy |
|---|---|---|
| Beer-Lambert generation | Analytical comparison | 0.19% error |
| Poisson solver | BCs and charge neutrality | Exact |
| Scharfetter-Gummel fluxes | J_sc vs theory | 0.92% error |
| SRH recombination | V_oc vs tau scaling | 80 mV/decade (n_id=1.56) |
| Ideality factor | V_oc vs illumination | n_id = 1.56 (SRH-dominated) |
| J_sc linearity | J_sc vs Phi | CV < 0.1% |
| Ion conservation | Integral before/after bias | 0.03% error |
| Ion zero-flux BCs | dP/dt at boundaries | Exact zero |
| Steric limit | P < P_lim check | Satisfied |
| Ion redistribution | Profile change under bias | max/P0 = 1.66 |
| Impedance passivity | Re(Z) > 0 | All frequencies |

## Parameter Sensitivity (All Physically Correct)

### tau sensitivity (V_oc increases ~80 mV/decade)
| tau (s) | V_oc (V) | J_sc (mA/cm2) | FF | PCE (%) |
|:---:|:---:|:---:|:---:|:---:|
| 1e-9 | 0.586 | 22.4 | 0.595 | 7.8 |
| 1e-8 | 0.664 | 22.8 | 0.719 | 10.9 |
| 1e-7 | 0.743 | 22.9 | 0.796 | 13.5 |
| 1e-6 | 0.843 | 22.9 | 0.802 | 15.5 |
| 1e-5 | 0.905 | 22.9 | 0.837 | 17.3 |

### Illumination sensitivity (ideality factor n_id = 1.56)
| Suns | V_oc (V) | J_sc (mA/cm2) |
|:---:|:---:|:---:|
| 0.1 | 0.659 | 2.3 |
| 0.5 | 0.716 | 11.4 |
| 1.0 | 0.743 | 22.9 |
| 2.0 | 0.776 | 45.8 |
| 5.0 | 0.816 | 114.4 |

## Recommendation

To achieve quantitative agreement with IonMonger (V_oc within 5%), the simulator needs:

1. **Band offsets at interfaces** — Add conduction band (chi) and valence band (chi + Eg) parameters per layer, and modify the Scharfetter-Gummel flux at interface faces to include the energy barrier via thermionic emission or modified boundary conditions.

2. **Interface recombination** — Add surface recombination velocity parameters (v_n, v_p) at each interface, applied as a localized recombination term at the interface node.

These are structural enhancements, not bug fixes. The existing physics engine is correct within the scope of its model assumptions.
