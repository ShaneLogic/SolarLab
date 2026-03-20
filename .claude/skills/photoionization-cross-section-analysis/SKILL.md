---
name: photoionization-cross-section-analysis
description: Calculate and measure photoionization cross-sections for defect centers, including deep centers, excited states, and experimental characterization using constant photoconductivity. Use this for analyzing photon-defect interactions, defect spectroscopy, or comparing optical properties of different impurities.
---

# Photoionization Cross-Section Analysis

## When to Use
- Calculating photon absorption cross-sections for defects
- Characterizing deep center or impurity energy levels
- Comparing optical properties of different impurities experimentally
- Analyzing transition probabilities between defect states

## Deep Centers (Continuum Transitions)

### Formula (Lucovsky Model)
```
σ = (1/(hν)³) × (F_eff/F)² × s₀ × √(hν - |Ei|)
```
Where:
- hν: photon energy
- Ei: ionization energy of the deep center
- F_eff/F: ratio of effective local field to radiation field (accounts for local screening)
- s₀: reference cross-section (~10⁻¹⁶ cm² near band edge)

### Assumptions
- Simple parabolic band near edge
- Square well potential for deep center
- Local field ratio ≈ 1 for shallow levels

### Limitations
- Expression becomes more complicated far from band edge
- Cross-section reduces rapidly with increasing photon energy

## Excited States (Bound-Bound Transitions)

### Hydrogenic Model
**Applicability:** Quasi-hydrogen defects behaving like hydrogen atoms

**Formula:**
```
σ ∝ fba / (hν)³
```
Where fba is the oscillator strength for the specific transition.

### Oscillator Strengths (Reference Values)
| Transition | fba |
|------------|-----|
| 1s → 2p | 0.4162 |
| 1s → 3p | 0.0791 |

**Procedure:**
1. Identify initial and final states
2. Obtain oscillator strength fba from reference tables
3. Calculate cross-section using photon energy

## Experimental Measurement: Constant Photoconductivity

### Method (Grimmeiss and Ledebo)
**Principle:** Adjust light intensity at different photon energies to maintain constant photoconductivity.

### Procedure
1. Measure photoconductivity at each photon energy
2. Adjust intensity I(hν) to achieve constant photoconductivity level
3. Calculate relative cross-section:
   ```
   σ(hν) = C / I(hν)
   ```
   Where C is the constant determined by the fixed photoconductivity level

### Verification Assumptions
- Photo-ionization occurs into the SAME band for all impurities
- Resulting carriers have the SAME mobility
- Levels do not communicate (verify using photo-Hall effect)
- No competing transitions

### Output
Relative photo-ionization cross-section spectrum as function of photon energy

## Cross-Section Comparison
| Transition Type | Energy Dependence | Typical Magnitude |
|----------------|-------------------|-------------------|
| Deep center (near edge) | ∝ √(hν - |Ei|) | ~10⁻¹⁶ cm² |
| Excited state | ∝ 1/(hν)³ | Depends on fba |

## Key Insights
- Deep center cross-sections depend on local field screening
- Excited state transitions follow hydrogenic patterns
- Constant photoconductivity method provides direct comparison between different impurities