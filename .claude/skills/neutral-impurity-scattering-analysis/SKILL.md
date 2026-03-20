---
name: neutral-impurity-scattering-analysis
description: Calculate carrier mobility due to neutral impurity and lattice defect scattering in semiconductors at low temperatures (T < 100K). Use this skill when analyzing semiconductor transport properties at cryogenic temperatures where ionized impurity density has decreased due to carrier freeze-out and phonon scattering has diminished due to phonon freeze-out.
---

# Neutral Impurity Scattering Analysis

## When to Use This Skill

Apply this skill when:
- Temperature is below 100K
- Ionized impurity density has decreased due to carrier freeze-out
- Phonon scattering has decreased due to phonon freeze-out
- Neutral defect density N_n is known

## Prerequisites

Before executing, ensure:
- Neutral defect density N_n is known
- Temperature T < 100K for importance
- Ionized impurity density decreased due to carrier freeze-out

## Execution Procedure

### Step 1: Estimate Scattering Cross Section

1. Use hydrogen atom analogy for electron scattering
2. Typical cross section: s_n ≈ 10^(-15) cm²
3. Apply Erginsoy approximation for T > 20K:
   - s_n ≈ 20a_qH/|k|
   - where a_qH = Bohr radius, |k| = wave vector magnitude

### Step 2: Calculate Mobility

1. Use gas-kinetic estimate for collision time
2. Calculate mobility using standard formula (see references)
3. **Key Result**: Mobility is INDEPENDENT of temperature (for T > 20K)

### Step 3: Apply Temperature Corrections

**For T > 20K (Erginsoy regime):**
- Use standard Erginsoy approximation
- Temperature-independent mobility result

**For T < 20K:**
- Apply Blagosklonskaya correction
- Include ionization energy E_i of the impurity
- Account for energy-dependent screening

### Step 4: Consider Spin Effects

- Scattering differs for parallel vs antiparallel electron spins
- Triplet state vs singlet state affects results

### Step 5: Multi-Valley Considerations

For multi-valley semiconductors:
- Results similar to Blagosklonskaya at low temperatures
- Only slight mobility reduction at T > 10K

## Validity Constraints

- Erginsoy approximation: valid for T > 20K
- Below 20K: screening depends on energy
- Spin-dependent scattering affects results

## Output

Return temperature-independent mobility (for T > 20K) or corrected mobility values with validity range specified.