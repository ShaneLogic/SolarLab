---
name: Quasi-Equilibrium Carrier Analysis
description: Calculate minority carrier densities and quasi-Fermi level positions in semiconductors under high optical generation. Use when analyzing solar cells under sunlight, determining carrier statistics under optical excitation, or verifying constant lifetime conditions.
---

# Quasi-Equilibrium Carrier Analysis

## When to Use
- Calculating carrier densities under high optical generation (sunlight)
- Analyzing solar cell behavior under illumination
- Determining quasi-Fermi level splitting
- Verifying constant lifetime approximation validity
- Modeling photovoltaic device operation

## Prerequisites
- Optical generation rate (go)
- Minority carrier lifetime (τp)
- Recombination center density (Nr)
- Capture coefficients (ccn, ccr)

## Procedure

### 1. Calculate Minority Carrier Density

Under high optical generation:
```
Δp = go * τp
```

**Expected ranges**:
| Material Type | Lifetime | Δp Range |
|---------------|----------|----------|
| Direct gap (low lifetime) | ~10^-11 s | ~10^10 cm^-3 |
| Indirect gap (high lifetime) | >10^-6 s | >10^16 cm^-3 |

### 2. Determine Quasi-Fermi Level Split

The split between electron and hole quasi-Fermi levels:
- Proportional to logarithm of light intensity
- Majority quasi-Fermi level remains essentially unchanged
- Determined by density of shallow donors

Split magnitude:
```
ΔE_F = (kT/e) * ln( (n0 + Δn) * (p0 + Δp) / ni^2 )
```

### 3. Verify Constant Lifetime Condition

Check if lifetime remains constant:

**Required condition**:
```
n * ccn >> Nr * ccr
```

Where:
- n: Electron density
- ccn: Capture coefficient for electrons
- Nr: Density of recombination centers
- ccr: Capture coefficient for recombination centers

**Physical meaning**:
- Density of available recombination centers remains constant (nr ≈ Nr)
- Electron capture into centers is more rapid than minority carrier (hole) capture

### 4. Handle Hole Trapping Effects

If substantial hole trapping occurs:
- Quasi-Fermi level for holes may pin with increasing generation
- Pinning continues until traps are filled
- This modifies the simple quasi-equilibrium model

### 5. Assess Validity of Approximation

The quasi-equilibrium approximation assumes:
- ✓ High generation rate (sunlight absorption)
- ✓ Steady-state condition achieved
- ✓ Majority quasi-Fermi level unchanged
- ✓ Constant lifetime condition satisfied

## Output
- Minority carrier density Δp
- Quasi-Fermi level split magnitude
- Validity status of constant lifetime assumption
- Warning if hole trapping is significant

## Key Parameters

| Parameter | Symbol | Typical Units |
|-----------|--------|---------------|
| Optical generation rate | go | cm^-3 s^-1 |
| Minority carrier lifetime | τp | s |
| Recombination center density | Nr | cm^-3 |
| Electron capture coefficient | ccn | cm^3/s |
| Hole capture coefficient | ccr | cm^3/s |

## Physical Interpretation

Under optical excitation:
1. Electron-hole pairs generated at rate go
2. Minority carriers accumulate (Δp = goτp)
3. Quasi-Fermi levels split to accommodate non-equilibrium
4. Majority carrier density essentially unchanged
5. System reaches steady state with enhanced minority carrier population

## Limitations

- Assumes majority quasi-Fermi level remains unchanged
- Requires constant lifetime condition
- May not hold at very high injection levels
- Hole trapping effects require modified analysis