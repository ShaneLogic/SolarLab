# GR Rate Formulas - Detailed Reference

## Complete Formula Derivations

### Auxiliary Density Term n* (Eq 29.2)

The auxiliary density n* characterizes the recombination center occupancy:

```
n* = (ni² / Nr) × [1 + exp((Er - Ei) / kT)]
```

**Physical meaning:**
- Determines the carrier density at which recombination center statistics transition
- Depends on recombination center energy level relative to intrinsic Fermi level

### Generation Rate g(x) Detailed Analysis

**Full formula (Eq 29.1):**
```
g(x) = [ni²/(τp×n) + n/τn] / [1 + n/n*]
```

**Physical interpretation:**
- First term in numerator: hole generation via electron excitation from valence band
- Second term in numerator: electron generation component
- Denominator: accounts for recombination center occupancy effects

### Region-Specific Behavior

#### n-type Bulk Region (Eq 29.3)
When n >> p and n ≈ Nd:
```
g(x) ≈ ni²/(τp×n)
```
This represents the minority carrier generation rate in quasi-neutral bulk.

#### Barrier Region Transition
As n(x) decreases from Nd toward nc:
- Generation rate increases progressively
- The denominator term [1 + n/n*] becomes significant

#### Maximum Generation Rate (Eq 29.4)
When Er - Ei ≥ 0, the third term in denominator dominates:
```
g_max ≈ ni / τ_eff
```
Where τ_eff is an effective lifetime combining τn and τp contributions.

### Recombination Rate r(x) (Eq 29.5)

```
r(x) = (p/τp) × [1 + n/n*] / [1 + p/(ni²/n*)]
```

This follows SRH (Shockley-Read-Hall) statistics adapted for barrier region conditions.

### Net GR Rate Under Different Bias

**Reverse Bias:**
- Carrier densities drop below equilibrium values
- r(x) < g(x)
- Net generation contributes to reverse saturation current
- Important for photodiode and solar cell operation

**Forward Bias:**
- Carrier densities exceed equilibrium values
- r(x) > g(x)
- Net recombination affects forward current characteristics
- Contributes to ideality factor deviation from unity

**Zero Bias (Equilibrium):**
- r(x) = g(x) at all positions
- U(x) = 0 throughout the device
- Detailed balance maintained

## Parameter Reference Table

| Symbol | Description | Units | Typical Range |
|--------|-------------|-------|---------------|
| g(x) | Generation rate | cm⁻³s⁻¹ | 10⁸ - 10²² |
| r(x) | Recombination rate | cm⁻³s⁻¹ | 10⁸ - 10²² |
| U(x) | Net GR rate | cm⁻³s⁻¹ | -10²² to +10²² |
| n* | Auxiliary density | cm⁻³ | Device-dependent |
| τn, τp | Carrier lifetimes | s | 10⁻⁹ - 10⁻⁶ |
| Nr | Recombination center density | cm⁻³ | 10¹² - 10¹⁷ |
| ni | Intrinsic carrier density | cm⁻³ | Material-dependent |

## Calculation Workflow Summary

1. Obtain n(x) and p(x) profiles from Poisson/continuity equations
2. Compute n* from material parameters
3. Evaluate g(x) using appropriate approximation for each region
4. Evaluate r(x) from carrier densities
5. Calculate U(x) = g(x) - r(x)
6. Integrate U(x) over space-charge region for current contribution