---
name: photoconductivity-regime-modeling
description: Model photoconductivity in different regimes: intrinsic (band-to-band, bimolecular) at high generation rates, or extrinsic with deep traps (monomolecular to bimolecular transition). Use to determine dominant recombination mechanisms and extract trap densities.
---

# Photoconductivity Regime Modeling

Model photoconductivity in different regimes based on generation rates and trap characteristics to determine recombination mechanisms and extract material parameters.

## When to Use
- High generation rates with direct band-to-band transitions
- Photoconductivity dominated by single deep trap level
- Determining dominant recombination mechanism
- Extracting trap density from photoconductivity measurements
- Ambipolar photoconductors

## Regime Determination

**IF** high generation rates AND direct band-to-band recombination dominant:
→ Use **Intrinsic Photoconductivity** model

**ELSE IF** single deep trap level dominates:
→ Use **Deep Trap Photoconductivity** model

## Intrinsic Photoconductivity (Bimolecular Regime)

### Conditions
- High generation rates
- Direct band-to-band transitions
- Incremental densities >> thermally generated densities (n >> n_0, p >> p_0)

### Rate Equation
```
dn/dt = g - c_cv * (n * p - n_0 * p_0)
```

### Steady State Solution (dn/dt = 0)

With n = p (equal incremental densities):
```
n = g_0 / c_cv
```

This is the **bimolecular recombination** relation.

**Interpretation:** Intrinsic recombination occurs when other recombination paths are saturated.

### Conductivity
Photoconductivity is **ambipolar** (depends on both electrons and holes).

## Photoconductivity with Deep Traps

### Conditions
- Single deep trap level dominates
- Deep traps completely filled (p_t ≈ N_t)

### Balance Equations
Include trapping term (dp_t/dt) and neutrality condition:
```
p + p_t = n + n_a
```

### Steady State Assumptions
- Time derivatives vanish
- Trapping terms drop out: g = c_ca * n * p_a
- Deep traps completely filled: p_t ≈ N_t

### Electron Density Calculation

```
n = (1 / (2 * c_ca)) * (-N_t + sqrt(N_t^2 + 4 * g_0))
```

### Regimes

**Small Optical Generation** (g_0 < N_t^2 * c_ca / 4):
```
n = g_0 / (c_ca * N_t)
```
- Reduces to **monomolecular relation**
- Indicates each electron finds constant density of recombination sites

**High Optical Generation:**
```
n ≈ sqrt(g_0 / c_ca)
```
- Converts back to **bimolecular relation**

**Transition point:** Occurs at n ≈ N_t

### Trap Density Extraction

**Procedure:**
1. Plot n vs g_0 on log-log scale
2. Identify break between linear and square-root branches
3. Break point occurs at n ≈ N_t
4. Read trap density N_t from transition

## Key Parameters

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| Electron density | n | Free electron density |
| Hole density | p | Free hole density |
| Generation rate | g_0 | Optical generation rate |
| Bimolecular coefficient | c_cv | Band-to-band recombination coefficient |
| Activator coefficient | c_ca | Recombination coefficient via activator |
| Trap density | N_t | Deep trap density |
| Trapped holes | p_t | Density of holes in traps |

## Output
- Steady state carrier density (n)
- Identification of dominant regime (monomolecular vs bimolecular)
- Trap density (N_t) from photoconductivity vs generation plot
- Recombination mechanism classification