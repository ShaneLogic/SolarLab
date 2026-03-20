# Reaction Kinetics - Complete Derivation

## Balance Equations

### Complete Steady-State Equations (Eqs 31.9-31.13)

Considering all competing transitions:

```
Electron balance: g_o + c_vn n_r N_r - c_cn n (N_r - n_r) - (thermal terms) = 0
Hole balance:   g_o + c_vp (N_r - n_r) N_r - c_cv p n_r - (thermal terms) = 0
```

**Simplification:** Thermal excitation from deep levels (terms in parenthesis) are usually negligible.

### Neutrality Condition

**Eq 31.14 - Total charge neutrality:**
```
n - p + N_r^- - N_d = 0
```

**Eq 31.15 - Occupied recombination centers:**
```
N_r^- = N_r / [1 + (n c_cn)/(p c_cv)]
```

This relates the density of negatively charged centers to the carrier densities and capture coefficients.

### Simplified Balance

**Eq 31.16 - Electron balance:**
```
g_o = (n - n₀) c_cn n_r
```

**Eq 31.17 - Hole balance:**
```
g_o = (p - p₀) c_cv n_r
```

These state that the optical generation rate equals the net recombination rate for each carrier type.

### Carrier Density Solutions

#### Case 1: n_r ≈ N_r (Most recombination centers available)

**Minority carrier density (Eq 31.18):**
```
Δp = g_o τ_p
```

**Hole lifetime (Eq 31.19):**
```
τ_p = 1/(N_r c_cv)
```

This is the standard linear photoconductivity relation.

#### Case 2: General solution

Substitute n_r from neutrality condition into balance equations. This yields an implicit polynomial expression:

**Eq 31.21:**
```
g_o = c_cn c_cv N_r (n p - n_i²) / (c_cn n + c_cv p)
```

### High Generation Rate Limit

When n ≈ p (very high generation rates, neglecting trapping):

**Eq 31.24 - Bottleneck equation:**
```
n = p = √[g_o / (c_cr + c_rv)]
```

#### Physical Interpretation

- At very high generation, carrier density is limited by the **smaller** capture coefficient
- The slower capture process becomes the bottleneck
- This determines the maximum achievable carrier density

#### Special Cases

**If c_cn ≪ c_cv:**
```
n ≈ √(g_o / c_cn)
```
Electron capture is limiting.

**If c_cv ≪ c_cn:**
```
p ≈ √(g_o / c_cv)
```
Hole capture is limiting.

## Transition Between Regimes

| Generation Rate | Dominant Term | Carrier Density Relation |
|-----------------|---------------|--------------------------|
| Low | Linear term | Δp = g_o τ_p |
| Medium | Full solution | Use Eq 31.21 |
| High | Square root | n = p = √[g_o / (c_cr + c_rv)] |

## Application to Photoconductors

In photoconductors (low thermal majority carrier density):
- n and p both change significantly under illumination
- Full balance equations must be used
- Bottleneck equation applies at high light levels
- Sensitization (trap engineering) modifies c_cv or c_cn