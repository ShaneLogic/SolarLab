---
name: schottky-barrier-validation
description: Validate Schottky barrier approximations, classify current contributions, and correctly identify junction types. Use when verifying model accuracy, decomposing total current, or interpreting carrier density crossings in barrier analysis.
---

# Schottky Barrier Validation and Classification

## When to Use
- Verifying Schottky approximation validity
- Decomposing total current into components
- Interpreting carrier density crossings
- Identifying junction type correctly

## Validation Tasks

### 1. Schottky Approximation Accuracy

**Check validity condition:**
```
nc/Nd ratio determines approximation accuracy
```

| nc/Nd Ratio | Error (ΔFc/Fc) | Validity |
|-------------|----------------|----------|
| < 10⁻⁵ | < 5% | Excellent |
| 10⁻⁵ to 10⁻² | 5-10% | Acceptable |
| > 10⁻² | > 10% | Poor - avoid |

**Warning:** Near x = xD, Schottky approximation is unsatisfactory because space charge density ρ has not reached constant value e·Nd.

**Rule:** nc must be several orders of magnitude smaller than Nd for valid approximation.

### 2. Current Contribution Classification

**Decompose total current into four components:**

| Component | Symbol | Description |
|-----------|--------|-------------|
| Divergence-free majority current | jni | Controlled by n(x=0) = nj |
| Divergence-free minority current | jpi | Controlled by p(x=d1) = pj |
| Minority carrier GR-current | j⁽ᵖ⁾(x) | Generation-recombination for holes |
| Majority carrier GR-current | j⁽ⁿ⁾(x) | Generation-recombination for electrons |

**Note:** Each current has both drift and diffusion contributions in parts of the device.

### 3. Junction Type Identification

**Critical warning:** Carrier density crossing (p > n) does NOT imply pn-junction!

**Procedure:**
1. Observe crossing of n(x) and p(x) at position xc (bias-dependent)
2. Check for sign change in space charge (slope of field)
3. Evaluate space charge = sum of free AND trapped charges
4. Compare current magnitudes: jn >> jp (typically > 5 orders of magnitude)

**Conclusion:** If no space charge sign change, device remains n-type barrier regardless of p > n inversion.

## Decision Matrix

| Observation | Interpretation | Action |
|-------------|----------------|--------|
| nc/Nd > 10⁻² | Approximation unreliable | Use exact solution |
| p(x) crosses n(x) | Possible inversion | Check space charge sign |
| jn >> jp | Majority carrier dominated | Treat as n-type barrier |
| Space charge sign change | True pn-junction | Apply junction models |

## Common Misinterpretations

| Misinterpretation | Correct Understanding |
|-------------------|----------------------|
| p > n means pn-junction | Only space charge sign change indicates pn-junction |
| Schottky approximation always valid | Only valid when nc << Nd |
| All current components equal | jni typically dominates |
| Crossing position fixed | xc is bias-dependent |

## Output

| Validation | Result |
|------------|--------|
| Approximation check | Valid/Invalid with error estimate |
| Current decomposition | Four component values |
| Junction identification | n-type barrier or pn-junction |