---
name: schottky-diode-iv-analysis
description: Calculate current-voltage characteristics of Schottky diodes under various field conditions. Use when analyzing metal-semiconductor junctions, determining diode operating regimes, or evaluating non-ideal diode behavior.
---

# Schottky Diode Current-Voltage Analysis

## When to Use
- Analyzing Schottky diode I-V characteristics
- Determining current limiting mechanism (drift vs diffusion)
- Evaluating non-ideal diode behavior
- Designing metal-semiconductor junctions

## Decision Flow

```
IF μn·Fj >> v* (high field, high reverse bias)
  → Use Diffusion-Limited Equation
ELSE IF |μn·Fj| << v* (low field, forward/low reverse bias)
  → Use Drift-Limited Equation

Always calculate Shape Factor to quantify non-ideality
```

## Procedure

### Step 1: Determine Operating Regime

Calculate the ratio of drift velocity to thermal velocity:
- **Thermal velocity:** v* ≈ 4×10⁶ cm/s
- **Drift velocity:** μn·Fj

Compare magnitudes to select appropriate equation.

### Step 2: Apply Current Equation

**For High Field (μnFj >> v*):**
```
jn = e·nc·v*·[exp(eV/kT) - 1]
```
- Perfect current saturation in reverse direction
- "Ideal characteristics" behavior

**For Low Field (|μnFj| << v*):**
```
jn = e·nc·μn·Fj·[exp(eV/kT) - 1]
```
- Drift-limited transport
- No true saturation (Fj is bias-dependent)

### Step 3: Calculate Shape Factor

```
SF = 1 / [1 + v*/|μn·Fj|]
```

**Interpretation:**
- SF ≈ 1: Near-ideal behavior (larger reverse bias)
- SF < 1: Non-ideal behavior (low bias, forward bias)

### Step 4: Apply Modified Boundary Condition (Advanced)

For more accurate modeling, account for carrier density jump at interface:
```
nj = nc / [1 - jn/(e·nc·vn*)]
```

Modified I-V characteristic:
```
jn = e·μn·nc·Fj·[exp(eV/kT) - 1] / [1 + jn/(e·nc·vn*)]
```

## Key Parameters

| Parameter | Symbol | Typical Value | Units |
|-----------|--------|---------------|-------|
| Thermal velocity | v* | 4×10⁶ | cm/s |
| Carrier density at boundary | nc | Variable | cm⁻³ |
| Electron mobility | μn | Material-dependent | cm²/V·s |
| Junction field | Fj | Bias-dependent | V/cm |

## Output Interpretation

| Result | Meaning |
|--------|---------|
| Perfect saturation | Diffusion-limited regime |
| Square-root I-V dependence | DRO-range behavior |
| SF < 1 | Non-ideal characteristics |
| No saturation | Drift-limited regime |

## References
- See `references/schottky-equations.md` for detailed derivations