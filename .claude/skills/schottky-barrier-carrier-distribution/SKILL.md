---
name: schottky-barrier-carrier-distribution
description: Analyze carrier density distributions and operating regimes in Schottky barriers. Use when calculating electron/hole density profiles, identifying DRO or Boltzmann regions, analyzing quasi-Fermi levels, or understanding minority carrier behavior in metal-semiconductor junctions.
---

# Schottky Barrier Carrier Distribution Analysis

## When to Use
- Calculating carrier density profiles n(x) and p(x)
- Identifying operating regimes (Boltzmann, DRO, DO)
- Analyzing quasi-Fermi level behavior
- Understanding minority carrier distributions
- Characterizing barrier physics under bias

## Operating Regime Classification

| Regime | Bias Condition | Key Feature |
|--------|---------------|-------------|
| Modified Boltzmann | Low bias | Drift term important near x=0 |
| DRO-Range | Reverse bias, -V ≥ 2kT/e | Drift-only, square-root I-V |
| DO-Range | High reverse bias | Diffusion-only for minority carriers |

## Procedure

### Step 1: Identify Operating Regime

**Check bias conditions:**
1. Low bias → Modified Boltzmann Range
2. Reverse bias with -V ≥ 2kT/e → DRO-Range
3. High reverse bias → DO-Range for minority carriers

### Step 2: Calculate Electron Density Distribution

**General solution (Eq. 26.33):**
```
n(x) = nB(x) · [1 + (jn/(e·nc·F(xt))) · (LD/(x - xt))]
```

Where:
- nB(x) = Boltzmann solution
- LD = Debye length
- xt ≈ xD - LD

**Bias effects:**
- Reverse bias (jn < 0): S-shaped distribution
- Forward bias: Boltzmann solution steepens

### Step 3: Calculate Minority Carrier Distribution

**Boltzmann solution for holes (when jp is negligible):**
```
p(x) = p10 · exp[(ψn,j - ψn(x))/ψn,D]
```

**Constraint:** Drift and diffusion must compensate.

### Step 4: Analyze Quasi-Fermi Levels

**At metal/semiconductor interface:**
- Quasi-Fermi levels collapse (complete recombination)

**Electron quasi-Fermi level (EFn):**
- Reverse bias: Drops below EFp, parallel to Ec → DRO-range
- Forward bias: Slopes upward toward right electrode

**Hole quasi-Fermi level (EFp):**
- Low reverse bias: Flat → Boltzmann region
- High reverse bias: Slopes down, parallel to Ec → DO-range
- Forward bias: Slopes upward distinctly

### Step 5: Locate Inflection Point

**Position:**
```
xi = xD - LD
```

**Field at inflection point:**
```
F(xi) ≈ kT/(e·LD)  ≈ 10⁴ V/cm
```

**Maximum current:**
```
jmax ≈ e·μn·nc·F(xi)  ≈ tens of kA/cm²
```

## Key Parameters

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| Debye length | LD | Characteristic screening length |
| Barrier width | xD | Extent of space charge region |
| Diffusion potential | ψn,D | Built-in potential |
| Inflection point | xi | Position of maximum field slope |

## Visualization Reference

For carrier density distribution curves:
- Electron density n(x): Shows S-shape under reverse bias
- Hole density p(x): Exponential decrease in barrier
- Cross-over: p > n possible near interface at high reverse bias

## Output

| Calculation | Result |
|-------------|--------|
| n(x) profile | Electron density vs position |
| p(x) profile | Hole density vs position |
| Regime identification | Boltzmann/DRO/DO classification |
| Quasi-Fermi level behavior | Bias-dependent energy diagrams |