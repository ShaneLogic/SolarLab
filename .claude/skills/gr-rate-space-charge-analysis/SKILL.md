---
name: gr-rate-space-charge-analysis
description: Calculate generation, recombination, and net GR rates in Schottky barrier space-charge regions. Use when analyzing carrier dynamics under reverse/forward bias, computing spatial distributions of GR rates, or modeling current contributions from space-charge regions.
---

# GR Rate Dynamics in Space-Charge Region

## When to Use
Apply this skill when:
- Calculating generation or recombination rates in Schottky barrier regions
- Analyzing bias-dependent carrier dynamics
- Computing net GR rate U(x) for current modeling
- Determining spatial distributions of g(x) and r(x) across the barrier

## Prerequisites
- Carrier density distributions: n(x) and p(x)
- Recombination center parameters (density Nr, energy level Er)
- Carrier lifetimes: τn and τp
- Intrinsic carrier density ni

## Procedure

### Step 1: Calculate Auxiliary Density Term n*
```
n* = (ni² / Nr) × [1 + exp((Er - Ei) / kT)]
```
Where:
- ni = intrinsic carrier density
- Nr = recombination center density
- Er - Ei = energy difference from intrinsic level

### Step 2: Calculate Generation Rate g(x)
```
g(x) = [ni²/(τp×n) + n/τn] / [1 + n/n*]
```

**Region-specific approximations:**
- **n-type bulk (n >> p):** g(x) ≈ ni²/(τp×n)
- **Barrier region maximum:** g_max ≈ ni/τ_eff (when Er ≥ Ei and n decreases to nc)

### Step 3: Calculate Recombination Rate r(x)
```
r(x) = (p/τp) × [1 + n/n*] / [1 + p/(ni²/n*)]
```

### Step 4: Determine Net GR Rate U(x)
```
U(x) = g(x) - r(x)
```

**Bias-dependent behavior:**

| Bias Condition | Carrier Behavior | Net Rate | Physical Meaning |
|----------------|------------------|----------|------------------|
| Reverse Bias | n(x), p(x) decrease | U(x) > 0 | Net Generation |
| Forward Bias | n(x), p(x) increase | U(x) < 0 | Net Recombination |
| Zero Bias | Equilibrium | U(x) = 0 | Balance |

## Output
Spatial distribution of:
- Generation rate g(x)
- Recombination rate r(x)  
- Net GR rate U(x)

## Constraints
- Assumes thermal excitation for generation rate approximation
- Valid for Schottky barrier space-charge regions
- Requires known carrier density profiles n(x) and p(x)