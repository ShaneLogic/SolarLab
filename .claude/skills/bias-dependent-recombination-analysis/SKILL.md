---
name: Bias-Dependent Recombination Analysis
description: Interpret recombination behavior in pn-junctions under different bias conditions. Use when analyzing net recombination vs. generation, determining carrier traffic direction, or understanding space charge region behavior under forward or reverse bias.
---

# Bias-Dependent Recombination Analysis

## When to Use
- Interpreting the sign and magnitude of recombination rate U
- Analyzing pn-junction behavior under forward or reverse bias
- Determining whether a device is generating or recombining carriers
- Understanding space charge region dynamics

## Prerequisites
- Carrier densities n and p in the region of interest
- Intrinsic carrier density ni for the material
- Bias state of the device

## Procedure

### 1. Calculate Carrier Product

Determine the product of electron and hole densities:
```
carrier_product = n * p
```

### 2. Compare to Equilibrium Value

Calculate the equilibrium reference:
```
equilibrium_product = ni^2
```

### 3. Determine Recombination Behavior

**Decision Tree**:

| Condition | Sign of U | Physical Meaning |
|-----------|-----------|-----------------|
| n*p = ni^2 | U = 0 | Thermal equilibrium - no net generation or recombination |
| n*p < ni^2 | U < 0 | Net thermal generation - carriers being produced |
| n*p > ni^2 | U > 0 | Net recombination - carriers being consumed |

### 4. Assess Recombination Magnitude

Key principles:
- Recombination traffic is large ONLY when both n and p are high
- The **minority carrier limits the recombination rate**
- In pn-junctions, recombination peaks where n ≈ p (inner junction region)

### 5. Spatial Distribution Analysis

In pn-junctions:
- Recombination is substantially higher in the inner junction region
- Both carrier densities are on the same order of magnitude in this region
- Adjacent bulk regions show lower recombination due to minority carrier limitation

## Output
- Sign of U (generation vs. recombination)
- Relative magnitude assessment
- Identification of peak recombination region

## Physical Interpretation

**Reverse Bias (n*p < ni^2)**:
- Carrier product decreased below equilibrium
- Balance is disturbed
- Thermal generation supplies carriers
- Space charge region acts as carrier source

**Forward Bias (n*p > ni^2)**:
- Carrier product exceeds equilibrium
- Carrier surplus exists
- Net recombination through centers
- Space charge region acts as carrier sink

## Key Insight

The minority carrier density is the limiting factor for recombination. Even if majority carriers are abundant, recombination cannot proceed without minority carriers. This is why recombination peaks in junction regions where both carrier types have comparable densities.