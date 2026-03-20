---
name: junction-rectification-characteristics
description: Analyze current rectification behavior in asymmetrically doped junctions (nn-junctions). Use when designing rectifiers, evaluating I-V characteristics, or determining how doping step-size affects rectification efficiency and current scales.
---

# Junction Rectification Characteristics

Analyze current-voltage characteristics and evaluate rectification efficiency in asymmetrically doped semiconductor junctions.

## When to Use
- Designing or analyzing rectifying devices (diodes)
- nn-junctions with asymmetric doping profiles
- Evaluating I-V characteristics under AC or DC bias
- Determining effect of doping step-size on device performance

## Analysis Procedure

### 1. Obtain Current-Voltage Characteristics

- Derived from solutions of transport and Poisson equations
- Shows **non-ohmic behavior** due to expansion/contraction of resistive region
- Characteristic curve typically asymmetric around V = 0

### 2. Analyze Rectification Asymmetry

**Current behavior:**
- Forward bias: Current is **larger**
- Reverse bias: Current is **smaller**

**AC operation:**
- When sinusoidal AC voltage is supplied
- Asymmetry produces **net forward DC component**

### 3. Evaluate Impact of Doping Step-Size

**Step-size metric:** Donor density reduction factor between regions

**Effects of larger step:**
- More pronounced I-V curve asymmetry
- Stronger curvature (rectifying shape)
- Higher current scales

**Example comparison:**

| Doping Step | Current Scale | Rectification |
|-------------|---------------|---------------|
| Small (10:1) | mA/cm² | Moderate |
| Large (10^5:1) | kA/cm² | Strong |

**Doping example:** 10^17 cm⁻³ → 10^12 cm⁻³ (factor of 10^5)

### 4. Non-Ohmic Behavior Assessment

**Root cause:** Expansion/contraction of resistive region under bias
- Forward bias: Contraction (lower resistance)
- Reverse bias: Expansion (higher resistance)

## Design Considerations

**For improved rectification:**
- Increase doping step-size between junction regions
- Ensure abrupt (step-like) doping transition
- Optimize junction width relative to depletion region

**For specific current requirements:**
- Adjust absolute doping levels to scale current
- Larger absolute doping → higher current at given voltage

## Output
- Rectification efficiency assessment
- Current scale determination (mA/cm² to kA/cm²)
- I-V curve asymmetry quantification
- Doping step-size optimization recommendations