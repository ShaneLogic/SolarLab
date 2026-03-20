---
name: Solar Cell Collection Efficiency Models
description: Calculate and model carrier collection efficiency in solar cells using simple drift-field and Hecht-like models. Use when analyzing voltage-dependent collection, optimizing i-layer thickness, or fitting experimental collection efficiency data for Si, α-Si, aSiGe, or similar solar cell structures.
---

# Solar Cell Collection Efficiency Models

## When to Use
- Calculating voltage-dependent collection efficiency
- Fitting measured collection efficiency data
- Optimizing i-layer thickness in p-i-n cells
- Analyzing carrier drift and recombination in junction regions
- Classifying solar cells by measurable transport parameters

## Model Selection Guide

| Model | Best For | Parameters |
|-------|----------|------------|
| Simple Drift-Field | Wide junction regions, initial classification | S, F |
| Hecht-like | α-Si, aSiGe p-i-n cells | XC, V0 |
| Collection Zone | α-Si:H p-i-n optimization | μD, τ, F |

## Procedure

### Model 1: Simple Collection Efficiency

**Use when**: Junction region is sufficiently wide, transport is field-dominated

```
ηc(V) = μF0(V) / (S + μF0(V))
```

Where:
- μF0(V): Drift velocity (mobility × field) at voltage V
- S: Interface recombination velocity (adjustable)

**Advantages**:
- Only TWO adjustable parameters
- Useful for initial cell classification
- Relates directly to Voc, jsc, FF predictions

### Model 2: Hecht-like Recombination Loss

**Use when**: Fitting measured voltage-dependent collection efficiency, α-Si or aSiGe p-i-n cells

**Primary formulation**:
```
ηc(V) = (XC/V0) * [V0 - V + (V0/2)(1 - exp(-2(V0-V)/V0))]
```

**Alternative (linear field increase)**:
```
ηc(V) = 1 - (XC/(V0-V)) * [1 - exp(-(V0-V)/XC)]
```

Where:
- XC: Fitting parameter = LC/L (collection length to diffusion length ratio)
- V0: Characteristic voltage parameter (eV)
- LC: Collection length = sqrt(kT/e) * sqrt(μτ)

**Field distribution assumption**:
```
F(x) = (V0 - V) / d
```
Where d = i-layer thickness

**Advantages**:
- Only TWO fitting parameters
- Widely validated for α-Si cells
- Accounts for recombination in high-field region

### Model 3: Collection Zone Analysis

**Use when**: Optimizing α-Si:H p-i-n cell i-layer thickness

**Step 1: Determine absorption depth**
- >70% of blue components absorbed in first ~200 nm

**Step 2: Calculate carrier drift distance**
```
Δx = μD * F * τ
```
Where:
- μD: Drift mobility
- F: Acting field
- τ: Carrier lifetime

**Step 3: Define collection zone**
- Region where R ~ 0 (minimal recombination)
- Extends from p-layer interface
- Boundary: where R/g = 0.5

**Step 4: Calculate collection zone width**
- Increases from ~200 nm to ~400 nm
- As hole band mobility increases from 0.1 to 1.0 cm²/Vs

**Optimal i-layer thickness constraints**:
- Collection zone: 200-400 nm
- Blue absorption depth: ~200 nm

## Output
- Collection efficiency ηc(V) as function of voltage
- Fitting parameters (S, F) or (XC, V0)
- Optimal i-layer thickness recommendation
- Collection zone boundary definition

## Model Limitations

**Simple Model**:
- Only valid for sufficiently wide junction regions
- Assumes single recombination velocity characterizes interface

**Hecht-like Model**:
- Field distribution assumptions for α-Si may not hold for CdS/CdTe
- More sophisticated models require more parameters

**Collection Zone Model**:
- Carrier recombination assumed outside collection zone
- Limited absorption depth consideration

## Parameter Reference

| Parameter | Symbol | Typical Range |
|-----------|--------|---------------|
| Interface recombination velocity | S | Device-dependent |
| Drift velocity | μF0 | 10^4-10^6 cm/s |
| Collection length ratio | XC | 0.1-10 |
| Characteristic voltage | V0 | 0.5-1.0 V |
| Hole drift mobility | μD | 0.1-1.0 cm²/Vs |
| Carrier lifetime | τ | 10^-8-10^-6 s |