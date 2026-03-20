---
name: Frenkel-Poole Effect Analysis
description: Calculate barrier lowering and enhanced thermal ionization at Coulomb-attractive defect centers under electric fields. Use when analyzing field ionization mechanisms, trap emission, field quenching, or carrier generation in semiconductors with defects. This is the lowest-field field-ionization mechanism.
---

# Frenkel-Poole Effect Analysis

## When to Use
- Analyzing field-enhanced ionization at defect centers
- Calculating barrier lowering in Coulomb-attractive traps
- Designing devices with field quenching behavior
- Evaluating deep donor depletion mechanisms
- Working with CdS:Cu or similar trap systems

## Prerequisites
- External electric field present
- Defect with charge Z (Coulomb-attractive center)
- Knowledge of material permittivity

## Core Mechanism

The Frenkel-Poole effect lowers the ionization barrier at Coulomb-attractive centers when an electric field is applied, enabling enhanced thermal ionization at lower temperatures.

## Barrier Lowering Formula

### General Formula
```
δE = β × √F
```
Where:
```
β = (Z × e³ / π × ε × ε₀)^(1/2)
```

Or equivalently:
```
δE = Z × e³/² × F¹/²
```

### Parameters
- δE = Barrier lowering (eV)
- Z = Charge of defect center
- F = Electric field (V/cm)
- ε = Relative permittivity
- ε₀ = Vacuum permittivity

## Procedure

### Step 1: Identify Defect Characteristics
- Determine defect charge Z
- Confirm Coulomb-attractive nature
- Note material permittivity ε

### Step 2: Calculate Barrier Lowering
- Apply the Frenkel-Poole formula
- Typical field range: 10-100 kV/cm

### Step 3: Evaluate Thermal Ionization Enhancement
```
e_tc = s_n × v_th × exp[-(Ec - Et - δE) / kT]
```
Where:
- e_tc = Thermal ionization coefficient
- s_n = Capture cross-section
- v_th = Thermal velocity
- Ec - Et = Trap energy depth

### Step 4: Determine Critical Thresholds
- Calculate field for δE = 2kT (significant emission)
- Evaluate field quenching onset

## Critical Thresholds (CdS:Cu Example)

| Field | Effect |
|-------|--------|
| > 20 kV/cm | New process must be considered |
| 23 kV/cm | Barrier lowered by 2kT, hole release begins |
| > 50 kV/cm | Field quenching markedly lowers electron density |

## Example Calculations

### CdS with Doubly Charged Cu (Z=2)
- At F = 50 kV/cm, ε = 10:
- Distance from funnel center: 35 Å
- δE ≈ 30 meV

### Order of Magnitude
- Typical semiconductors: δE significant at ~10 kV/cm

## Applications
1. Field-enhanced deep donor depletion
2. Field quenching in photoconductors
3. Trap emptying in memory devices
4. Carrier generation in high-field regions