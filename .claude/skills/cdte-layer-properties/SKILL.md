---
name: cdte-layer-properties
description: Use when characterizing, modeling, or analyzing CdTe absorber layer properties in thin-film solar cells (CdS/CdTe heterojunctions). Provides physical parameters including electrical conductivity, carrier mobilities, effective masses, bandgap, dielectric constants, and layer specifications for device modeling and characterization tasks.
---

# CdTe Layer Physical Properties

## When to Use

Use this skill when:
- Characterizing CdTe absorber layers in CdS/CdTe solar cells
- Modeling device performance requiring CdTe material parameters
- Analyzing thin-film properties of deposited CdTe layers
- Setting up simulations with CdTe physical constants
- Calculating carrier transport or optical properties

## Prerequisites

- CdTe layer must be deposited (or being modeled)
- Properties may vary with deposition and treatment methods
- Values provided are for typical processed CdTe

## Physical Parameters

### Electrical Properties

| Property | Value | Notes |
|----------|-------|-------|
| Conductivity type | p-type | Native doping |
| Hole density | 10¹⁴ – 10¹⁵ cm⁻³ | Typical range |
| Hole mobility (μh) | 50 – 80 cm²/Vs | Hall mobility |
| Electron mobility (μe) | 500 – 1000 cm²/Vs | Hall mobility |

### Effective Masses

| Carrier | Effective Mass |
|---------|---------------|
| Electrons | mn = 0.09 m₀ |
| Holes | mp = 0.35 m₀ |

### Bandgap Properties

- Bandgap at 300K: Eg = 1.5 eV
- Temperature coefficient: Varies (see Birkmire and McCandless 1992)

### Dielectric Properties

| Property | Value |
|----------|-------|
| Static dielectric constant (εs) | 10 |
| High frequency dielectric constant (ε∞) | 7.1 |

### Device Layer Specifications

- Typical thickness range: 2 – 8 μm

### Common Deposition Methods

1. Close space sublimation (most typical for devices)
2. Vacuum evaporation
3. Other methods applicable (similar to CdS deposition)

## Usage Notes

- Actual properties depend on deposition conditions and post-deposition treatments
- Reference: Birkmire and McCandless (1992)
- Thermal expansion coefficient values available in source literature