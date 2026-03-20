---
name: cds-layer-deposition-optimization
description: Optimize CdS window layer deposition and thickness for CdS/CdTe solar cells. Use when designing or fabricating CdS/CdTe cells to balance Voc and jsc trade-offs.
---

# CdS Layer Deposition and Thickness Optimization

## When to Use
Apply this skill when:
- Depositing CdS window layer for CdS/CdTe solar cells
- Optimizing cell efficiency through layer thickness adjustment
- Selecting deposition method for CdS layer
- Troubleshooting Voc or jsc issues related to window layer

## Prerequisites
- Conducting glass substrate prepared
- CdS source material available
- Understanding of junction formation requirements

## Deposition Method Selection

Choose from these polycrystalline deposition methods:
- Vacuum deposition
- Vapor transport deposition
- Sputtering
- Close space sublimation
- Spray deposition
- Screen printing

**Key Insight**: Final cell efficiency is remarkably insensitive to deposition method choice after careful recrystallization.

## Thickness Specifications

- **Typical range**: 30-80nm
- **Standard value**: ~60nm
- **Recrystallization**: May be required before CdTe deposition in some cases

## Optimization Strategy

Balance two competing factors:
1. **Open circuit voltage (Voc)**: INCREASES with CdS thickness due to better junction formation and coverage
2. **Short circuit current (jsc)**: DECREASES with CdS thickness due to additional absorption of optically active light

CdS absorbs blue/UV photons that cannot contribute to CdTe photocurrent.

**Target**: Find minimum thickness that still provides good Voc.

## Execution Steps

1. Select deposition method based on equipment availability and process requirements
2. Deposit initial CdS layer at ~60nm thickness
3. Evaluate Voc and jsc performance
4. Adjust thickness based on trade-off analysis:
   - If Voc is insufficient: Increase thickness
   - If jsc is insufficient: Decrease thickness
5. Iterate to find optimal balance point

## Variables

- `CdS_thickness`: Layer thickness in nm (range: 30-80nm)
- `Voc`: Open circuit voltage (increases with CdS thickness)
- `jsc`: Short circuit current (decreases with CdS thickness)

## Expected Output

Optimized CdS thickness value that balances Voc and jsc requirements for maximum cell efficiency.