---
name: cds-cdte-solar-cell-design
description: Configure and analyze CdS/CdTe thin-film solar cell structures in superstrate configuration. Use when designing, evaluating, or troubleshooting CdS/CdTe polycrystalline thin-film solar cells with conducting glass substrates.
---

# CdS/CdTe Solar Cell Design

## When to Use
- Designing new CdS/CdTe thin-film solar cells
- Analyzing existing cell performance against benchmarks
- Evaluating structural parameters for superstrate configuration
- Troubleshooting performance issues in CdTe-based photovoltaics

## Prerequisites
- Conducting glass substrate available
- CdS source material for window layer
- CdTe source material for absorber layer
- Copper-containing back contact materials

## Constraints
- Applies ONLY to superstrate configuration (light enters through glass substrate)
- Does NOT apply to monocrystalline CdTe cells
- Does NOT apply to substrate configuration cells (backwall designs)

## Configuration Procedure

### Step 1: Verify Cell Type Compatibility
Confirm the target cell matches:
- Polycrystalline thin-film structure
- Superstrate configuration (conducting glass as base)
- CdS/CdTe material system

### Step 2: Configure Layer Stack
Build the structure from glass substrate upward:

1. **CdS Window Layer**
   - Deposit on conducting glass plate
   - Target thickness: 60nm (typical)

2. **CdTe Absorber Layer**
   - Cover the CdS layer completely
   - Target thickness: ~2μm

3. **Back Contact**
   - Apply copper-containing material as final layer

### Step 3: Validate Performance Against Benchmarks

| Parameter | Typical Range | High-Efficiency Target |
|-----------|---------------|------------------------|
| Efficiency | 8-16% | 14-18% |
| Voc | 500-860mV | 845mV+ |
| jsc | 16-26mA/cm² | 25-26mA/cm² |
| Fill Factor | 63-76% | 75%+ |

### Step 4: Diagnose Performance Issues
If parameters fall below expected ranges:
- Check CdS thickness uniformity (affects jsc)
- Verify CdTe layer completeness (affects absorption)
- Evaluate back contact quality (affects fill factor)
- Review processing method consistency

## High-Efficiency Benchmark
Reference cell at 16.5% efficiency:
- Voc: 845mV
- jsc: 25.88mA/cm²
- Fill factor: 75.51%