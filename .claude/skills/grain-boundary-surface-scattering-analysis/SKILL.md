---
name: grain-boundary-surface-scattering-analysis
description: Calculate effective carrier mobility in polycrystalline and thin-film semiconductors accounting for grain boundary barriers and surface scattering. Use when analyzing microcrystalline materials, thin semiconductor platelets, field-effect transistors, or any device where grain size is smaller than electrode spacing or mean free path is comparable to sample dimensions.
---

# Grain Boundary and Surface Scattering Analysis

## When to Use This Skill

Apply this skill when:
- Analyzing mobility in polycrystalline or microcrystalline semiconductors
- Grain size is smaller than electrode spacing
- Mean free path is comparable to sample dimensions
- Working with thin-film transistors or solar cells
- Low-temperature analysis of high-mobility semiconductors
- Modeling carrier transport across grain boundaries

## Prerequisites

Before applying this analysis, obtain:
- Grain size (from microscopy or XRD)
- Electrode spacing (device geometry)
- Platelet/film thickness d
- Mean free path λ (from bulk mobility)
- Temperature T (for thermal activation models)

## Analysis Workflow

### Step 1: Determine Dominant Scattering Mechanism

| Condition | Dominant Mechanism |
|-----------|-------------------|
| Grain size < electrode spacing | Grain boundary scattering |
| Mean free path ≈ sample thickness | Surface scattering |
| Both conditions present | Combined analysis required |

### Step 2: Grain Boundary Analysis

**Physical Model:**
- Grain boundaries contain high trap density
- Traps become occupied → interface charges
- Space-charge triple layer forms (see references)
- Potential barrier V_b impedes carrier flow

**Barrier Height Estimation:**
```
V_b ∝ Σ_i × L_D
```
Where:
- Σ_i = surface charge density at interface
- L_D = Debye length

**Note:** Σ_i is often unknown; extract V_b from experimental data instead.

**Mobility Calculation:**
```
μ_b = μ_0 × exp(-eV_b / kT)
```
Where:
- μ_b = effective mobility across grain boundaries
- μ_0 = intra-grain mobility (higher than μ_b)
- eV_b = barrier height in eV
- kT = thermal energy

**Procedure:**
1. Measure mobility vs. temperature
2. Plot ln(μ) vs. 1/T
3. Extract barrier height from slope
4. Use μ_0 from single-crystal reference or high-T extrapolation

### Step 3: Surface Scattering Analysis

**Identify Surface Interaction Type:**
- Perfect scattering surfaces
- Recombination-active surfaces
- Space-charge surfaces (extends few Debye lengths into bulk)

**Specular vs. Non-Specular Scattering:**

| Scattering Type | Effect on Mobility |
|-----------------|-------------------|
| Specular (s = 1) | No reduction |
| Non-specular (s = 0) | Maximum reduction |
| Mixed (0 < s < 1) | Partial reduction |

**Combined Relaxation Time:**
```
1/τ = 1/τ_B + 1/τ_s
```

**Mobility Ratio Calculation:**
```
μ/μ_B = f(d/λ, s)
```
Where:
- d = platelet thickness
- λ = mean free path
- s = fraction of specular scattering events

See references for complete formula and graphical solution.

### Step 4: Combined Analysis (When Both Mechanisms Present)

1. Calculate grain boundary mobility μ_b
2. Calculate surface scattering factor μ/μ_B
3. Apply surface factor to intra-grain mobility μ_0
4. Result gives effective mobility through both mechanisms

## Key Variables

| Variable | Symbol | Units | Description |
|----------|--------|-------|-------------|
| Effective mobility | μ_b | cm²/V·s | Mobility across grain boundaries |
| Intra-grain mobility | μ_0 | cm²/V·s | Mobility within single grain |
| Barrier height | V_b | V | Potential barrier at grain boundary |
| Specularity | s | - | Fraction of specular scattering (0-1) |
| Mean free path | λ | cm | Average carrier path between collisions |
| Thickness | d | cm | Platelet or film thickness |

## Output

Return effective mobility values with:
- Dominant scattering mechanism identified
- Barrier height (if grain boundary dominated)
- Specularity parameter (if surface scattering dominated)
- Comparison to bulk single-crystal mobility

## Applications

- Field-effect transistor modeling
- Solar cell efficiency prediction
- Thin-film device optimization
- Polycrystalline material characterization