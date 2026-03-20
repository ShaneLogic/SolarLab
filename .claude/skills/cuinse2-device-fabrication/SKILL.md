---
name: cuinse2-device-fabrication
description: Apply validated deposition methods and layer stack construction techniques for CuInSe2/CIGS thin-film solar cell fabrication. Use when fabricating, designing, or troubleshooting CuInSe2-based photovoltaic devices, selecting between co-evaporation and precursor reaction methods, or optimizing device layer architecture.
---

# CuInSe2 Device Fabrication

## When to Use
- Fabricating CuInSe2 or Cu(InGa)Se2 (CIGS) thin-film solar cells
- Selecting deposition method for absorber layer formation
- Designing or optimizing device layer stack architecture
- Troubleshooting efficiency issues in chalcopyrite-based devices

## Primary Deposition Methods

### Method A: Boeing Co-evaporation
1. Set up separate evaporation sources for Cu, In, and Se elements
2. Co-evaporate elements onto heated substrate
3. Control deposition rates to achieve target composition
4. Forms CuInSe2 directly during deposition

### Method B: ARCO Solar Precursor Reaction
1. Deposit Cu/In precursor layers onto substrate
2. Perform reactive anneal in H2Se atmosphere
3. CuInSe2 forms through solid-state reaction

## Device Layer Stack Construction

Build the device from substrate upward:

### 1. Substrate Selection
- **Early designs**: Ceramic substrates
- **Modern standard**: Soda-lime glass (enables Na diffusion for efficiency gains)

### 2. Back Contact
- Deposit sputtered Molybdenum (Mo) layer
- Provides stable, conductive rear contact

### 3. Absorber Layer
- Deposit CuInSe2 or Cu(InGa)Se2 using Method A or B
- Ga incorporation enables bandgap tuning

### 4. Window Layer
- Deposit CdS buffer layer (optimized thickness: ~50nm)
- Follow with In-doped CdS as current carrier
- Alternative: High-resistance ZnO or ITO for Cd-free designs

### 5. Front Contact
- Deposit transparent conductive oxide: doped ZnO (preferred) or ITO

## Key Process Variables

| Variable | Type | Description |
|----------|------|-------------|
| cds_thickness | length | CdS layer thickness (evolved from 2μm to optimized 50nm) |
| anneal_gas | chemical | Reactive annealing gas (H2Se for precursor method) |

## Critical Optimization Notes

- CdS thickness reduction from 2μm to 50nm significantly improved current collection
- Soda-lime glass substrate provides beneficial Na incorporation
- Method selection depends on equipment availability and production scale