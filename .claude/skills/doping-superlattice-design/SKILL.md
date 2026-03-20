---
name: doping-superlattice-design
description: Design and configure doping superlattices (n-i-p-i structures) using periodic doping variations in semiconductor materials. Use when constructing superlattices through doping modulation rather than composition changes, particularly when determining lattice constants and miniband characteristics for GaAs, Si, InP, or similar semiconductors.
---

# Doping Superlattice Design

## When to Use This Skill

Use this skill when:
- Constructing superlattices using periodic doping variations rather than composition changes
- Designing n-i-p-i structures in semiconductor materials
- Determining appropriate lattice constants for doping superlattices
- Analyzing miniband characteristics in doping-modulated structures

## Prerequisites

- Host semiconductor material (e.g., GaAs, Si, InP)
- Knowledge of the material's Debye length for space charge calculations

## Design Procedure

### 1. Define Structure Configuration

Create a periodic modulation of bands:

1. Design alternating n-type and p-type doping layers
2. Optionally insert i-layers (undoped or compensated intrinsic layers) between n and p layers
3. This creates an n-i-p-i structure with improved charge carrier separation

### 2. Determine Miniband Characteristics

The structure will exhibit minibands similar to compositional superlattices:

- Minibands become wider with:
  - Shorter superlattice constants (smaller d)
  - Lower potential barriers
- Miniband width affects carrier transport and optical properties

### 3. Calculate Spatial Scale

Determine the superlattice period (d) based on the Debye length:

1. The lattice constant depends on the Debye length for space charge variation
2. Typical lattice constant range: **300 to 3,000 Å**
3. This is significantly larger than compositional superlattice constants

### 4. Select Appropriate Materials

Doping superlattices can be produced in various semiconductors:

- GaAs (Gallium Arsenide)
- Si (Silicon)
- InP (Indium Phosphide)
- PbTe (Lead Telluride)
- Other suitable semiconductor materials

## Design Considerations

- Lattice constants are determined by Debye length, not by crystal lattice matching
- The periodic potential modulation creates the superlattice effects
- n-i-p-i configuration provides better separation of charge carriers

## Expected Output

A superlattice structure with:
- Period (d) in the range of 300-3,000 Å
- n-i-p-i configuration (when i-layers are included)
- Defined miniband characteristics based on the chosen period and doping levels