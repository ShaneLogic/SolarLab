---
name: CIGS-CdS Heterojunction Formation and Analysis
description: Form and analyze Cu(InGa)Se2/CdS heterojunctions for thin-film solar cells. Use when designing junction structures, depositing CdS buffer layers, evaluating band alignment effects on carrier collection, or troubleshooting low Jsc/FF in CIGS devices.
---

# CIGS-CdS Heterojunction Formation and Analysis

## When to Use This Skill

Use this skill when:
- Designing or optimizing Cu(InGa)Se2/CdS heterojunction structures
- Selecting CdS deposition parameters for buffer layers
- Analyzing band alignment and interface transport properties
- Troubleshooting low Jsc or FF in CIGS solar cells
- Evaluating lattice mismatch effects with varying Ga content

## Junction Formation Process

### Step 1: CdS Buffer Layer Deposition

Use chemical bath deposition (CBD) for optimal results:
- Target thickness: **~50 nm** for undoped CdS layer
- Method: Ion-by-ion growth process
- Expected result: Dense, pinhole-free films with grain size of tens of nanometers
- Crystal structure: Mixed cubic/hexagonal or predominantly hexagonal

### Step 2: Device Structure Configuration

Implement the optimized structure:
```
p-Cu(InGa)Se2 / undoped CdS (~50nm) / doped ZnO
```

### Step 3: Interface Formation Assessment

The interface exhibits pseudo-epitaxial characteristics:
- Epitaxial relationship: (112) chalcopyrite || (111) cubic or (002) hexagonal CdS
- Expect some intermixing of elements at the interface

## Band Alignment Analysis

### Conduction Band Offset (ΔEc - Spike)

Evaluate the conduction band spike:
- **Critical threshold**: ΔEc > 0.5 eV impedes electron collection
- **Favorable condition**: ΔEc < 0.5 eV allows thermionic emission transport
- **Impact**: Large spikes sharply reduce Jsc and FF

### Valence Band Offset (ΔEv)

- Value: **Approximately -0.9 eV**
- Consistent across polycrystalline, epitaxial, and single-crystal films
- Independent of surface orientation or CdS deposition method

### ZnO/CdS Interface

- Assumed conduction-band offset: **-0.3 eV**

## Lattice Matching Considerations

Adjust expectations based on Ga content:

| Material | (112) Spacing (nm) |
|----------|-------------------|
| Pure CuInSe2 | 0.334 |
| CuIn0.7Ga0.3Se2 | 0.331 |
| CuIn0.5Ga0.5Se2 | 0.328 |
| CdS (111) cubic / (002) hex | 0.336 |

**Key insight**: Lattice mismatch increases with Ga content, affecting interface quality.

## Decision Criteria

| Condition | Action |
|-----------|--------|
| Optimizing optical transmission | Reduce undoped CdS to ~50 nm |
| Troubleshooting poor Jsc/FF | Check if ΔEc exceeds 0.5 eV threshold |
| High Ga content (>30%) | Account for increased lattice mismatch |

## Common Issues and Solutions

| Issue | Likely Cause | Solution |
|-------|--------------|----------|
| Low Jsc and FF | ΔEc spike > 0.5 eV | Review band alignment, adjust interface treatment |
| Poor interface quality | High Ga content | Consider alternative buffer layers or interface engineering |
| Optical losses | CdS too thick | Reduce to ~50 nm |
| Pinholes in CdS | Non-optimal CBD | Optimize bath chemistry and deposition time |