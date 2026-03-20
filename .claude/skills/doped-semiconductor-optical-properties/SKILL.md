---
name: Doped Semiconductor Optical Properties
description: Analyze absorption coefficient and band edge effects in heavily doped semiconductors, including Burstein-Moss shift and band tailing. Use when designing optical filters, calculating absorption edges in doped materials, or analyzing how doping affects optical transitions and band structure.
---

# Doped Semiconductor Optical Properties

## When to Use
- Calculating absorption in heavily doped semiconductors
- Designing optical filters with precise cut-off wavelengths
- Analyzing Burstein-Moss shift in low-effective-mass materials
- Evaluating band tailing effects from impurity potentials
- Comparing n-type vs p-type optical behavior

## Two Key Effects

### 1. Burstein-Moss Effect (Band-Gap Widening)
Filling of conduction band states shifts absorption edge to higher energies.

### 2. Band Tailing (Band-Gap Shrinking)
Random impurity potential deforms band edges, creating tail states.

## Burstein-Moss Effect

### Conditions for Observation
- Low effective mass semiconductor
- Moderate to high donor doping (n-type)
- Low density of states at band edge

### Procedure

**Step 1: Check Material Properties**
- Verify low effective mass (e.g., GaAs, InSb)
- Confirm n-type doping

**Step 2: Calculate Fermi Level Shift**
With moderate doping, Fermi level shifts above conduction band edge:
```
EF - Ec ∝ (n/Nc)^(2/3)
```

**Step 3: Determine Absorption Edge Shift**
- Optical excitation requires empty states
- Filled states block low-energy transitions
- Absorption edge shifts to higher energy

**Step 4: Apply to Design**
- Fine-tune absorption edge for optical filters
- Example: HgTe with Al doping for precise long-wavelength cut-off

### Material Comparison

| Material Type | Dominant Effect | Reason |
|---------------|-----------------|--------|
| n-type GaAs | Burstein-Moss shift | Low electron mass |
| p-type GaAs | Band gap shrinking | Heavy hole mass |

## Absorption Coefficient in Heavily Doped Semiconductors

### Symmetry Breaking
- Translational symmetry broken by random impurity potential
- k is no longer a good quantum number
- Use energy E as label instead

### Absorption Formula (Abram et al. 1978)
```
α(E) = (Π × Nv(E) × Nc(E + hν)) / hν
```
Where:
- Π = Transition probability (matrix element sum)
- Nv(E) = Density of states near valence band edge
- Nc(E + hν) = Density of states near conduction band edge
- hν = Photon energy

### Procedure for Absorption Calculation

**Step 1: Account for Band Edge Deformation**
- Band edges deformed from ideal distribution
- Tail states extend into band gap

**Step 2: Apply Occupancy Factors**
```
fn(E) = Fermi distribution for electrons
fp(E + hν) = Fermi distribution for holes
```

**Step 3: Analyze Asymmetric Excitation**
For n-type material:
- Transitions from valence band tail
- To states above Fermi level in conduction band

## Design Applications

### Optical Filter Design
1. Select base material (e.g., HgTe)
2. Choose dopant type and concentration
3. Calculate Burstein-Moss shift
4. Fine-tune for desired cut-off wavelength

### Example: HgTe Optical Filter
- Base material: HgTe
- Dopant: Al (n-type)
- Result: Precise long-wavelength cut-off control