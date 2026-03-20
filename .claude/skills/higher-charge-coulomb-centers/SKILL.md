---
name: higher-charge-coulomb-centers
description: Calculate ionization energies and analyze binding properties of multi-level impurity centers (double donors, double acceptors) with higher Coulomb charge. Use when working with substitutional impurities that donate or accept more than one carrier, such as Group VI elements in silicon or germanium, where hydrogen-like approximation becomes less accurate due to tight binding.
---

# Higher Charged Coulomb-Attractive Centers

## When to Use This Skill
Use this skill when:
- Analyzing substitutional impurities with valence difference greater than 1 relative to host
- Working with double donors (e.g., S, Se, Te in Si) or double acceptors
- Calculating ionization energies for multi-level centers
- Evaluating whether hydrogen-like approximations remain valid
- Determining which energy levels are active based on occupancy

## Core Workflow

### 1. Identify Multi-Level Centers

Identify substitutional impurities that donate or accept more than one carrier:
- Double donors: Group VI elements in Group IV hosts (S, Se, Te in Si or Ge)
- Double acceptors: Group II elements in Group IV hosts (Zn, Cd, Hg in Si or Ge)
- Verify that valence difference from host is > 1

### 2. Assess Binding Characteristics

Recognize that higher charge centers exhibit:
- Eigenfunctions closer to the impurity core
- Stronger binding requiring core corrections (similar to deep level centers)
- Reduced accuracy of simple hydrogen-like models

### 3. Calculate Binding Energy Scaling

For a center with charge Z (number of electrons/holes bound):
- First electron binding energy scales as: E ∝ Z²
- For Z = 2 (double donor/acceptor): first electron binds at 4× depth of standard donor
- Apply central cell corrections due to tight binding

### 4. Determine Active Energy Levels

Analyze based on occupancy state:

**For Double Donors (Z = +2):**
- Neutral state (two electrons): Only the shallow level is active
- Singly ionized (one electron): Remaining electron is more strongly bound
- Doubly ionized (no electrons): Deep level active

**For Double Acceptors (Z = -2):**
- Apply analogous logic with hole occupancy

### 5. Apply Core Corrections When Needed

When energies exceed ~100 meV (indicating tight binding):
- Include central cell potential effects
- Recognize quasi-hydrogen estimates may be inaccurate
- Consider measured experimental values over theoretical approximations

## Key Indicators

- **Quasi-hydrogen estimate deviation**: Actual energies often 3-5× higher than simple estimates
- **Central cell effects**: Significant when E > 100 meV in Ge, E > 150 meV in Si
- **Tight binding regime**: Requires treatment beyond effective mass approximation