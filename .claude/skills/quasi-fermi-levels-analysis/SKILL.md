---
name: quasi-fermi-levels-analysis
description: Calculate quasi-Fermi levels and demarcation lines for semiconductors under external excitation (light or bias) in non-equilibrium steady-state conditions. Use this for analyzing carrier distributions, classifying traps vs recombination centers, and verifying recombination model validity in devices like solar cells or Schottky barriers.
---

# Quasi-Fermi Levels and Demarcation Lines Analysis

## When to Use
- Analyzing non-equilibrium carrier distribution under external excitation (light or bias)
- Classifying defect centers as traps or recombination centers
- Computing barrier/junction behavior using recombination models
- Analyzing efficiency of solar cells
- Verifying Shockley-Read-Hall model validity

## Core Concepts

### Quasi-Fermi Levels
Under external excitation in steady state, the electron distribution is approximated by a Fermi-type distribution with **TWO quasi-Fermi levels**:
- EFn: Quasi-Fermi level for electrons
- EFp: Quasi-Fermi level for holes

Under thermal equilibrium: EFn = EFp = EF

### Demarcation Lines
Separate traps from recombination centers based on transition rates:
- **Traps**: Predominant transitions communicate with SAME band
- **Recombination centers**: Predominant transitions communicate BETWEEN two bands

## Execution Procedure

### 1. Calculate Quasi-Fermi Levels

**Electrons (Eq 27.21):**
```
n = Nc/[1 + exp((Ec - EFn)/kT)]
```

**Holes (Eq 27.22):**
```
p = Nv/[1 + exp((EFp - Ev)/kT)]
```

**Behavior under excitation:**
- Majority carriers: Slight split from EF (small density increase)
- Minority carriers: Substantial change from EF (large density increase)

### 2. Calculate Demarcation Lines

**Electron demarcation line (Eq 27.25):**
```
Ec - EDn = Ev - EFp + δi
```
Where δi = kT ln(s_ni√(m*n*)/s_pi√(m*p*))

**Hole demarcation line (Eq 27.27):**
```
EFp - Ev = Ec - EFn + δj
```
Where δj = kT ln(s_pj√(m*p*)/s_nj√(m*n*))

### 3. Classify Centers

- **Electron traps**: Close to conduction band
- **Hole traps**: Close to valence band
- **Recombination centers**: Near middle of bandgap (between EDn and EDp)

### 4. Apply Critical Validity Check

**WARNING**: Do NOT approximate all centers with one effective recombination center:
- Demarcation lines lie well below quasi-Fermi levels and spread over wide energy range
- This energy range shifts with respect to band edges throughout space charge layers
- Different device regions involve recombination through different centers with different parameters

**Verification required:**
- Specific energy level
- Spectrum analysis
- Recombination cross-sections

## Constraint: Thermal Equilibrium

At thermal equilibrium:
- EFn = EFp = EF
- EDn = EDp = ED
- No recombination center range exists

**Mass action law (Eq 27.28):**
```
n₀p₀ = NvNc exp[-(Ec - Ev)/kT] = nᵢ²
```

## Parameter Ranges

- Capture cross-sections: 10⁻¹³ to 10⁻²² cm²
- Coulomb-repulsive centers: ≈ 10⁻²⁰ to 10⁻²² cm²
- Tightly bound centers: ≈ 10⁻¹⁸ cm² or below
- δi, δj variation: ~0.5 eV across different centers