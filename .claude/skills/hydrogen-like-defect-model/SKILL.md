---
name: hydrogen-like-defect-model
description: Calculate eigenfunctions and energy spectra for shallow impurity defects (donors/acceptors) in semiconductors using the hydrogen-like model with effective mass approximation. Use when modeling shallow level centers, phosphorus-in-silicon type donors, or acceptors where the defect wavefunction extends beyond nearest neighbors.
---

# Hydrogen-Like Defect Model

## When to Use

Apply this skill when:
- Modeling shallow impurities or defects in a semiconductor host
- Calculating bound state energies for donors (e.g., P in Si) or acceptors
- The defect eigenfunction is expected to extend beyond nearest neighbors
- Only mixing with the nearest band is relevant

## Prerequisites

Before executing, ensure you have:
- Effective mass value (m_n) for the relevant carrier
- Static dielectric constant (ε_st) of the host material
- Band edge energy (E_c for donors, E_v for acceptors)

## Procedure

### 1. Identify the Physical System

Determine the defect type:
- **Donor**: Substitutional atom that donates an electron (becomes positively charged)
- **Acceptor**: Substitutional atom that accepts an electron (becomes negatively charged)

The carrier (electron or hole) becomes a quasi-free Bloch particle near the band edge but can localize near the charged defect.

### 2. Set Up the Hamiltonian

The system is described by:
```
H = H_0 + V
```

Where:
- H_0 = Unperturbed host lattice Hamiltonian
- V = Attractive Coulomb potential of the defect (screened)

### 3. Apply Dielectric Screening

Select the appropriate dielectric constant:
- **Static dielectric constant (ε_st)**: For trapped electrons/holes (ions have time to respond)
- **Optical dielectric constant (ε_opt)**: For Bloch electrons in ideal lattice (electronic response only)

Use ε_st for bound state calculations.

### 4. Construct the Envelope Function

For shallow levels, construct the solution from a wave packet of Bloch functions from the nearest band only:
- Near band minima, the periodic part u(k) varies slowly
- Extract u(k) as constant
- Introduce envelope function F(r)

F(r) satisfies a modified Schrödinger equation for the quasi-hydrogen model.

### 5. Calculate Effective Bohr Radius

```
a_qH = ε_st × (m_0 / m_n) × a_0
```

Where a_0 = 0.529 Å (Bohr radius).

### 6. Calculate Ground State Wavefunction

The 1s ground state envelope function:
```
F(r) = (1 / √(π × a_qH³)) × exp(-r / a_qH)
```

### 7. Calculate Energy Spectrum

Effective Rydberg energy:
```
Ry* = (m_n / (m_0 × ε_st²)) × Ry
```

Where Ry = 13.6 eV (hydrogen Rydberg).

Bound state energy levels (measured from band edge):
```
E_n = -Ry* / n²
```

These are bound states below E_c (for donors) or above E_v (for acceptors).

## Output

The skill produces:
- Envelope function F(r) describing spatial localization
- Energy spectrum E_n for bound states
- Effective Bohr radius a_qH characterizing defect extent

## Constraints

- Applies only to shallow defects (wavefunction extends beyond nearest neighbors)
- Assumes mixing primarily with the nearest band only
- Requires validity of effective mass approximation