---
name: carrier-capture-cross-section-models
description: Calculate carrier capture cross-sections using gas-kinetic models and Coulomb-attractive center theory. Use this for determining recombination rates, carrier lifetimes, and analyzing temperature-dependent capture at defect centers.
---

# Carrier Capture Cross-Section Models

## When to Use
- Modeling carrier capture rates at defect centers
- Calculating carrier lifetimes from recombination center density
- Analyzing temperature-dependent capture processes
- Evaluating recombination at donors/acceptors (Coulomb-attractive centers)

## Gas-Kinetic Model

### Capture Cross-Section
```
s = π × r₀²
```
Where r₀ is the radius of the electron eigenstate at kT below the band edge.

### Capture Conditions
Carrier capture requires:
1. Carrier approaches within kT of band edge (usually satisfied in thermal equilibrium)
2. Carrier approaches within distance < r₀ of recombination center

### Mean Free Path Between Captures
```
λ = 1 / [(N_r - n_r) × s]
```
Where (N_r - n_r) is the density of free recombination centers.

### Carrier Lifetime
```
τ_r = λ / v_th
```
Where v_th is the thermal velocity.

### Limitations
- Assumes thermal equilibrium
- Low field conditions
- At higher fields or optical excitation, energy-dependent capture must be considered

## Coulomb-Attractive Centers

### Energy Spectrum
```
E_n = E_c - E_Ry / n_q²
```
Where:
- E_Ry: Rydberg energy of the defect
- n_q: principal quantum number

### Determine Relevant Quantum Number
Find n_kT: the smallest integer where E_n > kT relative to continuum.

### Eigenstate Radius
```
r₀ = 2 × n_kT² × a_B
```
Where a_B is the quasi-Bohr radius.

### Room Temperature Approximation
Since ground state is often slightly below kT:
```
r₀ ≈ 2 × a_B
```

### Capture Cross-Section Formula
```
s ≈ 4 × a_B² × (E_Ry / kT)²
```

### Key Characteristics
- **Mass independence:** Cross-section independent of effective mass m*
- **Temperature dependence:** s ∝ 1/T² (decreases with increasing temperature)
- **Low temperature enhancement:** Can achieve 'giant' cross-sections
  - Example: ~4×10⁻¹² cm² at 70K

## Temperature Dependence Comparison
| Model | Temperature Dependence | Physical Origin |
|-------|----------------------|-----------------|
| Gas-kinetic (neutral) | Weak (thermal velocity) | Geometric capture |
| Coulomb-attractive | s ∝ 1/T² | Coulomb focusing decreases with T |

## Practical Application

### Step-by-Step for Coulomb-Attractive Centers
1. Calculate quasi-Bohr radius a_B from defect parameters
2. Determine Rydberg energy E_Ry
3. At room temperature, use n_q = 1 approximation
4. Calculate cross-section using temperature-dependent formula
5. For low temperatures, compute actual n_kT for accuracy

### Lifetime Calculation
1. Determine capture cross-section s from appropriate model
2. Calculate mean free path λ from center density
3. Compute lifetime τ_r = λ/v_th
4. For multiple mechanisms, use Matthiessen's rule: 1/τ_total = Σ 1/τ_i