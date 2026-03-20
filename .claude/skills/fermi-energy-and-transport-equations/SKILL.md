---
name: fermi-energy-and-transport-equations
description: Apply Fermi energy concepts, quasi-Fermi levels, transport equations, thermoelectric power, and cyclotron resonance to analyze semiconductor transport. Use when deriving transport equations, calculating thermoelectric effects, measuring effective mass with magnetic fields, or analyzing systems at/near thermal equilibrium.
---

# Fermi Energy and Transport Equations

## When to Use
- Analyzing carrier transport equations in semiconductors
- Deriving transport coefficients and applying Onsager relations
- Calculating thermoelectric power for different semiconductor types
- Using magnetic fields to determine effective mass (cyclotron resonance)
- Distinguishing between equilibrium and non-equilibrium transport

## Fermi Energy and Quasi-Fermi Energy

### Systems in Thermal Equilibrium
- Use electrochemical energy EF (Fermi energy)
- EF includes both potential energy AND changes in carrier density
- Permits simplified expression in transport equations

### Systems Deviating from Thermal Equilibrium
- **MUST replace EF with quasi-Fermi energies**
- Use EFn for electrons (quasi-Fermi energy for electrons)
- Use EFp for holes (quasi-Fermi energy for holes)
- Selection depends on which carrier type is being analyzed

### Key Distinction
| Type | Energy Level | Valid Condition |
|------|--------------|-----------------|
| Fermi energy (EF) | Single energy level | Only at thermal equilibrium |
| Quasi-Fermi (EFn, EFp) | Separate energy levels | When system is perturbed from equilibrium |

## General Transport Equations Framework

### Governing Equations (Eq. 16.8 and 16.9)
- Carrier current j equation
- Energy (heat) current w equation
- Both contain density of states and distribution functions

### Conservation Laws
**Conservation of number of carriers (Eq. 16.10):**
- Governs carrier continuity

**Conservation of energy (Eq. 16.11):**
- ρ = density, u = specific internal energy

### Steady-State Solutions (Linear Combinations)

**Only electric fields (Eq. 16.12):**
- Simple relationship between current and field

**Electric AND thermal fields (Eq. 16.13):**
- Coupled equations with cross-terms

**With magnetic field (Eq. 16.14):**
- Full tensor form with magnetic field contributions

### Key Parameters
- φ = electrochemical potential (distinguished from EF = eφ, the electrochemical energy)
- For steady state: Replace φ with φn for electrons, φp for holes (Sect. 26.5.2)

### Transport Coefficients
- α_ik, β_ik', and γ_ik are well-known transport coefficients
- α_11 = σ_c (electrical conductivity)
- α_22 = K_c (thermal conductivity, subscript c = n or p)
- Total thermal conductivity: κ = (α_11α_22 − α_12α_21)/α_11 (Beer 1963)

### Onsager Relations (Eq. 16.15)
- Connect different transport coefficients
- Obtained from reciprocity of effects
- α_ki is the transposed tensor of α_ik

### Tensor Properties
- In anisotropic semiconductors, each transport parameter is a TENSOR
- Example: σ_n = σ_ink = enμ_ik
- Magnitudes depend on relative orientation of different fields
- For anisotropic semiconductors: Also depends on crystallographic orientation

## Thermoelectric Power

### Key Difference from Metals
- Thermoelectric power in semiconductors is USUALLY MUCH LARGER than in metals

### n-type Semiconductors (Eq. 16.20, upper)
```
α = (k/e)[(r + 5/2) − ln(n/N_c)]
```

### p-type Semiconductors (Eq. 16.20, lower)
```
α = (k/e)[(r + 5/2) − ln(p/N_v)]
```

### Scattering Mechanism Parameter r
| r value | Mechanism |
|---------|-----------|
| 1 | Amorphous semiconductors (Fritzsche 1979) |
| 2 | Acoustic phonon scattering |
| 3 | (Polar) optical phonon scattering |
| 4 | Ionized impurity scattering |
| 2.5 | Neutral impurity scattering |

### Ambipolar Semiconductor (Eq. 16.21)
- When both electrons and holes contribute significantly
- More complex expression involving both carrier types
- Reference: Smith (1952) and Tauc (1954)

### Calculation Steps
1. Identify semiconductor type (n-type, p-type, or ambipolar)
2. Determine dominant scattering mechanism
3. Select appropriate r value
4. Apply corresponding formula with carrier concentration and effective density of states
5. Result in units consistent with k/e ≈ 86 μV/K

## Cyclotron Resonance

### Condition
- When magnetic field is strong enough
- Mean free path is long enough for carriers to complete cyclic paths
- Strong resonances observed at cyclotron frequency

### Cyclotron Frequency Formula
```
ω_c = eB/m_n
```
- e = electron charge
- B = magnetic induction
- m_n = effective mass

### Cyclotron-Resonance Linewidth
- Decreases rapidly as more cycles are completed before scattering
- Scattering with relaxation time τ_m acts as damping parameter
- τ_m = 1/γ determines resulting lineshape

### Applications
- Well suited for determining effective mass in different crystallographic directions
- For derivation of resonance conditions, see McKelvey (1966)

### Alternative Expression (Bohr Magneton)
- Free electrons in vacuo: ω_c = eB/m_0
- Using Bohr magneton μ_B = eh/(2m_0): ω_c = 2μ_BB/h

## Key Variables
| Variable | Description |
|----------|-------------|
| EF | Fermi energy (electrochemical energy at equilibrium) |
| EFn | Quasi-Fermi energy for electrons |
| EFp | Quasi-Fermi energy for holes |
| j | Carrier current density |
| w | Energy (heat) current |
| φ | Electrochemical potential |
| α_ik | Transport coefficient tensor |
| α | Thermoelectric power (Seebeck coefficient) |
| r | Scattering mechanism parameter |
| ω_c | Cyclotron frequency |