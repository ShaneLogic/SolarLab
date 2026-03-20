---
name: photon-assisted-tunneling-band-measurement
description: Measure band structure energies using photon-assisted tunneling and calculate rectangular barrier tunneling probabilities. Use when analyzing electron penetration through potential barriers, measuring characteristic band energies, or studying quantum tunneling effects in semiconductors.
---

# Photon-Assisted Tunneling and Barrier Tunneling

## When to Use
- Measuring band structure energies using optical excitation
- Calculating electron transmission probability through potential barriers
- Analyzing Franz-Keldysh effect for band energy measurement
- Studying quantum tunneling phenomena
- Determining barrier penetration characteristics

## Photon-Assisted Tunneling for Band Energy Measurement

Use this technique when measuring energy of characteristic points in the E(k) behavior of any band.

### Procedure
1. **Initiate Optical Excitation**
   - Optically excite an electron to a state close to ANY higher band
   - NOT restricted to valence-to-conduction band transitions near principal edge

2. **Complete Transition via Tunneling**
   - Complete the transition through the tunneling mechanism

3. **Measure Energy**
   - Utilize Franz-Keldysh effect as method for measuring energy
   - Measures characteristic points in E(k) behavior

### Application Scope
- Not restricted to valence-to-conduction band transitions near principal edge
- Applies to any band's E(k) behavior

## Rectangular Barrier Tunneling

Use when electron wave impinges on rectangular potential barrier.

### Prerequisites
- Quantum mechanics basics
- Understanding of wave function properties

### Barrier Definition
- Height: eV_0
- Width: a

### Three Regions
1. Before the barrier
2. Inside the barrier
3. After the barrier

### Wavenumber Definitions

**Outside barrier (k_0):**
```
k_0 = sqrt(2mE) / ħ
```

**Inside barrier (k_1):**
```
k_1 = sqrt(2m(eV_0 - E)) / ħ
```

### Transmission Probability (T_e)

**Exact Formula:**
```
T_e = 1 / [1 + ((k_0² + k_1²)² / (4k_0²k_1²)) × sinh²(k_1 × a)]
```

**Approximation for k_1 × a >> 1:**
```
T_e ≈ 16 × (k_0 × k_1 / (k_0² + k_1²))² × exp(-2 × k_1 × a)
```

**Approximation for eV_0 >> E:**
```
T_e ≈ 16 × (E / eV_0) × (1 - E / eV_0) × exp(-2 × k_1 × a)
```

### Example Calculation
- Thermal electrons (E = kT at 300K)
- Impinging on 10 Å thick barrier
- Barrier height: 1 Volt
- Result: Attenuated by factor of 2.2 × 10^(-3)

## Key Variables
| Variable | Description |
|----------|-------------|
| E(k) | Energy-momentum relationship of the band |
| eV_0 | Barrier height |
| a | Barrier width |
| E | Kinetic energy of tunneling electron |
| k_0 | Wavenumber outside barrier |
| k_1 | Wavenumber inside barrier |
| T_e | Transmission probability |