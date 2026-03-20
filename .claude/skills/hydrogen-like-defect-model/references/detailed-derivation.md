# Hydrogen-Like Defect Model: Detailed Derivation

## Physical Model

A substitutional donor (e.g., Phosphorus in Silicon) differs in valence from the host atom. The extra electron is donated to the host crystal, leaving behind a positively charged ion. The donated electron becomes a quasi-free Bloch electron near the conduction band edge but can localize near the positively charged defect due to Coulomb attraction.

## Hamiltonian Formulation

The total Hamiltonian:
```
H = H_0 + V
```

Where H_0 is the unperturbed host lattice Hamiltonian satisfying:
```
H_0 ψ_k = E(k) ψ_k
```

And V is the screened Coulomb potential:
```
V(r) = -e² / (4πε_0 ε_st r)
```

## Dielectric Screening Details

### Static Dielectric Constant (ε_st)
- Used for trapped electrons/holes
- The bound carrier causes a shift of surrounding ions according to its averaged Coulomb potential
- Ions have sufficient time to respond to the slowly varying bound state

### Optical Dielectric Constant (ε_opt)
- Used for Bloch electrons in an ideal lattice
- Interaction only with the electronic part of the dielectric response
- Ions cannot follow rapid electronic motion

## Wave Packet Construction

The solution near the defect is constructed from a wave packet of Bloch functions:
```
ψ(r) = Σ_k A(k) ψ_k(r)
```

For shallow levels:
- Summation over bands is dropped (mixing only with nearest band)
- Near band minima at k_0, expand E(k) to second order

## Envelope Function Approximation

Write the Bloch function as:
```
ψ_k(r) = u_k(r) exp(ik·r)
```

For k near k_0:
- u_k(r) ≈ u_{k_0}(r) (varies slowly)
- Pull u_{k_0}(r) out of the sum as constant

Define the envelope function:
```
F(r) = Σ_k A(k) exp(ik·r)
```

The envelope function satisfies:
```
[-(ℏ²/2m_n)∇² + V(r)] F(r) = E F(r)
```

This is the hydrogen-like Schrödinger equation with:
- Effective mass m_n
- Screened Coulomb potential V(r)

## Solutions

### Ground State (1s)
```
F(r) = (1/√(πa_qH³)) exp(-r/a_qH)
```

### Effective Bohr Radius
```
a_qH = ε_st × (m_0/m_n) × a_0
```

Typical values:
- For Si: ε_st ≈ 12, m_n ≈ 0.26m_0 → a_qH ≈ 24 Å
- For GaAs: ε_st ≈ 13, m_n ≈ 0.067m_0 → a_qH ≈ 100 Å

### Energy Spectrum

```
E_n = -Ry*/n²
```

Where:
```
Ry* = (m_n/(m_0 ε_st²)) × Ry
```

Typical values:
- For Si donors: Ry* ≈ 30 meV
- For GaAs donors: Ry* ≈ 6 meV

## Key Variables Reference

| Variable | Type | Description |
|----------|------|-------------|
| H_0 | Operator | Unperturbed Hamiltonian |
| V | Potential | Screened Coulomb potential |
| ε_st | Constant | Static dielectric constant |
| m_n | Mass | Effective mass of carrier |
| a_qH | Length | Effective Bohr radius |
| Ry* | Energy | Effective Rydberg energy |
| E_n | Energy | Bound state energy level |

## Equation Reference

Original equations: Eq. 9.1 - 9.10

## Domain Tags

- defect_physics
- shallow_impurities
- effective_mass_theory