---
name: Fermi Energy and Electron Density Calculation
description: Calculate electron density, Fermi potential, and effective density of states in semiconductors using Boltzmann approximation. Use when analyzing space-charge regions, determining carrier concentrations, or relating Fermi level position to band structure in thermal equilibrium.
---

# Fermi Energy and Electron Density Calculation

## When to Use
- Calculating electron density in semiconductors
- Determining Fermi potential in space-charge regions
- Analyzing band structure relationships
- Working with thermal equilibrium conditions

## Prerequisites
- Thermal equilibrium conditions
- Boltzmann approximation validity: (Ec - Ev) > 3kT
- Known effective mass values

## Core Formulas

### 1. Electrochemical Potential (Fermi Potential)
```
ψn = ψ - (kT/e) × ln(Nc/n)
```
Where:
- ψn = Electrochemical potential (Fermi potential)
- ψ = Electrostatic potential
- Nc = Effective density of states
- n = Electron density

### 2. Effective Density of States
```
Nc = 2 × (2π × mn × kT / h²)^(3/2)
```
Where:
- mn = Effective mass of electrons at conduction band edge
- k = Boltzmann constant
- T = Temperature
- h = Planck's constant

### 3. Electron Density (Boltzmann Approximation)
```
n = Nc × exp[-(Ec - EF) / (kT)]
```
Where:
- Ec = Energy at conduction band edge
- EF = Fermi energy

## Procedure

### Step 1: Verify Validity Conditions
- Confirm thermal equilibrium
- Check that (Ec - Ev) > 3kT for Boltzmann approximation

### Step 2: Calculate Effective Density of States
- Determine effective mass mn for the material
- Apply Nc formula with temperature

### Step 3: Determine Electron Density
- Use Fermi level position relative to conduction band
- Apply Boltzmann approximation formula

### Step 4: Calculate Fermi Potential
- Relate electrostatic and electrochemical potentials
- Account for carrier density variations

## Important Notes
- Equation (25.26) holds throughout entire crystal in thermal equilibrium
- Includes space-charge regions
- Boltzmann approximation breaks down in degenerate semiconductors