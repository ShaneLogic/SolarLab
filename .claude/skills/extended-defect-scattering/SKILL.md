---
name: extended-defect-scattering
description: Calculate mobility degradation and scattering effects in semiconductors with extended defects including dislocations and defect clusters. Use when analyzing carrier transport in deformed crystals, semiconductors with dislocations, or materials containing defect clusters and associates that interfere with carrier transport.
---

# Extended Defect Scattering Analysis

## When to Use
Apply this skill when:
- Analyzing semiconductor mobility in deformed crystals
- Calculating carrier transport in materials with dislocations
- Evaluating scattering from defect clusters or associates
- Determining directional mobility dependence due to aligned dislocations
- Assessing strain field effects on carrier mobility

## Prerequisites
- Dislocation density N (cm⁻²)
- Distance between charged defect centers along dislocation core (a_t)
- Dislocation orientation information (for anisotropy analysis)
- Temperature of operation (determines core charge state)
- Material parameters: permittivity (ε), carrier mass (m), carrier density

## Execution Workflow

### 1. Identify Scattering Mechanisms
Determine the dominant mechanisms present:
- **Core charge scattering**: From reconstructed core states attracting electrons/holes
- **Strain field scattering**: From stress field around dislocations
- **Piezoelectric scattering**: For piezoelectric crystals, from charges induced by strain field
- **Defect cluster scattering**: From larger defect associates or inclusions

### 2. Determine Core Charge State
- Core states attract electrons or holes based on Fermi level position
- In Ge and Si: positively charged at low temperatures, negatively charged at higher temperatures
- Find neutrality temperature where core is neutral (strain field effects persist)

### 3. Calculate Debye Screening Length
```
L_D = √(εε₀·kT / e²·p₀)
```
Where:
- ε = relative permittivity
- ε₀ = vacuum permittivity
- k = Boltzmann constant
- T = temperature
- e = elementary charge
- p₀ = hole density in bulk

### 4. Compute Dislocation Mobility (Podor 1966 Formula)
```
μ_disl = 75 × (ε₀/ε_s)^(3/2) × (e/N·a_t²) × (m_n/m₀) × (kT/e) / √n
```
Where:
- N = dislocation density (cm⁻²)
- a_t = distance between charged defect centers along dislocation core
- ε_s = semiconductor permittivity
- m_n = effective mass of carriers
- m₀ = free electron mass
- n = carrier density

### 5. Assess Anisotropy Effects
For materials with aligned dislocations:
- **Mobility parallel to dislocation array** ≈ mobility without dislocations
- **Mobility perpendicular to dislocation array** is substantially reduced
- This directional dependence is due to anisotropic deformation potential influence

### 6. Model Defect Cluster Scattering
Choose appropriate model based on cluster characteristics:
- **Small associates**: Model as neutral center with larger effective diameter
- **Charged clusters**: Model with space charge extending up to several Debye lengths
- **Different phase inclusions**: Account for electron affinity differences
- **Large clusters**: Range from point-defect-like scattering to carrier repulsion (cluster size comparable to or larger than mean free path)

### 7. Include Strain Field Effects
- At neutrality temperature, mobility is still lower than undeformed material
- Strain field scattering persists even without core charge
- Strain field influences band edge via deformation potential

## Output Interpretation
Results provide:
- Mobility degradation factor due to extended defects
- Directional mobility values (if anisotropy present)
- Relative contribution of each scattering mechanism
- Screening effects from surrounding carriers

## Key Constraints
- Core charge state depends on temperature (check neutrality temperature)
- Strain field effects persist at all temperatures
- Anisotropy requires knowledge of dislocation orientation
- Screening by free carriers reduces effective scattering cross-section
- Multiple scattering mechanisms may combine additively or dominantly