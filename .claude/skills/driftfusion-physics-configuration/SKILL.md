---
name: driftfusion-physics-configuration
description: Configure carrier statistics, generation models, and recombination mechanisms for Driftfusion simulations. Use this skill when defining the physical models governing carrier transport, generation, and recombination.
---

# Driftfusion Physics Configuration

Configure carrier statistics, optical generation, and recombination models for device simulation.

## When to Use
- Setting up carrier statistics (Boltzmann vs Blakemore)
- Defining optical generation profiles
- Configuring bulk and interface recombination models
- Analyzing carrier density profiles at interfaces

## Carrier Statistics Configuration

Set `distro_fun` for equilibrium boundary and initial carrier densities:

1. **'Boltz'**: Boltzmann approximation ($\gamma=0$)
   - Pros: Marginally faster calculations

2. **'Blakemore'**: Blakemore approximation ($\gamma>0$)
   - Pros: Extended domain of validity
   - **Recommendation**: Use Blakemore statistics

## Optical Generation Configuration

1. **Select Optical Model** (`optical_model`):
   - `'uniform'`: Uniform volumetric generation rate `g0`
   - `'Beer_Lambert'`: Exponential decay with absorption coefficient

2. **Generation Profiles**:
   - `gx1`: Bias light profile
   - `gx2`: Pulse light profile
   - Multiplied by intensities `int1` and `int2`

3. **External Profiles**:
   - Can overwrite `gx1` or `gx2` in `par` after creation
   - Must be interpolated to subinterval grid points ($x_{i+1/2}$)
   - **Recommendation**: Set generation rate to zero within interface regions

## Bulk Recombination Models

### Band-to-Band Recombination
```matlab
rbtb = B * (n * p - ni^2)
```
- `B`: Band-to-band recombination coefficient
- `ni^2`: Thermal generation term (ensures np ≥ ni^2)

### Shockley-Read-Hall (SRH) Recombination
```matlab
rSRH = (n * p - ni^2) / [τn,SRH * (p + pt) + τp,SRH * (n + nt)]
```
- `τn,SRH`, `τp,SRH`: SRH time constants
- `nt`, `pt`: Carrier densities at trap energy `Et`

**SRH Assumptions**:
- Trapped carriers in thermal equilibrium with bands
- Trapping/de-trapping rate fast compared to simulation timescale
- Trapped carriers negligible compared to free carriers

## Volumetric Surface Recombination (VSR)

1. **Convert Surface Flux to Volumetric Rate**:
   - Start with abrupt interface flux `Rint`
   - Distribute across zone of thickness `dvsr`
   - Base: `rvsr = Rint / dvsr`

2. **High Mobility Approximation**:
```matlab
rvsr ≈ (n * p - ni^2) / [τn,vsr * (p + pt) + τp,vsr * (n + nt)]
```
- `dvsr` subsumed into `τn,vsr` and `τp,vsr`

3. **VSR Mode Configuration** (`par.vsr_mode`):
   - `= 1`: `Et`, `ni`, `nt`, `pt` constant (calculated from interface energy levels)
   - `= 0`: Standard bulk SRH (graded `Et`, exponential `ni`, `nt`, `pt`)

4. **Self-Consistency Check**:
   - Run `compare_rec_flux` after solution
   - Compares interfacial recombination fluxes vs integrated VSR rate
   - Warning if difference > `par.RelTol_vsr` AND fluxes > `par.AbsTol_vsr`
   - **User Action**: Increase mobilities or reduce recombination coefficients

## Interface Carrier Density Profiles

For discrete interfaces:

1. **Translated Coordinates**: Define `xn`, `xp` in direction of carrier density decay

2. **Profile Analysis**:
   - Pure exponential `p(x)`: High hole mobility (negligible `jp,s` and `r`)
   - Curved `n(x)`: `jn,s` and `r` similar order to `ns` near `x1`

3. **Infinite Mobility Limit** ($\mu_{n,p} \to \infty$):
   - Equations converge to exponential forms
   - Carrier density change: $\Delta n = N_{CB} \cdot e^{\alpha \cdot d_{int}}$

## Output
- Configured carrier statistics
- Generation profiles `gx1`, `gx2`
- Recombination rates for bulk and interface regions
- Consistency warnings for VSR approximations