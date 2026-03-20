---
name: small-sinusoidal-excitation-analysis
description: Analyze photoconductors and trap systems using small sinusoidal modulation to determine trap density, time constants, and carrier mobility. Use when characterizing trap levels via phase shift analysis, measuring time constant ratios from amplitude response, or determining carrier mobility through modulation spectroscopy techniques.
---

# Small Sinusoidal Excitation Analysis

This skill applies small sinusoidal modulation techniques to probe trap levels and carrier dynamics in photoconductors and reaction-kinetic systems.

## When to Use
- Determining trap density in photoconductors
- Calculating time constant ratios from modulation response
- Measuring carrier mobility when traps are saturated
- Characterizing trap levels via phase shift analysis

## Prerequisites
- Constant bias light source (g₀)
- Superimposed small modulated light signal (g₁)
- Small signal condition to allow linearization
- ωτ << 1 for time constant ratio calculations

## Setup Procedure

1. **Configure excitation source**
   - Establish constant bias light with generation rate g₀
   - Superimpose oscillating light signal with amplitude g₁ and angular frequency ω
   - Total generation rate: g(t) = g₀ + g₁ exp(iωt)

2. **Prepare measurement system**
   - Ensure bias light moves quasi-Fermi level to desired position
   - Verify modulated light signal is small enough for linear response
   - Set up detection for electron density modulation n(t)

## Analysis Workflow

### Determine Trap Density

1. Measure electron density modulation: n(t) = n₀ + n₁ exp(iωt)

2. Introduce n(t) into reaction-kinetic differential equation

3. Extract trap density from **phase shift** between excitation and response

4. Calculate coefficients:
   - A = n₁/g₁ = 1 / (1/τₙ + iω(1 + dnₜ/dn))
   - B = dnₜ/dn

### Determine Time Constants

1. Ensure condition ωτ << 1 is satisfied

2. Measure amplitude ratio of photocurrent modulation:
   - |j_ac| / |j_dc| = g₁ / g₀ × (τ / τₙ)

3. Calculate time constant ratio from amplitude response

### Determine Carrier Mobility (Special Case)

1. Verify one of these conditions:
   - Light intensity high enough to fill all traps (dnₜ/dn = 0)
   - Quasi-Fermi level falls between trap levels (dnₜ/dn << 1)

2. When condition met, τₙ = τ

3. Calculate mobility:
   - μₙ = j_ac / (2ω e g₀ n₀)

## Key Outputs
- **Trap density**: Extracted from phase shift between excitation and response
- **Time constant ratio**: Calculated from amplitude of photocurrent modulation
- **Carrier mobility**: Determined when traps are filled or quasi-Fermi level is between trap levels

## Constraints
- Small signal requirement: Modulation must be small enough to linearize the system response
- Frequency limitation: ωτ << 1 required for specific time constant calculations
- Trap saturation: Mobility measurement requires traps to be filled or bypassed