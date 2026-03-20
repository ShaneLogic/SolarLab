# Impedance Simulation Parameters

## Voltage Protocol Parameters

| Parameter | Symbol | Typical Range | Unit |
|-----------|--------|---------------|------|
| Minimum frequency | f_min | 0.01 - 1 | Hz |
| Maximum frequency | f_max | 10⁵ - 10⁷ | Hz |
| DC voltage | V_DC | -1 - 1.5 | V |
| AC amplitude | V_AC | 0.01 - 0.1 | V |
| Number of frequencies | N_freq | 20 - 200 | - |
| Number of periods | N_per | 2 - 10 | - |

## Selection Guidelines

**Frequency Range:**
- Use logarithmic spacing to capture dynamics
- Ensure Nyquist plots close (Z' → 0 as f → ∞)
- Low frequencies may require long simulation times

**AC Amplitude:**
- Small enough for linear response (typically < 20 mV)
- Large enough to overcome numerical noise

**Number of Periods:**
- More periods = better signal-to-noise ratio
- Trade-off with total simulation time

## Extracted Parameters

**Electronic Ideality Factor:**
- Derived from high-frequency feature radius
- Values: 1 (ideal) to >2 (trap-assisted)
- Relates to recombination mechanism

**Ectypal Factor:**
- Derived from low-frequency feature
- Characterizes ionic screening strength
- Relates to ion density and mobility

**Time Constants:**
- τ_high = R_high * C_high (electronic)
- τ_low = R_low * C_low (ionic)
- Ratio τ_low/τ_high indicates relative timescales