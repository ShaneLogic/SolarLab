---
name: impedance-spectroscopy-analysis
description: Perform virtual impedance spectroscopy simulations using IonMonger and interpret Nyquist plots for perovskite solar cells. Use when measuring device impedance, analyzing ionic vs electronic behavior, extracting recombination parameters, or validating simulation against experimental IS data.
---

# Impedance Spectroscopy Analysis

Use this skill when:
- Simulating impedance response of perovskite solar cells in IonMonger
- Interpreting experimental or simulated impedance spectra
- Extracting recombination characteristics (ideality factors, barriers)
- Analyzing ionic vs electronic contributions to device behavior
- Understanding frequency-dependent transport mechanisms

## Impedance Simulation Protocol

### 1. Configure Voltage Parameters

Define the simulation protocol in the parameters file using the `applied_voltage` input:

```matlab
applied_voltage = {
    'min_freq',       % Minimum frequency (Hz)
    'max_freq',       % Maximum frequency (Hz) 
    'DC_voltage',     % Steady-state bias voltage (V)
    'AC_amplitude',   % Perturbation amplitude (V)
    'num_freqs',      % Number of sample frequencies
    'num_periods'     % Periods to simulate per frequency
};
```

### 2. Run Equilibration Phase

- Allow the cell to equilibrate at the DC voltage
- IonMonger autonomously detects when steady-state is reached
- This steady-state solution serves as the initial condition for transient phase

### 3. Execute Transient Stimulation

- Iterate over all sample frequencies from min_freq to max_freq
- For each frequency, simulate response to AC voltage perturbation
- Store solution for each frequency in a structure array

**Optional Parallel Execution:**
- Requires MATLAB Parallel Computing Toolbox
- Significantly reduces compute time by distributing frequency iterations

### 4. Post-Process Impedance Data

After simulation completes:

1. **Extract current density data**
   - For each frequency, extract the final two periodic cycles of J(t)

2. **Fit sinusoidal response**
   - Use `FourierFit.m` to fit a sinusoid to the current density data
   - Extract amplitude and phase information

3. **Calculate impedance**
   - `Z(ω) = V_AC / I_AC(ω)`
   - Complex impedance as function of frequency

**Performance note:** Same solver as transient simulations (100 frequencies ≈ 48.7s on Ryzen 5 5600X)

## Interpret Nyquist Plots

### Standard Two-Feature Model

Most PSC impedance spectra show two primary features:

#### High-Frequency Feature
- **Location**: Rightmost semicircle (largest frequencies)
- **Excludes**: Transient effects of slow-moving ionic charge
- **Provides**: 
  - Sources of electron and hole recombination
  - Quantified via **electronic ideality factor** (Bennett et al.)
- **Physical meaning**: Electronic transport and recombination timescales

#### Low-Frequency Feature  
- **Location**: Leftmost semicircle (smallest frequencies)
- **Captures**: How ions impact cell response
- **Provides**:
  - Potential barrier to recombination (**ectypal factor**)
  - Density and mobility of ionic species
- **Physical meaning**: Ionic migration and screening effects

### Additional Features

Check for less common features that appear in specific parameter regimes:

- **Third semicircle**: Additional process or interface effect
- **Intermediate loop**: Arc between high- and low-frequency features
- **Note**: IonMonger reproduces these features for particular parameter sets

### Feature Interpretation Guidelines

| Feature | Frequency Range | Information | Key Parameters |
|---------|----------------|-------------|----------------|
| High-frequency | 10⁴-10⁶ Hz | Electronic recombination | Electronic ideality factor |
| Low-frequency | 10⁻²-10² Hz | Ionic migration, barriers | Ectypal factor, ion density |
| Intermediate | 10²-10⁴ Hz | Interface effects, traps | Additional capacitance/resistance |

## Output Data Structure

The post-processing produces:
- Complex impedance array `Z(f)` for all frequencies
- Nyquist plot data (real vs imaginary components)
- Bode plot data (magnitude and phase vs frequency)
- Extracted parameters (ideality factors, time constants)

Ready for visualization in Section 7.3 plotting routines.