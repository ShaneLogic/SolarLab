---
name: ionmonger-software-operations
description: Operate IonMonger simulation software including version selection (Lite vs Full), data import/export, compatibility handling, and visualization tools. Use when running PSC drift-diffusion simulations, managing parameter files, or analyzing simulation results.
---

# IonMonger Software Operations

## When to Use
- Running PSC drift-diffusion simulations with IonMonger
- Selecting appropriate software version (Lite vs Full) based on simulation needs
- Importing simulation data into Python for analysis
- Loading legacy parameter files into newer IonMonger versions
- Visualizing simulation results with plotting tools
- Ensuring backward compatibility across software versions

## Version Selection: Lite vs Full

### IonMonger Lite Mode

**Interface:** Checkbox-based GUI for plot selection

**Supported Features:**
- Applied voltage/time plots
- Light intensity/time plots
- Current density/time or voltage plots
- Spatial distributions (anion vacancy, electron, hole, electric potential)
- Nyquist/frequency plots
- Animation generation (via animate_sections.m)

**Limitations:**
- No advanced protocols
- No OCV tracking
- No non-Boltzmann statistics in transport layers
- No batch simulations
- No solver resolution adjustments
- No band offset control at metal contacts
- No parasitic resistances
- Cannot resume saved simulations

### Full Version Mode

**Interface:** Requires editing parameters file and running `master.m`

**Exclusive Features:**
- Advanced protocols (asymmetric sweeps, multiple consecutive sweeps, time-dependent illumination)
- Open-circuit voltage tracking
- Non-Boltzmann statistical models in transport layers
- Batch simulations (iterating variables)
- Adjustable solver resolution and error tolerances
- Band offsets at metal contacts
- Parasitic resistances
- Resuming saved simulations

**Performance:** Uses same numerical solver as Lite (identical performance/accuracy)

## Python Data Import Workflow

**Prerequisites:**
- Completed simulation
- Python installation
- IonMonger_import.py script

**Procedure:**
1. Locate saved solution file (.mat) containing all inputs and data
2. Execute `IonMonger_import.py` script
3. Script unpacks data and extracts all major variables
4. Use example code within script to generate plots or perform analysis

**Note:** Python has no direct equivalent to MATLAB struct data types; script handles conversion automatically.

## Backward Compatibility

**Loading Legacy Parameter Files:**
- v2.0 accepts parameter files from previous versions
- Missing parameters for new model extensions are replaced by default values (see Table 1)
- Automatic defaulting ensures seamless migration

**Verification:**
- Electric potential distributions from v1.0 and v2.0 overlap exactly on high-res grids
- All output variables verified across multiple parameter sets

**Testing Framework:**
- Test 1 (Regression): Compares output with saved data from original code
- Test 2 (Integration): Checks successful execution with different protocols

## Analysis and Visualization Tools

**Location:** Code\Plotting folder

**Available Tools:**

1. `plotrecombination.m`:
   - Generates two figures
   - Shows recombination rates as functions of time and space

2. `plotdstrbns.m`:
   - Plots spatial distributions of four model variables
   - Variables: P (anion vacancy), φ (potential), n (electron), p (hole)

3. `plotIS.m`:
   - Generates Nyquist and frequency plots (Fig 4)
   - Creates 3D plot of impedance spectrum (Fig 3)

4. `plotbands.m`:
   - Generates band diagram
   - Includes reference energies and Quasi-Fermi Levels (QFLs)

5. `animatesections.m`:
   - Animates solution
   - Saves as MP4
   - Parameters (length, resolution, frame rate) specified in GUIDE.md