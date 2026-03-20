# IonMonger Feature Comparison: Lite vs Full

| Feature | Lite | Full |
|---------|------|------|
| **Interface** | Checkbox GUI | Code editing (master.m) |
| Basic Protocols | ✓ | ✓ |
| Advanced Protocols | ✗ | ✓ |
| OCV Tracking | ✗ | ✓ |
| Non-Boltzmann Stats (TL) | ✗ | ✓ |
| Batch Simulations | ✗ | ✓ |
| Solver Resolution | Fixed | Adjustable |
| Band Offsets | ✗ | ✓ |
| Parasitic Resistances | ✗ | ✓ |
| Resume Saved Sims | ✗ | ✓ |
| Animation | ✓ | ✓ |
| Basic Plots | ✓ | ✓ |
| 3D Impedance Plots | ✓ | ✓ |
| **Solver** | ode15s | ode15s (same) |

## Default Parameters for Missing Fields (Table 1)

When loading v1.0 parameter files into v2.0, new model parameters default to:

*Note: Refer to original Table 1 in documentation for specific default values.*

## Python Import Script Details

**File:** IonMonger_import.py

**Workflow:**
1. Reads .mat file using scipy.io.loadmat
2. Unpacks MATLAB struct to Python dict
3. Extracts major simulation variables
4. Provides example plotting code

**Target Audience:** Python users without MATLAB expertise

## Plotting Tool Requirements

**MATLAB Required for:**
- All plotrecombination.m visualizations
- All plotdstrbns.m visualizations
- All plotIS.m visualizations  
- All plotbands.m visualizations
- All animatesections.m video generation

**Animation Parameters (GUIDE.md):**
- Length: Animation duration
- Resolution: Spatial resolution
- Frame rate: Playback speed