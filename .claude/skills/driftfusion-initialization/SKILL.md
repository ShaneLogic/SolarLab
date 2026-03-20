---
name: driftfusion-initialization
description: Initialize the Driftfusion simulation environment and create parameter objects. Use this skill when starting a new MATLAB session or setting up device properties for simulation.
---

# Driftfusion Initialization

Initialize the Driftfusion environment and create parameter objects for device simulation.

## When to Use
- Starting a new MATLAB session for Driftfusion simulations
- Loading device properties from CSV files
- Setting up the simulation workspace before running protocols

## System Initialization

1. Navigate to the Drift-fusion parent folder
2. Execute `initialise_df`:
   ```matlab
   initialise_df
   ```
3. The function performs:
   - Adds program folders to MATLAB path
   - Sets plotting defaults

**Critical Constraint**: Must run from parent folder (not subfolders) and BEFORE loading any saved data objects.

## Parameter Object Creation

1. Create parameters object:
   ```matlab
   par = pc('path/to/file.csv')
   ```
2. CSV import process:
   - Default values in `pc` class are overwritten by CSV values
   - `import_properties` reads present properties from file

3. Property definition rules:
   - Layer-specific properties must be cell or numerical arrays
   - Array size must equal number of layers INCLUDING interface layers
   - Example: 3 material layers + 2 heterojunctions = 5 element arrays

4. Error checking:
   - Number of rows in `layer_type` column validates all other entries
   - Ensures properties defined for each discrete layer (excluding electrode pseudo-layers)

5. Extensibility:
   - New properties must be added to both `pc` class and `import_properties` list

## Output
- Configured MATLAB workspace with proper paths
- Parameters object `par` populated with device-specific properties