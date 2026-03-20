---
name: driftfusion-device-structure
description: Define device layer structure, configure spatial and time meshes, and build device structures with interface grading. Use this skill when setting up the physical geometry and discretization of a simulation device.
---

# Driftfusion Device Structure Configuration

Define layer types, configure meshes, and build device structures with graded interfaces.

## When to Use
- Specifying device geometry and layer composition
- Setting up spatial discretization for the solver
- Configuring interface properties between material layers
- Defining time mesh for simulation protocols

## Layer Type Definition

Define `layer_type` property for each layer. Four types supported:

1. **'electrode'**: Pseudo-layer defining system boundaries (not discrete, not visualized)
2. **'layer'**: Slab of semiconductor with spatially constant properties
3. **'active'**: Layer flagged as device active layer (stores index in `active_layer` property)
4. **'interface'**: Interfacial region between different material layers

**Critical Rule**: Interface layers MUST be included between materials with different energy levels and eDOS values (at heterojunctions).

## Spatial Mesh Configuration

Call `meshgen_x` to generate spatial mesh:

1. **Grid Structure**:
   - N intervals (integer points `xi`)
   - N-1 subintervals (`x_sub`) where $x_{i+1/2} = (x_{i+1} + x_i) / 2$
   - Solver calculates variables on subintervals, fluxes on integer intervals

2. **Mesh Types** (`xmesh_type`):
   - `'linear'`: Linear piece-wise spacing
   - `'erf-linear'`: Error function (bulk) + linear (interface) spacing

3. **Parameters**:
   - `layer_points`: Points per layer (constraint: ≥ 3)
   - `xmesh_coeff`: Controls boundary point density (constraint: > 0)

4. **Recommendations**:
   - Use `'erf-linear'` for high ionic defect densities
   - Reduce `xmesh_coeff` if ionic depletion extends into bulk

## Time Mesh Configuration

1. **Solver Behavior**: `pdepe` uses adaptive time stepping, interpolates to user-defined mesh

2. **Convergence Factors**:
   - Weakly dependent on mesh spacing
   - Strongly dependent on `tmax` and `MaxStepFactor`

3. **Mesh Types** (`tmesh_type`): Linear, logarithmic, etc.

4. **Usage Guidelines**:
   - Change frequently for intermediate solutions with different timescales
   - Use linear for J-V scans (matches voltage change)
   - Use logarithmic for large time derivatives

## Device Structure Building

1. **Structure Creation**:
   - `build_device` and `build_property` create device structures
   - `dev`: Integer intervals ($x_i$), used for initial conditions
   - `dev_sub`: Subintervals ($x_{i+1/2}$), used by solver

2. **Interface Grading Options** (set in `build_device`):
   - `'zeroed'`: Property set to zero throughout interface
   - `'constant'`: Property set to user-defined constant (requires CSV value)
   - `'lin_graded'`: Linear grading using adjoining layer values (no interface value needed)
   - `'exp_graded'`: Exponential grading using adjoining layer values

3. **Input Requirements**:
   - `'lin_graded'`: Interface value NOT needed
   - `'constant'`: Interface value IS needed (must be in CSV)
   - Recommendation: Specify all values for all layers to future-proof

## Output
- Spatial grids `x` and `x_sub`
- Time mesh defined for solver
- Populated `dev` and `dev_sub` structures with graded interface properties