# Quantum Mechanics of 2D Electron Systems

## Cyclotron Motion Equations

**Cyclotron frequency**:
ωc = eB/m*

**Cyclotron radius**:
r = mv/eB = √(2m*E)/eB

**Cyclotron energy**:
Ec = ħωc = ħeB/m*

## Landau Level Details

### Energy Levels

The quantized energy states are:

E(nq) = (nq + 1/2)ħωc

where:
- nq = 0, 1, 2, ... (Landau quantum number)
- ħωc is the cyclotron energy

### Degeneracy Calculation

Each Landau level can accommodate:

nL = eB/h electrons per unit area

**Physical interpretation**: This represents the highest packing of cyclotron orbits without overlap in the 2D plane for nq = 0.

### Electron Distribution

Given total electron density n in conduction band:

- Level 0: up to nL electrons
- Level 1: up to nL electrons
- Level nq: up to nL electrons

**Filling factor**:
ν = n / nL = nh / eB

When ν is an integer, all lower Landau levels are completely filled and the highest occupied level is exactly full.

## Conductivity Tensor

### Tensor Components

σ = [[σxx, σxy], [-σxy, σxx]]

### Relations

**Hall conductivity**:
σxy = n e / B (classical)
σxy = ν e²/h (quantized)

**Longitudinal conductivity**:
σxx = 0 when ν is integer (plateau condition)

### Resistance Tensor

R = [[ρxx, ρxy], [-ρxy, ρxx]]

where ρ = σ⁻¹

**Hall resistance on plateaus**:
Rxy = h/νe²

## Scattering Mechanisms

### Allowed Scattering

Scattering only occurs for electrons in the highest Landau level when:
1. The level is incompletely filled, OR
2. Thermal energy kT is not much less than ħωc

### Implications

- On quantized plateaus: σxx → 0 (no scattering)
- Between plateaus: σxx peaks (maximum scattering)

## Trochoid Motion

### Trajectory Description

In orthogonal E and B fields:
- Electric field Fx in x-direction
- Magnetic field Bz in z-direction

Electrons move in y-direction with constant drift velocity:
vd = Fx / B

The trajectory is a trochoid (flat spiral) combining:
- Circular cyclotron motion (radius r)
- Linear drift motion (velocity vd)

### Quantum Wells

### Formation

Quantum wells create 2D electron systems through:
1. Heterostructure interfaces (e.g., GaAs/AlGaAs)
2. Bandgap engineering creating potential wells
3. Quantum confinement in one dimension

### Thickness Requirement

Third dimension < 100Å ensures:
- Quantum confinement energy >> kT
- Discrete subbands in z-direction
- True 2D electron behavior

## Experimental Considerations

### Temperature Requirements

Typical experimental conditions:
- T < 4K (liquid helium temperatures)
- For high B fields: T can be higher but still ħωc >> kT

### Magnetic Field Range

- Typically 1-15 Tesla for conventional semiconductors
- Higher fields for smaller effective mass m*
- B field must be uniform over sample area

### Sample Quality

High mobility required for observing quantized Hall effect:
- μ > 10,000 cm²/V·s (typical)
- Low impurity concentration
- Smooth interfaces in heterostructures

## Equations Reference

### Key Equations

| Symbol | Equation | Description |
|--------|----------|-------------|
| ωc | eB/m* | Cyclotron frequency |
| nL | eB/h | Landau level degeneracy |
| E(nq) | (nq + 1/2)ħωc | Landau level energy |
| ν | nh/eB | Filling factor |
| σxy | νe²/h | Quantized Hall conductivity |
| Rxy | h/νe² | Quantized Hall resistance |

### Units

- n, nL: electrons per unit area (m⁻²)
- B: magnetic field (Tesla)
- e: elementary charge (1.602×10⁻¹⁹ C)
- h: Planck constant (6.626×10⁻³⁴ J·s)
- ħ: reduced Planck constant (h/2π)
- m*: effective mass (kg)
- σ: conductivity (Siemens)
- R: resistance (Ohms)