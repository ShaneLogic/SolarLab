# Junction Types and Structures Reference

## 1. Homojunctions

### pn-Junction
- Standard diode structure with p-type and n-type regions of same semiconductor
- Built-in potential: Vbi = (kT/q) ln(NA·ND/ni²)
- Depletion width: W = √[(2ε(Vbi-V)(NA+ND))/(q·NA·ND)]

### pin Structure
- Intrinsic (undoped) layer between p and n regions
- Extended depletion region for improved carrier collection
- Applications: Photodiodes (pin photodiode), solar cells (pin cell)

## 2. Heterojunctions

### Material Interface Types
- **Hetero-boundaries**: Interface between two different semiconductors
- **Frontwall solar cell**: Heterojunction with illumination through wider bandgap material
- **Backwall solar cell**: Heterojunction with illumination through narrower bandgap material

### Band Alignment
- **Valence band jump**: Discontinuity in valence band edge at interface
- **Conduction band offset**: Discontinuity in conduction band edge
- Total bandgap difference: ΔEg = ΔEv + ΔEc

## 3. Metal-Semiconductor Junctions

### Schottky Barrier
- Rectifying contact between metal and semiconductor
- Barrier height: φB = φM - χS (ideal case)
  - φM: Metal work function
  - χS: Semiconductor electron affinity

### Modified Schottky Barrier
- Accounts for interface states and Fermi-level pinning
- Modified Schottky diode equation includes ideality factor

### Large Area Metal Contacts
- Used for device terminals
- May be rectifying or ohmic depending on barrier height

## 4. Junction Physics Equations

### Poisson Equation
∇²ψ = -ρ/ε

Where:
- ψ: Electrostatic potential
- ρ: Space charge density
- ε: Permittivity

Applications:
- Depletion region analysis
- Potential distribution calculation
- Field determination in junctions

### Space-Charge Concepts

**Space Charge Regions**:
- Ionized dopants create fixed charge
- Depletion approximation: ρ ≈ q(ND - NA) in ionized regions

**Space-Charge-Limited Current**:
- Current limited by injected charge carriers
- SCLC equation: J = (9/8) εμV²/L³
- Important in insulators and low-doped regions

### Junction Capacitance

**Depletion Capacitance**:
Cj = εA/W

Voltage dependence:
Cj ∝ (Vbi - V)^(-1/2) for abrupt junction

### Rectifying Characteristic

Forward bias: Exponential current increase
I = Is[exp(qV/nkT) - 1]

Reverse bias: Saturation current Is
- Limited by minority carrier generation
- Breakdown at high reverse voltage

## 5. Key Parameters Summary

| Parameter | Symbol | Typical Units |
|-----------|--------|---------------|
| Built-in potential | Vbi | V |
| Barrier height | φB | eV |
| Depletion width | W | μm |
| Doping concentration | N | cm⁻³ |
| Capacitance | C | F/cm² |
| Ideality factor | n | dimensionless |