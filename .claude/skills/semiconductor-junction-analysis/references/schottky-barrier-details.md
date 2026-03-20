# Schottky Barrier Device Physics

## Schottky Approximation

The Schottky approximation simplifies the analysis of metal-semiconductor junctions:

### Assumptions
1. Depletion approximation valid
2. No free carriers in depletion region
3. Sharp boundary at depletion edge

### Potential Distribution

In the depletion region:
ψ(x) = (qN/2ε)(W - x)²

Where:
- W: Depletion width
- N: Doping concentration
- x: Distance from metal interface

### Electric Field

Maximum field at interface:
Emax = qNW/ε = √(2qNφB/ε)

## Modified Schottky Diode Equation

Accounting for non-ideal behavior:

I = AA*T² exp(-qφB/kT) [exp(qV/nkT) - 1]

Where:
- A: Junction area
- A*: Effective Richardson constant
- T: Temperature
- φB: Barrier height
- n: Ideality factor (typically 1.0-1.5)

## Schottky Barrier Height Determination

### Methods

1. **I-V Measurement**:
   φB = (kT/q) ln(AA*T²/Is)

2. **C-V Measurement**:
   Plot 1/C² vs V
   Extrapolate to Vbi
   φB = Vbi + (kT/q) ln(NC/N)

3. **Photoresponse**:
   Threshold energy = barrier height

## Device Applications

### Schottky-Barrier Devices
- Fast switching diodes
- Microwave detectors
- Solar cells (MS structures)
- Field-effect transistors (MESFETs)

### Contact Engineering
- **Rectifying contacts**: High barrier height
- **Ohmic contacts**: Low barrier or tunneling
- **Modified barriers**: Interface engineering for specific characteristics