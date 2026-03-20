# Band Inversion Physics Reference

## Physical Mechanism

### Field Quenching Effect
Field quenching occurs when a strong electric field causes a semiconductor layer to change its conductivity type. In CdS/CdTe systems:

1. **Normal State**: CdS is n-type, forming a standard heterojunction with p-type CdTe
2. **Quenched State**: Strong fields cause CdS to invert to p-type near the interface

### Band Structure Changes

#### Fermi Level Position
- **n-type CdS**: Fermi level near conduction band
- **p-type inverted CdS**: Fermi level shifts toward valence band
- **Constraint**: Fermi level remains constant (horizontal) in equilibrium

#### Band Bending Mechanics
When Fermi level shifts but must remain horizontal:
1. Valence band curves upward
2. Conduction band curves upward
3. Both bands follow the Fermi level constraint

### Band Disconnection Phenomenon

#### Forward Bias State
```
CdS (n-type)          CdTe (p-type)
    |                    |
CB ----+            +---- CB
       |            |
VB ----+            +---- VB
    |                    |
   Connected conduction bands
```

#### Quenched State
```
CdS (p-type)          CdTe (p-type)
    |                    |
CB ----+                 
       |            +---- CB
VB ----+            |
       |            +---- VB
    |                    |
   Disconnected conduction bands
```

### Benefits of Band Disconnection
1. **Limits electron back-diffusion**: Electrons cannot easily flow from CdTe CB to CdS CB
2. **Reduces leakage current**: The barrier prevents unwanted current paths
3. **Improves device performance**: Better collection efficiency

## Dipole Moment Considerations

### Interface Dipole
- Band offsets are determined by interface dipole moments
- Dipole moment can change with photoconductivity state
- Experimental evidence from Schottky barriers supports this mechanism

### Measurement Indicators
- Changes in open-circuit voltage
- Variations in fill factor
- Temperature-dependent current-voltage characteristics