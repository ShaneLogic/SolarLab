# Graded Band-Gap Theory

## Band Gap Engineering

### Composition Dependence
For alloy Ax B_{1-x}:
```
Eg(x) = x × Eg(A) + (1-x) × Eg(B) - b × x(1-x)
```
Where b is the bowing parameter.

### Common Graded Systems

| System | ΔEg (eV) | Applications |
|--------|----------|--------------|
| AlGaAs | 0.35 | HBTs, lasers |
| InGaAs | 0.6 | Detectors |
| ZnSeS | 0.3 | LEDs |
| SiGe | 0.5 | HBTs |

## Asymmetry Factor Determination

### Definition
```
AE = ΔEc / ΔEg
```

### Values for Common Systems

| Material | AE | Notes |
|----------|-----|-------|
| AlGaAs | 0.6-0.7 | Conduction band dominates |
| InGaAs | 0.4-0.5 | More symmetric |
| SiGe | 0.2-0.3 | Valence band dominates |

## Quasi-Field Applications

### HBT (Heterojunction Bipolar Transistor)
- Graded base creates accelerating field
- Reduces base transit time
- Improves high-frequency performance

### Photodetector
- Graded absorption region
- Accelerates carriers
- Reduces transit time
- Improves bandwidth

## Current Balance in Equilibrium

### Electron Current
```
Jn = e × n × μn × En + e × Dn × dn/dx = 0
```

### Hole Current
```
Jp = e × p × μp × Ep - e × Dp × dp/dx = 0
```

### Einstein Relation
```
Dn/μn = Dp/μp = kT/e
```

## Acousto-electric Effect Details

### Condition for Wave Generation
```
v_drift > v_sound
```

### Typical Sound Velocities

| Material | v_sound (m/s) |
|----------|---------------|
| GaAs | 5200 |
| Si | 9000 |
| CdS | 4500 |

### Applications
- Acousto-electric amplifiers
- Delay lines
- Signal processing