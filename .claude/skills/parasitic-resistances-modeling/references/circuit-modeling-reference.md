# Parasitic Loss Analysis

## Series Resistance (Rs) Effects

**Current equation with Rs:**
```
J(V) = J_diode(V - J*Rs*A)
```

**Impact on J-V parameters:**
- V_OC: Nearly unchanged (J ≈ 0 at V_OC)
- J_SC: Reduces by factor ~1 - Rs/R_sh
- FF: Significantly reduced
- P_max: Reduced due to FF loss

**Maximum power point condition:**
```
V_mpp = V_mpp,ideal - J_mpp*Rs*A
```

## Parallel Resistance (Rp) Effects

**Current with Rp (shunt):**
```
J(V) = J_diode(V) + V/Rp
```

**Impact on J-V parameters:**
- V_OC: Reduced by leakage current
- J_SC: Slightly increased by V_OC/Rp
- FF: Significantly reduced (curve softening)
- P_max: Reduced due to V_OC and FF loss

## Combined Effects

**Complete single-diode model:**
```
J = J_ph - J_0[exp(q(V+J*Rs*A)/nkT) - 1] - (V+J*Rs*A)/Rp
```

## Extraction Methods

**From J-V curve:**
1. Rs: Slope at V_OC (or V >> V_OC)
2. Rp: Slope at J_SC (or V ≈ 0)
3. Fit full curve using least-squares

**From impedance:**
1. High-frequency intercept → Rs
2. Low-frequency intercept → Rp
3. More accurate than J-V methods

## Common Causes

| Loss Type | Physical Origin | Mitigation |
|-----------|----------------|------------|
| High Rs | Poor contacts, thin TCO | Thick electrodes, annealing |
| Low Rp | Pinholes, defects | Improved processing, passivation |
| Band mismatch | Wrong electrode material | Interlayers, workfunction tuning |
| Interface losses | Dipoles, traps | Surface treatment, buffer layers |