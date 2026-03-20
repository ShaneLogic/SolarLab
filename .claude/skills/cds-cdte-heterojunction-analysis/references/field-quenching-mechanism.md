# Field Quenching Mechanism Details

## Physical Basis

Field quenching in copper-doped CdS is a cascade process that reduces photoconductivity under high electric fields.

## Step-by-Step Process

### Step 1: Optical Excitation Baseline
- Material: Cu-doped CdS (n-type)
- Under optical excitation (solar cell operation)
- Electron density: ~10^18 cm^-3
- High photoconductivity state

### Step 2: Field Application
- Electric field develops across CdS junction region
- Field increases from bulk toward junction interface

### Step 3: Barrier Lowering (at 23 kV/cm)
- Frenkel-Poole effect lowers barrier by δE = 2kT
- Trapped holes in slow copper centers gain enough energy
- Holes released into valence band

### Step 4: Hole Capture
- Released holes captured by fast recombination centers
- These centers have high capture cross-section for holes

### Step 5: Enhanced Recombination
- Fast centers now filled with holes
- Conduction electrons recombine through these centers
- Electron density decreases

### Step 6: Quenched State (above 50 kV/cm)
- Photoconductive electron density markedly reduced
- Material transitions from high-conductivity to low-conductivity state

## Mathematical Description

### Frenkel-Poole Barrier Lowering
```
δE = sqrt( (e^3 * F) / (pi * epsilon) )
```

At F = 23 kV/cm, δE ≈ 2kT (thermal energy at room temperature)

### Carrier Density Evolution
```
n(x) = n0 * exp( -integral of recombination rate )
```

## Implications for Solar Cells

1. **Current Saturation**: Field quenching limits current at high bias
2. **Band Alignment**: Creates bias-dependent electron affinity change
3. **Interface Behavior**: Conduction band position shifts with operating point
4. **Efficiency Impact**: Can limit fill factor at maximum power point

## Experimental Evidence

- Work function measurements at Au/CdS contacts show 0.25 eV change
- Photoconductivity measurements confirm electron density reduction
- Current-voltage characteristics show saturation effects