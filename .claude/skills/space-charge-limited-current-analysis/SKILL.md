---
name: Space-Charge-Limited Current Analysis
description: Calculate current density in thin semiconductor devices with injecting contacts when carrier density significantly exceeds doping density. Use this skill when analyzing nn+ junctions, thin semiconductor devices, or devices with injecting contacts under sufficient forward bias where the injected carrier density dominates over the background doping.
---

# Space-Charge-Limited Current Analysis

## When to Use

Apply this skill when:
- Analyzing nn+ junctions with sufficient step size
- Working with thin semiconductor devices
- Evaluating devices with injecting contacts
- Forward bias conditions where n >> Nd₁ in the entire lowly doped region

## Prerequisites Check

Before applying this analysis, verify:
1. **Sufficient forward bias** is applied
2. **Thin enough region 1** - the low-doped region can be swept by electrons
3. **Carrier density condition**: n >> Nd₁ throughout the entire low-doped region
4. **Field condition**: F₀ >> F(-d₁)

## Procedure

### Step 1: Verify Applicability

Confirm the device meets "thin device" criteria:
- Entire low-doped region can be swamped with electrons
- Drift current dominates over diffusion current
- Carrier density at injecting boundary is sufficiently above bulk carrier density

### Step 2: Gather Parameters

Collect the following parameters:
- **V**: Applied voltage (V)
- **d₁** or **L**: Width of low conductivity region or device length (cm)
- **ε**: Dielectric constant (dimensionless)
- **ε₀**: Permittivity of free space (F/cm)
- **μn**: Electron mobility (cm²/Vs)

### Step 3: Calculate Current Density

For nn+ junction with low conductivity region width d₁:

```
jn = (ε × ε₀ × μn × V²) / (2 × d₁³)
```

For homogeneous semiconductor of length L with injecting contact:

```
jn = (ε × ε₀ × μn × V²) / (2 × L³)
```

### Step 4: Interpret Results

Key relationships:
- Current ∝ V² (square of applied voltage)
- Current ∝ 1/d₁³ (inversely proportional to third power of width)
- Thinner devices show steeper current increase with bias

## Constraints

- Only valid for "thin devices" where entire low-doped region can be swept by electrons
- Does not depend on doping density or step size beyond minimum range
- Drift current must be much larger than diffusion current

## Output

Returns current density value in A/cm² calculated from voltage and device geometry.