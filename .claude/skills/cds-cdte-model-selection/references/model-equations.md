# Model Equations for CdS/CdTe Solar Cell Analysis

## IFC Model (Eq. 35.3)

The IFC (Interface Controlled) model equation:

```
ηc = [1 - exp(-WC/L)] × [1 - (V/V0)]
```

Where:
- ηc = collection efficiency
- WC = depletion width
- L = carrier diffusion length
- V = applied voltage
- V0 = characteristic voltage

**Fitting Parameters:**
- NA (acceptor density)
- V0 (characteristic voltage)

**Example Best-Fit Values:**
- NA = 2.5 × 10^15 cm^-3
- V0 = 0.72 eV

**Assessment:** Does NOT approach measured curve completely

---

## n-p Model (Eq. 35.3)

The n-p (heterojunction) model uses the same equation form as IFC but with different physical interpretation:

```
ηc = [1 - exp(-WC/L)] × [1 - (V/V0)]
```

**Fitting Parameters:**
- NA (acceptor density)
- V0 (characteristic voltage)

**Example Best-Fit Values:**
- NA = 1.6 × 10^16 cm^-3
- V0 = 0.84 eV

**Assessment:** Does NOT approach measured curve completely

---

## n-i-p Model (Eq. 35.5)

The n-i-p model uses a different equation structure:

```
ηc = XC × [1 - (V/V0)]
```

Where:
- XC = collection length ratio parameter (dimensionless)
- V = applied voltage
- V0 = characteristic voltage

**Fitting Parameters:**
- XC (collection length ratio)
- V0 (characteristic voltage)

**Example Best-Fit Values:**
- XC = 22.5
- V0 = 0.84 eV

**Assessment:** DOES approximate measured curve successfully

**Advantages:**
- Uses ONLY TWO adjustable parameters
- Best approximation among the three models
- Describes physics of CdS/CdTe solar cell operation most accurately

---

## Degradation Parameter Changes

### Before Degradation (Cu/Ni contact)
- XC = 22.5
- V0 = 0.84 eV

### After 25 days at 80°C (open circuit condition)
- Cu/Ni contact: Parameters change, but copper layer reduces degradation
- Ni only contact: Different parameter values, typically worse degradation

**Key Finding:** Copper layer is essential to reduce degradation in open circuit condition

---

## Physical Interpretation Guidelines

### Parameters with Direct Physical Meaning
These can be trusted when measured directly:
- Hall mobility (from Hall effect experiment)
- Optical absorption constant (from direct optical experiments)

### Parameters from Model Fitting
Interpret with caution:
- NA (acceptor density) - may differ from actual material properties
- XC (collection length ratio) - empirical parameter, not directly physical
- V0 (characteristic voltage) - related to but not identical to built-in potential

**Recommendation:** Use fitted parameters for:
- Relative comparisons between samples
- Tracking degradation trends
- Process optimization feedback

Do NOT use for:
- Absolute material property claims
- Direct comparison with literature values from different measurement techniques