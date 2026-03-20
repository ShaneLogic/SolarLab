# Schottky Diode Equation Reference

## Complete Equation Set

### Diffusion-Limited Equation (Eq. 26.50)
Applies when μnFj >> v* (high field, sufficient reverse bias):

```
jn = e·nc·v*·[exp(eV/kT) - 1]
```

**Characteristics:**
- Simple exponential behavior with forward bias
- Perfect current saturation in reverse direction
- Referred to as "ideal characteristics"

**Family parameter behavior (from Fig. 26.13):**
- nc values: 12.6, 25, 50, 100×10⁴ cm⁻³ for curves 1-4
- Higher nc → higher saturation current

### Drift-Limited Equation (Eq. 26.49)
Applies when |μnFj| << v* (low field, forward/low reverse bias):

```
jn = e·nc·μn·Fj·[exp(eV/kT) - 1]
```

**Characteristics:**
- Current limited by drift velocity
- Fj is bias-dependent → no true saturation
- Classical behavior for low-field conditions

### Shape Factor (Eq. 26.51)

```
SF = 1 / [1 + v*/|μnFj|]
```

**Physical interpretation:**
- Separates classical diode equation from modifying factor
- Results from current continuity at metal/semiconductor interface
- Current at left side: emission-limited (v*)
- Current at right side: drift-limited (μnFj)

**Modification is more pronounced when:**
- μn is smaller
- V is smaller
- Nd is smaller
- T is smaller
- ε is larger
- nc is larger

### Modified Diode Equation (Eq. 26.47)

**Thermal velocity:**
```
vn* = √(kT / 2πm*)
```

**Richardson-Dushman emission current:**
```
jR = e·nc·vn*
```

**Carrier density at semiconductor interface:**
```
nj = nc / [1 - jn/(e·nc·vn*)]
```

**Modified I-V characteristic:**
```
jn = e·μn·nc·Fj·[exp(eV/kT) - 1] / [1 + jn/(e·nc·vn*)]
```

## References
- Böer 1985a, 1985b
- Equation numbers from Chapter 26