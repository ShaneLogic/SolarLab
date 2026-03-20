# Collection Efficiency Model Derivations

## Simple Collection Efficiency Model

### Derivation

Starting from carrier continuity in the junction region:
- Carriers drift under built-in field F
- Recombination at interface characterized by velocity S

At steady state:
```
J_drift = e * n * μ * F
J_recomb = e * n * S
```

Collection efficiency:
```
ηc = J_collected / J_generated
ηc = μF / (S + μF)
```

### Application Notes

- Valid when field is approximately constant
- Best near maximum power point
- Two-parameter model enables quick device comparison

## Hecht-like Model Derivation

### Physical Basis

The Hecht equation describes carrier collection in the presence of trapping:

Original Hecht equation (for detectors):
```
Q = Q0 * (μτV/L²) * [1 - exp(-L²/μτV)]
```

### Adaptation for Solar Cells

For p-i-n cells with field-dependent collection:

1. Field varies linearly across i-layer:
```
F(x) = (V0 - V) / d
```

2. Collection length from mobility-lifetime product:
```
LC = sqrt( (kT/e) * μτ )
```

3. Collection efficiency:
```
ηc(V) = (XC/V0) * [V0 - V + (V0/2)(1 - exp(-2(V0-V)/V0))]
```

### Parameter Fitting

To extract XC and V0 from measured ηc(V):

1. Plot ηc vs. (V0 - V)
2. Fit to Hecht-like equation
3. XC relates to carrier transport quality
4. V0 relates to built-in potential

## Collection Zone Model Details

### Absorption Profile

Blue light absorption in α-Si:H:
```
I(x) = I0 * exp(-αx)
```

For α ≈ 10^5 cm^-1:
- 70% absorbed in ~200 nm
- Generation profile: g(x) = α * I(x)

### Carrier Drift

Under field F:
```
Δx = μD * F * τ
```

For typical α-Si:H:
- μD (holes) = 0.1-1.0 cm²/Vs
- τ = 10^-7-10^-6 s
- F = 10^4-10^5 V/cm
- Δx = 100-1000 nm

### Collection Zone Boundary

Defined where recombination rate equals half generation rate:
```
R/g = 0.5
```

Inside zone: R << g (efficient collection)
Outside zone: R ≈ g (carriers lost to recombination)

### Optimization Strategy

1. Ensure i-layer thickness ≥ collection zone width
2. Ensure i-layer thickness ≤ absorption depth for blue light
3. Balance between absorption and collection
4. Typical optimum: 300-500 nm for α-Si:H