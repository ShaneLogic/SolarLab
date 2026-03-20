# Ion Migration in Perovskites

## Physical Mechanisms

**Ion types:**
- Vacancies (V_I^+, V_I^-): Most mobile
- Interstitials (I_i^+, I_i^-): Less common
- Complex defects: May migrate

**Activation energies:**
- Iodide vacancies: ~0.1-0.6 eV
- Interstitial iodine: ~0.2-0.5 eV
- MA cation: ~0.8-1.2 eV (negligible at RT)

**Timescales:**
- Electronic transport: ns - μs
- Ionic migration: ms - s
- This separation causes hysteresis

## Modified PNP Equations

**Standard PNP:**
```
∂P/∂t = -∂F_P/∂x
F_P = -D_I (∂P/∂x) + (qD_I P / k_B T)(∂φ/∂x)
```

**Steric-enhanced (this skill):**
```
∂P/∂t = -∂F_P/∂x
F_P = -D_I (∂P/∂x)/(1-P/P_lim) + (qD_I P / k_B T)(∂φ/∂x)
```

## Alternative Steric Formulations

**Modified Drift Model:**
```
F_P = -D_I (∂P/∂x) + (qD_I P / k_B T) ln(P_lim/(P_lim-P))(∂φ/∂x)
```

- Same steady state as diffusion model
- Different dynamics
- Configurable via NonlinearFP parameter

## Numerical Considerations

**Stability issues:**
- As P → P_lim, denominator → 0
- Use adaptive time stepping
- Monitor CFL condition

**Recommended solver:**
- Implicit or semi-implicit schemes
- Sufficient grid resolution
- Flux limiters at high density

## Material-Specific Values

| Material | P_lim (m⁻³) | D_I (m²/s) | Notes |
|----------|------------|------------|-------|
| MAPbI3 | ~5×10²⁶ | 10⁻¹³ | Common reference |
| FAPbI3 | ~4×10²⁶ | 10⁻¹³ | Similar to MAPbI3 |
| CsPbI3 | ~6×10²⁶ | 10⁻¹⁴ | Lower mobility |
| Mixed halide | Variable | 10⁻¹³-10⁻¹² | Depends on composition |

**P_lim estimation:**
```
P_lim ≈ 1 / (unit_cell_volume)
For MAPbI3: a ≈ 6.3 Å, V ≈ 250 Å³
P_lim ≈ 4 × 10²⁷ m⁻³ (theoretical)
Practical limits ~10²⁶-10²⁷ m⁻³ (defects, grain boundaries)