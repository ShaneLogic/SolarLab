# Series Resistance Correction Examples

## Example Data (Table 35.1, Hegedus et al. 2007)

### As Measured Values
- Voc = 0.805V
- jsc = 23.7 mA/cm²
- FF = 69.8%
- η = 13.3%

### After Series Resistance (R) Correction
- Voc = 0.805V (unchanged)
- jsc = 23.7 mA/cm² (unchanged)
- FF = 71.5% (improved)
- η = 12.4% (slightly decreased)

### After R + η(V) Correction
- Voc = 0.826V (improved)
- jsc = 23.8 mA/cm² (improved)
- FF = 81.9% (significant improvement)
- η = 16.1% (significant improvement)

### After j0 Shift by jsc (Final Correction)
- Voc = 0.826V
- jsc = 24.0 mA/cm²
- FF = 81.0%
- η = 16.0%

## Key Observations

1. **Series resistance alone** primarily improves fill factor with minimal impact on other parameters
2. **Combined corrections** (R + η(V)) reveal substantial performance improvements across all metrics
3. **Final j0 shift** provides minor refinements to achieve the most accurate representation

## Variable Definitions

| Variable | Type | Description |
|----------|------|-------------|
| Voc | float | Open circuit voltage (V) |
| jsc | float | Short circuit current density (mA/cm²) |
| FF | float | Fill factor (%) |
| η | float | Efficiency (%) |
| R | float | Series resistance |
| j0 | float | Dark current saturation density |
| η(V) | function | Collection efficiency as function of voltage |