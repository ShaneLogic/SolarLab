# Semiconductor vs Photoconductor - Detailed Comparison

## Material Properties Comparison

| Property | Typical Semiconductor (e.g., Si) | Typical Photoconductor (e.g., CdS) |
|----------|--------------------------------|----------------------------------|
| Thermal majority carrier density | High | Low (insulating in dark) |
| Bandgap | Narrower (~1.1 eV for Si) | Wider (~2.4 eV for CdS) |
| Donor distance from conduction band | Smaller | Larger |
| Donor state | Not depleted | Depleted due to compensation |

## Response to Optical Generation

| Aspect | Typical Semiconductor | Typical Photoconductor |
|--------|---------------------|----------------------|
| Majority carrier density change | Insignificant | Significant increase |
| Minority carrier density change | Large increase | Large increase |
| EFn (electron quasi-Fermi) | Negligible shift | Significant shift |
| EFp (hole quasi-Fermi) | Significant shift | Significant shift |

## Sensitization in Photoconductors

### Mechanism
Photoconductors achieve high photosensitivity through sensitization:

1. **Doping with minority carrier traps**
   - Example: Cu in CdS
   - Traps are strategically placed in bandgap

2. **Capture cross-section engineering**
   - Small cross-section for majority carrier capture
   - Prevents immediate recombination

3. **Resulting behavior**
   - Very low recombination rate
   - Long carrier lifetimes
   - High photosensitivity

### Energy Band Diagram

```
Conduction Band (Ec)
    |
    |  ↓ Electrons excited by light
    |
    |     Donors (depleted in dark)
    |
    |  Minority carrier traps (Cu levels)
    |
Valence Band (Ev)
```

## Practical Implications

### For Device Design

**Semiconductors**: Better for
- Electronic switching
- High-frequency operation
- Low dark current requirements

**Photoconductors**: Better for
- High-sensitivity photodetection
- Low-light imaging
- Photoconductive switches

### For Material Selection

Consider:
- Operating temperature
- Required dark current
- Target sensitivity
- Response time requirements

## Material Examples

| Material | Type | Bandgap (eV) | Comments |
|----------|------|--------------|----------|
| Si | Semiconductor | 1.1 | Ubiquitous, mature technology |
| Ge | Semiconductor | 0.66 | High dark current |
| CdS | Photoconductor | 2.4 | Classic photoconductor |
| CdSe | Photoconductor | 1.74 | Visible light response |
| PbS | Photoconductor | 0.37 | IR detection |