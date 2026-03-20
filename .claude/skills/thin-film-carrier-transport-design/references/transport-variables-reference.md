# Carrier Transport Variables Reference

## μe - Electron Drift Mobility
- Units: cm²/Vs
- a-Si:H: ~2 cm²/Vs
- nc-Si:H: Higher than a-Si:H (improved crystallinity)

## μh - Hole Drift Mobility
- Units: cm²/Vs
- a-Si:H: ~0.01 cm²/Vs
- nc-Si:H: ~1 cm²/Vs

## μτ - Mobility-Lifetime Product
- Units: cm²/V
- Characterizes how far a carrier travels before recombination
- Collection length Lc = μτ × E (electric field)
- Typically μτ_e >> μτ_h for both a-Si:H and nc-Si:H

## Collection Length
- Electrons: Lc_e = μ_e × τ_e × E
- Holes: Lc_h = μ_h × τ_h × E
- For efficient collection: i-layer thickness < Lc_h

## Typical Values

| Parameter | a-Si:H | nc-Si:H |
|-----------|--------|----------|
| μe (cm²/Vs) | ~2 | >2 |
| μh (cm²/Vs) | ~0.01 | ~1 |
| μτ_e (cm²/V) | ~10⁻⁶ | ~10⁻⁵ |
| μτ_h (cm²/V) | ~10⁻⁸ | ~10⁻⁶ |
