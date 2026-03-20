# Carrier Lifetime Spatial Distribution Details

## Equation Reference Map
- Eq. 27.45: τₙ₀ = 1 / (cₙ × Nᵣ)
- Eq. 27.46: τₚ₀ = 1 / (cₚ × Nᵣ)
- Eq. 27.47: General τₙ polynomial expression
- Eq. 27.48: τₙ = (p - p₀) / U
- Eq. 27.49: τₚ = (n - n₀) / U
- Eq. 27.50: U = Δp / τₚ = Δn / τₙ

## Demarcation Lines

Lifetimes depend on the spread of demarcation lines in the bandgap:
- **Electron demarcation line:** Separates traps acting as electron recombination centers from those acting as hole traps
- **Hole demarcation line:** Similar separation for holes

The position of these lines depends on:
- Trap energy level (Eₜ)
- Quasi-Fermi levels
- Temperature

## Physical Interpretation of Spatial Variation

### Why Lifetime Varies Spatially:
1. **Carrier concentrations change** across the device (n and p are not constant)
2. **Quasi-Fermi levels split** differently in different regions
3. **Trap occupancy statistics** depend on local carrier populations

### At the pn-Junction Interface:
- Both n and p are small (depletion region)
- E_F ≈ Eᵢ near the metallurgical junction
- This configuration maximizes the denominator in the general lifetime expression
- Result: Maximum lifetime occurs here

### In Bulk Regions:
- One carrier type dominates (majority carriers)
- Minority carrier concentration is small
- Lifetime simplifies to τₙ₀ or τₚ₀
- Nearly constant throughout the bulk

## Edge Cases

1. **Very low recombination center density:** Lifetimes become extremely long (material approaches "perfect" semiconductor)
2. **Very high Nᵣ:** Lifetimes approach zero (highly defective material)
3. **Traps at mid-gap:** Most effective recombination centers, minimize lifetime
4. **Traps near band edges:** Less effective recombination centers, have minimal impact on lifetime