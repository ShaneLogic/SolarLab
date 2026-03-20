# Statistical Model Reference

## Parabolic Band Model Details

The parabolic band model originates from Taylor expansion of dispersion relation near band extrema.

**Density of States:**
- For E < E_c: `ĝ_c(E) = 0`
- For E ≥ E_c: `ĝ_c(E) = (g_c√2/√π) * √((E - E_c)/k_B T)`

**Fermi-Dirac Integral Definitions:**
- Distribution: `f(E) = 1/(exp((E - E_f)/k_B T) + 1)`
- Integral of order 1/2: `F_{1/2}(ξ) = (2/√π)∫₀^∞ √(η)/(1 + exp(η - ξ)) dη`

## Gaussian Band Model Details

**Physical Origin:**
- Weak molecular bonds create disordered structure
- Electron transport via hopping between molecular sites
- For large molecules, resembles band transport in Gaussian band

**Density of States:**
```
ĝ_c(E) = g_c * exp(-(E - E_L)²/σ²) / (s√π)
```
where σ = s*k_B T is the dimensional width

**Gauss-Fermi Integral:**
- Fermi-Dirac: `G_s(ξ)`
- Boltzmann approximation: `S(ξ) = exp(ξ + s²/2)`

## Available Statistical Functions

| Function Name | Symbol | Inverse | Model |
|--------------|--------|---------|-------|
| exp | exp | ln | Boltzmann |
| F12 | Fermi-Dirac 1/2 | F12inv | Parabolic |
| F32 | Fermi-Dirac 3/2 | F32inv | Parabolic (higher order) |
| G | Gauss-Fermi | Ginv | Gaussian |
| BL | Blakemore | BLinv | Ion vacancies |
| FD | Full Fermi-Dirac | FDLinv | General |