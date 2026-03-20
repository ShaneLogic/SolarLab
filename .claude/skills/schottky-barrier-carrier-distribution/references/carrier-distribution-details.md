# Carrier Distribution Detailed Reference

## Modified Boltzmann Range (Eqs. 26.53-26.54)

**Conditions:**
- Sufficiently low bias
- Drift and diffusion currents very large compared to net current

**Boltzmann distribution:**
```
n(x) = nc·exp[-e(ψn(x) - ψn,D)/kT]
```

**Key insight:** Near x=0, drift current term remains important even for low currents.

## DRO-Range Analysis (Eqs. 26.55-26.56)

**Conditions:**
- Larger reverse bias: -V ≥ 2kT/e
- Exponential term negligible
- μnFj << v*

**Current in DRO-Range:**
```
jn = e·nc·μn·Fj
```

**Explicit bias dependence:**
```
jn = e·nc·√[(2eμnNd/εst)·|V|]
```

**Square-root dependence** of reverse current on bias.

**Applications:**
- Determine Nd if other parameters known
- Verify nc and work function information

## Electron Density with Current (Eqs. 26.28-26.33)

**Differential equation:** Linear in n(x) when Schottky approximation for F(x) introduced.

**General solution:**
```
n(x) = Boltzmann term + correction term (linear in current)
```

Involves Dawson's integral D(ξ).

**Simplification for ξ > 4:**
```
D(ξ) ≈ 1/(2ξ)
```

**Correction term behavior:**
- Zero at x = xD
- Maximum at xt ≈ xD - LD
- Drops hyperbolically for xt < xD - 3LD

## Minority Carrier Boltzmann Solution

**Transport equation balance:**
```
epμpF = μpkT·dp/dx
```

**Solution:**
```
p(x) = p10·exp[(ψn,j - ψn(x))/ψn,D]
```

**Note:** Parameters ψn,j, Fj, LD controlled by electron distribution (majority carriers).

## Quasi-Fermi Level Regions

### Majority Carrier (Electrons)
- **Reverse bias:** EFn drops below EFp (constant), changes parallel to Ec → DRO-range
- **Forward bias:** EFn slopes upward toward right electrode

### Minority Carrier (Holes)
- **Low reverse bias:** EFp flat → Boltzmann region
- **High reverse bias:** EFp slopes down, parallel to Ec → DO-range
- **Forward bias:** EFp slopes upward distinctly

### Voltage Drop
- Almost all voltage drop occurs in barrier near interface (DRO-range)

### Depletion Status
- **Reverse bias:** EFp above EFn → minority carrier depletion
- **High reverse bias:** Minority quasi-Fermi level enters majority carrier band
- **Forward bias:** EFn above EFp → carrier accumulation