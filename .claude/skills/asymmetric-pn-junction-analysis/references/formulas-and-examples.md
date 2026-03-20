# Asymmetric PN Junction Formulas and Examples

## Field Distribution Equations

### Space Charge Region Width (Schottky Relation)
```
ln = sqrt( (2 * epsilon * psi_n,Dn) / (e * Nd) )
```
Where:
- ln: Width of lower doped space charge region (cm)
- epsilon: Dielectric constant (F/cm)
- psi_n,Dn: Diffusion potential (V)
- e: Elementary charge (C)
- Nd: Donor density in lower doped region (cm^-3)

### Maximum Electric Field
```
F_max = sqrt( (2 * e * Nd * (psi_n,Dn - V)) / epsilon )
```
Where:
- V: Applied bias voltage (V)

### Diffusion Potential
```
psi_n,Dn = (kT/e) * ln( (Na * Nd) / n_i^2 )
```
Example value: 0.38 V for typical Si junction

## Quantitative Examples

### Example 1: Asymmetric Doping Impact
Configuration:
- Nd = 10^16 cm^-3 (n-type)
- Na = 10^18 cm^-3 (p-type)
- Asymmetry ratio: 100

Results:
- Diffusion voltage increase: +120 mV
- Voc increase: ~12 mV only
- Recombination overshoot shifts into lower doped region

### Example 2: Asymmetric Recombination Centers
Configuration:
- Nr (n-type) = 10^17 cm^-3
- Nr (p-type) = 10^18 cm^-3 (10x increase)

Results:
- Minority carrier density decrease: 17x (superlinear)
- Voc reduction: 68 mV

### Example 3: Surface Recombination Effect
Configuration:
- Strong surface recombination at one surface
- Optical generation present

Results:
- Minority carrier density decrease: 0.4x
- Voc reduction: 23 mV
- Diode quality factor A = 1.7

### Example 4: Thick Device Carrier Dynamics
Configuration:
- Thin heavily doped n-type front layer
- Thick p-type base (d2 > Ln)
- go = 2×10^20 cm^-3 s^-1
- Nr1 = 10^17 cm^-3, Nr2 = 10^16 cm^-3

Key observations:
- Carrier crossover shifts into p-type bulk
- Field maximum remains at junction interface
- Quasi-Fermi level split: 0.533 eV (vs. 0.654 eV theoretical)

## Space Charge Shape Characteristics

In asymmetrically doped junctions:

**Higher doped region**:
- No longer block-shaped
- Carriers diffusing out produce triangular space charge layer
- Non-linear field distribution

**Lower doped region**:
- Hole density increases above donor density
- Spike of positive space charge near junction interface
- Gradual field slope up to spike at doping boundary

## Bias Effects

With applied bias:
- Electrostatic potential distribution deforms
- Bias-dependent step size develops
- Most changes occur in lower doped region
- Field increases rapidly as open circuit voltage approaches
- Field can reach tunneling values in reverse bias