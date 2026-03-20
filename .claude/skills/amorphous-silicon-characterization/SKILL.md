---
name: Amorphous Silicon Characterization
description: Determine optical band gap and band tail characteristics in amorphous silicon and its alloys using Tauc plot, E04 method, and mobility edge analysis. Use when characterizing a-Si:H materials, analyzing Urbach tails, or determining electronic structure parameters for amorphous semiconductor devices.
---

# Amorphous Silicon Characterization

## When to Use
- Measuring optical band gap of a-Si:H or a-Si alloys
- Analyzing band tail widths (Urbach tails)
- Determining mobility edges in amorphous semiconductors
- Characterizing material quality for device applications
- Comparing electrical vs optical band gaps

## Prerequisites
- Optical absorption coefficient data α(hν)
- Photon energy data (hν)
- Understanding of band structure concepts

## Key Concepts

### Band Tails
Localized states extending into the band gap due to disorder.

### Mobility Edges
Energy separating localized from delocalized states.

### Optical vs Electrical Band Gap
- Optical (Tauc): From absorption measurements
- Electrical: From conductivity or internal photoemission
- Electrical gap is typically 50-100 meV larger

## Method 1: Tauc Plot

### Procedure
1. Plot (hν × α(hν))^1/2 on y-axis
2. Plot photon energy hν on x-axis
3. Identify linear region of the plot
4. Extrapolate linear region to x-axis
5. X-intercept = Tauc optical band gap (ET)

### Formula Basis
```
(αhν)^1/2 = B(hν - ET)
```

Where B incorporates transition matrix elements.

### Notes
- Most widely used method
- Assumes parabolic band edges
- Proportionality constant B not usually studied separately

## Method 2: E04 Method

### Procedure
1. Obtain absorption coefficient vs photon energy data
2. Identify photon energy where α = 3 × 10^3 cm^-1
3. This energy value = E04

### Advantages
- Simpler than Tauc analysis
- No extrapolation required
- Useful for comparative studies

## Band Tail Analysis

### Valence Band Tail (EV)
- Described as 'Urbach' tail of spectrum
- Typical value for a-Si:H: EV = 50 meV
- Contributes to low hole mobility

### Conduction Band Tail (EC)
- Smaller than valence band tail
- Best a-Si:H: EC ≈ 22 meV
- Increases markedly for a-SiGe alloys

### Mobility Edge Definition
- Energy separating localized from delocalized electrons
- Referred to as conduction and valence band mobility edges (Mott 1987)
- Differs slightly from optical band gap

## Procedure Summary

### Complete Characterization

**Step 1: Obtain Optical Data**
- Measure absorption coefficient α(hν)
- Cover energy range around expected band gap

**Step 2: Determine Optical Band Gap**
- Apply Tauc plot method for ET
- Apply E04 method for comparison

**Step 3: Analyze Band Tails**
- Extract Urbach tail energy from low-α region
- Compare EV and EC values

**Step 4: Relate to Electrical Properties**
- Note that electrical gap > Tauc gap by 50-100 meV
- Use internal photoemission for electrical gap if needed

## Typical Values for a-Si:H

| Parameter | Value | Notes |
|-----------|-------|-------|
| ET (Tauc) | 1.7-1.8 eV | Deposition dependent |
| E04 | ~1.6 eV | Lower than Tauc |
| EV | 50 meV | Urbach tail |
| EC | 22 meV | Best material |
| Electrical gap | 1.8-1.9 eV | 50-100 meV > ET |

## Material Quality Indicators

- Smaller band tails → better electronic quality
- EC increases in a-SiGe alloys (degraded transport)
- EV dominates hole transport limitation