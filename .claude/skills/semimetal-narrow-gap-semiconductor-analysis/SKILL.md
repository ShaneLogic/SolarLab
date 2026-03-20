---
name: semimetal-narrow-gap-semiconductor-analysis
description: Classify and characterize materials with very small band overlap or very small bandgaps (less than ~0.5 eV). Use this skill when analyzing semimetals like graphite and bismuth, narrow gap semiconductors like gray tin and lead chalcogenides (PbS, PbSe, PbTe), HgCdTe alloys, or PbSnTe alloys with band inversion properties.
---

# Semimetal and Narrow Gap Semiconductor Analysis

## When to Use
Apply this skill when:
- Material exhibits very small band overlap (semimetals) or very small bandgap (< 0.5 eV)
- Analyzing graphite, bismuth, gray tin, or lead chalcogenides
- Working with HgCdTe or PbSnTe alloy systems
- Bandgap can be tuned to zero through alloying, pressure, or temperature
- Material shows weak electronic conduction characteristic of few charge carriers

## Classification Framework

### Semimetals
Materials with very small overlap between conduction and valence bands:
- Density of states near Fermi level is very small but non-zero
- Only relatively few electrons control conduction
- Examples: Graphite, Bismuth

### Narrow Gap Semiconductors
Materials with very small positive bandgap:
- Show relatively high conductivity compared to other semiconductors
- Bandgap can vanish through alloying, pressure, or elevated temperature
- When bandgap vanishes, semiconductor transitions to metal
- Examples: Gray tin (0-0.08 eV), PbS, PbSe, PbTe

### Gapless Semiconductors
Special case where bands just touch:
- Top of valence band touches bottom of conduction band
- Density of states at Fermi level is zero at T = 0K
- Distinct from semimetals (which have small but non-zero density of states)

## Analysis Procedure

1. **Determine Material Type**
   - Check if material is elemental (graphite, bismuth), compound (PbS, PbSe, PbTe), or alloy
   - Identify the crystal structure relevant to band analysis

2. **Measure or Retrieve Band Parameters**
   - Determine bandgap at relevant temperature
   - Identify band extrema locations (Γ-point vs L-point)
   - For anisotropic materials, obtain parallel and perpendicular effective masses

3. **Apply Alloy Formulas** (if applicable)
   - Use empirical relations for HgCdTe or PbSnTe systems
   - Verify composition (ζ) and temperature (T) are within valid ranges
   - Account for bowing effects in alloy bandgap

4. **Check Critical Transitions**
   - For PbSnTe: verify if ζ > 0.62 (metal transition point)
   - For HgCdTe: check if Eg dips below zero (semimetal regime)
   - Consider pressure coefficients if pressure is variable

5. **Characterize Electronic Properties**
   - Calculate carrier density based on band structure
   - Evaluate conduction strength based on density of states at Fermi level
   - Predict behavior under temperature or pressure changes

## Key Design Principles
- Narrow bandgap materials can be designed by alloying two narrow gap semiconductors
- Band inversion enables topological phase transitions in alloys like PbSnTe
- Alloy composition provides continuous tuning from semiconductor to metal