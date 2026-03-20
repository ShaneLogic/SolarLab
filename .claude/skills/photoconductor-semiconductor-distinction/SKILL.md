---
name: photoconductor-semiconductor-distinction
description: Classify materials as either typical semiconductors or photoconductors based on thermal majority carrier density and bandgap width. Use this when analyzing material response to light, understanding photosensitivity mechanisms, or designing photodetectors and photoconductive devices.
---

# Photoconductor vs Semiconductor Classification

## When to Use
- Analyzing material response to light
- Designing photodetectors or photosensitive devices
- Understanding photosensitivity mechanisms
- Explaining quasi-Fermi level behavior under illumination
- Selecting materials for optoelectronic applications

## Classification Decision Tree

### Typical Semiconductors (e.g., Si)

**Characteristics:**
- High density of thermally created majority carriers
- Bandgap is relatively narrow
- Donors are not depleted

**Under optical generation:**
- Insignificant increase in majority carrier density
- Large increase in minority carrier density
- **Quasi-Fermi level behavior**: Only minority Fermi level shifts significantly

**Result:** Split of quasi-Fermi levels by shifting minority level only

### Typical Photoconductors (e.g., CdS)

**Characteristics:**
- Low density of thermally created majority carriers (acts as insulator in dark)
- Wider bandgap than typical semiconductors
- Donor distance from conduction band is larger
- Donors are depleted due to compensation

**Under optical generation:**
- Compensation is partially lifted
- Majority carrier density increases significantly
- **Quasi-Fermi level behavior**: Both quasi-Fermi levels shift significantly

**Result:** Both EFn and EFp shift substantially under illumination

## Execution Procedure

### 1. Analyze Thermal Majority Carrier Density

- **High density** → Typical semiconductor behavior
- **Low density** → Photoconductor behavior

### 2. Check Bandgap Width

- **Narrower bandgap** → Typical semiconductor
- **Wider bandgap** → Photoconductor

### 3. Examine Donor State

- **Donors not depleted** → Typical semiconductor
- **Donors depleted due to compensation** → Photoconductor

### 4. Analyze Illumination Response

- **Majority carriers: small change** → Typical semiconductor
- **Majority carriers: large increase** → Photoconductor

## Sensitization Mechanism

**Photoconductors are often sensitized** by doping with minority carrier traps:
- Example: Cu in CdS
- Traps have small cross-section to capture majority carriers
- Results in very low recombination rate
- Enables high photosensitivity

## Examples

| Material Type | Example | Thermal Majority Carriers | Bandgap |
|---------------|---------|--------------------------|---------|
| Semiconductor | Si | High | ~1.1 eV |
| Photoconductor | CdS | Low | ~2.4 eV |

## Key Variables

- **majority_carrier_density**: Density of thermally created majority carriers
- **bandgap**: Energy difference between valence and conduction bands
- **compensation**: Ratio of compensating defects to dopants