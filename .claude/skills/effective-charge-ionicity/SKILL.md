---
name: effective-charge-ionicity
description: Calculate effective charge (e*) and determine ionicity in III-V, II-VI, and Group IV semiconductors based on wavefunction coefficients. Use when analyzing charge distribution in mixed bonding systems or calculating static vs dynamic effective charges.
---

# Effective Charge and Ionicity in Semiconductor Groups

## When to Use
- Analyzing charge distribution in semiconductor materials
- Determining the degree of ionicity vs covalency in mixed bonds
- Calculating effective charge for different semiconductor groups
- Understanding bonding characteristics of III-V, II-VI, or Group IV materials

## Semiconductor Group Classification

### Group IV Semiconductors (e.g., Si, Ge)
- Wavefunction coefficients: a = b
- Purely covalent bonding
- **Effective charge e* = 0**

### III-V Semiconductors (e.g., GaAs, InP)
- Coefficient ratio: a/b = √3
- Mixed ionic-covalent bonding
- Calculate e* from ratio

### II-VI Semiconductors (e.g., ZnO, CdS)
- Coefficient ratio: a/b = √5
- Higher ionic character than III-V
- Calculate e* from ratio

## Effective Charge Definitions

**Static effective charge (e*):**
- Represents fraction of ionicity averaged over time
- Value ranges from 0 (purely covalent) to 1 (fully ionic)

**Dynamic effective charge:**
- Always less than static charge
- Reduced by fraction on order of b/a compared to purely ionic compound

## Reference Values (Static Effective Charge e*/e)

| Material | Semiconductor Group | e*/e |
|----------|---------------------|------|
| ZnO | II-VI | 0.60 |
| CdS | II-VI | 0.49 |
| ZnS | II-VI | 0.47 |
| GaAs | III-V | 0.46 |
| Si | Group IV | 0.00 |

## Calculation Approach

1. **Identify semiconductor group** (IV, III-V, or II-VI)
2. **Determine coefficient ratio** a/b for the group
3. **Apply effective charge relationship** based on ratio
4. **Distinguish static vs dynamic** charge as needed

## Key Constraints

- Must distinguish between static and dynamic effective charge
- Static charge is the time-averaged ionicity
- Dynamic charge is smaller, accounting for bond polarization during atomic motion

## Variables
| Symbol | Description |
|--------|-------------|
| e* | Static effective ion charge |
| a/b | Ratio of covalent to ionic wavefunction coefficients |