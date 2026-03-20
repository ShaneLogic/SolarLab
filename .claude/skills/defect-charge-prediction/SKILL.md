---
name: defect-charge-prediction
description: Predicts whether point defects (vacancies and interstitials) in ionic compounds and semiconductors act as donors or acceptors. Use this skill when analyzing defect chemistry, determining the charge character of vacancies or interstitials, or applying the 8-N rule to mixed bonding systems.
---

# Defect Charge Prediction

## When to Use This Skill

Use this skill when you need to:
- Determine the charge character of vacancies or interstitials in ionic compounds
- Predict donor/acceptor behavior in semiconductors with mixed bonding
- Apply the 8-N rule to analyze point defects

## The 8-N Rule Principle

Elements tend to complete their outer shell (8 electrons) by sharing with neighbors:
- **Surplus electrons**: Donated to the lattice → Donor behavior
- **Missing electrons**: Attracted from elsewhere in the lattice → Acceptor behavior

## Prediction Procedure

### Step 1: Identify the Defect Type

Determine whether you are analyzing:
- Metal-ion interstitial
- Metal-ion vacancy
- Nonmetal-ion vacancy

### Step 2: Apply Prediction Rules

#### Metal-ion Interstitials
1. Identify the interstitial metal ion
2. Predict: **Usually act as Donors**
3. Reasoning: Interstitial metal ions tend to donate valence electron(s), reducing their radius and causing less lattice deformation
4. Example: Cd_i in CdS acts as a donor

#### Metal-ion Vacancies
1. Identify the missing metal ion
2. Predict: **Act as Acceptors**
3. Example: V_Cd in CdS acts as an acceptor

#### Nonmetal-ion Vacancies
1. Identify the missing nonmetal ion
2. Predict: **Usually act as Donors**
3. Example: V_S in CdS acts as a donor

### Step 3: Apply 8-N Rule for Mixed Bonding

For elements with mixed bonding character:
1. Determine N (number of valence electrons)
2. Calculate electron surplus or deficit relative to 8
3. Surplus → Donor; Deficit → Acceptor

## Variables

| Variable | Type | Description |
|----------|------|-------------|
| N | Integer | Number of valence electrons of the element |

## Constraints

- The 8-N rule is a simplified judgment and may not apply to all cases
- Requires understanding of valence electron theory
- Best suited for ionic compounds and semiconductors with mixed bonding

## Output

Returns the predicted charge character (Donor/Acceptor) for the specified vacancy or interstitial.