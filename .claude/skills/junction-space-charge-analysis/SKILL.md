---
name: junction-space-charge-analysis
description: Calculate and analyze space charge distributions in semiconductor junctions with doping steps, both at equilibrium and under bias. Use for nn+-junctions, heterojunctions, or any junction with abrupt doping transitions.
---

# Junction Space Charge Analysis

Calculate space charge distributions at equilibrium and determine how applied bias affects space charge regions in semiconductor junctions with doping steps.

## When to Use
- Junction with abrupt step in donor density (e.g., nn+-junction, heterojunction)
- Analyzing space charge double layer formation
- Determining accumulation vs depletion under bias
- Evaluating quasi-neutrality conditions

## Equilibrium Space Charge Formation

### 1. Identify Regions
- **Region 1**: Lower carrier density (lower doping)
- **Region 2**: Higher carrier density (higher doping)

### 2. Carrier Leakage Process
Carriers 'leak out' from higher density region (Region 2) into lower density region (Region 1).

### 3. Calculate Space Charge in Each Region

**High Doped Region (Region 2):**
```
Q_2 = q(N_d2 - n_2) ≈ qN_d2  (since n_2 << N_d2)
```
- Electrons leave the region
- Remaining ionized uncompensated donors create **positive space charge**

**Low Doped Region (Region 1):**
```
Q_1 = q(N_d1 - n_1)
```
- Electrons accumulate in adjacent region
- Creates **negative space charge**

### 4. Interface Behavior
- Abrupt flip of space charge sign at doping boundary (x=0)
- Doping step produces space charge **double layer**
- Field spike at interface maintains neutrality over entire junction at zero bias

## Bias Effects on Space Charge

### Forward Bias (Anode at left)
- Electrons in higher doped region (Region 2) are 'blown' from junction interface into Region 1
- **Result**: Region 1 becomes 'surplus' or 'accumulation' space charge region (negative charge increases)
- In series connection, accumulation layer dominates resistance

### Reverse Bias (Cathode at left)
- Electrons are pulled away from the junction
- **Result**: Region 2 becomes slightly wider depletion region (positive charge increases)

### Quasi-Neutrality Analysis

**Vanishing Net Current (Equilibrium):**
- Total positive space charge = Total negative space charge
- Quasi-neutrality holds in entire device

**Applied Bias:**
- Quasi-neutrality no longer holds
- Forward bias: Net charge becomes **negative** (negative > positive)
- Reverse bias: Net charge becomes **positive** (positive > negative)

## Output
- Double layer space charge profile (positive in high density, negative in low density)
- Type of space charge region under bias (accumulation vs depletion)
- Net charge state deviation from neutrality
- Field spike characteristics at interface