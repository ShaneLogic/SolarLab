# Discrepancy Analysis Formulas

## Percentage Difference Calculation

$$\text{Percentage Difference} = 100 \times \frac{J_{ASA} - J_{DF}}{J_{ASA}}$$

**Variables:**
- $J_{ASA}$: Current density from ASA simulator
- $J_{DF}$: Current density from Driftfusion simulator

## Expected Differences by Parameter Set

### Parameter Set 1 (PS1)
- **Magnitude:** ~1% for $J > 10^{-12}$ A cm⁻²
- **Thickness sensitivity:** Minimal (PS1b shows little change)

### Parameter Set 2 (PS2)
- **Magnitude:** Up to ~5% for $J > 10^{-12}$ mA cm⁻²
- **Root causes:**
  - Electron density change at absorber-ETL interface: $>10^7$ orders of magnitude
  - eDOS transition: $N_{CB} = 10^{18} \rightarrow 10^{20}$ cm⁻³
  - Conduction band energy shift: $\Delta E_{CB} = 0.3$ eV

## Numerical Method Differences

**Linear Discretization (PDEPE, Driftfusion):**
- Cannot calculate carrier density changes within interfaces to high accuracy
- Graded interface approach

**Internal Boundary Conditions (ASA):**
- Abrupt interface treatment
- Higher accuracy at interfaces

## Mitigation Effectiveness

**Uniform eDOS Strategy (PS2b):**
- Set $N_{CB} = 10^{18}$ cm⁻³ across all layers
- Result: Significant reduction in deviation

**Preserved Metrics:**
- Open-circuit voltage ($V_{OC}$)
- Ideality factor
- Fill factor

Despite percentage differences in PS2a, key J-V curve metrics remain preserved.