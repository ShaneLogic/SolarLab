---
name: solar-cell-series-resistance-correction
description: Apply series resistance corrections to CdS/CdTe solar cell I-V measurements to obtain accurate performance parameters. Use when series resistance affects measurement accuracy or when determining true cell performance potential from measured I-V characteristics.
---

# Solar Cell Series Resistance Correction

Correct for series resistance effects in CdS/CdTe solar cell measurements to obtain accurate performance parameters including Voc, jsc, fill factor, and efficiency.

## When to Use

Apply this skill when:
- Measuring CdS/CdTe solar cell performance
- Series resistance may affect measurement accuracy
- Need to determine true cell performance potential
- Comparing measured values to theoretical models

## Prerequisites

Before applying corrections, obtain:
- Full AM1 current-voltage characteristic
- Dark current-voltage characteristic
- Measured values for: Voc, jsc, FF, and efficiency (η)

## Correction Procedure

### Step 1: Calculate Difference Curve

Subtract the dark current-voltage curve from the AM1 characteristic:

```
I_corrected(V) = I_AM1(V) - I_dark(V)
```

This eliminates the influence of series resistance and provides a corrected curve for model fitting.

### Step 2: Apply Series Resistance Correction

Correct the measured performance parameters for series resistance R:

- Voc and jsc typically remain unchanged
- Fill factor (FF) increases
- Efficiency may slightly decrease

### Step 3: Apply Collection Efficiency and Series Resistance Correction

Apply both corrections simultaneously:
- Collection efficiency voltage dependence η(V)
- Series resistance R

This step reveals the true potential performance of the cell with significant improvements in all parameters.

### Step 4: Apply Dark Current Saturation Shift

Shift the dark current saturation density j0 by jsc:

```
j0_corrected = j0 + jsc
```

This provides the final refined accuracy for all performance parameters.

## Interpret Results

- Series resistance correction primarily affects fill factor
- Full correction (R + η(V)) shows true potential performance
- Final correction refines accuracy of all parameters
- Compare corrected values to measured values to assess impact of series resistance

## Constraints

- Correction assumes linear series resistance behavior
- May not capture all non-ideal effects
- Results are most accurate for CdS/CdTe solar cells