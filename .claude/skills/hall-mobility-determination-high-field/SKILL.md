---
name: hall-mobility-determination-high-field
description: Measure Hall mobility in CdS crystals at different field strengths using Hall electrodes. Use when you need to characterize mobility in both high-field domain regions and low-field regions, typically showing 1/F behavior above 30 kV/cm.
---

# Hall Mobility Determination in High-Field Domains

Measure electron mobility in CdS crystals under varying field conditions by positioning the high-field domain over Hall electrodes and switching between high-field and low-field measurements.

## When to Use

- Working with **CdS crystals equipped with Hall electrodes**
- Need to measure **mobility at different field strengths**
- Characterizing **field-dependent mobility behavior**
- Studying transitions between low-field (~500 cm²/Vs) and high-field regimes

## Procedure

1. **Apply high-field bias**
   - Apply sufficient voltage to cause the high-field domain to extend over the entire region between cathode and Hall electrodes
   - Ensure the measured part of the sample is within the high-field domain
   - Record the high-field mobility data point

2. **Switch to low-field measurement**
   - Change the polarity of the applied voltage
   - This positions the measured sample region in the **low-field area**
   - Record the low-field mobility data point (expected ~500 cm²/Vs)

3. **Compare data points**
   - Analyze the pair of mobility measurements
   - Document the difference between high-field and low-field values
   - Plot mobility vs. field strength to characterize behavior

4. **Extend field range (optional)**
   - For wider range of domain fields without changing samples:
     - Use crystals with electrodes of different work functions
     - Use a virtual cathode configuration
   - Expected behavior: **1/F relationship** above 30 kV/cm

## Key Observations

- Low-field mobility: ~500 cm²/Vs (typical)
- High-field behavior: mobility decreases as 1/F above 30 kV/cm
- Polarity switching enables dual-region measurements on single sample
- Different samples may be needed for different field ranges unless using virtual cathode

## Variables

| Variable | Type | Description |
|----------|------|-------------|
| mobility | Coefficient | Hall mobility (cm²/Vs) |
| domain_field_range | Electric Field | Range of fields measured |

## Constraints

- Different samples often needed for different field ranges unless using virtual cathode technique
- Requires precise electrode placement for accurate measurements