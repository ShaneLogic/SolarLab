---
name: domain-field-determination-voltage-width
description: Determine electric field strength in stationary high-field domains in CdS by measuring domain width versus applied voltage. Use when you have optically observable stationary domains and need to calculate the constant field strength within the domain.
---

# Domain Field Determination via Voltage-Width Relation

Determine the constant electric field strength within a stationary high-field domain by analyzing the linear relationship between domain width and applied voltage.

## When to Use

- You have a **stationary high-field domain** in a CdS crystal (not a moving domain)
- The domain is **optically visible** (appears as a darkened band)
- You can **vary the applied bias** and observe domain width changes
- You need to determine the **field strength value** within the domain

## Procedure

1. **Measure initial domain width**
   - Observe the high-field domain optically
   - Measure the width of the darkened band

2. **Apply varying bias and measure width changes**
   - Gradually increase the applied voltage (bias)
   - Observe and record the change in domain width at each voltage level
   - Note that the width increases **linearly** with bias

3. **Verify current behavior**
   - Monitor current through the crystal during measurements
   - Confirm that current remains **constant** while the high-field domain is present
   - This confirms the domain field is constant

4. **Calculate domain field**
   - The increased bias is absorbed entirely in the increased domain width
   - The domain field remains constant throughout the domain
   - Calculate from the **slope** of domain width vs. applied voltage plot
   - Result is expressed in kV/cm (e.g., 80 kV/cm for cathode-adjacent domains)

## Key Observations

- Domain width varies linearly with applied bias
- Current remains constant during domain presence
- Domain field is independent of applied voltage
- Calculation requires only the slope of the width-voltage relationship

## Variables

| Variable | Type | Description |
|----------|------|-------------|
| width | Length | Width of the high-field domain |
| bias | Voltage | Applied voltage/bias |
| domain_field | Electric Field | Constant field strength within the domain |