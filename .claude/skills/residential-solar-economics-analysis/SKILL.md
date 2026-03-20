---
name: residential-solar-economics-analysis
description: Calculate the economic feasibility and payback period for residential solar installations. Use when evaluating whether to install solar panels for a single-family home, comparing panel types (thin-film vs crystalline silicon), estimating total costs including tax incentives, or determining the amortization period for solar investments.
---

# Residential Solar Economics Analysis

## When to Use
Use this skill when:
- Evaluating residential solar investment costs
- Comparing thin-film vs crystalline silicon panel economics
- Calculating total installation costs including tax incentives
- Determining system sizing requirements
- Estimating payback period and amortization

## Prerequisites
Gather the following information before starting:
- Geographic location data (for sunshine hours)
- State-specific tax regulations and incentives
- Current utility electricity rate (cents/kWh)
- Desired percentage of utility power savings

## Procedure

### Step 1: Calculate Base Panel Cost
Select panel type and calculate base cost:
- **Thin-film panels**: ~$2 per peak Watt
- **Crystalline silicon panels**: ~$4 per peak Watt

### Step 2: Calculate Total Installation Cost
1. Add installation costs: $2 to $5 per Watt to panel cost
2. Base consumer cost: ~$7,000 to $9,000 per installed kW
3. Apply tax incentives (30-50% reduction)
4. Calculate final capital outlay

### Step 3: Determine System Sizing
Based on 50% utility power savings and 3 hours/day average sunshine:
- Required installation: 2 to 5 kW for single-family home
- Initial gross cost range: $14,000 to $45,000

### Step 4: Calculate Payback Period
Using 10 cents/kWh savings and 3 hours/day sunshine:
- Annual Savings: ~$100 per year per kW
- Amortization Period: 10 to 15 years
- Post-amortization: Energy is essentially free

## Output
Provide:
- Total initial capital outlay ($)
- Amortization period (years)
- Annual energy savings ($)
- Post-amortization benefits

## Key Variables
- `panel_type`: Thin-film or Crystalline Silicon
- `tax_rate`: Combined federal and state tax incentive percentage (30-50%)
- `sunshine_hours`: Daily average sunshine hours (default: 3)
- `utility_rate`: Current electricity cost in cents/kWh