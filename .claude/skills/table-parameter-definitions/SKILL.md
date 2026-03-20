---
name: table-parameter-definitions
description: Interprets and maps parameter definitions from the footer/definition section of Part 107 tables. Use when you need to identify parameter values (mc, tsε, tpoε, ffp,em, T∂/E∂, p∂/E∂) from table definition rows and understand their primary, secondary, and tertiary values.
---

# Table Parameter Definitions

## When to Use
Use this skill when:
- Interpreting the structure of Part 107 tables
- Reading parameter definitions from the footer/definition section of a table
- Mapping parameter names to their corresponding values
- Identifying primary, secondary, and tertiary parameter values

## Procedure

1. **Locate Parameter Definition Rows**
   - Identify the parameter definition rows at the bottom of the table
   - These rows appear in the footer/definition section

2. **Map Primary Parameter Values**
   - For each parameter header, find its corresponding value in the row below
   - Map parameters to their primary values:
     - `mc` → `m`
     - `tsε` → `m`
     - `tpoε` → `c`
     - `ffp,em` → `(`
     - `T∂/E∂` → `(ffpe`
     - `p∂/E∂` → `)-mc(N`

3. **Identify Secondary Values**
   - Check subsequent rows for secondary values:
     - `tsε` secondary: `c`
     - `tpoε` secondary: `n,`
     - `ffp,em` secondary: `(ffpe`
     - `T∂/E∂` secondary: `μμ`
     - `p∂/E∂` secondary: `m`

4. **Identify Tertiary Values**
   - Check for tertiary values where applicable:
     - `ffp,em` tertiary: `μμ`
     - `T∂/E∂` tertiary: `m`
     - `p∂/E∂` tertiary: `)-mc(N`

## Output
Returns the parameter definitions and their associated values (primary, secondary, tertiary) for the specified table parameters.