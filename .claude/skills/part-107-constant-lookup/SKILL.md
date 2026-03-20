---
name: part-107-constant-lookup
description: Retrieve specific constant values from Part 107 Table A.13 (Tables of Constants) for engineering data records identified by their record ID. Use this skill when you need to look up alphanumeric constants from the table for IDs such as '2201843', '2201x244', '2201x005', or similar Part 107 data records.
---

# Part 107 Constant Lookup

## When to Use
Use this skill when you need to retrieve constant values from the Part 107 Table A.13 (Tables of Constants) for specific data record IDs. Common trigger phrases include:
- "Retrieve constants for ID [x]"
- "Look up Part 107 constants"
- "Get values from table A.13"
- "Find engineering constants for record ID"

## Prerequisites
- Access to Part 107 Table A.13 (Tables of Constants)
- Valid record ID to look up

## Procedure

1. **Identify the target row**
   - Locate the row starting with the specified record ID
   - Note that some IDs use '×' notation (e.g., '2201×244' or '2201×005')

2. **Extract column values**
   - For each column position specified in the reference data, extract the alphanumeric constant value
   - Common columns include: 2nd, 3rd, 4th, 5th, 6th, 10th, 11th, 12th, 18th, and 19th columns
   - Note that some rows may have multiple entries in the same column

3. **Capture parameter values**
   - Extract associated parameter values from the bottom section of the record
   - Standard parameters include: mc, tsε, tpoε
   - Common parameter values: mc='m', tsε='m', tpoε='c'

4. **Return the complete constant set**
   - Compile all extracted values into a structured response
   - Include the record ID, all column values, and parameter values

## Notes
- Values may contain special characters including ε, ᣕ, −, ⊥, =, and semicolons
- Multiple values in a single cell are separated by semicolons
- Some entries use approximation notation when exact column positions vary
- Refer to the detailed lookup table in references/ for specific column mappings per record ID