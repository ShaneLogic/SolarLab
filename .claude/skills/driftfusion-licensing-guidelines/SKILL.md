---
name: driftfusion-licensing-guidelines
description: Understand and comply with Driftfusion software licensing terms, including the open-source AGPL v3.0 frontend and proprietary MATLAB pdepe solver backend. Use when using, modifying, or distributing Driftfusion code.
---

# Driftfusion Licensing Guidelines

## When to Use
Use this skill when you need to:
- Understand your rights and obligations when using Driftfusion
- Determine if you can modify or redistribute the code
- Clarify licensing terms for the frontend vs. backend components
- Ensure compliance with software licenses

## Dual Licensing Structure

### Frontend License
- **Status:** Open-source
- **License:** GNU Affero General Public License v3.0 (AGPL v3.0)
- **Rights:** Allows use, modification, and distribution under AGPL terms

### Backend Solver License
- **Component:** MATLAB's Partial Differential Equation solver (pdepe)
- **Owner:** MathWorks, Inc.
- **License:** MathWorks Software License Agreement (proprietary)
- **Restriction:** Strictly prohibits modification and distribution of the solver component itself

## Key Restrictions
1. The `pdepe` solver component cannot be modified or distributed
2. Users must comply with both AGPL v3.0 (for frontend) and MathWorks license (for solver access)

## User Obligations
- Users are encouraged to provide feedback and/or contribute to the continued development and dissemination of the project
- Ensure any modifications or distributions respect both licensing frameworks

## Variables
- `frontend_license`: GNU Affero General Public License v3.0
- `solver_license`: MathWorks proprietary license