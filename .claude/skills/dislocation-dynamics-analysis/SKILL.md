---
name: dislocation-dynamics-analysis
description: Analyze dislocation motion types, velocity factors, and generation mechanisms in crystals under stress. Use when analyzing crystal deformation, predicting dislocation behavior, determining plastic deformation mechanisms, or evaluating dislocation mobility in materials science and solid-state physics contexts.
---

# Dislocation Dynamics Analysis

## When to Use This Skill
Use this skill when:
- Analyzing crystal deformation mechanisms
- Predicting dislocation movement behavior under stress
- Determining plastic deformation pathways
- Evaluating dislocation generation sources
- Assessing mobility of dislocations in crystalline materials
- Investigating the relationship between stress and dislocation velocity

## Core Analysis Workflow

### 1. Identify Motion Type
Determine whether the dislocation undergoes **glide** or **climb**:

**Glide (Conservative Motion):**
- Motion is parallel to the Burgers vector or within the slip plane
- Essential for plastic deformation
- Does not require atom diffusion

**Climb (Non-Conservative Motion):**
- Motion is normal to both Burgers vector and dislocation line
- Changes height of inserted plane
- Requires atoms to move to/from core (diffusion-dependent)

### 2. Assess Obstacles and Pinning
Check for obstacles that may pin the dislocation:
- Crossing dislocations
- Vacancies
- Foreign atoms
- Other defects

When pinned, the dislocation must climb over the obstacle to continue motion.

### 3. Determine Velocity Factors
Analyze factors affecting dislocation velocity:
- Nucleation rate of double kinks
- Motion of double kinks along the dislocation line
- Electrical charge of defects
- Fermi level position (especially in semiconductors)

### 4. Evaluate Generation Mechanisms
Identify if dislocation generation is occurring:
- Check for Frank-Read sources (dislocations pinned at two ends)
- Observe bowing under shear stress
- Look for sequential dislocation loop formation

### 5. Classify Mobility
Determine if the dislocation is:
- **Glissile**: Can move easily
- **Sessile**: Cannot glide (e.g., partials with Burgers vector inclined to stacking fault)

## Key Variables
- **Shear Stress**: Applied parallel to Burgers vector
- **Fermi Level**: Influences dislocation velocity in semiconductors
- **Kink**: Step along dislocation line aiding motion

## Output
Provide predictions of:
- Dislocation movement type (glide or climb)
- Velocity factors and relative magnitude
- Generation capability and mechanism
- Mobility classification (glissile or sessile)