---
name: driftfusion-tool-capabilities
description: Evaluate Driftfusion's capabilities for 1D drift-diffusion simulation of mixed ionic-electronic conducting layers, including interface architecture, physical models, and verification status.
---

# Driftfusion Tool Capabilities

Use this skill when you need to:
- Evaluate whether Driftfusion is suitable for your simulation needs
- Understand the tool's architecture and limitations
- Verify simulation results against analytical models
- Configure mixed ionic-electronic conducting device simulations

## Core Capabilities

**Platform:**
- 1D drift-diffusion simulation tool
- Based on MATLAB's `pdepe` toolbox
- Requires MATLAB environment

**Device Support:**
- Mixed ionic-electronic conducting layers
- Up to two ionic carrier species
- Virtually any number of material layers

## Interface Architecture

**Approach:** Discrete interlayer interface

**Features:**
- Material properties can be graded between adjoining layers
- Easy specification of interface-specific properties
- Flexible layer configuration

**Trade-off:** Introduces marginal errors compared to abrupt interface models

## Physical Models

**Included Models:**
- Default models for carrier densities and fluxes
- Analytical approximations for interface regions
- Volumetric recombination scheme for surface recombination approximation

**Carrier Handling:**
- All carriers resolved in all regions
- Ionic carriers can be modeled as inert in transport layers
- Ionic charge compensated by static background charged density

## System Features

**Protocol Functions:**
- Time-dependent voltage conditions
- Time-dependent light conditions

**Analysis Tools:**
- Built-in analysis functions
- Built-in plotting functions
- Parallel calculation support for multiple solutions

## Verification Status

**Comparison Models:**
- Verified against two analytical models
- Verified against two existing numerical models

**Results:**
- Good agreement in all cases
- General device behavior reproduced
- Marginal errors from:
  - Discrete interface treatment
  - Linear discretization scheme

## Limitations

**Dimensionality:** One-dimensional simulation only

**Interface Treatment:** Discrete interfaces introduce small numerical errors compared to abrupt interface models