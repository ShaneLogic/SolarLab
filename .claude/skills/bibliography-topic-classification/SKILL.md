---
name: bibliography-topic-classification
description: Classifies research citations and journal names into domain categories (Photovoltaics, Semiconductor Physics, Surface Science, Materials Science). Use when analyzing bibliography subject matter, performing topic modeling on citation lists, determining research focus from references, or conducting bibliometric domain analysis.
---

# Bibliography Topic Classification

## When to Use
Use this skill when:
- Analyzing the subject matter of a bibliography or reference list
- Performing bibliometric analysis on citation data
- Determining the core research domains of a document from its citations
- Tagging citations with domain labels for topic modeling

## Prerequisites
- Parsed citation source titles (journal names, conference proceedings, book titles)

## Classification Procedure

### Step 1: Scan for Photovoltaic Research
Identify references related to solar energy conversion.

**Keywords to match:**
- 'Photovolt.'
- 'Sol.Energy'
- 'Solar'

**Specific sources:**
- IEEE Photovolt. Spec. Conf.
- Sol. Energy Mater. Sol. Cells
- Prog. Photovolt.

**Assign tag:** `Photovoltaics`

### Step 2: Scan for Semiconductor Physics
Identify references related to solid-state physics and semiconductor properties.

**Keywords to match:**
- 'Semicond.'
- 'Semimetals'
- 'SolidState'
- 'Phys.Rev.'

**Specific sources:**
- Semiconductors and Semimetals
- Phys. Rev. Lett.
- Solid State Commun.
- J. Appl. Phys.

**Assign tag:** `Semiconductor Physics`

### Step 3: Scan for Materials Science & Surface Science
Identify references on material synthesis, characterization, and surface analysis.

**Keywords to match:**
- 'Surf. Sci.'
- 'Thin Solid Films'
- 'J. Cryst. Growth'

**Specific sources:**
- Appl. Phys. Lett.
- J. Vac. Sci. Technol.

**Assign tags:** `Surface Science` or `Materials Science`

### Step 4: Assign Domain Tags
For each reference:
1. Match source title against keyword patterns
2. Identify all applicable domains
3. Assign one or more domain tags

## Output Format
For each classified reference:
```
Source: [journal/conference name]
Matched: [keywords or sources matched]
Tags: [domain tags]
```

## Interpretation
- High density of references in a domain indicates a core topic of the parent document
- Multiple domain tags on a single reference indicate interdisciplinary work
- Classification is limited to explicitly identified journals and sources