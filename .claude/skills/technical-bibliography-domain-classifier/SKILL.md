---
name: technical-bibliography-domain-classifier
description: Classify technical references and bibliography entries into subject domains based on journal names, keywords, and author patterns. Use when analyzing subject coverage of a technical bibliography, organizing research literature databases, or performing subject indexing for semiconductor physics and related fields.
---

# Technical Bibliography Domain Classifier

## When to Use

Use this skill when you need to:
- Analyze subject coverage of a technical bibliography
- Classify research papers into subject domains
- Organize literature databases by topic area
- Identify domain distribution in reference collections

## Classification Workflow

1. **Extract identifying features** from each reference:
   - Journal or source publication name
   - Title keywords and topic terms
   - Author names (for known domain experts)

2. **Match against domain patterns** using the reference guide:
   - Compare journal names to known domain-specific publications
   - Identify topic keywords associated with each domain
   - Cross-reference authors with their primary research areas

3. **Assign primary domain** based on strongest evidence:
   - Journal match carries highest weight
   - Topic keywords provide supporting evidence
   - Author patterns confirm classification

4. **Handle multi-domain references**:
   - Some references span multiple domains (e.g., surface science AND electronic devices)
   - Assign primary domain based on main focus
   - Note secondary domains when applicable

## Domain Categories

The classifier covers six primary domains in semiconductor physics and applications:

| Domain | Focus Area |
|--------|------------|
| Semiconductor Physics | Band theory, transport, excitons, defects |
| Photovoltaics & Solar Cells | Solar cell fabrication, efficiency, thin films |
| Surface Science | Surface structure, adsorption, interfaces |
| Materials Science | Crystal growth, thin films, material properties |
| Electronic Devices | Transistors, diodes, device physics |
| Optical Properties | Luminescence, absorption, spectroscopy |

## Output Format

Provide classification with supporting evidence:
```
Primary Domain: [domain name]
Evidence:
  - Journal: [journal name] → [domain association]
  - Topics: [matching keywords]
  - Authors: [recognized domain experts]
Confidence: [high/medium/low]
```

## Constraints

- Domain boundaries may overlap (e.g., optical properties and semiconductor physics)
- Some references span multiple domains
- Classification relies on recognized publication patterns
- Novel or interdisciplinary journals may require manual classification

## Reference Materials

See `references/domain-mapping-guide.md` for:
- Complete journal-to-domain mappings
- Topic keyword lists per domain
- Example author associations