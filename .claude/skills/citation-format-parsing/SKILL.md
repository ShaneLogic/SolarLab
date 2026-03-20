---
name: citation-format-parsing
description: Parse academic citation strings to identify source type (book, journal article, book chapter, or conference paper) and extract structured metadata including authors, title, and publication details. Use when analyzing reference lists, building bibliographies, extracting publication information from citation text, or converting unstructured citations to structured format.
---

# Citation Format Parsing

## When to Use
Use this skill when you need to:
- Parse a reference string and identify its source type
- Extract structured metadata from academic citations
- Build or clean bibliographic databases
- Convert citation text to machine-readable format

## Prerequisites
- Published material with bibliographic details available
- Citation string in standard academic format

## Parsing Procedure

### Step 1: Identify the Author Block
Locate the author names at the beginning of the citation:
- Format varies: initials + surname (e.g., "S.M. Sze") or surname + initials
- Multiple authors indicated by "et al." (e.g., "M. Sugiyama et al.")

### Step 2: Identify the Title
Extract the title text following the author block:
- Often italicized or in plain text
- Example: "Physics of Semiconductor Devices"

### Step 3: Determine Source Type
Apply decision tree based on delimiters:

**IF format is `(Publisher, City, Year)`:**
- Source type: BOOK
- Parse Publisher, City, and Year from parentheses
- Example: "(Wiley, New York, 1981)"

**IF format is `JournalName Volume, Pages (Year)`:**
- Source type: JOURNAL ARTICLE
- Parse Journal name (often abbreviated), Volume, Pages, and Year
- Example: "Phys. Rev. 106, 882 (1957)"
- Note: Volume numbers may be bolded in physics journals

**IF format contains `in [Title], ed. by [Editor]`:**
- Source type: BOOK CHAPTER or CONFERENCE PAPER
- Parse containing work title and editor information
- Example: "in Semiconductor Interfaces..., ed. by G. LeLay..."

### Step 4: Handle Special Notations
- "et al.": Indicates multiple authors
- Bold volume numbers: Common in physics journal citations
- Abbreviated journal names: Standard in scientific citations

## Output Structure
Return a structured metadata object containing:
- `authors`: List of author names (String)
- `title`: Title of the work (String)
- `source_details`: Object with Publisher/Journal, Volume, Pages, Year

## Constraints
- Format varies slightly between books and journals
- Some citations may have non-standard formatting requiring manual interpretation