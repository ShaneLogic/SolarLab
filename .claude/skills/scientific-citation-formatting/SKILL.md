---
name: scientific-citation-formatting
description: Parse, classify, and format scientific citations for journal articles, books, monographs, and conference proceedings according to academic standards
---

# Scientific Citation Formatting

Use this skill when working with bibliographic citations in scientific literature.

## When to Use

- Extracting metadata from existing citations (authors, journal, year, pages)
- Formatting new citations in proper academic style
- Classifying entries by publication type (journal, book, conference)
- Parsing bibliography entries for data extraction

## Identify Publication Type

Before formatting, classify the entry type:

| Type | Indicators | Example |
|------|------------|---------|
| **Journal Article** | Abbreviated journal name, volume in bold, page numbers | Phys. Rev. B **90**, 2263 (2006) |
| **Book/Monograph** | Publisher name, location in parentheses, italicized title | *Title* (Publisher, City, Year) |
| **Conference Paper** | "Proc." or "in" keyword, italicized conference name, "pp." for pages | in *Proc. Conf.* (Year), pp. 123-126 |

## Format Journal Articles

**Structure:** Author(s), Journal**Volume**,Page(Year)

1. List authors: Initials. Surname format (e.g., A.D. Katnani)
2. Append abbreviated journal name (e.g., Sol. Energy Mater. Sol. Cells, Phys. Rev. B)
3. Add volume number in bold (marked with **)
4. Add comma and page number(s)
5. Add publication year in parentheses
6. Add letter suffix (a, b) if multiple works by same author in same year

**Example:** `J. Gray, Y. Lee, Sol. Energy Mater. Sol. Cells **90**, 2263–2271 (2006)`

## Format Books and Monographs

**Structure:** Author, *Title* (Publisher, City, Year)

1. List author name(s)
2. Add "(ed.)" after editor name for edited volumes
3. Append italicized book title (marked with *)
4. Add parenthetical publication details: Publisher, City, Year

**Example:** `W.A. Harrison, *Electronic Structure and the Properties of Solids* (Freeman, San Francisco, 1980)`

**Note:** For theses, include "Thesis," and the institution (e.g., F.M. Klaassen, Thesis, Vrije Universiteit to Amsterdam (1961))

## Format Conference Proceedings

**Structure:** Authors, in *Conference Title* (Year), pp. PageRange

1. List authors separated by commas
2. Add "in" keyword
3. Append italicized conference/proceedings title
4. Add conference year in parentheses
5. Add comma and page range prefixed with "pp."

**Example:** `J. Kessler, H. Dittrich, in *Proc. 10th Euro. Conf. Photovoltaic Solar Energy Conversion* (1991), pp. 879–882`

## Parse Existing Citations

When extracting data from raw citation strings:

1. **Author Identification**: Parse initial segment for names (format: Initial.Lastname, handle backslash-period artifacts)
2. **Source Title Identification**: Identify journal abbreviation or book title following authors
3. **Volume**: Find bolded number following journal title
4. **Pages**: Extract page number or range (books use "p." notation)
5. **Year**: Locate four-digit year at end in parentheses
