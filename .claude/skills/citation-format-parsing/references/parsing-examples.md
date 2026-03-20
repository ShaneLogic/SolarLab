# Citation Parsing Examples

## Book Citation
**Input:**
```
S.M. Sze, Physics of Semiconductor Devices (Wiley, New York, 1981)
```

**Parsed Output:**
```json
{
  "authors": "S.M. Sze",
  "title": "Physics of Semiconductor Devices",
  "source_details": {
    "type": "book",
    "publisher": "Wiley",
    "city": "New York",
    "year": "1981"
  }
}
```

---

## Journal Article Citation
**Input:**
```
J. Bardeen, L.N. Cooper, J.R. Schrieffer, Phys. Rev. 106, 882 (1957)
```

**Parsed Output:**
```json
{
  "authors": "J. Bardeen, L.N. Cooper, J.R. Schrieffer",
  "title": "(implied from context or preceding text)",
  "source_details": {
    "type": "journal_article",
    "journal": "Phys. Rev.",
    "volume": "106",
    "pages": "882",
    "year": "1957"
  }
}
```

---

## Book Chapter / Conference Paper
**Input:**
```
M. Sugiyama et al., in Semiconductor Interfaces at the Atomic Scale, ed. by G. LeLay (Springer, Berlin, 1993)
```

**Parsed Output:**
```json
{
  "authors": "M. Sugiyama et al.",
  "title": "(chapter title may precede 'in')",
  "source_details": {
    "type": "book_chapter",
    "containing_work": "Semiconductor Interfaces at the Atomic Scale",
    "editor": "G. LeLay",
    "publisher": "Springer",
    "city": "Berlin",
    "year": "1993"
  }
}
```

---

## Edge Cases

### Multiple Authors with Et Al.
- "M. Sugiyama et al." indicates the primary author with additional authors omitted
- Preserve the "et al." notation in the authors field

### Bold Volume Numbers
- Physics journals often use bold formatting: `**47**, 1`
- Extract volume number without formatting markers

### Abbreviated Journal Names
- Common abbreviations: Phys. Rev., J. Appl. Phys., Nature
- Preserve original abbreviation; do not expand

### Missing Elements
- Some citations may lack city or page information
- Return null or empty string for missing fields

## Format Variations by Source Type

| Source Type | Key Identifier | Typical Format |
|-------------|----------------|----------------|
| Book | (Publisher, City, Year) | Author, Title (Publisher, City, Year) |
| Journal | JournalName Vol, Pages (Year) | Author, Title, Journal Vol, Pages (Year) |
| Chapter | "in" + "ed. by" | Author, Chapter Title, in Book Title, ed. by Editor (Publisher, City, Year) |