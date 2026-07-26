# Building the manual

`solarlabPubManual.md` is the single source of truth. The `.tex` and the
dated `SolarLabManual<YYMMDD>.pdf` files are build artifacts — there is no
script, no CI job, and nothing in the repository references them, so they
only change when someone runs the build below.

All styling (report class, Arial, 0.85 in margins, TOC, coloured links)
lives in the markdown's YAML front matter, so the build takes no hidden
flags. Bump `date:` in that front matter before building.

```bash
cd docs/manual

pandoc solarlabPubManual.md -o solarlabPubManual.tex \
    --standalone --pdf-engine=xelatex --resource-path=".:../.."

pandoc solarlabPubManual.md -o SolarLabManual<YYMMDD>.pdf \
    --pdf-engine=xelatex --resource-path=".:../.."
```

Requires `pandoc` and a LaTeX distribution providing `xelatex`.

## `--resource-path` is not optional

The markdown mixes two relative bases for its figures:

- `figures/*.png` — relative to `docs/manual`
- `perovskite-sim/docs/images/*.png` — relative to the **repository root**

Run from `docs/manual` without `--resource-path=".:../.."` and pandoc
cannot see the second group. It then substitutes the alt text for those
three figures, prints a `[WARNING] Could not fetch resource` line, and
**still exits 0** — so a silently degraded PDF looks like a successful
build. Check the output for warnings, and sanity-check the page count
against the previous edition.

## Mint a new dated file; do not overwrite

Each `SolarLabManual<YYMMDD>.pdf` is the record of what was published
under that date. When the content has changed, add a new dated file and
leave the old one alone rather than swapping its contents. Nothing
references these filenames, so a new one costs no cross-reference updates.

To confirm a rebuild actually carries the intended changes, diff the
extracted text rather than trusting the file size:

```bash
pdftotext SolarLabManual<new>.pdf - > /tmp/new.txt
pdftotext SolarLabManual<old>.pdf - > /tmp/old.txt
# each new passage present in new.txt and absent from old.txt,
# each superseded claim gone from new.txt
```
