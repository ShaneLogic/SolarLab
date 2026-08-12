# Building the manual

`solarlabPubManual.md` is the single source of truth. The `.tex` and dated
`SolarLabManual<YYMMDD>.pdf` files are build artifacts; there is no CI build,
so they change only when someone runs the workflow below. The root and
`docs/` READMEs link the current dated edition explicitly.

The current publication is
[`SolarLabManual260811.pdf`](SolarLabManual260811.pdf): 88 pages, dated
2026-08-11, with seven generated figures.

All styling (report class, Arial, 0.85 in margins, TOC, coloured links)
lives in the markdown's YAML front matter, so the build takes no hidden
flags. Bump `date:` in that front matter before building.

```bash
cd docs/manual

XDG_CACHE_HOME=/tmp/solarlab-cache \
MPLCONFIGDIR=/tmp/solarlab-mpl-cache \
    python generate_manual_figures.py

pandoc solarlabPubManual.md -o solarlabPubManual.tex \
    --standalone --pdf-engine=xelatex --resource-path=".:../.."

pandoc solarlabPubManual.md -o SolarLabManual<YYMMDD>.pdf \
    --pdf-engine=xelatex --resource-path=".:../.."
```

Requires `pandoc` and a LaTeX distribution providing `xelatex`.

The figure generator writes matched vector `.pdf` and 180-dpi `.png` files
under `figures/`. The manual embeds the PDFs; GitHub READMEs embed the PNGs.
Numerical panels read the reproducibility registry and the registered CBO
result directly. Generation stops if an expected grid ladder or certification
state has changed, so neither output can silently reuse a hard-coded plot.

The CBO input is currently a local, ignored result:

```text
perovskite-sim/outputs/interface-cbo/scan-fermi-edge-qf-grid-40-50-60.json
```

It is not present in the tracked remote tree. A clean clone can read the
committed rendered figures but cannot regenerate `cbo_interface_validation`
until that exact JSON is restored. Do not substitute another scan merely to
make the build run.

## Figure map

| Generated name | Current use | Replaces in GitHub README |
|---|---|---|
| `architecture_flow` | schema-to-result architecture | earlier `architecture_flow.png` |
| `device_contact_boundary` | layer order, signed contact potential, carrier BC | `perovskite-sim/docs/images/device_structure.png` |
| `band_interface_convention` | bent bands, QF splitting, default/QF interface closures | `band_diagram.png` and `transport_equations.png` |
| `solver_topology` | transient, steady, QF, QF-frequency, and 2D drivers | `solver_pipeline.png` |
| `csi_qf_convergence` | restricted-QF internal J-V/C-V convergence | no older equivalent |
| `cbo_interface_validation` | internal grid pass and external SCAPS-shape failure | no older equivalent |
| `twod_scope` | registered parity domain and omitted physics | no older equivalent |

`perovskite-sim/docs/images/ui_layout.png` has no 260811 counterpart and is
retained. The superseded physical/solver PNGs are also retained as historical
assets; current READMEs no longer reference them.

## `--resource-path` is not optional

Manual figures currently live under `docs/manual/figures`, but keep the
repository root in `--resource-path` so future source-linked assets resolve in
the same way. Pandoc can substitute alt text for a missing image and still exit
successfully. Check the output for warnings and confirm that every expected
figure appears in the rendered PDF.

## Mint a new dated file; do not overwrite

Each `SolarLabManual<YYMMDD>.pdf` is the record of what was published under
that date. When the content changes, add a new dated file and leave the old one
alone rather than swapping its contents. Then update the current-edition links
in `README.md`, `docs/README.md`, and `perovskite-sim/README.md` in the same
change.

To confirm a rebuild actually carries the intended changes, diff the
extracted text rather than trusting the file size:

```bash
pdftotext SolarLabManual<new>.pdf - > /tmp/new.txt
pdftotext SolarLabManual<old>.pdf - > /tmp/old.txt
# each new passage present in new.txt and absent from old.txt,
# each superseded claim gone from new.txt
```
