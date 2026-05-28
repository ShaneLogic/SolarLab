const fs = require("fs");
const path = require("path");
const pptxgen = require("/Users/shane/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/pptxgenjs");

const root = path.resolve(__dirname, "../../..");
const outDir = path.join(root, "docs", "manual", "slides");
const outPath = path.join(outDir, "scaps_1d_input_parameters_editable.pptx");
fs.mkdirSync(outDir, { recursive: true });

const C = {
  navy: "162033",
  slate: "334155",
  muted: "64748B",
  line: "CBD5E1",
  grid: "E2E8F0",
  blue: "2563EB",
  green: "16803C",
  teal: "0F766E",
  amber: "A16207",
  orange: "B45309",
  red: "DC2626",
  purple: "6D28D9",
  white: "FFFFFF",
  offWhite: "F8FAFC",
  bg: "F7F9FC",
  header: "E2E8F0",
  strip: "EAF2FF",
};

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "SolarLab";
pptx.company = "SolarLab";
pptx.subject = "SCAPS-1D input parameter overview";
pptx.title = "SCAPS-1D Input Parameters";
pptx.lang = "en-US";
pptx.theme = { headFontFace: "Arial", bodyFontFace: "Arial", lang: "en-US" };
pptx.margin = 0;
pptx.defineLayout({ name: "CUSTOM_WIDE", width: 13.333, height: 7.5 });
pptx.layout = "CUSTOM_WIDE";

const slide = pptx.addSlide();
slide.background = { color: C.bg };

const textBoxes = [];

function track(txt, x, y, w, h, role = "text") {
  textBoxes.push({ txt: String(txt || ""), x, y, w, h, role });
}

function text(txt, x, y, w, h, opt = {}) {
  track(txt, x, y, w, h, opt.role);
  slide.addText(txt, {
    x, y, w, h,
    margin: opt.margin ?? 0,
    fit: opt.fit || "shrink",
    fontFace: "Arial",
    fontSize: opt.fontSize || 11,
    color: opt.color || C.navy,
    bold: opt.bold || false,
    italic: opt.italic || false,
    align: opt.align || "left",
    valign: opt.valign || "top",
    paraSpaceAfterPt: 0,
    paraSpaceBeforePt: 0,
  });
}

function sub(txt, opt = {}) {
  return { text: txt, options: { subscript: true, ...opt } };
}

function rich(parts, x, y, w, h, opt = {}) {
  const runs = parts.flatMap((part) => (typeof part === "string" ? [{ text: part }] : part));
  const plain = runs.map((r) => r.text || "").join("");
  track(plain, x, y, w, h, opt.role || "rich-text");
  const base = {
    fontFace: "Arial",
    fontSize: opt.fontSize || 11,
    color: opt.color || C.navy,
    bold: opt.bold || false,
    italic: opt.italic || false,
  };
  slide.addText(
    runs.map((run) => ({ text: run.text, options: { ...base, ...(run.options || {}) } })),
    {
      x, y, w, h,
      margin: opt.margin ?? 0,
      fit: opt.fit || "shrink",
      fontFace: "Arial",
      align: opt.align || "left",
      valign: opt.valign || "mid",
      paraSpaceAfterPt: 0,
      paraSpaceBeforePt: 0,
    },
  );
}

function rect(x, y, w, h, fill, line = C.line, opt = {}) {
  slide.addShape(opt.round ? pptx.ShapeType.roundRect : pptx.ShapeType.rect, {
    x, y, w, h,
    rectRadius: opt.round || 0,
    fill: { color: fill, transparency: opt.fillTransparency || 0 },
    line: { color: line, pt: opt.pt || 1, transparency: opt.lineTransparency || 0 },
  });
}

function hline(x1, y, x2, color = C.grid, pt = 0.75) {
  slide.addShape(pptx.ShapeType.line, {
    x: x1, y, w: x2 - x1, h: 0,
    line: { color, pt, beginArrowType: "none", endArrowType: "none" },
  });
}

function vline(x, y1, y2, color = C.grid, pt = 0.75) {
  slide.addShape(pptx.ShapeType.line, {
    x, y: y1, w: 0, h: y2 - y1,
    line: { color, pt, beginArrowType: "none", endArrowType: "none" },
  });
}

function paramCell(parts) {
  return parts;
}

const rows = [
  {
    block: "Cell / stack",
    color: C.blue,
    params: paramCell(["layer order, thickness ", { text: "d", options: { italic: true } }, ", area ", { text: "A", options: { italic: true } }, ", temperature ", { text: "T", options: { italic: true } }]),
    role: "1D geometry and operating temperature",
  },
  {
    block: "Layer material",
    color: C.purple,
    params: paramCell([{ text: "E", options: { italic: true } }, sub("g"), ", χ, ε", sub("r"), ", ", { text: "N", options: { italic: true } }, sub("C"), ", ", { text: "N", options: { italic: true } }, sub("V"), ", ", { text: "v", options: { italic: true } }, sub("th,n"), ", ", { text: "v", options: { italic: true } }, sub("th,p")]),
    role: "band alignment, electrostatics, carrier statistics",
  },
  {
    block: "Transport",
    color: C.teal,
    params: paramCell(["μ", sub("n"), ", μ", sub("p"), ", ", { text: "N", options: { italic: true } }, sub("A"), ", ", { text: "N", options: { italic: true } }, sub("D"), "; grading ", { text: "P", options: { italic: true } }, "(", { text: "x", options: { italic: true } }, ") or ", { text: "P", options: { italic: true } }, "(", { text: "y", options: { italic: true } }, ")"]),
    role: "drift-diffusion mobility and fixed doping",
  },
  {
    block: "Recombination",
    color: C.green,
    params: paramCell([{ text: "B", options: { italic: true } }, sub("rad"), ", ", { text: "C", options: { italic: true } }, sub("n"), "/", { text: "C", options: { italic: true } }, sub("p"), ", bulk defects ", { text: "N", options: { italic: true } }, sub("t"), ", ", { text: "E", options: { italic: true } }, sub("t"), ", σ", sub("n"), "/σ", sub("p")]),
    role: "SRH / radiative / Auger loss channels",
  },
  {
    block: "Interface",
    color: C.amber,
    params: paramCell(["interface defects, tunneling, ", { text: "m", options: { italic: true } }, sub("e"), ", ", { text: "m", options: { italic: true } }, sub("h")]),
    role: "heterojunction recombination and tunneling",
  },
  {
    block: "Contacts",
    color: C.orange,
    params: paramCell(["work function Φ", sub("m"), " or flat-band; ", { text: "S", options: { italic: true } }, sub("n"), ", ", { text: "S", options: { italic: true } }, sub("p"), "; optical filter"]),
    role: "boundary carrier extraction and optical loss",
  },
  {
    block: "Optical input",
    color: C.red,
    params: paramCell(["spectrum, illumination side, ", { text: "R", options: { italic: true } }, "/", { text: "T", options: { italic: true } }, ", absorption α(λ)"]),
    role: "generation profile G(x)",
  },
  {
    block: "Simulation setup",
    color: C.slate,
    params: paramCell(["working point ", { text: "V", options: { italic: true } }, ", ", { text: "f", options: { italic: true } }, ", ", { text: "T", options: { italic: true } }, "; J-V / C-V / C-f / QE ranges; mesh settings"]),
    role: "defines which measurement is solved",
  },
];

text("SCAPS-1D Input Parameters", 0.58, 0.23, 8.6, 0.42, { fontSize: 23.5, bold: true, role: "title" });
text("Main editable inputs used to define a one-dimensional thin-film solar-cell simulation", 0.60, 0.76, 11.9, 0.23, {
  fontSize: 11.7,
  color: C.slate,
  role: "subtitle",
});

const table = { x: 0.56, y: 1.20, w: 12.23, h: 5.42 };
const headerH = 0.50;
const rowH = (table.h - headerH) / rows.length;
const col1 = 1.95;
const col2 = 5.15;
const col3 = table.w - col1 - col2;
const c0 = table.x;
const c1 = c0 + col1;
const c2 = c1 + col2;
const c3 = table.x + table.w;

rect(table.x, table.y, table.w, table.h, C.white, C.line, { round: 0.07, pt: 1.1 });
rect(table.x, table.y, table.w, headerH, C.header, C.header, { lineTransparency: 100 });
text("Input block", c0 + 0.18, table.y + 0.16, col1 - 0.32, 0.18, { fontSize: 12.2, bold: true });
text("Typical parameters", c1 + 0.20, table.y + 0.16, col2 - 0.38, 0.18, { fontSize: 12.2, bold: true });
text("Physical role", c2 + 0.20, table.y + 0.16, col3 - 0.38, 0.18, { fontSize: 12.2, bold: true });

vline(c1, table.y, table.y + table.h);
vline(c2, table.y, table.y + table.h);
hline(table.x, table.y + headerH, table.x + table.w);

rows.forEach((row, idx) => {
  const y = table.y + headerH + idx * rowH;
  if (idx % 2 === 0) {
    rect(table.x, y, table.w, rowH, C.offWhite, C.offWhite, { lineTransparency: 100 });
  }
  rect(table.x, y, 0.075, rowH, row.color, row.color, { lineTransparency: 100 });
  hline(table.x, y + rowH, table.x + table.w);
  text(row.block, c0 + 0.18, y + rowH / 2 - 0.11, col1 - 0.32, 0.22, {
    fontSize: 11.4,
    bold: true,
    color: row.color,
    valign: "mid",
    role: "input-block",
  });
  rich(row.params, c1 + 0.20, y + rowH / 2 - 0.13, col2 - 0.36, 0.26, {
    fontSize: 10.75,
    color: C.navy,
    role: "parameter-list",
  });
  text(row.role, c2 + 0.20, y + rowH / 2 - 0.11, col3 - 0.38, 0.22, {
    fontSize: 10.6,
    color: C.slate,
    valign: "mid",
    role: "physical-role",
  });
});

rect(0, 6.95, 13.333, 0.55, C.strip, C.strip, { lineTransparency: 100 });
rich([
  "SCAPS-1D input = ",
  { text: "device structure", options: { bold: true, color: C.blue } },
  " + ",
  { text: "material/defect physics", options: { bold: true, color: C.green } },
  " + ",
  { text: "contact/optical conditions", options: { bold: true, color: C.orange } },
  " + ",
  { text: "measurement setup", options: { bold: true, color: C.slate } },
], 0.62, 7.10, 12.05, 0.20, {
  fontSize: 11.2,
  bold: true,
  color: C.navy,
  align: "center",
  role: "takeaway",
});

slide.addNotes(`Suggested narration:
This slide compresses the SCAPS-1D input space into the blocks a beginner should recognize first. The cell stack defines the one-dimensional geometry; layer material and transport parameters define band alignment and drift-diffusion behavior; recombination and interface parameters define loss channels; contact and optical inputs define boundary extraction and generation; finally, the simulation setup tells SCAPS which measurement, such as J-V, C-V, C-f, or QE, to solve.`);

function checkLayout() {
  const ignore = new Set(["title", "subtitle", "takeaway"]);
  const overlaps = [];
  function hit(a, b, pad = 0.005) {
    return !(
      a.x + a.w + pad <= b.x ||
      b.x + b.w + pad <= a.x ||
      a.y + a.h + pad <= b.y ||
      b.y + b.h + pad <= a.y
    );
  }
  for (let i = 0; i < textBoxes.length; i += 1) {
    for (let j = i + 1; j < textBoxes.length; j += 1) {
      const a = textBoxes[i];
      const b = textBoxes[j];
      if (ignore.has(a.role) || ignore.has(b.role)) continue;
      if (hit(a, b)) overlaps.push([a.role, a.txt, b.role, b.txt]);
    }
  }
  if (overlaps.length) {
    console.warn("Potential text box overlaps:", overlaps.slice(0, 8));
  }
}

checkLayout();

pptx.writeFile({ fileName: outPath }).then(() => {
  console.log(outPath);
});
