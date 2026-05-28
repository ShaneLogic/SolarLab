const fs = require("fs");
const path = require("path");
const pptxgen = require("/Users/shane/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/pptxgenjs");

const root = path.resolve(__dirname, "../../..");
const outDir = path.join(root, "docs", "manual", "slides");
const outPath = path.join(outDir, "scaps_1d_outputs_and_figures.pptx");
fs.mkdirSync(outDir, { recursive: true });

const C = {
  navy: "162033",
  slate: "334155",
  muted: "64748B",
  line: "CBD5E1",
  grid: "E2E8F0",
  blue: "2563EB",
  blueLight: "DBEAFE",
  green: "16803C",
  greenLight: "DCFCE7",
  teal: "0F766E",
  amber: "A16207",
  amberLight: "FEF3C7",
  orange: "B45309",
  red: "B42318",
  redLight: "FEE4E2",
  purple: "6D28D9",
  purpleLight: "EDE9FE",
  white: "FFFFFF",
  offWhite: "F8FAFC",
  bg: "F7F9FC",
  strip: "EAF2FF",
};

const pptx = new pptxgen();
pptx.defineLayout({ name: "CUSTOM_WIDE", width: 13.333, height: 7.5 });
pptx.layout = "CUSTOM_WIDE";
pptx.author = "SolarLab";
pptx.company = "SolarLab";
pptx.subject = "SCAPS-1D output result overview";
pptx.title = "SCAPS-1D Outputs and Figures";
pptx.lang = "en-US";
pptx.theme = { headFontFace: "Arial", bodyFontFace: "Arial", lang: "en-US" };
pptx.margin = 0;

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
    fontSize: opt.fontSize || 10,
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
  const runs = parts.flatMap((p) => (typeof p === "string" ? [{ text: p }] : p));
  const plain = runs.map((r) => r.text || "").join("");
  track(plain, x, y, w, h, opt.role || "rich-text");
  const base = {
    fontFace: "Arial",
    fontSize: opt.fontSize || 9,
    color: opt.color || C.navy,
    bold: opt.bold || false,
    italic: opt.italic || false,
  };
  slide.addText(
    runs.map((r) => ({ text: r.text, options: { ...base, ...(r.options || {}) } })),
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

function line(x1, y1, x2, y2, color = C.muted, opt = {}) {
  slide.addShape(pptx.ShapeType.line, {
    x: x1,
    y: y1,
    w: x2 - x1,
    h: y2 - y1,
    line: {
      color,
      pt: opt.pt || 1,
      dash: opt.dash || "solid",
      beginArrowType: opt.beginArrow || "none",
      endArrowType: opt.endArrow || "none",
      transparency: opt.transparency || 0,
    },
  });
}

function plotSegments(box, points, color, opt = {}) {
  for (let i = 0; i < points.length - 1; i += 1) {
    const [x1, y1] = points[i];
    const [x2, y2] = points[i + 1];
    line(
      box.x + x1 * box.w,
      box.y + (1 - y1) * box.h,
      box.x + x2 * box.w,
      box.y + (1 - y2) * box.h,
      color,
      { pt: opt.pt || 1.3, dash: opt.dash || "solid" },
    );
  }
}

function chartFrame(x, y, w, h, title, opt = {}) {
  rect(x, y, w, h, C.white, C.line, { round: 0.06, pt: 1.0 });
  text(title, x + 0.13, y + 0.12, w - 0.26, 0.16, {
    fontSize: 8.4,
    bold: true,
    color: C.slate,
    role: "chart-title",
  });
  const box = { x: x + 0.34, y: y + 0.42, w: w - 0.58, h: h - 0.74 };
  line(box.x, box.y + box.h, box.x + box.w, box.y + box.h, C.slate, { pt: 0.8 });
  line(box.x, box.y + box.h, box.x, box.y, C.slate, { pt: 0.8 });
  text(opt.xlabel || "x", box.x + box.w - 0.18, box.y + box.h + 0.06, 0.25, 0.10, {
    fontSize: 5.9,
    color: C.muted,
    align: "right",
    role: "axis-label",
  });
  text(opt.ylabel || "", box.x - 0.25, box.y - 0.02, 0.22, 0.11, {
    fontSize: 5.9,
    color: C.muted,
    align: "right",
    role: "axis-label",
  });
  return box;
}

function addMiniChart(x, y, w, h, title, series, opt = {}) {
  rect(x, y, w, h, C.white, C.line, { round: 0.06, pt: 1.0 });
  text(title, x + 0.13, y + 0.12, w - 0.26, 0.16, {
    fontSize: 8.4,
    bold: true,
    color: C.slate,
    role: "chart-title",
  });
  const labels = opt.labels || ["0", "", "", "", "", "1"];
  slide.addChart(
    pptx.ChartType.line,
    series.map((s) => ({
      name: s.name,
      labels,
      values: s.values,
    })),
    {
      x: x + 0.25,
      y: y + 0.40,
      w: w - 0.45,
      h: h - 0.62,
      chartColors: series.map((s) => s.color),
      showLegend: false,
      showTitle: false,
      showValue: false,
      lineSize: opt.lineSize || 2.25,
      lineSmooth: true,
      lineDataSymbol: "none",
      catAxisHidden: true,
      valAxisHidden: true,
      catAxisLineShow: false,
      valAxisLineShow: false,
      valAxisMinVal: opt.yMin ?? 0,
      valAxisMaxVal: opt.yMax ?? 1,
      valGridLine: { style: "none" },
      catGridLine: { style: "none" },
      chartArea: {
        fill: { color: C.white, transparency: 100 },
        border: { color: C.white, transparency: 100 },
        roundedCorners: false,
      },
      plotArea: {
        fill: { color: C.white, transparency: 100 },
        border: { color: C.white, transparency: 100 },
      },
      layout: { x: 0.02, y: 0.02, w: 0.94, h: 0.90 },
    },
  );
  text(opt.xlabel || "", x + w - 0.34, y + h - 0.16, 0.25, 0.10, {
    fontSize: 5.9,
    color: C.muted,
    align: "right",
    role: "axis-label",
  });
  text(opt.ylabel || "", x + 0.10, y + 0.35, 0.24, 0.10, {
    fontSize: 5.9,
    color: C.muted,
    role: "axis-label",
  });
}

function outputCard(x, y, w, h, title, color, bodyParts, role) {
  rect(x, y, w, h, C.white, C.line, { round: 0.07, pt: 0.9 });
  rect(x, y, 0.07, h, color, color, { lineTransparency: 100 });
  text(title, x + 0.17, y + 0.11, w - 0.30, 0.15, {
    fontSize: 8.7,
    bold: true,
    color,
    role: "card-title",
  });
  rich(bodyParts, x + 0.17, y + 0.34, w - 0.30, 0.22, {
    fontSize: 7.25,
    color: C.navy,
    role: "output-variables",
  });
  text(role, x + 0.17, y + 0.63, w - 0.30, h - 0.70, {
    fontSize: 7.05,
    color: C.slate,
    role: "card-body",
  });
}

text("SCAPS-1D Outputs and Typical Figures", 0.48, 0.23, 9.8, 0.38, {
  fontSize: 22.0,
  bold: true,
  role: "title",
});
text("Result outputs can be grouped as internal physics profiles, terminal device curves, spectroscopy, and optical response.", 0.50, 0.72, 12.10, 0.24, {
  fontSize: 11.2,
  color: C.slate,
  role: "subtitle",
});

rect(0.45, 1.08, 12.43, 5.72, C.white, C.line, { round: 0.08, pt: 1.1 });
text("A. Output result families", 0.72, 1.26, 4.9, 0.18, {
  fontSize: 10.8,
  bold: true,
  color: C.slate,
  role: "section-title",
});
text("B. Figure examples", 6.16, 1.26, 5.9, 0.18, {
  fontSize: 10.8,
  bold: true,
  color: C.slate,
  role: "section-title",
});

const leftX = 0.70;
const cardW = 5.10;
const cardH = 1.06;
const gap = 0.18;
const y0 = 1.56;

outputCard(
  leftX,
  y0,
  cardW,
  cardH,
  "Working-point profiles",
  C.blue,
  [{ text: "E", options: { italic: true } }, sub("C"), ", ", { text: "E", options: { italic: true } }, sub("V"), ", ", { text: "E", options: { italic: true } }, sub("Fn"), ", ", { text: "E", options: { italic: true } }, sub("Fp"), ", n, p, ρ, E, ", { text: "J", options: { italic: true } }, sub("n"), ", ", { text: "J", options: { italic: true } }, sub("p"), ", G, R"],
  "Energy bands, quasi-Fermi levels, carrier densities, field, current density, generation and recombination versus position.",
);
outputCard(
  leftX,
  y0 + cardH + gap,
  cardW,
  cardH,
  "J-V and performance metrics",
  C.red,
  [{ text: "J", options: { italic: true } }, "(", { text: "V", options: { italic: true } }, "), ", { text: "J", options: { italic: true } }, sub("bulk"), ", ", { text: "J", options: { italic: true } }, sub("ifr"), ", ", { text: "J", options: { italic: true } }, sub("SRH"), ", ", { text: "V", options: { italic: true } }, sub("oc"), ", ", { text: "J", options: { italic: true } }, sub("sc"), ", FF, η"],
  "Terminal current-voltage behavior and loss-current decomposition; illuminated runs report photovoltaic figures of merit.",
);
outputCard(
  leftX,
  y0 + 2 * (cardH + gap),
  cardW,
  cardH,
  "Admittance and capacitance",
  C.green,
  ["C(", { text: "V", options: { italic: true } }, "), G(", { text: "V", options: { italic: true } }, "), W, ", { text: "N", options: { italic: true } }, sub("app"), ", C(", { text: "f", options: { italic: true } }, "), G(", { text: "f", options: { italic: true } }, "), Z, ", { text: "E", options: { italic: true } }, sub("t"), ", ", { text: "N", options: { italic: true } }, sub("t")],
  "C-V, C-f, conductance, impedance, Mott-Schottky, apparent doping and admittance spectroscopy outputs.",
);
outputCard(
  leftX,
  y0 + 3 * (cardH + gap),
  cardW,
  cardH,
  "QE and optical response",
  C.purple,
  ["QE(λ), photon energy hν, generation profile G(", { text: "x", options: { italic: true } }, "), band-gap estimate ", { text: "E", options: { italic: true } }, sub("g")],
  "Spectral collection response versus wavelength or photon energy; useful for optical loss and band-gap interpretation.",
);

addMiniChart(6.15, 1.55, 2.88, 2.10, "Energy-band panel", [
  { name: "Ec", values: [0.82, 0.77, 0.73, 0.70, 0.63, 0.56], color: C.amber },
  { name: "Ev", values: [0.24, 0.23, 0.22, 0.20, 0.18, 0.16], color: C.purple },
  { name: "EFn", values: [0.58, 0.57, 0.56, 0.55, 0.53, 0.52], color: C.blue },
  { name: "EFp", values: [0.42, 0.41, 0.40, 0.39, 0.37, 0.36], color: C.red },
], { xlabel: "x", ylabel: "E", yMin: 0, yMax: 1 });
rich([{ text: "E", options: { italic: true, color: C.amber, bold: true } }, sub("C", { color: C.amber, bold: true })], 8.57, 1.94, 0.24, 0.10, { fontSize: 5.9, color: C.amber });
rich([{ text: "E", options: { italic: true, color: C.purple, bold: true } }, sub("V", { color: C.purple, bold: true })], 8.57, 3.17, 0.24, 0.10, { fontSize: 5.9, color: C.purple });

addMiniChart(9.35, 1.55, 2.88, 2.10, "J-V curve and metrics", [
  { name: "light", values: [0.72, 0.71, 0.67, 0.55, 0.18, -0.08], color: C.red },
  { name: "dark", values: [0.44, 0.43, 0.44, 0.51, 0.68, 0.90], color: C.slate },
], { xlabel: "V", ylabel: "J", yMin: -0.15, yMax: 1.0 });
rich([{ text: "V", options: { italic: true } }, sub("oc")], 11.38, 3.18, 0.30, 0.10, { fontSize: 5.9, color: C.muted });
rich([{ text: "J", options: { italic: true } }, sub("sc")], 9.65, 2.42, 0.30, 0.10, { fontSize: 5.9, color: C.muted });

addMiniChart(6.15, 4.02, 2.88, 2.10, "C-V / Mott-Schottky", [
  { name: "C", values: [0.78, 0.69, 0.58, 0.46, 0.35, 0.28], color: C.blue },
  { name: "MS", values: [0.20, 0.28, 0.40, 0.53, 0.67, 0.80], color: C.green },
], { xlabel: "V", ylabel: "C", yMin: 0, yMax: 1 });
text("C", 8.48, 4.98, 0.16, 0.10, { fontSize: 5.9, color: C.blue, bold: true, role: "plot-label" });
rich(["1/C", { text: "2", options: { superscript: true, color: C.green, bold: true } }], 8.35, 4.55, 0.32, 0.10, { fontSize: 5.9, color: C.green, bold: true });

addMiniChart(9.35, 4.02, 2.88, 2.10, "QE spectrum", [
  { name: "QE", values: [0.08, 0.42, 0.78, 0.83, 0.72, 0.04], color: C.purple },
], { xlabel: "λ", ylabel: "QE", yMin: 0, yMax: 1 });
rich(["band edge / ", { text: "E", options: { italic: true } }, sub("g")], 10.77, 5.02, 0.85, 0.10, { fontSize: 5.8, color: C.muted });

text("schematic examples; exact curves depend on device definition and working point", 6.22, 6.34, 5.95, 0.16, {
  fontSize: 7.0,
  color: C.muted,
  italic: true,
  align: "center",
  role: "figure-note",
});

rect(0, 6.93, 13.333, 0.57, C.strip, C.strip, { lineTransparency: 100 });
rich([
  "Takeaway: SCAPS-1D outputs include ",
  { text: "spatial profiles", options: { bold: true, color: C.blue } },
  ", ",
  { text: "J-V metrics", options: { bold: true, color: C.red } },
  ", ",
  { text: "capacitance/admittance spectra", options: { bold: true, color: C.green } },
  ", and ",
  { text: "QE optical response", options: { bold: true, color: C.purple } },
  ".",
], 0.64, 7.09, 12.05, 0.22, {
  fontSize: 11.0,
  bold: true,
  align: "center",
  color: C.navy,
  role: "takeaway",
});

slide.addNotes(`Suggested narration:
SCAPS-1D does not only output a J-V curve. At a selected working point, it can save internal state profiles such as band edges, quasi-Fermi levels, carrier densities, electric field, currents, generation and recombination. For terminal device analysis, the IV panel gives current-density curves, recombination-current components, and photovoltaic metrics such as Voc, Jsc, FF and efficiency. It also supports capacitance and admittance outputs, including C-V, C-f, Mott-Schottky, apparent doping, impedance and admittance-spectroscopy quantities. Finally, the QE panel gives spectral quantum-efficiency curves versus wavelength or photon energy.`);

function checkLayout() {
  const ignore = new Set(["title", "subtitle", "takeaway", "axis-label", "plot-label"]);
  const overlaps = [];
  function hit(a, b, pad = 0.006) {
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
  if (overlaps.length) console.warn("Potential text overlaps:", overlaps.slice(0, 10));
}

checkLayout();
pptx.writeFile({ fileName: outPath }).then(() => console.log(outPath));
