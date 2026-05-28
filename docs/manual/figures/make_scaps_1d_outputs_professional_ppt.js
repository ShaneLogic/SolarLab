const fs = require("fs");
const path = require("path");
const pptxgen = require("/Users/shane/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/pptxgenjs");

const root = path.resolve(__dirname, "../../..");
const outDir = path.join(root, "docs", "manual", "slides");
const outPath = path.join(outDir, "scaps_1d_outputs_professional_examples.pptx");
const legacyOutPath = path.join(outDir, "scaps_1d_outputs_and_figures.pptx");
fs.mkdirSync(outDir, { recursive: true });

const pptx = new pptxgen();
pptx.defineLayout({ name: "CUSTOM_WIDE", width: 13.333, height: 7.5 });
pptx.layout = "CUSTOM_WIDE";
pptx.author = "SolarLab";
pptx.company = "SolarLab";
pptx.subject = "Professional SCAPS-1D result figure examples";
pptx.title = "Representative SCAPS-1D Result Figures";
pptx.lang = "en-US";
pptx.theme = { headFontFace: "Arial", bodyFontFace: "Arial", lang: "en-US" };
pptx.margin = 0;

const C = {
  bg: "F7F9FC",
  white: "FFFFFF",
  navy: "162033",
  slate: "334155",
  muted: "64748B",
  line: "CBD5E1",
  grid: "E2E8F0",
  strip: "EAF2FF",
  blue: "2563EB",
  blueLight: "DBEAFE",
  red: "B42318",
  redLight: "FEE4E2",
  green: "16803C",
  greenLight: "DCFCE7",
  amber: "A16207",
  amberLight: "FEF3C7",
  orange: "EA580C",
  purple: "6D28D9",
  purpleLight: "EDE9FE",
  teal: "0F766E",
  black: "111827",
};

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
    fontSize: opt.fontSize || 9,
    color: opt.color || C.navy,
    bold: opt.bold || false,
    italic: opt.italic || false,
    align: opt.align || "left",
    valign: opt.valign || "top",
    paraSpaceAfterPt: 0,
    paraSpaceBeforePt: 0,
  });
}

function run(txt, options = {}) {
  return { text: txt, options };
}

function sub(txt, options = {}) {
  return { text: txt, options: { subscript: true, ...options } };
}

function sup(txt, options = {}) {
  return { text: txt, options: { superscript: true, ...options } };
}

function rich(parts, x, y, w, h, opt = {}) {
  const base = {
    fontFace: "Arial",
    fontSize: opt.fontSize || 9,
    color: opt.color || C.navy,
    bold: opt.bold || false,
    italic: opt.italic || false,
  };
  const runs = parts.flatMap((p) => (typeof p === "string" ? [{ text: p }] : [p]));
  const plain = runs.map((r) => r.text || "").join("");
  track(plain, x, y, w, h, opt.role || "rich-text");
  slide.addText(
    runs.map((r) => ({ text: r.text, options: { ...base, ...(r.options || {}) } })),
    {
      x, y, w, h,
      margin: opt.margin ?? 0,
      fit: opt.fit || "shrink",
      fontFace: "Arial",
      align: opt.align || "left",
      valign: opt.valign || "top",
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

function badge(txt, x, y, w, color, fill) {
  rect(x, y, w, 0.24, fill, color, { round: 0.06, pt: 1 });
  text(txt, x + 0.05, y + 0.055, w - 0.10, 0.10, {
    fontSize: 6.6,
    bold: true,
    color,
    align: "center",
    role: "badge",
  });
}

function panel(x, y, w, h, tag, title, accent, tint) {
  rect(x, y, w, h, C.white, C.line, { round: 0.08, pt: 1.05 });
  rect(x, y, 0.08, h, accent, accent, { lineTransparency: 100 });
  badge(tag, x + 0.20, y + 0.16, 0.38, accent, tint);
  text(title, x + 0.67, y + 0.17, w - 0.90, 0.20, {
    fontSize: 10.0,
    bold: true,
    color: C.slate,
    role: "panel-title",
  });
  return {
    chart: { x: x + 0.25, y: y + 0.62, w: 3.25, h: h - 0.87 },
    note: { x: x + 3.72, y: y + 0.62, w: w - 3.98, h: h - 0.87 },
  };
}

function addLineChart(box, series, opts = {}) {
  slide.addChart(
    pptx.ChartType.line,
    series.map((s) => ({
      name: s.name,
      labels: opts.labels,
      values: s.values,
    })),
    {
      x: box.x,
      y: box.y,
      w: box.w,
      h: box.h,
      chartColors: series.map((s) => s.color),
      showLegend: false,
      showTitle: false,
      showValue: false,
      lineSize: opts.lineSize || 2.05,
      lineSmooth: opts.lineSmooth ?? true,
      lineDataSymbol: opts.symbol || "none",
      lineDataSymbolSize: 3,
      catAxisHidden: false,
      valAxisHidden: false,
      catAxisLabelFontFace: "Arial",
      valAxisLabelFontFace: "Arial",
      catAxisLabelFontSize: opts.axisFont || 5.6,
      valAxisLabelFontSize: opts.axisFont || 5.6,
      catAxisLabelColor: C.muted,
      valAxisLabelColor: C.muted,
      catAxisLineColor: C.slate,
      valAxisLineColor: C.slate,
      catAxisLineSize: 0.6,
      valAxisLineSize: 0.6,
      catAxisMajorTickMark: "none",
      valAxisMajorTickMark: "none",
      valAxisMinVal: opts.yMin,
      valAxisMaxVal: opts.yMax,
      valAxisMajorUnit: opts.yUnit,
      valAxisLabelFormatCode: opts.yFmt || "General",
      catAxisLabelRotate: opts.rotateX || 0,
      valGridLine: { color: C.grid, size: 0.45, style: "solid" },
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
      layout: { x: 0.10, y: 0.08, w: 0.82, h: 0.78 },
      displayBlanksAs: "span",
    },
  );
}

function legend(items, x, y, w, opt = {}) {
  const step = opt.step || 0.52;
  items.forEach((item, i) => {
    const xx = x + i * step;
    line(xx, y + 0.055, xx + 0.16, y + 0.055, item.color, { pt: 1.6 });
    rich(item.label, xx + 0.19, y, w / items.length - 0.12, 0.13, {
      fontSize: 5.9,
      color: C.slate,
      role: "legend",
    });
  });
}

function metricCard(x, y, labelParts, value, color) {
  rect(x, y, 0.98, 0.36, "F8FAFC", C.line, { round: 0.04, pt: 0.75 });
  rich(labelParts, x + 0.07, y + 0.055, 0.33, 0.11, {
    fontSize: 6.0,
    color: C.muted,
    role: "metric-label",
  });
  text(value, x + 0.41, y + 0.050, 0.50, 0.13, {
    fontSize: 7.0,
    bold: true,
    color,
    align: "right",
    role: "metric-value",
  });
}

function bullet(txt, x, y, w, opt = {}) {
  rect(x, y + 0.045, 0.035, 0.035, opt.color || C.muted, opt.color || C.muted, { lineTransparency: 100 });
  text(txt, x + 0.10, y, w - 0.10, opt.h || 0.17, {
    fontSize: opt.fontSize || 6.7,
    color: opt.textColor || C.slate,
    role: "bullet",
  });
}

text("Representative SCAPS-1D Result Figures", 0.48, 0.23, 8.55, 0.34, {
  fontSize: 21.0,
  bold: true,
  role: "title",
});
text("A professional report should connect terminal metrics to spectral collection, band alignment, and spatial loss mechanisms.", 0.50, 0.70, 11.95, 0.22, {
  fontSize: 10.4,
  color: C.slate,
  role: "subtitle",
});
badge("editable", 11.60, 0.30, 0.82, C.blue, C.blueLight);
badge("SCAPS-1D", 12.48, 0.30, 0.70, C.slate, "EEF2F7");

const pA = panel(0.50, 1.12, 6.00, 2.58, "A", "Terminal J-V and device metrics", C.red, C.redLight);
addLineChart(pA.chart, [
  { name: "Light", color: C.red, values: [24.1, 24.0, 23.8, 23.3, 22.2, 18.4, 9.2, 0.0, -4.4] },
  { name: "Dark × 10", color: C.slate, values: [0.0, 0.0, 0.1, 0.3, 0.9, 2.2, 5.8, 13.0, 24.0] },
], {
  labels: ["0", "0.2", "0.4", "0.6", "0.8", "1.0", "1.1", "1.2", "1.3"],
  yMin: -5,
  yMax: 26,
  yUnit: 5,
  yFmt: "0",
  lineSmooth: false,
});
rich([run("Voltage "), run("V", { italic: true }), run(" (V)")], pA.chart.x + 1.20, pA.chart.y + pA.chart.h - 0.08, 0.90, 0.12, {
  fontSize: 6.2,
  color: C.muted,
  align: "center",
  role: "axis-label",
});
rich([run("J", { italic: true }), run(" (mA cm"), sup("−2"), run(")")], pA.chart.x - 0.04, pA.chart.y - 0.02, 0.82, 0.12, {
  fontSize: 6.2,
  color: C.muted,
  role: "axis-label",
});
legend([
  { color: C.red, label: ["light"] },
  { color: C.slate, label: ["dark × 10"] },
], pA.chart.x + 0.06, pA.chart.y + pA.chart.h - 0.23, 1.7, { step: 0.75 });
metricCard(pA.note.x, pA.note.y, [run("V", { italic: true }), sub("oc")], "1.18 V", C.red);
metricCard(pA.note.x + 1.12, pA.note.y, [run("J", { italic: true }), sub("sc")], "24.1", C.red);
metricCard(pA.note.x, pA.note.y + 0.46, ["FF"], "78.6%", C.red);
metricCard(pA.note.x + 1.12, pA.note.y + 0.46, ["η"], "22.3%", C.red);
bullet("Primary performance figure; report scan direction and illumination.", pA.note.x, pA.note.y + 1.02, pA.note.w, { color: C.red, h: 0.25 });
bullet("Dark curve diagnoses diode quality, shunt leakage, and series resistance.", pA.note.x, pA.note.y + 1.34, pA.note.w, { color: C.slate, h: 0.32 });

const pB = panel(6.84, 1.12, 6.00, 2.58, "B", "Spectral response: EQE / QE", C.purple, C.purpleLight);
addLineChart(pB.chart, [
  { name: "EQE", color: C.purple, values: [3, 34, 72, 86, 88, 83, 70, 24, 1] },
], {
  labels: ["300", "400", "500", "600", "700", "800", "850", "900", "950"],
  yMin: 0,
  yMax: 100,
  yUnit: 20,
  yFmt: "0",
});
rich(["Wavelength λ (nm)"], pB.chart.x + 1.05, pB.chart.y + pB.chart.h - 0.08, 1.08, 0.12, {
  fontSize: 6.2,
  color: C.muted,
  align: "center",
  role: "axis-label",
});
rich(["EQE (%)"], pB.chart.x - 0.02, pB.chart.y - 0.02, 0.55, 0.12, {
  fontSize: 6.2,
  color: C.muted,
  role: "axis-label",
});
legend([{ color: C.purple, label: ["external quantum efficiency"] }], pB.chart.x + 0.08, pB.chart.y + pB.chart.h - 0.22, 2.2, { step: 0.60 });
rect(pB.note.x, pB.note.y, 2.00, 0.52, "F8FAFC", C.line, { round: 0.04, pt: 0.75 });
rich([run("Integrated "), run("J", { italic: true }), sub("sc,EQE"), run(" ≈ 23.8 mA cm"), sup("−2")], pB.note.x + 0.10, pB.note.y + 0.13, 1.80, 0.14, {
  fontSize: 7.0,
  bold: true,
  color: C.purple,
  role: "metric-value",
});
bullet("Short-wavelength drop: front-layer absorption or surface recombination.", pB.note.x, pB.note.y + 0.72, pB.note.w, { color: C.purple, h: 0.30 });
bullet("Long-wavelength roll-off: absorber band gap and optical thickness.", pB.note.x, pB.note.y + 1.08, pB.note.w, { color: C.purple, h: 0.30 });
bullet("Use with optical constants to separate reflection and collection losses.", pB.note.x, pB.note.y + 1.44, pB.note.w, { color: C.slate, h: 0.30 });

const pC = panel(0.50, 4.02, 6.00, 2.48, "C", "Band diagram and quasi-Fermi levels", C.blue, C.blueLight);
addLineChart(pC.chart, [
  { name: "EC", color: C.amber, values: [-3.78, -3.82, -3.92, -3.98, -4.03, -4.12, -4.18, -4.22] },
  { name: "EV", color: C.purple, values: [-5.42, -5.46, -5.54, -5.60, -5.65, -5.70, -5.76, -5.82] },
  { name: "EFn", color: C.blue, values: [-4.05, -4.04, -4.03, -4.02, -4.00, -3.98, -3.96, -3.95] },
  { name: "EFp", color: C.red, values: [-5.20, -5.18, -5.14, -5.10, -5.05, -5.00, -4.96, -4.92] },
], {
  labels: ["0", "100", "200", "300", "400", "500", "600", "700"],
  yMin: -6.0,
  yMax: -3.5,
  yUnit: 0.5,
  yFmt: "0.0",
  lineSmooth: true,
});
rich(["Position ", run("x", { italic: true }), " (nm)"], pC.chart.x + 1.08, pC.chart.y + pC.chart.h - 0.08, 0.95, 0.12, {
  fontSize: 6.2,
  color: C.muted,
  align: "center",
  role: "axis-label",
});
rich(["Energy relative to ", run("E", { italic: true }), sub("vac"), " (eV)"], pC.chart.x - 0.04, pC.chart.y - 0.02, 1.25, 0.12, {
  fontSize: 6.2,
  color: C.muted,
  role: "axis-label",
});
legend([
  { color: C.amber, label: [run("E", { italic: true }), sub("C")] },
  { color: C.purple, label: [run("E", { italic: true }), sub("V")] },
  { color: C.blue, label: [run("E", { italic: true }), sub("Fn")] },
  { color: C.red, label: [run("E", { italic: true }), sub("Fp")] },
], pC.chart.x + 0.03, pC.chart.y + pC.chart.h - 0.23, 3.0, { step: 0.68 });
bullet("Band offsets identify transport barriers at ETL/absorber/HTL interfaces.", pC.note.x, pC.note.y, pC.note.w, { color: C.blue, h: 0.34 });
bullet("Quasi-Fermi-level splitting under illumination relates directly to voltage.", pC.note.x, pC.note.y + 0.42, pC.note.w, { color: C.blue, h: 0.36 });
bullet("A flat or kinked quasi-Fermi level can reveal extraction bottlenecks.", pC.note.x, pC.note.y + 0.88, pC.note.w, { color: C.slate, h: 0.36 });
rich([run("Report at a fixed bias, e.g. "), run("V", { italic: true }), " = 0.9 V"], pC.note.x, pC.note.y + 1.48, pC.note.w, 0.14, {
  fontSize: 6.7,
  color: C.muted,
  italic: true,
  role: "figure-note",
});

const pD = panel(6.84, 4.02, 6.00, 2.48, "D", "Loss localization: generation and recombination", C.green, C.greenLight);
addLineChart(pD.chart, [
  { name: "G", color: C.teal, values: [1.00, 0.78, 0.61, 0.48, 0.38, 0.30, 0.24, 0.19] },
  { name: "RSRH", color: C.red, values: [0.05, 0.10, 0.18, 0.22, 0.20, 0.17, 0.14, 0.10] },
  { name: "Rinterface", color: C.orange, values: [0.04, 0.08, 0.18, 0.75, 0.26, 0.12, 0.08, 0.05] },
], {
  labels: ["0", "100", "200", "300", "400", "500", "600", "700"],
  yMin: 0,
  yMax: 1.05,
  yUnit: 0.25,
  yFmt: "0.00",
});
rich(["Position ", run("x", { italic: true }), " (nm)"], pD.chart.x + 1.08, pD.chart.y + pD.chart.h - 0.08, 0.95, 0.12, {
  fontSize: 6.2,
  color: C.muted,
  align: "center",
  role: "axis-label",
});
rich(["Normalized rate"], pD.chart.x - 0.01, pD.chart.y - 0.02, 0.72, 0.12, {
  fontSize: 6.2,
  color: C.muted,
  role: "axis-label",
});
legend([
  { color: C.teal, label: [run("G", { italic: true })] },
  { color: C.red, label: [run("R", { italic: true }), sub("SRH")] },
  { color: C.orange, label: [run("R", { italic: true }), sub("interface")] },
], pD.chart.x + 0.05, pD.chart.y + pD.chart.h - 0.23, 2.8, { step: 0.76 });
bullet("Compare where photons create carriers with where carriers are lost.", pD.note.x, pD.note.y, pD.note.w, { color: C.green, h: 0.32 });
bullet("Sharp interface peak suggests defect-rich or misaligned junction loss.", pD.note.x, pD.note.y + 0.40, pD.note.w, { color: C.orange, h: 0.34 });
bullet("Pair with current profiles to check whether loss is transport- or recombination-limited.", pD.note.x, pD.note.y + 0.86, pD.note.w, { color: C.slate, h: 0.44 });
rect(pD.note.x, pD.note.y + 1.50, pD.note.w, 0.25, C.greenLight, C.green, { round: 0.04, pt: 0.75 });
rich([run("Use for: "), run("loss diagnosis", { bold: true }), run(" and "), run("stack optimization", { bold: true })], pD.note.x + 0.10, pD.note.y + 1.565, pD.note.w - 0.20, 0.09, {
  fontSize: 6.5,
  color: C.green,
  align: "center",
  role: "callout",
});

rect(0, 6.90, 13.333, 0.60, C.strip, C.strip, { lineTransparency: 100 });
rich([
  run("Takeaway: ", { bold: true }),
  run("SCAPS-1D figures should link "),
  run("terminal metrics", { bold: true, color: C.red }),
  run(" to "),
  run("optical collection", { bold: true, color: C.purple }),
  run(", "),
  run("band alignment", { bold: true, color: C.blue }),
  run(", and "),
  run("recombination-loss localization", { bold: true, color: C.green }),
  run("."),
], 0.64, 7.08, 12.05, 0.18, {
  fontSize: 10.5,
  color: C.navy,
  align: "center",
  role: "takeaway",
});

slide.addNotes(`Suggested narration:
This slide shows professional SCAPS-1D result figures rather than generic curve sketches. The J-V panel reports terminal performance and diagnostic dark behavior. The EQE panel shows spectral collection and integrated current. The band diagram connects material parameters and interface band offsets to extraction barriers and quasi-Fermi-level splitting. The generation/recombination panel localizes where carriers are created and where they are lost, which is essential for identifying whether the device is limited by optics, transport, bulk recombination, or interface recombination.`);

function checkLayout() {
  const ignore = new Set([
    "title",
    "subtitle",
    "badge",
    "axis-label",
    "legend",
    "takeaway",
    "metric-label",
    "metric-value",
    "figure-note",
  ]);
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
  if (overlaps.length) {
    console.warn("Potential text overlaps:", JSON.stringify(overlaps.slice(0, 12), null, 2));
  }
}

checkLayout();
pptx.writeFile({ fileName: outPath }).then(() => {
  fs.copyFileSync(outPath, legacyOutPath);
  console.log(outPath);
  console.log(legacyOutPath);
});
