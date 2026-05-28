const fs = require("fs");
const path = require("path");
const pptxgen = require("/Users/shane/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/pptxgenjs");

const root = path.resolve(__dirname, "../../..");
const outDir = path.join(root, "docs", "manual", "slides");
const outPath = path.join(outDir, "scaps_1d_main_outputs_four_panels.pptx");
const legacyOutPath = path.join(outDir, "scaps_1d_outputs_and_figures.pptx");
fs.mkdirSync(outDir, { recursive: true });

const pptx = new pptxgen();
pptx.defineLayout({ name: "CUSTOM_WIDE", width: 13.333, height: 7.5 });
pptx.layout = "CUSTOM_WIDE";
pptx.author = "SolarLab";
pptx.company = "SolarLab";
pptx.subject = "SCAPS-1D main output modules";
pptx.title = "Main Output Modules in SCAPS-1D";
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
  purple: "6D28D9",
  purpleLight: "EDE9FE",
  teal: "0F766E",
  tealLight: "CCFBF1",
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

function badge(txt, x, y, w, color, fill, opt = {}) {
  rect(x, y, w, 0.24, fill, color, { round: 0.06, pt: 1 });
  text(txt, x + 0.05, y + 0.052, w - 0.10, 0.10, {
    fontSize: opt.fontSize || 6.6,
    bold: true,
    color,
    align: "center",
    role: "badge",
  });
}

function panel(x, y, w, h, tag, title, accent, tint) {
  rect(x, y, w, h, C.white, C.line, { round: 0.08, pt: 1.05 });
  rect(x, y, 0.08, h, accent, accent, { lineTransparency: 100 });
  badge(tag, x + 0.18, y + 0.15, 0.38, accent, tint);
  text(title, x + 0.66, y + 0.16, w - 0.86, 0.20, {
    fontSize: 10.0,
    bold: true,
    color: C.slate,
    role: "panel-title",
  });
  return {
    chart: { x: x + 0.24, y: y + 0.58, w: 3.10, h: h - 0.84 },
    note: { x: x + 3.62, y: y + 0.58, w: w - 3.86, h: h - 0.84 },
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
      catAxisLabelFontSize: opts.axisFont || 5.4,
      valAxisLabelFontSize: opts.axisFont || 5.4,
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

function legend(items, x, y, opt = {}) {
  const step = opt.step || 0.72;
  items.forEach((item, i) => {
    const xx = x + i * step;
    line(xx, y + 0.055, xx + 0.16, y + 0.055, item.color, { pt: 1.55, dash: item.dash || "solid" });
    rich(item.label, xx + 0.19, y, item.w || 0.62, 0.13, {
      fontSize: 5.8,
      color: C.slate,
      role: "legend",
    });
  });
}

function stackedLegend(items, x, y, w, opt = {}) {
  text(opt.title || "Series", x, y, w, 0.10, {
    fontSize: opt.titleSize || 5.9,
    bold: true,
    color: C.muted,
    role: "legend",
  });
  const rowH = opt.rowH || 0.17;
  items.forEach((item, i) => {
    const yy = y + 0.16 + i * rowH;
    line(x, yy + 0.055, x + 0.20, yy + 0.055, item.color, {
      pt: 1.65,
      dash: item.dash || "solid",
    });
    rich(item.label, x + 0.27, yy, w - 0.27, 0.11, {
      fontSize: opt.fontSize || 6.0,
      color: C.slate,
      role: "legend",
    });
  });
}

function chartOverlayLegend(items, x, y, w, opt = {}) {
  const rowH = opt.rowH || 0.145;
  const h = 0.13 + rowH * items.length;
  rect(x, y, w, h, C.white, opt.border || C.line, {
    round: 0.035,
    pt: 0.55,
    fillTransparency: opt.fillTransparency ?? 6,
  });
  if (opt.title) {
    text(opt.title, x + 0.06, y + 0.035, w - 0.12, 0.08, {
      fontSize: opt.titleSize || 5.2,
      bold: true,
      color: C.muted,
      role: "legend",
    });
  }
  const y0 = y + (opt.title ? 0.135 : 0.065);
  items.forEach((item, i) => {
    const yy = y0 + i * rowH;
    line(x + 0.08, yy + 0.045, x + 0.27, yy + 0.045, item.color, {
      pt: 1.45,
      dash: item.dash || "solid",
    });
    rich(item.label, x + 0.32, yy - 0.004, w - 0.38, 0.09, {
      fontSize: opt.fontSize || 5.35,
      color: C.slate,
      role: "legend",
    });
  });
}

function bullet(parts, x, y, w, opt = {}) {
  rect(x, y + 0.045, 0.035, 0.035, opt.color || C.muted, opt.color || C.muted, { lineTransparency: 100 });
  if (Array.isArray(parts)) {
    rich(parts, x + 0.10, y, w - 0.10, opt.h || 0.18, {
      fontSize: opt.fontSize || 6.7,
      color: opt.textColor || C.slate,
      role: "bullet",
    });
  } else {
    text(parts, x + 0.10, y, w - 0.10, opt.h || 0.18, {
      fontSize: opt.fontSize || 6.7,
      color: opt.textColor || C.slate,
      role: "bullet",
    });
  }
}

function miniMetric(x, y, labelParts, value, color) {
  rect(x, y, 0.90, 0.31, "F8FAFC", C.line, { round: 0.04, pt: 0.7 });
  rich(labelParts, x + 0.06, y + 0.046, 0.34, 0.10, {
    fontSize: 5.8,
    color: C.muted,
    role: "metric-label",
  });
  text(value, x + 0.40, y + 0.043, 0.43, 0.11, {
    fontSize: 6.4,
    bold: true,
    color,
    align: "right",
    role: "metric-value",
  });
}

text("SCAPS-1D Main Output Modules", 0.48, 0.22, 7.75, 0.34, {
  fontSize: 21.0,
  bold: true,
  role: "title",
});
text("Four primary characterization outputs used to evaluate solar-cell performance, junction electrostatics, defect response, and spectral collection.", 0.50, 0.69, 11.95, 0.22, {
  fontSize: 10.3,
  color: C.slate,
  role: "subtitle",
});
badge("editable", 11.54, 0.29, 0.82, C.blue, C.blueLight);
badge("SCAPS-1D", 12.42, 0.29, 0.74, C.slate, "EEF2F7", { fontSize: 6.3 });

const pA = panel(0.50, 1.10, 6.00, 2.55, "A", "I-V / J-V characteristics", C.red, C.redLight);
addLineChart(pA.chart, [
  { name: "light", color: C.red, values: [23.8, 23.7, 23.2, 21.7, 17.2, 8.4, 0.0, -6.8] },
  { name: "dark", color: C.slate, values: [0.0, 0.0, 0.1, 0.4, 1.5, 5.4, 15.0, 26.0] },
], {
  labels: ["0", "0.2", "0.4", "0.6", "0.8", "1.0", "1.15", "1.3"],
  yMin: -8,
  yMax: 28,
  yUnit: 6,
  yFmt: "0",
  lineSmooth: false,
});
rich([run("Voltage "), run("V", { italic: true }), run(" (V)")], pA.chart.x + 1.06, pA.chart.y + pA.chart.h - 0.08, 0.86, 0.12, {
  fontSize: 6.1,
  color: C.muted,
  align: "center",
  role: "axis-label",
});
rich([run("Current density "), run("J", { italic: true }), run(" (mA cm"), sup("−2"), run(")")], pA.chart.x - 0.03, pA.chart.y - 0.02, 1.40, 0.12, {
  fontSize: 6.1,
  color: C.muted,
  role: "axis-label",
});
miniMetric(pA.note.x, pA.note.y, [run("V", { italic: true }), sub("oc")], "1.15 V", C.red);
miniMetric(pA.note.x + 1.01, pA.note.y, [run("J", { italic: true }), sub("sc")], "23.8", C.red);
miniMetric(pA.note.x, pA.note.y + 0.39, ["FF"], "78%", C.red);
miniMetric(pA.note.x + 1.01, pA.note.y + 0.39, ["η"], "21.3%", C.red);
chartOverlayLegend([
  { color: C.red, label: ["light ", run("J", { italic: true }), "-", run("V", { italic: true })] },
  { color: C.slate, label: ["dark ", run("J", { italic: true }), "-", run("V", { italic: true })] },
], pA.chart.x + 0.46, pA.chart.y + 0.72, 0.98, { border: C.red });
bullet("SCAPS core terminal output: light/dark curves and photovoltaic metrics.", pA.note.x, pA.note.y + 0.88, pA.note.w, { color: C.red, h: 0.34 });
bullet("I = J × area; sign convention controls whether the curve appears upward or downward.", pA.note.x, pA.note.y + 1.28, pA.note.w, { color: C.slate, h: 0.42 });

const pB = panel(6.84, 1.10, 6.00, 2.55, "B", "C-V and Mott-Schottky analysis", C.blue, C.blueLight);
addLineChart(pB.chart, [
  { name: "C", color: C.blue, values: [0.92, 0.82, 0.70, 0.59, 0.49, 0.42, 0.38, 0.37] },
  { name: "1/C2", color: C.green, values: [0.12, 0.20, 0.31, 0.43, 0.57, 0.70, 0.82, 0.88] },
], {
  labels: ["−0.4", "−0.2", "0", "0.2", "0.4", "0.6", "0.8", "1.0"],
  yMin: 0,
  yMax: 1.0,
  yUnit: 0.2,
  yFmt: "0.0",
});
rich([run("Bias "), run("V", { italic: true }), run(" (V)")], pB.chart.x + 1.13, pB.chart.y + pB.chart.h - 0.08, 0.72, 0.12, {
  fontSize: 6.1,
  color: C.muted,
  align: "center",
  role: "axis-label",
});
rich(["Normalized ", run("C", { italic: true }), ", 1/", run("C", { italic: true }), sup("2")], pB.chart.x - 0.03, pB.chart.y - 0.02, 1.10, 0.12, {
  fontSize: 6.1,
  color: C.muted,
  role: "axis-label",
});
miniMetric(pB.note.x, pB.note.y, [run("V", { italic: true }), sub("bi")], "0.96 V", C.blue);
miniMetric(pB.note.x + 1.01, pB.note.y, [run("N", { italic: true }), sub("eff")], "2×10¹⁶", C.blue);
chartOverlayLegend([
  { color: C.blue, label: [run("C", { italic: true }), "(", run("V", { italic: true }), ")"] },
  { color: C.green, label: ["1/", run("C", { italic: true }), sup("2")] },
], pB.chart.x + 1.95, pB.chart.y + 1.07, 0.72, { border: C.blue });
bullet("Extract depletion capacitance, built-in voltage, and apparent doping.", pB.note.x, pB.note.y + 0.56, pB.note.w, { color: C.blue, h: 0.37 });
bullet(["Linear 1/", run("C", { italic: true }), sup("2"), " versus ", run("V", { italic: true }), " is used only in the depletion regime."], pB.note.x, pB.note.y + 0.99, pB.note.w, { color: C.green, h: 0.42 });
bullet("Fully depleted thin absorbers need careful interpretation.", pB.note.x, pB.note.y + 1.47, pB.note.w, { color: C.slate, h: 0.29, fontSize: 6.35 });

const pC = panel(0.50, 3.96, 6.00, 2.55, "C", "C-f / admittance spectroscopy", C.green, C.greenLight);
addLineChart(pC.chart, [
  { name: "C", color: C.green, values: [0.95, 0.94, 0.91, 0.83, 0.63, 0.42, 0.31, 0.28] },
  { name: "G", color: C.amber, values: [0.12, 0.17, 0.29, 0.56, 0.82, 0.63, 0.34, 0.16] },
], {
  labels: ["10", "10²", "10³", "10⁴", "10⁵", "10⁶", "10⁷", "10⁸"],
  yMin: 0,
  yMax: 1.0,
  yUnit: 0.2,
  yFmt: "0.0",
});
rich(["Frequency ", run("f", { italic: true }), " (Hz)"], pC.chart.x + 1.02, pC.chart.y + pC.chart.h - 0.08, 0.90, 0.12, {
  fontSize: 6.1,
  color: C.muted,
  align: "center",
  role: "axis-label",
});
rich(["Normalized ", run("C", { italic: true }), " / ", run("G", { italic: true })], pC.chart.x - 0.03, pC.chart.y - 0.02, 0.94, 0.12, {
  fontSize: 6.1,
  color: C.muted,
  role: "axis-label",
});
rich([run("From admittance: "), run("Y", { italic: true }), " = 1/", run("Z", { italic: true }), " = ", run("G", { italic: true }), " + jω", run("C", { italic: true })], pC.note.x, pC.note.y, pC.note.w, 0.18, {
  fontSize: 6.9,
  bold: true,
  color: C.green,
  role: "formula",
});
chartOverlayLegend([
  { color: C.green, label: [run("C", { italic: true }), "(", run("f", { italic: true }), ")"] },
  { color: C.amber, label: [run("G", { italic: true }), "(", run("f", { italic: true }), ")"] },
], pC.chart.x + 1.83, pC.chart.y + 0.20, 0.70, { border: C.green });
bullet("Frequency dispersion indicates slow traps, ions, or interfacial charging.", pC.note.x, pC.note.y + 0.43, pC.note.w, { color: C.green, h: 0.43 });
bullet("Temperature-dependent C-f / G-f supports activation-energy analysis.", pC.note.x, pC.note.y + 0.92, pC.note.w, { color: C.amber, h: 0.39 });
bullet("Here, C-f is derived from impedance/admittance.", pC.note.x, pC.note.y + 1.37, pC.note.w, { color: C.slate, h: 0.27, fontSize: 6.35 });

const pD = panel(6.84, 3.96, 6.00, 2.55, "D", "QE / spectral response", C.purple, C.purpleLight);
addLineChart(pD.chart, [
  { name: "QE", color: C.purple, values: [4, 35, 70, 86, 88, 82, 62, 20, 2] },
], {
  labels: ["300", "400", "500", "600", "700", "800", "850", "900", "950"],
  yMin: 0,
  yMax: 100,
  yUnit: 20,
  yFmt: "0",
});
rich(["Wavelength λ (nm)"], pD.chart.x + 1.00, pD.chart.y + pD.chart.h - 0.08, 1.05, 0.12, {
  fontSize: 6.1,
  color: C.muted,
  align: "center",
  role: "axis-label",
});
rich(["QE (%)"], pD.chart.x - 0.02, pD.chart.y - 0.02, 0.55, 0.12, {
  fontSize: 6.1,
  color: C.muted,
  role: "axis-label",
});
chartOverlayLegend([
  { color: C.purple, label: ["QE(λ)"] },
], pD.chart.x + 2.04, pD.chart.y + 0.22, 0.72, { border: C.purple });
miniMetric(pD.note.x, pD.note.y, [run("J", { italic: true }), sub("sc,QE")], "23.5", C.purple);
rect(pD.note.x + 1.01, pD.note.y, 0.90, 0.31, "F8FAFC", C.line, { round: 0.04, pt: 0.7 });
rich([run("E", { italic: true }), sub("g")], pD.note.x + 1.07, pD.note.y + 0.046, 0.24, 0.10, {
  fontSize: 5.8,
  color: C.muted,
  role: "metric-label",
});
text("edge", pD.note.x + 1.38, pD.note.y + 0.043, 0.46, 0.11, {
  fontSize: 6.4,
  bold: true,
  color: C.purple,
  align: "right",
  role: "metric-value",
});
bullet("QE is plotted versus wavelength λ or photon energy hν, not electrical frequency.", pD.note.x, pD.note.y + 0.50, pD.note.w, { color: C.purple, h: 0.38 });
bullet(["Integrated QE gives a consistency check on ", run("J", { italic: true }), sub("sc"), " under AM1.5G illumination."], pD.note.x, pD.note.y + 0.94, pD.note.w, { color: C.purple, h: 0.36 });
bullet("Short- and long-wavelength roll-off locate optical and collection losses.", pD.note.x, pD.note.y + 1.34, pD.note.w, { color: C.slate, h: 0.31 });

rect(0, 6.88, 13.333, 0.62, C.strip, C.strip, { lineTransparency: 100 });
rich([
  run("Note: ", { bold: true }),
  run("band diagrams, carrier densities, electric field, current components, generation and recombination are "),
  run("diagnostic working-point profiles", { bold: true, color: C.blue }),
  run(", not the four primary SCAPS-1D output modules."),
], 0.64, 7.06, 12.05, 0.20, {
  fontSize: 10.1,
  color: C.navy,
  align: "center",
  role: "takeaway",
});

slide.addNotes(`Suggested narration:
This slide summarizes the four main SCAPS-1D output modules. I-V/J-V gives terminal photovoltaic performance and dark diode behavior. C-V, usually shown with the Mott-Schottky 1/C^2 plot, extracts built-in voltage and apparent doping under depletion assumptions. C-f/admittance spectroscopy probes frequency-dependent capacitance and conductance, which are often linked to traps, ionic motion, and interfacial charging. QE is the spectral collection response versus wavelength or photon energy and can be integrated to cross-check Jsc. Internal profiles such as band diagrams and recombination profiles are still important diagnostic outputs, but they should be presented separately from these four main modules.`);

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
