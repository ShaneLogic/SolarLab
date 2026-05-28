const fs = require("fs");
const path = require("path");
const pptxgen = require("/Users/shane/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/pptxgenjs");

const root = path.resolve(__dirname, "../../..");
const outDir = path.join(root, "docs", "manual", "slides");
const outPath = path.join(outDir, "jv_stepwise_transient_sweep_animation.pptx");
const gifPath = path.join(root, "docs", "manual", "figures", "jv_stepwise_transient_sweep.gif");
const previewPath = path.join(root, "docs", "manual", "figures", "jv_stepwise_transient_sweep_preview.png");
fs.mkdirSync(outDir, { recursive: true });

if (!fs.existsSync(gifPath)) {
  throw new Error(`Missing animation: ${gifPath}. Run make_jv_stepwise_sweep_animation.py first.`);
}

const C = {
  navy: "162033",
  slate: "334155",
  muted: "64748B",
  line: "CBD5E1",
  blue: "2563EB",
  blueLight: "DBEAFE",
  green: "16803C",
  greenLight: "DCFCE7",
  amber: "A16207",
  amberLight: "FEF3C7",
  purple: "6D28D9",
  purpleLight: "EDE9FE",
  white: "FFFFFF",
  bg: "F7F9FC",
  strip: "EAF2FF",
};

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "SolarLab";
pptx.company = "SolarLab";
pptx.subject = "Animated explanation of SolarLab stepwise J-V sweep";
pptx.title = "Animated J-V Sweep: Stepwise Fixed-Voltage Radau Transient Solves";
pptx.lang = "en-US";
pptx.theme = { headFontFace: "Arial", bodyFontFace: "Arial", lang: "en-US" };
pptx.margin = 0;

const slide = pptx.addSlide();
slide.background = { color: C.bg };
const boxes = [];

function track(txt, x, y, w, h, role) {
  boxes.push({ txt: String(txt || ""), x, y, w, h, role: role || "text" });
}

function text(txt, x, y, w, h, opt = {}) {
  track(txt, x, y, w, h, opt.role);
  slide.addText(txt, {
    x, y, w, h,
    margin: opt.margin ?? 0,
    fit: opt.fit || "shrink",
    fontFace: "Arial",
    fontSize: opt.fontSize || 12,
    color: opt.color || C.navy,
    bold: opt.bold || false,
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
  track(plain, x, y, w, h, opt.role);
  const base = {
    fontFace: "Arial",
    fontSize: opt.fontSize || 12,
    color: opt.color || C.navy,
    bold: opt.bold || false,
  };
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

function card(x, y, w, h, label, bodyParts, fill, accent, opt = {}) {
  rect(x, y, w, h, fill, accent, { pt: 1.0, round: 0.08 });
  rect(x, y, 0.08, h, accent, accent, { lineTransparency: 100 });
  text(label, x + 0.18, y + (opt.labelY ?? 0.12), w - 0.34, opt.labelH ?? 0.22, {
    fontSize: opt.labelSize || 9.8,
    bold: true,
    color: accent,
    role: "card-label",
  });
  rich(bodyParts, x + 0.18, y + (opt.bodyY ?? 0.43), w - 0.34, opt.bodyH ?? h - 0.51, {
    fontSize: opt.bodySize || 9.8,
    color: C.navy,
    fit: "shrink",
    role: "card-body",
  });
}

const R = {
  Vapp: [{ text: "V" }, sub("app")],
  Vk: [{ text: "V" }, sub("k")],
  Vk1: [{ text: "V" }, sub("k+1")],
  Vmax: [{ text: "V" }, sub("max")],
  Yk1: [{ text: "Y" }, sub("k+1")],
  tk: [{ text: "t" }, sub("k")],
  tk1: [{ text: "t" }, sub("k+1")],
};

function concat(...parts) {
  return parts.flatMap((p) => (typeof p === "string" ? [{ text: p }] : p));
}

text("Animated J-V Sweep: Fixed-Voltage Radau Calls Chained in Time", 0.46, 0.22, 12.35, 0.34, {
  fontSize: 21.5,
  bold: true,
  role: "title",
});
rich(concat(
  "Actual SolarLab mode: hold ", R.Vk,
  " fixed, integrate one transient, record J, then warm-start the next voltage."
), 0.47, 0.66, 12.1, 0.25, {
  fontSize: 12.2,
  color: C.slate,
  role: "subtitle",
});

// Large animation area.
rect(0.48, 1.03, 12.38, 5.08, C.white, C.line, { round: 0.08 });
text("Stepwise Transient Sweep", 0.75, 1.19, 3.2, 0.25, {
  fontSize: 12.8,
  bold: true,
  color: C.slate,
  role: "panel-title",
});
text("GIF animation; use slideshow mode for playback.", 9.70, 1.21, 2.85, 0.18, {
  fontSize: 8.8,
  color: C.muted,
  align: "right",
  role: "side-note",
});
slide.addImage({
  path: gifPath,
  x: 0.78,
  y: 1.50,
  w: 11.78,
  h: 3.96,
  sizing: { type: "contain", x: 0.78, y: 1.50, w: 11.78, h: 3.96 },
});

// Compact editable summary cards.
card(0.60, 6.18, 3.85, 0.64, "STAIRCASE VOLTAGE", concat("sampled plateaus; not one continuous ramp"), C.blueLight, C.blue, {
  labelSize: 9.6,
  bodySize: 8.8,
  bodyY: 0.37,
  bodyH: 0.20,
});
card(4.74, 6.18, 3.85, 0.64, "ONE RADAU CALL", concat("[", R.tk, ", ", R.tk1, "] at fixed ", R.Vk), C.purpleLight, C.purple, {
  labelSize: 9.6,
  bodySize: 8.9,
  bodyY: 0.37,
  bodyH: 0.20,
});
card(8.88, 6.18, 3.85, 0.64, "STATE CARRY-OVER", concat(R.Yk1, " seeds the next voltage point"), C.greenLight, C.green, {
  labelSize: 9.6,
  bodySize: 8.9,
  bodyY: 0.37,
  bodyH: 0.20,
});

// Formula strip.
slide.addShape(pptx.ShapeType.rect, {
  x: 0,
  y: 6.96,
  w: 13.333,
  h: 0.54,
  fill: { color: C.strip },
  line: { color: C.strip, transparency: 100 },
});
rich(concat(
  [{ text: "Numerical mode: ", options: { bold: true, color: C.amber } }],
  "piecewise-constant ", R.Vapp, ".  Δt = |", R.Vk1, " - ", R.Vk,
  "| / scan rate; forward 0 -> ", R.Vmax, ", reverse ", R.Vmax, " -> 0."
), 0.55, 7.10, 12.20, 0.20, {
  fontSize: 10.6,
  bold: true,
  color: C.navy,
  align: "center",
  role: "takeaway",
});

slide.addNotes(`Suggested narration:
This slide separates the actual SolarLab J-V sweep implementation from the idealized continuous-ramp picture. The frontend supplies a scan rate and a finite number of voltage samples. The backend converts those into a dwell time per sample, dt = |V_{k+1} - V_k| / scan_rate.

At each sample, V_app is held fixed. Radau integrates the coupled drift-diffusion state over that dwell interval, the terminal current is recorded after the settle, and the final state becomes the initial condition for the next voltage sample. This state carry-over is what allows forward and reverse sweeps to differ when ions or slow coupled variables lag the voltage history.

The important distinction is that this is not a single continuous V_app(t) ramp inside one solve_ivp call. It is a chain of fixed-voltage transient solves: forward from 0 to V_max, then reverse from V_max back to 0.`);

function overlap(a, b, pad = 0.015) {
  return !(
    a.x + a.w + pad <= b.x ||
    b.x + b.w + pad <= a.x ||
    a.y + a.h + pad <= b.y ||
    b.y + b.h + pad <= a.y
  );
}

const warnings = [];
for (let i = 0; i < boxes.length; i++) {
  for (let j = i + 1; j < boxes.length; j++) {
    const a = boxes[i];
    const b = boxes[j];
    const cardInternal =
      (a.role === "card-label" && b.role === "card-body") ||
      (a.role === "card-body" && b.role === "card-label");
    if (!cardInternal && overlap(a, b)) {
      warnings.push(`${a.role}:${a.txt.slice(0, 30)} <-> ${b.role}:${b.txt.slice(0, 30)}`);
    }
  }
}
if (warnings.length) {
  console.warn("Potential text-box overlaps:");
  for (const warning of warnings) console.warn("  " + warning);
  process.exitCode = 2;
}

pptx.writeFile({ fileName: outPath });
console.log(outPath);
console.log(previewPath);
