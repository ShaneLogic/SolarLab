const fs = require("fs");
const path = require("path");
const pptxgen = require("/Users/shane/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/pptxgenjs");

const root = path.resolve(__dirname, "../../..");
const outDir = path.join(root, "docs", "manual", "slides");
const outPath = path.join(outDir, "radau_transient_solve_overview.pptx");
fs.mkdirSync(outDir, { recursive: true });

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "SolarLab";
pptx.company = "SolarLab";
pptx.subject = "Radau transient solve overview";
pptx.title = "Implicit Radau Time Integration of the Semi-Discrete 2D Drift-Diffusion System";
pptx.lang = "en-US";
pptx.theme = { headFontFace: "Arial", bodyFontFace: "Arial", lang: "en-US" };
pptx.margin = 0;

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
  red: "B42318",
  redLight: "FEE4E2",
  white: "FFFFFF",
  offWhite: "F8FAFC",
  bg: "F7F9FC",
  strip: "EAF2FF",
};

const slide = pptx.addSlide();
slide.background = { color: C.bg };

// Track text boxes so we can run a simple geometry overlap check for the
// editable layout. Intentional decorative rectangles/arrows are not included.
const textBoxes = [];

function text(txt, x, y, w, h, opt = {}) {
  textBoxes.push({ txt, x, y, w, h, role: opt.role || "text" });
  slide.addText(txt, {
    x, y, w, h,
    fontFace: "Arial",
    margin: opt.margin ?? 0,
    fit: opt.fit || "shrink",
    color: opt.color || C.navy,
    fontSize: opt.fontSize || 14,
    bold: opt.bold || false,
    italic: opt.italic || false,
    align: opt.align || "left",
    valign: opt.valign || "top",
    paraSpaceAfterPt: opt.paraSpaceAfterPt || 0,
    paraSpaceBeforePt: 0,
  });
}

function richText(runs, x, y, w, h, opt = {}) {
  const plain = runs.map((r) => r.text || "").join("");
  textBoxes.push({ txt: plain, x, y, w, h, role: opt.role || "rich-text" });
  const base = {
    fontFace: "Arial",
    fontSize: opt.fontSize || 12,
    color: opt.color || C.navy,
    bold: opt.bold || false,
    italic: opt.italic || false,
  };
  slide.addText(
    runs.map((r) => ({
      text: r.text,
      options: {
        ...base,
        ...(r.options || {}),
      },
    })),
    {
      x, y, w, h,
      fontFace: "Arial",
      margin: opt.margin ?? 0,
      fit: opt.fit || "shrink",
      align: opt.align || "left",
      valign: opt.valign || "top",
      paraSpaceAfterPt: 0,
      paraSpaceBeforePt: 0,
    },
  );
}

function sub(text, opt = {}) {
  return { text, options: { subscript: true, ...opt } };
}

function sup(text, opt = {}) {
  return { text, options: { superscript: true, ...opt } };
}

const R = {
  Vapp: [{ text: "V" }, sub("app")],
  Y0: [{ text: "Y" }, sub("0")],
  Yn: [{ text: "Y" }, sub("n")],
  Yn1: [{ text: "Y" }, sub("n+1")],
  tend: [{ text: "t" }, sub("end")],
  t0: [{ text: "t" }, sub("0")],
  t1: [{ text: "t" }, sub("1")],
  tN: [{ text: "t" }, sub("N")],
  tk: [{ text: "t" }, sub("k")],
  tk1: [{ text: "t" }, sub("k+1")],
  Vk: [{ text: "V" }, sub("k")],
  Vk1: [{ text: "V" }, sub("k+1")],
  Vmax: [{ text: "V" }, sub("max")],
  Yk: [{ text: "Y" }, sub("k")],
  Yk1: [{ text: "Y" }, sub("k+1")],
  Ki: [{ text: "K" }, sub("i")],
  Kj: [{ text: "K" }, sub("j")],
  ci: [{ text: "c" }, sub("i")],
  aij: [{ text: "a" }, sub("ij")],
  bi: [{ text: "b" }, sub("i")],
};

function concatRuns(...parts) {
  return parts.flatMap((p) => (typeof p === "string" ? [{ text: p }] : p));
}

function rect(x, y, w, h, fill, line = C.line, opt = {}) {
  slide.addShape(opt.round ? pptx.ShapeType.roundRect : pptx.ShapeType.rect, {
    x, y, w, h,
    rectRadius: opt.round || 0,
    fill: { color: fill, transparency: opt.fillTransparency || 0 },
    line: { color: line, pt: opt.pt || 1, transparency: opt.lineTransparency || 0 },
  });
}

function arrow(x1, y1, x2, y2, color = C.muted, opt = {}) {
  slide.addShape(pptx.ShapeType.line, {
    x: x1, y: y1, w: x2 - x1, h: y2 - y1,
    line: {
      color,
      pt: opt.pt || 1.4,
      beginArrowType: "none",
      endArrowType: opt.noHead ? "none" : "triangle",
      dash: opt.dash || "solid",
    },
  });
}

function card(x, y, w, h, label, body, fill, accent, opt = {}) {
  rect(x, y, w, h, fill, accent, { pt: 1.15, round: 0.08 });
  rect(x, y, 0.08, h, accent, accent, { lineTransparency: 100 });
  text(label, x + 0.18, y + 0.13, w - 0.32, opt.labelH || 0.25, {
    fontSize: opt.labelSize || 9.8,
    bold: true,
    color: accent,
    role: "card-label",
  });
  text(body, x + 0.18, y + (opt.bodyY || 0.47), w - 0.34, h - (opt.bodyY || 0.47) - 0.10, {
    fontSize: opt.bodySize || 11.0,
    color: C.navy,
    fit: "shrink",
    role: "card-body",
  });
}

function richCard(x, y, w, h, label, bodyRuns, fill, accent, opt = {}) {
  rect(x, y, w, h, fill, accent, { pt: 1.15, round: 0.08 });
  rect(x, y, 0.08, h, accent, accent, { lineTransparency: 100 });
  text(label, x + 0.18, y + 0.13, w - 0.32, opt.labelH || 0.25, {
    fontSize: opt.labelSize || 9.8,
    bold: true,
    color: accent,
    role: "card-label",
  });
  richText(bodyRuns, x + 0.18, y + (opt.bodyY || 0.47), w - 0.34, h - (opt.bodyY || 0.47) - 0.10, {
    fontSize: opt.bodySize || 11.0,
    color: C.navy,
    fit: "shrink",
    role: "card-body",
  });
}

function smallPill(x, y, w, label, fill, accent) {
  rect(x, y, w, 0.28, fill, accent, { pt: 0.9, round: 0.12 });
  text(label, x + 0.05, y + 0.07, w - 0.10, 0.10, {
    fontSize: 8.4,
    bold: true,
    color: accent,
    align: "center",
    role: "pill",
  });
}

// Header
text("Implicit Radau Time Integration of the Semi-Discrete 2D Drift-Diffusion System", 0.46, 0.23, 12.30, 0.38, {
  fontSize: 22.0,
  bold: true,
  role: "title",
});
richText(concatRuns("Outer solver at fixed ", R.Vapp, "; repeatedly evaluates the RHS operator F(Y) until a quasi-steady carrier state is reached."), 0.47, 0.69, 12.0, 0.26, {
  fontSize: 11.8,
  color: C.slate,
  role: "subtitle",
});

// Left panel: outer workflow.
rect(0.45, 1.08, 6.28, 5.54, C.white, C.line, { round: 0.08 });
text("Outer Solver: Voltage-Specific Transient Settle", 0.73, 1.25, 5.60, 0.24, {
  fontSize: 12.2,
  bold: true,
  color: C.slate,
  role: "panel-title",
});

richCard(0.82, 1.68, 2.22, 0.90, "INPUT", concatRuns(R.Y0, ", ", R.Vapp), C.blueLight, C.blue, { bodySize: 13.0, labelSize: 9.4 });
arrow(3.10, 2.13, 3.55, 2.13, C.muted);
richCard(3.62, 1.68, 2.58, 0.90, "INITIAL-VALUE PROBLEM", concatRuns("dY/dt = F(Y; ", R.Vapp, ")"), C.purpleLight, C.purple, { bodySize: 12.2, labelSize: 8.8 });

arrow(4.91, 2.63, 4.91, 3.00, C.muted);
richCard(3.10, 3.05, 3.62, 1.10, "RADAU TRANSIENT SOLVE", concatRuns("implicit time stepping\n", R.Y0, "  ->  Y(", R.tend, ")"), C.offWhite, C.slate, { bodySize: 12.2, labelSize: 9.4 });
arrow(4.91, 4.20, 4.91, 4.56, C.muted);

richCard(0.82, 4.62, 2.52, 0.95, "SETTLED STATE", concatRuns("Y(", R.tend, ")\nquasi-steady n, p"), C.greenLight, C.green, { bodySize: 11.1, labelSize: 9.2 });
arrow(3.42, 5.10, 3.88, 5.10, C.muted);
richCard(3.98, 4.62, 2.22, 0.95, "OUTPUT", concatRuns("J(", R.Vapp, ")\nterminal current"), C.amberLight, C.amber, { bodySize: 10.6, labelSize: 9.2 });

// Left panel time axis, kept sparse to avoid label collision.
arrow(0.92, 6.08, 6.04, 6.08, C.line, { pt: 2.0, noHead: true });
[
  [R.t0, 1.10],
  [R.t1, 2.45],
  ["...", 3.80],
  [R.tN, 5.15],
  [R.tend, 5.88],
].forEach(([lab, x]) => {
  arrow(x, 5.96, x, 6.20, C.slate, { pt: 1.0, noHead: true });
  if (typeof lab === "string") {
    text(lab, x - 0.22, 6.28, 0.44, 0.12, { fontSize: 8.6, color: C.slate, align: "center", role: "axis-label" });
  } else {
    richText(lab, x - 0.22, 6.28, 0.44, 0.12, { fontSize: 8.6, color: C.slate, align: "center", role: "axis-label" });
  }
});

// Right panel: what happens inside one Radau step.
rect(6.98, 1.08, 5.90, 5.54, C.white, C.line, { round: 0.08 });
text("Inside One Radau Step", 7.26, 1.25, 2.35, 0.24, {
  fontSize: 12.2,
  bold: true,
  color: C.slate,
  role: "panel-title",
});
text("Radau is an implicit Runge-Kutta method.", 10.04, 1.27, 2.45, 0.18, {
  fontSize: 8.8,
  color: C.muted,
  align: "right",
  role: "side-note",
});

// Formula block, deliberately split into two shorter lines with large box.
rect(7.32, 1.64, 5.18, 1.13, C.purpleLight, C.purple, { pt: 1.15, round: 0.08 });
text("implicit stage equations", 7.55, 1.79, 2.1, 0.18, {
  fontSize: 9.4,
  bold: true,
  color: C.purple,
  role: "formula-label",
});
richText(concatRuns(R.Ki, " = F(", [{ text: "t" }, sub("n")], " + ", R.ci, " h,  ", R.Yn, " + h Σ", sub("j"), " ", R.aij, " ", R.Kj, ")"), 7.55, 2.10, 4.74, 0.22, {
  fontSize: 10.6,
  color: C.navy,
  role: "formula",
});
richText(concatRuns(R.Yn1, " = ", R.Yn, " + h Σ", sub("i"), " ", R.bi, " ", R.Ki), 7.55, 2.42, 4.74, 0.20, {
  fontSize: 10.6,
  color: C.navy,
  role: "formula",
});

// Stage nodes and RHS calls.
rect(7.32, 3.03, 5.18, 1.24, C.offWhite, C.line, { pt: 1.0, round: 0.08 });
text("stage evaluations call F(Y)", 7.55, 3.17, 2.3, 0.18, {
  fontSize: 9.4,
  bold: true,
  color: C.blue,
  role: "stage-label",
});
arrow(7.72, 3.72, 11.95, 3.72, C.purple, { pt: 1.4, noHead: true });
[
  [[{ text: "K" }, sub("1")], 8.28],
  [[{ text: "K" }, sub("2")], 9.83],
  [[{ text: "K" }, sub("3")], 11.38],
].forEach(([lab, x]) => {
  slide.addShape(pptx.ShapeType.ellipse, {
    x: x - 0.16,
    y: 3.55,
    w: 0.32,
    h: 0.32,
    fill: { color: C.purpleLight },
    line: { color: C.purple, pt: 1.0 },
  });
  richText(lab, x - 0.13, 3.64, 0.26, 0.08, { fontSize: 8.6, color: C.purple, align: "center", role: "stage-node" });
});

card(7.32, 4.56, 2.58, 1.02, "INNER RHS OPERATOR", "F(Y): Poisson + R/G\n+ SG flux divergence\n+ contact BCs", C.blueLight, C.blue, { bodySize: 9.4, labelSize: 8.8, bodyY: 0.42 });
arrow(9.98, 5.06, 10.40, 5.06, C.muted);
card(10.48, 4.56, 2.02, 1.02, "WHY IMPLICIT?", "stiff transport\nstrong coupling\nstable larger steps", C.redLight, C.red, { bodySize: 9.4, labelSize: 8.8, bodyY: 0.42 });

smallPill(7.50, 5.91, 2.18, "run_transient_2d(...)", C.greenLight, C.green);
smallPill(9.93, 5.91, 2.18, "solve_ivp(method='Radau')", C.greenLight, C.green);

// Bottom takeaway strip: general Radau role plus the concrete SolarLab J-V
// sweep mode implemented by the frontend/backend path.
slide.addShape(pptx.ShapeType.rect, {
  x: 0,
  y: 6.73,
  w: 13.333,
  h: 0.77,
  fill: { color: C.strip },
  line: { color: C.strip, transparency: 100 },
});
richText(concatRuns("Takeaway: RHS assembly defines F(Y); Radau repeatedly evaluates F(Y) to advance the stiff 2D carrier-transport system toward a ", R.Vapp, "-specific quasi-steady state."), 0.50, 6.87, 12.25, 0.18, {
  fontSize: 9.2,
  bold: true,
  color: C.navy,
  align: "center",
  role: "takeaway",
});
richText(concatRuns(
  [{ text: "Actual J-V sweep mode: ", options: { bold: true, color: C.amber } }],
  "stepwise transient sweep. Hold ", R.Vk, " fixed on [", R.tk, ", ", R.tk1, "]; Radau advances ", R.Yk, " -> ", R.Yk1,
  ".  Δt = |", R.Vk1, " - ", R.Vk, "| / scan rate; forward 0 -> ", R.Vmax, ", then reverse ", R.Vmax, " -> 0."
), 0.62, 7.15, 12.05, 0.18, {
  fontSize: 8.6,
  color: C.slate,
  align: "center",
  role: "takeaway-mode",
});

slide.addNotes(`Suggested narration:
This slide explains the outer time-integration layer of the 2D solver. After spatial discretization, the coupled drift-diffusion-Poisson PDE system becomes a stiff semi-discrete initial-value problem, dY/dt = F(Y; V_app), where Y contains all electron and hole densities on the 2D grid.

The previous RHS assembly slide explains how F(Y) is constructed from Poisson, recombination, generation, Scharfetter-Gummel fluxes, and boundary conditions. Here, Radau uses that F(Y) inside an implicit time-stepping scheme. At each time step, Radau solves coupled nonlinear stage equations K_i, and each stage evaluation calls the RHS operator.

For one fixed applied voltage V_app, the transient solver advances Y from the initial state Y0 to a settled or quasi-steady state Y(t_end). The terminal current J(V_app) is then extracted from the final spatial snapshot. This is why the RHS assembly is the inner kernel, while Radau is the outer solver that actually advances the system in time.

In the current frontend J-V sweep, this fixed-voltage transient solve is chained across voltage samples rather than solved as a continuous voltage ramp. The backend holds V_k fixed for one Radau interval, computes dt = |V_{k+1} - V_k| / scan_rate, carries the final state Y_k forward to Y_{k+1}, and then repeats this from 0 to V_max and back from V_max to 0. Therefore the scan rate changes the physical dwell time per voltage step and can affect hysteresis.

Academic phrasing:
The spatially discretized 2D drift-diffusion-Poisson equations form a stiff semi-discrete ODE system. The RHS operator F(Y) is assembled at each nonlinear stage evaluation, and the initial-value problem is advanced using an implicit Radau IIA time integrator.`);

function overlap(a, b, pad = 0.02) {
  return !(
    a.x + a.w + pad <= b.x ||
    b.x + b.w + pad <= a.x ||
    a.y + a.h + pad <= b.y ||
    b.y + b.h + pad <= a.y
  );
}

// Ignore label/body overlaps inside the same card by requiring role equality
// not to be a card internal pair. This check is intentionally conservative
// and flags only unrelated text boxes.
const warnings = [];
for (let i = 0; i < textBoxes.length; i++) {
  for (let j = i + 1; j < textBoxes.length; j++) {
    const a = textBoxes[i];
    const b = textBoxes[j];
    const sameCardInternal =
      (a.role === "card-label" && b.role === "card-body") ||
      (a.role === "card-body" && b.role === "card-label");
    if (!sameCardInternal && overlap(a, b, 0.01)) {
      warnings.push(`${a.role}:${a.txt.slice(0, 28)} <-> ${b.role}:${b.txt.slice(0, 28)}`);
    }
  }
}
if (warnings.length) {
  console.warn("Potential text-box overlaps:");
  for (const w of warnings) console.warn("  " + w);
  process.exitCode = 2;
}

pptx.writeFile({ fileName: outPath });
