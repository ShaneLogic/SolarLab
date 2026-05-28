const fs = require("fs");
const path = require("path");
const pptxgen = require("/Users/shane/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/pptxgenjs");

const root = path.resolve(__dirname, "../../..");
const outDir = path.join(root, "docs", "manual", "slides");
const outPath = path.join(outDir, "complete_2d_solver_workflow.pptx");
fs.mkdirSync(outDir, { recursive: true });

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "SolarLab";
pptx.company = "SolarLab";
pptx.subject = "Complete 2D solver workflow";
pptx.title = "Complete 2D Solver Workflow";
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
  white: "FFFFFF",
  offWhite: "F8FAFC",
  bg: "F7F9FC",
  strip: "EAF2FF",
};

const SYM = {
  Vk: "Vₖ",
  Y0: "Y₀",
  Yi: "Y⁽ⁱ⁾",
  Yts: "Y(tₛ)",
  JVk: "J(Vₖ)",
  JV: "J-V",
  FY: "F(Y)",
  dYdt: "dY/dt",
  dn: "dn/dt",
  dp: "dp/dt",
  rho: "ρ",
  phi: "φ",
  Jn: "Jₙ",
  Jp: "Jₚ",
  divJ: "∇·J",
};

const slide = pptx.addSlide();
slide.background = { color: C.bg };
const textBoxes = [];

function text(txt, x, y, w, h, opt = {}) {
  textBoxes.push({ txt, x, y, w, h, role: opt.role || "text" });
  slide.addText(txt, {
    x,
    y,
    w,
    h,
    fontFace: "Arial",
    margin: opt.margin ?? 0,
    fit: opt.fit || "shrink",
    color: opt.color || C.navy,
    fontSize: opt.fontSize || 14,
    bold: opt.bold || false,
    italic: opt.italic || false,
    align: opt.align || "left",
    valign: opt.valign || "top",
    paraSpaceAfterPt: 0,
    paraSpaceBeforePt: 0,
    breakLine: opt.breakLine || false,
  });
}

function rect(x, y, w, h, fill, line = C.line, opt = {}) {
  slide.addShape(opt.round ? pptx.ShapeType.roundRect : pptx.ShapeType.rect, {
    x,
    y,
    w,
    h,
    rectRadius: opt.round || 0,
    fill: { color: fill, transparency: opt.fillTransparency || 0 },
    line: { color: line, pt: opt.pt || 1, transparency: opt.lineTransparency || 0 },
  });
}

function arrow(x1, y1, x2, y2, color = C.muted, opt = {}) {
  slide.addShape(pptx.ShapeType.line, {
    x: x1,
    y: y1,
    w: x2 - x1,
    h: y2 - y1,
    line: {
      color,
      pt: opt.pt || 1.3,
      beginArrowType: "none",
      endArrowType: opt.noHead ? "none" : "triangle",
      dash: opt.dash || "solid",
      transparency: opt.transparency || 0,
    },
  });
}

function card(x, y, w, h, label, body, fill, accent, opt = {}) {
  rect(x, y, w, h, fill, accent, { pt: 1.1, round: 0.08 });
  rect(x, y, 0.08, h, accent, accent, { lineTransparency: 100 });
  text(label, x + 0.18, y + 0.10, w - 0.32, opt.labelH || 0.20, {
    fontSize: opt.labelSize || 8.8,
    bold: true,
    color: accent,
    role: "card-label",
  });
  text(body, x + 0.18, y + (opt.bodyY || 0.34), w - 0.34, h - (opt.bodyY || 0.34) - 0.08, {
    fontSize: opt.bodySize || 8.8,
    color: C.navy,
    fit: "shrink",
    role: "card-body",
  });
}

function miniStep(x, y, w, h, idx, title, body, fill, accent) {
  rect(x, y, w, h, fill, accent, { pt: 0.85, round: 0.06 });
  text(idx, x + 0.09, y + 0.10, 0.22, 0.18, {
    fontSize: 8.0,
    bold: true,
    color: accent,
    align: "center",
    role: "mini-index",
  });
  text(title, x + 0.35, y + 0.09, w - 0.44, 0.18, {
    fontSize: 7.7,
    bold: true,
    color: accent,
    role: "mini-title",
  });
  text(body, x + 0.35, y + 0.33, w - 0.44, h - 0.40, {
    fontSize: 7.2,
    color: C.navy,
    fit: "shrink",
    role: "mini-body",
  });
}

function overlap(a, b, pad = 0.01) {
  return !(
    a.x + a.w + pad <= b.x ||
    b.x + b.w + pad <= a.x ||
    a.y + a.h + pad <= b.y ||
    b.y + b.h + pad <= a.y
  );
}

text("Complete 2D Solver Workflow: RHS Assembly + Radau Integration", 0.46, 0.22, 12.35, 0.34, {
  fontSize: 20.8,
  bold: true,
  role: "title",
});
text("The full 2D J-V calculation is an outer voltage sweep; for each applied voltage, an implicit Radau solver repeatedly evaluates the 2D RHS physics operator.", 0.47, 0.68, 12.05, 0.28, {
  fontSize: 11.2,
  color: C.slate,
  role: "subtitle",
});

rect(0.45, 1.05, 12.43, 5.72, C.white, C.line, { round: 0.08 });

// Left column: complete workflow boundary.
rect(0.72, 1.25, 3.25, 5.22, C.offWhite, C.line, { pt: 1.0, round: 0.08 });
text("A. Outer 2D J-V Sweep", 0.96, 1.40, 2.70, 0.22, {
  fontSize: 11.3,
  bold: true,
  color: C.slate,
  role: "section-title",
});
text("complete calculation boundary", 1.18, 1.63, 2.25, 0.15, {
  fontSize: 7.7,
  color: C.muted,
  align: "center",
  role: "section-note",
});

card(0.96, 1.82, 2.78, 0.80, "PREPROCESS", "device stack\n2D mesh\nmaterial arrays", C.blueLight, C.blue, { bodySize: 8.0, bodyY: 0.36 });
arrow(2.35, 2.67, 2.35, 2.87, C.muted, { pt: 1.0 });
card(0.96, 2.91, 2.78, 0.80, "INITIALIZE", `1D warm start\nbroadcast ${SYM.Y0}\nfixed ion background`, C.greenLight, C.green, { bodySize: 8.0, bodyY: 0.36 });
arrow(2.35, 3.76, 2.35, 3.96, C.muted, { pt: 1.0 });
card(0.96, 4.00, 2.78, 0.80, "VOLTAGE LOOP", `for each ${SYM.Vk}\ncall fixed-voltage\ntransient solve`, C.amberLight, C.amber, { bodySize: 8.0, bodyY: 0.36 });
arrow(2.35, 4.85, 2.35, 5.05, C.muted, { pt: 1.0 });
card(0.96, 5.09, 2.78, 0.80, "POSTPROCESS", `collect ${SYM.JVk}\nassemble ${SYM.JV} curve\nextract metrics`, C.purpleLight, C.purple, { bodySize: 8.0, bodyY: 0.36 });

// Right column: fixed-voltage core solve.
rect(4.17, 1.25, 8.27, 5.22, C.offWhite, C.line, { pt: 1.0, round: 0.08 });
text(`B. Fixed-Voltage Core Solve at ${SYM.Vk}`, 4.42, 1.40, 4.50, 0.22, {
  fontSize: 11.3,
  bold: true,
  color: C.slate,
  role: "section-title",
});
text(`Radau repeatedly calls ${SYM.FY}`, 9.36, 1.43, 2.50, 0.16, {
  fontSize: 8.1,
  color: C.muted,
  align: "right",
  role: "section-note",
});

card(4.45, 1.82, 7.66, 0.82, "IMPLICIT RADAU TIME INTEGRATION", `${SYM.dYdt} = ${SYM.FY};  adaptive stiff IVP solve\nadvance carrier state until ${SYM.Yts}`, C.purpleLight, C.purple, {
  bodySize: 9.0,
  labelSize: 8.6,
  bodyY: 0.35,
});

text(`stage state ${SYM.Yi}`, 4.98, 2.75, 1.35, 0.15, {
  fontSize: 7.8,
  color: C.muted,
  align: "right",
  role: "stage-label",
});
arrow(6.58, 2.68, 6.58, 2.96, C.blue, { pt: 1.1 });
text(`${SYM.FY} returned`, 9.90, 2.75, 1.55, 0.15, {
  fontSize: 7.8,
  color: C.muted,
  align: "left",
  role: "stage-label",
});
arrow(9.68, 2.96, 9.68, 2.68, C.green, { pt: 1.1 });

rect(4.45, 3.02, 7.66, 1.54, C.blueLight, C.blue, { pt: 1.05, round: 0.08 });
rect(4.45, 3.02, 0.08, 1.54, C.blue, C.blue, { lineTransparency: 100 });
text("2D RHS ASSEMBLY  F(Y)", 4.65, 3.16, 2.80, 0.20, {
  fontSize: 8.8,
  bold: true,
  color: C.blue,
  role: "rhs-title",
});
text(`${SYM.dYdt} = [${SYM.dn}, ${SYM.dp}]`, 9.52, 3.17, 2.15, 0.18, {
  fontSize: 8.5,
  bold: true,
  color: C.blue,
  align: "right",
  role: "rhs-formula",
});

miniStep(4.72, 3.52, 1.28, 0.62, "1", "State", "n(y,x), p(y,x)", C.white, C.blue);
miniStep(6.15, 3.52, 1.28, 0.62, "2", "Poisson", `${SYM.rho} -> ${SYM.phi}`, C.white, C.blue);
miniStep(7.58, 3.52, 1.28, 0.62, "3", "Sources", "G - R(n,p)", C.white, C.blue);
miniStep(9.01, 3.52, 1.28, 0.62, "4", "Flux", `${SYM.Jn}, ${SYM.Jp} by SG`, C.white, C.blue);
miniStep(10.44, 3.52, 1.28, 0.62, "5", "Balance", `${SYM.divJ} + BCs`, C.white, C.blue);

for (const x of [6.01, 7.44, 8.87, 10.30]) {
  arrow(x, 3.83, x + 0.12, 3.83, C.blue, { pt: 0.75 });
}

arrow(8.28, 4.58, 8.28, 4.82, C.muted, { pt: 1.0 });
card(4.45, 4.88, 3.42, 0.72, "CONVERGENCE CHECK", "adaptive step control\nsettling or end-time criterion", C.greenLight, C.green, {
  bodySize: 8.2,
  labelSize: 8.1,
});
arrow(7.96, 5.24, 8.34, 5.24, C.muted, { pt: 1.0 });
card(8.42, 4.88, 3.69, 0.72, "CURRENT EXTRACTION", `terminal ${SYM.JVk}\nreturned to voltage loop`, C.amberLight, C.amber, {
  bodySize: 8.2,
  labelSize: 8.1,
});

rect(4.45, 5.86, 7.66, 0.34, C.white, C.line, { pt: 0.8, round: 0.05 });
text("Core relationship: Radau controls time stepping and nonlinear stage solves; RHS Assembly supplies the spatially discretized semiconductor physics.", 4.66, 5.96, 7.24, 0.12, {
  fontSize: 7.8,
  color: C.muted,
  align: "center",
  role: "callout",
});

// One explicit bridge from the outer loop into the zoomed core.
arrow(3.76, 4.31, 4.25, 4.31, C.amber, { pt: 1.1 });
text(`fixed ${SYM.Vk}`, 3.78, 4.08, 0.52, 0.14, {
  fontSize: 7.3,
  color: C.amber,
  bold: true,
  align: "center",
  role: "bridge-label",
});

slide.addShape(pptx.ShapeType.rect, {
  x: 0,
  y: 6.93,
  w: 13.333,
  h: 0.57,
  fill: { color: C.strip },
  line: { color: C.strip, transparency: 100 },
});
text(`Takeaway: RHS Assembly is the inner physics operator; Radau is the transient time integrator; the voltage sweep wraps both to produce the complete 2D ${SYM.JV} result.`, 0.50, 7.08, 12.25, 0.20, {
  fontSize: 10.7,
  bold: true,
  color: C.navy,
  align: "center",
  role: "takeaway",
});

slide.addNotes(`Suggested narration:
This slide separates the complete SolarLab 2D computation into an outer J-V sweep and a fixed-voltage transient core. The outer workflow builds the mesh and material arrays, initializes the state, loops over applied voltage points, and postprocesses terminal currents into a J-V curve.

Inside each fixed-voltage solve, the Radau method is the implicit time integrator for the semi-discrete method-of-lines system. Radau proposes stage states and repeatedly calls the RHS operator F(Y).

The RHS assembly is not the time integrator by itself. It is the spatially discretized physics evaluation: unpack carrier densities, compute charge density, solve Poisson, evaluate generation and recombination, compute Scharfetter-Gummel fluxes, take current divergence, impose boundary conditions, and return dY/dt.

Academic phrasing:
The complete workflow consists of spatial discretization and material-array construction, warm-start initialization, voltage-stepped implicit Radau integration of the semi-discrete drift-diffusion system, repeated RHS assembly at nonlinear stage evaluations, and terminal-current extraction from the converged transient state.`);

const warnings = [];
for (let i = 0; i < textBoxes.length; i++) {
  for (let j = i + 1; j < textBoxes.length; j++) {
    const a = textBoxes[i];
    const b = textBoxes[j];
    const sameCardInternal =
      (a.role === "card-label" && b.role === "card-body") ||
      (a.role === "card-body" && b.role === "card-label");
    const sameMiniInternal =
      a.role.startsWith("mini-") &&
      b.role.startsWith("mini-") &&
      Math.abs(a.x - b.x) < 0.04 &&
      Math.abs(a.y - b.y) < 0.70;
    if (!sameCardInternal && !sameMiniInternal && overlap(a, b, 0.01)) {
      warnings.push(`${a.role}:${a.txt.slice(0, 28)} <-> ${b.role}:${b.txt.slice(0, 28)}`);
    }
  }
}

pptx.writeFile({ fileName: outPath })
  .then(() => {
    if (warnings.length) {
      console.warn("Potential text-box overlaps:");
      for (const w of warnings) console.warn("  " + w);
      process.exitCode = 2;
    }
  })
  .catch((err) => {
    console.error(err);
    process.exit(1);
  });
