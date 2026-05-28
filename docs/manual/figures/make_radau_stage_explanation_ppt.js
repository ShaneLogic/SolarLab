const fs = require("fs");
const path = require("path");
const pptxgen = require("/Users/shane/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/pptxgenjs");

const root = path.resolve(__dirname, "../../..");
const outDir = path.join(root, "docs", "manual", "slides");
const outPath = path.join(outDir, "radau_internal_stage_explained.pptx");
const gifPath = path.join(root, "docs", "manual", "figures", "radau_stage_process.gif");
const previewPath = path.join(root, "docs", "manual", "figures", "radau_stage_process_preview.png");
fs.mkdirSync(outDir, { recursive: true });

if (!fs.existsSync(gifPath)) {
  throw new Error(`Missing animation: ${gifPath}. Run make_radau_stage_animation.py first.`);
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
  red: "B42318",
  redLight: "FEE4E2",
  white: "FFFFFF",
  offWhite: "F8FAFC",
  bg: "F7F9FC",
  strip: "EAF2FF",
};

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "SolarLab";
pptx.company = "SolarLab";
pptx.subject = "Radau internal stages in solve_ivp";
pptx.title = "Inside solve_ivp(method='Radau'): What K1-K3 Mean";
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

function sup(txt, opt = {}) {
  return { text: txt, options: { superscript: true, ...opt } };
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
      valign: opt.valign || "top",
      paraSpaceAfterPt: 0,
      paraSpaceBeforePt: 0,
      breakLine: opt.breakLine || false,
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
    },
  });
}

function concat(...parts) {
  return parts.flatMap((p) => (typeof p === "string" ? [{ text: p }] : p));
}

const R = {
  solveIvp: [{ text: "solve_ivp", options: { fontFace: "Courier New" } }],
  Yn: [{ text: "Y" }, sub("n")],
  Yn1: [{ text: "Y" }, sub("n+1")],
  Ki: [{ text: "K" }, sub("i")],
  K1: [{ text: "K" }, sub("1")],
  K2: [{ text: "K" }, sub("2")],
  K3: [{ text: "K" }, sub("3")],
  Kj: [{ text: "K" }, sub("j")],
  tn: [{ text: "t" }, sub("n")],
  ci: [{ text: "c" }, sub("i")],
  aij: [{ text: "a" }, sub("ij")],
  bi: [{ text: "b" }, sub("i")],
  Vk: [{ text: "V" }, sub("k")],
  Vapp: [{ text: "V" }, sub("app")],
  dt: [{ text: "Δt" }],
};

function formulaBox(x, y, w, h) {
  rect(x, y, w, h, C.purpleLight, C.purple, { pt: 1.2, round: 0.08 });
  text("Radau stage equations", x + 0.20, y + 0.15, w - 0.40, 0.22, {
    fontSize: 10.5,
    bold: true,
    color: C.purple,
    role: "formula-label",
  });
  text("Kᵢ = F(tₙ + cᵢh,  Yₙ + hΣⱼ aᵢⱼKⱼ; Vₖ)", x + 0.26, y + 0.52, w - 0.52, 0.34, {
    fontSize: 16.0,
    color: C.navy,
    role: "formula",
  });
  text("Yₙ₊₁ = Yₙ + hΣᵢ bᵢKᵢ", x + 0.26, y + 0.98, w - 0.52, 0.30, {
    fontSize: 16.0,
    color: C.navy,
    role: "formula",
  });
}

function bulletCard(x, y, w, h, label, parts, fill, accent, opt = {}) {
  rect(x, y, w, h, fill, accent, { pt: 1.1, round: 0.07 });
  rect(x, y, 0.08, h, accent, accent, { lineTransparency: 100 });
  text(label, x + 0.18, y + 0.12, w - 0.36, 0.22, {
    fontSize: opt.labelSize || 9.5,
    bold: true,
    color: accent,
    role: "card-label",
  });
  rich(parts, x + 0.18, y + 0.42, w - 0.36, h - 0.52, {
    fontSize: opt.bodySize || 9.8,
    color: C.navy,
    fit: "shrink",
    role: "card-body",
  });
}

function plainCard(x, y, w, h, label, body, fill, accent, opt = {}) {
  rect(x, y, w, h, fill, accent, { pt: 1.1, round: 0.07 });
  rect(x, y, 0.08, h, accent, accent, { lineTransparency: 100 });
  text(label, x + 0.18, y + 0.12, w - 0.36, 0.20, {
    fontSize: opt.labelSize || 9.5,
    bold: true,
    color: accent,
    role: "card-label",
  });
  text(body, x + 0.18, y + 0.40, w - 0.36, h - 0.50, {
    fontSize: opt.bodySize || 9.5,
    color: C.navy,
    fit: "shrink",
    role: "card-body",
  });
}

function flowStep(x, y, w, label, body, fill, accent) {
  rect(x, y, w, 0.42, fill, accent, { pt: 1.0, round: 0.06 });
  text(label, x + 0.16, y + 0.09, w - 0.32, 0.12, {
    fontSize: 7.8,
    bold: true,
    color: accent,
    role: "flow-label",
  });
  text(body, x + 0.16, y + 0.25, w - 0.32, 0.10, {
    fontSize: 7.8,
    color: C.navy,
    role: "flow-body",
  });
}

// Header
text("Inside solve_ivp(method='Radau'): What K1-K3 Mean", 0.46, 0.22, 12.35, 0.36, {
  fontSize: 21.5,
  bold: true,
  role: "title",
});
rich(concat(
  "A J-V voltage plateau is one IVP: hold Vₖ",
  " fixed, then Radau advances the state through many internal steps."
), 0.47, 0.66, 12.1, 0.25, {
  fontSize: 12.2,
  color: C.slate,
  role: "subtitle",
});

// Left: animation
rect(0.48, 1.05, 7.48, 5.25, C.white, C.line, { round: 0.08 });
text("Animated View: One Internal Radau Step", 0.75, 1.22, 4.1, 0.25, {
  fontSize: 12.6,
  bold: true,
  color: C.slate,
  role: "panel-title",
});
text("GIF animation; use slideshow mode for playback.", 5.05, 1.24, 2.58, 0.18, {
  fontSize: 8.6,
  color: C.muted,
  align: "right",
  role: "side-note",
});
slide.addImage({
  path: gifPath,
  x: 0.78,
  y: 1.55,
  w: 6.88,
  h: 3.87,
  sizing: { type: "contain", x: 0.78, y: 1.55, w: 6.88, h: 3.87 },
});
rich(concat(
  "One outer plateau time ", R.dt,
  " contains many adaptive internal steps ", "h", ". Each internal step solves a coupled set of stages."
), 0.78, 5.62, 6.88, 0.30, {
  fontSize: 10.6,
  color: C.slate,
  role: "left-caption",
});

// Right: formula and parameter legend
rect(8.20, 1.05, 4.64, 5.25, C.white, C.line, { round: 0.08 });
text("Equation And Symbols", 8.48, 1.22, 3.5, 0.25, {
  fontSize: 12.6,
  bold: true,
  color: C.slate,
  role: "panel-title",
});
formulaBox(8.46, 1.56, 4.10, 1.52);

plainCard(8.46, 3.28, 1.90, 0.68, "STATE", "Y_n: current state\nY_n+1: updated state", C.blueLight, C.blue, { bodySize: 8.9 });
plainCard(10.66, 3.28, 1.90, 0.68, "STEP SIZE", "h: Radau internal\nadaptive time step", C.amberLight, C.amber, { bodySize: 8.9 });
plainCard(8.46, 4.12, 1.90, 0.78, "STAGES", "K1, K2, K3:\ninternal slopes from F", C.purpleLight, C.purple, { bodySize: 8.9 });
plainCard(10.66, 4.12, 1.90, 0.78, "WEIGHTS", "c_i: stage locations\na_ij: coupling\nb_i: final average", C.greenLight, C.green, { bodySize: 8.3 });

rect(8.46, 5.08, 4.10, 0.82, C.offWhite, C.line, { round: 0.07 });
text("Why it is implicit", 8.67, 5.22, 3.68, 0.18, {
  fontSize: 9.8,
  bold: true,
  color: C.red,
  role: "implicit-label",
});
rich(concat(
  "Every K_i depends on all K_j. Newton iteration solves the coupled stages together, giving stability for stiff drift-diffusion dynamics."
), 8.67, 5.46, 3.68, 0.25, {
  fontSize: 9.0,
  color: C.navy,
  role: "implicit-body",
});

// Flow strip
rect(0.48, 6.46, 12.36, 0.62, C.white, C.line, { round: 0.08 });
const flowY = 6.61;
flowStep(0.76, 6.56, 2.15, "1  FIX VOLTAGE", "V_app = V_k", C.blueLight, C.blue);
arrow(3.04, flowY + 0.08, 3.42, flowY + 0.08, C.muted, { pt: 1.2 });
flowStep(3.52, 6.56, 2.15, "2  SOLVE IVP", "dY/dt = F(Y; V_k)", C.purpleLight, C.purple);
arrow(5.80, flowY + 0.08, 6.18, flowY + 0.08, C.muted, { pt: 1.2 });
flowStep(6.28, 6.56, 2.15, "3  UPDATE STATE", "Y_n -> Y_n+1", C.greenLight, C.green);
arrow(8.56, flowY + 0.08, 8.94, flowY + 0.08, C.muted, { pt: 1.2 });
flowStep(9.04, 6.56, 3.48, "4  RECORD CURRENT", "compute J(V_k) after settle", C.amberLight, C.amber);

// Bottom takeaway
slide.addShape(pptx.ShapeType.rect, {
  x: 0,
  y: 7.18,
  w: 13.333,
  h: 0.32,
  fill: { color: C.strip },
  line: { color: C.strip, transparency: 100 },
});
rich(concat(
  [{ text: "Takeaway: ", options: { bold: true, color: C.amber } }],
  "K1, K2, K3 are not voltage samples; they are coupled internal slopes used to advance one Radau step from Y_n to Y_n+1."
), 0.55, 7.27, 12.22, 0.12, {
  fontSize: 9.4,
  color: C.navy,
  align: "center",
  role: "takeaway",
});

slide.addNotes(`Suggested narration:
This slide zooms into one internal step inside solve_ivp(method='Radau'). In the J-V sweep, the outer loop first holds the applied voltage V_k fixed and defines an initial-value problem dY/dt = F(Y; V_k). Radau then advances this IVP using adaptive internal steps h.

Within one internal step, K1, K2, and K3 are not three measured J-V points. They are three implicit Runge-Kutta stage slopes. The stages are located at different positions within the interval [t_n, t_n + h], and each stage state contains a weighted combination of all stage slopes. This coupling is why the method is implicit.

SciPy's Radau method solves the coupled stage equations with Newton iterations, repeatedly evaluating the drift-diffusion RHS F(Y). After the stage slopes are self-consistent, the weighted average of the stages updates Y_n to Y_{n+1}. This implicit coupling is useful for stiff carrier transport, recombination, electrostatics, and slow ionic relaxation.`);

function overlap(a, b, pad = 0.012) {
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
    const sameCard =
      a.role.startsWith("card-") &&
      b.role.startsWith("card-") &&
      Math.abs(a.x - b.x) < 0.05 &&
      Math.abs(a.w - b.w) < 0.05;
    if (!sameCard && overlap(a, b)) {
      warnings.push(`${a.role}:${a.txt.slice(0, 34)} <-> ${b.role}:${b.txt.slice(0, 34)}`);
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
