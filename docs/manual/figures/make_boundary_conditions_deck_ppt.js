const fs = require("fs");
const path = require("path");
const pptxgen = require("/Users/shane/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/pptxgenjs");

const root = path.resolve(__dirname, "../../..");
const outDir = path.join(root, "docs", "manual", "slides");
const outPath = process.env.OUT_PPTX
  ? path.resolve(process.env.OUT_PPTX)
  : path.join(outDir, "2d_boundary_conditions_deck.pptx");
const onlySlide = process.env.ONLY_SLIDE ? Number(process.env.ONLY_SLIDE) : 0;
fs.mkdirSync(outDir, { recursive: true });

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "SolarLab";
pptx.company = "SolarLab";
pptx.subject = "2D boundary conditions";
pptx.title = "2D Boundary Conditions in SolarLab";
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

const SYM = {
  phi: "φ",
  n: "n",
  p: "p",
  Jn: "Jₙ",
  Jp: "Jₚ",
  Jx: "Jₓ",
  Jy: "Jᵧ",
  Sn: "Sₙ",
  Sp: "Sₚ",
  Vapp: "Vₐₚₚ",
  Vbi: "Vᵦᵢ",
  Ly: "Lᵧ",
  Lx: "Lₓ",
  neq: "nₑq",
  peq: "pₑq",
  neqAscii: "n_eq",
  peqAscii: "p_eq",
  Cs: "cₛ",
  Ceq: "cₑq",
  chi: "χ",
  Eg: "E₉",
};

function addSlide(title, subtitle, takeaway) {
  const slide = pptx.addSlide();
  slide.background = { color: C.bg };
  const boxes = [];

  function text(txt, x, y, w, h, opt = {}) {
    boxes.push({ txt, x, y, w, h, role: opt.role || "text" });
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
      paraSpaceAfterPt: 0,
      paraSpaceBeforePt: 0,
    });
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
      x: x1, y: y1, w: x2 - x1, h: y2 - y1,
      line: {
        color,
        pt: opt.pt || 1.25,
        beginArrowType: opt.begin ? "triangle" : "none",
        endArrowType: opt.noHead ? "none" : "triangle",
        dash: opt.dash || "solid",
        transparency: opt.transparency || 0,
      },
    });
  }

  function card(x, y, w, h, label, body, fill, accent, opt = {}) {
    rect(x, y, w, h, fill, accent, { pt: opt.pt || 1.1, round: 0.08 });
    rect(x, y, 0.08, h, accent, accent, { lineTransparency: 100 });
    text(label, x + 0.18, y + 0.12, w - 0.32, opt.labelH || 0.22, {
      fontSize: opt.labelSize || 9.0,
      bold: true,
      color: accent,
      role: "card-label",
    });
    text(body, x + 0.18, y + (opt.bodyY || 0.42), w - 0.34, h - (opt.bodyY || 0.42) - 0.08, {
      fontSize: opt.bodySize || 9.0,
      color: C.navy,
      fit: "shrink",
      role: "card-body",
    });
  }

  function pill(x, y, w, label, fill, accent, opt = {}) {
    rect(x, y, w, opt.h || 0.32, fill, accent, { pt: 0.8, round: 0.12 });
    text(label, x + 0.05, y + 0.08, w - 0.10, 0.12, {
      fontSize: opt.fontSize || 8.2,
      bold: true,
      color: accent,
      align: "center",
      role: "pill",
    });
  }

  function device(x, y, w, h, opt = {}) {
    rect(x, y, w, h, C.white, C.line, { pt: 1.0, round: 0.06 });
    const h1 = opt.h1 || h * 0.22;
    const h2 = opt.h2 || h * 0.56;
    const h3 = h - h1 - h2;
    rect(x, y, w, h1, C.greenLight, C.line, { pt: 0.6 });
    rect(x, y + h1, w, h2, C.blueLight, C.line, { pt: 0.6 });
    rect(x, y + h1 + h2, w, h3, C.amberLight, C.line, { pt: 0.6 });
    text("HTL", x + 0.18, y + h1 / 2 - 0.08, 0.72, 0.16, { fontSize: 8.4, bold: true, color: C.green, role: "dev-label" });
    text("absorber", x + 0.18, y + h1 + h2 / 2 - 0.08, 1.05, 0.16, { fontSize: 8.4, bold: true, color: C.blue, role: "dev-label" });
    text("ETL", x + 0.18, y + h1 + h2 + h3 / 2 - 0.08, 0.72, 0.16, { fontSize: 8.4, bold: true, color: C.amber, role: "dev-label" });
    if (opt.grid) {
      for (let i = 1; i < 6; i++) line(x + (w * i) / 6, y, x + (w * i) / 6, y + h, C.line, { pt: 0.45, noHead: true });
      for (let j = 1; j < 4; j++) line(x, y + (h * j) / 4, x + w, y + (h * j) / 4, C.line, { pt: 0.45, noHead: true });
    }
  }

  function overlap(a, b, pad = 0.01) {
    return !(
      a.x + a.w + pad <= b.x ||
      b.x + b.w + pad <= a.x ||
      a.y + a.h + pad <= b.y ||
      b.y + b.h + pad <= a.y
    );
  }

  text(title, 0.46, 0.22, 12.35, 0.34, { fontSize: 21.0, bold: true, role: "title" });
  text(subtitle, 0.47, 0.68, 12.05, 0.26, { fontSize: 11.2, color: C.slate, role: "subtitle" });
  slide.addShape(pptx.ShapeType.rect, {
    x: 0, y: 6.93, w: 13.333, h: 0.57,
    fill: { color: C.strip },
    line: { color: C.strip, transparency: 100 },
  });
  text(takeaway, 0.50, 7.08, 12.25, 0.20, { fontSize: 10.6, bold: true, color: C.navy, align: "center", role: "takeaway" });

  return { slide, text, rect, line, card, pill, device, boxes, overlap };
}

function addNotes(slide, note) {
  slide.addNotes(note);
}

// ---------------------------------------------------------------------------
// Slide 1. BC map.
// ---------------------------------------------------------------------------
if (!onlySlide || onlySlide === 1) {
  const s = addSlide(
    "Boundary Condition Map: Where BCs Enter the 2D Solver",
    "First separate boundary conditions by physical location: electrode contacts, lateral domain edges, and internal material interfaces.",
    "Takeaway: contact BC, lateral BC, and internal interface treatment are different objects and should not be explained in one crowded diagram.",
  );
  const { rect, text, line, card, pill, device, slide } = s;
  rect(0.48, 1.10, 12.36, 5.48, C.white, C.line, { round: 0.08 });

  device(4.36, 1.68, 4.62, 3.76, { grid: true });
  text("2D drift-diffusion domain", 5.48, 1.36, 2.38, 0.22, { fontSize: 11.0, bold: true, color: C.slate, align: "center", role: "map-title" });

  pill(4.68, 1.22, 3.98, `Poisson: ${SYM.phi}(0,x)=0, ${SYM.phi}(${SYM.Ly},x)=${SYM.Vbi}-${SYM.Vapp}`, C.greenLight, C.green, { fontSize: 7.2, h: 0.30 });
  pill(4.78, 5.68, 3.78, "carrier contacts: ohmic Dirichlet or Robin S", C.purpleLight, C.purple, { fontSize: 7.6, h: 0.30 });

  line(4.16, 3.05, 3.40, 3.05, C.blue, { begin: true });
  line(9.18, 3.05, 9.94, 3.05, C.blue);
  text("x=0", 3.77, 3.18, 0.38, 0.14, { fontSize: 7.5, color: C.muted, align: "right", role: "axis" });
  text(`x=${SYM.Lx}`, 9.20, 3.18, 0.55, 0.14, { fontSize: 7.5, color: C.muted, role: "axis" });
  line(4.22, 1.95, 4.22, 5.30, C.slate, { pt: 1.0, noHead: true });
  text("y transport direction", 3.18, 2.08, 0.80, 0.32, { fontSize: 7.5, color: C.slate, align: "right", role: "axis" });

  // Internal interfaces.
  line(4.36, 2.48, 8.98, 2.48, C.red, { pt: 1.2, noHead: true, dash: "dash" });
  line(4.36, 4.48, 8.98, 4.48, C.red, { pt: 1.2, noHead: true, dash: "dash" });
  text("internal material interfaces", 6.00, 4.60, 1.60, 0.14, { fontSize: 7.2, color: C.red, align: "center", role: "interface-label" });

  card(0.84, 1.64, 2.70, 1.06, "1. Contact BC", "outer electrode rows\ny=0 and y=Ly\ncontrols carrier exchange", C.purpleLight, C.purple, { bodySize: 8.3 });
  line(3.58, 2.17, 4.36, 1.62, C.purple, { pt: 1.0, dash: "dash" });
  line(3.58, 2.17, 4.36, 5.46, C.purple, { pt: 1.0, dash: "dash" });

  card(0.84, 3.02, 2.70, 1.06, "2. Lateral BC", "left and right domain edges\nperiodic or Neumann\nselected by lateral_bc", C.blueLight, C.blue, { bodySize: 8.3 });
  line(3.58, 3.55, 4.36, 3.05, C.blue, { pt: 1.0 });

  card(0.84, 4.40, 2.70, 1.06, "3. Interface Treatment", "inside the stack\nband offsets and recombination\nnot the same as contact BC", C.redLight, C.red, { bodySize: 8.0 });
  line(3.58, 4.92, 4.36, 4.48, C.red, { pt: 1.0 });

  card(9.78, 1.64, 2.72, 1.08, "Fixed in 2D Poisson", `${SYM.phi}=0 and ${SYM.phi}=${SYM.Vbi}-${SYM.Vapp}\nDirichlet electrode potentials`, C.greenLight, C.green, { bodySize: 8.2 });
  line(9.62, 2.18, 8.98, 1.62, C.green, { pt: 1.0, dash: "dash" });
  line(9.62, 2.18, 8.98, 5.46, C.green, { pt: 1.0, dash: "dash" });

  card(9.78, 3.02, 2.72, 1.08, "Configurable in 2D Run", "`lateral_bc`\nperiodic or neumann\nused by Poisson + continuity", C.blueLight, C.blue, { bodySize: 8.2 });
  line(9.62, 3.55, 8.98, 3.05, C.blue, { pt: 1.0 });

  card(9.78, 4.40, 2.72, 1.08, "Configurable by S", `${SYM.Sn}, ${SYM.Sp} fields\nmissing: ohmic pin\nfinite: Robin contact`, C.purpleLight, C.purple, { bodySize: 8.2 });
  line(9.62, 4.94, 8.98, 5.44, C.purple, { pt: 1.0 });

  addNotes(slide, "This opening slide is a location map. It deliberately avoids detailed formulas. The audience should first learn that SolarLab has three different BC-related concepts: contact boundary conditions at the electrodes, lateral boundary conditions at x edges, and internal interface treatment between layers.");
}

// ---------------------------------------------------------------------------
// Slide 2. Contact BC.
// ---------------------------------------------------------------------------
if (!onlySlide || onlySlide === 2) {
  const s = addSlide(
    "Contact Boundary Conditions: Ohmic, Blocking, and Robin",
    "Contact BCs describe how the external electrodes exchange electrons and holes with the simulated device boundary rows.",
    "Takeaway: SolarLab switches contact behavior through the S fields; missing S gives the ohmic Dirichlet pin, S=0 blocks flux, and finite S gives Robin selective contact.",
  );
  const { rect, text, line, card, pill, slide } = s;
  rect(0.48, 1.10, 12.36, 5.48, C.white, C.line, { round: 0.08 });

  const cols = [
    { x: 0.78, accent: C.green, fill: C.greenLight, title: "Ohmic Dirichlet", mode: "missing or null S", formula: `${SYM.n}=${SYM.neq},  ${SYM.p}=${SYM.peq}`, body: "ideal carrier reservoir\nboundary rows are pinned\nRHS sets dn/dt and dp/dt to zero" },
    { x: 4.57, accent: C.red, fill: C.redLight, title: "Blocking Neumann", mode: "explicit S = 0", formula: `${SYM.Jn}=0,  ${SYM.Jp}=0`, body: "no carrier flux through contact\nboundary density can evolve\nperfect blocking limit" },
    { x: 8.36, accent: C.purple, fill: C.purpleLight, title: "Robin Selective Contact", mode: "finite S in mode=full", formula: `Jcontact = ± q S (${SYM.Cs}-${SYM.Ceq})`, body: "finite exchange rate\nmatched carrier has large S\nwrong-sign carrier has small S" },
  ];

  for (const c of cols) {
    rect(c.x, 1.42, 3.18, 4.80, C.offWhite, C.line, { round: 0.08 });
    text(c.title, c.x + 0.22, 1.60, 2.70, 0.25, { fontSize: 12.0, bold: true, color: c.accent, align: "center", role: "col-title" });
    pill(c.x + 0.56, 1.96, 2.06, c.mode, c.fill, c.accent, { fontSize: 7.2 });

    // Contact cartoon: metal at left, semiconductor at right.
    rect(c.x + 0.34, 2.48, 0.58, 1.75, "E2E8F0", C.slate, { pt: 0.8 });
    rect(c.x + 0.92, 2.48, 1.88, 1.75, C.blueLight, C.blue, { pt: 1.0 });
    text("electrode", c.x + 0.35, 4.34, 0.65, 0.14, { fontSize: 7.0, color: C.slate, align: "center", role: "mini" });
    text("semiconductor", c.x + 1.18, 4.34, 1.36, 0.14, { fontSize: 7.0, color: C.blue, align: "center", role: "mini" });
    line(c.x + 0.92, 2.48, c.x + 0.92, 4.23, c.accent, { pt: 2.2, noHead: true });

    if (c.title === "Ohmic Dirichlet") {
      line(c.x + 1.12, 3.12, c.x + 2.55, 3.12, C.green, { pt: 1.4, noHead: true });
      line(c.x + 1.12, 3.55, c.x + 2.55, 3.55, C.green, { pt: 1.4, noHead: true });
      text("fixed boundary value", c.x + 1.08, 2.72, 1.60, 0.14, { fontSize: 7.2, color: C.green, align: "center", role: "mini" });
    } else if (c.title === "Blocking Neumann") {
      line(c.x + 0.72, 3.16, c.x + 0.72, 3.62, C.red, { pt: 2.0, noHead: true });
      line(c.x + 0.62, 3.16, c.x + 0.82, 3.16, C.red, { pt: 2.0, noHead: true });
      line(c.x + 0.62, 3.62, c.x + 0.82, 3.62, C.red, { pt: 2.0, noHead: true });
      text("no flux", c.x + 1.50, 3.22, 0.90, 0.14, { fontSize: 7.6, color: C.red, align: "center", role: "mini" });
    } else {
      line(c.x + 1.08, 3.12, c.x + 0.76, 3.12, C.purple, { pt: 1.5, begin: true });
      line(c.x + 0.76, 3.62, c.x + 1.08, 3.62, C.purple, { pt: 1.5 });
      text("finite exchange", c.x + 1.28, 3.27, 1.26, 0.14, { fontSize: 7.2, color: C.purple, align: "center", role: "mini" });
    }

    card(c.x + 0.36, 4.72, 2.46, 0.92, "BC equation", `${c.formula}\n${c.body}`, C.white, c.accent, { bodySize: 7.5, labelSize: 7.2, bodyY: 0.33 });
  }

  card(1.20, 5.86, 10.94, 0.36, "SolarLab code mapping", `Top/HTL: S_n_left, S_p_left     Bottom/ETL: S_n_right, S_p_right     Activation: mode = full`, C.offWhite, C.slate, { bodySize: 7.3, labelSize: 7.0, bodyY: 0.18 });

  addNotes(slide, "This is the contact BC slide. Explain that the figure is a local magnification of the electrode-device boundary, not the entire device. In SolarLab 2D, the same concept is applied along the full top and bottom boundary rows.");
}

// ---------------------------------------------------------------------------
// Slide 3. Lateral BC.
// ---------------------------------------------------------------------------
if (!onlySlide || onlySlide === 3) {
  const s = addSlide(
    "Lateral Boundary Conditions: Periodic Cell vs Zero-Flux Wall",
    "The lateral boundary controls what happens at x=0 and x=Lx; it is selected by the 2D run parameter lateral_bc.",
    "Takeaway: periodic represents a repeating lateral unit cell, while Neumann represents a closed sidewall with zero normal flux.",
  );
  const { rect, text, line, card, pill, device, slide } = s;
  rect(0.48, 1.10, 12.36, 5.48, C.white, C.line, { round: 0.08 });

  // Periodic side.
  rect(0.82, 1.44, 5.72, 4.80, C.offWhite, C.line, { round: 0.08 });
  text("Periodic Lateral Boundary", 1.08, 1.62, 3.2, 0.25, { fontSize: 12.2, bold: true, color: C.blue, role: "section" });
  pill(4.55, 1.64, 1.38, 'lateral_bc="periodic"', C.blueLight, C.blue, { fontSize: 6.9 });
  device(1.32, 2.32, 4.70, 2.38, { grid: true, h1: 0.50, h2: 1.36 });
  line(1.33, 2.70, 6.02, 2.70, C.blue, { pt: 2.0, begin: true });
  line(1.33, 3.74, 6.02, 3.74, C.blue, { pt: 2.0, begin: true });
  text("right edge wraps to left edge", 2.72, 2.08, 1.95, 0.16, { fontSize: 8.2, bold: true, color: C.blue, align: "center", role: "label" });
  card(1.36, 5.04, 4.58, 0.76, "Physical interpretation", "simulate one representative lateral period; no artificial sidewall is introduced", C.blueLight, C.blue, { bodySize: 8.0, labelSize: 7.5, bodyY: 0.30 });

  // Neumann side.
  rect(6.78, 1.44, 5.72, 4.80, C.offWhite, C.line, { round: 0.08 });
  text("Neumann Lateral Boundary", 7.04, 1.62, 3.2, 0.25, { fontSize: 12.2, bold: true, color: C.red, role: "section" });
  pill(10.52, 1.64, 1.38, 'lateral_bc="neumann"', C.redLight, C.red, { fontSize: 6.9 });
  device(7.28, 2.32, 4.70, 2.38, { grid: true, h1: 0.50, h2: 1.36 });
  line(7.28, 2.32, 7.28, 4.70, C.red, { pt: 3.0, noHead: true });
  line(11.98, 2.32, 11.98, 4.70, C.red, { pt: 3.0, noHead: true });
  text(`${SYM.Jx}=0`, 7.54, 3.34, 0.60, 0.16, { fontSize: 9.0, bold: true, color: C.red, role: "jzero" });
  text(`${SYM.Jx}=0`, 11.08, 3.34, 0.60, 0.16, { fontSize: 9.0, bold: true, color: C.red, role: "jzero" });
  card(7.32, 5.04, 4.58, 0.76, "Physical interpretation", "simulate an isolated lateral window; no carrier current or electric displacement leaves through sidewalls", C.redLight, C.red, { bodySize: 7.7, labelSize: 7.5, bodyY: 0.30 });

  card(2.28, 5.86, 8.78, 0.36, "Solver usage", "The same lateral_bc is passed into Poisson assembly and continuity divergence for a consistent x-edge convention.", C.offWhite, C.slate, { bodySize: 7.2, labelSize: 7.0, bodyY: 0.18 });

  addNotes(slide, "This slide should be very simple. The audience only needs to know that lateral_bc is a domain-edge choice. It is not the electrode contact model.");
}

// ---------------------------------------------------------------------------
// Slide 4. Internal interface.
// ---------------------------------------------------------------------------
if (!onlySlide || onlySlide === 4) {
  const s = addSlide(
    "Internal Material Interfaces: Not the Same as Contact BC",
    "Internal interfaces occur between layers such as HTL/absorber or absorber/ETL; they modify transport inside the device rather than at the external electrode.",
    "Takeaway: contact BC is an outer-boundary exchange model; internal interfaces are handled through band-offset-aware fluxes, thermionic capping, and optional interface recombination.",
  );
  const { rect, text, line, card, pill, slide } = s;
  rect(0.48, 1.10, 12.36, 5.48, C.white, C.line, { round: 0.08 });

  // Main interface cartoon.
  rect(0.88, 1.50, 7.12, 4.54, C.offWhite, C.line, { round: 0.08 });
  text("Local view around an internal interface", 1.16, 1.68, 3.30, 0.22, { fontSize: 11.5, bold: true, color: C.slate, role: "main-title" });
  rect(1.30, 2.18, 2.04, 2.76, "F7DDE2", C.line, { pt: 0.6 });
  rect(3.34, 2.18, 1.28, 2.76, "FFF2D8", C.line, { pt: 0.6 });
  rect(4.62, 2.18, 2.58, 2.76, "D7F0FA", C.line, { pt: 0.6 });
  text("Material 1", 1.76, 2.34, 1.0, 0.16, { fontSize: 8.2, color: C.slate, align: "center", role: "mat" });
  text("interface region", 3.45, 2.34, 1.04, 0.16, { fontSize: 8.2, color: C.amber, align: "center", role: "mat" });
  text("Material 2", 5.32, 2.34, 1.05, 0.16, { fontSize: 8.2, color: C.slate, align: "center", role: "mat" });
  line(3.34, 2.18, 3.34, 4.94, C.amber, { pt: 1.2, noHead: true, dash: "dash" });
  line(4.62, 2.18, 4.62, 4.94, C.amber, { pt: 1.2, noHead: true, dash: "dash" });

  // Carrier profiles drawn as polylines.
  const pPts = [
    [1.42, 2.65], [2.20, 2.78], [3.20, 2.94], [3.80, 3.55], [4.58, 4.62], [7.10, 4.62],
  ];
  const nPts = [
    [1.42, 4.70], [2.30, 4.48], [3.20, 4.26], [3.82, 3.76], [4.62, 3.28], [7.10, 3.20],
  ];
  for (let i = 0; i < pPts.length - 1; i++) line(pPts[i][0], pPts[i][1], pPts[i + 1][0], pPts[i + 1][1], C.red, { pt: 1.7, noHead: true });
  for (let i = 0; i < nPts.length - 1; i++) line(nPts[i][0], nPts[i][1], nPts[i + 1][0], nPts[i + 1][1], C.blue, { pt: 1.7, noHead: true });
  text("p(y)", 2.48, 2.84, 0.42, 0.14, { fontSize: 8.2, italic: true, color: C.red, role: "profile-label" });
  text("n(y)", 5.14, 3.04, 0.42, 0.14, { fontSize: 8.2, italic: true, color: C.blue, role: "profile-label" });
  line(2.70, 3.58, 3.28, 3.58, C.red, { pt: 1.2 });
  text("interface flux", 2.08, 3.46, 0.70, 0.14, { fontSize: 7.0, color: C.red, align: "right", role: "flux" });
  line(5.28, 3.68, 4.72, 3.68, C.blue, { pt: 1.2 });
  text("interface flux", 5.38, 3.56, 0.78, 0.14, { fontSize: 7.0, color: C.blue, role: "flux" });
  text("carrier density", 1.12, 1.96, 1.40, 0.14, { fontSize: 7.2, color: C.muted, role: "axis" });
  line(1.18, 4.98, 7.26, 4.98, C.slate, { pt: 1.0 });
  text("y transport coordinate", 3.40, 5.16, 1.68, 0.15, { fontSize: 7.6, color: C.slate, align: "center", role: "axis" });

  // Explanation cards.
  card(8.34, 1.50, 3.76, 1.05, "What this slide is about", "internal layer-layer interface\ninside the device domain\nnot the metal electrode contact", C.redLight, C.red, { bodySize: 8.3 });
  card(8.34, 2.80, 3.76, 1.05, "Transport treatment", "SG flux uses electron-affinity and bandgap offsets\nthermionic-emission capping can limit large interface fluxes", C.blueLight, C.blue, { bodySize: 7.8 });
  card(8.34, 4.10, 3.76, 1.05, "Recombination treatment", "1D path supports interface recombination velocities; 2D Stage-B focus uses material maps and interface-aware fluxes.", C.greenLight, C.green, { bodySize: 7.8 });

  rect(8.34, 5.48, 3.76, 0.48, C.offWhite, C.line, { pt: 0.8, round: 0.05 });
  text("Do not label this as contact BC unless the interface is the external electrode.", 8.56, 5.64, 3.32, 0.12, { fontSize: 7.8, color: C.muted, align: "center", role: "warning" });

  addNotes(slide, "This slide responds to the reference figure. The reference figure is best interpreted as an internal interface transport model, not as SolarLab's outer contact BC. Use it after the contact and lateral BC slides to avoid mixing concepts.");
}

// Geometry warning pass.
function overlap(a, b, pad = 0.01) {
  return !(
    a.x + a.w + pad <= b.x ||
    b.x + b.w + pad <= a.x ||
    a.y + a.h + pad <= b.y ||
    b.y + b.h + pad <= a.y
  );
}

// pptxgenjs has no global boxes; the per-slide builders track locally but we
// intentionally keep warnings visual via Quick Look in the generation workflow.
pptx.writeFile({ fileName: outPath })
  .then(() => console.log(outPath))
  .catch((err) => {
    console.error(err);
    process.exit(1);
  });
