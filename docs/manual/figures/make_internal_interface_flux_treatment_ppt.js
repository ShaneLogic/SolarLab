const fs = require("fs");
const path = require("path");
const pptxgen = require("/Users/shane/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/pptxgenjs");

const root = path.resolve(__dirname, "../../..");
const outDir = path.join(root, "docs", "manual", "slides");
const outPath = path.join(outDir, "internal_interface_flux_treatment_2d.pptx");
fs.mkdirSync(outDir, { recursive: true });

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "SolarLab";
pptx.company = "SolarLab";
pptx.subject = "2D internal interface flux treatment";
pptx.title = "2D Internal Interface: Conservative Heterointerface Flux";
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
  red: "B42318",
  redLight: "FEE4E2",
  purple: "6D28D9",
  purpleLight: "EDE9FE",
  white: "FFFFFF",
  offWhite: "F8FAFC",
  bg: "F7F9FC",
  strip: "EAF2FF",
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
    paraSpaceAfterPt: opt.paraSpaceAfterPt || 0,
    paraSpaceBeforePt: 0,
  });
}

function richText(parts, x, y, w, h, opt = {}) {
  const runs = parts.flatMap((p) => (typeof p === "string" ? [{ text: p }] : p));
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
      options: { ...base, ...(r.options || {}) },
    })),
    {
      x,
      y,
      w,
      h,
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

function sub(txt, opt = {}) {
  return { text: txt, options: { subscript: true, ...opt } };
}

function sup(txt, opt = {}) {
  return { text: txt, options: { superscript: true, ...opt } };
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

function line(x1, y1, x2, y2, color = C.muted, opt = {}) {
  slide.addShape(pptx.ShapeType.line, {
    x: x1,
    y: y1,
    w: x2 - x1,
    h: y2 - y1,
    line: {
      color,
      pt: opt.pt || 1.2,
      beginArrowType: opt.beginArrow || "none",
      endArrowType: opt.noHead ? "none" : opt.endArrow || "triangle",
      dash: opt.dash || "solid",
      transparency: opt.transparency || 0,
    },
  });
}

function label(x, y, w, txt, color = C.muted, opt = {}) {
  text(txt, x, y, w, opt.h || 0.16, {
    fontSize: opt.fontSize || 7.6,
    color,
    bold: opt.bold || false,
    italic: opt.italic || false,
    align: opt.align || "left",
    role: opt.role || "label",
  });
}

function panelTitle(txt, x, y, w) {
  text(txt, x, y, w, 0.24, {
    fontSize: 11.3,
    bold: true,
    color: C.slate,
    role: "panel-title",
  });
}

function pill(x, y, w, txt, fill, accent) {
  rect(x, y, w, 0.31, fill, accent, { pt: 0.8, round: 0.1 });
  text(txt, x + 0.05, y + 0.075, w - 0.10, 0.11, {
    fontSize: 7.5,
    bold: true,
    color: accent,
    align: "center",
    role: "pill",
  });
}

function card(x, y, w, h, title, body, fill, accent) {
  rect(x, y, w, h, fill, accent, { pt: 1.0, round: 0.08 });
  rect(x, y, 0.08, h, accent, accent, { lineTransparency: 100 });
  text(title, x + 0.18, y + 0.12, w - 0.32, 0.20, {
    fontSize: 8.2,
    bold: true,
    color: accent,
    role: "card-title",
  });
  text(body, x + 0.18, y + 0.39, w - 0.34, h - 0.46, {
    fontSize: 8.0,
    color: C.navy,
    role: "card-body",
  });
}

function fluxPair(x, y, opt = {}) {
  const color = opt.color || C.navy;
  const fs = opt.fontSize || 10.0;
  const subFs = opt.subFontSize || 6.2;
  const subDy = opt.subDy || 0.075;
  text("J", x, y, 0.15, 0.18, {
    fontSize: fs,
    bold: true,
    color,
    role: "math-main",
  });
  text("n,f", x + 0.12, y + subDy, 0.27, 0.10, {
    fontSize: subFs,
    bold: true,
    color,
    role: "math-sub",
  });
  text(",", x + 0.39, y + 0.02, 0.07, 0.14, {
    fontSize: fs * 0.85,
    bold: true,
    color,
    role: "math-comma",
  });
  text("J", x + 0.56, y, 0.15, 0.18, {
    fontSize: fs,
    bold: true,
    color,
    role: "math-main",
  });
  text("p,f", x + 0.68, y + subDy, 0.27, 0.10, {
    fontSize: subFs,
    bold: true,
    color,
    role: "math-sub",
  });
}

function richCard(x, y, w, h, title, body, fill, accent) {
  rect(x, y, w, h, fill, accent, { pt: 1.0, round: 0.08 });
  rect(x, y, 0.08, h, accent, accent, { lineTransparency: 100 });
  text(title, x + 0.18, y + 0.12, w - 0.32, 0.20, {
    fontSize: 8.2,
    bold: true,
    color: accent,
    role: "card-title",
  });
  richText(body, x + 0.18, y + 0.39, w - 0.34, h - 0.46, {
    fontSize: 8.0,
    color: C.navy,
    role: "card-body",
  });
}

function miniBox(x, y, w, h, title, fill, accent) {
  rect(x, y, w, h, fill, accent, { pt: 0.9, round: 0.06 });
  text(title, x + 0.12, y + 0.09, w - 0.24, 0.18, {
    fontSize: 7.8,
    bold: true,
    color: accent,
    role: "card-title",
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

// Header
text("2D Internal Interface: Band-Offset-Aware SG Face Flux", 0.46, 0.22, 12.00, 0.38, {
  fontSize: 22.0,
  bold: true,
  role: "title",
});
text("The interface is an internal finite-volume face: band parameters create a jump in the SG driving potential, and the resulting face flux enters the divergence conservatively.", 0.47, 0.68, 12.10, 0.28, {
  fontSize: 11.3,
  color: C.slate,
  role: "subtitle",
});

rect(0.45, 1.05, 12.43, 5.72, C.white, C.line, { round: 0.08 });

// Panel A: finite-volume stencil.
rect(0.75, 1.30, 4.03, 5.18, C.offWhite, C.line, { pt: 1.0, round: 0.08 });
panelTitle("A. Internal Face in the 2D Grid", 1.00, 1.46, 3.40);
label(3.52, 1.50, 0.86, "y-direction", C.muted, { align: "right" });

const ax = 1.12;
const ay = 2.00;
const aw = 3.25;
const ahTop = 1.44;
const faceY = ay + ahTop;
const ahBot = 1.44;
const fluxX = ax + aw * 0.56;

rect(ax, ay, aw, ahTop, C.redLight, C.red, { pt: 0.85, round: 0.04, fillTransparency: 6 });
rect(ax, faceY, aw, ahBot, "E0F2FE", C.blue, { pt: 0.85, round: 0.04, fillTransparency: 4 });
line(ax, faceY, ax + aw, faceY, C.slate, { pt: 2.1, noHead: true });

text("Material A, row j", ax + 0.16, ay + 0.16, 1.35, 0.18, {
  fontSize: 8.5,
  bold: true,
  color: C.red,
  role: "stencil-label",
});
text("Material B, row j+1", ax + 0.16, faceY + 0.33, 1.55, 0.18, {
  fontSize: 8.5,
  bold: true,
  color: C.blue,
  role: "stencil-label",
});

text("state: (n, p, φ) at row j", ax + 0.18, ay + 0.58, 1.55, 0.18, {
  fontSize: 8.2,
  color: C.navy,
  role: "stencil-text",
});
text("state: (n, p, φ) at row j+1", ax + 0.18, faceY + 0.75, 1.82, 0.18, {
  fontSize: 8.2,
  color: C.navy,
  role: "stencil-text",
});

label(ax + 0.14, faceY - 0.27, 1.02, "internal face f", C.slate, {
  fontSize: 7.5,
  bold: true,
  align: "left",
  role: "face-label",
});
line(fluxX, faceY - 0.44, fluxX, faceY + 0.44, C.slate, { pt: 1.45, beginArrow: "triangle", endArrow: "triangle" });
fluxPair(fluxX + 0.16, faceY - 0.29, { fontSize: 10.0, subFontSize: 6.2, subDy: 0.075 });
text("shared normal flux", fluxX + 0.16, faceY + 0.10, 0.98, 0.14, {
  fontSize: 7.4,
  color: C.slate,
  align: "left",
  role: "flux-label",
});
line(ax, faceY, ax + aw, faceY, C.slate, { pt: 2.4, noHead: true });

rect(1.08, 5.18, 3.33, 0.78, C.white, C.line, { pt: 0.8, round: 0.06 });
text("Numerical meaning: one shared face flux couples the two control volumes and enters their divergence balances with opposite signs.", 1.27, 5.32, 2.92, 0.34, {
  fontSize: 8.0,
  color: C.navy,
  align: "center",
  role: "stencil-note",
});

// Panel B: band-edge schematic.
rect(4.95, 1.30, 3.82, 5.18, C.offWhite, C.line, { pt: 1.0, round: 0.08 });
panelTitle("B. Heterojunction Physics", 5.20, 1.46, 2.70);
label(7.62, 1.50, 0.78, "energy", C.muted, { align: "right" });

const bx0 = 5.36;
const bxFace = 6.82;
const bx1 = 8.24;
const ecL = 2.20;
const ecR = 2.62;
const evL = 4.25;
const evR = 3.82;
line(bx0, ecL, bxFace, ecL, C.blue, { pt: 2.0, noHead: true });
line(bxFace, ecL, bxFace, ecR, C.blue, { pt: 2.0, noHead: true });
line(bxFace, ecR, bx1, ecR, C.blue, { pt: 2.0, noHead: true });
line(bx0, evL, bxFace, evL, C.red, { pt: 2.0, noHead: true });
line(bxFace, evL, bxFace, evR, C.red, { pt: 2.0, noHead: true });
line(bxFace, evR, bx1, evR, C.red, { pt: 2.0, noHead: true });
line(bxFace, 1.97, bxFace, 4.50, C.slate, { pt: 1.1, dash: "dash", noHead: true });

richText(["E", sub("C")], bx0 + 0.05, ecL - 0.24, 0.38, 0.16, {
  fontSize: 8.4,
  bold: true,
  color: C.blue,
  role: "band-label",
});
richText(["E", sub("V")], bx0 + 0.05, evL + 0.10, 0.38, 0.16, {
  fontSize: 8.4,
  bold: true,
  color: C.red,
  role: "band-label",
});
label(bxFace - 0.32, 4.57, 0.66, "face f", C.muted, { align: "center" });

line(bxFace + 0.18, ecL + 0.02, bxFace + 0.18, ecR - 0.02, C.blue, { pt: 1.0, beginArrow: "triangle", endArrow: "triangle" });
richText(["ΔE", sub("C")], bxFace + 0.27, (ecL + ecR) / 2 - 0.09, 0.52, 0.16, {
  fontSize: 8.4,
  bold: true,
  color: C.blue,
  role: "delta-ec",
});
line(bxFace - 0.20, evR + 0.02, bxFace - 0.20, evL - 0.02, C.red, { pt: 1.0, beginArrow: "triangle", endArrow: "triangle" });
richText(["ΔE", sub("V")], bxFace - 0.78, (evR + evL) / 2 - 0.09, 0.52, 0.16, {
  fontSize: 8.4,
  bold: true,
  color: C.red,
  role: "delta-ev",
});

rect(5.22, 4.90, 3.36, 1.08, C.white, C.line, { pt: 0.8, round: 0.06 });
richText(["Δψ", sub("n,f"), " = Δφ", sub("f"), " + Δχ", sub("f")], 5.45, 5.10, 1.90, 0.16, {
  fontSize: 8.2,
  bold: true,
  color: C.navy,
  role: "band-note",
});
richText(["Δψ", sub("p,f"), " = Δφ", sub("f"), " + Δχ", sub("f"), " + ΔE", sub("g,f")], 5.45, 5.38, 2.34, 0.16, {
  fontSize: 8.2,
  bold: true,
  color: C.navy,
  role: "band-note",
});
text("Band offsets become SG driving-potential jumps.", 5.45, 5.68, 2.90, 0.14, {
  fontSize: 7.3,
  color: C.slate,
  align: "center",
  role: "band-note",
});

// Panel C: numerical treatment.
rect(8.95, 1.30, 3.62, 5.18, C.offWhite, C.line, { pt: 1.0, round: 0.08 });
panelTitle("C. Numerical Treatment", 9.20, 1.46, 2.80);
label(11.42, 1.50, 0.72, "RHS path", C.muted, { align: "right" });

card(9.24, 1.84, 3.06, 0.56, "POISSON FIELD", "solve φ from charge density ρ", C.blueLight, C.blue);
line(10.77, 2.45, 10.77, 2.61, C.muted, { pt: 1.0 });
rect(9.24, 2.65, 3.06, 0.86, C.purpleLight, C.purple, { pt: 1.0, round: 0.08 });
rect(9.24, 2.65, 0.08, 0.86, C.purple, C.purple, { lineTransparency: 100 });
text("BAND-OFFSET DRIVING POTENTIALS", 9.42, 2.77, 2.68, 0.20, {
  fontSize: 8.2,
  bold: true,
  color: C.purple,
  role: "card-title",
});
richText(["ψ", sub("n"), " = φ + χ"], 9.50, 3.17, 1.00, 0.16, {
  fontSize: 8.6,
  bold: true,
  color: C.navy,
  role: "card-body",
});
richText(["ψ", sub("p"), " = φ + χ + E", sub("g")], 10.58, 3.17, 1.28, 0.16, {
  fontSize: 8.6,
  bold: true,
  color: C.navy,
  role: "card-body",
});
line(10.77, 3.56, 10.77, 3.68, C.muted, { pt: 1.0 });
miniBox(9.24, 3.72, 3.06, 0.60, "FACE JUMP ENTERS SG EXPONENT", C.offWhite, C.slate);
richText(["ξ", sub("n,f"), " = Δψ", sub("n,f"), "/V", sub("T")], 9.48, 4.02, 1.12, 0.13, {
  fontSize: 7.8,
  bold: true,
  color: C.navy,
  role: "card-body",
});
richText(["ξ", sub("p,f"), " = Δψ", sub("p,f"), "/V", sub("T")], 10.70, 4.02, 1.18, 0.13, {
  fontSize: 7.8,
  bold: true,
  color: C.navy,
  role: "card-body",
});
line(10.77, 4.37, 10.77, 4.53, C.muted, { pt: 1.0 });
rect(9.24, 4.57, 3.06, 0.66, C.amberLight, C.amber, { pt: 1.0, round: 0.08 });
rect(9.24, 4.57, 0.08, 0.66, C.amber, C.amber, { lineTransparency: 100 });
text("BERNOULLI SG FACE FLUX", 9.42, 4.68, 2.62, 0.17, {
  fontSize: 8.0,
  bold: true,
  color: C.amber,
  role: "card-title",
});
fluxPair(9.55, 4.94, { fontSize: 8.8, subFontSize: 5.6, subDy: 0.066 });
text("via B(ξ)", 10.58, 4.97, 0.70, 0.13, {
  fontSize: 7.6,
  color: C.navy,
  role: "card-body",
});
line(10.77, 5.27, 10.77, 5.42, C.muted, { pt: 1.0 });
richCard(9.24, 5.46, 3.06, 0.58, "CAP + DIVERGENCE", ["|J", sup("SG"), "| ≤ |J", sup("TE"), "| if needed; then ∇·J"], C.greenLight, C.green);

pill(9.47, 6.16, 1.00, "not BC", C.white, C.slate);
pill(10.61, 6.16, 1.32, "internal flux", C.white, C.slate);

// Bottom takeaway.
rect(0, 6.93, 13.333, 0.57, C.strip, C.strip, { lineTransparency: 100 });
richText(
  ["Takeaway: band parameters χ and E", sub("g"), " turn offsets into SG driving-potential jumps; Bernoulli flux plus cap gives a conservative face current."],
  0.50,
  7.07,
  12.25,
  0.22,
  {
    fontSize: 11.6,
    bold: true,
    color: C.navy,
    align: "center",
    role: "takeaway",
  },
);

slide.addNotes(`Suggested narration:
This slide should replace the earlier carrier-density-curve sketch. The 2D code represents a material interface as an internal y-face between two neighboring finite-volume rows, not as a separate contact boundary condition. The left panel emphasizes the stencil: one face flux J_n,f and J_p,f couples row j and row j+1. The middle panel shows the heterojunction physics: chi and Eg change across the interface, producing conduction- and valence-band offsets. Those offsets enter the SG driving-potential jump: Delta psi_n,f = Delta phi_f + Delta chi_f and Delta psi_p,f = Delta phi_f + Delta chi_f + Delta E_g,f. The right panel maps this to the code path: Poisson gives phi, material arrays give chi and Eg, the solver constructs psi_n = phi + chi and psi_p = phi + chi + Eg, SG uses xi = Delta psi / V_T inside the Bernoulli flux, and thermionic-emission capping is applied at heterofaces when needed. The same face flux then enters the finite-volume divergence on the two sides with opposite signs, which is the conservative coupling.`);

const ignored = new Set([
  "title",
  "subtitle",
  "panel-title",
  "label",
  "pill",
  "takeaway",
  "stencil-label",
  "stencil-text",
  "face-label",
  "flux-label",
  "stencil-note",
  "band-label",
  "delta-ec",
  "delta-ev",
  "band-note",
  "math-main",
  "math-sub",
  "math-comma",
]);
const overlaps = [];
for (let i = 0; i < textBoxes.length; i += 1) {
  for (let j = i + 1; j < textBoxes.length; j += 1) {
    const a = textBoxes[i];
    const b = textBoxes[j];
    if (ignored.has(a.role) || ignored.has(b.role)) continue;
    if (overlap(a, b, 0.015)) overlaps.push(`${a.role}:${a.txt} <-> ${b.role}:${b.txt}`);
  }
}
if (overlaps.length) {
  console.error(`Potential text overlaps:\n${overlaps.join("\n")}`);
  process.exit(1);
}

async function main() {
  await pptx.writeFile({ fileName: outPath });
  console.log(outPath);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
