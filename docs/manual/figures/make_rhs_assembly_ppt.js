const fs = require("fs");
const path = require("path");
const pptxgen = require("/Users/shane/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/pptxgenjs");

const root = path.resolve(__dirname, "../../..");
const figuresDir = path.resolve(__dirname);
const outDir = path.join(root, "docs", "manual", "slides");
const gifPath = path.join(figuresDir, "rhs_assembly_2d.gif");
const previewPath = path.join(figuresDir, "rhs_assembly_2d_preview.png");
const outPath = path.join(outDir, "2d_rhs_assembly_overview.pptx");

fs.mkdirSync(outDir, { recursive: true });

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "SolarLab";
pptx.company = "SolarLab";
pptx.subject = "2D RHS Assembly overview";
pptx.title = "2D RHS Assembly: From Carrier Fields to Time Derivatives";
pptx.lang = "en-US";
pptx.theme = {
  headFontFace: "Arial",
  bodyFontFace: "Arial",
  lang: "en-US",
};
pptx.margin = 0;

const slide = pptx.addSlide();
slide.background = { color: "F7F9FC" };

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
};

function text(txt, x, y, w, h, opt = {}) {
  slide.addText(txt, {
    x, y, w, h,
    fontFace: "Arial",
    margin: 0,
    fit: "shrink",
    color: opt.color || C.navy,
    fontSize: opt.fontSize || 14,
    bold: opt.bold || false,
    italic: opt.italic || false,
    breakLine: opt.breakLine || false,
    align: opt.align || "left",
    valign: opt.valign || "top",
    paraSpaceAfterPt: opt.paraSpaceAfterPt || 0,
    paraSpaceBeforePt: 0,
  });
}

function rect(x, y, w, h, fill, line = C.line, radius = 0) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x, y, w, h,
    rectRadius: radius,
    fill: { color: fill },
    line: { color: line, pt: 1 },
  });
}

function arrow(x1, y1, x2, y2, color = C.muted) {
  slide.addShape(pptx.ShapeType.line, {
    x: x1, y: y1, w: x2 - x1, h: y2 - y1,
    line: { color, pt: 1.5, beginArrowType: "none", endArrowType: "triangle" },
  });
}

function miniCard(x, y, w, h, label, body, fill, accent) {
  rect(x, y, w, h, fill, accent);
  slide.addShape(pptx.ShapeType.rect, {
    x, y, w: 0.07, h,
    fill: { color: accent },
    line: { color: accent, transparency: 100 },
  });
  text(label, x + 0.18, y + 0.13, w - 0.28, 0.23, {
    fontSize: 10.5,
    bold: true,
    color: accent,
  });
  text(body, x + 0.18, y + 0.40, w - 0.30, h - 0.47, {
    fontSize: 13,
    color: C.navy,
  });
}

// Title block
text("2D RHS Assembly: From Carrier Fields to Time Derivatives", 0.46, 0.23, 12.2, 0.42, {
  fontSize: 25,
  bold: true,
});
text("The semi-discrete operator F(Y) evaluated repeatedly inside the Radau transient solver.", 0.47, 0.69, 12.0, 0.26, {
  fontSize: 12.2,
  color: C.slate,
});

// Left animated visualization card
rect(0.45, 1.10, 7.05, 5.65, C.white, C.line);
text("Animated RHS pipeline", 0.73, 1.27, 3.4, 0.26, {
  fontSize: 12.5,
  bold: true,
  color: C.slate,
});
text("10 stages, 5 s per stage", 5.95, 1.27, 1.1, 0.24, {
  fontSize: 8.8,
  color: C.muted,
  align: "right",
});

// PowerPoint supports animated GIFs as images. If a viewer only renders the
// first frame, the PNG preview remains next to the GIF in the figures folder.
slide.addImage({
  path: gifPath,
  x: 0.72,
  y: 1.62,
  w: 6.52,
  h: 3.67,
});
text("Y -> rho -> phi -> R,G -> SG fluxes -> div J -> dn/dt, dp/dt -> dY/dt", 0.76, 5.55, 6.45, 0.32, {
  fontSize: 12.2,
  bold: true,
  color: C.blue,
  align: "center",
});
text("Note: the GIF is a conceptual visualization of the code path, not a literal simulation snapshot.", 0.76, 6.08, 6.45, 0.34, {
  fontSize: 9.8,
  color: C.muted,
  align: "center",
});

// Right-side structured explanation
miniCard(
  7.82, 1.10, 4.95, 0.92,
  "INPUT STATE",
  "Y = [ n(y,x), p(y,x) ]",
  C.blueLight,
  C.blue,
);
arrow(10.30, 2.05, 10.30, 2.33);
miniCard(
  7.82, 2.35, 4.95, 1.48,
  "COUPLED PHYSICS",
  "rho -> phi by Poisson\nG - R source/sink terms\nJ_n, J_p from Scharfetter-Gummel fluxes",
  C.greenLight,
  C.green,
);
arrow(10.30, 3.88, 10.30, 4.16);
miniCard(
  7.82, 4.18, 4.95, 0.92,
  "OUTPUT RHS",
  "dY/dt = [ dn/dt, dp/dt ]",
  C.amberLight,
  C.amber,
);

// Equation panel
rect(7.82, 5.36, 4.95, 1.19, "F8FAFC", C.line);
text("Core equations", 8.05, 5.51, 1.8, 0.24, {
  fontSize: 10.8,
  bold: true,
  color: C.slate,
});
text("∇·(ε∇φ) = −ρ", 8.05, 5.83, 1.8, 0.24, {
  fontSize: 12.0,
  color: C.navy,
});
text("∂n/∂t =  (1/q)∇·Jₙ + G − R", 9.60, 5.75, 2.95, 0.24, {
  fontSize: 10.6,
  color: C.navy,
});
text("∂p/∂t = −(1/q)∇·Jₚ + G − R", 9.60, 6.05, 2.95, 0.24, {
  fontSize: 10.6,
  color: C.navy,
});

// Bottom takeaway strip
slide.addShape(pptx.ShapeType.rect, {
  x: 0,
  y: 6.93,
  w: 13.333,
  h: 0.57,
  fill: { color: "EAF2FF" },
  line: { color: "EAF2FF", transparency: 100 },
});
text("Takeaway: RHS assembly is the physics-numerics operator F(Y); Radau repeatedly calls it to advance the 2D drift-diffusion-Poisson system in time.", 0.50, 7.08, 12.25, 0.22, {
  fontSize: 12.2,
  bold: true,
  color: C.navy,
  align: "center",
});

slide.addNotes(`Suggested narration:
1. The 2D state Y stores all electron and hole densities on the lateral-stack grid.
2. RHS assembly unpacks Y, computes charge density, and solves Poisson for the electrostatic potential.
3. Generation and recombination provide local source and sink terms.
4. Scharfetter-Gummel fluxes compute drift-diffusion currents on grid faces.
5. Current divergence plus G-R gives dn/dt and dp/dt.
6. Contact boundary conditions are applied before packing the result as dY/dt.
7. Radau calls this operator many times during one transient settle step at a fixed applied voltage.`);

pptx.writeFile({ fileName: outPath });

