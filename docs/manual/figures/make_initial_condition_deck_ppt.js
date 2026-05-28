const fs = require("fs");
const path = require("path");
const pptxgen = require("/Users/shane/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/pptxgenjs");

const root = path.resolve(__dirname, "../../..");
const outDir = path.join(root, "docs", "manual", "slides");
const outPath = path.join(outDir, "initial_condition_explained.pptx");
fs.mkdirSync(outDir, { recursive: true });

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

function createDeck() {
  const pptx = new pptxgen();
  pptx.layout = "LAYOUT_WIDE";
  pptx.author = "SolarLab";
  pptx.company = "SolarLab";
  pptx.subject = "SolarLab initial condition explanation";
  pptx.title = "Initial Condition in SolarLab";
  pptx.lang = "en-US";
  pptx.theme = { headFontFace: "Arial", bodyFontFace: "Arial", lang: "en-US" };
  pptx.margin = 0;
  return pptx;
}

function slideTools(slide, pptx) {
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
      x: x1, y: y1, w: x2 - x1, h: y2 - y1,
      line: {
        color,
        pt: opt.pt || 1.25,
        dash: opt.dash || "solid",
        beginArrowType: opt.beginArrow || "none",
        endArrowType: opt.noHead ? "none" : opt.endArrow || "triangle",
      },
    });
  }

  function header(title, subtitle) {
    text(title, 0.46, 0.22, 12.1, 0.34, { fontSize: 20.8, bold: true, role: "title" });
    text(subtitle, 0.47, 0.66, 12.15, 0.25, { fontSize: 11.6, color: C.slate, role: "subtitle" });
    rect(0.45, 1.06, 12.43, 5.68, C.white, C.line, { round: 0.08 });
  }

  function panel(x, y, w, h, title) {
    rect(x, y, w, h, C.offWhite, C.line, { pt: 1.0, round: 0.08 });
    text(title, x + 0.24, y + 0.17, w - 0.48, 0.24, {
      fontSize: 12.5,
      bold: true,
      color: C.slate,
      role: "panel-title",
    });
  }

  function card(x, y, w, h, title, body, fill, accent, opt = {}) {
    rect(x, y, w, h, fill, accent, { pt: 1.05, round: 0.08 });
    rect(x, y, 0.08, h, accent, accent, { lineTransparency: 100 });
    text(title, x + 0.20, y + 0.14, w - 0.40, 0.22, {
      fontSize: opt.titleSize || 9.4,
      bold: true,
      color: accent,
      role: "card-title",
    });
    if (body) {
      text(body, x + 0.20, y + 0.46, w - 0.40, h - 0.58, {
        fontSize: opt.bodySize || 10.0,
        color: C.navy,
        role: "card-body",
      });
    }
  }

  function formulaCard(x, y, w, h, title, parts, body, fill, accent, opt = {}) {
    card(x, y, w, h, title, "", fill, accent, opt);
    rich(parts, x + 0.24, y + 0.50, w - 0.48, opt.formulaH || 0.26, {
      fontSize: opt.formulaSize || 14,
      bold: true,
      color: C.navy,
      role: "formula",
    });
    if (body) {
      text(body, x + 0.24, y + 0.88, w - 0.48, h - 0.98, {
        fontSize: opt.bodySize || 10.0,
        color: C.navy,
        role: "card-body",
      });
    }
  }

  function stepBox(x, y, w, h, title, body, fill, accent, opt = {}) {
    rect(x, y, w, h, fill, accent, { pt: 1.05, round: 0.08 });
    text(title, x + 0.15, y + 0.13, w - 0.30, 0.22, {
      fontSize: opt.titleSize || 9.6,
      bold: true,
      color: accent,
      align: "center",
      role: "step-title",
    });
    text(body, x + 0.18, y + 0.47, w - 0.36, h - 0.58, {
      fontSize: opt.bodySize || 9.2,
      color: C.navy,
      align: opt.align || "center",
      role: "step-body",
    });
  }

  function pill(x, y, w, h, label, fill, accent, opt = {}) {
    rect(x, y, w, h, fill, accent, { pt: 0.9, round: 0.09 });
    text(label, x + 0.07, y + h / 2 - 0.075, w - 0.14, 0.15, {
      fontSize: opt.fontSize || 8.5,
      bold: true,
      color: accent,
      align: "center",
      role: "pill",
    });
  }

  function takeaway(parts) {
    rect(0, 6.93, 13.333, 0.57, C.strip, C.strip, { lineTransparency: 100 });
    rich(parts, 0.54, 7.06, 12.2, 0.24, {
      fontSize: 11.6,
      bold: true,
      color: C.navy,
      align: "center",
      role: "takeaway",
    });
  }

  function check() {
    const ignore = new Set(["title", "subtitle", "takeaway"]);
    const overlaps = [];
    function touch(a, b, pad = 0.01) {
      return !(
        a.x + a.w + pad <= b.x ||
        b.x + b.w + pad <= a.x ||
        a.y + a.h + pad <= b.y ||
        b.y + b.h + pad <= a.y
      );
    }
    for (let i = 0; i < boxes.length; i += 1) {
      for (let j = i + 1; j < boxes.length; j += 1) {
        const a = boxes[i];
        const b = boxes[j];
        if (ignore.has(a.role) || ignore.has(b.role)) continue;
        if (touch(a, b)) overlaps.push(`${a.role}:${a.txt} <-> ${b.role}:${b.txt}`);
      }
    }
    if (overlaps.length) throw new Error(`Potential text overlaps:\n${overlaps.join("\n")}`);
  }

  return { text, rich, sub, sup, rect, line, header, panel, card, formulaCard, stepBox, pill, takeaway, check };
}

function addConceptSlide(pptx) {
  const slide = pptx.addSlide();
  slide.background = { color: C.bg };
  const t = slideTools(slide, pptx);

  t.header(
    "Initial Condition: A Physically Prepared Starting State",
    "SolarLab does not start from arbitrary carrier values; it first constructs a stable device state before transient solving.",
  );

  t.panel(0.75, 1.30, 4.10, 5.15, "A. What the Solver Needs");
  t.card(
    1.08,
    1.92,
    3.45,
    1.05,
    "STATE VECTOR",
    "The transient solver needs carrier and ion densities at t = 0.",
    C.blueLight,
    C.blue,
    { bodySize: 10.2 },
  );
  t.rich(["1D:  Y = [n, p, P]"], 1.25, 3.33, 2.95, 0.26, {
    fontSize: 14.0,
    bold: true,
    color: C.navy,
    align: "center",
    role: "formula",
  });
  t.rich(["2D:  Y = [n(y,x), p(y,x)]"], 1.02, 3.88, 3.55, 0.26, {
    fontSize: 13.0,
    bold: true,
    color: C.navy,
    align: "center",
    role: "formula",
  });
  t.card(
    1.08,
    4.72,
    3.45,
    0.88,
    "WHY IT MATTERS",
    "A poor initial state creates artificial transients or stiff solver failure.",
    C.redLight,
    C.red,
    { bodySize: 9.6 },
  );

  t.panel(5.05, 1.30, 7.05, 5.15, "B. Preparation Logic");
  const y = 2.35;
  t.stepBox(5.40, y, 1.55, 1.02, "1", "dark quasi-neutral state", C.blueLight, C.blue);
  t.line(6.95, y + 0.51, 7.46, y + 0.51, C.muted, { pt: 1.3 });
  t.stepBox(7.48, y, 1.55, 1.02, "2", "optional light-soaked state", C.amberLight, C.amber);
  t.line(9.03, y + 0.51, 9.54, y + 0.51, C.muted, { pt: 1.3 });
  t.stepBox(9.56, y, 1.72, 1.02, "3", "transient J-V solve", C.greenLight, C.green);
  t.card(
    5.42,
    4.02,
    2.28,
    0.85,
    "DARK RUN",
    "Use the dark equilibrium directly.",
    C.offWhite,
    C.slate,
    { bodySize: 9.5 },
  );
  t.card(
    8.04,
    4.02,
    2.92,
    0.85,
    "ILLUMINATED RUN",
    "First settle carriers under light, then start the voltage sweep.",
    C.amberLight,
    C.amber,
    { bodySize: 9.5 },
  );
  t.pill(6.00, 5.32, 1.38, 0.34, "less artificial transient", C.white, C.blue);
  t.pill(7.75, 5.32, 1.30, 0.34, "better Radau start", C.white, C.green);
  t.pill(9.42, 5.32, 1.28, 0.34, "physical memory", C.white, C.purple);

  t.takeaway([
    "Takeaway: initial condition is a pre-conditioned physical state, ",
    "not a random numerical guess.",
  ]);

  slide.addNotes(`Suggested narration:
Initial condition means the state given to the transient solver at t=0. In SolarLab this state is not arbitrary. The code prepares a dark quasi-neutral state first. For illuminated simulations it then allows carriers to settle under light before the actual J-V sweep starts. This avoids mixing an artificial dark-to-light transient into the intended voltage-sweep dynamics.`);
  t.check();
}

function addOneDSlide(pptx) {
  const slide = pptx.addSlide();
  slide.background = { color: C.bg };
  const t = slideTools(slide, pptx);

  t.header(
    "1D Initial Condition: Dark Equilibrium and Light-Soaked State",
    "The 1D solver prepares n(x), p(x), and the initial ion profile before running the voltage sweep.",
  );

  t.panel(0.75, 1.30, 5.65, 5.15, "A. Dark Quasi-Neutral Equilibrium");
  t.card(1.08, 1.95, 4.95, 1.30, "CARRIER BALANCE", "", C.blueLight, C.blue);
  t.text("n p = nᵢ²        n - p = Nᴰ - Nᴬ", 1.32, 2.48, 4.45, 0.28, {
    fontSize: 14.2,
    bold: true,
    color: C.navy,
    role: "formula",
  });
  t.text("Local mass action plus approximate charge neutrality.", 1.32, 2.90, 4.45, 0.18, {
    fontSize: 10.2,
    color: C.navy,
    role: "card-body",
  });

  t.card(1.08, 3.54, 4.95, 1.02, "ION BACKGROUND", "", C.greenLight, C.green);
  t.text("P = initial ion profile", 1.32, 3.96, 2.40, 0.25, {
    fontSize: 13.5,
    bold: true,
    color: C.navy,
    role: "formula",
  });
  t.text("configured from the device stack", 3.95, 3.99, 1.80, 0.16, {
    fontSize: 9.8,
    color: C.navy,
    role: "card-body",
  });
  t.text("used as the initial ionic background", 1.32, 4.27, 2.35, 0.14, {
    fontSize: 8.8,
    color: C.muted,
    italic: true,
    role: "card-body",
  });

  t.card(1.08, 4.88, 4.95, 0.86, "CONTACT PINNING", "", C.purpleLight, C.purple);
  t.text("n, p at contacts → equilibrium values", 1.32, 5.32, 4.25, 0.20, {
    fontSize: 11.8,
    bold: true,
    color: C.navy,
    role: "formula",
  });

  t.panel(6.65, 1.30, 5.55, 5.15, "B. Illuminated Pre-Settle");
  t.stepBox(7.05, 2.02, 1.46, 0.88, "dark state", "Ydark", C.blueLight, C.blue, { bodySize: 11.0 });
  t.line(8.51, 2.46, 9.02, 2.46, C.muted, { pt: 1.3 });
  t.stepBox(9.04, 2.02, 1.64, 0.88, "turn on light", "G > 0", C.amberLight, C.amber, { bodySize: 11.0 });
  t.line(10.68, 2.46, 11.18, 2.46, C.muted, { pt: 1.3 });
  t.stepBox(11.02, 2.02, 0.96, 0.88, "steady state", "Ylight", C.greenLight, C.green, { titleSize: 8.5, bodySize: 10.8 });

  t.card(7.05, 3.50, 4.85, 1.16, "SHORT TRANSIENT SETTLE", "", C.amberLight, C.amber);
  t.text("Ylight = Radau(Ydark, 10⁻³ s)", 7.30, 4.00, 4.30, 0.25, {
    fontSize: 13.0,
    bold: true,
    color: C.navy,
    role: "formula",
  });
  t.text("Carriers relax quickly; ions move little.", 7.30, 4.39, 4.30, 0.16, {
    fontSize: 10.0,
    color: C.navy,
    role: "card-body",
  });
  t.card(
    7.05,
    4.83,
    4.85,
    0.85,
    "J-V WARM START",
    "Each voltage point starts from the previous settled state, so device memory is preserved.",
    C.offWhite,
    C.slate,
    { bodySize: 9.7 },
  );

  t.takeaway([
    "Takeaway: 1D starts from quasi-neutral dark equilibrium; ",
    "illuminated runs first settle carriers under light.",
  ]);

  slide.addNotes(`Suggested narration:
For the 1D system the state contains electrons, holes, and ions. The dark initializer solves a local analytic balance: mass action np=ni^2 and quasi-neutrality n-p=ND-NA. The ion density is initialized from the configured P_ion,0 profile. For illuminated J-V the code integrates from this dark state under light for a short time, producing a light-soaked carrier state before the voltage sweep. The sweep itself is warm-started point by point.`);
  t.check();
}

function addTwoDSlide(pptx) {
  const slide = pptx.addSlide();
  slide.background = { color: C.bg };
  const t = slideTools(slide, pptx);

  t.header(
    "2D Initial Condition: Extrude the 1D State into a 2D Mesh",
    "The 2D solver starts laterally uniform, then lets grain boundaries and lateral BCs shape the transient solution.",
  );

  t.panel(0.75, 1.30, 5.05, 5.15, "A. 1D Seed on the y Grid");
  const x = 1.52;
  const y = 2.10;
  t.line(x, y, x, y + 2.70, C.slate, { pt: 1.4, noHead: true });
  t.line(x - 0.08, y + 2.70, x + 0.08, y + 2.70, C.slate, { pt: 1.2, noHead: true });
  t.line(x - 0.08, y, x + 0.08, y, C.slate, { pt: 1.2, noHead: true });
  t.rect(x + 0.45, y + 0.05, 2.95, 0.60, C.redLight, C.red, { pt: 1.0 });
  t.rect(x + 0.45, y + 0.65, 2.95, 1.40, C.purpleLight, C.purple, { pt: 1.0 });
  t.rect(x + 0.45, y + 2.05, 2.95, 0.60, "E0F2FE", C.blue, { pt: 1.0 });
  t.text("HTL", x + 1.70, y + 0.26, 0.44, 0.14, { fontSize: 9.2, bold: true, color: C.red, align: "center" });
  t.text("absorber", x + 1.52, y + 1.23, 0.78, 0.14, { fontSize: 9.2, bold: true, color: C.purple, align: "center" });
  t.text("ETL", x + 1.70, y + 2.26, 0.44, 0.14, { fontSize: 9.2, bold: true, color: C.blue, align: "center" });
  t.rich(["known 1D profiles: n(y), p(y), P(y)"], 1.10, 5.24, 4.10, 0.24, {
    fontSize: 12.0,
    bold: true,
    color: C.navy,
    align: "center",
    role: "formula",
  });

  t.panel(6.05, 1.30, 6.15, 5.15, "B. Broadcast into 2D");
  const gx = 6.70;
  const gy = 2.05;
  const gw = 4.80;
  const gh = 2.68;
  t.rect(gx, gy, gw, gh, C.offWhite, C.slate, { pt: 1.0, round: 0.04 });
  for (let i = 1; i < 6; i += 1) {
    const xx = gx + (gw * i) / 6;
    t.line(xx, gy, xx, gy + gh, C.line, { pt: 0.8, noHead: true });
  }
  for (let j = 1; j < 4; j += 1) {
    const yy = gy + (gh * j) / 4;
    t.line(gx, yy, gx + gw, yy, C.line, { pt: 0.8, noHead: true });
  }
  t.line(gx + 0.80, gy + 0.28, gx + 0.80, gy + gh - 0.28, C.red, { pt: 1.8, dash: "dash", noHead: true });
  t.text("optional GB", gx + 0.54, gy + 1.20, 0.55, 0.15, {
    fontSize: 8.5,
    bold: true,
    color: C.red,
    align: "center",
  });
  t.rich(["x"], gx + gw - 0.05, gy + gh + 0.18, 0.18, 0.12, { fontSize: 9.0, color: C.muted });
  t.rich(["y"], gx - 0.25, gy - 0.08, 0.18, 0.12, { fontSize: 9.0, color: C.muted });

  t.card(6.62, 5.05, 2.65, 0.88, "CARRIERS", "", C.blueLight, C.blue);
  t.text("copy n(y) across x", 6.86, 5.44, 2.10, 0.18, {
    fontSize: 10.8,
    bold: true,
    color: C.navy,
    role: "formula",
  });
  t.text("copy p(y) across x", 6.86, 5.68, 2.10, 0.18, {
    fontSize: 10.8,
    bold: true,
    color: C.navy,
    role: "formula",
  });
  t.card(9.58, 5.05, 2.30, 0.88, "IONS", "", C.greenLight, C.green);
  t.text("freeze P(y)", 9.82, 5.44, 1.80, 0.18, {
    fontSize: 10.8,
    bold: true,
    color: C.navy,
    role: "formula",
  });
  t.text("into Poisson", 9.82, 5.68, 1.30, 0.18, {
    fontSize: 10.8,
    bold: true,
    color: C.navy,
    role: "formula",
  });

  t.takeaway([
    "Takeaway: 2D begins as a laterally uniform extrusion of the 1D prepared state; ",
    "2D physics then creates lateral variation.",
  ]);

  slide.addNotes(`Suggested narration:
The 2D solver does not invent a separate arbitrary 2D initial state. It first solves the corresponding 1D problem on the same vertical y-grid. The electron and hole profiles are copied across the lateral x direction, producing a laterally uniform initial state. The ion profile is not dynamic in the current 2D Stage-A/B solver; it is frozen into the Poisson background as P_static. Grain boundaries and lateral boundary conditions affect the subsequent transient evolution, not the initial copy operation itself.`);
  t.check();
}

async function main() {
  const pptx = createDeck();
  addConceptSlide(pptx);
  addOneDSlide(pptx);
  addTwoDSlide(pptx);
  await pptx.writeFile({ fileName: outPath });
  console.log(outPath);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
