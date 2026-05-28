const fs = require("fs");
const path = require("path");
const pptxgen = require("/Users/shane/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/pptxgenjs");

const root = path.resolve(__dirname, "../../..");
const outDir = path.join(root, "docs", "manual", "slides");
const outContact = path.join(outDir, "external_contact_bc_2d.pptx");
const outLateral = path.join(outDir, "lateral_bc_2d.pptx");
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

function deck(title, subject) {
  const pptx = new pptxgen();
  pptx.layout = "LAYOUT_WIDE";
  pptx.author = "SolarLab";
  pptx.company = "SolarLab";
  pptx.subject = subject;
  pptx.title = title;
  pptx.lang = "en-US";
  pptx.theme = { headFontFace: "Arial", bodyFontFace: "Arial", lang: "en-US" };
  pptx.margin = 0;
  return pptx;
}

function tools(slide, pptx) {
  const boxes = [];

  function addBox(txt, x, y, w, h, role) {
    boxes.push({ txt: String(txt || ""), x, y, w, h, role: role || "text" });
  }

  function text(txt, x, y, w, h, opt = {}) {
    addBox(txt, x, y, w, h, opt.role);
    slide.addText(txt, {
      x,
      y,
      w,
      h,
      margin: opt.margin ?? 0,
      fit: opt.fit || "shrink",
      fontFace: "Arial",
      fontSize: opt.fontSize || 12,
      color: opt.color || C.navy,
      bold: opt.bold || false,
      italic: opt.italic || false,
      align: opt.align || "left",
      valign: opt.valign || "top",
      rotate: opt.rotate || 0,
      breakLine: opt.breakLine || false,
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
    addBox(plain, x, y, w, h, opt.role);
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
        x,
        y,
        w,
        h,
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
        dash: opt.dash || "solid",
        transparency: opt.transparency || 0,
        beginArrowType: opt.beginArrow || "none",
        endArrowType: opt.noHead ? "none" : opt.endArrow || "triangle",
      },
    });
  }

  function header(title, subtitle) {
    text(title, 0.46, 0.22, 12.10, 0.34, {
      fontSize: 20.8,
      bold: true,
      role: "title",
    });
    text(subtitle, 0.47, 0.66, 12.15, 0.25, {
      fontSize: 11.7,
      color: C.slate,
      role: "subtitle",
    });
    rect(0.45, 1.06, 12.43, 5.68, C.white, C.line, { round: 0.08 });
  }

  function panel(x, y, w, h, title) {
    rect(x, y, w, h, C.offWhite, C.line, { pt: 1.0, round: 0.08 });
    text(title, x + 0.24, y + 0.17, w - 0.48, 0.24, {
      fontSize: 12.7,
      bold: true,
      color: C.slate,
      role: "panel-title",
    });
  }

  function card(x, y, w, h, title, body, fill, accent, opt = {}) {
    rect(x, y, w, h, fill, accent, { pt: 1.05, round: 0.08 });
    rect(x, y, 0.08, h, accent, accent, { lineTransparency: 100 });
    text(title, x + 0.20, y + 0.14, w - 0.40, 0.22, {
      fontSize: opt.titleSize || 9.5,
      bold: true,
      color: accent,
      role: "card-title",
    });
    if (body) {
      text(body, x + 0.20, y + 0.45, w - 0.40, h - 0.55, {
        fontSize: opt.bodySize || 10.0,
        color: C.navy,
        role: "card-body",
      });
    }
  }

  function formulaCard(x, y, w, h, title, formulaRuns, body, fill, accent, opt = {}) {
    card(x, y, w, h, title, "", fill, accent, opt);
    rich(formulaRuns, x + 0.25, y + 0.48, w - 0.50, opt.formulaH || 0.34, {
      fontSize: opt.formulaSize || 14.0,
      bold: true,
      color: C.navy,
      role: "formula",
    });
    if (body) {
      text(body, x + 0.25, y + 0.90, w - 0.50, h - 1.02, {
        fontSize: opt.bodySize || 10.0,
        color: C.navy,
        role: "card-body",
      });
    }
  }

  function statementBox(x, y, w, h, title, body, fill, accent, opt = {}) {
    rect(x, y, w, h, fill, accent, { pt: 1.0, round: 0.08 });
    rect(x, y, 0.08, h, accent, accent, { lineTransparency: 100 });
    const label = body ? `${title}: ${body}` : title;
    text(label, x + 0.20, y + h / 2 - 0.10, w - 0.40, 0.20, {
      fontSize: opt.fontSize || 9.8,
      bold: opt.bold || false,
      color: opt.color || C.navy,
      role: "statement",
    });
  }

  function pill(x, y, w, h, txt, fill, accent, opt = {}) {
    rect(x, y, w, h, fill, accent, { pt: 0.9, round: 0.09 });
    text(txt, x + 0.07, y + h / 2 - 0.075, w - 0.14, 0.15, {
      fontSize: opt.fontSize || 8.7,
      bold: true,
      color: accent,
      align: "center",
      role: "pill",
    });
  }

  function takeaway(parts) {
    rect(0, 6.93, 13.333, 0.57, C.strip, C.strip, { lineTransparency: 100 });
    rich(parts, 0.54, 7.06, 12.20, 0.24, {
      fontSize: 11.8,
      bold: true,
      color: C.navy,
      align: "center",
      role: "takeaway",
    });
  }

  function check() {
    const ignore = new Set(["title", "subtitle", "takeaway"]);
    const overlaps = [];
    function touches(a, b, pad = 0.01) {
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
        if (touches(a, b)) {
          overlaps.push(`${a.role}:${a.txt} <-> ${b.role}:${b.txt}`);
        }
      }
    }
    if (overlaps.length) {
      throw new Error(`Potential text overlaps:\n${overlaps.join("\n")}`);
    }
  }

  return { text, rich, sub, rect, line, header, panel, card, formulaCard, statementBox, pill, takeaway, check };
}

function buildContactDeck() {
  const pptx = deck(
    "External Contact BC: Ohmic Pinning and Selective Robin Flux",
    "2D external contact boundary conditions",
  );
  const slide = pptx.addSlide();
  slide.background = { color: C.bg };
  const t = tools(slide, pptx);

  t.header(
    "External Contact BC: Ohmic Pinning and Selective Robin Flux",
    "External contacts are electrode reservoirs at the top and bottom rows; they set how carriers enter or leave the 2D domain.",
  );

  t.panel(0.75, 1.28, 5.10, 5.36, "A. Contact Rows in the 2D Device");

  const sx = 1.42;
  const sy = 1.98;
  const sw = 3.78;
  t.rect(sx, sy, sw, 0.42, "E2E8F0", C.slate, { pt: 1.1, round: 0.03 });
  t.rect(sx, sy + 0.42, sw, 0.86, C.redLight, C.red, { pt: 1.0, fillTransparency: 5 });
  t.rect(sx, sy + 1.28, sw, 1.62, C.purpleLight, C.purple, { pt: 1.0, fillTransparency: 8 });
  t.rect(sx, sy + 2.90, sw, 0.86, "E0F2FE", C.blue, { pt: 1.0, fillTransparency: 4 });
  t.rect(sx, sy + 3.76, sw, 0.42, "E2E8F0", C.slate, { pt: 1.1, round: 0.03 });
  t.line(sx, sy + 0.42, sx + sw, sy + 0.42, C.red, { pt: 2.2, noHead: true });
  t.line(sx, sy + 3.76, sx + sw, sy + 3.76, C.blue, { pt: 2.2, noHead: true });

  t.text("top electrode reservoir", sx + 0.72, sy + 0.14, 2.30, 0.16, {
    fontSize: 9.2,
    bold: true,
    color: C.slate,
    align: "center",
  });
  t.text("HTL / contact layer", sx + 0.74, sy + 0.78, 2.30, 0.16, {
    fontSize: 9.4,
    bold: true,
    color: C.red,
    align: "center",
  });
  t.text("absorber", sx + 1.28, sy + 1.98, 1.22, 0.16, {
    fontSize: 9.6,
    bold: true,
    color: C.purple,
    align: "center",
  });
  t.text("ETL / contact layer", sx + 0.74, sy + 3.22, 2.30, 0.16, {
    fontSize: 9.4,
    bold: true,
    color: C.blue,
    align: "center",
  });
  t.text("bottom electrode reservoir", sx + 0.62, sy + 3.91, 2.55, 0.16, {
    fontSize: 9.2,
    bold: true,
    color: C.slate,
    align: "center",
  });

  t.line(sx + 0.70, sy + 0.24, sx + 0.70, sy + 0.72, C.red, {
    pt: 1.8,
    beginArrow: "triangle",
    endArrow: "triangle",
  });
  t.line(sx + sw - 0.70, sy + 3.94, sx + sw - 0.70, sy + 3.46, C.blue, {
    pt: 1.8,
    beginArrow: "triangle",
    endArrow: "triangle",
  });
  t.text("carrier exchange", sx + 0.92, sy + 0.42, 1.42, 0.16, {
    fontSize: 9.2,
    bold: true,
    color: C.red,
  });
  t.text("carrier exchange", sx + sw - 2.26, sy + 3.46, 1.42, 0.16, {
    fontSize: 9.2,
    bold: true,
    color: C.blue,
  });
  t.rich(["contact rows: y = 0 and y = L", t.sub("y")], 1.34, 6.13, 3.90, 0.18, {
    fontSize: 10.4,
    color: C.slate,
    align: "center",
  });

  t.panel(6.08, 1.28, 6.50, 5.36, "B. Boundary Model Used by the RHS");

  t.formulaCard(
    6.42,
    1.92,
    5.80,
    1.35,
    "OHMIC / DIRICHLET LIMIT",
    ["n = n", t.sub("eq"), "      p = p", t.sub("eq")],
    "Ideal reservoir: boundary carrier densities are pinned; the contact rows are not time-evolved.",
    C.blueLight,
    C.blue,
    { formulaSize: 15.0, bodySize: 10.5 },
  );

  t.formulaCard(
    6.42,
    3.52,
    5.80,
    1.47,
    "SELECTIVE / ROBIN CONTACT",
    ["Jₙ = q Sₙ(n - n", t.sub("eq"), ")     Jₚ = q Sₚ(p - p", t.sub("eq"), ")"],
    "Finite surface velocity S controls carrier exchange with the electrode reservoir.",
    C.amberLight,
    C.amber,
    { formulaSize: 13.4, formulaH: 0.34, bodySize: 10.5 },
  );

  t.line(6.62, 5.45, 11.98, 5.45, C.slate, { pt: 1.7, noHead: true });
  t.line(6.62, 5.45, 6.62, 5.31, C.slate, { pt: 1.1, noHead: true });
  t.line(9.28, 5.45, 9.28, 5.31, C.slate, { pt: 1.1, noHead: true });
  t.line(11.98, 5.45, 11.98, 5.31, C.slate, { pt: 1.1, noHead: true });
  t.text("S = 0", 6.34, 5.64, 0.66, 0.14, {
    fontSize: 9.0,
    bold: true,
    color: C.slate,
    align: "center",
  });
  t.text("finite S", 8.90, 5.64, 0.82, 0.14, {
    fontSize: 9.0,
    bold: true,
    color: C.purple,
    align: "center",
  });
  t.text("S -> infinity", 11.39, 5.64, 1.12, 0.14, {
    fontSize: 9.0,
    bold: true,
    color: C.green,
    align: "center",
  });
  t.text("blocking", 6.28, 5.91, 0.78, 0.13, {
    fontSize: 8.4,
    color: C.muted,
    align: "center",
  });
  t.text("selective", 8.96, 5.91, 0.70, 0.13, {
    fontSize: 8.4,
    color: C.muted,
    align: "center",
  });
  t.text("ohmic", 11.57, 5.91, 0.70, 0.13, {
    fontSize: 8.4,
    color: C.muted,
    align: "center",
  });

  t.takeaway([
    "Takeaway: external contact BCs model carrier exchange with electrodes; ",
    "Dirichlet pins densities, Robin uses finite surface velocity S.",
  ]);

  slide.addNotes(`Suggested narration:
This slide separates the physical location of the contact boundary from the mathematical boundary model. The contacts live at the top and bottom rows of the 2D mesh. In the ohmic Dirichlet limit, the electrode is treated as an ideal reservoir and the boundary carrier densities are pinned to equilibrium values. In the selective Robin formulation, the contact current is proportional to the difference between the boundary carrier density and its equilibrium reservoir value. The parameter S is a surface recombination or exchange velocity: S = 0 is blocking, finite S is selective, and very large S approaches the ohmic limit.`);

  t.check();
  return pptx;
}

function buildLateralDeck() {
  const pptx = deck(
    "2D Lateral BC: Periodic Window and Zero-Flux Sidewalls",
    "2D lateral boundary conditions",
  );
  const slide = pptx.addSlide();
  slide.background = { color: C.bg };
  const t = tools(slide, pptx);

  t.header(
    "2D Lateral BC: Periodic Window and Zero-Flux Sidewalls",
    "The lateral BC closes the finite x-window of the 2D mesh; it is not an electrode contact condition.",
  );

  t.panel(0.75, 1.28, 5.75, 3.98, "A. Periodic Lateral Boundary");
  const px = 1.30;
  const py = 1.98;
  const pw = 4.62;
  const ph = 2.10;
  t.rect(px, py, pw, ph, C.purpleLight, C.purple, { pt: 1.1, round: 0.04, fillTransparency: 8 });
  t.line(px + 1.28, py + 0.28, px + 1.28, py + ph - 0.28, C.red, {
    pt: 2.0,
    dash: "dash",
    noHead: true,
  });
  t.text("GB", px + 1.12, py + 0.97, 0.34, 0.16, {
    fontSize: 9.8,
    bold: true,
    color: C.red,
    align: "center",
  });
  t.line(px + pw, py + 0.62, px, py + 0.62, C.blue, { pt: 1.8, dash: "dash" });
  t.line(px, py + 1.48, px + pw, py + 1.48, C.blue, { pt: 1.8, dash: "dash" });
  t.text("wrap face flux", px + 1.82, py + 0.41, 1.25, 0.16, {
    fontSize: 9.4,
    bold: true,
    color: C.blue,
    align: "center",
  });
  t.text("x = 0", px - 0.03, py + ph + 0.18, 0.55, 0.14, {
    fontSize: 9.0,
    color: C.muted,
  });
  t.rich(["x = L", t.sub("x")], px + pw - 0.62, py + ph + 0.18, 0.62, 0.14, {
    fontSize: 9.0,
    color: C.muted,
  });
  t.statementBox(
    1.12,
    4.48,
    5.02,
    0.58,
    "MEANING",
    "Left and right sides are copies of a repeating unit cell.",
    C.blueLight,
    C.blue,
    { fontSize: 10.0 },
  );

  t.panel(6.83, 1.28, 5.75, 3.98, "B. Neumann / Zero-Flux Sidewalls");
  const nx = 7.40;
  const ny = 1.98;
  const nw = 4.58;
  const nh = 2.10;
  t.rect(nx, ny, nw, nh, "E0F2FE", C.blue, { pt: 1.1, round: 0.04, fillTransparency: 4 });
  t.line(nx, ny, nx, ny + nh, C.slate, { pt: 2.4, noHead: true });
  t.line(nx + nw, ny, nx + nw, ny + nh, C.slate, { pt: 2.4, noHead: true });
  t.line(nx + 0.90, ny + 0.76, nx + 0.30, ny + 0.76, C.red, { pt: 1.6 });
  t.line(nx + nw - 0.90, ny + 1.34, nx + nw - 0.30, ny + 1.34, C.red, { pt: 1.6 });
  t.text("closed sidewall", nx + 0.14, ny + 0.15, 1.18, 0.15, {
    fontSize: 8.8,
    bold: true,
    color: C.slate,
  });
  t.text("closed sidewall", nx + nw - 1.32, ny + 0.15, 1.18, 0.15, {
    fontSize: 8.8,
    bold: true,
    color: C.slate,
    align: "right",
  });
  t.rich(["Jₓ = 0"], nx + 1.62, ny + 0.94, 1.34, 0.24, {
    fontSize: 15.8,
    bold: true,
    color: C.navy,
    align: "center",
    role: "formula",
  });
  t.statementBox(
    7.20,
    4.48,
    5.02,
    0.58,
    "MEANING",
    "Exterior side flux is zero; carriers do not leak out laterally.",
    C.greenLight,
    C.green,
    { fontSize: 10.0 },
  );

  t.panel(0.75, 5.37, 11.83, 1.18, "C. What Changes in RHS Assembly");
  t.statementBox(
    1.05,
    5.86,
    3.46,
    0.54,
    "PERIODIC",
    "add SG wrap flux at x sides",
    C.blueLight,
    C.blue,
    { fontSize: 9.3 },
  );
  t.statementBox(
    4.92,
    5.86,
    3.20,
    0.54,
    "NEUMANN",
    "set exterior side flux to zero",
    C.greenLight,
    C.green,
    { fontSize: 9.3 },
  );
  t.statementBox(
    8.55,
    5.86,
    3.50,
    0.54,
    "INTERIOR",
    "interior physics unchanged",
    C.amberLight,
    C.amber,
    { fontSize: 9.3 },
  );

  t.takeaway([
    "Takeaway: periodic BC makes a repeating 2D unit cell; ",
    "Neumann BC closes the sidewalls with zero lateral flux.",
  ]);

  slide.addNotes(`Suggested narration:
This slide should be read as the lateral counterpart to the external contact slide. The lateral boundary condition does not model electrodes. It closes the finite computational window in the x direction. Periodic BC connects the left and right sides with a wrap face, so flux leaving one side enters the other side. Neumann BC instead sets the exterior side flux to zero. The interior Scharfetter-Gummel face fluxes and recombination terms are unchanged; only the side-boundary divergence bookkeeping changes.`);

  t.check();
  return pptx;
}

async function main() {
  await buildContactDeck().writeFile({ fileName: outContact });
  console.log(outContact);
  await buildLateralDeck().writeFile({ fileName: outLateral });
  console.log(outLateral);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
