const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const sharp = require("sharp");

const OUT_DIR = path.join(__dirname, "..", "report", "poster");
fs.mkdirSync(OUT_DIR, { recursive: true });

const pptx = new pptxgen();
pptx.author = "Bahaa Hamdan, Tarek El Moura, Anthony John Kanaan";
pptx.company = "American University of Beirut";
pptx.subject = "Early-warning prediction of asthma exacerbations";
pptx.title = "Early-Warning Prediction of Asthma Exacerbations";
pptx.lang = "en-US";
pptx.layout = "LAYOUT_WIDE";
pptx.defineLayout({ name: "A0_LANDSCAPE", width: 48, height: 36 });
pptx.layout = "A0_LANDSCAPE";
pptx.theme = {
  headFontFace: "Aptos Display",
  bodyFontFace: "Aptos",
  lang: "en-US"
};
pptx.margin = 0;

const C = {
  aub: "7A1736",
  aubDark: "3B1B18",
  copper: "B8672E",
  gold: "D7A95A",
  teal: "177E89",
  tealDark: "0D4D56",
  ink: "1C1B1A",
  text: "3C3835",
  muted: "706A64",
  cream: "F7F2EA",
  panel: "FFFFFF",
  rule: "DED4C8",
  soft: "F1E8DF",
  warn: "A84D24",
  green: "4A7C59",
  blue: "315D7C"
};

function rgb(hex) { return hex.replace("#", ""); }
function addText(slide, text, x, y, w, h, opt = {}) {
  slide.addText(text, {
    x, y, w, h,
    fontFace: opt.fontFace || "Aptos",
    fontSize: opt.fontSize || 14,
    color: rgb(opt.color || C.text),
    bold: !!opt.bold,
    italic: !!opt.italic,
    margin: opt.margin ?? 0.04,
    breakLine: opt.breakLine,
    fit: opt.fit || "shrink",
    valign: opt.valign || "top",
    align: opt.align || "left",
    paraSpaceAfterPt: opt.paraSpaceAfterPt ?? 0,
    paraSpaceBeforePt: opt.paraSpaceBeforePt ?? 0,
    bullet: opt.bullet,
    rotate: opt.rotate,
    transparency: opt.transparency
  });
}
function addRect(slide, x, y, w, h, fill, line = fill, opt = {}) {
  slide.addShape(pptx.ShapeType.rect, {
    x, y, w, h,
    rectRadius: opt.radius || 0,
    fill: { color: rgb(fill), transparency: opt.transparency || 0 },
    line: { color: rgb(line), transparency: opt.lineTransparency || 0, width: opt.lineWidth || 0.8 }
  });
}
function addRound(slide, x, y, w, h, fill, line = fill, opt = {}) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x, y, w, h,
    rectRadius: opt.radius || 0.12,
    fill: { color: rgb(fill), transparency: opt.transparency || 0 },
    line: { color: rgb(line), transparency: opt.lineTransparency || 0, width: opt.lineWidth || 1 }
  });
}
function addLine(slide, x1, y1, x2, y2, color = C.rule, width = 1.4, opt = {}) {
  slide.addShape(pptx.ShapeType.line, {
    x: x1, y: y1, w: x2 - x1, h: y2 - y1,
    line: { color: rgb(color), width, beginArrowType: opt.beginArrow, endArrowType: opt.endArrow, transparency: opt.transparency || 0 }
  });
}
function sectionLabel(slide, text, x, y, w) {
  addText(slide, text.toUpperCase(), x, y, w, 0.32, {
    fontSize: 8.2,
    bold: true,
    color: C.aub,
    paraSpaceAfterPt: 0,
    margin: 0,
    fit: "resize"
  });
  addLine(slide, x, y + 0.35, x + w, y + 0.35, C.gold, 1.2);
}
function panel(slide, title, x, y, w, h, opt = {}) {
  addRound(slide, x, y, w, h, opt.fill || C.panel, opt.line || C.rule, { radius: 0.16, lineWidth: 1.1 });
  sectionLabel(slide, title, x + 0.38, y + 0.28, w - 0.76);
}
function bullet(slide, text, x, y, w, h, opt = {}) {
  slide.addText(text, {
    x, y, w, h,
    fontFace: "Aptos",
    fontSize: opt.fontSize || 11,
    color: rgb(opt.color || C.text),
    margin: 0.02,
    fit: "shrink",
    bullet: { type: "bullet" },
    breakLine: false,
    paraSpaceAfterPt: 2
  });
}

const slide = pptx.addSlide();
slide.background = { color: C.cream };

// Header
addRect(slide, 0, 0, 48, 4.25, C.aubDark, C.aubDark);
addRect(slide, 0, 0, 48, 0.18, C.gold, C.gold);
addText(slide, "Early-Warning Prediction of Asthma Exacerbations", 1.2, 0.55, 32.5, 1.15, {
  fontFace: "Georgia",
  fontSize: 31,
  bold: true,
  color: "FFFFFF",
  margin: 0,
  fit: "shrink"
});
addText(slide, "using wearable and digital-health data", 1.25, 1.78, 21.5, 0.55, {
  fontSize: 15.5,
  color: "E9DCCC",
  margin: 0,
  italic: true
});
addText(slide, "Bahaa Hamdan  |  Tarek El Moura  |  Anthony John Kanaan", 1.25, 2.48, 27.5, 0.45, {
  fontSize: 12.8,
  color: "FFFFFF",
  bold: true,
  margin: 0
});
addText(slide, "EECE 693 Neural Networks / Deep Learning - Spring 2026", 1.25, 3.05, 24.5, 0.38, {
  fontSize: 10.8,
  color: "DDC6B1",
  margin: 0
});
addRound(slide, 36.1, 0.64, 10.5, 2.55, "FFFFFF", "FFFFFF", { radius: 0.18, transparency: 0 });
addText(slide, "AUB", 36.65, 0.85, 2.7, 0.95, {
  fontFace: "Georgia",
  fontSize: 28,
  bold: true,
  color: C.aub,
  margin: 0
});
addText(slide, "American University\nof Beirut", 39.45, 0.88, 5.9, 0.95, {
  fontFace: "Aptos Display",
  fontSize: 14.5,
  bold: true,
  color: C.ink,
  margin: 0,
  fit: "shrink"
});
addLine(slide, 36.65, 2.1, 45.6, 2.1, C.rule, 1.0);
addText(slide, "Public AAMOS-00 asthma monitoring dataset", 36.65, 2.33, 8.8, 0.38, {
  fontSize: 9.6,
  color: C.muted,
  margin: 0
});

// Main message strip
addRound(slide, 1.05, 4.65, 45.9, 1.25, C.panel, C.rule, { radius: 0.15 });
addText(slide, "Core claim", 1.55, 4.92, 3.1, 0.35, { fontSize: 9, bold: true, color: C.aub, margin: 0 });
addText(slide, "We built a patient-safe early-warning pipeline that predicts pre-exacerbation risk from multimodal historical signals. The best held-out signal was promising, but statistically fragile because the test set contained only four positive events.", 4.05, 4.78, 32.2, 0.65, {
  fontSize: 13.1,
  color: C.ink,
  bold: true,
  margin: 0.02,
  fit: "shrink"
});
addText(slide, "Not deployment-ready", 38.2, 4.86, 7.7, 0.42, {
  fontSize: 12.5,
  color: C.warn,
  bold: true,
  align: "center",
  margin: 0
});
addText(slide, "needs larger prospective validation", 38.2, 5.23, 7.7, 0.32, {
  fontSize: 8.6,
  color: C.muted,
  align: "center",
  margin: 0
});

// Left column panels
panel(slide, "Clinical Task", 1.05, 6.45, 13.15, 6.2);
addText(slide, "Goal", 1.48, 7.1, 2.2, 0.35, { fontSize: 10.5, bold: true, color: C.ink, margin: 0 });
addText(slide, "Predict upcoming asthma exacerbation risk before clinically meaningful worsening, using only historical data available before onset.", 1.48, 7.52, 11.95, 0.95, {
  fontSize: 11.2,
  color: C.text,
  margin: 0.02
});
addText(slide, "Why hard?", 1.48, 8.82, 2.8, 0.35, { fontSize: 10.5, bold: true, color: C.ink, margin: 0 });
bullet(slide, "Rare positives: event windows are sparse.", 1.72, 9.28, 11.3, 0.38);
bullet(slide, "Patient-specific baselines and triggers.", 1.72, 9.82, 11.3, 0.38);
bullet(slide, "Missingness reflects behavior, adherence, charging, and device dropout.", 1.72, 10.36, 11.3, 0.62);
addRound(slide, 1.48, 11.34, 11.9, 0.78, C.soft, C.soft, { radius: 0.12 });
addText(slide, "Primary selection metric: validation PR-AUC", 1.75, 11.55, 11.35, 0.3, {
  fontSize: 10.2,
  bold: true,
  color: C.aub,
  margin: 0
});

panel(slide, "Data & Labels", 1.05, 13.1, 13.15, 7.9);
addText(slide, "AAMOS-00 multimodal sources", 1.48, 13.78, 10.6, 0.32, {
  fontSize: 10.7,
  bold: true,
  color: C.ink,
  margin: 0
});
const sources = [
  ["Smartwatch", "heart rate, steps, activity, intensity"],
  ["Smart inhaler", "reliever medication timestamps"],
  ["Peak flow", "PEF readings"],
  ["Questionnaires", "daily / weekly symptom evidence"],
  ["Environment", "weather and exposure context"],
  ["Patient info", "static demographic / clinical descriptors"]
];
let sy = 14.25;
for (const [name, desc] of sources) {
  addText(slide, name, 1.55, sy, 3.2, 0.26, { fontSize: 8.7, bold: true, color: C.aub, margin: 0 });
  addText(slide, desc, 4.55, sy, 8.55, 0.26, { fontSize: 8.3, color: C.text, margin: 0 });
  sy += 0.52;
}
addLine(slide, 1.48, 17.5, 13.35, 17.5, C.rule, 0.8);
addText(slide, "Event-episode labeling", 1.48, 17.85, 6.8, 0.32, {
  fontSize: 10.7,
  bold: true,
  color: C.ink,
  margin: 0
});
bullet(slide, "Positive: historical window immediately before probable event onset.", 1.72, 18.32, 11.4, 0.52, { fontSize: 8.9 });
bullet(slide, "Negative: stable periods with no event/positive-week overlap.", 1.72, 18.9, 11.4, 0.52, { fontSize: 8.9 });
bullet(slide, "Washout: post-event recovery days excluded from stable-negative sampling.", 1.72, 19.48, 11.4, 0.68, { fontSize: 8.9 });

panel(slide, "Sampling Grid", 1.05, 21.45, 13.15, 5.15);
const gridX = 1.48, gridY = 22.1;
const rows = [
  ["Event threshold", "2, 3, 4"],
  ["History length", "3, 7, 14 days"],
  ["Washout", "0, 7, 14 days"],
  ["Example T3/L14/W14", "53 pos / 1,071 neg"]
];
for (let i = 0; i < rows.length; i++) {
  const y = gridY + i * 0.76;
  addRound(slide, gridX, y, 11.9, 0.58, i % 2 ? "FBF8F4" : C.soft, "FFFFFF", { radius: 0.08, lineTransparency: 100 });
  addText(slide, rows[i][0], gridX + 0.28, y + 0.16, 4.5, 0.25, { fontSize: 8.7, bold: true, color: C.ink, margin: 0 });
  addText(slide, rows[i][1], gridX + 5.25, y + 0.16, 6.2, 0.25, { fontSize: 8.7, color: C.text, margin: 0 });
}

// Center pipeline
panel(slide, "Full Pipeline", 15.0, 6.45, 18.3, 14.55);
addText(slide, "Every model is evaluated under the same patient-level split and validation-first selection discipline.", 15.43, 7.1, 17.3, 0.55, {
  fontSize: 10.5,
  color: C.text,
  margin: 0
});
const px = 15.72, py = 8.08, nodeW = 4.45, nodeH = 1.0, gapX = 1.2, gapY = 1.18;
const pipe = [
  ["1", "Raw\nAAMOS-00"],
  ["2", "Clean +\nstandardize"],
  ["3", "Event\nepisodes"],
  ["4", "Lookback\nfeatures"],
  ["5", "Patient-safe\nsplit"],
  ["6", "Tune\nmodels"],
  ["7", "Validate\nPR-AUC"],
  ["8", "Final held-\nout test"]
];
for (let i = 0; i < pipe.length; i++) {
  const col = i % 4;
  const row = Math.floor(i / 4);
  const x = px + col * (nodeW + gapX);
  const y = py + row * (nodeH + gapY);
  const fill = i < 3 ? C.aub : i < 6 ? C.copper : C.tealDark;
  addRound(slide, x, y, nodeW, nodeH, fill, fill, { radius: 0.16 });
  addText(slide, pipe[i][0], x + 0.16, y + 0.25, 0.45, 0.3, { fontSize: 10, bold: true, color: "FFFFFF", align: "center", margin: 0 });
  addText(slide, pipe[i][1], x + 0.72, y + 0.16, nodeW - 0.95, 0.58, { fontSize: 9.0, bold: true, color: "FFFFFF", margin: 0, fit: "shrink" });
  if (i < pipe.length - 1 && col < 3) {
    addLine(slide, x + nodeW + 0.1, y + nodeH / 2, x + nodeW + gapX - 0.12, y + nodeH / 2, C.muted, 1.2, { endArrow: "triangle" });
  }
}
addLine(slide, px + 3 * (nodeW + gapX) + nodeW / 2, py + nodeH + 0.1, px + 3 * (nodeW + gapX) + nodeW / 2, py + nodeH + gapY - 0.16, C.muted, 1.2, { endArrow: "triangle" });

// Timeline
addRound(slide, 15.55, 13.05, 17.35, 2.35, "FBF8F4", C.rule, { radius: 0.14 });
addText(slide, "Label timing", 15.9, 13.35, 3.5, 0.3, { fontSize: 9, bold: true, color: C.aub, margin: 0 });
addLine(slide, 16.0, 14.35, 31.75, 14.35, C.ink, 1.2, { endArrow: "triangle" });
addRound(slide, 16.15, 13.9, 5.3, 0.9, C.teal, C.teal, { radius: 0.12 });
addText(slide, "Input history\nE-L ... E-1", 16.45, 14.05, 4.75, 0.45, { fontSize: 8.7, bold: true, color: "FFFFFF", align: "center", margin: 0 });
addRect(slide, 22.1, 13.75, 0.16, 1.15, C.aub, C.aub);
addText(slide, "Event onset\nE", 21.45, 14.95, 1.5, 0.45, { fontSize: 8.2, bold: true, color: C.aub, align: "center", margin: 0 });
addRound(slide, 23.35, 13.9, 3.35, 0.9, C.copper, C.copper, { radius: 0.12 });
addText(slide, "Event\nepisode", 23.68, 14.08, 2.65, 0.4, { fontSize: 8.3, bold: true, color: "FFFFFF", align: "center", margin: 0 });
addRound(slide, 27.35, 13.9, 3.9, 0.9, C.warn, C.warn, { radius: 0.12 });
addText(slide, "Post-event washout\nexcluded negatives", 27.55, 14.04, 3.55, 0.42, { fontSize: 7.6, bold: true, color: "FFFFFF", align: "center", margin: 0 });

// Models and protocol
addText(slide, "Model families", 15.55, 16.25, 4.5, 0.32, { fontSize: 10.2, bold: true, color: C.ink, margin: 0 });
const modelCards = [
  ["Classical", "LR - RF - XGBoost"],
  ["Sequence DL", "RNN - GRU - LSTM - CNN"],
  ["Selection", "validation PR-AUC first"],
  ["Safeguards", "gap - Brier - leakage - seeds"]
];
for (let i = 0; i < modelCards.length; i++) {
  const x = 15.55 + (i % 2) * 8.55;
  const y = 16.75 + Math.floor(i / 2) * 1.28;
  addRound(slide, x, y, 8.15, 0.92, i < 2 ? C.soft : "FFFFFF", C.rule, { radius: 0.12 });
  addText(slide, modelCards[i][0], x + 0.28, y + 0.16, 2.3, 0.24, { fontSize: 8.5, bold: true, color: C.aub, margin: 0 });
  addText(slide, modelCards[i][1], x + 2.65, y + 0.16, 5.1, 0.32, { fontSize: 8.3, color: C.text, margin: 0 });
}

panel(slide, "Evaluation Discipline", 15.0, 21.45, 18.3, 5.15);
const evals = [
  ["Metric", "PR-AUC primary; ROC-AUC, F1, recall, Brier as companions"],
  ["Threshold", "chosen on validation only; applied unchanged to test"],
  ["Split", "patient-level train / validation / test separation"],
  ["Stress tests", "label-shuffle leakage probe, sensor ablation, bootstrap CIs, multi-seed DL"]
];
for (let i = 0; i < evals.length; i++) {
  const y = 22.1 + i * 0.86;
  addText(slide, evals[i][0], 15.48, y, 2.8, 0.25, { fontSize: 8.8, bold: true, color: C.aub, margin: 0 });
  addText(slide, evals[i][1], 18.35, y, 14.3, 0.34, { fontSize: 8.7, color: C.text, margin: 0 });
}

// Right column results
panel(slide, "Held-Out Test Results", 34.1, 6.45, 12.85, 10.5);
addText(slide, "Validation-selected v2 second run", 34.53, 7.08, 6.7, 0.28, {
  fontSize: 8.4,
  color: C.muted,
  margin: 0
});
const tableX = 34.5, tableY = 7.65;
const cols = [2.05, 2.2, 2.1, 1.55, 1.65, 1.8];
const headers = ["Model", "T/L/W", "PR-AUC", "ROC-AUC", "F1", "Test"];
let cx = tableX;
for (let i = 0; i < headers.length; i++) {
  addRound(slide, cx, tableY, cols[i], 0.52, C.aubDark, C.aubDark, { radius: 0.05 });
  addText(slide, headers[i], cx + 0.07, tableY + 0.15, cols[i] - 0.14, 0.18, { fontSize: 7.3, bold: true, color: "FFFFFF", align: "center", margin: 0 });
  cx += cols[i];
}
const results = [
  ["RF", "4/14/14", "0.555", "0.934", "0.667*", "325 (4+)"],
  ["XGB", "3/14/14", "0.526", "0.875", "0.308", "324 (4+)"],
  ["RNN", "3/14/14", "0.261+/-.148", "0.902+/-.057", "0.000", "324 (4+)"],
  ["LR", "4/14/0", "0.022", "0.615", "0.041", "350 (4+)"]
];
for (let r = 0; r < results.length; r++) {
  const y = tableY + 0.62 + r * 0.68;
  cx = tableX;
  const fill = r === 0 ? "FFF5E4" : r % 2 ? "FFFFFF" : "F9F5EF";
  for (let c = 0; c < headers.length; c++) {
    addRect(slide, cx, y, cols[c], 0.56, fill, "FFFFFF", { lineTransparency: 100 });
    addText(slide, results[r][c], cx + 0.05, y + 0.16, cols[c] - 0.1, 0.18, {
      fontSize: c === 0 ? 8.0 : 7.2,
      bold: r === 0 || c === 0,
      color: c === 2 && r === 0 ? C.aub : C.text,
      align: "center",
      margin: 0,
      fit: "shrink"
    });
    cx += cols[c];
  }
}
addText(slide, "*RF value is validation-tuned F1. XGB F1 shown is untuned test F1; tuned F1 was lower.", 34.55, 11.18, 11.7, 0.45, {
  fontSize: 7.2,
  color: C.muted,
  margin: 0
});
addRound(slide, 34.55, 12.0, 11.85, 1.58, C.soft, C.rule, { radius: 0.12 });
addText(slide, "Wide uncertainty", 34.9, 12.25, 3.7, 0.28, { fontSize: 9, bold: true, color: C.warn, margin: 0 });
addText(slide, "RF PR-AUC 95% CI: 0.033 to 1.000 because the held-out test set had only four positives.", 34.9, 12.66, 10.9, 0.55, {
  fontSize: 8.9,
  color: C.text,
  margin: 0
});
addText(slide, "Takeaway", 34.55, 14.18, 2.6, 0.28, { fontSize: 9, bold: true, color: C.aub, margin: 0 });
addText(slide, "Promising rank signal exists, but clinical reliability is not established.", 37.0, 14.16, 8.9, 0.35, {
  fontSize: 9.2,
  bold: true,
  color: C.ink,
  margin: 0
});

panel(slide, "Interpreting Multimodal Signal", 34.1, 17.45, 12.85, 5.95);
bullet(slide, "All-sensor tabular RF produced the strongest validation-selected held-out result.", 34.78, 18.08, 11.25, 0.55, { fontSize: 9.1 });
bullet(slide, "Ablation patterns were not monotonic; sensor-source importance is not cleanly separable.", 34.78, 18.82, 11.25, 0.55, { fontSize: 9.1 });
bullet(slide, "Questionnaire-derived predictors may be close to symptom-based labels and must be checked for proxy-label effects.", 34.78, 19.56, 11.25, 0.68, { fontSize: 9.1 });
bullet(slide, "High accuracy alone is not meaningful in this rare-event setting.", 34.78, 20.42, 11.25, 0.5, { fontSize: 9.1 });
addRound(slide, 34.55, 21.55, 11.85, 0.9, "FFFFFF", C.gold, { radius: 0.13, lineWidth: 1.2 });
addText(slide, "Scientific reading: best signal found, not deployment readiness.", 34.85, 21.82, 11.25, 0.28, {
  fontSize: 9.1,
  bold: true,
  color: C.aub,
  align: "center",
  margin: 0
});

panel(slide, "Limitations & Next Steps", 34.1, 23.85, 12.85, 5.95);
const lims = [
  ["Small cohort", "~22 enrolled; 20 modeled users after preprocessing"],
  ["Few events", "test positives often 4 or fewer"],
  ["Stability", "leakage probes near-random overall, but individual configs unstable"],
  ["Next", "patient-level CV, raw-minute/downsampled representations, external validation"]
];
for (let i = 0; i < lims.length; i++) {
  const y = 24.55 + i * 1.0;
  addText(slide, lims[i][0], 34.55, y, 2.8, 0.28, { fontSize: 8.7, bold: true, color: C.aub, margin: 0 });
  addText(slide, lims[i][1], 37.25, y, 8.85, 0.38, { fontSize: 8.4, color: C.text, margin: 0 });
}

// Bottom conclusion band
addRect(slide, 0, 31.0, 48, 5.0, C.aubDark, C.aubDark);
addText(slide, "What we contributed", 1.25, 31.55, 7.5, 0.4, { fontSize: 12.2, bold: true, color: "FFFFFF", margin: 0 });
const contribs = [
  ["RIGOROUS SAMPLING", "event-episode windows, post-event washout, patient-safe splits"],
  ["MODEL COMPARISON", "LR/RF/XGB + RNN/GRU/LSTM/CNN under the same validation rule"],
  ["TRUST CHECKS", "calibration, train-val gap, leakage probes, ablation, bootstrap CIs"],
  ["HONEST RESULT", "promising signal with explicit uncertainty and non-deployment caveat"]
];
for (let i = 0; i < contribs.length; i++) {
  const x = 1.25 + i * 11.6;
  addRound(slide, x, 32.25, 10.55, 2.15, i === 3 ? C.copper : "4A2923", i === 3 ? C.copper : "4A2923", { radius: 0.16 });
  addText(slide, contribs[i][0], x + 0.42, 32.55, 9.75, 0.28, { fontSize: 8.3, bold: true, color: C.gold, align: "center", margin: 0 });
  addText(slide, contribs[i][1], x + 0.55, 33.05, 9.45, 0.78, { fontSize: 9.2, color: "FFFFFF", align: "center", margin: 0, fit: "shrink" });
}
addText(slide, "Code and results: v2 event-episode protocol - final numbers from v2_results_second_run/v2/tables - Poster generated as editable PowerPoint", 1.25, 35.15, 37.0, 0.3, {
  fontSize: 7.6,
  color: "D8C8B9",
  margin: 0
});
addText(slide, "A+ target: clear pipeline, credible validation, restrained claims", 38.2, 35.12, 8.55, 0.35, {
  fontSize: 7.6,
  color: C.gold,
  bold: true,
  align: "right",
  margin: 0
});

// Small decorative but structural rules.
addLine(slide, 14.55, 6.45, 14.55, 29.8, C.rule, 1.0);
addLine(slide, 33.65, 6.45, 33.65, 29.8, C.rule, 1.0);

(async () => {
const pptxPath = path.join(OUT_DIR, "Early_Warning_Asthma_AUB_Poster.pptx");
await pptx.writeFile({ fileName: pptxPath });

// Create a lightweight preview PNG with the same section geometry and main text.
const W = 2400, H = 1800;
function esc(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function svgText(text, x, y, size, color, weight = 400, anchor = "start", family = "Arial") {
  const lines = String(text).split("\n");
  return lines.map((line, i) =>
    `<text x="${x}" y="${y + i * size * 1.18}" font-family="${family}" font-size="${size}" font-weight="${weight}" fill="#${color}" text-anchor="${anchor}">${esc(line)}</text>`
  ).join("");
}
function svgBlock(text, x, y, size, color, width, weight = 400, family = "Arial", gap = 1.18) {
  const maxChars = Math.max(10, Math.floor(width / (size * 0.52)));
  const lines = [];
  for (const raw of String(text).split("\n")) {
    const words = raw.split(/\s+/);
    let line = "";
    for (const word of words) {
      const next = line ? `${line} ${word}` : word;
      if (next.length > maxChars && line) {
        lines.push(line);
        line = word;
      } else {
        line = next;
      }
    }
    if (line) lines.push(line);
  }
  return lines.map((line, i) =>
    `<text x="${x}" y="${y + i * size * gap}" font-family="${family}" font-size="${size}" font-weight="${weight}" fill="#${color}">${esc(line)}</text>`
  ).join("");
}
function svgRect(x, y, w, h, fill, stroke = fill, rx = 8, sw = 1) {
  return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" fill="#${fill}" stroke="#${stroke}" stroke-width="${sw}"/>`;
}
function svgLine(x1, y1, x2, y2, color, sw = 3, arrow = false) {
  return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#${color}" stroke-width="${sw}"${arrow ? ' marker-end="url(#arrow)"' : ""}/>`;
}
let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">`;
svg += `<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#${C.muted}"/></marker></defs>`;
svg += svgRect(0, 0, W, H, C.cream, C.cream, 0, 0);
svg += svgRect(0, 0, W, 213, C.aubDark, C.aubDark, 0, 0);
svg += svgRect(0, 0, W, 9, C.gold, C.gold, 0, 0);
svg += svgText("Early-Warning Prediction of Asthma Exacerbations", 60, 78, 54, "FFFFFF", 700, "start", "Georgia");
svg += svgText("using wearable and digital-health data", 63, 132, 25, "E9DCCC", 400);
svg += svgText("Bahaa Hamdan  |  Tarek El Moura  |  Anthony John Kanaan", 63, 168, 22, "FFFFFF", 700);
svg += svgRect(1805, 32, 525, 128, "FFFFFF", "FFFFFF", 10, 1);
svg += svgText("AUB", 1832, 91, 50, C.aub, 700, "start", "Georgia");
svg += svgText("American University\nof Beirut", 1970, 79, 26, C.ink, 700);
svg += svgRect(52, 233, 2295, 63, C.panel, C.rule, 10, 1);
svg += svgText("Core claim", 78, 264, 16, C.aub, 700);
svg += svgText("Patient-safe early-warning pipeline; promising signal, but statistically fragile with only four positive held-out events.", 205, 268, 23, C.ink, 700);
const p = [[52,322,657,310,"Clinical Task"],[52,655,657,395,"Data & Labels"],[52,1072,657,258,"Sampling Grid"],[750,322,915,728,"Full Pipeline"],[750,1072,915,258,"Evaluation Discipline"],[1705,322,642,525,"Held-Out Test Results"],[1705,872,642,298,"Interpreting Multimodal Signal"],[1705,1192,642,298,"Limitations & Next Steps"]];
for (const [x,y,w,h,t] of p) { svg += svgRect(x,y,w,h,C.panel,C.rule,10,1); svg += svgText(t.toUpperCase(), x+20, y+32, 15, C.aub, 700); }
svg += svgBlock("Goal: predict upcoming asthma exacerbation risk before clinically meaningful worsening.\nHard because positives are rare, baselines are patient-specific, and missingness can reflect behavior, adherence, charging, or sensor dropout.\nPrimary selection metric: validation PR-AUC.", 78, 390, 19, C.text, 570, 400);
svg += svgBlock("Sources: smartwatch heart rate, steps, activity and intensity; smart inhaler use; peak flow; questionnaires; environment; patient descriptors.\nLabels: positive windows before event onset; stable negatives away from events; post-event washout excluded from negative sampling.", 78, 725, 18, C.text, 570, 400);
svg += svgBlock("Grid explored: event threshold T = 2/3/4, history length L = 3/7/14 days, washout W = 0/7/14 days.\nExample T3/L14/W14: 53 positive and 1,071 negative samples.", 78, 1140, 19, C.text, 570, 400);
svg += svgText("Full pipeline", 780, 385, 24, C.ink, 700);
const nodes = ["Raw data","Clean","Labels","Features","Split","Tune","Select","Test"];
for (let i=0;i<nodes.length;i++) {
  const x=785+(i%4)*140, y=430+Math.floor(i/4)*105;
  svg += svgRect(x,y,115,50, i<3?C.aub:(i<6?C.copper:C.tealDark), i<3?C.aub:(i<6?C.copper:C.tealDark), 10, 1);
  svg += svgText(nodes[i], x+57, y+31, 16, "FFFFFF", 700, "middle");
  if (i < 3) svg += svgLine(x + 118, y + 25, x + 135, y + 25, C.muted, 2, true);
  if (i >= 4 && i < 7) svg += svgLine(x + 118, y + 25, x + 135, y + 25, C.muted, 2, true);
}
svg += svgLine(1360, 455, 1360, 535, C.muted, 2, true);
svg += svgBlock("Clean multi-source data -> create event episodes -> build lookback features -> split by patient -> tune LR/RF/XGB and sequence DL -> select by validation PR-AUC -> evaluate once on held-out test.", 785, 670, 18, C.text, 780, 400);
svg += svgBlock("Label timing: input history (E-L ... E-1) feeds the model; event onset E defines the positive boundary; post-event washout is excluded from stable-negative sampling.", 785, 820, 18, C.aub, 700, 700);
svg += svgBlock("Selection discipline: PR-AUC first; recall, F1, ROC-AUC, Brier, train-validation gap, leakage probe, ablation, and multi-seed stability as safeguards. Thresholds are chosen on validation only.", 780, 1140, 18, C.text, 790, 400);
svg += svgText("Best held-out result", 1728, 382, 20, C.ink, 700);
svg += svgText("RF  PR-AUC 0.555   ROC-AUC 0.934   tuned F1 0.667", 1728, 430, 21, C.aub, 700);
svg += svgText("95% CI for RF PR-AUC: 0.033 to 1.000", 1728, 487, 18, C.warn, 700);
svg += svgText("Do not infer deployment readiness.", 1728, 532, 18, C.text, 700);
svg += svgBlock("XGB: PR-AUC 0.526, ROC-AUC 0.875, F1 0.308\nRNN mean: PR-AUC 0.261+/-0.148\nLR: PR-AUC 0.022", 1728, 595, 17, C.text, 560, 400);
svg += svgBlock("All-sensor tabular RF produced the strongest validation-selected result, but ablation patterns were not monotonic. Accuracy is not meaningful for choosing models in this rare-event problem.", 1728, 940, 18, C.text, 560, 400);
svg += svgBlock("Small cohort, very few held-out positives, proxy-label risks from symptom questionnaires, and unstable configurations. Next: patient-level CV, raw-minute/downsampled experiments, external validation.", 1728, 1260, 18, C.text, 560, 400);
svg += svgRect(0,1550,W,250,C.aubDark,C.aubDark,0,0);
svg += svgText("What we contributed", 60, 1616, 28, "FFFFFF", 700);
svg += svgText("RIGOROUS SAMPLING     MODEL COMPARISON     TRUST CHECKS     HONEST RESULT", 60, 1690, 30, C.gold, 700);
svg += svgBlock("event-episode windows and washout     LR/RF/XGB plus RNN/GRU/LSTM/CNN     leakage, ablation, CIs, calibration     promising signal with restrained claims", 60, 1740, 20, "E9DCCC", 2050, 400);
svg += `</svg>`;

const previewPath = path.join(OUT_DIR, "Early_Warning_Asthma_AUB_Poster_preview.png");
await sharp(Buffer.from(svg)).png().toFile(previewPath);

console.log(JSON.stringify({ pptxPath, previewPath }, null, 2));
})().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
