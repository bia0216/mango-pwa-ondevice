// On-device models (prototype 2): the 3 YOLO models run INSIDE the browser with ONNX Runtime Web.
// It does the same steps and uses the same rules as the Colab server (cell 7 of mango_models_compare.ipynb),
// and returns the same JSON shapes, so app.js can show the results with its existing code.
// Model files: models/detector.onnx, models/ripeness_yolo.onnx, models/disease_yolo.onnx, models/models.json
// (exported and checked in mango_ondevice_export.ipynb: same answers as the original models).
(() => {
  "use strict";

  const ORT_VERSION = "1.20.1";
  const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
  const DET = { size: 640, conf: 0.35, iou: 0.7, max: 12, minSide: 32, pad: 114 };   // same as the server
  const CROP_PAD = 0.05;                                 // 5% border, like the training crops
  const DISEASE_MIN = 0.50, HEALTHY_MIN = 0.60, NOT_MANGO_MIN = 0.60, LEAF_GREEN_MIN = 0.10;
  const RIPE = ["unripe", "ripe", "overripe"];
  const D_ORDER = ["healthy", "anthracnose", "stem_end_rot", "leaf_healthy", "leaf_bacterial_canker", "leaf_other"];
  const PRETTY = { healthy: "Healthy", anthracnose: "Anthracnose", stem_end_rot: "Stem-end rot",
                   leaf_healthy: "Leaf: healthy", leaf_bacterial_canker: "Leaf: bacterial canker", leaf_other: "Leaf: other disease" };
  const R_COLORS = { unripe: "rgb(63,127,34)", ripe: "rgb(214,160,0)", overripe: "rgb(139,74,28)", "not a mango": "rgb(130,130,130)" };
  const V_COLORS = { "Disease found": "rgb(200,50,40)", "Check by hand": "rgb(130,130,130)", "Healthy": "rgb(63,127,34)",
                     "Not a mango or leaf": "rgb(110,110,110)" };
  const V_ORDER = { "Disease found": 0, "Check by hand": 1, "Healthy": 2, "Not a mango or leaf": 3 };
  const LABEL = "On this phone";

  // ---------- loading (once; the service worker keeps the files for offline use) ----------
  let loading = null, meta = null;
  const S = {};                                          // sessions: detector, ripeness, disease

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = src; s.crossOrigin = "anonymous";        // CORS, so the service worker can keep it for offline use
      s.onload = resolve;
      s.onerror = () => reject(new Error("Could not load ONNX Runtime. Open the app once with internet so it can be saved for offline use."));
      document.head.appendChild(s);
    });
  }

  function ready(onStatus) {
    if (!loading) {
      loading = (async () => {
        if (!window.ort) await loadScript(ORT_BASE + "ort.min.js");
        ort.env.wasm.wasmPaths = ORT_BASE;
        ort.env.wasm.numThreads = 1;                     // GitHub Pages is not cross-origin isolated
        const r = await fetch("models/models.json");
        if (!r.ok) throw new Error("models/models.json not found: put the models folder next to index.html.");
        meta = (await r.json()).models;
        for (const [key, name] of [["detector", "detector"], ["ripeness", "ripeness_yolo"], ["disease", "disease_yolo"]]) {
          if (onStatus) onStatus(`Loading ${key} model…`);
          const m = await fetch("models/" + meta[name].file);
          if (!m.ok) throw new Error(`models/${meta[name].file} not found.`);
          S[key] = await ort.InferenceSession.create(new Uint8Array(await m.arrayBuffer()), { executionProviders: ["wasm"] });
          S[key].names = meta[name].names;
          S[key].inputName = meta[name].input.name;
        }
      })().catch((e) => { loading = null; throw e; });
    }
    return loading;
  }

  // ---------- images ----------
  async function toBitmap(blob, maxSide) {
    const bmp = await createImageBitmap(blob, { imageOrientation: "from-image" });
    const s = Math.min(1, maxSide / Math.max(bmp.width, bmp.height));
    if (s === 1) return canvasOf(bmp, bmp.width, bmp.height);
    return canvasOf(bmp, Math.round(bmp.width * s), Math.round(bmp.height * s));
  }
  function canvasOf(src, w, h) {
    const c = document.createElement("canvas");
    c.width = w; c.height = h;
    const g = c.getContext("2d");
    g.imageSmoothingQuality = "high";
    g.drawImage(src, 0, 0, w, h);
    return c;
  }
  function toTensor(c) {                                 // canvas -> [1,3,H,W] float32, RGB, 0..1
    const { width: w, height: h } = c;
    const d = c.getContext("2d").getImageData(0, 0, w, h).data;
    const out = new Float32Array(3 * w * h), n = w * h;
    for (let i = 0; i < n; i++) {
      out[i] = d[i * 4] / 255; out[n + i] = d[i * 4 + 1] / 255; out[2 * n + i] = d[i * 4 + 2] / 255;
    }
    return new ort.Tensor("float32", out, [1, 3, h, w]);
  }

  // classifier input: shorter side -> 224, centre crop (checked: same answer as the original on 100% of 40 photos)
  function resizeCrop(src, size = 224) {
    const w = src.width, h = src.height, s = size / Math.min(w, h);
    const sw = Math.max(size, Math.round(w * s)), sh = Math.max(size, Math.round(h * s));
    const c = document.createElement("canvas");
    c.width = size; c.height = size;
    const g = c.getContext("2d");
    g.imageSmoothingQuality = "high";
    g.drawImage(src, -Math.floor((sw - size) / 2), -Math.floor((sh - size) / 2), sw, sh);
    return c;
  }

  async function classify(sess, src) {
    const out = await sess.run({ [sess.inputName]: toTensor(resizeCrop(src)) });
    const p = Array.from(out[Object.keys(out)[0]].data);
    const probs = {};
    sess.names.forEach((n, i) => { probs[n] = p[i]; });  // map by NAME (YOLO sorts class names A-Z)
    return probs;
  }

  // ---------- detector ----------
  function iou(a, b) {
    const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])), iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
    const inter = ix * iy;
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9);
  }

  async function findMangoes(img) {
    const size = DET.size, r = Math.min(size / img.width, size / img.height);
    const nw = Math.round(img.width * r), nh = Math.round(img.height * r);
    const dx = Math.floor((size - nw) / 2), dy = Math.floor((size - nh) / 2);
    const c = document.createElement("canvas");
    c.width = size; c.height = size;
    const g = c.getContext("2d");
    g.fillStyle = `rgb(${DET.pad},${DET.pad},${DET.pad})`;
    g.fillRect(0, 0, size, size);
    g.imageSmoothingQuality = "high";
    g.drawImage(img, dx, dy, nw, nh);
    const out = await S.detector.run({ [S.detector.inputName]: toTensor(c) });
    const t = out[Object.keys(out)[0]];                  // [1, 5, N]: cx, cy, w, h, score
    const N = t.dims[2], d = t.data;
    let boxes = [];
    for (let i = 0; i < N; i++) {
      const sc = d[4 * N + i];
      if (sc < DET.conf) continue;
      const cx = d[i], cy = d[N + i], w = d[2 * N + i], h = d[3 * N + i];
      boxes.push([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, sc]);
    }
    boxes.sort((a, b) => b[4] - a[4]);
    const keep = [];
    for (const b of boxes) if (keep.every((k) => iou(b, k) < DET.iou)) keep.push(b);
    boxes = keep.map((b) => [
      Math.max(0, (b[0] - dx) / r), Math.max(0, (b[1] - dy) / r),
      Math.min(img.width, (b[2] - dx) / r), Math.min(img.height, (b[3] - dy) / r), b[4]]);
    boxes = boxes.filter((b) => Math.min(b[2] - b[0], b[3] - b[1]) >= DET.minSide);
    boxes.sort((a, b) => (b[2] - b[0]) * (b[3] - b[1]) - (a[2] - a[0]) * (a[3] - a[1]));
    return boxes.slice(0, DET.max);
  }

  function crop(img, b) {                                // 5% border, like crop_box on the server
    const w = b[2] - b[0], h = b[3] - b[1];
    const x0 = Math.floor(Math.max(0, b[0] - CROP_PAD * w)), y0 = Math.floor(Math.max(0, b[1] - CROP_PAD * h));
    const x1 = Math.floor(Math.min(img.width, b[2] + CROP_PAD * w)), y1 = Math.floor(Math.min(img.height, b[3] + CROP_PAD * h));
    const c = document.createElement("canvas");
    c.width = Math.max(1, x1 - x0); c.height = Math.max(1, y1 - y0);
    c.getContext("2d").drawImage(img, x0, y0, c.width, c.height, 0, 0, c.width, c.height);
    return c;
  }

  function drawBoxes(img, items, colorOf) {              // numbered boxes, coloured by result -> JPEG data URL
    const c = canvasOf(img, img.width, img.height), g = c.getContext("2d");
    const lw = Math.max(3, Math.round(Math.max(c.width, c.height) / 220));
    const fs = Math.max(18, Math.round(Math.max(c.width, c.height) / 28));
    g.font = `bold ${fs}px sans-serif`;
    g.textBaseline = "top";
    for (const it of items) {
      if (!it.box) continue;
      const [x1, y1, x2, y2] = it.box, col = colorOf(it);
      g.lineWidth = lw; g.strokeStyle = col;
      g.strokeRect(x1, y1, x2 - x1, y2 - y1);
      const t = String(it.n), tw = g.measureText(t).width;
      g.fillStyle = col; g.fillRect(x1, y1, tw + 2 * lw, fs + 2 * lw);
      g.fillStyle = "#fff"; g.fillText(t, x1 + lw, y1 + lw);
    }
    return c.toDataURL("image/jpeg", 0.9);
  }

  // ---------- ripeness (same rule as the server: not_mango on top -> "not a mango"; else the 3 classes renormalised) ----------
  function ripenessItem(probs) {
    const top = Object.keys(probs).reduce((a, b) => (probs[a] >= probs[b] ? a : b));
    if (top === "not_mango") return { label: "not a mango", conf: probs.not_mango, probs: {} };
    const sum = RIPE.reduce((s, k) => s + (probs[k] || 0), 0) || 1;
    const rp = {};
    RIPE.forEach((k) => { rp[k] = (probs[k] || 0) / sum; });
    const k = RIPE.reduce((a, b) => (rp[a] >= rp[b] ? a : b));
    return { label: k, conf: rp[k], probs: rp };
  }

  async function ripenessScan(blob) {
    await ready();
    const t0 = performance.now();
    const img = await toBitmap(blob, 1280);
    const boxes = await findMangoes(img);
    const parts = boxes.length ? boxes.map((b) => [b, crop(img, b)]) : [[null, img]];
    const items = [];
    for (let i = 0; i < parts.length; i++) {
      const [b, c] = parts[i];
      const it = ripenessItem(await classify(S.ripeness, c));
      it.n = i + 1; it.key = it.label;
      it.box = b ? b.slice(0, 4).map(Math.round) : null;
      it.det = b ? Math.round(b[4] * 1000) / 1000 : null;
      items.push(it);
    }
    const mangoes = items.filter((it) => it.label !== "not a mango");
    const counts = {};
    RIPE.forEach((k) => { counts[k] = mangoes.filter((it) => it.label === k).length; });
    const summary = !mangoes.length ? "No mango found"
      : mangoes.length === 1 ? "1 mango: " + mangoes[0].label
      : `${mangoes.length} mangoes: ` + Object.entries(counts).filter(([, n]) => n).sort((a, b) => b[1] - a[1]).map(([k, n]) => `${n} ${k}`).join(", ");
    const res = { detector: true, whole_photo: !boxes.length, count: mangoes.length, counts, summary, items,
                  model: LABEL, ms: Math.round(performance.now() - t0) };
    const picture = boxes.length ? { url: drawBoxes(img, items, (it) => R_COLORS[it.label] || "rgb(150,150,150)") } : null;
    return [res, picture];
  }

  // live mode: whole frame, one answer, in the Gradio "Label" shape that app.js already reads
  async function predict(blob) {
    await ready();
    const img = await toBitmap(blob, 640);
    const it = ripenessItem(await classify(S.ripeness, img));
    if (it.label === "not a mango") return { label: "not a mango", confidences: [{ label: "not a mango", confidence: 1 }] };
    return { label: it.label, confidences: RIPE.map((k) => ({ label: k, confidence: it.probs[k] })) };
  }

  // ---------- disease (same verdict rule and checks as the server) ----------
  function verdict(disease, conf) {
    const part = disease.startsWith("leaf_") ? "leaf" : "mango";
    const healthy = disease === "healthy" || disease === "leaf_healthy";
    if (!healthy && conf >= DISEASE_MIN)
      return ["Disease found", `Signs of ${PRETTY[disease].replace("Leaf: ", "").toLowerCase()} on this ${part}. Keep it apart and ask an expert.`];
    if (healthy && conf >= HEALTHY_MIN) return ["Healthy", `No signs of the diseases we check for on this ${part}.`];
    return ["Check by hand", "The disease model is not sure. Look closely or ask an expert."];
  }

  async function diseaseOne(c) {
    const raw = await classify(S.disease, c);
    const probs = {};
    D_ORDER.forEach((k) => { probs[k] = raw[k] || 0; });
    const k = D_ORDER.reduce((a, b) => (probs[a] >= probs[b] ? a : b));
    const [v, advice] = verdict(k, probs[k]);
    return { disease: k, name: PRETTY[k], conf: probs[k], part: k.startsWith("leaf_") ? "leaf" : "mango",
             probs, verdict: v, advice, key: v };
  }

  async function notMango(c) {                           // the ripeness model's own not_mango class
    return (await classify(S.ripeness, c)).not_mango || 0;
  }

  function greenShare(src) {                              // share of clearly green pixels (rough "could this be a leaf?")
    const c = canvasOf(src, 128, 128), d = c.getContext("2d").getImageData(0, 0, 128, 128).data;
    let green = 0;
    for (let i = 0; i < d.length; i += 4) {
      const r = d[i] / 255, g = d[i + 1] / 255, b = d[i + 2] / 255;
      const mx = Math.max(r, g, b), mn = Math.min(r, g, b), delta = mx - mn;
      if (mx < 0.15 || mx === 0 || delta / mx < 0.20) continue;
      let h = mx === r ? ((g - b) / delta) % 6 : mx === g ? (b - r) / delta + 2 : (r - g) / delta + 4;
      h = (h * 60 + 360) % 360;
      if (h >= 60 && h <= 170) green++;
    }
    return green / (128 * 128);
  }

  async function diseaseScan(blob) {
    await ready();
    const t0 = performance.now();
    const img = await toBitmap(blob, 1280);
    const boxes = await findMangoes(img);
    const items = [];
    if (!boxes.length) {                                 // no mango found: whole photo (e.g. a leaf)
      const small = canvasOf(img, Math.round(img.width * Math.min(1, 640 / Math.max(img.width, img.height))),
                                  Math.round(img.height * Math.min(1, 640 / Math.max(img.width, img.height))));
      const g = await diseaseOne(small);
      g.n = 1; g.box = null;
      if (g.part === "mango") {
        g.not_mango = Math.round((await notMango(small)) * 1000) / 1000;
        if (g.not_mango >= NOT_MANGO_MIN)
          Object.assign(g, { verdict: "Not a mango or leaf", key: "Not a mango or leaf",
            advice: "Our models found no mango here and it doesn't look like a mango. Take a close photo of one mango, or of one mango leaf on its own." });
      } else {
        g.green_share = Math.round(greenShare(small) * 1000) / 1000;
        if (g.green_share < LEAF_GREEN_MIN)
          g.warning = "This doesn't look much like a leaf (very little green). If it isn't a mango leaf, ignore this result.";
      }
      items.push(g);
    } else {
      for (let i = 0; i < boxes.length; i++) {
        const c = crop(img, boxes[i]);
        const g = await diseaseOne(c);
        g.n = i + 1; g.box = boxes[i].slice(0, 4).map(Math.round); g.det = Math.round(boxes[i][4] * 1000) / 1000;
        g.not_mango = Math.round((await notMango(c)) * 1000) / 1000;
        if (g.not_mango >= NOT_MANGO_MIN)
          Object.assign(g, { verdict: "Not a mango or leaf", key: "Not a mango or leaf",
            advice: "This doesn't look like a mango to our ripeness model, so no disease verdict is given." });
        items.push(g);
      }
    }
    const worst = items.slice().sort((a, b) => (V_ORDER[a.verdict] - V_ORDER[b.verdict]) || (b.conf - a.conf))[0];
    const found = items.filter((g) => g.verdict === "Disease found").length;
    let summary;
    if (items.length === 1) summary = worst.verdict === "Not a mango or leaf" ? "Not a mango or leaf" : worst.name;
    else summary = `${items.length} mangoes: ` + Object.keys(V_ORDER)
      .filter((v) => items.some((g) => g.verdict === v))
      .map((v) => `${items.filter((g) => g.verdict === v).length} ${v.toLowerCase()}`).join(", ");
    const res = Object.assign({}, worst, { available: true, names: PRETTY, count: items.length, found,
      whole_photo: !boxes.length, detector: true, summary, items, model: LABEL, heatmap: false,
      ms: Math.round(performance.now() - t0) });
    const picture = boxes.length ? { url: drawBoxes(img, items, (it) => V_COLORS[it.verdict] || "rgb(150,150,150)") } : null;
    return [res, picture];
  }

  window.MangoOnDevice = { LABEL, ORT_BASE, ready, ripenessScan, predict, diseaseScan };
})();
