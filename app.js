(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const LS_KEY = "mangoOD.server";
  const LOW_CONF = 0.6;          // below this, show a "low confidence" hint (UI choice, not from the model)
  const MAX_SIDE = 1024;         // photos are downscaled before upload (the model resizes to 224x224 anyway)
  const CLASS_INFO = {           // descriptions from the project's labeling rubric
    unripe:   { color: "var(--unripe)",   text: "Fully or almost fully green, firm and glossy." },
    ripe:     { color: "var(--ripe)",     text: "Mostly or fully yellow, smooth and plump." },
    overripe: { color: "var(--overripe)", text: "Wrinkled, soft or collapsing look, or heavy dark patches or bruising." },
  };

  const el = {
    pill: $("serverPill"), pillText: $("serverText"),
    stage: $("stage"), placeholder: $("placeholder"), video: $("video"), preview: $("preview"), busy: $("busy"),
    mainControls: $("mainControls"), cameraControls: $("cameraControls"),
    cameraBtn: $("cameraBtn"), fileInput: $("fileInput"), snapBtn: $("snapBtn"), flipBtn: $("flipBtn"), closeCamBtn: $("closeCamBtn"),
    error: $("error"), resultCard: $("resultCard"), verdictLabel: $("verdictLabel"), verdictConf: $("verdictConf"),
    verdictText: $("verdictText"), lowConf: $("lowConf"), bars: $("bars"), againBtn: $("againBtn"),
    aiDesc: $("aiDesc"), aiDescText: $("aiDescText"), gradeLink: $("gradeLink"),
    settings: $("settings"), settingsBtn: $("settingsBtn"), serverInput: $("serverInput"), saveServerBtn: $("saveServerBtn"),
    settingsMsg: $("settingsMsg"), phoneBox: $("phoneBox"), qr: $("qr"), phoneLink: $("phoneLink"),
    installBtn: $("installBtn"), canvas: $("canvas"),
    liveBtn: $("liveBtn"), liveTag: $("liveTag"), liveText: $("liveText"),
    modelRow: $("modelRow"), modelSelect: $("modelSelect"), modelUsed: $("modelUsed"), mItems: $("mItems"),
    feedback: $("rFeedback"),
  };
  const LOW_ONE = el.lowConf.textContent;
  const LOW_MANY = "Low confidence on at least one mango. Check that mango in good light, close up.";
  const fileUrl = (f) => (f && (f.url || (f.path ? server + "/gradio_api/file=" + f.path : ""))) || "";

  // ---------- light / dark mode (index.html already set it before drawing) ----------
  const THEME_KEY = "mangoOD.theme";
  const themeBtn = $("themeBtn");
  function applyTheme(t) {
    document.documentElement.dataset.theme = t;
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.content = t === "light" ? "#FBF6EA" : "#0B140F";
    if (themeBtn) themeBtn.setAttribute("aria-label", t === "light" ? "Switch to dark mode" : "Switch to light mode");
  }
  applyTheme(document.documentElement.dataset.theme === "light" ? "light" : "dark");
  if (themeBtn) themeBtn.addEventListener("click", () => {
    const t = document.documentElement.dataset.theme === "light" ? "dark" : "light";
    applyTheme(t);
    try { localStorage.setItem(THEME_KEY, t); } catch (_) { /* ignore */ }
  });
  try {                                                  // follow the phone's setting until the user picks one
    window.matchMedia("(prefers-color-scheme: light)").addEventListener("change", (e) => {
      let saved = null;
      try { saved = localStorage.getItem(THEME_KEY); } catch (_) { /* ignore */ }
      if (!saved) applyTheme(e.matches ? "light" : "dark");
    });
  } catch (_) { /* old browsers */ }

  // ---------- model server (the Gradio app from notebook cell J) ----------
  let server = "";
  try { server = localStorage.getItem(LS_KEY) || ""; } catch (_) { /* storage blocked */ }

  function normalizeServer(u) {
    u = (u || "").trim();
    if (!u) return "";
    if (!/^https?:\/\//i.test(u)) u = "https://" + u;
    return new URL(u).origin;      // throws on an invalid URL
  }

  function saveServer(u) {
    server = normalizeServer(u);
    try { localStorage.setItem(LS_KEY, server); } catch (_) { /* ignore */ }
  }

  const params = new URLSearchParams(location.search);
  if (params.get("server")) {
    try { saveServer(params.get("server")); } catch (_) { /* ignore bad link */ }
    params.delete("server");
    const qs = params.toString();
    history.replaceState(null, "", location.pathname + (qs ? "?" + qs : "") + location.hash);
  }

  function setPill(state, text) {
    el.pill.dataset.state = state;
    el.pillText.textContent = text;
  }

  function timedFetch(url, opts = {}, ms = 30000) {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), ms);
    return fetch(url, { ...opts, signal: ctrl.signal }).finally(() => clearTimeout(t));
  }

  async function checkServer() {
    if (!server) { setPill("ok", "On-phone models · server optional (tap to add)"); return false; }
    setPill("checking", "Connecting to model…");
    try {
      const r = await timedFetch(server + "/config", { cache: "no-store" }, 10000);
      if (!r.ok) throw new Error(String(r.status));
      let cfg = null;
      try { cfg = await r.json(); } catch (_) { /* old server: no model list */ }
      setupModels(cfg);
      setPill("ok", "Model online · " + new URL(server).host);
      return true;
    } catch (_) {
      setPill("bad", "Model server offline (tap to change link)");
      return false;
    }
  }

  // ---------- model selector (shown only when the server offers a "Model" dropdown, e.g. the comparison notebook) ----------
  const MODEL_KEY = "mangoOD.model";
  let modelChoices = [];          // values the server accepts, e.g. ["MobileNetV2", "EfficientNetV2-B0", "YOLO26n-cls"]
  // the on-phone models (ondevice.js): always offered first, and the default ("phone first, server optional")
  const OD = window.MangoOnDevice || null;
  const DEVICE_R = OD ? OD.RIPENESS_MODELS : [], DEVICE_D = OD ? OD.DISEASE_MODELS : [];
  const isDev = (v) => !!OD && OD.isDevice(v);

  // fills a <select> from a Gradio dropdown; returns the accepted values
  function fillSelect(comp, select, row, key, device = []) {
    const raw = comp && comp.props && Array.isArray(comp.props.choices) ? comp.props.choices : [];
    // Gradio lists choices as ["Label", "value"] pairs (newer) or plain strings (older)
    const choices = raw.map((ch) => Array.isArray(ch) ? { label: String(ch[0]), value: String(ch[1] ?? ch[0]) }
                                                      : { label: String(ch), value: String(ch) });
    const values = choices.map((c) => c.value);
    const best = comp && comp.props && values.includes(String(comp.props.value)) ? String(comp.props.value) : "";
    select.innerHTML = "";
    const addGroup = (label, list, mark) => {
      if (!list.length) return;
      const grp = document.createElement("optgroup");
      grp.label = label;
      for (const c of list) {
        const o = document.createElement("option");
        o.value = c.value;
        o.textContent = c.label + (mark && c.value === best && values.length > 1 ? " (recommended)" : "");
        grp.appendChild(o);
      }
      select.appendChild(grp);
    };
    addGroup("On this phone (offline)", device.map((m) => ({ label: m.name + " · on phone", value: m.value })), false);
    addGroup("Colab server", choices, true);
    const all = device.map((m) => m.value).concat(values);
    let saved = "";
    try { saved = localStorage.getItem(key) || ""; } catch (_) { /* ignore */ }
    if (saved && all.includes(saved)) select.value = saved;   // the user's own choice wins
    else if (device.length) select.value = device[0].value;  // phone first (MobileNetV2 / EfficientNetV2-B0, like the server)
    else if (best) select.value = best;                     // otherwise the server's default (cell 7)
    row.hidden = all.length < 1;
    return values;                                   // server models only (the phone option is not sent to the server)
  }

  function setupModels(cfg) {
    const drops = cfg && Array.isArray(cfg.components) ? cfg.components.filter((c) => c.type === "dropdown") : [];
    const isDis = (c) => c.props && c.props.label === "Disease model";
    modelChoices = fillSelect(drops.find((c) => !isDis(c)), el.modelSelect, el.modelRow, MODEL_KEY, DEVICE_R);
    el.modelRow.hidden = !DEVICE_R.length && modelChoices.length === 0;   // ripeness: always show which model answered
    dModelChoices = fillSelect(drops.find(isDis), d.modelSelect, d.modelRow, D_MODEL_KEY, DEVICE_D);
  }
  const D_MODEL_KEY = "mangoOD.dmodel";
  let dModelChoices = [];

  el.modelSelect.addEventListener("change", () => {
    try { localStorage.setItem(MODEL_KEY, el.modelSelect.value); } catch (_) { /* ignore */ }
    el.resultCard.hidden = true;     // an old result belongs to the previous model
  });

  function currentModel() {
    const v = el.modelSelect.value;
    return v && (isDev(v) || modelChoices.includes(v)) ? v : "";
  }
  const onDevice = () => isDev(el.modelSelect.value);

  async function predict(blob) {
    const fd = new FormData();
    fd.append("files", blob, "mango.jpg");
    const up = await timedFetch(server + "/gradio_api/upload", { method: "POST", body: fd }, 60000);
    if (!up.ok) throw new Error("Upload to the model server failed (" + up.status + ").");
    const paths = await up.json();
    lastUploadPath = paths[0];                         // reused by describe() so the photo is uploaded once

    const call = await timedFetch(server + "/gradio_api/call/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data: modelChoices.length
        ? [{ path: paths[0], meta: { _type: "gradio.FileData" } }, currentModel()]   // photo + chosen model
        : [{ path: paths[0], meta: { _type: "gradio.FileData" } }] }),               // older one-model server
    }, 30000);
    if (!call.ok) throw new Error("The model server refused the request (" + call.status + ").");
    const { event_id } = await call.json();

    const res = await timedFetch(server + "/gradio_api/call/predict/" + event_id, {}, 120000);
    return parseSSE(await res.text());
  }

  let lastUploadPath = null;

  // Generic Gradio call: POST data, then read the "complete" event of the answer stream. Returns the data array or null.
  async function gradioCall(endpoint, data, ms = 60000) {
    const call = await timedFetch(server + "/gradio_api/call/" + endpoint, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ data }),
    }, 30000);
    if (!call.ok) return null;                       // e.g. 404: this server has no such endpoint
    const { event_id } = await call.json();
    const res = await timedFetch(server + "/gradio_api/call/" + endpoint + "/" + event_id, {}, ms);
    let event = null, out = null;
    for (const line of (await res.text()).split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:") && event === "complete") out = JSON.parse(line.slice(5).trim());
    }
    return out;
  }

  // ---------- disease scan: SEPARATE from ripeness (mentor), a mango OR a mango leaf, all from OUR model (cell 7) ----------
  // the class list comes from the server, so any disease model version works
  const D_COLORS = { healthy: "var(--unripe)", anthracnose: "var(--danger)", stem_end_rot: "var(--overripe)",
                     bacterial_canker: "#B784F5", black_tip: "#8FA3FF",
                     leaf_healthy: "#7FD49A", leaf_bacterial_canker: "#B784F5", leaf_other: "#E0A35C" };
  const VERDICT_STYLE = {
    "Healthy": { badge: "✓", cls: "g-a" }, "Disease found": { badge: "✕", cls: "g-x" }, "Check by hand": { badge: "?", cls: "g-q" },
    "Not a mango or leaf": { badge: "–", cls: "g-n" },
  };
  const d = {
    placeholder: $("dPlaceholder"), cam: $("dCam"), photo: $("dPhoto"), note: $("dNote"), busy: $("dBusy"),
    toggle: $("dToggle"), error: $("dError"), resultCard: $("dResultCard"), verdictCard: $("dVerdictCard"),
    label: $("dLabel"), conf: $("dConf"), bars: $("dBars"), badge: $("gBadge"), title: $("gTitle"), advice: $("gAdvice"),
    camInput: $("dCamInput"), fileInput: $("dFileInput"),
    summary: $("dSummary"), items: $("dItems"), which: $("dWhich"),
    modelSelect: $("dModelSelect"), modelRow: $("dModelRow"), warn: $("dWarn"), feedback: $("dFeedback"),
  };
  d.modelSelect.addEventListener("change", () => {
    try { localStorage.setItem(D_MODEL_KEY, d.modelSelect.value); } catch (_) { /* ignore */ }
  });
  const currentDModel = () => {
    const v = d.modelSelect.value;
    return v && (isDev(v) || dModelChoices.includes(v)) ? v : "";
  };
  const dOnDevice = () => isDev(d.modelSelect.value);

  function dError(msg) { if (d.error) { d.error.textContent = msg; d.error.hidden = !msg; } }

  async function uploadBlob(blob) {
    const fd = new FormData();
    fd.append("files", blob, "photo.jpg");
    const up = await timedFetch(server + "/gradio_api/upload", { method: "POST", body: fd }, 60000);
    if (!up.ok) throw new Error("Upload to the model server failed (" + up.status + ").");
    return (await up.json())[0];
  }

  function renderDisease(g, cam) {
    const notIt = g.verdict === "Not a mango or leaf";  // idea 4: the server's sanity check said "neither"
    d.label.textContent = notIt ? "Not a mango or leaf" : g.name;
    d.label.style.color = notIt ? "var(--muted)" : (D_COLORS[g.disease] || "var(--text)");
    d.conf.textContent = notIt ? "" : Math.round(g.conf * 100) + "% confident";
    d.warn.textContent = g.warning || "";
    d.warn.hidden = !g.warning;
    d.bars.innerHTML = "";
    for (const k of Object.keys(g.probs || {})) {
      const v = g.probs[k] || 0;
      const row = document.createElement("div");
      row.className = "bar-row";
      row.innerHTML = '<span class="name"></span><div class="bar-track"><div class="bar-fill"></div></div><span class="val"></span>';
      row.querySelector(".name").textContent = (g.names && g.names[k]) || k;
      row.querySelector(".val").textContent = (v * 100).toFixed(1) + "%";
      const fill = row.querySelector(".bar-fill");
      fill.style.background = D_COLORS[k] || "var(--muted)";
      d.bars.appendChild(row);
      requestAnimationFrame(() => requestAnimationFrame(() => { fill.style.width = (v * 100).toFixed(1) + "%"; }));
    }
    const st = VERDICT_STYLE[g.verdict] || { badge: "", cls: "" };
    d.badge.textContent = st.badge;
    d.badge.className = "grade-badge " + st.cls;
    d.title.textContent = g.verdict || "";
    d.advice.textContent = g.advice || "";
    const camUrl = fileUrl(cam);
    if (camUrl) {
      d.note.textContent = g.heatmap === false ? "Numbered boxes · " + (g.model || "this model") + " has no heat map"
                                               : "Red = where the disease model looked most";
      d.cam.src = camUrl; d.cam.hidden = false; d.photo.hidden = true; d.note.hidden = false;
      d.toggle.hidden = false; d.toggle.textContent = "Show photo";
    }
    d.resultCard.hidden = false;
    d.verdictCard.hidden = false;
  }

  // Several mangoes (detector on the server): one numbered chip per mango; the worst one is shown first.
  function renderDiseaseAll(g, cam) {
    const items = Array.isArray(g.items) && g.items.length ? g.items : [g];
    const multi = items.length > 1;
    let pick = items.findIndex((it) => it.n === g.n);
    if (pick < 0) pick = 0;
    d.summary.hidden = !multi;
    d.summary.textContent = multi ? (g.summary || items.length + " mangoes") : "";
    d.items.hidden = !multi;
    d.items.innerHTML = "";
    const show = (i, picture) => {
      [...d.items.children].forEach((b, j) => b.classList.toggle("on", j === i));
      renderDisease({ ...items[i], names: g.names, model: g.model, heatmap: g.heatmap }, picture);
      d.which.textContent = multi ? "Mango " + items[i].n + " of " + items.length : "Disease (our model)";
      if (fbD) { fbD = { ...fbD, n: items[i].n || 1 }; renderFeedback(d.feedback, fbD); }
    };
    if (multi) items.forEach((it, i) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "m-item";
      b.innerHTML = '<b class="m-num"></b><span class="m-name"></span><span class="m-val"></span>' + verdictBadge(it.verdict);
      b.querySelector(".m-num").textContent = it.n;
      b.querySelector(".m-num").style.background = VERDICT_COLORS[it.verdict] || "var(--muted)";
      b.querySelector(".m-name").textContent = it.name;
      b.querySelector(".m-val").textContent = Math.round(it.conf * 100) + "%";
      b.addEventListener("click", () => show(i, null));
      d.items.appendChild(b);
    });
    show(pick, cam);
  }

  async function scanDisease(file) {
    if (!file) return;
    dError("");
    if (!server && !dOnDevice()) { openSettings("Add the model server link first, then scan again."); return; }
    let jpeg;
    try { jpeg = await toJpeg(file); } catch (e) { dError(e.message || "Could not read this image."); return; }
    d.placeholder.hidden = true;
    d.photo.src = URL.createObjectURL(jpeg); d.photo.hidden = false;
    lastDPhoto = d.photo.src;
    fbD = null; renderFeedback(d.feedback, null);
    d.cam.hidden = true; d.note.hidden = true; d.toggle.hidden = true;
    d.resultCard.hidden = true; d.verdictCard.hidden = true;
    d.summary.hidden = true; d.items.hidden = true;
    d.busy.hidden = false;
    const device = dOnDevice();
    const busyText = d.busy.querySelector("span"), busyOld = busyText ? busyText.textContent : "";
    try {
      let out;
      if (device) {                                  // on the phone: no upload, no server
        out = await OD.diseaseScan(jpeg, d.modelSelect.value, (s) => { if (busyText) busyText.textContent = s; });
      } else {
        const path = await uploadBlob(jpeg);
        const file = { path, meta: { _type: "gradio.FileData" } };
        out = dModelChoices.length ? await gradioCall("disease_scan", [file, currentDModel()], 90000) : null;
        if (!out) out = await gradioCall("disease_scan", [file], 90000);   // older server: one disease model, photo only
      }
      const g = out && out[0];
      if (!g || g.available === false) {
        dError("The model server has no disease scan yet. In Colab: train the disease model (mango_disease.ipynb, D3), "
               + "run the newest cell 7 of mango_models_compare.ipynb, and paste the new link in Settings.");
        return;
      }
      renderDiseaseAll(g, out[1]);
      const all = Array.isArray(g.items) && g.items.length ? g.items : [g];
      const t = Date.now();
      all.forEach((it) => addDiseaseScan({ ...it, model: g.model || currentDModel() }, t, all.length));
      fbD = { t, n: g.n || 1, of: all.length, disease: true, names: g.names || {} };
      renderFeedback(d.feedback, fbD);
      if (!device) setPill("ok", "Model online · " + new URL(server).host);
    } catch (e) {
      const offline = !device && (e.name === "AbortError" || e instanceof TypeError);
      dError(offline ? "Can't reach the model server. Is cell 7 still running in Colab? Paste its newest link in Settings."
                     : (e.message || "Something went wrong."));
      if (offline) setPill("bad", "Model server offline (tap to change link)");
    } finally {
      d.busy.hidden = true;
      if (busyText) busyText.textContent = busyOld;
    }
  }

  for (const inp of [d.camInput, d.fileInput]) {
    if (inp) inp.addEventListener("change", () => { const f = inp.files && inp.files[0]; inp.value = ""; scanDisease(f); });
  }
  // in-app camera for the disease scan (same as the ripeness tab); the phone's own camera picker is the fallback
  const dv = { video: $("dVideo"), main: $("dMainControls"), cam: $("dCamControls"), flip: $("dFlipBtn") };
  let dStream = null, dFacing = "environment";
  function dStopCamera() {
    if (dStream) dStream.getTracks().forEach((t) => t.stop());
    dStream = null;
    dv.video.srcObject = null;
    dv.video.hidden = true;
    dv.main.hidden = false;
    dv.cam.hidden = true;
  }
  async function dOpenCamera() {
    dError("");
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) { d.camInput.click(); return; }
    try {
      dStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: dFacing }, width: { ideal: 1280 }, height: { ideal: 960 } }, audio: false });
    } catch (e) {
      dError("Camera not available (" + (e.name || "error") + "). Allow camera access, or use Upload photo.");
      return;
    }
    dv.video.srcObject = dStream;
    dv.video.hidden = false;
    d.placeholder.hidden = true; d.photo.hidden = true; d.cam.hidden = true; d.note.hidden = true; d.toggle.hidden = true;
    dv.main.hidden = true;
    dv.cam.hidden = false;
    try { await dv.video.play(); } catch (_) { /* autoplay quirks */ }
    try {
      const cams = (await navigator.mediaDevices.enumerateDevices()).filter((x) => x.kind === "videoinput");
      dv.flip.hidden = cams.length < 2;
    } catch (_) { dv.flip.hidden = true; }
  }
  async function dSnap() {
    const v = dv.video;
    if (!v.videoWidth) return;
    const c = document.createElement("canvas");
    c.width = v.videoWidth; c.height = v.videoHeight;
    c.getContext("2d").drawImage(v, 0, 0);
    const blob = await new Promise((r) => c.toBlob(r, "image/jpeg", 0.92));
    dStopCamera();
    scanDisease(blob);
  }
  $("dCamBtn").addEventListener("click", dOpenCamera);
  $("dSnapBtn").addEventListener("click", dSnap);
  $("dCloseCamBtn").addEventListener("click", () => {
    dStopCamera();
    if (d.photo.hidden && d.cam.hidden) d.placeholder.hidden = false;
  });
  dv.flip.addEventListener("click", async () => {
    dFacing = dFacing === "environment" ? "user" : "environment";
    dStopCamera();
    await dOpenCamera();
  });

  if (d.toggle) d.toggle.addEventListener("click", () => {
    const showCam = d.cam.hidden;
    d.cam.hidden = !showCam; d.photo.hidden = showCam; d.note.hidden = !showCam;
    d.toggle.textContent = showCam ? "Show photo" : "Show heat map";
  });

  // Optional text description written by OpenAI on the server (cell 7). The grade itself always comes from our model.
  // Servers without a "describe" endpoint, or without an OpenAI key, simply return nothing and the box stays hidden.
  async function describe(label) {
    el.aiDesc.hidden = true;
    if (!lastUploadPath || !label) return;
    try {
      const call = await timedFetch(server + "/gradio_api/call/describe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ data: [{ path: lastUploadPath, meta: { _type: "gradio.FileData" } }, label] }),
      }, 30000);
      if (!call.ok) return;
      const { event_id } = await call.json();
      const res = await timedFetch(server + "/gradio_api/call/describe/" + event_id, {}, 60000);
      let event = null, text = "";
      for (const line of (await res.text()).split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:") && event === "complete") {
          const data = JSON.parse(line.slice(5).trim());
          text = (data && typeof data[0] === "string") ? data[0].trim() : "";
        }
      }
      if (!text) return;
      el.aiDescText.textContent = text.replace(/\*\*/g, "");   // plain text, never HTML
      el.aiDesc.hidden = false;
    } catch (_) { /* description is optional */ }
  }

  function parseSSE(text) {
    let event = null;
    for (const line of text.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) {
        if (event === "complete") {
          const data = JSON.parse(line.slice(5).trim());
          if (!data || !data[0] || !data[0].confidences) throw new Error("The model returned no prediction.");
          return data[0];
        }
        if (event === "error") throw new Error("The model server reported an error while predicting.");
      }
    }
    throw new Error("No answer from the model server.");
  }

  // ---------- UI state ----------
  function showError(msg) { el.error.textContent = msg; el.error.hidden = !msg; }

  function resetView() {
    stopCamera();
    el.preview.hidden = true;
    el.preview.removeAttribute("src");
    el.placeholder.hidden = false;
    el.resultCard.hidden = true;
    el.busy.hidden = true;
    el.mainControls.hidden = false;
    el.cameraControls.hidden = true;
    showError("");
  }

  function renderResult(r, scroll = true, keepItems = false) {
    if (!keepItems) el.mItems.hidden = true;         // the per-mango chips belong to a several-mango result
    fbR = null; renderFeedback(el.feedback, null);   // set again once this scan is saved (not in live mode)
    el.lowConf.textContent = LOW_ONE;
    if (el.aiDesc) el.aiDesc.hidden = true;          // an old description belongs to the previous photo
    if (el.gradeLink) el.gradeLink.hidden = true;
    el.modelUsed.textContent = currentModel() ? "Model: " + currentModel() : "";
    el.modelUsed.hidden = !currentModel();
    if (!(r.label in CLASS_INFO)) {                // "not a mango", or any answer that is not a ripeness class
      el.verdictLabel.textContent = "Not a mango";
      el.verdictLabel.style.color = "var(--danger)";
      el.verdictLabel.style.textTransform = "none";
      el.verdictConf.textContent = "";
      el.verdictText.textContent = "This photo doesn't look like a mango, so no ripeness is given. Take a close photo of one mango.";
      el.lowConf.hidden = true;
      el.bars.innerHTML = "";
      el.resultCard.hidden = false;
      if (scroll) el.resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
      return;
    }
    el.verdictLabel.style.textTransform = "";
    const conf = {};
    for (const c of r.confidences) conf[c.label] = c.confidence;
    const top = r.label;
    const p = conf[top] ?? 0;
    const info = CLASS_INFO[top] || { color: "var(--accent)", text: "" };

    el.verdictLabel.textContent = top;
    el.verdictLabel.style.color = info.color;
    el.verdictConf.textContent = Math.round(p * 100) + "% confident";
    el.verdictText.textContent = info.text;
    el.lowConf.hidden = p >= LOW_CONF;

    el.bars.innerHTML = "";
    for (const name of Object.keys(CLASS_INFO)) {
      const v = conf[name] ?? 0;
      const row = document.createElement("div");
      row.className = "bar-row";
      row.innerHTML = '<span class="name"></span><div class="bar-track"><div class="bar-fill"></div></div><span class="val"></span>';
      row.querySelector(".name").textContent = name;
      row.querySelector(".val").textContent = (v * 100).toFixed(1) + "%";
      const fill = row.querySelector(".bar-fill");
      fill.style.background = CLASS_INFO[name].color;
      el.bars.appendChild(row);
      requestAnimationFrame(() => requestAnimationFrame(() => { fill.style.width = (v * 100).toFixed(1) + "%"; }));
    }
    el.resultCard.hidden = false;
    if (scroll) el.resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  // ---------- several mangoes in one photo (cell 7 "ripeness_scan": detector + one grade per mango) ----------
  const itemToR = (it) => ({ label: it.label,
    confidences: Object.entries(it.probs || {}).map(([label, confidence]) => ({ label, confidence })) });

  function renderMulti(res, picture) {
    const items = res.items;
    const url = fileUrl(picture);                    // the photo with numbered boxes (none = whole photo was graded)
    if (url) { el.preview.src = url; el.preview.hidden = false; el.placeholder.hidden = true; }
    if (items.length === 1) { renderResult(itemToR(items[0])); return; }
    fbR = null; renderFeedback(el.feedback, null);
    const show = (i) => {
      [...el.mItems.children].forEach((b, j) => b.classList.toggle("on", j === i));
      const it = items[i];
      if (fbR) { fbR = { ...fbR, n: it.n }; renderFeedback(el.feedback, fbR); }
      el.bars.innerHTML = "";
      if (!(it.label in CLASS_INFO)) return;
      for (const name of Object.keys(CLASS_INFO)) {
        const v = (it.probs || {})[name] ?? 0;
        const row = document.createElement("div");
        row.className = "bar-row";
        row.innerHTML = '<span class="name"></span><div class="bar-track"><div class="bar-fill"></div></div><span class="val"></span>';
        row.querySelector(".name").textContent = name;
        row.querySelector(".val").textContent = (v * 100).toFixed(1) + "%";
        const fill = row.querySelector(".bar-fill");
        fill.style.background = CLASS_INFO[name].color;
        el.bars.appendChild(row);
        requestAnimationFrame(() => requestAnimationFrame(() => { fill.style.width = (v * 100).toFixed(1) + "%"; }));
      }
    };
    if (el.aiDesc) el.aiDesc.hidden = true;
    el.modelUsed.textContent = currentModel() ? "Model: " + currentModel() : "";
    el.modelUsed.hidden = !currentModel();
    el.verdictLabel.textContent = res.count ? res.count + (res.count === 1 ? " mango" : " mangoes") : "No mango found";
    el.verdictLabel.style.color = "var(--text)";
    el.verdictLabel.style.textTransform = "none";
    el.verdictConf.textContent = "";
    el.verdictText.textContent = (res.summary || "").replace(/^\d+ mangoes?: /, "") + ". Tap a number to see that mango’s bars.";
    el.lowConf.textContent = LOW_MANY;
    el.lowConf.hidden = !items.some((it) => it.label in CLASS_INFO && it.conf < LOW_CONF);
    el.mItems.innerHTML = "";
    items.forEach((it, i) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "m-item";
      b.innerHTML = '<b class="m-num"></b><span class="m-name"></span><span class="m-val"></span>';
      b.querySelector(".m-num").textContent = it.n;
      b.querySelector(".m-num").style.background = COLORS[it.label] || "var(--muted)";
      b.querySelector(".m-name").textContent = cap(it.label);
      b.querySelector(".m-val").textContent = it.label in CLASS_INFO ? Math.round(it.conf * 100) + "%" : "";
      b.addEventListener("click", () => show(i));
      el.mItems.appendChild(b);
    });
    el.mItems.hidden = false;
    show(0);
    el.resultCard.hidden = false;
    el.resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  // ---------- images ----------
  async function toJpeg(fileOrBlob) {
    let bmp;
    try {
      bmp = await createImageBitmap(fileOrBlob, { imageOrientation: "from-image" });
    } catch (_) {
      bmp = await new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error("This file is not an image the browser can open."));
        img.src = URL.createObjectURL(fileOrBlob);
      });
    }
    const w = bmp.width, h = bmp.height;
    const s = Math.min(1, MAX_SIDE / Math.max(w, h));
    const c = el.canvas;
    c.width = Math.round(w * s);
    c.height = Math.round(h * s);
    const g = c.getContext("2d");
    g.drawImage(bmp, 0, 0, c.width, c.height);
    return new Promise((resolve) => c.toBlob(resolve, "image/jpeg", 0.92));
  }

  async function handleImage(fileOrBlob) {
    showError("");
    el.resultCard.hidden = true;
    let jpeg;
    try {
      jpeg = await toJpeg(fileOrBlob);
    } catch (e) {
      showError(e.message || "Could not read this image.");
      return;
    }
    el.preview.src = URL.createObjectURL(jpeg);
    lastRPhoto = el.preview.src;
    el.preview.hidden = false;
    el.placeholder.hidden = true;

    if (!server && !onDevice()) {
      pendingBlob = jpeg;
      openSettings("Add the model server link first. Your photo will be analysed right after.");
      return;
    }
    await analyse(jpeg);
  }

  let pendingBlob = null;

  async function analyse(jpeg) {
    el.busy.hidden = false;
    el.cameraBtn.disabled = true;
    if (onDevice()) {                                // on the phone: same result shape as the server's ripeness_scan
      const busyText = el.busy.querySelector("span"), busyOld = busyText ? busyText.textContent : "";
      try {
        const [res, picture] = await OD.ripenessScan(jpeg, el.modelSelect.value, (s) => { if (busyText) busyText.textContent = s; });
        renderMulti(res, picture);
        const t = addMultiScan(res);
        const on = el.mItems.querySelector(".m-item.on");
        const idx = on ? [...el.mItems.children].indexOf(on) : 0;
        fbR = { t, n: (res.items[idx] || res.items[0]).n || 1, of: res.items.length };
        renderFeedback(el.feedback, fbR);
      } catch (e) {
        showError(e.message || "The on-phone model could not run.");
      } finally {
        if (busyText) busyText.textContent = busyOld;
        el.busy.hidden = true;
        el.cameraBtn.disabled = false;
      }
      return;
    }
    try {
      if (modelChoices.length) {                     // the comparison server (cell 7) can find and grade EVERY mango
        const path = await uploadBlob(jpeg);
        lastUploadPath = path;
        const out = await gradioCall("ripeness_scan", [{ path, meta: { _type: "gradio.FileData" } }, currentModel()], 90000);
        const res = out && out[0];
        if (res && Array.isArray(res.items) && res.items.length) {
          setPill("ok", "Model online · " + new URL(server).host);
          try {
            renderMulti(res, out[1]);
            const t = addMultiScan(res);
            const on = el.mItems.querySelector(".m-item.on");     // the mango whose bars are shown
            const idx = on ? [...el.mItems.children].indexOf(on) : 0;
            fbR = { t, n: (res.items[idx] || res.items[0]).n || 1, of: res.items.length };
            renderFeedback(el.feedback, fbR);
            if (res.items.length === 1) describe(res.items[0].label);   // a description only for a single mango
          } catch (uiErr) {
            showError("The model answered (" + res.summary + "), but the app could not show it: " + uiErr.message +
                      ". Reload the page; if it stays, upload ALL files from mango-pwa.zip again.");
          }
          return;
        }                                            // older server without "ripeness_scan": one answer per photo
      }
      const r = await predict(jpeg);
      setPill("ok", "Model online · " + new URL(server).host);
      try {                                          // a display problem must not look like "server offline"
        renderResult(r);
        fbR = { t: addScan(r), n: 1, of: 1 };        // saved on this device for the Overview tab
        renderFeedback(el.feedback, fbR);
        const lbl = r.label in CLASS_INFO ? r.label : "not a mango";
        describe(lbl);                               // optional OpenAI text; not awaited (disease is a separate scan)
      } catch (uiErr) {
        showError("The model answered (" + r.label + "), but the app could not show it: " + uiErr.message +
                  ". Reload the page; if it stays, upload ALL files from mango-pwa.zip again.");
      }
    } catch (e) {
      const offline = e.name === "AbortError" || e instanceof TypeError;
      showError(offline
        ? "Can't reach the model server. Is cell J still running in Colab? Its link changes on every run: paste the new one in Settings (gear icon)."
        : (e.message || "Something went wrong."));
      if (offline) setPill("bad", "Model server offline (tap to change link)");
    } finally {
      el.busy.hidden = true;
      el.cameraBtn.disabled = false;
    }
  }

  // ---------- camera ----------
  let stream = null;
  let facing = "environment";

  async function openCamera() {
    showError("");
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      el.fileInput.click();          // e.g. not on HTTPS: fall back to the phone's own camera picker
      return;
    }
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: facing }, width: { ideal: 1280 }, height: { ideal: 960 } },
        audio: false,
      });
    } catch (e) {
      showError("Camera not available (" + (e.name || "error") + "). Allow camera access, or use Upload photo.");
      return;
    }
    el.video.srcObject = stream;
    el.video.hidden = false;
    el.preview.hidden = true;
    el.placeholder.hidden = true;
    el.resultCard.hidden = true;
    el.mainControls.hidden = true;
    el.cameraControls.hidden = false;
    try { await el.video.play(); } catch (_) { /* autoplay quirks */ }
    try {
      const cams = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
      el.flipBtn.hidden = cams.length < 2;
    } catch (_) { el.flipBtn.hidden = true; }
  }

  function stopCamera() {
    stopLive();
    if (stream) stream.getTracks().forEach((t) => t.stop());
    stream = null;
    el.video.srcObject = null;
    el.video.hidden = true;
  }

  async function snap() {
    const v = el.video;
    if (!v.videoWidth) return;
    const c = el.canvas;
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    c.getContext("2d").drawImage(v, 0, 0);
    const blob = await new Promise((resolve) => c.toBlob(resolve, "image/jpeg", 0.92));
    stopCamera();
    el.mainControls.hidden = false;
    el.cameraControls.hidden = true;
    handleImage(blob);
  }

  // ---------- live mode: predict on camera frames continuously ----------
  let liveOn = false;
  const LIVE_SIDE = 640;          // frames are downscaled before upload to keep it fast
  const LIVE_PAUSE_MS = 400;      // pause between predictions (one request at a time)

  function frameBlob() {
    const v = el.video;
    if (!v.videoWidth) return Promise.resolve(null);
    const s = Math.min(1, LIVE_SIDE / Math.max(v.videoWidth, v.videoHeight));
    const c = document.createElement("canvas");
    c.width = Math.round(v.videoWidth * s);
    c.height = Math.round(v.videoHeight * s);
    c.getContext("2d").drawImage(v, 0, 0, c.width, c.height);
    return new Promise((resolve) => c.toBlob(resolve, "image/jpeg", 0.85));
  }

  function showLive(r) {
    if (!(r.label in CLASS_INFO)) {
      el.liveTag.dataset.cls = "none";
      el.liveText.textContent = "Not a mango";
    } else {
      const c = r.confidences.find((x) => x.label === r.label);
      const pct = c ? Math.round(c.confidence * 100) : 0;
      el.liveTag.dataset.cls = r.label;
      el.liveText.textContent = r.label.charAt(0).toUpperCase() + r.label.slice(1) + " · " + pct + "%";
    }
    renderResult(r, false);
  }

  async function liveLoop() {
    while (liveOn && stream) {
      const blob = await frameBlob();
      if (blob) {
        try {
          const device = onDevice();
          const r = device ? await OD.predict(blob, el.modelSelect.value) : await predict(blob);
          if (!liveOn) break;
          showLive(r);
          if (!device) setPill("ok", "Model online · " + new URL(server).host);
        } catch (e) {
          stopLive();
          if (onDevice()) { showError("Live mode stopped: " + (e.message || "the on-phone model could not run.")); break; }
          showError("Live mode stopped: can't reach the model server. Is the Colab cell still running? Paste its new link in Settings if it changed.");
          setPill("bad", "Model server offline (tap to change link)");
          break;
        }
      }
      await new Promise((res) => setTimeout(res, LIVE_PAUSE_MS));
    }
  }

  function startLive() {
    if (!server && !onDevice()) { openSettings("Add the model server link first, then start live mode."); return; }
    showError("");
    liveOn = true;
    el.liveBtn.textContent = "Stop live";
    el.liveBtn.classList.add("live-on");
    delete el.liveTag.dataset.cls;
    el.liveText.textContent = "Live · analysing…";
    el.liveTag.hidden = false;
    liveLoop();
  }

  function stopLive() {
    liveOn = false;
    el.liveBtn.textContent = "Start live";
    el.liveBtn.classList.remove("live-on");
    el.liveTag.hidden = true;
  }

  el.liveBtn.addEventListener("click", () => (liveOn ? stopLive() : startLive()));
  el.cameraBtn.addEventListener("click", openCamera);
  el.snapBtn.addEventListener("click", snap);
  el.flipBtn.addEventListener("click", async () => {
    const wasLive = liveOn;
    facing = facing === "environment" ? "user" : "environment";
    stopCamera();
    await openCamera();
    if (wasLive && stream) startLive();
  });
  el.closeCamBtn.addEventListener("click", resetView);
  el.againBtn.addEventListener("click", () => { resetView(); el.stage.scrollIntoView({ behavior: "smooth", block: "center" }); });

  el.fileInput.addEventListener("change", () => {
    const f = el.fileInput.files && el.fileInput.files[0];
    el.fileInput.value = "";
    if (f) handleImage(f);
  });

  // drag & drop and paste (handy on a laptop during the demo)
  ["dragenter", "dragover"].forEach((t) => el.stage.addEventListener(t, (e) => { e.preventDefault(); el.stage.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((t) => el.stage.addEventListener(t, (e) => { e.preventDefault(); el.stage.classList.remove("drag"); }));
  el.stage.addEventListener("drop", (e) => {
    const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (f && f.type.startsWith("image/")) { stopCamera(); el.mainControls.hidden = false; el.cameraControls.hidden = true; handleImage(f); }
  });
  window.addEventListener("paste", (e) => {
    const item = [...(e.clipboardData ? e.clipboardData.items : [])].find((i) => i.type.startsWith("image/"));
    if (item) { stopCamera(); el.mainControls.hidden = false; el.cameraControls.hidden = true; handleImage(item.getAsFile()); }
  });

  // ---------- settings ----------
  function openSettings(msg) {
    el.serverInput.value = server;
    el.settingsMsg.textContent = msg || "";
    renderPhoneBox();
    if (typeof el.settings.showModal === "function") el.settings.showModal();
    else el.settings.setAttribute("open", "");
  }

  let qrLoading = null;
  function loadQrLib() {
    if (window.QRCode) return Promise.resolve();
    if (!qrLoading) {
      qrLoading = new Promise((resolve, reject) => {
        const s = document.createElement("script");
        s.src = "https://cdnjs.cloudflare.com/ajax/libs/qrcodejs/1.0.0/qrcode.min.js";
        s.onload = resolve;
        s.onerror = () => { qrLoading = null; reject(new Error("QR library failed to load")); };
        document.head.appendChild(s);
      });
    }
    return qrLoading;
  }

  async function renderPhoneBox() {
    if (!server) { el.phoneBox.hidden = true; return; }
    const link = location.origin + location.pathname + "?server=" + encodeURIComponent(server);
    el.phoneLink.href = link;
    el.phoneLink.textContent = link;
    el.phoneBox.hidden = false;
    el.qr.innerHTML = "";
    try {
      await loadQrLib();
      new window.QRCode(el.qr, { text: link, width: 180, height: 180, correctLevel: window.QRCode.CorrectLevel.M });
    } catch (_) {
      el.qr.textContent = "QR code unavailable offline. Use the link below.";
    }
  }

  el.settingsBtn.addEventListener("click", () => openSettings());
  el.pill.addEventListener("click", () => openSettings());
  el.saveServerBtn.addEventListener("click", async (e) => {
    e.preventDefault();
    try {
      saveServer(el.serverInput.value);
    } catch (_) {
      el.settingsMsg.textContent = "That doesn't look like a valid link.";
      return;
    }
    el.settingsMsg.textContent = "Testing…";
    const ok = await checkServer();
    el.settingsMsg.textContent = ok ? "Connected. You're ready." : "Can't reach that link. Is cell J running in Colab?";
    renderPhoneBox();
    if (ok && pendingBlob) {
      const b = pendingBlob;
      pendingBlob = null;
      el.settings.close();
      analyse(b);
    }
  });

  // ---------- install (PWA) ----------
  let deferredPrompt = null;
  window.addEventListener("beforeinstallprompt", (e) => {
    e.preventDefault();
    deferredPrompt = e;
    el.installBtn.hidden = false;
  });
  el.installBtn.addEventListener("click", async () => {
    if (!deferredPrompt) return;
    deferredPrompt.prompt();
    await deferredPrompt.userChoice;
    deferredPrompt = null;
    el.installBtn.hidden = true;
  });
  window.addEventListener("appinstalled", () => { el.installBtn.hidden = true; });

  if ("serviceWorker" in navigator) {
    window.addEventListener("load", () => { navigator.serviceWorker.register("sw.js").catch(() => {}); });
  }

  // ---------- scan history (kept only in this browser; feeds the Overview tab) ----------
  const HIST_KEY = "mangoOD.history";
  const COLORS = { unripe: "#5BC27A", ripe: "#F2B42C", overripe: "#C9733A", "not a mango": "#9FB3A5" };
  const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
  const esc = (s) => String(s).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));

  function loadScans() {
    try { return JSON.parse(localStorage.getItem(HIST_KEY) || "[]"); } catch (_) { return []; }
  }
  function saveScans(s) {
    try { localStorage.setItem(HIST_KEY, JSON.stringify(s.slice(-200))); } catch (_) { /* storage blocked */ }
  }
  // batches: a sorter starts a batch, scans its mangoes, then downloads a report
  const BATCH_KEY = "mangoOD.batch";
  function currentBatch() {
    let b = null;
    try { b = JSON.parse(localStorage.getItem(BATCH_KEY) || "null"); } catch (_) { /* ignore */ }
    if (!b || !b.id) {
      b = { id: 1, started: Date.now() };
      try { localStorage.setItem(BATCH_KEY, JSON.stringify(b)); } catch (_) { /* ignore */ }
    }
    return b;
  }
  function newBatch() {
    const b = { id: currentBatch().id + 1, started: Date.now() };
    try { localStorage.setItem(BATCH_KEY, JSON.stringify(b)); } catch (_) { /* ignore */ }
    return b;
  }

  function addScan(r) {
    const known = r.label in CLASS_INFO;
    const c = known ? (r.confidences.find((x) => x.label === r.label) || {}).confidence : null;
    const s = loadScans();
    const t = Date.now();
    s.push({ t, label: known ? r.label : "not a mango", conf: c ?? null, model: currentModel(), batch: currentBatch().id });
    saveScans(s);
    return t;
  }
  function addMultiScan(res) {                       // one history row per mango found in the photo
    const s = loadScans();
    const t = Date.now(), of = res.items.length;
    for (const it of res.items) {
      const known = it.label in CLASS_INFO;
      s.push({ t, label: known ? it.label : "not a mango", conf: known ? it.conf : null, model: currentModel(),
               batch: currentBatch().id, n: it.n, of });
    }
    saveScans(s);
    return t;
  }
  function addDiseaseScan(g, t = Date.now(), of = 1) {   // a separate disease scan (mango or leaf), one row per mango
    const s = loadScans();
    s.push({ t, type: "disease", label: g.disease, name: g.name, part: g.part || "mango",
             conf: g.conf, verdict: g.verdict, batch: currentBatch().id, n: g.n, of, model: g.model || "" });
    saveScans(s);
  }
  const mangoOf = (x) => (x.of > 1 ? "mango " + x.n + " of " + x.of : "");
  const isDisease = (x) => x.type === "disease";
  function ago(t) {
    const m = Math.round((Date.now() - t) / 60000);
    if (m < 1) return "just now";
    if (m < 60) return m + " min ago";
    const h = Math.round(m / 60);
    return h < 24 ? h + " h ago" : Math.round(h / 24) + " d ago";
  }

  const VERDICT_COLORS = { "Healthy": "#5BC27A", "Disease found": "#FF8A6B", "Check by hand": "#B9C6BC", "Not a mango or leaf": "#8A8F8B" };
  let donutMode = "ripeness";
  function verdictBadge(v) {
    const st = VERDICT_STYLE[v];
    return st ? `<b class="mini-badge ${st.cls}" title="${esc(v)}">${st.badge}</b>` : "";
  }
  const scanName = (x) => isDisease(x) ? (x.name || x.label) : cap(x.label);
  const scanColor = (x) => isDisease(x) ? (VERDICT_COLORS[x.verdict] || "") : (COLORS[x.label] || "");

  function renderBatch(s) {
    const b = currentBatch();
    const mine = s.filter((x) => x.batch === b.id);
    const dis = mine.filter(isDisease);
    $("batchName").textContent = "Batch " + b.id;
    $("batchCount").textContent = mine.length + (mine.length === 1 ? " scan" : " scans");
    const parts = [];
    if (mine.length - dis.length) parts.push((mine.length - dis.length) + " ripeness");
    if (dis.length) parts.push(dis.length + " disease (" + dis.filter((x) => x.verdict === "Disease found").length + " found)");
    $("batchGrades").textContent = "Started " + new Date(b.started).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" })
      + (parts.length ? " · " + parts.join(" · ") : "");
    $("reportBtn").disabled = !mine.length;
  }

  function csvCell(v) {
    const t = v == null ? "" : String(v);
    return /[",\n]/.test(t) ? '"' + t.replace(/"/g, '""') + '"' : t;
  }
  function downloadReport() {
    const b = currentBatch();
    const mine = loadScans().filter((x) => x.batch === b.id);
    if (!mine.length) return;
    const head = ["batch", "time", "scan", "mango", "result", "confidence", "verdict", "model", "user_check", "correct_label"];
    const rows = mine.map((x) => [b.id, new Date(x.t).toLocaleString(),
      isDisease(x) ? "disease (" + (x.part || "mango") + ")" : "ripeness", mangoOf(x), scanName(x),
      x.conf != null ? x.conf.toFixed(3) : "", isDisease(x) ? x.verdict || "" : "", x.model || "",
      x.fb || "", x.fb === "wrong" ? x.truth || "" : ""]);
    const csv = [head, ...rows].map((r) => r.map(csvCell).join(",")).join("\r\n");
    const url = URL.createObjectURL(new Blob(["﻿" + csv], { type: "text/csv;charset=utf-8" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = `mango-batch-${b.id}-${new Date().toISOString().slice(0, 10)}.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  }

  function renderOverview() {
    const s = loadScans();
    const last = s[s.length - 1];
    $("lastLabel").textContent = last ? scanName(last) + (last.conf != null ? " · " + Math.round(last.conf * 100) + "%" : "") : "No scans yet";
    $("lastLabel").style.color = last ? scanColor(last) : "";
    $("lastMeta").textContent = last
      ? (isDisease(last) ? "Disease scan (" + (last.part || "mango") + ") · " + (last.verdict || "") + " · " : "Ripeness scan · ")
        + (last.model ? last.model + " · " : "") + ago(last.t)
      : "Scan a mango to see it here.";
    renderBatch(s);

    // donut: ripeness scans by result, or disease scans by verdict
    const colors = donutMode === "disease" ? VERDICT_COLORS : COLORS;
    const inScope = scope === "batch" ? s.filter((x) => x.batch === currentBatch().id) : s;
    const pool = inScope.filter((x) => (donutMode === "disease") === isDisease(x));
    const keyOf = donutMode === "disease" ? (x) => x.verdict : (x) => x.label;
    const keys = Object.keys(colors);
    const counts = keys.map((k) => pool.filter((x) => keyOf(x) === k).length);
    const total = counts.reduce((a, b) => a + b, 0);
    const C = 2 * Math.PI * 60;
    let off = 0;
    let svg = '<g transform="rotate(-90 75 75)" fill="none" stroke-width="16"><circle cx="75" cy="75" r="60" style="stroke:var(--fill-3)"/>';
    keys.forEach((k, i) => {
      if (!counts[i]) return;
      const len = counts[i] / total * C;
      svg += `<circle cx="75" cy="75" r="60" stroke="${colors[k]}" stroke-dasharray="${len.toFixed(1)} ${(C - len).toFixed(1)}" stroke-dashoffset="${(-off).toFixed(1)}"/>`;
      off += len;
    });
    svg += `</g><text x="75" y="76" text-anchor="middle" style="fill:var(--text)" font-family="Space Grotesk, sans-serif" font-size="28" font-weight="700">${total}</text>`
         + `<text x="75" y="96" text-anchor="middle" style="fill:var(--muted)" font-family="Inter, sans-serif" font-size="11">${donutMode === "disease" ? "checked" : "mangoes"}</text>`;
    $("donut").innerHTML = svg;
    $("donutLegend").innerHTML = keys.map((k, i) => `<li><i style="background:${colors[k]}"></i>${cap(k)} <b>${counts[i]}</b></li>`).join("");
    // idea 5: one-line summary of the crate / batch, plus how many results the user confirmed
    const b = currentBatch();
    $("donutTitle").textContent = scope === "batch" ? "Batch " + b.id + " summary" : "All scans";
    const parts = keys.map((k, i) => counts[i] ? counts[i] + " " + k.toLowerCase() : "").filter(Boolean);
    const checked = pool.filter((x) => x.fb), right = checked.filter((x) => x.fb === "correct").length;
    $("donutSummary").textContent = total
      ? (donutMode === "disease" ? total + " checked: " : total + (total === 1 ? " mango: " : " mangoes: ")) + parts.join(" · ")
        + (checked.length ? ` · you confirmed ${right} of ${checked.length} as right` : "")
      : (scope === "batch" ? "No " + (donutMode === "disease" ? "disease" : "ripeness") + " scans in this batch yet." : "No scans yet.");

    // bars: scans per day, last 7 days
    const days = [];
    for (let d = 6; d >= 0; d--) {
      const day = new Date();
      day.setHours(0, 0, 0, 0);
      day.setDate(day.getDate() - d);
      days.push(day.getTime());
    }
    const per = days.map((d0) => s.filter((x) => x.t >= d0 && x.t < d0 + 864e5).length);
    const max = Math.max(1, ...per);
    $("weekBars").innerHTML = days.map((d0, i) =>
      `<div class="day"><span>${per[i] || ""}</span><div class="bar${i === 6 ? " today" : ""}" style="height:${Math.round(per[i] / max * 72)}%"></div>`
      + `<span>${new Date(d0).toLocaleDateString(undefined, { weekday: "short" })}</span></div>`).join("");

    // recent scans
    const rec = s.slice(-5).reverse();
    $("recentList").innerHTML = rec.length
      ? rec.map((x) => `<li><span class="rec-main">${isDisease(x) ? verdictBadge(x.verdict) : ""}<span style="color:${scanColor(x)}">${esc(scanName(x))}</span></span><span class="meta">`
          + `${mangoOf(x) ? esc(mangoOf(x)) + " · " : ""}${x.conf != null ? Math.round(x.conf * 100) + "% · " : ""}${x.model ? esc(x.model) + " · " : ""}${ago(x.t)}</span></li>`).join("")
      : '<li><span class="muted">No scans yet</span></li>';
  }

  function setDonutMode(m) {
    donutMode = m;
    $("segGrade").classList.toggle("on", m === "disease");
    $("segRipe").classList.toggle("on", m !== "disease");
    $("segGrade").setAttribute("aria-pressed", String(m === "disease"));
    $("segRipe").setAttribute("aria-pressed", String(m !== "disease"));
    renderOverview();
  }
  let scope = "batch";                               // idea 5: donut for this batch (crate) or all scans
  function setScope(v) {
    scope = v;
    $("scopeBatch").classList.toggle("on", v === "batch");
    $("scopeAll").classList.toggle("on", v !== "batch");
    $("scopeBatch").setAttribute("aria-pressed", String(v === "batch"));
    $("scopeAll").setAttribute("aria-pressed", String(v !== "batch"));
    renderOverview();
  }
  $("scopeBatch").addEventListener("click", () => setScope("batch"));
  $("scopeAll").addEventListener("click", () => setScope("all"));
  $("segGrade").addEventListener("click", () => setDonutMode("disease"));
  $("segRipe").addEventListener("click", () => setDonutMode("ripeness"));
  $("reportBtn").addEventListener("click", downloadReport);
  $("newBatchBtn").addEventListener("click", () => {
    const n = loadScans().filter((x) => x.batch === currentBatch().id).length;
    if (n && !confirm(`Start a new batch? Batch ${currentBatch().id} (${n} scans) stays in the history; download its report first if you need it.`)) return;
    const b = newBatch();
    $("batchMsg").textContent = "Batch " + b.id + " started. New scans are counted here.";
    $("batchMsg").hidden = false;
    renderOverview();
  });

  $("clearHistory").addEventListener("click", () => {
    if (confirm("Clear the scan history on this device?")) { saveScans([]); renderOverview(); }
  });

  // ---------- idea 2: "Was this right?" (saved with the scan, exported in the CSV) ----------
  const R_TRUTH = [["unripe", "Unripe"], ["ripe", "Ripe"], ["overripe", "Overripe"], ["not a mango", "Not a mango"]];
  let fbR = null, fbD = null;                        // {t, n, of, disease, names} of the result on screen
  const sameScan = (x, tg) => x.t === tg.t && (x.n || 1) === (tg.n || 1) && isDisease(x) === !!tg.disease;

  function saveFeedback(tg, fb, truth) {
    const s = loadScans();
    const x = s.find((y) => sameScan(y, tg));
    if (!x) return;
    x.fb = fb;
    if (fb === "wrong" && truth) x.truth = truth; else delete x.truth;
    saveScans(s);
  }

  function renderFeedback(box, tg) {
    if (!box) return;
    if (!tg) { box.hidden = true; box.innerHTML = ""; return; }
    const x = loadScans().find((y) => sameScan(y, tg));
    if (!x) { box.hidden = true; return; }
    const who = tg.of > 1 ? "mango " + tg.n : "this";
    const opts = tg.disease
      ? Object.entries(tg.names || {}).map(([k, v]) => [k, v]).concat([["not a mango or leaf", "Not a mango or leaf"]])
      : R_TRUTH;
    const btn = (attr, val, label, on) => `<button type="button" class="fb-btn${on ? " on" : ""}" ${attr}="${esc(val)}">${esc(label)}</button>`;
    let html;
    if (!x.fb) {
      html = `<span class="fb-q">Was ${esc(who)} right?</span>` + btn("data-fb", "correct", "👍 Yes") + btn("data-fb", "wrong", "👎 No");
    } else if (x.fb === "wrong" && !x.truth) {
      html = `<span class="fb-q">What is it really?</span><span class="fb-opts">`
        + opts.filter(([k]) => k !== (tg.disease ? x.label : x.label)).map(([k, v]) => btn("data-truth", k, v)).join("") + "</span>";
    } else {
      const truth = x.truth ? (opts.find(([k]) => k === x.truth) || [0, x.truth])[1] : "";
      html = `<span class="fb-q">${x.fb === "correct" ? "✓ Saved: right" : "Saved: really " + esc(truth)}</span>`
        + btn("data-fb", "undo", "Change");
    }
    box.innerHTML = html;
    box.hidden = false;
    box.querySelectorAll("[data-fb]").forEach((b) => b.addEventListener("click", () => {
      const v = b.dataset.fb;
      if (v === "undo") { const s = loadScans(); const y = s.find((z) => sameScan(z, tg)); if (y) { delete y.fb; delete y.truth; saveScans(s); } }
      else saveFeedback(tg, v);
      renderFeedback(box, tg);
    }));
    box.querySelectorAll("[data-truth]").forEach((b) => b.addEventListener("click", () => {
      saveFeedback(tg, "wrong", b.dataset.truth);
      renderFeedback(box, tg);
    }));
  }

  // ---------- idea 3: share or save the result as a picture ----------
  async function loadImg(src) {
    if (!src) return null;
    try {
      const blob = src.startsWith("blob:") || src.startsWith("data:") ? await (await fetch(src)).blob()
                                                                      : await (await timedFetch(src, {}, 20000)).blob();
      return await createImageBitmap(blob);
    } catch (_) { return null; }                       // e.g. the server link expired: fall back to our own photo
  }

  function wrapText(g, text, x, y, maxW, lineH, maxLines = 4) {
    const words = String(text || "").split(/\s+/);
    let line = "", n = 0;
    for (const w of words) {
      const test = line ? line + " " + w : w;
      if (g.measureText(test).width > maxW && line) {
        g.fillText(line, x, y); y += lineH; line = w;
        if (++n >= maxLines - 1) { line = words.slice(words.indexOf(w)).join(" "); break; }
      } else line = test;
    }
    if (line) { while (g.measureText(line).width > maxW && line.length > 3) line = line.slice(0, -4) + "…"; g.fillText(line, x, y); y += lineH; }
    return y;
  }

  async function shareResult(kind) {
    const W = 1080, H = 1350, c = document.createElement("canvas");
    c.width = W; c.height = H;
    const g = c.getContext("2d");
    g.fillStyle = "#FFF8E7"; g.fillRect(0, 0, W, H);
    g.fillStyle = "#F5A623"; g.fillRect(0, 0, W, 120);
    g.fillStyle = "#2A1A00"; g.font = "700 48px 'Space Grotesk', sans-serif";
    g.fillText("Mango Grader", 56, 78);
    g.font = "500 28px Inter, sans-serif";
    g.textAlign = "right";
    g.fillText(kind === "disease" ? "Disease scan" : "Ripeness scan", W - 56, 76);
    g.textAlign = "left";
    // the picture on screen (heat map or numbered boxes when shown), else our own photo
    const imgEl = kind === "disease" ? (!d.cam.hidden ? d.cam : d.photo) : el.preview;
    const bmp = (await loadImg(imgEl && imgEl.src)) || (await loadImg(kind === "disease" ? lastDPhoto : lastRPhoto));
    const box = { x: 56, y: 160, w: W - 112, h: 660 };
    g.fillStyle = "#EFE4C8"; g.fillRect(box.x, box.y, box.w, box.h);
    if (bmp) {
      const s = Math.min(box.w / bmp.width, box.h / bmp.height);
      const w = bmp.width * s, h = bmp.height * s;
      g.drawImage(bmp, box.x + (box.w - w) / 2, box.y + (box.h - h) / 2, w, h);
    }
    let y = box.y + box.h + 80;
    const big = kind === "disease" ? (d.summary.hidden ? d.label.textContent : d.summary.textContent) : el.verdictLabel.textContent;
    const conf = kind === "disease" ? (d.summary.hidden ? d.conf.textContent : "") : el.verdictConf.textContent;
    g.fillStyle = "#1F2A22"; g.font = "700 64px 'Space Grotesk', sans-serif";
    y = wrapText(g, cap(big || ""), 56, y, W - 112, 72, 2);
    if (conf) { g.fillStyle = "#5B6A5F"; g.font = "600 34px Inter, sans-serif"; g.fillText(conf, 56, y); y += 56; }
    g.font = "500 32px Inter, sans-serif"; g.fillStyle = "#3E4C42";
    const chips = [...(kind === "disease" ? d.items : el.mItems).querySelectorAll(".m-item")];
    const multi = !(kind === "disease" ? d.items : el.mItems).hidden && chips.length > 1;
    if (multi) {                                     // several mangoes: one line each
      g.font = "600 32px Inter, sans-serif";
      chips.slice(0, 6).forEach((b) => {
        const num = b.querySelector(".m-num");
        g.fillStyle = num.style.background && !num.style.background.startsWith("var") ? num.style.background : "#9A5800";
        g.beginPath(); g.arc(76, y - 11, 20, 0, 7); g.fill();
        g.fillStyle = "#FFFFFF"; g.textAlign = "center"; g.fillText(num.textContent, 76, y);
        g.textAlign = "left"; g.fillStyle = "#1F2A22";
        const badge = b.querySelector(".mini-badge");
        g.fillText([b.querySelector(".m-name").textContent, b.querySelector(".m-val").textContent,
                    badge ? "· " + badge.title : ""].filter(Boolean).join("  "), 112, y);
        y += 50;
      });
      if (chips.length > 6) { g.fillText("… and " + (chips.length - 6) + " more", 112, y); y += 50; }
    } else if (kind === "disease") {
      if (!d.verdictCard.hidden && d.title.textContent) {
        g.font = "700 36px Inter, sans-serif"; g.fillStyle = "#1F2A22";
        g.fillText("Verdict: " + d.title.textContent, 56, y); y += 50;
        g.font = "500 30px Inter, sans-serif"; g.fillStyle = "#3E4C42";
        y = wrapText(g, d.advice.textContent, 56, y, W - 112, 42, 3);
      }
    } else {
      y = wrapText(g, el.verdictText.textContent, 56, y, W - 112, 42, 3);
    }
    g.font = "500 24px Inter, sans-serif"; g.fillStyle = "#6B7A6F";
    const model = kind === "disease" ? (currentDModel() ? "Model: " + currentDModel() : "") : el.modelUsed.textContent;
    g.fillText([new Date().toLocaleString(), model].filter(Boolean).join(" · "), 56, H - 84);
    g.fillText("From our own student model · not an official grade", 56, H - 46);
    const blob = await new Promise((r) => c.toBlob(r, "image/png"));
    const now = new Date(), p2 = (v) => String(v).padStart(2, "0");
    const stamp = `${now.getFullYear()}-${p2(now.getMonth() + 1)}-${p2(now.getDate())}-${p2(now.getHours())}${p2(now.getMinutes())}`;
    const file = new File([blob], `mango-${kind}-${stamp}.png`, { type: "image/png" });
    try {
      if (navigator.canShare && navigator.canShare({ files: [file] })) {
        await navigator.share({ files: [file], title: "Mango Grader result" });
        return;
      }
    } catch (e) { if (e && e.name === "AbortError") return; }   // the user closed the share sheet
    const url = URL.createObjectURL(file);
    const a = document.createElement("a");
    a.href = url; a.download = file.name;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  }
  let lastRPhoto = "", lastDPhoto = "";
  $("rShare").addEventListener("click", () => shareResult("ripeness"));
  $("dShare").addEventListener("click", () => shareResult("disease"));

  // ---------- idea 1: photo tips are open until the first scan, then closed (tap to open) ----------
  if (!loadScans().length) { $("rTips").open = true; $("dTips").open = true; }

  // ---------- tabs: Overview · Ripeness · Disease · Insights (sidebar on laptop, bottom bar on phone) ----------
  const TABS = ["overview", "ripeness", "disease", "insights"];
  function showTab(name, updateHash = true) {
    if (!TABS.includes(name)) name = "overview";
    document.querySelectorAll(".view").forEach((v) => { v.hidden = v.dataset.view !== name; });
    document.querySelectorAll("[data-tab]").forEach((b) => {
      const on = b.dataset.tab === name;
      b.classList.toggle("active", on);
      if (on) b.setAttribute("aria-current", "page"); else b.removeAttribute("aria-current");
    });
    if (name !== "ripeness" && stream) {           // leaving the scan tab turns the camera off
      stopCamera();
      el.mainControls.hidden = false;
      el.cameraControls.hidden = true;
      if (el.preview.hidden) el.placeholder.hidden = false;
    }
    if (name !== "disease" && dStream) {             // leaving the disease tab turns its camera off too
      dStopCamera();
      if (d.photo.hidden && d.cam.hidden) d.placeholder.hidden = false;
    }
    if (name === "overview") renderOverview();
    if (updateHash && location.hash !== "#" + name) history.replaceState(null, "", "#" + name);
    window.scrollTo({ top: 0 });
  }
  document.querySelectorAll("[data-tab]").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
  document.querySelectorAll("[data-goto]").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.goto)));
  document.querySelectorAll("[data-scroll]").forEach((b) => b.addEventListener("click", () => {   // Overview step 4
    const t = $(b.dataset.scroll);
    if (t) t.scrollIntoView({ behavior: "smooth", block: "center" });
  }));
  window.addEventListener("hashchange", () => showTab(location.hash.slice(1), false));

  // Overview quick actions jump to the scan tab
  document.querySelectorAll("[data-quick]").forEach((b) => b.addEventListener("click", async () => {
    showTab("ripeness");
    if (b.dataset.quick === "upload") { el.fileInput.click(); return; }
    await openCamera();
    if (b.dataset.quick === "live" && stream) startLive();
  }));

  showTab(location.hash.slice(1) || "overview", false);
  setupModels(null);                                 // "On this phone" is available even without a server
  checkServer();
})();
