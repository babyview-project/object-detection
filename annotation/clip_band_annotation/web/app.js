/* Object check: two-step crop / scene annotation of YOLOE detections.
 *
 * Runs entirely in the browser. The RA picks their bundle folder (manifest.json,
 * crop/<id>.jpg, frame/<id>.jpg); images are shown from local files via blob URLs and
 * the page's Content-Security-Policy forbids any network request. Answers are kept in
 * localStorage (one key per bundle) after every click and exported as CSV.
 *
 * Per page (one category): step "crop" shows the crops alone; step "frame" shows the
 * same items reshuffled, each beside its full frame with the box drawn. The RA clicks
 * the items that are NOT the category. A step is locked once submitted.
 */
"use strict";

const APP_VERSION = "1.0.0";
const STUDY = "clip-band-context";
const BACKUP_EVERY = 10; // pages

const $ = (id) => document.getElementById(id);
const S = {
  manifest: null,
  files: new Map(), // "crop/<id>" | "frame/<id>" -> File
  saved: null,      // {rater, bundle_id, pos: {page, phase}, pages: {pi: {crop: {...}, frame: {...}}}}
  urls: [],
  storageOk: true,
  pendingImport: null,
};

/* ---------- storage ---------- */
const key = () => `objcheck:${S.manifest.bundle_id}`;
function load() {
  try {
    const raw = localStorage.getItem(key());
    return raw ? JSON.parse(raw) : null;
  } catch (e) { S.storageOk = false; return null; }
}
function save() {
  S.saved.updated = new Date().toISOString();
  try {
    localStorage.setItem(key(), JSON.stringify(S.saved));
    $("saved").textContent = `Saved ${new Date().toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"})}`;
  } catch (e) {
    S.storageOk = false;
    $("saved").textContent = "Not saved in this browser: export a backup now";
  }
}
const fresh = () => ({rater: S.manifest.rater, bundle_id: S.manifest.bundle_id, pos: {page: 0, phase: "crop"}, pages: {}});

/* ---------- helpers ---------- */
function hash32(str) { // FNV-1a
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 0x01000193); }
  return h >>> 0;
}
function shuffled(arr, seed) { // mulberry32, deterministic so a resumed page looks the same
  let a = seed >>> 0;
  const rnd = () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const out = arr.slice();
  for (let i = out.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [out[i], out[j]] = [out[j], out[i]]; }
  return out;
}
const pagesDone = () => S.manifest.pages.filter((_, i) => S.saved.pages[i]?.frame?.submitted).length;
function show(id) {
  for (const s of ["start", "confirm", "task", "done"]) $(s).hidden = s !== id;
  $("bar").hidden = id === "start";
}
function url(file) { const u = URL.createObjectURL(file); S.urls.push(u); return u; }
function freeUrls() { S.urls.forEach(URL.revokeObjectURL); S.urls = []; }
function esc(s) { return String(s).replace(/[&<>"]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c])); }

/* ---------- opening a bundle ---------- */
$("folder").addEventListener("change", async (ev) => {
  const msg = $("start-msg");
  msg.className = "msg";
  const list = [...ev.target.files];
  const mf = list.filter((f) => f.name === "manifest.json")
                 .sort((a, b) => a.webkitRelativePath.length - b.webkitRelativePath.length)[0];
  if (!mf) { msg.textContent = "No manifest.json in that folder. Choose the unzipped bundle_… folder."; return; }
  let m;
  try { m = JSON.parse(await mf.text()); } catch (e) { msg.textContent = "manifest.json could not be read."; return; }
  if (m.study !== STUDY || !Array.isArray(m.pages)) { msg.textContent = "This is not an object-check bundle."; return; }
  const root = mf.webkitRelativePath.slice(0, -"manifest.json".length);
  S.files.clear();
  for (const f of list) {
    const rel = f.webkitRelativePath.startsWith(root) ? f.webkitRelativePath.slice(root.length) : "";
    const mm = rel.match(/^(crop|frame)\/([^/]+)\.jpg$/);
    if (mm) S.files.set(`${mm[1]}/${mm[2]}`, f);
  }
  const missing = m.pages.flatMap((p) => p.items).filter((i) => !S.files.has(`crop/${i}`) || !S.files.has(`frame/${i}`));
  if (missing.length) { msg.textContent = `${missing.length} images are missing from this folder. Re-unzip the bundle and try again.`; return; }
  S.manifest = m;
  S.saved = load() || fresh();
  if (S.pendingImport) applyImport(S.pendingImport);
  const n = pagesDone();
  $("who").textContent = m.rater;
  $("confirm-text").textContent = n
    ? `Your bundle has ${m.n_items.toLocaleString()} objects on ${m.pages.length} pages. You have finished ${n} page${n === 1 ? "" : "s"}; you will continue where you left off.`
    : `Your bundle has ${m.n_items.toLocaleString()} objects on ${m.pages.length} pages (about 20 to 24 objects each, two steps per page).`;
  $("go").textContent = n ? "Continue" : "Start";
  if (!S.storageOk) $("confirm-text").textContent += " This browser is not saving progress (private window?): export a backup often.";
  show("confirm");
});

$("import").addEventListener("change", async (ev) => {
  const f = ev.target.files[0];
  if (!f) return;
  const rows = parseCsv(await f.text());
  const msg = $("start-msg");
  if (!rows.length || !("bundle_id" in rows[0])) { msg.textContent = "That file is not an object-check backup."; return; }
  S.pendingImport = rows;
  msg.className = "msg ok";
  msg.textContent = `Backup loaded (${rows.length} rows). Now choose your bundle folder.`;
});

function applyImport(rows) {
  const mine = rows.filter((r) => r.bundle_id === S.manifest.bundle_id);
  if (!mine.length) { $("start-msg").textContent = "That backup belongs to a different bundle; it was not imported."; return; }
  const pages = {};
  for (const r of mine) {
    const pg = (pages[r.page_index] ??= {});
    const st = (pg[r.phase] ??= {marked: [], started: r.step_started || null, submitted: r.step_submitted || null});
    if (r.marked_not_object === "1") st.marked.push(r.item_id);
  }
  S.saved.pages = pages; // the backup replaces this browser's copy for this bundle
  S.saved.pos = firstOpen();
  S.pendingImport = null;
  save();
}
function firstOpen() {
  for (let i = 0; i < S.manifest.pages.length; i++) {
    const p = S.saved.pages[i];
    if (!p?.crop?.submitted) return {page: i, phase: "crop"};
    if (!p?.frame?.submitted) return {page: i, phase: "frame"};
  }
  return {page: S.manifest.pages.length, phase: "crop"};
}

$("go").addEventListener("click", () => { S.saved.pos = firstOpen(); render(); });

/* ---------- task ---------- */
function step(pi, phase) {
  const pg = (S.saved.pages[pi] ??= {});
  return (pg[phase] ??= {marked: [], started: null, submitted: null});
}

function render() {
  freeUrls();
  const {page: pi, phase} = S.saved.pos;
  const total = S.manifest.pages.length;
  $("progress-fill").style.width = `${(100 * pagesDone()) / total}%`;
  if (pi >= total) { $("where").textContent = `${total} / ${total} pages`; show("done"); return; }
  const page = S.manifest.pages[pi];
  const st = step(pi, phase);
  if (!st.started) { st.started = new Date().toISOString(); save(); }
  $("where").textContent = `Page ${pi + 1} / ${total}`;
  const cat = esc(page.category);
  $("prompt").innerHTML = phase === "crop"
    ? `<span class="step">Step 1 of 2 · crops only</span>Click every crop that is <u>not</u> a <b>${cat}</b>.`
    : `<span class="step">Step 2 of 2 · in the scene (pink box)</span>Judge again: click every boxed object that is <u>not</u> a <b>${cat}</b>.`;
  const items = phase === "crop" ? page.items : shuffled(page.items, hash32(`${S.manifest.bundle_id}:${pi}`));
  const grid = $("grid");
  grid.className = phase === "frame" ? "scene" : "";
  grid.replaceChildren(...items.map((id) => card(id, phase, st)));
  const last = pi === total - 1;
  $("next").textContent = phase === "crop" ? "Done: see them in the scene →" : last ? "Finish" : "Next page →";
  updateCount(st);
  window.scrollTo(0, 0);
  show("task");
  if (pi > 0 && phase === "crop" && pi % BACKUP_EVERY === 0) {
    $("saved").textContent = `${pi} pages done: please Export backup`;
  }
}

function card(id, phase, st) {
  const c = document.createElement("div");
  c.className = "card";
  c.tabIndex = 0;
  c.setAttribute("role", "button");
  c.dataset.id = id;
  const crop = `<div class="crop"><img alt="detected object" src="${url(S.files.get(`crop/${id}`))}"></div>`;
  const frame = phase === "frame"
    ? `<div class="frame"><img alt="video frame with the object outlined" src="${url(S.files.get(`frame/${id}`))}"><button class="zoom-btn" type="button" aria-label="Enlarge frame">Enlarge</button></div>`
    : "";
  c.innerHTML = `<span class="badge">not it</span>${crop}${frame}`;
  const marked = new Set(st.marked);
  setMark(c, marked.has(id));
  const toggle = () => {
    const on = !c.classList.contains("marked");
    setMark(c, on);
    st.marked = on ? [...new Set([...st.marked, id])] : st.marked.filter((x) => x !== id);
    save();
    updateCount(st);
  };
  c.addEventListener("click", (e) => {
    if (e.target.closest(".zoom-btn")) { openZoom(S.files.get(`frame/${id}`)); return; }
    toggle();
  });
  c.addEventListener("keydown", (e) => { if (e.key === " " || e.key === "Enter") { e.preventDefault(); toggle(); } });
  return c;
}
function setMark(c, on) { c.classList.toggle("marked", on); c.setAttribute("aria-pressed", on ? "true" : "false"); }
function updateCount(st) {
  const n = st.marked.length;
  $("count").textContent = n ? `${n} marked as not it` : "None marked yet (fine if all are correct)";
}

$("next").addEventListener("click", () => {
  const pos = S.saved.pos;
  step(pos.page, pos.phase).submitted = new Date().toISOString();
  S.saved.pos = pos.phase === "crop" ? {page: pos.page, phase: "frame"} : {page: pos.page + 1, phase: "crop"};
  save();
  render();
});

/* ---------- zoom ---------- */
function openZoom(file) { $("zoom-img").src = url(file); $("zoom").hidden = false; }
function closeZoom() { $("zoom").hidden = true; $("zoom-img").removeAttribute("src"); }
$("zoom").addEventListener("click", closeZoom);
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !$("zoom").hidden) closeZoom(); });

/* ---------- export ---------- */
const COLS = ["study", "app_version", "rater", "bundle_id", "page_index", "category", "phase", "item_id",
              "marked_not_object", "step_started", "step_submitted", "step_seconds", "exported_at"];
function exportCsv() {
  if (!S.manifest) return;
  const now = new Date().toISOString();
  const lines = [COLS.join(",")];
  S.manifest.pages.forEach((p, pi) => {
    for (const phase of ["crop", "frame"]) {
      const st = S.saved.pages[pi]?.[phase];
      if (!st?.started) continue;
      const secs = st.submitted ? ((new Date(st.submitted) - new Date(st.started)) / 1000).toFixed(1) : "";
      const marked = new Set(st.marked);
      for (const id of p.items) {
        lines.push([STUDY, APP_VERSION, S.manifest.rater, S.manifest.bundle_id, pi, p.category, phase, id,
                    marked.has(id) ? 1 : 0, st.started || "", st.submitted || "", secs, now].join(","));
      }
    }
  });
  const blob = new Blob([lines.join("\n") + "\n"], {type: "text/csv"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `objcheck_${S.manifest.rater}_${S.manifest.bundle_id}_${now.slice(0, 16).replace(/[:T]/g, "-")}.csv`;
  document.body.append(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
  $("saved").textContent = "Backup exported: upload it to the lab drive";
}
$("export-top").addEventListener("click", exportCsv);
$("export-end").addEventListener("click", exportCsv);

function parseCsv(text) { // fields here never contain commas or quotes
  const [head, ...rows] = text.trim().split(/\r?\n/);
  const cols = head.split(",");
  return rows.map((r) => Object.fromEntries(r.split(",").map((v, i) => [cols[i], v])));
}
