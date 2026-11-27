"use strict";

const byId = id => document.getElementById(id);
const state = {pages: [], index: -1, block: -1, dirty: false, busy: false};
const coords = ["x0", "y0", "x1", "y1"];
const endpoint = () => "/api/page/" + state.pages[state.index].id;
const selected = () => state.page.blocks[state.block];
const option = (value, text) => new Option(text, value);
const on = (id, name, handler) => byId(id).addEventListener(name, handler);

function message(text, error = false) {
  byId("message").textContent = text;
  byId("message").classList.toggle("error", error);
}

async function api(url, init) {
  const response = await fetch(url, init);
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || response.statusText);
  return body;
}

function refreshStatus() {
  const row = state.pages[state.index];
  const count = state.pages.filter(page => page.accepted).length;
  byId("progress").textContent = count + "/" + state.pages.length + " accepted";
  byId("pages").replaceChildren(...state.pages.map((page, index) =>
    option(index, "PDF " + page.number + (page.accepted ? " (accepted)" : ""))));
  byId("pages").value = state.index;
  byId("page-state").textContent = state.dirty ? "Unsaved changes" :
    (row.accepted ? "Accepted" : "Pending") + " / revision " + state.version;
  byId("save").disabled = !state.dirty;
  byId("accept").disabled = row.accepted && !state.dirty;
  byId("previous").disabled = state.index <= 0;
  byId("next").disabled = state.index === state.pages.length - 1;
  byId("pending").disabled = state.pages.every((page, index) => page.accepted || index === state.index);
}

function historyOptions() {
  byId("history").replaceChildren(...[...state.versions].reverse().map(version =>
    option(version, version === 0 ? "Pre-draft" : "Revision " + version)));
}

function setPage(page) {
  state.page = page;
  state.block = page.blocks.length ? 0 : -1;
  byId("folio").value = page.printed_page ?? "";
  showBlocks();
}

function changed() {
  state.dirty = true;
  byId("checked").checked = false;
  refreshStatus();
}

function blockLabel(block, index) {
  const flag = state.flags.has(block.block_id) ? " [review]" : "";
  return (index + 1) + ". " + block.kind + flag + " - " +
    (block.text || block.latex || block.block_id).replace(/\s+/g, " ").slice(0, 96);
}

function showBlocks() {
  byId("blocks").replaceChildren(...state.page.blocks.map((block, index) =>
    option(index, blockLabel(block, index))));
  selectBlock(state.block);
}

function selectBlock(index, reveal = false) {
  state.block = index;
  byId("blocks").selectedIndex = index;
  const block = selected();
  byId("editor").hidden = !block;
  for (const name of ["remove", "up", "down"]) byId(name).disabled = !block;
  if (block) showEditor(block);
  drawRegions();
  if (reveal) {
    const target = byId("regions").children[index];
    if (target && byId("boxes").checked) {
      target.scrollIntoView({block: "center", inline: "nearest", behavior: "smooth"});
    }
  }
}

function showEditor(block) {
  byId("block-id").textContent = block.block_id;
  byId("kind").value = block.kind;
  byId("order").value = state.block + 1;
  byId("order").max = state.page.blocks.length;
  byId("text").value = block.text ?? "";
  byId("formula").hidden = block.kind !== "formula";
  byId("latex").value = block.latex ?? "";
  byId("equation").value = block.equation_number ?? "";
  byId("table-field").hidden = block.kind !== "table";
  byId("table").value = state.tableEdits[block.block_id] ??
    JSON.stringify(block.table?.rows ?? [], null, 2);
  byId("caption-field").hidden = block.kind !== "figure";
  byId("caption").replaceChildren(option("", "No linked caption"), ...state.page.blocks
    .filter(item => item.kind === "caption")
    .map(item => option(item.block_id, item.block_id + " - " + item.text.slice(0, 100))));
  byId("caption").value = block.caption_id ?? "";
  for (const key of coords) byId(key).value = Number(block.region[key].toFixed(2));
  showNotes(block);
}

function showNotes(block) {
  const disagreement = state.flags.get(block.block_id);
  const original = state.original.get(block.block_id);
  const notes = byId("notes");
  notes.replaceChildren();
  byId("disagreements").hidden = !disagreement && !original;
  if (disagreement) {
    const list = document.createElement("ul");
    for (const flag of disagreement.flags) {
      const item = document.createElement("li");
      item.textContent = flag;
      list.append(item);
    }
    notes.append(list);
  }
  const readings = [...(original ? [["Pre-draft", original.text || original.latex || ""]] : [])];
  for (const item of disagreement?.alternate ?? []) {
    readings.push(["Alternate (" + item.kind + ")", item.text]);
  }
  for (const [title, text] of readings) {
    const label = document.createElement("strong");
    label.textContent = title;
    const body = document.createElement("pre");
    body.textContent = text || "(empty)";
    notes.append(label, body);
  }
}

function regionStyle(element, region) {
  const {width, height} = state.page;
  element.style.left = 100 * region.x0 / width + "%";
  element.style.top = 100 * region.y0 / height + "%";
  element.style.width = 100 * (region.x1 - region.x0) / width + "%";
  element.style.height = 100 * (region.y1 - region.y0) / height + "%";
}

function drawRegions() {
  byId("regions").replaceChildren(...state.page.blocks.map((block, index) => {
    const element = document.createElement("button");
    element.type = "button";
    element.className = "region";
    element.title = blockLabel(block, index);
    element.setAttribute("aria-label", element.title);
    element.tabIndex = index === state.block ? 0 : -1;
    element.classList.toggle("selected", index === state.block);
    element.classList.toggle("flagged", state.flags.has(block.block_id));
    regionStyle(element, block.region);
    element.addEventListener("click", () => selectBlock(index));
    return element;
  }));
}

function zoom() {
  const value = byId("zoom").value;
  byId("surface").style.width = value === "fit" ? "100%" :
    state.page.width * Number(value) / 100 + "px";
}

async function load(index) {
  const row = state.pages[index];
  const url = "/api/page/" + row.id;
  const result = await api(url);
  const predraft = result.version ? await api(url + "?version=0") : result.page;
  state.index = index;
  state.version = result.version;
  state.versions = result.versions;
  state.flags = new Map(result.flags.map(flag => [flag.block_id, flag]));
  state.original = new Map(predraft.blocks.map(block => [block.block_id, structuredClone(block)]));
  state.tableEdits = result.page.table_edits ?? {};
  delete result.page.table_edits;
  state.dirty = false;
  row.accepted = result.accepted;
  byId("checked").checked = false;
  byId("reviewer").value = localStorage.getItem("pilot-reviewer") || result.reviewer;
  byId("page-title").textContent = "PDF " + row.number + " / " + row.page_class.replaceAll("_", " ");
  byId("image").src = "/scan/" + row.id;
  byId("surface").classList.toggle("inverted", row.orientation === 180);
  byId("scan-scroll").scrollTo(0, 0);
  byId("workspace").querySelector(".editing").scrollTo(0, 0);
  setDrawing(false);
  setPage(result.page);
  zoom();
  historyOptions();
  refreshStatus();
  message("");
}

function toSave(accept) {
  const page = structuredClone(state.page);
  if (!accept && Object.keys(state.tableEdits).length) page.table_edits = state.tableEdits;
  if (accept) for (const block of page.blocks.filter(item => item.kind === "table")) {
    const edit = state.tableEdits[block.block_id];
    if (edit !== undefined) {
      try {
        block.table = {rows: JSON.parse(edit)};
      } catch {
        throw new Error("Invalid table rows JSON in block " + block.block_id + ".");
      }
    }
  }
  return page;
}

async function save(accept) {
  if (accept && !byId("checked").checked) throw new Error("Confirm the full-page check first.");
  const reviewer = byId("reviewer").value.trim();
  const page = toSave(accept);
  const result = await api(endpoint(), {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({page, reviewer, accept, base: state.version})
  });
  state.version = result.version;
  state.versions.push(result.version);
  state.pages[state.index].accepted = accept;
  if (accept) {
    state.page = page;
    state.tableEdits = {};
    if (selected()) showEditor(selected());
  }
  state.dirty = false;
  byId("checked").checked = false;
  localStorage.setItem("pilot-reviewer", reviewer);
  historyOptions();
  refreshStatus();
  message(accept ? "Page accepted." : "Draft revision saved.");
}

async function action(callback) {
  if (state.busy) return;
  state.busy = true;
  byId("workspace").inert = true;
  try {
    await callback();
  } catch (error) {
    message(error.message, true);
    if (state.index >= 0) refreshStatus();
  } finally {
    state.busy = false;
    byId("workspace").inert = false;
  }
}

function navigate(index) {
  if (index === state.index || index < 0 || index >= state.pages.length) return;
  return action(async () => {
    if (state.dirty) await save(false);
    await load(index);
  });
}

function updateBlock(field, value) {
  const block = selected();
  block[field] = value;
  byId("blocks").options[state.block].text = blockLabel(block, state.block);
  changed();
}

function moveBlock(index) {
  if (!Number.isInteger(index) || index < 0 || index >= state.page.blocks.length ||
      index === state.block) return;
  const [block] = state.page.blocks.splice(state.block, 1);
  state.page.blocks.splice(index, 0, block);
  state.block = index;
  renumber();
}

function renumber() {
  state.page.blocks.forEach((block, index) => { block.reading_order = index; });
  showBlocks();
  changed();
}

function setDrawing(drawing) {
  state.drawing = drawing;
  state.start = null;
  byId("surface").classList.toggle("drawing", drawing);
  byId("draw").setAttribute("aria-pressed", drawing);
  byId("draw").textContent = drawing ? "Cancel drawing" : "Draw region";
}

function scanPoint(event) {
  const surface = byId("surface");
  const box = surface.getBoundingClientRect();
  const invert = surface.classList.contains("inverted");
  const fraction = (point, start, size) => Math.min(1, Math.max(0, (point - start) / size));
  const x = fraction(event.clientX, box.left, box.width);
  const y = fraction(event.clientY, box.top, box.height);
  return [(invert ? 1 - x : x) * state.page.width, (invert ? 1 - y : y) * state.page.height];
}

function drawnRegion(event) {
  const [x, y] = scanPoint(event);
  const [left, top] = state.start;
  return {
    x0: Math.round(Math.min(x, left)), y0: Math.round(Math.min(y, top)),
    x1: Math.round(Math.max(x, left)), y1: Math.round(Math.max(y, top))
  };
}

on("surface", "pointerdown", event => {
  if (!state.drawing || !selected()) return;
  event.preventDefault();
  state.start = scanPoint(event);
  event.currentTarget.setPointerCapture(event.pointerId);
  state.preview = document.createElement("div");
  state.preview.className = "region preview";
  byId("regions").append(state.preview);
});
on("surface", "pointermove", event => {
  if (state.start) regionStyle(state.preview, drawnRegion(event));
});
on("surface", "pointerup", event => {
  if (!state.start) return;
  const region = drawnRegion(event);
  if (region.x1 > region.x0 && region.y1 > region.y0) {
    selected().region = region;
    showEditor(selected());
    changed();
  }
  setDrawing(false);
  drawRegions();
});
on("surface", "pointercancel", () => { setDrawing(false); drawRegions(); });
on("draw", "click", () => {
  setDrawing(!state.drawing);
  if (state.drawing) byId("regions").hidden = false;
  else drawRegions();
});
on("rotate", "click", () => byId("surface").classList.toggle("inverted"));
on("zoom", "change", zoom);
on("boxes", "change", event => { byId("regions").hidden = !event.target.checked; });
on("previous", "click", () => navigate(state.index - 1));
on("next", "click", () => navigate(state.index + 1));
on("pending", "click", () => {
  const after = state.pages.findIndex((page, index) => index > state.index && !page.accepted);
  navigate(after >= 0 ? after : state.pages.findIndex((page, index) =>
    index !== state.index && !page.accepted));
});
on("pages", "change", event => navigate(Number(event.target.value)));
on("blocks", "change", event => selectBlock(event.target.selectedIndex, true));
on("folio", "input", event => { state.page.printed_page = event.target.value.trim() || null; changed(); });
on("kind", "change", event => {
  updateBlock("kind", event.target.value);
  if (selected().kind !== "figure") selected().caption_id = null;
  if (selected().kind !== "table") delete state.tableEdits[selected().block_id];
  showEditor(selected());
  drawRegions();
});
on("order", "change", event => moveBlock(event.target.valueAsNumber - 1));
on("up", "click", () => moveBlock(state.block - 1));
on("down", "click", () => moveBlock(state.block + 1));
on("text", "input", event => updateBlock("text", event.target.value));
on("latex", "input", event => updateBlock("latex", event.target.value || null));
on("equation", "input", event => updateBlock("equation_number", event.target.value || null));
on("caption", "change", event => updateBlock("caption_id", event.target.value || null));
on("table", "input", event => {
  state.tableEdits[selected().block_id] = event.target.value;
  changed();
});
for (const key of coords) on(key, "input", event => {
  selected().region[key] = event.target.valueAsNumber || 0;
  drawRegions();
  changed();
});
on("add", "click", () => {
  state.block = state.page.blocks.length;
  state.page.blocks.push({
    block_id: "review-" + crypto.randomUUID(),
    kind: "body", text: "", region: {x0: 0, y0: 0, x1: 0, y1: 0}
  });
  renumber();
  setDrawing(true);
  byId("boxes").checked = true;
  byId("regions").hidden = false;
});
on("remove", "click", () => {
  const [block] = state.page.blocks.splice(state.block, 1);
  delete state.tableEdits[block.block_id];
  for (const item of state.page.blocks) {
    if (item.caption_id === block.block_id) item.caption_id = null;
  }
  state.block = Math.min(state.block, state.page.blocks.length - 1);
  setDrawing(false);
  renumber();
});
on("save", "click", () => action(() => save(false)));
on("accept", "click", () => action(() => save(true)));
on("restore", "click", () => action(async () => {
  const revision = byId("history").value;
  if (state.dirty) await save(false);
  const page = await api(endpoint() + "?version=" + revision);
  state.tableEdits = page.table_edits ?? {};
  delete page.table_edits;
  setPage(page);
  changed();
  message("Loaded revision " + revision + " as an unsaved draft.");
}));
window.addEventListener("beforeunload", event => {
  if (state.dirty) event.preventDefault();
});

void action(async () => {
  const result = await api("/api/pages");
  state.pages = result.pages;
  byId("kind").replaceChildren(...result.kinds.map(kind => option(kind, kind)));
  const pending = state.pages.findIndex(page => !page.accepted);
  await load(Math.max(0, pending));
});
