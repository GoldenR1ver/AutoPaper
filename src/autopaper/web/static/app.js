const state = {
  data: null,
  mode: "papers",
  minWeight: 40,
  authors: new Set(),
  topics: new Set(),
  query: "",
  focus: null,
  positions: new Map(),
  camera: { x: 0, y: 0, k: 1 },
  drag: null,
  noteScope: null,
  noteMarkdown: "",
  noteUpdatedAt: "",
  noteUpdate: null,
  jumpHighlight: "",
  pendingSelection: null,
  updating: false,
  biblioNotice: "",
};

const canvas = document.querySelector("#graph");
const ctx = canvas.getContext("2d");
const app = document.querySelector("#app");

document.querySelector("#mode-papers").onclick = () => setMode("papers");
document.querySelector("#mode-network").onclick = () => setMode("network");
document.querySelector("#min-weight").oninput = (event) => {
  state.minWeight = Number(event.target.value);
  document.querySelector("#min-weight-label").textContent = String(state.minWeight);
  renderList();
  draw();
};
document.querySelector("#query").oninput = (event) => {
  state.query = event.target.value.trim();
  renderList();
  draw();
};
document.querySelector("#clear-authors").onclick = () => { state.authors.clear(); renderFilters(); renderList(); draw(); };
document.querySelector("#clear-topics").onclick = () => { state.topics.clear(); renderFilters(); renderList(); draw(); };
document.querySelector("#reset-layout").onclick = () => { state.positions.clear(); layout(true); draw(); };
document.querySelector("#reader-close").onclick = closeReader;
document.querySelector("#open-file-toggle").onclick = toggleOpenFileMenu;
document.querySelector("#open-file-web").onclick = closeOpenFileMenu;
document.querySelector("#open-file-local").onclick = openFileLocally;
document.querySelector("#open-file-annotate").onclick = annotateOriginal;
document.addEventListener("click", (event) => {
  if (!event.target.closest("#open-file")) closeOpenFileMenu();
});
document.querySelector("#library-open").onclick = openLibrary;
document.querySelector("#fill-biblio").onclick = fillAllBibliography;
document.querySelector("#dedupe-open").onclick = openDuplicates;
document.querySelector("#dedupe-close").onclick = () => document.querySelector("#dedupe-dialog").close();
document.querySelector("#dedupe-merge").onclick = mergeDuplicates;
document.querySelector("#upload-open").onclick = openUpload;
document.querySelector("#upload-close").onclick = () => document.querySelector("#upload-dialog").close();
document.querySelector("#files").onchange = suggestUpload;
document.querySelector("#folder").onchange = suggestUpload;
document.querySelector("#classify-topics").onclick = classifyTopics;
document.querySelector("#upload-form").onsubmit = submitUpload;

canvas.addEventListener("pointerdown", onPointerDown);
canvas.addEventListener("pointermove", onPointerMove);
canvas.addEventListener("pointerup", onPointerUp);
canvas.addEventListener("wheel", onWheel, { passive: false });
window.addEventListener("resize", () => { resize(); draw(); });

load();

async function load() {
  const response = await fetch("/api/graph");
  state.data = await response.json();
  if (state.focus && !state.data.papers.some((paper) => paper.key === state.focus.id && state.focus.kind === "paper")) {
    if (state.focus.kind === "paper") closeReader();
  }
  renderFilters();
  renderUploadChoices();
  layout(false);
  renderList();
  resize();
  draw();
}

function setMode(mode) {
  state.mode = mode;
  document.querySelector("#mode-papers").classList.toggle("active", mode === "papers");
  document.querySelector("#mode-network").classList.toggle("active", mode === "network");
  layout(false);
  draw();
}

function visiblePapers() {
  const query = state.query.toLowerCase();
  return state.data.papers.filter((paper) => {
    if (state.authors.size && !paper.authors.some((author) => state.authors.has(author.id))) return false;
    if (state.topics.size && !paper.topics.some((topic) => state.topics.has(topic.id))) return false;
    if (!query) return true;
    return `${paper.title} ${paper.one_sentence}`.toLowerCase().includes(query);
  });
}

function edgeWeight(left, right) {
  const edge = state.data.edges.find((item) =>
    (item.source === left && item.target === right) || (item.source === right && item.target === left)
  );
  return edge ? edge.weight : 0;
}

function visibleEdges(papers) {
  const keys = new Set(papers.map((paper) => paper.key));
  return state.data.edges.filter((edge) => {
    if (!keys.has(edge.source) || !keys.has(edge.target)) return false;
    if (edge.weight >= state.minWeight) return true;
    return edge.reason === "同一作者" || edge.reason === "同一主题";
  });
}

function focusWeight(paper) {
  if (!state.focus || state.focus.kind !== "paper" || state.focus.id === paper.key) return paper.score || 0;
  return edgeWeight(state.focus.id, paper.key);
}

function renderFilters() {
  fillChips(document.querySelector("#author-filters"), state.data.authors, state.authors, (author) => author.label);
  fillChips(document.querySelector("#topic-filters"), state.data.topics, state.topics, (topic) => topic.name, {
    label: "关系",
    onClick: (topic) => inspect({ kind: "topic", id: topic.id }, false),
  });
  const synthesis = currentSynthesis();
  const title = document.querySelector("#list-title");
  title.textContent = synthesis ? "这组论文" : "论文";
  renderProfileActions();
}

function renderProfileActions() {
  const box = document.querySelector("#profile-actions");
  box.innerHTML = "";
  if (state.authors.size === 1) {
    const author = state.data.authors.find((item) => item.id === [...state.authors][0]);
    if (author) appendProfileControls(box, "作者", "author", author.id, author.has_profile, author.profile_updated_at);
  }
  if (state.topics.size === 1) {
    const topic = state.data.topics.find((item) => item.id === [...state.topics][0]);
    if (topic) appendProfileControls(box, "主题", "topic", topic.id, topic.has_profile, topic.profile_updated_at);
  }
}

function appendProfileControls(box, label, kind, id, exists, updatedAt) {
  if (exists) {
    box.appendChild(profileButton(`查看${label}笔记`, kind, id, false));
    box.appendChild(profileButton(`更新${label}笔记`, kind, id, true));
    if (updatedAt) box.appendChild(dateLine(updatedAt));
    return;
  }
  box.appendChild(profileButton(`生成${label}笔记`, kind, id, true));
}

function dateLine(iso) {
  const date = document.createElement("span");
  date.className = "meta-line";
  date.textContent = `笔记更新于 ${formatUpdated(iso)}`;
  return date;
}

function formatUpdated(iso) {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return String(iso || "");
  const pad = (value) => String(value).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

function setReaderChrome(updatedAt, onUpdate) {
  const slot = document.querySelector("#note-updated");
  const button = document.querySelector("#update-note");
  state.noteUpdatedAt = updatedAt || "";
  state.noteUpdate = onUpdate || null;
  slot.textContent = updatedAt ? `笔记更新于 ${formatUpdated(updatedAt)}` : (onUpdate ? "还没有笔记" : "");
  button.hidden = !onUpdate;
  button.disabled = false;
  button.textContent = updatedAt ? "更新笔记" : "生成笔记";
  button.onclick = onUpdate || null;
}

function profileButton(label, kind, id, generate) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "chip";
  button.textContent = label;
  button.onclick = () => openProfile(kind, id, generate);
  return button;
}

async function openProfile(kind, id, generate) {
  clearDanger();
  app.classList.add("reading");
  document.querySelector("#reader").hidden = false;
  hideOpenFile();
  const body = document.querySelector("#reader-body");
  setReaderChrome("", null);
  body.innerHTML = `<p class="meta-line">${generate ? "正在更新笔记，请稍等…" : "正在打开笔记…"}</p>`;
  const response = await fetch(generate ? "/api/profile" : `/api/profile?kind=${encodeURIComponent(kind)}&id=${encodeURIComponent(id)}`, generate ? {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ kind, id }),
  } : undefined);
  const payload = await response.json();
  if (!response.ok) {
    body.innerHTML = `<p class="message error"></p>`;
    body.querySelector("p").textContent = payload.error || "没有生成笔记";
    return;
  }
  const record = (kind === "author" ? state.data.authors : state.data.topics).find((item) => item.id === id);
  const label = record ? (kind === "author" ? record.label : record.name) : id;
  state.noteMarkdown = payload.markdown || "";
  body.innerHTML = `<article class="note">${renderMarkdown(state.noteMarkdown)}</article>`;
  await mountNote(body.querySelector("article.note"), { scope: kind, id, label }, payload.updated_at, () => openProfile(kind, id, true));
  if (record) {
    record.has_profile = true;
    if (payload.updated_at) record.profile_updated_at = payload.updated_at;
  }
  armDanger(
    kind === "author" ? () => deleteAuthor(id, label) : () => deleteTopic(id, label),
    () => deleteNote({ scope: kind, id }),
    kind === "author" ? "删除作者" : "删除主题",
  );
  renderProfileActions();
}

function fillChips(container, items, selected, label, extra) {
  container.innerHTML = "";
  if (!items.length) {
    container.textContent = "还没有";
    return;
  }
  for (const item of items) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `chip${selected.has(item.id) ? " on" : ""}`;
    button.textContent = label(item);
    button.onclick = () => {
      if (selected.has(item.id)) selected.delete(item.id);
      else selected.add(item.id);
      renderFilters();
      renderList();
      draw();
    };
    if (!extra) {
      container.appendChild(button);
      continue;
    }
    const wrap = document.createElement("span");
    wrap.className = "chip-wrap";
    const more = document.createElement("button");
    more.type = "button";
    more.className = "text";
    more.textContent = extra.label;
    more.onclick = () => extra.onClick(item);
    wrap.append(button, more);
    container.appendChild(wrap);
  }
}

function renderList() {
  const papers = visiblePapers().slice().sort((a, b) => focusWeight(b) - focusWeight(a) || a.title.localeCompare(b.title, "zh"));
  const list = document.querySelector("#paper-list");
  list.innerHTML = "";
  document.querySelector("#graph-empty").hidden = state.data.papers.length > 0;
  document.querySelector("#graph-status").textContent = `${papers.length} 篇可见 · 相关度 ${state.minWeight} 以上`;
  for (const paper of papers) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `paper-row${state.focus && state.focus.kind === "paper" && state.focus.id === paper.key ? " active" : ""}`;
    const bits = [];
    if (state.focus && state.focus.kind === "paper" && state.focus.id !== paper.key) bits.push(`相关 ${focusWeight(paper)}`);
    else bits.push(paper.has_note ? "已有笔记" : "尚未生成笔记");
    if (paper.note_updated_at) bits.push(`更新于 ${formatUpdated(paper.note_updated_at)}`);
    button.innerHTML = `<strong></strong><span></span>`;
    button.querySelector("strong").textContent = paper.title;
    button.querySelector("span").textContent = bits.join(" · ");
    button.onclick = () => inspect({ kind: "paper", id: paper.key }, true);
    list.appendChild(button);
  }
}

function currentSynthesis() {
  if (state.authors.size !== 1 || state.topics.size !== 1) return null;
  const author = [...state.authors][0];
  const topic = [...state.topics][0];
  return state.data.syntheses.find((item) => item.author_id === author && item.topic_id === topic) || null;
}

function nodes() {
  const papers = visiblePapers();
  const items = papers.map((paper) => ({ kind: "paper", id: paper.key, label: paper.title, paper }));
  if (state.mode === "network") {
    const authorIds = new Set(papers.flatMap((paper) => paper.authors.map((author) => author.id)));
    const topicIds = new Set(papers.flatMap((paper) => paper.topics.map((topic) => topic.id)));
    for (const author of state.data.authors) {
      if (authorIds.has(author.id) || state.authors.has(author.id)) items.push({ kind: "author", id: author.id, label: author.label, author });
    }
    for (const topic of state.data.topics) {
      if (topicIds.has(topic.id) || state.topics.has(topic.id)) items.push({ kind: "topic", id: topic.id, label: topic.name, topic });
    }
  }
  return items;
}

function nodeKey(node) { return `${node.kind}:${node.id}`; }

function layout(reset) {
  const items = nodes();
  if (reset) state.positions.clear();
  const count = Math.max(items.length, 1);
  items.forEach((node, index) => {
    if (!reset && state.positions.has(nodeKey(node))) return;
    const angle = (Math.PI * 2 * index) / count;
    const radius = node.kind === "paper" ? 220 : 340;
    state.positions.set(nodeKey(node), { x: Math.cos(angle) * radius, y: Math.sin(angle) * radius });
  });
  const edges = simulationEdges(items);
  const velocity = new Map(items.map((node) => [nodeKey(node), { x: 0, y: 0 }]));
  for (let step = 0; step < 220; step += 1) {
    for (let i = 0; i < items.length; i += 1) {
      for (let j = i + 1; j < items.length; j += 1) {
        const a = state.positions.get(nodeKey(items[i]));
        const b = state.positions.get(nodeKey(items[j]));
        const va = velocity.get(nodeKey(items[i]));
        const vb = velocity.get(nodeKey(items[j]));
        let dx = a.x - b.x;
        let dy = a.y - b.y;
        let dist = Math.hypot(dx, dy) || 0.1;
        const force = Math.min(40, 1400 / dist);
        dx = (dx / dist) * force;
        dy = (dy / dist) * force;
        va.x += dx; va.y += dy; vb.x -= dx; vb.y -= dy;
      }
    }
    for (const edge of edges) {
      const a = state.positions.get(edge.source);
      const b = state.positions.get(edge.target);
      const va = velocity.get(edge.source);
      const vb = velocity.get(edge.target);
      if (!a || !b || !va || !vb) continue;
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const dist = Math.hypot(dx, dy) || 0.1;
      const pull = (dist - edge.length) * 0.02;
      const ux = (dx / dist) * pull;
      const uy = (dy / dist) * pull;
      va.x += ux; va.y += uy; vb.x -= ux; vb.y -= uy;
    }
    for (const node of items) {
      const key = nodeKey(node);
      const point = state.positions.get(key);
      const vel = velocity.get(key);
      vel.x = (vel.x - point.x * 0.01) * 0.75;
      vel.y = (vel.y - point.y * 0.01) * 0.75;
      point.x += vel.x;
      point.y += vel.y;
    }
  }
}

function simulationEdges(items) {
  const keys = new Set(items.map(nodeKey));
  const edges = [];
  for (const edge of visibleEdges(visiblePapers())) {
    const source = `paper:${edge.source}`;
    const target = `paper:${edge.target}`;
    if (keys.has(source) && keys.has(target)) {
      edges.push({ source, target, length: 240 - edge.weight * 1.4 });
    }
  }
  if (state.mode === "network") {
    for (const paper of visiblePapers()) {
      for (const author of paper.authors) {
        const target = `author:${author.id}`;
        if (keys.has(target)) edges.push({ source: `paper:${paper.key}`, target, length: 160 });
      }
      for (const topic of paper.topics) {
        const target = `topic:${topic.id}`;
        if (keys.has(target)) edges.push({ source: `paper:${paper.key}`, target, length: 190 });
      }
    }
  }
  return edges;
}

function resize() {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * ratio));
  canvas.height = Math.max(1, Math.floor(rect.height * ratio));
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
}

function worldToScreen(x, y) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: (x - state.camera.x) * state.camera.k + rect.width / 2,
    y: (y - state.camera.y) * state.camera.k + rect.height / 2,
  };
}

function screenToWorld(x, y) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: (x - rect.width / 2) / state.camera.k + state.camera.x,
    y: (y - rect.height / 2) / state.camera.k + state.camera.y,
  };
}

function draw() {
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  ctx.fillStyle = "#f3efe6";
  ctx.fillRect(0, 0, rect.width, rect.height);
  const items = nodes();
  const edges = simulationEdges(items);
  ctx.lineCap = "round";
  ctx.font = "12px Segoe UI, Microsoft YaHei, sans-serif";
  for (const edge of edges) {
    const a = state.positions.get(edge.source);
    const b = state.positions.get(edge.target);
    if (!a || !b) continue;
    const from = worldToScreen(a.x, a.y);
    const to = worldToScreen(b.x, b.y);
    const paperEdge = edge.source.startsWith("paper:") && edge.target.startsWith("paper:");
    const weight = paperEdge ? edgeWeight(edge.source.slice(6), edge.target.slice(6)) : 30;
    const strong = !paperEdge || weight >= state.minWeight;
    ctx.strokeStyle = paperEdge
      ? `rgba(31, 107, 74, ${strong ? 0.25 + weight / 140 : 0.18})`
      : "rgba(90, 78, 64, 0.28)";
    ctx.lineWidth = paperEdge ? (strong ? 1.4 + weight / 22 : 0.7) : 1;
    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);
    ctx.stroke();
  }
  for (const node of items) {
    const point = state.positions.get(nodeKey(node));
    if (!point) continue;
    const screen = worldToScreen(point.x, point.y);
    const active = state.focus && state.focus.kind === node.kind && state.focus.id === node.id;
    const radius = nodeRadius(node);
    ctx.beginPath();
    if (node.kind === "topic") {
      ctx.moveTo(screen.x, screen.y - radius);
      ctx.lineTo(screen.x + radius, screen.y);
      ctx.lineTo(screen.x, screen.y + radius);
      ctx.lineTo(screen.x - radius, screen.y);
      ctx.closePath();
    } else if (node.kind === "author") {
      roundRect(screen.x - radius * 1.15, screen.y - radius * 0.72, radius * 2.3, radius * 1.44, 10);
    } else {
      ctx.arc(screen.x, screen.y, radius, 0, Math.PI * 2);
    }
    ctx.fillStyle = node.kind === "author" ? "#9a4e32" : node.kind === "topic" ? "#2a4b7c" : "#1f6b4a";
    ctx.globalAlpha = faded(node) ? 0.35 : 1;
    ctx.fill();
    if (node.kind === "paper" && node.paper.has_note) {
      ctx.lineWidth = 3;
      ctx.strokeStyle = "#d7b15a";
      ctx.stroke();
    }
    if (active) {
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#1c1915";
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    ctx.fillStyle = "#1c1915";
    ctx.textAlign = "center";
    ctx.fillText(shortLabel(node.label), screen.x, screen.y + radius + 14);
  }
}

function nodeRadius(node) {
  if (node.kind !== "paper") return 16;
  const base = 15 + Math.min(node.paper.score, 100) / 12;
  if (!state.focus || state.focus.kind !== "paper") return base;
  if (state.focus.id === node.paper.key) return 24;
  return 12 + focusWeight(node.paper) / 8;
}

function faded(node) {
  if (!state.focus || state.focus.kind !== "paper" || node.kind !== "paper") return false;
  if (state.focus.id === node.id) return false;
  return focusWeight(node.paper) < state.minWeight;
}

function shortLabel(label) {
  const chars = [...label];
  return chars.length > 18 ? `${chars.slice(0, 17).join("")}…` : label;
}

function roundRect(x, y, width, height, radius) {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.arcTo(x + width, y, x + width, y + height, radius);
  ctx.arcTo(x + width, y + height, x, y + height, radius);
  ctx.arcTo(x, y + height, x, y, radius);
  ctx.arcTo(x, y, x + width, y, radius);
  ctx.closePath();
}

function hitNode(x, y) {
  let found = null;
  let best = 36;
  for (const node of nodes()) {
    const point = state.positions.get(nodeKey(node));
    if (!point) continue;
    const screen = worldToScreen(point.x, point.y);
    const distance = Math.hypot(screen.x - x, screen.y - y);
    const limit = (node.kind === "paper" ? nodeRadius(node) : 20) + 8;
    if (distance <= limit && distance < best) {
      best = distance;
      found = node;
    }
  }
  return found;
}

function onPointerDown(event) {
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  const node = hitNode(x, y);
  canvas.setPointerCapture(event.pointerId);
  canvas.classList.add("dragging");
  if (node) {
    state.drag = { kind: "node", node, sx: x, sy: y };
  } else {
    state.drag = { kind: "pan", sx: x, sy: y, cx: state.camera.x, cy: state.camera.y, node: null };
  }
}

function onPointerMove(event) {
  if (!state.drag) return;
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  if (state.drag.kind === "pan") {
    state.camera.x = state.drag.cx - (x - state.drag.sx) / state.camera.k;
    state.camera.y = state.drag.cy - (y - state.drag.sy) / state.camera.k;
  } else {
    const world = screenToWorld(x, y);
    const point = state.positions.get(nodeKey(state.drag.node));
    point.x = world.x;
    point.y = world.y;
  }
  draw();
}

function onPointerUp(event) {
  if (!state.drag) return;
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  const moved = Math.hypot(x - state.drag.sx, y - state.drag.sy) > 4;
  const node = state.drag.node;
  state.drag = null;
  canvas.classList.remove("dragging");
  if (moved) return;
  if (node) inspect(node, true);
  else if (state.focus) closeReader();
}

function onWheel(event) {
  event.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const before = screenToWorld(event.clientX - rect.left, event.clientY - rect.top);
  state.camera.k = Math.min(2.4, Math.max(0.45, state.camera.k * (event.deltaY > 0 ? 0.9 : 1.1)));
  const after = screenToWorld(event.clientX - rect.left, event.clientY - rect.top);
  state.camera.x += before.x - after.x;
  state.camera.y += before.y - after.y;
  draw();
}

async function inspect(node, center) {
  clearDanger();
  state.focus = { kind: node.kind, id: node.id };
  if (center) {
    const point = state.positions.get(nodeKey(node));
    if (point) { state.camera.x = point.x; state.camera.y = point.y; }
  }
  app.classList.add("reading");
  document.querySelector("#reader").hidden = false;
  renderList();
  resize();
  draw();
  const body = document.querySelector("#reader-body");
  const fileLink = document.querySelector("#open-file");
  hideOpenFile();
  if (node.kind === "paper") await showPaper(node.id, body, fileLink);
  else if (node.kind === "author") showAuthor(node.id, body);
  else showTopic(node.id, body);
}

function hideOpenFile() {
  const fileLink = document.querySelector("#open-file");
  const status = document.querySelector("#open-file-status");
  fileLink.hidden = true;
  delete fileLink.dataset.key;
  status.hidden = true;
  status.textContent = "";
  document.querySelector("#open-file-annotate").hidden = true;
  closeOpenFileMenu();
}

function closeOpenFileMenu() {
  const menu = document.querySelector("#open-file-menu");
  const toggle = document.querySelector("#open-file-toggle");
  menu.hidden = true;
  toggle.setAttribute("aria-expanded", "false");
}

function toggleOpenFileMenu(event) {
  event.stopPropagation();
  const menu = document.querySelector("#open-file-menu");
  const open = menu.hidden;
  menu.hidden = !open;
  event.currentTarget.setAttribute("aria-expanded", open ? "true" : "false");
}

async function openFileLocally() {
  const key = document.querySelector("#open-file").dataset.key;
  const status = document.querySelector("#open-file-status");
  status.hidden = false;
  status.textContent = "正在用本机阅读器打开…";
  const response = await fetch("/api/file/open", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ key }),
  });
  const payload = await response.json().catch(() => ({}));
  status.textContent = response.ok ? "已用本机阅读器打开。保存批注时请保存这个文件。" : (payload.error || "没有打开");
}

async function annotateOriginal() {
  const key = document.querySelector("#open-file").dataset.key;
  const status = document.querySelector("#open-file-status");
  const button = document.querySelector("#open-file-annotate");
  status.hidden = false;
  status.textContent = "正在阅读原文并写入批注，可能需要一两分钟…";
  button.disabled = true;
  try {
    const response = await fetch("/api/annotate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key }),
    });
    const payload = await response.json().catch(() => ({}));
    status.textContent = response.ok ? (payload.message || "已写入批注") : (payload.error || "没有写入批注");
  } finally {
    button.disabled = false;
  }
}

function closeReader() {
  hideOpenFile();
  clearDanger();
  state.focus = null;
  state.noteScope = null;
  hideSelectMenu();
  setReaderChrome("", null);
  app.classList.remove("reading");
  document.querySelector("#reader").hidden = true;
  renderList();
  resize();
  draw();
}

function clearDanger() {
  armDanger(null, null);
}

function armDanger(onRecord, onNote, recordLabel = "删除") {
  const record = document.querySelector("#delete-record");
  const note = document.querySelector("#delete-note");
  record.hidden = !onRecord;
  note.hidden = !onNote;
  record.textContent = recordLabel;
  note.textContent = "删除笔记";
  record.onclick = onRecord;
  note.onclick = onNote;
}

async function deletePaper(key) {
  const paper = state.data.papers.find((item) => item.key === key);
  const title = paper ? paper.title : "这篇论文";
  const payload = await confirmDelete(
    `/api/paper?key=${encodeURIComponent(key)}`,
    `删除《${title}》？论文库里的原文和笔记会一起删除，作者和主题会保留。`,
  );
  if (!payload) return;
  closeReader();
  await load();
}

async function deleteAuthor(id, label) {
  const payload = await confirmDelete(
    `/api/author?id=${encodeURIComponent(id)}`,
    `删除作者 ${label}？这个人的论文会留在库里。`,
  );
  if (!payload) return;
  state.authors.delete(id);
  closeReader();
  await load();
}

async function deleteTopic(id, label) {
  const payload = await confirmDelete(
    `/api/topic?id=${encodeURIComponent(id)}`,
    `删除主题「${label}」？主题下的论文会留在库里，从属和相关关系会去掉。`,
  );
  if (!payload) return;
  state.topics.delete(id);
  closeReader();
  await load();
}

async function deleteNote(query) {
  const params = new URLSearchParams(query);
  const payload = await confirmDelete(`/api/note?${params.toString()}`, "删除这份笔记？论文、作者或主题本身会保留。");
  if (!payload) return;
  await load();
  if (query.scope === "paper" && state.focus && state.focus.kind === "paper") {
    await showPaper(query.id, document.querySelector("#reader-body"), document.querySelector("#open-file"));
    return;
  }
  if (query.scope === "synthesis") {
    closeReader();
    return;
  }
  if (state.focus && state.focus.kind === query.scope) await inspect(state.focus, false);
}

async function confirmDelete(url, message) {
  if (!window.confirm(message)) return null;
  const response = await fetch(url, { method: "DELETE" });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    window.alert(payload.error || "没有删除");
    return null;
  }
  return payload;
}

async function openDuplicates() {
  const dialog = document.querySelector("#dedupe-dialog");
  const box = document.querySelector("#dedupe-groups");
  const message = document.querySelector("#dedupe-message");
  const button = document.querySelector("#dedupe-merge");
  box.innerHTML = "";
  message.textContent = "正在对照题名、副标题和 DOI…";
  button.hidden = true;
  dialog.showModal();
  const response = await fetch("/api/duplicates");
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    message.textContent = payload.error || "没有完成查重";
    return;
  }
  message.textContent = payload.message || "";
  const groups = payload.groups || [];
  button.hidden = !groups.length;
  for (const group of groups) {
    const section = document.createElement("section");
    section.className = "dedupe-group";
    for (const paper of group) {
      const line = document.createElement("p");
      line.textContent = paper.doi ? `${paper.title} · ${paper.doi}` : paper.title;
      section.appendChild(line);
    }
    box.appendChild(section);
  }
}

async function mergeDuplicates() {
  const message = document.querySelector("#dedupe-message");
  const button = document.querySelector("#dedupe-merge");
  button.disabled = true;
  message.textContent = "正在合并…";
  const response = await fetch("/api/duplicates", { method: "POST" });
  const payload = await response.json().catch(() => ({}));
  button.disabled = false;
  if (!response.ok) {
    message.textContent = payload.error || "没有合并";
    return;
  }
  message.textContent = payload.message || "已合并。";
  button.hidden = true;
  document.querySelector("#dedupe-groups").innerHTML = "";
  await load();
}

async function showPaper(key, body, fileLink) {
  body.innerHTML = "<p class='meta-line'>正在打开笔记…</p>";
  const response = await fetch(`/api/paper?key=${encodeURIComponent(key)}`);
  const payload = await response.json();
  if (!response.ok) {
    body.textContent = payload.error || "打不开这篇论文";
    return;
  }
  if (payload.paper.has_file) {
    fileLink.hidden = false;
    fileLink.dataset.key = key;
    document.querySelector("#open-file-web").href = `/api/file?key=${encodeURIComponent(key)}`;
    document.querySelector("#open-file-annotate").hidden = payload.paper.file_suffix !== ".pdf";
  }
  const synthesis = currentSynthesis();
  const synthesisDate = synthesis && synthesis.has_synthesis && synthesis.updated_at ? `综合笔记更新于 ${formatUpdated(synthesis.updated_at)}` : "";
  const synthesisBlock = synthesis
    ? `<p class="meta-line">${synthesisDate}</p><p><button type="button" id="open-synthesis">${synthesis.has_synthesis ? "阅读这组综合笔记" : "这组还没有综合笔记"}</button></p>`
    : "";
  const markdown = payload.markdown || fallbackNote(payload);
  state.noteMarkdown = markdown;
  body.innerHTML = `${synthesisBlock}${biblioForm(payload.paper)}<article class="note">${renderMarkdown(markdown)}</article><h2>相关论文</h2><div class="related" id="related"></div>`;
  const form = body.querySelector("#biblio-form");
  form.onsubmit = (event) => saveBibliography(event, key);
  body.querySelector("#biblio-fill").onclick = () => fillBibliography(key);
  if (state.biblioNotice) {
    body.querySelector("#biblio-message").textContent = state.biblioNotice;
    state.biblioNotice = "";
  }
  await mountNote(
    body.querySelector("article.note"),
    { scope: "paper", id: key, label: payload.paper.title },
    payload.updated_at,
    () => updatePaperNote(key),
  );
  armDanger(
    () => deletePaper(key),
    payload.markdown ? () => deleteNote({ scope: "paper", id: key }) : null,
    "删除论文",
  );
  const openSynthesis = document.querySelector("#open-synthesis");
  if (openSynthesis && synthesis && synthesis.has_synthesis) {
    openSynthesis.onclick = () => showSynthesis(synthesis, body);
  }
  const related = document.querySelector("#related");
  if (!payload.related.length) {
    related.textContent = "调低相关度后，附近的论文会显示在这里。";
    return;
  }
  for (const item of payload.related) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "paper-row";
    button.innerHTML = "<strong></strong><span></span>";
    button.querySelector("strong").textContent = item.title;
    button.querySelector("span").textContent = `${item.reason} · ${item.weight}`;
    button.onclick = () => inspect({ kind: "paper", id: item.key }, true);
    related.appendChild(button);
  }
}

function biblioForm(paper) {
  return `<form id="biblio-form" class="biblio">
    <label>年份<input name="year" inputmode="numeric" value="${escapeAttr(paper.year || "")}" /></label>
    <label>期刊或会议<input name="venue" value="${escapeAttr(paper.venue || "")}" /></label>
    <label>DOI<input name="doi" value="${escapeAttr(paper.doi || "")}" /></label>
    <label>链接<input name="url" value="${escapeAttr(paper.url || "")}" /></label>
    <div class="biblio-actions">
      <button type="submit" class="primary">保存题录</button>
      <button type="button" id="biblio-fill">从原文补全</button>
    </div>
    <p id="biblio-message" class="meta-line">链接留空且写了 DOI 时，会保存成 doi.org 链接。从原文补全只填写空着的字段。</p>
  </form>`;
}

function escapeAttr(value) {
  return String(value ?? "").replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
}

async function saveBibliography(event, key) {
  event.preventDefault();
  const form = event.currentTarget;
  const message = form.querySelector("#biblio-message");
  message.textContent = "正在保存…";
  const response = await fetch("/api/paper-meta", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      key,
      year: form.year.value.trim(),
      venue: form.venue.value.trim(),
      doi: form.doi.value.trim(),
      url: form.url.value.trim(),
    }),
  });
  const payload = await response.json();
  if (!response.ok) {
    message.textContent = payload.error || "没有保存";
    return;
  }
  state.biblioNotice = "已保存。笔记开头的题录已改，笔记更新日期不变。";
  await load();
  await showPaper(key, document.querySelector("#reader-body"), document.querySelector("#open-file"));
}

async function fillBibliography(key) {
  const message = document.querySelector("#biblio-message");
  if (message) message.textContent = "正在从原文读取…";
  const response = await fetch("/api/biblio", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ key }),
  });
  const payload = await response.json();
  if (!response.ok) {
    if (message) message.textContent = payload.error || "没有补全";
    return;
  }
  const found = (payload.papers[0] && payload.papers[0].recognized) || {};
  state.biblioNotice = `原文识别：年份 ${found.year || "未识别"}，期刊或会议 ${found.venue || "未识别"}，DOI ${found.doi || "未识别"}，链接 ${found.url || "未识别"}。空着的字段已写入。`;
  await load();
  await showPaper(key, document.querySelector("#reader-body"), document.querySelector("#open-file"));
}

async function fillAllBibliography() {
  const button = document.querySelector("#fill-biblio");
  const status = document.querySelector("#graph-status");
  button.disabled = true;
  status.textContent = "正在从原文补全题录…";
  const response = await fetch("/api/biblio", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ all: true }),
  });
  const payload = await response.json();
  button.disabled = false;
  if (!response.ok) {
    status.textContent = payload.error || "没有补全";
    return;
  }
  await load();
  status.textContent = `已检查 ${payload.count} 篇，写入了 ${payload.filled_count} 篇原先空着的题录。`;
  if (state.focus && state.focus.kind === "paper") {
    await showPaper(state.focus.id, document.querySelector("#reader-body"), document.querySelector("#open-file"));
  }
}

async function updatePaperNote(key) {
  if (state.updating) return;
  state.updating = true;
  const button = document.querySelector("#update-note");
  button.disabled = true;
  button.textContent = "正在更新…";
  document.querySelector("#note-updated").textContent = "正在更新笔记，请稍等…";
  const response = await fetch("/api/note", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ key }),
  });
  const payload = await response.json();
  state.updating = false;
  if (!response.ok) {
    document.querySelector("#note-updated").textContent = payload.error || "笔记没有更新";
    button.disabled = false;
    button.textContent = "更新笔记";
    return;
  }
  await load();
  await showPaper(key, document.querySelector("#reader-body"), document.querySelector("#open-file"));
}

function showAuthor(id, body) {
  const author = state.data.authors.find((item) => item.id === id);
  if (!author) return;
  state.noteScope = null;
  setReaderChrome("", null);
  body.innerHTML = `<article class="note"><h1></h1><p class="meta-line">点选左侧作者可以只看这个人的论文。别名在这里增删，显示为“主名（别名 / 别名）”。</p><div class="alias-list" id="alias-list"></div><div class="alias-row"><input id="alias-input" placeholder="新别名，多项用 ; 隔开" /><button type="button" class="primary" id="alias-add">添加</button></div></article>`;
  body.querySelector("h1").textContent = author.label;
  const list = body.querySelector("#alias-list");
  for (const alias of author.aliases) {
    const row = document.createElement("div");
    const name = document.createElement("span");
    name.textContent = alias;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "text";
    remove.textContent = "移除";
    remove.onclick = () => changeAlias(author.id, [], [alias]);
    row.append(name, remove);
    list.appendChild(row);
  }
  body.querySelector("#alias-add").onclick = () => {
    const value = body.querySelector("#alias-input").value.trim();
    if (!value) return;
    changeAlias(author.id, value.split(/[;；]/).map((item) => item.trim()).filter(Boolean), []);
  };
  body.querySelector("article").appendChild(profileButton(author.has_profile ? "查看作者笔记" : "生成作者笔记", "author", author.id, !author.has_profile));
  armDanger(
    () => deleteAuthor(author.id, author.label),
    author.has_profile ? () => deleteNote({ scope: "author", id: author.id }) : null,
    "删除作者",
  );
}

function showTopic(id, body) {
  state.noteScope = null;
  setReaderChrome("", null);
  const topic = state.data.topics.find((item) => item.id === id);
  const papers = state.data.papers.filter((paper) => paper.topics.some((item) => item.id === id));
  body.innerHTML = `<article class="note"><h1></h1><p class="meta-line">下位主题里的论文会自动出现在这里。圆点越大，这篇在该主题下的分越高。</p><div id="topic-relations"></div><h2>论文</h2><div class="related" id="topic-papers"></div></article>`;
  body.querySelector("h1").textContent = topic ? topic.name : id;
  if (topic) {
    body.querySelector("article").appendChild(profileButton(topic.has_profile ? "查看主题笔记" : "生成主题笔记", "topic", topic.id, !topic.has_profile));
    renderTopicRelations(topic, body.querySelector("#topic-relations"));
    armDanger(
      () => deleteTopic(topic.id, topic.name),
      topic.has_profile ? () => deleteNote({ scope: "topic", id: topic.id }) : null,
      "删除主题",
    );
  }
  const list = body.querySelector("#topic-papers");
  for (const paper of papers.sort((a, b) => (b.topics.find((item) => item.id === id)?.score || 0) - (a.topics.find((item) => item.id === id)?.score || 0))) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "paper-row";
    button.innerHTML = "<strong></strong><span></span>";
    button.querySelector("strong").textContent = paper.title;
    const link = paper.topics.find((item) => item.id === id);
    const score = link?.score ?? 0;
    button.querySelector("span").textContent = link?.inherited ? `经由下位主题 · 相关度 ${score}` : `相关度 ${score}`;
    button.onclick = () => inspect({ kind: "paper", id: paper.key }, true);
    list.appendChild(button);
  }
}

function renderTopicRelations(topic, box) {
  const others = state.data.topics.filter((item) => item.id !== topic.id);
  const parentIds = new Set((topic.parents || []).map((item) => item.id));
  const childIds = new Set((topic.children || []).map((item) => item.id));
  const relatedIds = new Set((topic.related || []).map((item) => item.id));
  box.innerHTML = `<h2>从属</h2><div id="parent-list"></div><div id="child-list"></div><div class="relation-form"><select id="parent-select"></select><button type="button" id="parent-add">设为上位主题</button></div><h2>相关主题</h2><div id="related-list"></div><div class="relation-form"><select id="related-select"></select><button type="button" id="related-add">手动相关</button></div>`;
  const parentList = box.querySelector("#parent-list");
  for (const parent of topic.parents || []) {
    parentList.appendChild(relationLine(`${parent.name} 的下位主题`, "取消", () => changeTopicParent(topic.id, parent.id, false)));
  }
  if (!(topic.parents || []).length) parentList.appendChild(meta("还没有上位主题。"));
  const childList = box.querySelector("#child-list");
  for (const child of topic.children || []) childList.appendChild(meta(`包含下位主题 ${child.name}`));
  fillRelationSelect(box.querySelector("#parent-select"), others.filter((item) => !parentIds.has(item.id) && !childIds.has(item.id)));
  box.querySelector("#parent-add").onclick = () => {
    const parentId = box.querySelector("#parent-select").value;
    if (parentId) changeTopicParent(topic.id, parentId, true);
  };
  const relatedList = box.querySelector("#related-list");
  for (const related of topic.related || []) {
    const how = [related.shared ? "有共同论文" : "", related.manual ? "手动" : ""].filter(Boolean).join(" · ");
    const remove = related.manual ? () => changeTopicRelated(topic.id, related.id, false) : null;
    relatedList.appendChild(relationLine(`${related.name}（${how}）`, remove ? "取消手动" : "", remove));
  }
  if (!(topic.related || []).length) relatedList.appendChild(meta("还没有相关主题。"));
  fillRelationSelect(box.querySelector("#related-select"), others.filter((item) => !relatedIds.has(item.id)));
  box.querySelector("#related-add").onclick = () => {
    const otherId = box.querySelector("#related-select").value;
    if (otherId) changeTopicRelated(topic.id, otherId, true);
  };
}

function meta(text) {
  const line = document.createElement("p");
  line.className = "meta-line";
  line.textContent = text;
  return line;
}

function relationLine(label, action, onAction) {
  const row = document.createElement("div");
  row.className = "relation-line";
  const name = document.createElement("span");
  name.textContent = label;
  row.appendChild(name);
  if (onAction) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "text";
    button.textContent = action;
    button.onclick = onAction;
    row.appendChild(button);
  }
  return row;
}

function fillRelationSelect(select, topics) {
  select.innerHTML = "";
  const empty = document.createElement("option");
  empty.value = "";
  empty.textContent = topics.length ? "选择主题" : "没有可选主题";
  select.appendChild(empty);
  for (const topic of topics) {
    const option = document.createElement("option");
    option.value = topic.id;
    option.textContent = topic.name;
    select.appendChild(option);
  }
}

async function changeTopicParent(childId, parentId, add) {
  const response = await fetch(add ? "/api/topic-parent" : `/api/topic-parent?child=${encodeURIComponent(childId)}&parent=${encodeURIComponent(parentId)}`, add ? {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ child: childId, parent: parentId }),
  } : { method: "DELETE" });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    alert(payload.error || "从属关系没有保存");
    return;
  }
  await load();
  inspect({ kind: "topic", id: childId }, false);
}

async function changeTopicRelated(leftId, rightId, add) {
  const response = await fetch(add ? "/api/topic-related" : `/api/topic-related?left=${encodeURIComponent(leftId)}&right=${encodeURIComponent(rightId)}`, add ? {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ left: leftId, right: rightId }),
  } : { method: "DELETE" });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    alert(payload.error || "相关关系没有保存");
    return;
  }
  await load();
  inspect({ kind: "topic", id: leftId }, false);
}

async function showSynthesis(synthesis, body) {
  clearDanger();
  hideOpenFile();
  body.innerHTML = "<p class='meta-line'>正在打开笔记…</p>";
  const response = await fetch(`/api/synthesis?author=${encodeURIComponent(synthesis.author_id)}&topic=${encodeURIComponent(synthesis.topic_id)}`);
  const payload = await response.json();
  if (!response.ok) {
    body.innerHTML = `<p class="message error"></p>`;
    body.querySelector("p").textContent = payload.error || "没有这份综合笔记";
    setReaderChrome("", () => updateSynthesisNote(synthesis));
    return;
  }
  state.noteMarkdown = payload.markdown || "";
  body.innerHTML = `<article class="note">${renderMarkdown(state.noteMarkdown)}</article>`;
  const label = `${synthesis.author_label || synthesis.author_id} / ${synthesis.topic_name || synthesis.topic_id}`;
  await mountNote(
    body.querySelector("article.note"),
    { scope: "synthesis", id: `${synthesis.author_id}|${synthesis.topic_id}`, label },
    payload.updated_at,
    () => updateSynthesisNote(synthesis),
  );
  armDanger(null, () => deleteNote({ scope: "synthesis", author: synthesis.author_id, topic: synthesis.topic_id }));
}

async function updateSynthesisNote(synthesis) {
  if (state.updating) return;
  state.updating = true;
  const body = document.querySelector("#reader-body");
  body.innerHTML = "<p class='meta-line'>正在更新笔记，请稍等…</p>";
  const response = await fetch("/api/synthesis", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ author: synthesis.author_id, topic: synthesis.topic_id }),
  });
  const payload = await response.json();
  state.updating = false;
  if (!response.ok) {
    body.innerHTML = `<p class="message error"></p>`;
    body.querySelector("p").textContent = payload.error || "综合笔记没有更新";
    return;
  }
  await load();
  const fresh = state.data.syntheses.find((item) => item.author_id === synthesis.author_id && item.topic_id === synthesis.topic_id) || synthesis;
  await showSynthesis(fresh, document.querySelector("#reader-body"));
}

async function changeAlias(authorId, add, remove) {
  const response = await fetch("/api/alias", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ author: authorId, add, remove }),
  });
  const payload = await response.json();
  if (!response.ok) {
    alert(payload.error || "别名没有保存");
    return;
  }
  await load();
  inspect({ kind: "author", id: authorId }, false);
}

function fallbackNote(payload) {
  const reading = payload.reading || {};
  return `# ${payload.paper.title}\n\n${reading.one_sentence || "这篇论文还没有笔记。上传之后可以运行 autopaper notes 生成。"}\n`;
}

function renderMarkdown(markdown) {
  const lines = String(markdown || "").split(/\r?\n/);
  const html = [];
  let list = false;
  const flush = () => { if (list) { html.push("</ul>"); list = false; } };
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) { flush(); continue; }
    if (trimmed.startsWith("### ")) { flush(); html.push(`<h3>${escapeHtml(trimmed.slice(4))}</h3>`); continue; }
    if (trimmed.startsWith("## ")) { flush(); html.push(`<h2>${escapeHtml(trimmed.slice(3))}</h2>`); continue; }
    if (trimmed.startsWith("# ")) { flush(); html.push(`<h1>${escapeHtml(trimmed.slice(2))}</h1>`); continue; }
    if (trimmed.startsWith("- ")) {
      if (!list) { html.push("<ul>"); list = true; }
      html.push(`<li>${escapeHtml(trimmed.slice(2))}</li>`);
      continue;
    }
    flush();
    html.push(`<p>${escapeHtml(trimmed)}</p>`);
  }
  flush();
  return html.join("");
}

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));
}

function openUpload() {
  document.querySelector("#upload-message").textContent = "";
  document.querySelector("#upload-dialog").showModal();
  if (selectedFiles().length) suggestUpload();
}

function mountChoice(container, items, label, selected) {
  container.innerHTML = "";
  const chosen = selected || new Set();
  for (const item of items) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `chip${chosen.has(item.id) ? " on" : ""}`;
    button.dataset.token = item.id;
    button.textContent = label(item);
    button.onclick = () => button.classList.toggle("on");
    container.appendChild(button);
  }
}

function updateFileLabel() {
  const files = selectedFiles();
  document.querySelector("#file-label").textContent = files.length ? `已选 ${files.length} 个文件` : "选择文件，或选择整个文件夹。同名论文会自动跳过。";
}

async function suggestUpload() {
  updateFileLabel();
  const files = selectedFiles();
  const message = document.querySelector("#upload-message");
  if (!files.length) {
    message.textContent = "";
    return;
  }
  message.className = "message";
  message.textContent = "正在分别阅读作者栏、摘要和结论…";
  const form = new FormData();
  for (const file of files.slice(0, 40)) form.append("file", file, file.name);
  const response = await fetch("/api/upload/suggest", { method: "POST", body: form });
  const payload = await response.json().catch(() => ({}));
  renderAssignments(files, response.ok ? payload.papers || [] : []);
  if (!response.ok) {
    message.className = "message error";
    message.textContent = payload.error || "没有给出建议";
    return;
  }
  message.className = "message";
  message.textContent = payload.message || "";
}

function renderAssignments(files, suggestions) {
  const box = document.querySelector("#upload-papers");
  box.innerHTML = "";
  if (!state.data) return;
  files.forEach((file, index) => {
    const suggestion = suggestions[index] && suggestions[index].name === file.name ? suggestions[index] : {};
    const article = document.createElement("article");
    article.className = "upload-paper";
    article.dataset.name = file.name;
    const title = document.createElement("h3");
    title.textContent = file.name;
    const note = document.createElement("p");
    note.className = "meta-line";
    note.textContent = suggestion.message || "还没有建议，请手选作者和主题。";
    const authorLabel = document.createElement("p");
    authorLabel.className = "meta-line";
    authorLabel.textContent = "作者";
    const authors = document.createElement("div");
    authors.className = "chips authors";
    const topicLabel = document.createElement("p");
    topicLabel.className = "meta-line";
    topicLabel.textContent = "研究主题";
    const topics = document.createElement("div");
    topics.className = "chips topics";
    const newAuthor = document.createElement("input");
    newAuthor.className = "new-author";
    newAuthor.type = "text";
    newAuthor.placeholder = "这篇的新作者，多项用 ; 隔开";
    const newTopic = document.createElement("input");
    newTopic.className = "new-topic";
    newTopic.type = "text";
    newTopic.placeholder = "这篇的新主题，多项用 ; 隔开";
    article.append(title, note, authorLabel, authors, newAuthor, topicLabel, topics, newTopic);
    box.appendChild(article);
    const roles = new Map((suggestion.authors || []).map((item) => [item.id, item.reason || ""]));
    mountChoice(authors, state.data.authors, (item) => authorChipLabel(item, roles), new Set((suggestion.authors || []).map((item) => item.id)));
    mountChoice(topics, state.data.topics, (item) => item.name, new Set((suggestion.topics || []).map((item) => item.id)));
    mountSuggested(authors, suggestion.new_authors || []);
    mountSuggested(topics, suggestion.new_topics || []);
  });
}

function authorChipLabel(item, roles) {
  const role = roles.get(item.id) || "";
  return /^(一作|二作|通信作者)( · (一作|二作|通信作者))?$/.test(role) ? `${item.label} · ${role}` : item.label;
}

function mountSuggested(container, items) {
  for (const item of items) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `chip suggested${item.selected === false ? "" : " on"}`;
    button.dataset.token = item.name;
    button.textContent = item.role ? `${item.name} · ${item.role}` : item.name;
    button.onclick = () => button.classList.toggle("on");
    container.appendChild(button);
  }
}

async function classifyTopics() {
  const button = document.querySelector("#classify-topics");
  button.disabled = true;
  button.textContent = "归类中…";
  try {
    const response = await fetch("/api/topics/classify", { method: "POST" });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      alert(payload.error || "没有完成再分类");
      return;
    }
    await load();
    alert(payload.message || "已按已有主题再分类");
  } finally {
    button.disabled = false;
    button.textContent = "再分类";
  }
}

function selectedFiles() {
  return [...document.querySelector("#files").files, ...document.querySelector("#folder").files];
}

async function submitUpload(event) {
  event.preventDefault();
  const message = document.querySelector("#upload-message");
  const files = selectedFiles();
  if (!files.length) { message.className = "message error"; message.textContent = "请先选择论文文件。"; return; }
  const assignments = [...document.querySelectorAll("#upload-papers .upload-paper")].map((row) => ({
    name: row.dataset.name,
    authors: [
      ...[...row.querySelectorAll(".authors .chip.on")].map((item) => item.dataset.token),
      row.querySelector(".new-author").value.trim(),
    ].filter(Boolean),
    topics: [
      ...[...row.querySelectorAll(".topics .chip.on")].map((item) => item.dataset.token),
      row.querySelector(".new-topic").value.trim(),
    ].filter(Boolean),
  }));
  const missing = assignments.find((item) => !item.authors.length || !item.topics.length);
  if (assignments.length !== files.length || missing) {
    message.className = "message error";
    message.textContent = missing ? `请为《${missing.name}》选择或填写作者和主题。` : "请先等待每篇的作者和主题建议。";
    return;
  }
  const form = new FormData();
  for (const file of files) form.append("file", file, file.name);
  form.append("assignments", JSON.stringify(assignments));
  message.className = "message";
  message.textContent = "正在归类并生成笔记，请稍等…";
  const response = await fetch("/api/upload", { method: "POST", body: form });
  const payload = await response.json();
  if (!response.ok) { message.className = "message error"; message.textContent = payload.error || "上传失败"; return; }
  const noteStatus = payload.note_status ? ` ${payload.note_status}` : "";
  const recognized = (payload.bibliography || []).map((item) => `${item.title}：${item.summary || ""}`).join("\n");
  message.className = payload.note_status && payload.note_status.includes("失败") ? "message error" : "message ok";
  message.textContent = `上传 ${payload.uploaded.length} 篇，跳过 ${payload.skipped.length} 篇。${noteStatus}${recognized ? `\n${recognized}` : ""}`;
  document.querySelector("#files").value = "";
  document.querySelector("#folder").value = "";
  document.querySelector("#upload-papers").innerHTML = "";
  updateFileLabel();
  await load();
}

document.querySelector("#reader-body").addEventListener("mouseup", onNoteMouseUp);
document.querySelector("#reader").addEventListener("scroll", hideSelectMenu);
document.querySelector("#select-menu").addEventListener("mousedown", (event) => {
  if (event.target.closest("textarea, input")) return;
  event.preventDefault();
});
for (const button of document.querySelectorAll("#select-menu .color-dot")) {
  button.onclick = () => persistHighlight("highlight", button.dataset.color, "");
}
document.querySelector("#select-annotate").onclick = () => {
  document.querySelector("#ask-box").hidden = true;
  document.querySelector("#annotate-box").hidden = false;
  document.querySelector("#annotate-text").focus();
};
document.querySelector("#select-ask").onclick = openAsk;
document.querySelector("#ask-custom-send").onclick = () => askFollowup("custom");
document.querySelector("#ask-custom").addEventListener("keydown", (event) => {
  if (event.key === "Enter") askFollowup("custom");
});
document.querySelector("#ask-save").onclick = saveAskNote;
const ASK_QUESTIONS = [
  ["meaning", "这段在说什么"],
  ["whole", "和全文的关系"],
  ["author", "和作者的关系"],
  ["topic", "和主题的关系"],
  ["method", "方法能支持什么"],
  ["evidence", "主张和证据"],
  ["gap", "还缺什么"],
  ["corpus", "和其他论文的关系"],
  ["term", "术语"],
  ["reproduce", "能否沿用"],
];
for (const [ident, label] of ASK_QUESTIONS) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "chip";
  button.textContent = label;
  button.onclick = () => askFollowup(ident);
  document.querySelector("#ask-questions").appendChild(button);
}
document.querySelector("#annotate-save").onclick = () => {
  const comment = document.querySelector("#annotate-text").value.trim();
  if (!comment) return;
  document.querySelector("#annotate-text").value = "";
  persistHighlight("annotation", "", comment);
};
document.addEventListener("mousedown", (event) => {
  if (event.target.closest("#select-menu")) return;
  hideSelectMenu();
});

function hideSelectMenu() {
  const menu = document.querySelector("#select-menu");
  if (!menu) return;
  menu.hidden = true;
  document.querySelector("#annotate-box").hidden = true;
  document.querySelector("#ask-box").hidden = true;
}

function onNoteMouseUp(event) {
  if (event.target.closest("#select-menu")) return;
  const article = document.querySelector("#reader-body article.note");
  const selection = window.getSelection();
  if (!state.noteScope || !article || !selection || selection.rangeCount === 0 || selection.isCollapsed) return;
  if (!article.contains(selection.anchorNode)) return;
  const range = selection.getRangeAt(0);
  const blocks = [...article.querySelectorAll("p, li, h1, h2, h3")].filter((block) => {
    try { return range.intersectsNode(block); } catch (error) { return false; }
  });
  const parts = [];
  for (const block of blocks) {
    const paragraph = plain(block.textContent);
    if (!paragraph) continue;
    const selected = plain(textInside(block, range)) || paragraph;
    parts.push({ paragraph, selected_text: selected });
  }
  if (!parts.length) return;
  state.pendingSelection = { parts };
  state.askLabel = "";
  state.askAnswer = "";
  document.querySelector("#annotate-box").hidden = true;
  document.querySelector("#ask-box").hidden = true;
  document.querySelector("#ask-answer").textContent = "";
  document.querySelector("#ask-save").disabled = true;
  placeMenu(range.getBoundingClientRect());
}

function placeMenu(rect) {
  const menu = document.querySelector("#select-menu");
  menu.hidden = false;
  const width = Math.min(440, window.innerWidth - 16);
  let left = Math.max(8, rect.left);
  if (left + width > window.innerWidth - 8) left = Math.max(8, window.innerWidth - width - 8);
  let top = rect.bottom + 8;
  menu.style.left = `${left}px`;
  menu.style.top = `${top}px`;
  const height = menu.offsetHeight;
  if (top + height > window.innerHeight - 8) {
    menu.style.top = `${Math.max(8, rect.top - height - 8)}px`;
  }
}

function openAsk() {
  document.querySelector("#annotate-box").hidden = true;
  document.querySelector("#ask-box").hidden = false;
  document.querySelector("#ask-answer").textContent = state.askAnswer || "选择一个问题，或写下自己的问题。";
  document.querySelector("#ask-save").disabled = !state.askAnswer;
  nudgeMenu();
}

function nudgeMenu() {
  const menu = document.querySelector("#select-menu");
  const rect = menu.getBoundingClientRect();
  if (rect.right > window.innerWidth - 8) menu.style.left = `${Math.max(8, window.innerWidth - rect.width - 8)}px`;
  if (rect.bottom > window.innerHeight - 8) menu.style.top = `${Math.max(8, window.innerHeight - rect.height - 8)}px`;
}

async function askFollowup(question) {
  if (!state.pendingSelection || !state.noteScope) return;
  const custom = document.querySelector("#ask-custom").value.trim();
  if (question === "custom" && !custom) return;
  const answer = document.querySelector("#ask-answer");
  answer.textContent = "正在追问…";
  document.querySelector("#ask-save").disabled = true;
  const response = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      scope: state.noteScope.scope,
      scope_id: state.noteScope.id,
      scope_label: state.noteScope.label,
      question,
      custom,
      items: state.pendingSelection.parts,
      note: state.noteMarkdown,
    }),
  });
  const payload = await response.json();
  if (!response.ok) {
    state.askAnswer = "";
    answer.textContent = payload.error || "没有得到回答";
    return;
  }
  state.askLabel = payload.question || "追问";
  state.askAnswer = payload.answer || "";
  answer.textContent = state.askAnswer;
  document.querySelector("#ask-save").disabled = !state.askAnswer;
  nudgeMenu();
}

function saveAskNote() {
  if (!state.askAnswer) return;
  persistHighlight("annotation", "blue", `追问：${state.askLabel}\n${state.askAnswer}`);
}

function textInside(block, range) {
  const slice = range.cloneRange();
  if (!block.contains(range.startContainer)) slice.setStart(block, 0);
  if (!block.contains(range.endContainer)) slice.setEnd(block, block.childNodes.length);
  return slice.toString();
}

function plain(text) {
  return String(text || "").replace(/\s+/g, " ").trim();
}

async function persistHighlight(kind, color, comment) {
  const pending = state.pendingSelection;
  if (!pending || !state.noteScope) return;
  hideSelectMenu();
  const response = await fetch("/api/highlights", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      scope: state.noteScope.scope,
      scope_id: state.noteScope.id,
      scope_label: state.noteScope.label,
      kind,
      color,
      comment,
      items: pending.parts,
    }),
  });
  const payload = await response.json();
  state.pendingSelection = null;
  window.getSelection()?.removeAllRanges();
  if (!response.ok) {
    alert(payload.error || "没有保存");
    return;
  }
  const article = document.querySelector("#reader-body article.note");
  if (!article) return;
  article.innerHTML = renderMarkdown(state.noteMarkdown);
  await mountNote(article, state.noteScope, state.noteUpdatedAt, state.noteUpdate);
}

async function mountNote(article, scope, updatedAt, onUpdate) {
  state.noteScope = scope;
  setReaderChrome(updatedAt, onUpdate);
  const response = await fetch(`/api/highlights?scope=${encodeURIComponent(scope.scope)}&id=${encodeURIComponent(scope.id)}`);
  const payload = await response.json();
  const highlights = payload.highlights || [];
  const missed = applyHighlights(article, highlights);
  if (missed) {
    const hint = document.createElement("p");
    hint.className = "meta-line";
    hint.textContent = `有 ${missed} 条高亮或批注的原段落在当前笔记里没有对上，仍保存在我的笔记库。`;
    article.prepend(hint);
  }
  if (state.jumpHighlight) {
    const target = article.querySelector(`[data-highlight-id="${cssEscape(state.jumpHighlight)}"]`);
    const jump = state.jumpHighlight;
    state.jumpHighlight = "";
    if (target) target.scrollIntoView({ block: "center" });
    else if (!highlights.some((item) => item.id === jump) || missed) {
      const hint = document.createElement("p");
      hint.className = "meta-line";
      hint.textContent = "这条高亮对应的段落在当前笔记里没有对上，原文仍保存在我的笔记库。";
      article.prepend(hint);
    }
  }
}

function cssEscape(value) {
  return window.CSS && CSS.escape ? CSS.escape(value) : String(value).replace(/"/g, "");
}

function applyHighlights(article, highlights) {
  const blocks = [...article.querySelectorAll("p, li, h1, h2, h3")];
  let missed = 0;
  for (const item of highlights) {
    const paragraph = plain(item.paragraph);
    const selected = plain(item.selected_text);
    const block = blocks.find((node) => {
      const text = plain(node.textContent);
      if (!text) return false;
      if (paragraph && (text === paragraph || text.includes(paragraph))) return true;
      return Boolean(selected && text.includes(selected));
    });
    if (!block) {
      missed += 1;
      continue;
    }
    if (!(selected && wrapExact(block, selected, item))) {
      block.classList.add("hl-block", `hl-${item.color || "note"}`);
      if (!block.dataset.highlightId) block.dataset.highlightId = item.id;
    }
    if (item.kind === "annotation" && item.comment) {
      const aside = document.createElement("aside");
      aside.className = "annotation";
      aside.dataset.highlightId = item.id;
      aside.textContent = item.comment;
      block.after(aside);
    }
  }
  return missed;
}

function wrapExact(root, needle, item) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  let node = walker.nextNode();
  while (node) {
    const value = node.nodeValue || "";
    const index = value.indexOf(needle);
    if (index >= 0) {
      const range = document.createRange();
      range.setStart(node, index);
      range.setEnd(node, index + needle.length);
      const mark = document.createElement("mark");
      mark.className = item.color ? `hl hl-${item.color}` : (item.kind === "annotation" ? "hl hl-note" : "hl hl-yellow");
      mark.dataset.highlightId = item.id;
      range.surroundContents(mark);
      return true;
    }
    node = walker.nextNode();
  }
  return false;
}

async function openLibrary() {
  app.classList.add("reading");
  document.querySelector("#reader").hidden = false;
  hideOpenFile();
  state.focus = null;
  state.noteScope = null;
  hideSelectMenu();
  setReaderChrome("", null);
  const body = document.querySelector("#reader-body");
  body.innerHTML = "<p class='meta-line'>正在打开我的笔记库…</p>";
  renderList();
  resize();
  draw();
  const response = await fetch("/api/highlights");
  const payload = await response.json();
  renderLibrary(body, payload.highlights || []);
}

function renderLibrary(body, highlights) {
  body.innerHTML = "<article class='note'><h1>我的笔记库</h1><p class='meta-line'>高亮和批注按论文、作者、主题分别保存。更新笔记不会清掉这里的内容。</p></article>";
  const article = body.querySelector("article");
  if (!highlights.length) {
    const empty = document.createElement("p");
    empty.textContent = "还没有高亮或批注。打开一篇笔记，用鼠标左键划取文字即可。";
    article.appendChild(empty);
    return;
  }
  const names = { paper: "论文", author: "作者", topic: "主题", synthesis: "综合笔记" };
  const colors = { yellow: "黄色高亮", green: "绿色高亮", blue: "蓝色高亮", pink: "粉色高亮", orange: "橙色高亮" };
  for (const scope of ["paper", "author", "topic", "synthesis"]) {
    const items = highlights.filter((item) => item.scope === scope);
    if (!items.length) continue;
    const group = document.createElement("section");
    group.className = "library-group";
    const heading = document.createElement("h2");
    heading.textContent = names[scope];
    group.appendChild(heading);
    for (const item of items) {
      const card = document.createElement("div");
      card.className = "library-item";
      const title = document.createElement("div");
      title.className = "row";
      const swatch = document.createElement("i");
      swatch.className = `swatch color-dot ${item.color || "orange"}`;
      const label = document.createElement("strong");
      const kindLabel = String(item.comment || "").startsWith("追问：") ? "追问" : (item.kind === "annotation" ? "批注" : (colors[item.color] || "高亮"));
      label.textContent = `${item.scope_label || "未命名"} · ${kindLabel}`;
      title.append(swatch, label);
      const paragraph = document.createElement("p");
      paragraph.textContent = item.paragraph;
      card.append(title, paragraph);
      if (item.kind === "annotation" && item.comment) {
        const comment = document.createElement("p");
        comment.className = "annotation";
        comment.textContent = item.comment;
        card.appendChild(comment);
      }
      const when = document.createElement("p");
      when.className = "meta-line";
      when.textContent = `保存于 ${formatUpdated(item.updated_at)}`;
      const actions = document.createElement("div");
      actions.className = "row";
      const open = document.createElement("button");
      open.type = "button";
      open.className = "text";
      open.textContent = "打开原文";
      open.onclick = () => openHighlight(item);
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "text";
      remove.textContent = "移除";
      remove.onclick = () => removeHighlight(item.id);
      actions.append(open, remove);
      card.append(when, actions);
      group.appendChild(card);
    }
    article.appendChild(group);
  }
}

function openHighlight(item) {
  state.jumpHighlight = item.id;
  if (item.scope === "paper") {
    inspect({ kind: "paper", id: item.scope_id }, true);
    return;
  }
  if (item.scope === "author" || item.scope === "topic") {
    openProfile(item.scope, item.scope_id, false);
    return;
  }
  const [authorId, topicId] = String(item.scope_id).split("|");
  const synthesis = (state.data.syntheses || []).find((entry) => entry.author_id === authorId && entry.topic_id === topicId) || {
    author_id: authorId,
    topic_id: topicId,
    author_label: item.scope_label,
    topic_name: "",
  };
  app.classList.add("reading");
  document.querySelector("#reader").hidden = false;
  showSynthesis(synthesis, document.querySelector("#reader-body"));
}

async function removeHighlight(id) {
  const response = await fetch(`/api/highlights?id=${encodeURIComponent(id)}`, { method: "DELETE" });
  const payload = await response.json();
  if (!response.ok) {
    alert(payload.error || "没有移除");
    return;
  }
  await openLibrary();
}
