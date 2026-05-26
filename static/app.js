const API_BASE = "";
const STORAGE_KEY = "research_rag_recent";

let currentResponse = null;
let activeSourceIdx = 0;
const recentQueries = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");

function togglePanel(side) {
  const app = document.getElementById("app");
  const panel = document.getElementById("panel-" + side);
  const btn = document.getElementById("btn-" + side);
  const cls = side + "-open";
  if (app.classList.contains(cls)) {
    app.classList.remove(cls);
    panel.style.display = "none";
    btn.classList.remove("active");
  } else {
    app.classList.add(cls);
    panel.style.display = "flex";
    btn.classList.add("active");
  }
}

async function loadHealth() {
  try {
    const r = await fetch("/health");
    const data = await r.json();
    const qdrantOk = data.qdrant_reachable === true || data.qdrant?.status === "ok";
    const ollamaOk = data.ollama_reachable === true || data.ollama?.status === "ok";
    setStatus("qdrant", qdrantOk, qdrantOk ? "online" : "offline");
    setStatus("ollama", ollamaOk, ollamaOk ? "online" : "offline");
    const points = data.points_count ?? data.qdrant?.points_count ?? "-";
    document.getElementById("val-chunks").textContent =
      typeof points === "number" ? points.toLocaleString() : points;
    document.getElementById("val-papers").textContent = data.papers ?? "15";
  } catch (e) {
    setStatus("qdrant", false, "error");
    setStatus("ollama", false, "error");
  }
}

function setStatus(name, ok, text) {
  document.getElementById("dot-" + name).className =
    "status-dot " + (ok ? "ok" : "bad");
  document.getElementById("val-" + name).textContent = text;
}

function askSuggestion(q) {
  document.getElementById("query-input").value = q;
  sendQuery();
}

function newSession() {
  document.getElementById("chat-inner").innerHTML = `
    <div class="welcome">
      <div class="welcome-mark"><i class="ti ti-book-2"></i></div>
      <h1>Research RAG Assistant</h1>
      <p>Ask questions about the indexed academic papers. Every claim is grounded in a cited paper, page, and section.</p>
    </div>`;
  document.getElementById("chat-head").style.visibility = "hidden";
  document.getElementById("sources-body").innerHTML =
    '<div style="color: #666; font-size: 12px; text-align: center; padding: 40px 0;">No sources yet</div>';
  document.getElementById("sources-badge").style.display = "none";
  document.getElementById("query-input").value = "";
  currentResponse = null;
}

async function sendQuery() {
  const input = document.getElementById("query-input");
  const question = input.value.trim();
  if (!question) return;

  const sendBtn = document.getElementById("send-btn");
  sendBtn.disabled = true;
  input.disabled = true;

  document.getElementById("chat-head").style.visibility = "visible";
  document.getElementById("head-question").textContent = question;
  document.getElementById("head-sources").textContent = "…";
  document.getElementById("head-latency").textContent = "…";
  document.getElementById("head-cites").textContent = "…";

  document.getElementById("chat-inner").innerHTML = `
    <div class="loading">
      <div class="spinner"></div>
      <span>Retrieving relevant chunks, reranking, generating answer with Qwen2.5:7b…</span>
    </div>`;

  document.getElementById("sources-body").innerHTML = `
    <div style="color: #666; font-size: 12px; text-align: center; padding: 40px 0;">Retrieving…</div>`;

  try {
    const t0 = performance.now();
    const r = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const elapsed = (performance.now() - t0) / 1000;

    if (!r.ok) {
      const errTxt = await r.text();
      throw new Error(`HTTP ${r.status}: ${errTxt}`);
    }
    const data = await r.json();
    currentResponse = data;
    renderAnswer(data, elapsed);
    renderSources(data.retrieved_chunks || data.retrieved || []);
    pushRecent(question);
  } catch (e) {
    document.getElementById("chat-inner").innerHTML = `
      <div class="error-card">
        <i class="ti ti-alert-triangle"></i>
        <div>
          <div style="font-weight: 500; margin-bottom: 4px;">Query failed</div>
          <div style="font-size: 12px; color: #f5c4b3; font-family: 'JetBrains Mono', monospace;">${escapeHtml(String(e))}</div>
        </div>
      </div>`;
  } finally {
    sendBtn.disabled = false;
    input.disabled = false;
    input.focus();
  }
}

function renderAnswer(data, elapsed) {
  const answerText = data.answer || "(empty)";
  const citations = data.citations || [];
  const checkData = data.citation_check || { all_valid: true, n_extracted: 0, n_valid: 0 };

  const isAbstain = data.abstained === true;
  // If the system abstained, suppress citation pills inside the answer
  const html = isAbstain
    ? `<p>${escapeHtml(answerText).replace(/\[Paper:[^\]]+\]/g, "").trim()}</p>`
    : formatAnswerWithCitations(answerText, citations);

  const latency = data.latency_ms?.total_ms ?? data.latency_ms?.total ?? elapsed * 1000;
  document.getElementById("head-sources").textContent = `${(data.retrieved_chunks||data.retrieved||[]).length} sources`;
  document.getElementById("head-latency").textContent = `${(latency/1000).toFixed(1)}s`;
  document.getElementById("head-cites").textContent = isAbstain ? "abstained" : `${checkData.n_extracted} cites`;

  const validBadge = isAbstain
    ? `<span class="badge-ok"><i class="ti ti-shield-check"></i>abstained (no fabrication)</span>`
    : (checkData.all_valid
        ? `<span class="badge-ok"><i class="ti ti-circle-check"></i>${checkData.n_valid}/${checkData.n_extracted} citations valid</span>`
        : `<span class="badge-bad"><i class="ti ti-alert-circle"></i>${checkData.n_valid}/${checkData.n_extracted} citations valid</span>`);

  const retrieval = (data.latency_ms?.retrieval_ms ?? data.latency_ms?.retrieval ?? 0);
  const rerank = (data.latency_ms?.rerank_ms ?? data.latency_ms?.rerank ?? 0);
  const generation = (data.latency_ms?.generation_ms ?? data.latency_ms?.generation ?? 0);

  document.getElementById("chat-inner").innerHTML = `
    <div class="answer">
      ${html}
      <div class="answer-footer">
        ${validBadge}
        <span>retrieval ${Math.round(retrieval)}ms</span>
        <span>rerank ${Math.round(rerank)}ms</span>
        <span>generation ${(generation/1000).toFixed(1)}s</span>
      </div>
      <div class="answer-actions">
        <button class="action" onclick="copyAnswer()"><i class="ti ti-copy"></i>Copy</button>
        <button class="action" onclick="togglePanel('right')"><i class="ti ti-list-details"></i>Sources</button>
      </div>
    </div>`;

  document.getElementById("sources-badge").style.display = "block";
}

function formatAnswerWithCitations(text, allowedCitations) {
  let citeMap = new Map();
  allowedCitations.forEach((c, i) => citeMap.set(c, i + 1));

  let safe = escapeHtml(text);

  safe = safe.replace(/\[Paper:\s*[^\]]+\]/g, (match) => {
    const idx = citeMap.get(match);
    if (idx !== undefined) {
      return `<span class="cite" onclick="focusSource(${idx-1})" title="${escapeHtml(match)}">${idx}</span>`;
    }
    return `<span class="cite" style="border-color:#d85a30; color:#f0997b;" title="${escapeHtml(match)} (not in retrieved set)">?</span>`;
  });

  safe = safe.replace(/\[(CLS|SEP|MASK|PAD|UNK)\]/g, '<span class="code-tag">[$1]</span>');

  const paras = safe.split(/\n\s*\n/);
  return paras.map(p => `<p>${p.replace(/\n/g, "<br>")}</p>`).join("");
}

function renderSources(retrieved) {
  const body = document.getElementById("sources-body");
  if (!retrieved.length) {
    body.innerHTML = '<div style="color: #666; font-size: 12px; text-align: center; padding: 40px 0;">No sources</div>';
    document.getElementById("sources-title").textContent = "Sources";
    return;
  }
  document.getElementById("sources-title").textContent = `Sources · ${retrieved.length} chunks`;

  body.innerHTML = retrieved.map((c, i) => {
    const rerank = c.score_breakdown?.reranker ?? c.score;
    const dense = c.score_breakdown?.dense_score ?? c.score_breakdown?.dense ?? null;
    const sparse = c.score_breakdown?.sparse_score ?? c.score_breakdown?.sparse ?? null;
    const pills = [];
    if (dense !== null) pills.push(`<span class="score-pill dense">dense ${Number(dense).toFixed(2)}</span>`);
    if (sparse !== null && sparse > 0) pills.push(`<span class="score-pill sparse">sparse ${Number(sparse).toFixed(2)}</span>`);
    pills.push(`<span class="score-pill rerank">rerank ${Number(rerank).toFixed(3)}</span>`);
    const title = c.metadata?.paper_title || c.source || "(untitled)";
    const src = c.source || c.metadata?.source || "?";
    const page = c.page ?? c.metadata?.page ?? "?";
    const section = c.section ?? c.metadata?.section ?? "?";
    return `
      <div class="source-card ${i === 0 ? "active" : ""}" onclick="focusSource(${i})" data-idx="${i}">
        <div class="source-head">
          <span class="source-rank">${i + 1}</span>
          <span class="source-score">${Number(rerank).toFixed(3)}</span>
        </div>
        <div class="source-title">${escapeHtml(title)}</div>
        <div class="source-meta">${escapeHtml(String(src).replace(".pdf","").slice(0,40))} · p.${page} · §${escapeHtml(String(section))}</div>
        <div class="score-breakdown">${pills.join("")}</div>
        ${(c.text || c.text_preview) ? `<div class="source-text">${escapeHtml(c.text || c.text_preview).slice(0, 400)}…</div>` : ""}
      </div>`;
  }).join("");
}

function focusSource(idx) {
  document.querySelectorAll(".source-card").forEach((el, i) => {
    el.classList.toggle("active", i === idx);
  });
  if (!document.getElementById("app").classList.contains("right-open")) {
    togglePanel("right");
  }
  const card = document.querySelector(`.source-card[data-idx="${idx}"]`);
  if (card) card.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function copyAnswer() {
  if (!currentResponse) return;
  navigator.clipboard.writeText(currentResponse.answer || "");
}

function pushRecent(q) {
  const existing = recentQueries.findIndex(r => r === q);
  if (existing >= 0) recentQueries.splice(existing, 1);
  recentQueries.unshift(q);
  if (recentQueries.length > 10) recentQueries.length = 10;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(recentQueries));
  renderRecent();
}

function renderRecent() {
  const list = document.getElementById("recent-list");
  if (!recentQueries.length) {
    list.innerHTML = '<div style="color: #555; font-size: 11px; padding: 8px 10px;">No recent queries</div>';
    return;
  }
  list.innerHTML = recentQueries.map((q, i) =>
    `<div class="recent-item ${i === 0 ? "active" : ""}" onclick="askSuggestion(${JSON.stringify(q).replace(/"/g,"&quot;")})">${escapeHtml(q)}</div>`
  ).join("");
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

loadHealth();
setInterval(loadHealth, 30000);
renderRecent();
