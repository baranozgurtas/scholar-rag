const API_BASE = "";
const STORAGE_KEY = "scholar_rag_recent";

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
      <h1>Scholar RAG</h1>
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
  const checkData = data.citation_check || { all_valid: false, n_extracted: 0, n_valid: 0 };

  const isAbstain = data.abstained === true;
  // If the system abstained, suppress citation pills inside the answer
  const html = isAbstain
    ? `<p>${escapeHtml(answerText).replace(/\[Paper:[^\]]+\]/g, "").trim()}</p>`
    : formatAnswerWithCitations(answerText, data.retrieved_chunks || data.retrieved || []);

  const latency = data.latency_ms?.total_ms ?? data.latency_ms?.total ?? elapsed * 1000;
  document.getElementById("head-sources").textContent = `${(data.retrieved_chunks||data.retrieved||[]).length} sources`;
  document.getElementById("head-latency").textContent = `${(latency/1000).toFixed(1)}s`;
  const isFailure = data.outcome === "retrieval_failed";
  document.getElementById("head-cites").textContent =
    isFailure ? "failed" : isAbstain ? "abstained" : `${checkData.n_extracted} cites`;

  // Why the system did not answer. Tag checks are structural: a matching tag
  // means the cited page was in the supplied context, not that it supports the claim.
  const ABSTAIN_REASONS = {
    model_abstained: "model declined: context judged insufficient",
    mixed_abstention: "withheld: model both declined and answered",
    uncited_answer: "withheld: answer had no citation tags",
    invalid_citation: "withheld: cited a passage outside the supplied context",
    generation_error: "withheld: generation failed",
    no_context: "no passages retrieved",
    low_rerank_score: "abstained: top rerank score below threshold",
  };
  const abstainLabel = ABSTAIN_REASONS[data.outcome] || "abstained";
  const validBadge = isFailure
    ? `<span class="badge-bad"><i class="ti ti-alert-triangle"></i>failed: retrieval stage error (${escapeHtml(data.outcome_detail || "unknown")})</span>`
    : isAbstain
    ? `<span class="badge-ok"><i class="ti ti-shield-check"></i>${abstainLabel}</span>`
    : (checkData.all_valid
        ? `<span class="badge-ok" title="Tags match supplied passages; this does not verify that the passages support each claim."><i class="ti ti-circle-check"></i>${checkData.n_valid}/${checkData.n_extracted} citation tags match retrieved passages</span>`
        : `<span class="badge-bad"><i class="ti ti-alert-circle"></i>${checkData.n_valid}/${checkData.n_extracted} citation tags match retrieved passages</span>`);

  const retrieval = (data.latency_ms?.retrieval_ms ?? data.latency_ms?.retrieval ?? 0);
  const rerank = (data.latency_ms?.rerank_ms ?? data.latency_ms?.rerank ?? 0);
  const generation = (data.latency_ms?.generation_ms ?? data.latency_ms?.generation ?? 0);

  document.getElementById("chat-inner").innerHTML = `
    <div class="answer">
      ${html}
      <div class="answer-footer">
        ${validBadge}
        ${data.retrieval_mode === "dense_only" ? `<span title="The query's sparse encoding had no positive weights.">dense only</span>` : ""}
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

// Index of the retrieved chunk a citation tag refers to, or -1. Mirrors the
// matching rules in rag/guards/citation_checker.py: exact tag, same title and
// page, or a whole-word shortened title (>= 6 chars) on the same page. Both
// sides are NFKC-normalized there and here, so "Eﬀects" matches "Effects".
function matchChunkIndex(retrieved, title, page, section) {
  const norm = s => String(s).normalize("NFKC").trim().toLowerCase().replace(/\s+/g, " ");
  const t = norm(title), p = String(page).trim(), s = norm(section);
  const tp = retrieved.map(c => [norm(c.paper_title ?? c.metadata?.paper_title ?? ""), String(c.page ?? c.metadata?.page)]);
  let i = retrieved.findIndex((c, k) => tp[k][0] === t && tp[k][1] === p && norm(c.section ?? c.metadata?.section ?? "") === s);
  if (i < 0) i = tp.findIndex(([ct, cp]) => ct === t && cp === p);
  // Python's `\w` on str is Unicode letters, numbers and "_"; JS `\w` is
  // ASCII-only, so spell the class out. Length is in code points, like len().
  if (i < 0 && [...t].length >= 6) {
    const W = "[\\p{L}\\p{N}_]";
    const re = new RegExp(`(?<!${W})${t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}(?!${W})`, "u");
    i = tp.findIndex(([ct, cp]) => cp === p && re.test(ct));
  }
  return i;
}

function formatAnswerWithCitations(text, retrieved) {
  let safe = escapeHtml(text);

  // Bracketed or parenthesized tags. The pill number is the rank of the
  // retrieved chunk the tag matches, so it lines up with the source cards.
  safe = safe.replace(/[\[(]Paper:\s*([^|\])]+?)\s*\|\s*p\.([\d-]+)\s*\|\s*§([\w-]+)\s*[\])]/g, (match, title, page, section) => {
    const unescape = s => s.replace(/&amp;/g, "&").replace(/&#39;/g, "'").replace(/&quot;/g, '"');
    const idx = matchChunkIndex(retrieved, unescape(title), page, unescape(section));
    if (idx >= 0) {
      return `<span class="cite" onclick="focusSource(${idx})" title="${match}">${idx + 1}</span>`;
    }
    return `<span class="cite" style="border-color:#d85a30; color:#f0997b;" title="${match} (not in retrieved set)">?</span>`;
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
    const title = c.paper_title || c.metadata?.paper_title || c.source || "(untitled)";
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
