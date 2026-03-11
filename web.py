#!/usr/bin/env python3
"""
Agent Memory Viewer — browse and manage the memory database.
Run: python3 web.py [--port 7337]
"""

import os
import json
import argparse
from datetime import datetime
from flask import Flask, jsonify, request, render_template_string

import threading
import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

DB_URL = os.environ.get(
    "AGENT_MEMORY_DB_URL",
    "postgresql://agent:agent_memory_pwd@localhost:5433/agent_memory"
)
EMBED_MODEL = os.environ.get("AGENT_MEMORY_EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5")

CATEGORIES = ["convention", "gotcha", "pattern", "preference", "dependency", "optimization", "decision"]

app = Flask(__name__)

# ── Embedding server ───────────────────────────────────────────────────────────

_embed_model = None
_embed_lock = threading.Lock()


def get_embed_model():
    global _embed_model
    with _embed_lock:
        if _embed_model is None:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            from sentence_transformers import SentenceTransformer
            _embed_model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True)
    return _embed_model


@app.post("/api/embed")
def api_embed():
    data = request.get_json() or {}
    text = data.get("text", "")
    mode = data.get("mode", "document")
    if not text:
        return jsonify({"error": "text required"}), 400
    try:
        model = get_embed_model()
        prefix = "search_query: " if mode == "query" else "search_document: "
        vec = model.encode(prefix + text, normalize_embeddings=True)
        return jsonify({"embedding": vec.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def get_conn():
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = True
    register_vector(conn)
    return conn


def serialize(row):
    out = {}
    for k, v in row.items():
        if k.endswith("_embedding") or (hasattr(v, "tolist") and callable(v.tolist)):
            continue
        out[k] = v.isoformat() if isinstance(v, datetime) else v
    return out


# ── Projects ──────────────────────────────────────────────────────────────────

@app.get("/api/projects")
def api_projects():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT project FROM memories ORDER BY project")
    projects = [r[0] for r in cur.fetchall()]
    conn.close()
    return jsonify(projects)


# ── Stats ──────────────────────────────────────────────────────────────────────

@app.get("/api/stats")
def api_stats():
    project = request.args.get("project", "")
    conn = get_conn()
    cur = conn.cursor()
    where = "WHERE project = %s" if project else ""
    params = [project] if project else []
    cur.execute(f"SELECT COUNT(*) FROM memories {where}", params)
    total = cur.fetchone()[0]
    cur.execute(f"SELECT COUNT(*) FROM memories {where} {'AND' if project else 'WHERE'} status = 'active'", params)
    active = cur.fetchone()[0]
    conn.close()
    return jsonify({"total": total, "active": active})


@app.get("/api/stats/full")
def api_stats_full():
    project = request.args.get("project", "")
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    proj_filter = "WHERE project = %s" if project else ""
    proj_and    = "AND project = %s" if project else ""
    p = [project] if project else []

    cur.execute(f"SELECT COUNT(*) AS n FROM memories {proj_filter}", p)
    total = cur.fetchone()["n"]

    cur.execute(f"SELECT category, COUNT(*) AS n FROM memories {proj_filter} GROUP BY category ORDER BY n DESC", p)
    by_category = {r["category"]: r["n"] for r in cur.fetchall()}

    cur.execute(f"SELECT status, COUNT(*) AS n FROM memories {proj_filter} GROUP BY status ORDER BY n DESC", p)
    by_status = {r["status"]: r["n"] for r in cur.fetchall()}

    cur.execute(f"""
        SELECT ROUND(AVG(confidence)::numeric, 3) AS avg,
               ROUND(MIN(confidence)::numeric, 3) AS min,
               ROUND(MAX(confidence)::numeric, 3) AS max
        FROM memories WHERE status = 'active' {proj_and}
    """, p)
    conf_row = cur.fetchone()
    confidence = {k: float(v) if v is not None else None for k, v in conf_row.items()}

    cur.execute(f"""
        SELECT id, category, content, confidence, last_referenced
        FROM memories WHERE status = 'active' {proj_and}
        ORDER BY last_referenced DESC NULLS LAST
        LIMIT 8
    """, p)
    top_referenced = [serialize(r) for r in cur.fetchall()]

    cur.execute(f"SELECT COALESCE(SUM(LENGTH(content)), 0) AS c FROM memories {proj_filter}", p)
    est_tokens = cur.fetchone()["c"] // 4

    conn.close()
    return jsonify({
        "total": total,
        "by_category": by_category,
        "by_status": by_status,
        "confidence": confidence,
        "top_referenced": top_referenced,
        "estimated_tokens": est_tokens,
    })


# ── Memories ──────────────────────────────────────────────────────────────────

@app.get("/api/memories")
def api_memories():
    project  = request.args.get("project", "")
    status   = request.args.get("status", "")
    category = request.args.get("category", "")
    search   = request.args.get("search", "")
    limit    = int(request.args.get("limit", 50))
    offset   = int(request.args.get("offset", 0))

    clauses, params = [], []
    if project:  clauses.append("project = %s");     params.append(project)
    if status:   clauses.append("status = %s");      params.append(status)
    if category: clauses.append("category = %s");    params.append(category)
    if search:   clauses.append("content ILIKE %s"); params.append(f"%{search}%")
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cur.execute(f"SELECT COUNT(*) AS n FROM memories {where}", params)
    total = cur.fetchone()["n"]

    cur.execute(f"""
        SELECT id, project, category, content, confidence, status, last_referenced, created_at
        FROM memories {where}
        ORDER BY confidence DESC, created_at DESC
        LIMIT %s OFFSET %s
    """, params + [limit, offset])
    rows = [serialize(r) for r in cur.fetchall()]
    conn.close()
    resp = jsonify(rows)
    resp.headers["X-Total-Count"] = total
    return resp


@app.patch("/api/memories/<mem_id>")
def update_memory(mem_id):
    data = request.get_json() or {}
    allowed = {"category", "content", "confidence", "status"}
    updates = {k: v for k, v in data.items() if k in allowed}
    if not updates:
        return jsonify({"error": "no valid fields"}), 400
    set_clause = ", ".join(f"{k} = %s" for k in updates)
    params = list(updates.values()) + [mem_id]
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(f"UPDATE memories SET {set_clause}, updated_at = NOW() WHERE id = %s", params)
    conn.close()
    return jsonify({"ok": True}) if cur.rowcount else (jsonify({"error": "not found"}), 404)


@app.delete("/api/memories/<mem_id>")
def delete_memory(mem_id):
    hard = request.args.get("hard") == "true"
    conn = get_conn()
    cur = conn.cursor()
    if hard:
        cur.execute("DELETE FROM memories WHERE id = %s", (mem_id,))
    else:
        cur.execute("UPDATE memories SET status = 'archived', updated_at = NOW() WHERE id = %s", (mem_id,))
    conn.close()
    return jsonify({"ok": True}) if cur.rowcount else (jsonify({"error": "not found"}), 404)


# ── UI ─────────────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent Memory</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'SF Mono', 'Fira Code', monospace; font-size: 13px;
         background: #0d1117; color: #c9d1d9; min-height: 100vh; }

  header { padding: 16px 24px; border-bottom: 1px solid #21262d;
           display: flex; align-items: center; gap: 16px; }
  header h1 { font-size: 15px; color: #58a6ff; font-weight: 600; }
  header .subtitle { color: #6e7681; font-size: 12px; }
  .spacer { flex: 1; }
  .stats-bar { display: flex; gap: 8px; }
  .stat { background: #161b22; border: 1px solid #21262d; border-radius: 6px;
          padding: 3px 10px; color: #8b949e; font-size: 11px; }
  .stat span { color: #58a6ff; font-weight: 600; }

  .toolbar { padding: 10px 24px; border-bottom: 1px solid #21262d;
             display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
  select, input { background: #161b22; border: 1px solid #30363d; color: #c9d1d9;
                  padding: 5px 10px; border-radius: 6px; font-size: 12px;
                  font-family: inherit; outline: none; }
  select:focus, input:focus { border-color: #58a6ff; }
  input[type=text] { width: 200px; }

  .tabs { display: flex; padding: 0 24px; border-bottom: 1px solid #21262d; }
  .tab { padding: 10px 14px; cursor: pointer; color: #8b949e; font-size: 12px;
         border-bottom: 2px solid transparent; transition: all 0.15s; user-select: none; }
  .tab:hover { color: #c9d1d9; }
  .tab.active { color: #58a6ff; border-bottom-color: #58a6ff; }
  .tab.active.cat-tab-gotcha      { color: #f85149; border-bottom-color: #f85149; }
  .tab.active.cat-tab-pattern     { color: #3fb950; border-bottom-color: #3fb950; }
  .tab.active.cat-tab-convention  { color: #58a6ff; border-bottom-color: #58a6ff; }
  .tab.active.cat-tab-decision    { color: #f0883e; border-bottom-color: #f0883e; }
  .tab.active.cat-tab-preference  { color: #d29922; border-bottom-color: #d29922; }
  .tab.active.cat-tab-dependency  { color: #8b5cf6; border-bottom-color: #8b5cf6; }
  .tab.active.cat-tab-optimization{ color: #ff7b72; border-bottom-color: #ff7b72; }

  .filter-bar { padding: 8px 24px; border-bottom: 1px solid #21262d;
                display: flex; gap: 8px; align-items: center; min-height: 38px; }

  .content { padding: 16px 24px; overflow-x: auto; }

  table { width: 100%; border-collapse: collapse; }
  th { text-align: left; padding: 8px 10px; color: #8b949e; font-size: 11px;
       font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;
       border-bottom: 1px solid #21262d; background: #0d1117; }
  td { padding: 9px 10px; border-bottom: 1px solid #161b22; vertical-align: top; }
  tr:hover td { background: #161b22; }

  .badge { display: inline-block; padding: 2px 7px; border-radius: 12px;
           font-size: 11px; font-weight: 500; white-space: nowrap; }
  .badge-active     { background: #1f3a1f; color: #3fb950; }
  .badge-stale      { background: #3a2f1a; color: #d29922; }
  .badge-archived   { background: #1e1e1e; color: #6e7681; }
  .badge-superseded { background: #2a1a2a; color: #8b5cf6; }

  .conf-bar { display: flex; align-items: center; gap: 5px; }
  .bar { height: 4px; background: #21262d; border-radius: 2px; width: 52px; flex-shrink: 0; }
  .bar-fill { height: 100%; border-radius: 2px; }
  .conf-val { color: #8b949e; font-size: 11px; min-width: 28px; }

  .cat-convention  { color: #58a6ff; }
  .cat-gotcha      { color: #f85149; }
  .cat-pattern     { color: #3fb950; }
  .cat-preference  { color: #d29922; }
  .cat-dependency  { color: #8b5cf6; }
  .cat-optimization{ color: #ff7b72; }
  .cat-decision    { color: #f0883e; }

  .text-muted { color: #6e7681; }
  .truncate { max-width: 440px; overflow: hidden; text-overflow: ellipsis;
              white-space: nowrap; cursor: pointer; }
  .truncate:hover { color: #58a6ff; }
  .expand-box { display: none; margin-top: 6px; background: #0d1117; padding: 10px;
                border-radius: 6px; white-space: pre-wrap; word-break: break-word;
                color: #c9d1d9; font-size: 12px; border: 1px solid #21262d;
                cursor: default; max-height: 300px; overflow-y: auto; }

  .actions { display: flex; gap: 6px; white-space: nowrap; }
  .btn { padding: 3px 8px; border-radius: 5px; font-size: 11px; cursor: pointer;
         border: 1px solid #30363d; background: #161b22; color: #8b949e;
         font-family: inherit; transition: all 0.15s; }
  .btn:hover { background: #21262d; color: #c9d1d9; }
  .btn-del  { border-color: #3a1a1a; color: #f85149; }
  .btn-del:hover  { background: #3a1a1a; }
  .btn-edit { border-color: #1a2a3a; color: #58a6ff; }
  .btn-edit:hover { background: #1a2a3a; }

  .pagination { display: flex; align-items: center; gap: 10px; padding: 12px 0 4px;
                color: #8b949e; font-size: 12px; }
  .pagination .btn { padding: 4px 12px; }
  .page-info { flex: 1; text-align: center; }

  .empty   { padding: 48px; text-align: center; color: #6e7681; }
  .err-msg { padding: 20px 24px; color: #f85149; font-size: 12px; }

  /* stats */
  .stats-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                gap: 12px; margin-bottom: 20px; }
  .stat-card { background: #161b22; border: 1px solid #21262d; border-radius: 8px;
               padding: 14px 16px; }
  .stat-card h4 { color: #8b949e; font-size: 11px; text-transform: uppercase;
                  letter-spacing: 0.06em; margin-bottom: 10px; }
  .stat-card .big { font-size: 24px; font-weight: 700; color: #58a6ff; }
  .stat-card .sub { color: #6e7681; font-size: 11px; margin-top: 2px; }
  .kv-list { display: flex; flex-direction: column; gap: 5px; }
  .kv { display: flex; justify-content: space-between; align-items: center; }
  .kv-key { color: #8b949e; font-size: 12px; }
  .kv-val { color: #c9d1d9; font-size: 12px; font-weight: 600; }
  .mini-bar-wrap { width: 100%; background: #21262d; border-radius: 2px;
                   height: 4px; margin-top: 3px; }
  .mini-bar { height: 100%; border-radius: 2px; background: #58a6ff; }
  .stats-section { margin-bottom: 20px; }
  .stats-section h3 { color: #8b949e; font-size: 11px; text-transform: uppercase;
                      letter-spacing: 0.06em; margin-bottom: 10px; padding-bottom: 6px;
                      border-bottom: 1px solid #21262d; }

  /* modal */
  .overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.6);
             display: none; align-items: center; justify-content: center; z-index: 100; }
  .overlay.open { display: flex; }
  .modal { background: #161b22; border: 1px solid #30363d; border-radius: 10px;
           padding: 24px; width: 520px; max-width: 95vw; max-height: 85vh; overflow-y: auto; }
  .modal h3 { color: #c9d1d9; font-size: 14px; margin-bottom: 16px; }
  .field { margin-bottom: 14px; }
  .field label { display: block; color: #8b949e; font-size: 11px; text-transform: uppercase;
                 letter-spacing: 0.05em; margin-bottom: 5px; }
  .field input, .field select, .field textarea {
    width: 100%; background: #0d1117; border: 1px solid #30363d; color: #c9d1d9;
    padding: 7px 10px; border-radius: 6px; font-size: 12px; font-family: inherit; outline: none; }
  .field input:focus, .field select:focus, .field textarea:focus { border-color: #58a6ff; }
  .field textarea { resize: vertical; min-height: 80px; }
  .modal-footer { display: flex; gap: 8px; justify-content: flex-end; margin-top: 18px; }
  .btn-primary { background: #1f3a5f; color: #58a6ff; border-color: #2d5a8a; }
  .btn-primary:hover { background: #2d5a8a; }
</style>
</head>
<body>

<header>
  <h1>🧠 Agent Memory</h1>
  <div class="subtitle">local vector store</div>
  <div class="spacer"></div>
  <div class="stats-bar" id="statsBar"></div>
</header>

<div class="toolbar">
  <select id="projectSelect" onchange="onProjectChange()">
    <option value="">all projects</option>
  </select>
  <input type="text" id="searchInput" placeholder="search..." oninput="debounceLoad()">
</div>

<div class="tabs" id="tabBar"></div>

<div class="filter-bar" id="filterBar"></div>
<div class="content" id="content"></div>

<!-- Edit modal -->
<div class="overlay" id="overlay" onclick="if(event.target===this)closeModal()">
  <div class="modal">
    <h3 id="modalTitle">Edit Memory</h3>
    <div id="modalBody"></div>
    <div class="modal-footer">
      <button class="btn" onclick="closeModal()">Cancel</button>
      <button class="btn btn-primary" onclick="saveModal()">Save</button>
    </div>
  </div>
</div>

<script>
const CATEGORIES = ['convention','gotcha','pattern','preference','dependency','optimization','decision'];
const TABS = ['memories', 'stats'];
const TAB_LABELS = { memories: 'Memories', stats: 'Stats' };
let currentTab = 'memories';
let statusFilter = '';
let debounceTimer;
let editRow = null;
let rowCache = [];
let currentPage = 0;
let totalRows = 0;
const PAGE_SIZE = 50;

function esc(s) {
  if (s == null) return '';
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

// ── Init ──────────────────────────────────────────────────────────────────────

async function init() {
  try {
    const res = await fetch('/api/projects');
    const projects = await res.json();
    const sel = document.getElementById('projectSelect');
    projects.forEach(p => {
      const o = document.createElement('option');
      o.value = p; o.textContent = p;
      sel.appendChild(o);
    });
    if (projects.length === 1) sel.value = projects[0];
  } catch(e) {}
  renderTabs();
  loadStats();
  renderFilterBar();
  loadTab();
}

function onProjectChange() { loadStats(); loadTab(); }
function debounceLoad() { clearTimeout(debounceTimer); debounceTimer = setTimeout(loadTab, 280); }

async function loadStats() {
  try {
    const p = document.getElementById('projectSelect').value;
    const res = await fetch('/api/stats?project=' + encodeURIComponent(p));
    const s = await res.json();
    document.getElementById('statsBar').innerHTML =
      `<div class="stat">total <span>${s.total}</span></div>` +
      `<div class="stat">active <span>${s.active}</span></div>`;
  } catch(e) {}
}

// ── Tabs & Filters ────────────────────────────────────────────────────────────

function renderTabs() {
  document.getElementById('tabBar').innerHTML = TABS.map(t =>
    `<div class="tab${t===currentTab?' active':''} cat-tab-${t}"
          onclick="switchTab('${t}')">${TAB_LABELS[t]}</div>`
  ).join('');
}

function switchTab(tab) {
  currentTab = tab;
  statusFilter = '';
  categoryFilter = '';
  currentPage = 0;
  document.getElementById('searchInput').value = '';
  renderTabs();
  renderFilterBar();
  loadTab();
}

function renderFilterBar() {
  const bar = document.getElementById('filterBar');
  if (currentTab === 'stats') { bar.innerHTML = ''; return; }
  bar.innerHTML = `
    <select onchange="setStatus(this.value)">
      <option value="">all status</option>
      <option>active</option><option>stale</option>
      <option>archived</option><option>superseded</option>
    </select>
    <select onchange="setCategoryFilter(this.value)">
      <option value="">all categories</option>
      ${CATEGORIES.map(c => `<option>${c}</option>`).join('')}
    </select>`;
}

let categoryFilter = '';
function setStatus(v) { statusFilter = v; currentPage = 0; loadTab(); }
function setCategoryFilter(v) { categoryFilter = v; currentPage = 0; loadTab(); }

function buildQuery() {
  const p = document.getElementById('projectSelect').value;
  const s = document.getElementById('searchInput').value;
  const params = new URLSearchParams();
  if (p) params.set('project', p);
  if (s) params.set('search', s);
  if (statusFilter) params.set('status', statusFilter);
  if (categoryFilter) params.set('category', categoryFilter);
  if (currentTab !== 'stats') {
    params.set('limit', PAGE_SIZE);
    params.set('offset', currentPage * PAGE_SIZE);
  }
  return params.toString();
}

// ── Load ──────────────────────────────────────────────────────────────────────

async function loadTab() {
  const content = document.getElementById('content');
  content.innerHTML = '<div class="empty" style="color:#6e7681">loading...</div>';
  try {
    if (currentTab === 'stats') { await loadStatsFull(); return; }
    const res = await fetch(`/api/memories?${buildQuery()}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const rows = await res.json();
    totalRows = parseInt(res.headers.get('X-Total-Count') || rows.length);
    rowCache = rows;
    content.innerHTML = renderMemoriesTable(rows) + renderPagination();
  } catch(e) {
    content.innerHTML = `<div class="err-msg">Error: ${esc(e.message)}</div>`;
  }
}

async function loadStatsFull() {
  const content = document.getElementById('content');
  try {
    const res = await fetch('/api/stats/full?' + buildQuery());
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    content.innerHTML = renderStats(await res.json());
  } catch(e) {
    content.innerHTML = `<div class="err-msg">Error: ${esc(e.message)}</div>`;
  }
}

// ── Render ────────────────────────────────────────────────────────────────────

function badge(s) { return `<span class="badge badge-${s}">${esc(s)}</span>`; }

function confBar(v) {
  const pct = Math.round(v * 100);
  const color = v >= 0.7 ? '#58a6ff' : v >= 0.4 ? '#d29922' : '#f85149';
  return `<div class="conf-bar">
    <div class="bar"><div class="bar-fill" style="width:${pct}%;background:${color}"></div></div>
    <span class="conf-val">${pct}%</span>
  </div>`;
}

function fmtDate(iso) {
  if (!iso) return '<span class="text-muted">—</span>';
  const d = new Date(iso);
  return `<span class="text-muted" title="${d.toISOString()}">${d.toLocaleDateString()} ${d.toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'})}</span>`;
}

function expandCell(idx, field, text) {
  if (!text) return '<span class="text-muted">—</span>';
  const id = `x-${idx}-${field}`;
  const safe = esc(text);
  const short = safe.length > 120 ? safe.slice(0, 120) + '…' : safe;
  return `<div>
    <div class="truncate" onclick="toggleExpand('${id}')">${short}</div>
    <div class="expand-box" id="${id}">${safe}</div>
  </div>`;
}

function toggleExpand(id) {
  const el = document.getElementById(id);
  el.style.display = el.style.display === 'block' ? 'none' : 'block';
}

function renderMemoriesTable(rows) {
  if (!rows.length) return '<div class="empty">no records found</div>';
  return `<table><thead><tr>
    <th>Category</th><th>Content</th><th>Confidence</th><th>Status</th><th>Last Ref.</th><th></th>
  </tr></thead><tbody>${rows.map((r, i) => `<tr>
    <td><span class="cat-${r.category}">${esc(r.category)}</span></td>
    <td>${expandCell(i, 'c', r.content)}</td>
    <td>${confBar(r.confidence)}</td>
    <td>${badge(r.status)}</td>
    <td>${fmtDate(r.last_referenced)}</td>
    <td><div class="actions">
      <button class="btn btn-edit" onclick="openEdit(${i})">edit</button>
      <button class="btn btn-del"  onclick="archiveMemory('${r.id}')">archive</button>
      <button class="btn btn-del"  onclick="deleteMemory('${r.id}')" title="permanent delete">✕</button>
    </div></td>
  </tr>`).join('')}</tbody></table>`;
}

function renderPagination() {
  const pages = Math.ceil(totalRows / PAGE_SIZE);
  if (pages <= 1) return '';
  const start = currentPage * PAGE_SIZE + 1;
  const end = Math.min((currentPage + 1) * PAGE_SIZE, totalRows);
  return `<div class="pagination">
    <button class="btn" onclick="gotoPage(${currentPage-1})" ${currentPage===0?'disabled':''}>← prev</button>
    <span class="page-info">${start}–${end} of ${totalRows}</span>
    <button class="btn" onclick="gotoPage(${currentPage+1})" ${currentPage>=pages-1?'disabled':''}>next →</button>
  </div>`;
}

function gotoPage(p) { currentPage = p; loadTab(); }

function renderStats(d) {
  const pct = (n, total) => total ? Math.round(n / total * 100) : 0;
  const kvRows = (obj, total) => Object.entries(obj).map(([k, v]) =>
    `<div class="kv"><span class="kv-key cat-${k}">${k}</span><span class="kv-val">${v}</span></div>
     <div class="mini-bar-wrap"><div class="mini-bar" style="width:${pct(v,total)}%"></div></div>`
  ).join('');
  const kvRowsPlain = (obj, total) => Object.entries(obj).map(([k, v]) =>
    `<div class="kv"><span class="kv-key">${k}</span><span class="kv-val">${v}</span></div>
     <div class="mini-bar-wrap"><div class="mini-bar" style="width:${pct(v,total)}%"></div></div>`
  ).join('');

  const confHtml = d.confidence.avg !== null ? `
    <div class="kv"><span class="kv-key">avg</span><span class="kv-val">${Math.round(d.confidence.avg*100)}%</span></div>
    <div class="kv"><span class="kv-key">min</span><span class="kv-val">${Math.round(d.confidence.min*100)}%</span></div>
    <div class="kv"><span class="kv-key">max</span><span class="kv-val">${Math.round(d.confidence.max*100)}%</span></div>
  ` : '<div class="text-muted">no active memories</div>';

  const topRefHtml = d.top_referenced.length ? `
    <table><thead><tr><th>Category</th><th>Content</th><th>Confidence</th><th>Last Ref.</th></tr></thead>
    <tbody>${d.top_referenced.map((r, i) => `<tr>
      <td><span class="cat-${r.category}">${esc(r.category)}</span></td>
      <td>${expandCell('sr'+i, 'c', r.content)}</td>
      <td>${confBar(r.confidence)}</td>
      <td>${fmtDate(r.last_referenced)}</td>
    </tr>`).join('')}</tbody></table>
  ` : '<div class="text-muted">none yet</div>';

  return `
  <div class="stats-grid">
    <div class="stat-card">
      <h4>Total Memories</h4>
      <div class="big">${d.total}</div>
      <div class="sub">${d.by_status.active||0} active · ${d.by_status.stale||0} stale · ${d.by_status.archived||0} archived · ${d.by_status.superseded||0} superseded</div>
    </div>
    <div class="stat-card">
      <h4>Confidence (active)</h4>
      <div class="kv-list">${confHtml}</div>
    </div>
    <div class="stat-card">
      <h4>Est. Tokens</h4>
      <div class="big">${d.estimated_tokens.toLocaleString()}</div>
      <div class="sub">≈ chars / 4</div>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:20px">
    <div class="stat-card">
      <h4>By Category</h4>
      <div class="kv-list">${Object.keys(d.by_category).length ? kvRows(d.by_category, d.total) : '<span class="text-muted">—</span>'}</div>
    </div>
    <div class="stat-card">
      <h4>By Status</h4>
      <div class="kv-list">${Object.keys(d.by_status).length ? kvRowsPlain(d.by_status, d.total) : '<span class="text-muted">—</span>'}</div>
    </div>
  </div>
  <div class="stats-section">
    <h3>Most Recently Referenced</h3>
    ${topRefHtml}
  </div>`;
}

// ── Edit Modal ────────────────────────────────────────────────────────────────

function openEdit(idx) {
  editRow = rowCache[idx];
  if (!editRow) return;
  document.getElementById('modalBody').innerHTML = `
    <div class="field"><label>Category</label>
      <select id="f-category">
        ${CATEGORIES.map(c => `<option ${editRow.category===c?'selected':''}>${c}</option>`).join('')}
      </select></div>
    <div class="field"><label>Content</label>
      <textarea id="f-content" rows="6">${esc(editRow.content||'')}</textarea></div>
    <div class="field"><label>Confidence (0–1)</label>
      <input id="f-confidence" type="number" min="0" max="1" step="0.05" value="${editRow.confidence||1}"></div>
    <div class="field"><label>Status</label>
      <select id="f-status">
        ${['active','stale','archived','superseded'].map(s =>
          `<option ${editRow.status===s?'selected':''}>${s}</option>`).join('')}
      </select></div>`;
  document.getElementById('overlay').classList.add('open');
}

function closeModal() {
  document.getElementById('overlay').classList.remove('open');
  editRow = null;
}

async function saveModal() {
  if (!editRow) return;
  const data = {
    category:   document.getElementById('f-category').value,
    content:    document.getElementById('f-content').value,
    confidence: parseFloat(document.getElementById('f-confidence').value),
    status:     document.getElementById('f-status').value,
  };
  try {
    const res = await fetch(`/api/memories/${editRow.id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });
    if (!res.ok) throw new Error(await res.text());
    closeModal(); loadTab(); loadStats();
  } catch(e) { alert('Save failed: ' + e.message); }
}

async function archiveMemory(id) {
  if (!confirm('Archive this memory?')) return;
  try {
    const res = await fetch(`/api/memories/${id}`, { method: 'DELETE' });
    if (!res.ok) throw new Error(await res.text());
    loadTab(); loadStats();
  } catch(e) { alert('Failed: ' + e.message); }
}

async function deleteMemory(id) {
  if (!confirm('Permanently delete this memory?')) return;
  try {
    const res = await fetch(`/api/memories/${id}?hard=true`, { method: 'DELETE' });
    if (!res.ok) throw new Error(await res.text());
    loadTab(); loadStats();
  } catch(e) { alert('Failed: ' + e.message); }
}

init();
</script>
</body>
</html>"""


@app.get("/")
def index():
    return render_template_string(HTML)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7337)))
    args = parser.parse_args()
    print(f"Agent Memory Viewer → http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)
