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

CATEGORIES = ["convention", "gotcha", "pattern", "preference", "decision"]

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


# ── Map (UMAP 2D projection) ─────────────────────────────────────────────────

_umap_cache = {}  # cache_key -> {hash, points, reducer, embeddings, mins, ranges, metas}


def _load_map_data(project, status):
    """Load embeddings + metas from DB, compute or reuse cached UMAP."""
    import hashlib
    import numpy as np

    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    clauses, params = ["status = %s"], [status]
    if project:
        clauses.append("project = %s")
        params.append(project)
    where = "WHERE " + " AND ".join(clauses)

    cur.execute(f"""
        SELECT id, category, content, confidence, status,
               content_embedding::text AS embedding_text
        FROM memories {where}
        ORDER BY created_at
    """, params)
    rows = cur.fetchall()
    conn.close()

    if len(rows) < 5:
        return None, "Need at least 5 memories for projection"

    embeddings = []
    metas = []
    for r in rows:
        vec_str = r["embedding_text"]
        if vec_str:
            vec = [float(x) for x in vec_str.strip("[]").split(",")]
            embeddings.append(vec)
            metas.append({
                "id": r["id"],
                "category": r["category"],
                "content": r["content"],
                "confidence": round(r["confidence"], 3),
            })

    if len(embeddings) < 5:
        return None, "Not enough embeddings"

    ids_hash = hashlib.md5("".join(m["id"] for m in metas).encode()).hexdigest()
    cache_key = f"{project}:{status}"

    if cache_key in _umap_cache and _umap_cache[cache_key]["hash"] == ids_hash:
        cache = _umap_cache[cache_key]
        # Update metas in cache (confidence may change)
        cache["metas"] = metas
        return cache, None

    import umap
    X = np.array(embeddings)
    n_neighbors = min(15, len(X) - 1)
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1,
                        metric="cosine", random_state=42)
    coords = reducer.fit_transform(X)

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1
    normed = (coords - mins) / ranges

    points = []
    for i, m in enumerate(metas):
        points.append({**m, "x": round(float(normed[i, 0]), 4), "y": round(float(normed[i, 1]), 4)})

    cache = {
        "hash": ids_hash, "points": points, "reducer": reducer,
        "embeddings": X, "mins": mins, "ranges": ranges, "metas": metas,
    }
    _umap_cache[cache_key] = cache
    return cache, None


@app.get("/api/map")
def api_map():
    project = request.args.get("project", "")
    status = request.args.get("status", "active")
    cache, err = _load_map_data(project, status)
    if err:
        return jsonify({"points": [], "error": err})
    return jsonify({"points": cache["points"]})


@app.post("/api/map/query")
def api_map_query():
    """Embed a query, project into UMAP space, return position + top-k similarities."""
    import numpy as np

    data = request.get_json() or {}
    query_text = data.get("query", "").strip()
    project = data.get("project", "")
    top_k = int(data.get("top_k", 10))

    if not query_text:
        return jsonify({"error": "query required"}), 400

    cache, err = _load_map_data(project, "active")
    if err:
        return jsonify({"error": err}), 400

    # Embed the query
    model = get_embed_model()
    query_vec = model.encode("search_query: " + query_text, normalize_embeddings=True)
    query_vec = np.array(query_vec, dtype=np.float32)

    # Cosine similarities (embeddings are already normalized)
    X = cache["embeddings"]
    sims = X @ query_vec

    # Approximate 2D position: weighted average of top-N neighbors' known positions
    # Much faster than reducer.transform() and visually equivalent
    top_n = min(8, len(sims))
    nn_indices = np.argsort(-sims)[:top_n]
    nn_sims = sims[nn_indices]
    # Sharpen weights so nearest neighbor dominates
    weights = nn_sims ** 4
    weights /= weights.sum()
    pts = cache["points"]
    qx = round(float(sum(weights[i] * pts[int(nn_indices[i])]["x"] for i in range(top_n))), 4)
    qy = round(float(sum(weights[i] * pts[int(nn_indices[i])]["y"] for i in range(top_n))), 4)

    # Top-k by similarity
    indices = np.argsort(-sims)[:top_k]
    top = []
    for idx in indices:
        m = cache["metas"][int(idx)]
        top.append({
            **m,
            "similarity": round(float(sims[idx]), 4),
            "x": cache["points"][int(idx)]["x"],
            "y": cache["points"][int(idx)]["y"],
        })

    # All similarities for radius visualization
    sim_list = [round(float(s), 4) for s in sims]

    return jsonify({
        "query": {"x": qx, "y": qy},
        "top": top,
        "similarities": sim_list,
    })


# ── Graph (force-directed) ────────────────────────────────────────────────────

@app.get("/api/graph")
def api_graph():
    """Return nodes + edges for force-directed graph. Edges where cosine sim >= threshold."""
    import numpy as np

    project = request.args.get("project", "")
    threshold = float(request.args.get("threshold", 0.75))

    cache, err = _load_map_data(project, "active")
    if err:
        return jsonify({"nodes": [], "edges": [], "error": err})

    X = cache["embeddings"]
    metas = cache["metas"]

    # Pairwise cosine similarities (embeddings are normalized)
    sims = X @ X.T

    edges = []
    for i in range(len(metas)):
        for j in range(i + 1, len(metas)):
            s = float(sims[i, j])
            if s >= threshold:
                edges.append({"source": i, "target": j, "sim": round(s, 4)})

    nodes = [{"id": m["id"], "category": m["category"], "content": m["content"],
              "confidence": m["confidence"]} for m in metas]

    return jsonify({"nodes": nodes, "edges": edges})


# ── Clusters (HDBSCAN) ────────────────────────────────────────────────────────

_cluster_cache = {}

@app.get("/api/clusters")
def api_clusters():
    import numpy as np
    from sklearn.cluster import HDBSCAN
    from collections import Counter
    import re

    project = request.args.get("project", "")
    min_size = int(request.args.get("min_size", 3))

    cache, err = _load_map_data(project, "active")
    if err:
        return jsonify({"clusters": [], "error": err})

    X = cache["embeddings"]
    metas = cache["metas"]

    cache_key = f"{project}:{min_size}:{len(metas)}"
    if cache_key in _cluster_cache:
        return jsonify(_cluster_cache[cache_key])

    # Use UMAP 2D projection + euclidean for clustering — raw 768-dim cosine
    # merges too aggressively in a single-domain corpus
    points = cache["points"]
    coords = np.array([[p["x"], p["y"]] for p in points])
    clusterer = HDBSCAN(min_cluster_size=min_size, metric="euclidean")
    labels = clusterer.fit_predict(coords)

    # Stop words for label extraction
    stop = {'the','a','an','is','are','was','were','be','been','in','on','at','to','for',
            'of','with','and','or','not','no','this','that','from','by','as','it','its',
            'der','die','das','ein','eine','und','oder','nicht','ist','mit','von','zu',
            'den','dem','des','im','am','auf','für','als','bei','nach','über','auch',
            'nur','noch','wird','hat','kann','sind','wie','wenn','aber','so','dann',
            'true','false','null','none','the','must','should','will','can','may'}

    def extract_words(texts):
        words = []
        for t in texts:
            words.extend(re.findall(r'[A-Za-zÄÖÜäöüß]{4,}', t.lower()))
        return Counter(w for w in words if w not in stop)

    def extract_label(texts, n=4):
        freq = extract_words(texts)
        return " · ".join(w for w, _ in freq.most_common(n))

    cluster_ids = sorted(set(labels))
    clusters = []
    for cid in cluster_ids:
        indices = [i for i, l in enumerate(labels) if l == cid]
        cat_counts = Counter(metas[i]["category"] for i in indices)

        # Compute cluster tightness (avg pairwise sim)
        if len(indices) > 1:
            sub = X[indices]
            sims = sub @ sub.T
            tightness = float((sims.sum() - len(indices)) / (len(indices) * (len(indices) - 1)))
        else:
            tightness = 1.0

        texts = [metas[i]["content"] for i in indices]
        label = extract_label(texts)
        word_freq = extract_words(texts)
        top_words = [{"word": w, "count": c} for w, c in word_freq.most_common(20)]
        dominant_cat = cat_counts.most_common(1)[0][0] if cat_counts else "convention"

        cluster = {
            "id": int(cid),
            "label": label,
            "size": len(indices),
            "tightness": round(tightness, 3),
            "categories": dict(cat_counts),
            "dominant_category": dominant_cat,
            "words": top_words,
            "memories": [{
                "id": metas[i]["id"],
                "category": metas[i]["category"],
                "content": metas[i]["content"],
                "confidence": metas[i]["confidence"],
            } for i in indices],
        }
        if cid == -1:
            cluster["label"] = "unclustered"
        clusters.append(cluster)

    # Sort: noise last, then by size desc
    clusters.sort(key=lambda c: (c["id"] == -1, -c["size"]))

    result = {"clusters": clusters, "total_memories": len(metas),
              "clustered": sum(c["size"] for c in clusters if c["id"] != -1),
              "noise": sum(1 for l in labels if l == -1)}
    _cluster_cache[cache_key] = result
    return jsonify(result)


# ── Timeline ─────────────────────────────────────────────────────────────────

@app.get("/api/timeline")
def api_timeline():
    project = request.args.get("project", "")
    status = request.args.get("status", "")

    clauses, params = [], []
    if project: clauses.append("project = %s"); params.append(project)
    if status:  clauses.append("status = %s");  params.append(status)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cur.execute(f"""
        SELECT id, category, content, confidence, status, created_at, last_referenced
        FROM memories {where}
        ORDER BY created_at ASC
    """, params)
    rows = [serialize(r) for r in cur.fetchall()]
    conn.close()

    return jsonify({"memories": rows})


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

  /* map */
  .map-wrap { position: relative; width: 100%; height: calc(100vh - 220px); min-height: 400px; }
  .map-canvas { width: 100%; height: 100%; cursor: crosshair; }
  .map-tooltip { position: absolute; pointer-events: none; background: #161b22; border: 1px solid #30363d;
                 border-radius: 8px; padding: 10px 12px; font-size: 12px; max-width: 360px;
                 box-shadow: 0 4px 16px rgba(0,0,0,0.4); z-index: 50; display: none;
                 line-height: 1.5; }
  .map-tooltip .tt-cat { font-weight: 600; margin-bottom: 4px; }
  .map-tooltip .tt-content { color: #8b949e; word-break: break-word; }
  .map-tooltip .tt-conf { color: #6e7681; font-size: 11px; margin-top: 4px; }
  .map-legend { position: absolute; bottom: 12px; left: 12px; display: flex; gap: 12px;
                background: rgba(13,17,23,0.85); padding: 6px 12px; border-radius: 6px;
                border: 1px solid #21262d; font-size: 11px; align-items: center; }
  .map-legend-item { display: flex; align-items: center; gap: 5px; cursor: pointer; user-select: none; }
  .map-legend-dot { width: 8px; height: 8px; border-radius: 50%; }
  .map-legend-item.dimmed { opacity: 0.3; }
  .map-loading { position: absolute; inset: 0; display: flex; align-items: center;
                 justify-content: center; color: #6e7681; font-size: 13px; }

  .map-container { display: flex; height: calc(100vh - 220px); min-height: 400px; }
  .map-main { flex: 1; position: relative; min-width: 0; }
  .map-drawer { width: 360px; flex-shrink: 0; border-left: 1px solid #21262d;
                background: #0d1117; display: flex; flex-direction: column; overflow: hidden; }
  .map-drawer-header { padding: 12px 14px; border-bottom: 1px solid #21262d; }
  .map-drawer-header input { width: 100%; background: #161b22; border: 1px solid #30363d;
                              color: #c9d1d9; padding: 8px 10px; border-radius: 6px;
                              font-size: 12px; font-family: inherit; outline: none; }
  .map-drawer-header input:focus { border-color: #58a6ff; }
  .map-drawer-header input::placeholder { color: #484f58; }
  .map-drawer-results { flex: 1; overflow-y: auto; padding: 0; }
  .map-drawer-empty { padding: 24px 14px; color: #484f58; font-size: 12px; text-align: center; }
  .map-result { padding: 10px 14px; border-bottom: 1px solid #161b22; cursor: pointer;
                transition: background 0.1s; }
  .map-result:hover { background: #161b22; }
  .map-result.highlighted { background: #1a2332; }
  .map-result-head { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
  .map-result-sim { font-size: 11px; font-weight: 600; color: #58a6ff; min-width: 36px; }
  .map-result-cat { font-size: 11px; }
  .map-result-conf { font-size: 10px; color: #484f58; margin-left: auto; }
  .map-result-content { font-size: 11px; color: #8b949e; line-height: 1.5;
                        display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical;
                        overflow: hidden; word-break: break-word; }
  .map-result-content.expanded { -webkit-line-clamp: unset; }
  .sim-threshold { font-size: 10px; color: #484f58; padding: 6px 14px; border-bottom: 1px solid #161b22;
                   display: flex; align-items: center; gap: 6px; }
  .sim-threshold .dot-inline { width: 6px; height: 6px; border-radius: 50;
                               background: #58a6ff; display: inline-block; }

  /* clusters */
  .cluster-list { display: flex; flex-direction: column; gap: 10px; }
  .cluster-card { background: #161b22; border: 1px solid #21262d; border-radius: 8px;
                  overflow: hidden; }
  .cluster-header { padding: 12px 16px; cursor: pointer; display: flex; align-items: center;
                    gap: 12px; transition: background 0.1s; user-select: none; }
  .cluster-header:hover { background: #1c2129; }
  .cluster-size { font-size: 20px; font-weight: 700; color: #58a6ff; min-width: 36px; text-align: right; }
  .cluster-info { flex: 1; min-width: 0; }
  .cluster-label { font-size: 13px; color: #c9d1d9; margin-bottom: 3px; }
  .cluster-meta { font-size: 11px; color: #6e7681; display: flex; gap: 10px; flex-wrap: wrap; }
  .cluster-cats { display: flex; gap: 4px; }
  .cluster-cat-pill { font-size: 10px; padding: 1px 6px; border-radius: 8px; }
  .cluster-tightness { font-size: 11px; color: #484f58; }
  .cluster-body { display: none; border-top: 1px solid #21262d; padding: 0; }
  .cluster-body.open { display: block; }
  .cluster-mem { padding: 8px 16px; border-bottom: 1px solid #161b22; font-size: 12px;
                 color: #8b949e; line-height: 1.5; display: flex; gap: 10px; }
  .cluster-mem:last-child { border-bottom: none; }
  .cluster-mem-cat { font-size: 11px; min-width: 70px; flex-shrink: 0; }
  .cluster-mem-content { word-break: break-word; }
  .cluster-noise { color: #484f58; font-style: italic; }
  .cluster-summary { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
                     gap: 10px; margin-bottom: 16px; }
  .cluster-stat { background: #161b22; border: 1px solid #21262d; border-radius: 8px;
                  padding: 12px 14px; }
  .cluster-stat h4 { font-size: 10px; text-transform: uppercase; letter-spacing: 0.06em;
                     color: #6e7681; margin-bottom: 6px; }
  .cluster-stat .big { font-size: 22px; font-weight: 700; color: #58a6ff; }
  .cluster-stat .sub { font-size: 11px; color: #484f58; margin-top: 2px; }
  .cluster-chevron { color: #484f58; transition: transform 0.15s; font-size: 14px; }
  .cluster-chevron.open { transform: rotate(90deg); }
  .cluster-chart { position: relative; min-height: 350px; height: calc(50vh - 80px);
                   margin: 0 -24px 20px; padding: 0; width: calc(100% + 48px);
                   background: #161b22; border-top: 1px solid #21262d;
                   border-bottom: 1px solid #21262d; overflow: hidden; }
  .cluster-chart canvas { width: 100%; height: 100%; }
  .cluster-controls { display: flex; gap: 12px; align-items: center; margin-bottom: 12px; }
  .cluster-controls label { color: #8b949e; font-size: 11px; display: flex; align-items: center; gap: 6px; }
  .cluster-controls .val { color: #58a6ff; font-weight: 600; min-width: 16px; }

  /* timeline */
  .tl-container { padding: 0; }
  .tl-controls { display: flex; gap: 10px; align-items: center; margin-bottom: 16px; flex-wrap: wrap; }
  .tl-controls label { color: #8b949e; font-size: 11px; display: flex; align-items: center; gap: 6px; }
  .tl-chart { position: relative; height: calc(50vh - 100px); min-height: 240px;
              margin: 0 -24px 20px; width: calc(100% + 48px);
              background: #161b22; border-top: 1px solid #21262d;
              border-bottom: 1px solid #21262d; overflow: hidden; }
  .tl-canvas { width: 100%; height: 100%; }
  .tl-day-list { display: flex; flex-direction: column; gap: 2px; }
  .tl-day { border-bottom: 1px solid #161b22; }
  .tl-day-header { padding: 10px 0 6px; font-size: 12px; font-weight: 600; color: #8b949e;
                   border-bottom: 1px solid #21262d; margin-bottom: 4px; display: flex;
                   align-items: center; gap: 8px; cursor: pointer; user-select: none; }
  .tl-day-count { font-size: 11px; color: #484f58; font-weight: 400; }
  .tl-day-items { display: none; }
  .tl-day-items.open { display: block; }
  .tl-item { padding: 6px 0 6px 16px; font-size: 12px; color: #8b949e; line-height: 1.5;
             display: flex; gap: 10px; border-left: 2px solid #21262d; margin-left: 4px; }
  .tl-item-time { font-size: 11px; color: #484f58; min-width: 44px; flex-shrink: 0; }
  .tl-item-cat { font-size: 11px; min-width: 70px; flex-shrink: 0; }
  .tl-item-content { word-break: break-word; }
  .tl-bar-label { font-size: 10px; color: #484f58; }

  /* graph */
  .graph-container { display: flex; height: calc(100vh - 220px); min-height: 400px; }
  .graph-main { flex: 1; position: relative; min-width: 0; }
  .graph-canvas { width: 100%; height: 100%; cursor: grab; }
  .graph-canvas:active { cursor: grabbing; }
  .graph-controls { position: absolute; top: 12px; right: 12px; display: flex; flex-direction: column;
                    gap: 6px; background: rgba(13,17,23,0.9); padding: 10px 12px; border-radius: 8px;
                    border: 1px solid #21262d; font-size: 11px; z-index: 10; }
  .graph-controls label { color: #8b949e; display: flex; align-items: center; gap: 6px; }
  .graph-controls input[type=range] { width: 100px; accent-color: #58a6ff; }
  .graph-controls .val { color: #58a6ff; font-weight: 600; min-width: 28px; text-align: right; }
  .graph-stats { position: absolute; top: 12px; left: 12px; background: rgba(13,17,23,0.9);
                 padding: 8px 12px; border-radius: 8px; border: 1px solid #21262d;
                 font-size: 11px; color: #6e7681; }
  .graph-stats span { color: #58a6ff; font-weight: 600; }

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
const CATEGORIES = ['convention','gotcha','pattern','preference','decision'];
const TABS = ['memories', 'stats', 'map', 'graph', 'clusters', 'timeline'];
const TAB_LABELS = { memories: 'Memories', stats: 'Stats', map: 'Map', graph: 'Graph', clusters: 'Clusters', timeline: 'Timeline' };
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
  // Restore tab from URL
  const urlTab = new URLSearchParams(window.location.search).get('tab');
  if (urlTab && TABS.includes(urlTab)) currentTab = urlTab;

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
  const url = new URL(window.location);
  url.searchParams.set('tab', tab);
  history.replaceState(null, '', url);
  renderTabs();
  renderFilterBar();
  loadTab();
}

function renderFilterBar() {
  const bar = document.getElementById('filterBar');
  if (currentTab !== 'memories') { bar.innerHTML = ''; return; }
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
    if (currentTab === 'stats')    { await loadStatsFull(); return; }
    if (currentTab === 'map')      { await loadMap(); return; }
    if (currentTab === 'graph')    { await loadGraph(); return; }
    if (currentTab === 'clusters') { await loadClusters(); return; }
    if (currentTab === 'timeline') { await loadTimeline(); return; }
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

// ── Map ───────────────────────────────────────────────────────────────────────

const CAT_COLORS = {
  convention: '#58a6ff', gotcha: '#f85149', pattern: '#3fb950',
  preference: '#d29922', decision: '#f0883e'
};
let mapPoints = [];
let mapHidden = new Set();
let mapHover = -1;
let mapCanvas, mapCtx, mapTooltip;
let mapQuery = null;       // {x, y} — projected query point
let mapQuerySims = [];     // per-point similarities to current query
let mapQueryTop = [];      // top-k results
let mapHighlightIdx = -1;  // hovered result in drawer

async function loadMap() {
  const content = document.getElementById('content');
  content.innerHTML = `
    <div class="map-container">
      <div class="map-main">
        <div class="map-loading" id="mapLoading">Computing UMAP projection…</div>
        <canvas class="map-canvas" id="mapCanvas"></canvas>
        <div class="map-tooltip" id="mapTooltip">
          <div class="tt-cat"></div>
          <div class="tt-content"></div>
          <div class="tt-conf"></div>
        </div>
        <div class="map-legend" id="mapLegend"></div>
      </div>
      <div class="map-drawer">
        <div class="map-drawer-header">
          <div style="display:flex;gap:6px;align-items:center">
            <input type="text" id="mapQueryInput" placeholder="Type a query to explore…"
                   onkeydown="if(event.key==='Enter')runMapQuery()" style="flex:1">
            <button class="btn" id="mapResetBtn" onclick="resetMapQuery()" style="display:none;padding:5px 8px;font-size:11px" title="Reset">✕</button>
          </div>
        </div>
        <div class="map-drawer-results" id="mapDrawerResults">
          <div class="map-drawer-empty">Enter a query to see where it lands in the embedding space and find the most similar memories.</div>
        </div>
      </div>
    </div>`;

  mapCanvas = document.getElementById('mapCanvas');
  mapCtx = mapCanvas.getContext('2d');
  mapTooltip = document.getElementById('mapTooltip');

  // Legend
  const legend = document.getElementById('mapLegend');
  legend.innerHTML = CATEGORIES.map(c =>
    `<div class="map-legend-item" data-cat="${c}" onclick="toggleMapCat('${c}')">
       <div class="map-legend-dot" style="background:${CAT_COLORS[c]}"></div>
       <span>${c}</span>
     </div>`
  ).join('') + `<div style="color:#484f58;margin-left:8px;border-left:1px solid #21262d;padding-left:10px" id="mapPointCount"></div>`;

  // Fetch
  const p = document.getElementById('projectSelect').value;
  const params = new URLSearchParams();
  if (p) params.set('project', p);
  params.set('status', 'active');

  try {
    const res = await fetch('/api/map?' + params);
    const data = await res.json();
    document.getElementById('mapLoading').style.display = 'none';

    if (data.error) {
      document.getElementById('mapLoading').style.display = 'flex';
      document.getElementById('mapLoading').textContent = data.error;
      return;
    }

    mapPoints = data.points;
    document.getElementById('mapPointCount').textContent = mapPoints.length + ' points';
    resizeMap();
    drawMap();

    mapCanvas.onmousemove = onMapMouseMove;
    mapCanvas.onmouseleave = () => { mapHover = -1; drawMap(); mapTooltip.style.display = 'none'; };
    window.addEventListener('resize', () => { resizeMap(); drawMap(); });
  } catch(e) {
    document.getElementById('mapLoading').textContent = 'Failed to load map: ' + e.message;
  }
}

function resizeMap() {
  const wrap = mapCanvas.parentElement;
  const dpr = window.devicePixelRatio || 1;
  mapCanvas.width = wrap.clientWidth * dpr;
  mapCanvas.height = wrap.clientHeight * dpr;
  mapCanvas.style.width = wrap.clientWidth + 'px';
  mapCanvas.style.height = wrap.clientHeight + 'px';
  mapCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function toScreen(nx, ny) {
  const w = mapCanvas.clientWidth;
  const h = mapCanvas.clientHeight;
  const pad = 40;
  return [pad + nx * (w - pad * 2), pad + ny * (h - pad * 2)];
}

function drawMap() {
  const w = mapCanvas.clientWidth;
  const h = mapCanvas.clientHeight;
  mapCtx.clearRect(0, 0, w, h);

  const hasQuery = mapQuery !== null;
  const topIds = new Set(mapQueryTop.map(t => t.id));

  // Draw 0.85 similarity radius (convex hull of matching points)
  if (hasQuery && mapQuerySims.length) {
    const [qx, qy] = toScreen(mapQuery.x, mapQuery.y);
    // Find max screen distance among points with sim >= 0.85
    let maxDist = 0;
    for (let i = 0; i < mapPoints.length; i++) {
      if (mapQuerySims[i] >= 0.85) {
        const [px, py] = toScreen(mapPoints[i].x, mapPoints[i].y);
        const d = Math.hypot(px - qx, py - qy);
        if (d > maxDist) maxDist = d;
      }
    }
    if (maxDist > 0) {
      mapCtx.beginPath();
      mapCtx.arc(qx, qy, maxDist + 10, 0, Math.PI * 2);
      mapCtx.fillStyle = 'rgba(88, 166, 255, 0.04)';
      mapCtx.fill();
      mapCtx.setLineDash([6, 4]);
      mapCtx.strokeStyle = 'rgba(88, 166, 255, 0.25)';
      mapCtx.lineWidth = 1;
      mapCtx.stroke();
      mapCtx.setLineDash([]);
    }
  }

  // Draw points
  for (let i = 0; i < mapPoints.length; i++) {
    const p = mapPoints[i];
    if (mapHidden.has(p.category)) continue;
    const [px, py] = toScreen(p.x, p.y);

    let r = 4.5, alpha = 0.7;

    if (hasQuery) {
      const isTop = topIds.has(p.id);
      const isAbove = mapQuerySims[i] >= 0.85;
      if (!isTop && !isAbove) { alpha = 0.12; r = 3; }
      else if (isAbove) { alpha = 0.55; r = 4; }
      if (isTop) { alpha = 0.9; r = 5.5; }
    }

    if (i === mapHover) { r = 7; alpha = 1.0; }
    if (i === mapHighlightIdx) { r = 8; alpha = 1.0; }

    mapCtx.beginPath();
    mapCtx.arc(px, py, r, 0, Math.PI * 2);
    mapCtx.fillStyle = hexAlpha(CAT_COLORS[p.category] || '#8b949e', alpha);
    mapCtx.fill();

    if (i === mapHover || i === mapHighlightIdx) {
      mapCtx.strokeStyle = '#c9d1d9';
      mapCtx.lineWidth = 1.5;
      mapCtx.stroke();
    }
  }

  // Draw query point
  if (hasQuery) {
    const [qx, qy] = toScreen(mapQuery.x, mapQuery.y);
    // Outer glow
    mapCtx.beginPath();
    mapCtx.arc(qx, qy, 12, 0, Math.PI * 2);
    mapCtx.fillStyle = 'rgba(88, 166, 255, 0.15)';
    mapCtx.fill();
    // Inner dot
    mapCtx.beginPath();
    mapCtx.arc(qx, qy, 5, 0, Math.PI * 2);
    mapCtx.fillStyle = '#fff';
    mapCtx.fill();
    mapCtx.strokeStyle = '#58a6ff';
    mapCtx.lineWidth = 2;
    mapCtx.stroke();
    // Crosshair
    mapCtx.beginPath();
    mapCtx.moveTo(qx - 18, qy); mapCtx.lineTo(qx - 8, qy);
    mapCtx.moveTo(qx + 8, qy); mapCtx.lineTo(qx + 18, qy);
    mapCtx.moveTo(qx, qy - 18); mapCtx.lineTo(qx, qy - 8);
    mapCtx.moveTo(qx, qy + 8); mapCtx.lineTo(qx, qy + 18);
    mapCtx.strokeStyle = 'rgba(88, 166, 255, 0.5)';
    mapCtx.lineWidth = 1;
    mapCtx.stroke();
  }
}

function hexAlpha(hex, a) {
  const r = parseInt(hex.slice(1,3),16);
  const g = parseInt(hex.slice(3,5),16);
  const b = parseInt(hex.slice(5,7),16);
  return `rgba(${r},${g},${b},${a})`;
}

function onMapMouseMove(e) {
  const rect = mapCanvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  const w = mapCanvas.clientWidth;
  const h = mapCanvas.clientHeight;
  const pad = 40;

  let closest = -1, closestDist = 20; // 20px hit radius
  for (let i = 0; i < mapPoints.length; i++) {
    const p = mapPoints[i];
    if (mapHidden.has(p.category)) continue;
    const px = pad + p.x * (w - pad * 2);
    const py = pad + p.y * (h - pad * 2);
    const d = Math.hypot(mx - px, my - py);
    if (d < closestDist) { closestDist = d; closest = i; }
  }

  if (closest !== mapHover) {
    mapHover = closest;
    drawMap();
  }

  if (closest >= 0) {
    const p = mapPoints[closest];
    const tt = mapTooltip;
    tt.querySelector('.tt-cat').innerHTML =
      `<span class="cat-${p.category}">${p.category}</span>`;
    tt.querySelector('.tt-content').textContent = p.content;
    tt.querySelector('.tt-conf').textContent = `confidence: ${Math.round(p.confidence * 100)}%`;
    tt.style.display = 'block';

    // Position tooltip
    let tx = e.clientX - rect.left + 14;
    let ty = e.clientY - rect.top - 10;
    if (tx + 370 > w) tx = e.clientX - rect.left - 374;
    if (ty + 100 > h) ty = h - 110;
    if (ty < 0) ty = 4;
    tt.style.left = tx + 'px';
    tt.style.top = ty + 'px';
  } else {
    mapTooltip.style.display = 'none';
  }
}

function toggleMapCat(cat) {
  if (mapHidden.has(cat)) mapHidden.delete(cat);
  else mapHidden.add(cat);

  document.querySelectorAll('.map-legend-item').forEach(el => {
    el.classList.toggle('dimmed', mapHidden.has(el.dataset.cat));
  });
  drawMap();
}

// ── Map Query ─────────────────────────────────────────────────────────────────

let mapQueryTimer;

async function runMapQuery() {
  const input = document.getElementById('mapQueryInput');
  const query = input.value.trim();
  const results = document.getElementById('mapDrawerResults');

  if (!query) { resetMapQuery(); return; }

  results.innerHTML = '<div class="map-drawer-empty">Embedding query…</div>';
  document.getElementById('mapResetBtn').style.display = 'block';

  const p = document.getElementById('projectSelect').value;
  try {
    const res = await fetch('/api/map/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, project: p, top_k: 10 })
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();

    mapQuery = data.query;
    mapQuerySims = data.similarities;
    mapQueryTop = data.top;
    mapHighlightIdx = -1;
    drawMap();
    renderMapResults(data.top);
  } catch(e) {
    results.innerHTML = `<div class="map-drawer-empty" style="color:#f85149">Error: ${esc(e.message)}</div>`;
  }
}

function resetMapQuery() {
  mapQuery = null; mapQuerySims = []; mapQueryTop = []; mapHighlightIdx = -1;
  document.getElementById('mapQueryInput').value = '';
  document.getElementById('mapResetBtn').style.display = 'none';
  document.getElementById('mapDrawerResults').innerHTML =
    '<div class="map-drawer-empty">Enter a query to see where it lands in the embedding space and find the most similar memories.</div>';
  drawMap();
}

function renderMapResults(top) {
  const results = document.getElementById('mapDrawerResults');
  if (!top.length) {
    results.innerHTML = '<div class="map-drawer-empty">No results found.</div>';
    return;
  }

  const aboveCount = mapQuerySims.filter(s => s >= 0.85).length;

  results.innerHTML =
    `<div class="sim-threshold">
       <span style="color:#58a6ff;font-weight:600">${aboveCount}</span> memor${aboveCount === 1 ? 'y' : 'ies'} within 0.85 cosine similarity
     </div>` +
    top.map((r, i) => {
      const simPct = Math.round(r.similarity * 100);
      const simColor = r.similarity >= 0.85 ? '#58a6ff' : r.similarity >= 0.7 ? '#d29922' : '#6e7681';
      return `<div class="map-result" data-idx="${i}"
                   onmouseenter="highlightMapResult(${i})"
                   onmouseleave="highlightMapResult(-1)"
                   onclick="toggleResultExpand(this)">
        <div class="map-result-head">
          <span class="map-result-sim" style="color:${simColor}">${simPct}%</span>
          <span class="map-result-cat cat-${r.category}">${r.category}</span>
          <span class="map-result-conf">${Math.round(r.confidence * 100)}% conf</span>
        </div>
        <div class="map-result-content">${esc(r.content)}</div>
      </div>`;
    }).join('');
}

function highlightMapResult(drawerIdx) {
  if (drawerIdx < 0) {
    mapHighlightIdx = -1;
  } else {
    // Find the point index in mapPoints matching this top result's id
    const targetId = mapQueryTop[drawerIdx]?.id;
    mapHighlightIdx = mapPoints.findIndex(p => p.id === targetId);
  }
  drawMap();
}

function toggleResultExpand(el) {
  const c = el.querySelector('.map-result-content');
  c.classList.toggle('expanded');
}

// ── Clusters ───────────────────────────────────────────────────────────────────

let clusterMinSize = 3;
let clusterData = null;

async function loadClusters() {
  const content = document.getElementById('content');
  content.innerHTML = '<div class="empty" style="color:#6e7681">Computing clusters…</div>';

  const p = document.getElementById('projectSelect').value;
  const params = new URLSearchParams();
  if (p) params.set('project', p);
  params.set('min_size', clusterMinSize);

  try {
    const res = await fetch('/api/clusters?' + params);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    clusterData = await res.json();
    content.innerHTML = renderClusters(clusterData);
    drawClusterChart(clusterData);
  } catch(e) {
    content.innerHTML = `<div class="err-msg">Error: ${esc(e.message)}</div>`;
  }
}

function onClusterMinSizeChange(val) {
  clusterMinSize = parseInt(val);
  document.getElementById('clMinSizeVal').textContent = clusterMinSize;
  // Clear cache and reload
  loadClusters();
}

function renderClusters(data) {
  const pct = (n) => data.total_memories ? Math.round(n / data.total_memories * 100) : 0;

  const realClusters = data.clusters.filter(c => c.id !== -1);
  const noiseCluster = data.clusters.find(c => c.id === -1);

  let html = `<div class="cluster-controls">
    <label>Min cluster size
      <input type="range" id="clMinSizeSlider" min="2" max="10" step="1"
             value="${clusterMinSize}" oninput="onClusterMinSizeChange(this.value)">
      <span class="val" id="clMinSizeVal">${clusterMinSize}</span>
    </label>
  </div>`;

  html += `<div class="cluster-chart">
    <canvas id="clusterCanvas"></canvas>
  </div>`;

  html += `<div class="cluster-summary">
    <div class="cluster-stat"><h4>Clusters</h4><div class="big">${realClusters.length}</div></div>
    <div class="cluster-stat"><h4>Clustered</h4><div class="big">${data.clustered}</div>
      <div class="sub">${pct(data.clustered)}% of memories</div></div>
    <div class="cluster-stat"><h4>Unclustered</h4><div class="big">${data.noise}</div>
      <div class="sub">${pct(data.noise)}% noise</div></div>
    <div class="cluster-stat"><h4>Avg Size</h4><div class="big">${realClusters.length ? Math.round(data.clustered / realClusters.length) : 0}</div></div>
  </div>`;

  html += '<div class="cluster-list">';

  for (const c of data.clusters) {
    const isNoise = c.id === -1;
    const catPills = Object.entries(c.categories).sort((a,b) => b[1] - a[1]).map(([cat, n]) =>
      `<span class="cluster-cat-pill" style="background:${hexAlpha(CAT_COLORS[cat]||'#8b949e',0.15)};color:${CAT_COLORS[cat]||'#8b949e'}">${cat} ${n}</span>`
    ).join('');

    html += `<div class="cluster-card${isNoise ? ' cluster-noise' : ''}">
      <div class="cluster-header" onclick="toggleCluster(this)">
        <div class="cluster-size" style="${isNoise ? 'color:#484f58' : ''}">${c.size}</div>
        <div class="cluster-info">
          <div class="cluster-label">${isNoise ? '⊘ unclustered' : esc(c.label)}</div>
          <div class="cluster-meta">
            <div class="cluster-cats">${catPills}</div>
            ${!isNoise ? `<span class="cluster-tightness">tightness: ${Math.round(c.tightness * 100)}%</span>` : ''}
          </div>
        </div>
        <span class="cluster-chevron">›</span>
      </div>
      <div class="cluster-body">
        ${c.memories.map(m => `<div class="cluster-mem">
          <span class="cluster-mem-cat cat-${m.category}">${m.category}</span>
          <span class="cluster-mem-content">${esc(m.content)}</span>
        </div>`).join('')}
      </div>
    </div>`;
  }

  html += '</div>';
  return html;
}

function toggleCluster(header) {
  const body = header.nextElementSibling;
  const chevron = header.querySelector('.cluster-chevron');
  body.classList.toggle('open');
  chevron.classList.toggle('open');
}

function drawClusterChart(data) {
  const canvas = document.getElementById('clusterCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const wrap = canvas.parentElement;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = wrap.clientWidth * dpr;
  canvas.height = wrap.clientHeight * dpr;
  canvas.style.width = wrap.clientWidth + 'px';
  canvas.style.height = wrap.clientHeight + 'px';
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const w = wrap.clientWidth, h = wrap.clientHeight;
  const clusters = data.clusters.filter(c => c.id !== -1 && c.words && c.words.length);
  if (!clusters.length) return;

  // Collect all words across clusters, pick top ones per cluster
  // Scale font size by word count, color by dominant category
  const allWords = [];
  const maxClusterSize = Math.max(...clusters.map(c => c.size));

  for (const c of clusters) {
    const color = CAT_COLORS[c.dominant_category] || '#8b949e';
    const clusterWeight = c.size / maxClusterSize;
    for (let i = 0; i < Math.min(c.words.length, 8); i++) {
      const ww = c.words[i];
      const importance = ww.count * clusterWeight;
      allWords.push({
        text: ww.word,
        count: ww.count,
        importance: importance,
        color: color,
        cluster: c.label,
        clusterId: c.id,
      });
    }
  }

  // Deduplicate: keep highest importance version of each word
  const wordMap = {};
  for (const w of allWords) {
    if (!wordMap[w.text] || w.importance > wordMap[w.text].importance) {
      wordMap[w.text] = w;
    }
  }
  const words = Object.values(wordMap).sort((a, b) => b.importance - a.importance);

  // Font size scaling
  const maxImp = words.length ? words[0].importance : 1;
  const minFont = 11, maxFont = 48;

  // Place words using spiral placement
  const placed = []; // [{x, y, w, h, word}]
  const cx = w / 2, cy = h / 2;

  function fontSize(word) {
    return minFont + (word.importance / maxImp) * (maxFont - minFont);
  }

  function measureWord(word) {
    const size = fontSize(word);
    ctx.font = `600 ${size}px monospace`;
    const metrics = ctx.measureText(word.text);
    return { w: metrics.width + 8, h: size * 1.2 + 4 };
  }

  function overlaps(x, y, ww, hh) {
    for (const p of placed) {
      if (x < p.x + p.w && x + ww > p.x && y < p.y + p.h && y + hh > p.y) return true;
    }
    return false;
  }

  // Spiral placement
  for (const word of words.slice(0, 80)) {
    const m = measureWord(word);
    let angle = 0;
    let radius = 0;
    let found = false;

    for (let step = 0; step < 1200; step++) {
      const x = cx + Math.cos(angle) * radius - m.w / 2;
      const y = cy + Math.sin(angle) * radius - m.h / 2;

      if (x >= 4 && x + m.w <= w - 4 && y >= 4 && y + m.h <= h - 4 && !overlaps(x, y, m.w, m.h)) {
        placed.push({ x, y, w: m.w, h: m.h, word });
        found = true;
        break;
      }
      angle += 0.35;
      radius += 0.6;
    }
  }

  // Draw
  for (const p of placed) {
    const size = fontSize(p.word);
    ctx.font = `600 ${size}px monospace`;
    ctx.fillStyle = hexAlpha(p.word.color, 0.35 + (p.word.importance / maxImp) * 0.55);
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(p.word.text, p.x + 4, p.y + 2);
  }
}

// ── Timeline ───────────────────────────────────────────────────────────────────

let tlCanvas, tlCtx, tlData = [], tlHiddenCats = new Set();

async function loadTimeline() {
  const content = document.getElementById('content');
  content.innerHTML = '<div class="empty" style="color:#6e7681">Loading timeline…</div>';

  const p = document.getElementById('projectSelect').value;
  const params = new URLSearchParams();
  if (p) params.set('project', p);

  try {
    const res = await fetch('/api/timeline?' + params);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    tlData = data.memories;
    content.innerHTML = renderTimeline(data.memories);
    initTimelineChart(data.memories);
  } catch(e) {
    content.innerHTML = `<div class="err-msg">Error: ${esc(e.message)}</div>`;
  }
}

function renderTimeline(memories) {
  // Group by day
  const days = {};
  for (const m of memories) {
    const d = m.created_at.split('T')[0];
    if (!days[d]) days[d] = [];
    days[d].push(m);
  }

  const dayKeys = Object.keys(days).sort().reverse();

  let html = `<div class="tl-container">
    <div class="tl-controls">
      ${CATEGORIES.map(c =>
        `<label><input type="checkbox" checked onchange="toggleTlCat('${c}', this.checked)">
         <span class="cat-${c}">${c}</span></label>`
      ).join('')}
      <label style="color:#484f58;margin-left:12px">
        <input type="checkbox" onchange="toggleTlSuperseded(this.checked)"> show superseded
      </label>
    </div>
    <div class="tl-chart"><canvas class="tl-canvas" id="tlCanvas"></canvas></div>
    <div class="tl-day-list" id="tlDayList">`;

  for (const day of dayKeys) {
    const items = days[day];
    const dateStr = new Date(day + 'T12:00:00').toLocaleDateString('en', {weekday:'short', month:'short', day:'numeric'});
    html += `<div class="tl-day" data-day="${day}">
      <div class="tl-day-header" onclick="toggleTlDay(this)">
        ${dateStr} <span class="tl-day-count">${items.length} memor${items.length===1?'y':'ies'}</span>
      </div>
      <div class="tl-day-items">`;

    for (const m of items) {
      const time = new Date(m.created_at).toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
      const statusBadge = m.status !== 'active'
        ? ` <span class="badge badge-${m.status}" style="font-size:9px;padding:1px 5px">${m.status}</span>` : '';
      html += `<div class="tl-item" data-cat="${m.category}" data-status="${m.status}">
        <span class="tl-item-time">${time}</span>
        <span class="tl-item-cat cat-${m.category}">${m.category}${statusBadge}</span>
        <span class="tl-item-content">${esc(m.content)}</span>
      </div>`;
    }
    html += '</div></div>';
  }

  html += '</div></div>';
  return html;
}

function initTimelineChart(memories) {
  tlCanvas = document.getElementById('tlCanvas');
  if (!tlCanvas) return;
  tlCtx = tlCanvas.getContext('2d');

  const wrap = tlCanvas.parentElement;
  const dpr = window.devicePixelRatio || 1;
  tlCanvas.width = wrap.clientWidth * dpr;
  tlCanvas.height = wrap.clientHeight * dpr;
  tlCanvas.style.width = wrap.clientWidth + 'px';
  tlCanvas.style.height = wrap.clientHeight + 'px';
  tlCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

  drawTimelineChart(memories);
}

function drawTimelineChart(memories) {
  if (!tlCtx) return;
  const w = tlCanvas.clientWidth, h = tlCanvas.clientHeight;
  tlCtx.clearRect(0, 0, w, h);

  // Aggregate by day + category
  const filtered = memories.filter(m => !tlHiddenCats.has(m.category));
  const days = {};
  for (const m of filtered) {
    if (m.status === 'superseded' && !document.querySelector('.tl-controls input[type=checkbox]:last-of-type')?.checked) continue;
    const d = m.created_at.split('T')[0];
    if (!days[d]) days[d] = {};
    if (!days[d][m.category]) days[d][m.category] = 0;
    days[d][m.category]++;
  }

  const dayKeys = Object.keys(days).sort();
  if (!dayKeys.length) return;

  const pad = {l: 40, r: 16, t: 12, b: 28};
  const barW = Math.max(4, Math.min(24, (w - pad.l - pad.r) / dayKeys.length - 2));
  const gap = 2;

  // Max stacked height
  let maxH = 0;
  for (const d of dayKeys) {
    const total = Object.values(days[d]).reduce((a,b) => a+b, 0);
    if (total > maxH) maxH = total;
  }
  if (maxH === 0) return;

  const chartH = h - pad.t - pad.b;
  const chartW = w - pad.l - pad.r;

  // Y-axis gridlines
  const yTicks = Math.min(maxH, 4);
  for (let i = 0; i <= yTicks; i++) {
    const y = pad.t + chartH - (i / yTicks) * chartH;
    tlCtx.beginPath();
    tlCtx.moveTo(pad.l, y);
    tlCtx.lineTo(w - pad.r, y);
    tlCtx.strokeStyle = '#21262d';
    tlCtx.lineWidth = 0.5;
    tlCtx.stroke();
    const val = Math.round(maxH * i / yTicks);
    tlCtx.fillStyle = '#484f58';
    tlCtx.font = '10px monospace';
    tlCtx.textAlign = 'right';
    tlCtx.fillText(val, pad.l - 6, y + 3);
  }

  // Bars
  const totalBarSpace = dayKeys.length * (barW + gap);
  const offsetX = totalBarSpace < chartW ? (chartW - totalBarSpace) / 2 : 0;

  for (let i = 0; i < dayKeys.length; i++) {
    const d = dayKeys[i];
    const x = pad.l + offsetX + i * (barW + gap);
    let y = pad.t + chartH;

    const catOrder = ['convention', 'decision', 'pattern', 'gotcha', 'preference'];
    for (const cat of catOrder) {
      const count = days[d][cat] || 0;
      if (!count) continue;
      const barH = (count / maxH) * chartH;
      y -= barH;
      tlCtx.fillStyle = hexAlpha(CAT_COLORS[cat] || '#8b949e', 0.75);
      tlCtx.fillRect(x, y, barW, barH);
    }

    // Date label (show every Nth depending on density)
    const labelEvery = Math.max(1, Math.floor(dayKeys.length / 12));
    if (i % labelEvery === 0) {
      tlCtx.save();
      tlCtx.fillStyle = '#484f58';
      tlCtx.font = '9px monospace';
      tlCtx.textAlign = 'center';
      const label = d.slice(5); // MM-DD
      tlCtx.fillText(label, x + barW/2, h - pad.b + 14);
      tlCtx.restore();
    }
  }
}

function toggleTlDay(header) {
  header.nextElementSibling.classList.toggle('open');
}

let tlShowSuperseded = false;
function toggleTlCat(cat, checked) {
  if (checked) tlHiddenCats.delete(cat); else tlHiddenCats.add(cat);
  applyTlFilters();
  drawTimelineChart(tlData);
}

function toggleTlSuperseded(checked) {
  tlShowSuperseded = checked;
  applyTlFilters();
  // Re-fetch with all statuses
  if (checked) {
    const p = document.getElementById('projectSelect').value;
    const params = new URLSearchParams();
    if (p) params.set('project', p);
    fetch('/api/timeline?' + params).then(r => r.json()).then(data => {
      tlData = data.memories;
      drawTimelineChart(tlData);
    });
  } else {
    drawTimelineChart(tlData);
  }
}

function applyTlFilters() {
  document.querySelectorAll('.tl-item').forEach(el => {
    const cat = el.dataset.cat;
    const status = el.dataset.status;
    const hide = tlHiddenCats.has(cat) || (status === 'superseded' && !tlShowSuperseded);
    el.style.display = hide ? 'none' : '';
  });
  // Update day counts
  document.querySelectorAll('.tl-day').forEach(dayEl => {
    const visible = dayEl.querySelectorAll('.tl-item:not([style*="display: none"])').length;
    const count = dayEl.querySelector('.tl-day-count');
    if (count) count.textContent = `${visible} memor${visible===1?'y':'ies'}`;
  });
}

// ── Graph ──────────────────────────────────────────────────────────────────────

let graphNodes = [], graphEdges = [];
let graphCanvas, graphCtx;
let graphSim = null;       // animation frame id
let graphThreshold = 0.75;
let graphDragNode = -1;
let graphHoverNode = -1;
let graphPan = {x: 0, y: 0};
let graphZoom = 1;
let graphDragging = false;
let graphLastMouse = {x: 0, y: 0};
let graphPanning = false;
let graphQueryData = null;    // {query: {x,y}, top: [...], similarities: [...]}
let graphQuerySims = [];      // per-node similarities
let graphQueryTop = [];       // top-k results
let graphHighlightIdx = -1;   // hovered result in drawer

async function loadGraph() {
  const content = document.getElementById('content');
  content.innerHTML = `
    <div class="graph-container">
      <div class="graph-main">
        <div class="map-loading" id="graphLoading">Loading graph…</div>
        <canvas class="graph-canvas" id="graphCanvas"></canvas>
        <div class="map-tooltip" id="graphTooltip">
          <div class="tt-cat"></div>
          <div class="tt-content"></div>
          <div class="tt-conf"></div>
        </div>
        <div class="graph-controls">
          <label>Threshold
            <input type="range" id="graphThresholdSlider" min="0.60" max="0.95" step="0.01"
                   value="${graphThreshold}" oninput="onGraphThresholdChange(this.value)">
            <span class="val" id="graphThresholdVal">${graphThreshold}</span>
          </label>
        </div>
        <div class="graph-stats" id="graphStats"></div>
        <div class="map-legend" id="graphLegend"></div>
      </div>
      <div class="map-drawer">
        <div class="map-drawer-header">
          <div style="display:flex;gap:6px;align-items:center">
            <input type="text" id="graphQueryInput" placeholder="Type a query to fire synapses…"
                   onkeydown="if(event.key==='Enter')runGraphQuery()" style="flex:1">
            <button class="btn" id="graphResetBtn" onclick="resetGraphQuery()" style="display:none;padding:5px 8px;font-size:11px" title="Reset">✕</button>
          </div>
        </div>
        <div class="map-drawer-results" id="graphDrawerResults">
          <div class="map-drawer-empty">Enter a query to see synapses fire across the graph.</div>
        </div>
      </div>
    </div>`;

  graphCanvas = document.getElementById('graphCanvas');
  graphCtx = graphCanvas.getContext('2d');

  // Legend
  const legend = document.getElementById('graphLegend');
  legend.innerHTML = CATEGORIES.map(c =>
    `<div class="map-legend-item" data-cat="${c}" onclick="toggleGraphCat('${c}')">
       <div class="map-legend-dot" style="background:${CAT_COLORS[c]}"></div>
       <span>${c}</span>
     </div>`
  ).join('');

  resizeGraph();
  window.addEventListener('resize', () => { resizeGraph(); drawGraph(); });
  graphCanvas.onmousemove = onGraphMouseMove;
  graphCanvas.onmousedown = onGraphMouseDown;
  graphCanvas.onmouseup = onGraphMouseUp;
  graphCanvas.onmouseleave = () => { graphHoverNode = -1; graphPanning = false; drawGraph();
    document.getElementById('graphTooltip').style.display = 'none'; };
  graphCanvas.onwheel = onGraphWheel;

  await fetchGraph();
}

async function fetchGraph() {
  if (graphSim) { cancelAnimationFrame(graphSim); graphSim = null; }
  document.getElementById('graphLoading').style.display = 'flex';

  const p = document.getElementById('projectSelect').value;
  const params = new URLSearchParams();
  if (p) params.set('project', p);
  params.set('threshold', graphThreshold);

  try {
    const res = await fetch('/api/graph?' + params);
    const data = await res.json();
    document.getElementById('graphLoading').style.display = 'none';

    if (data.error) {
      document.getElementById('graphLoading').style.display = 'flex';
      document.getElementById('graphLoading').textContent = data.error;
      return;
    }

    graphEdges = data.edges;

    // Initialize node positions randomly if fresh, or keep existing
    const w = graphCanvas.clientWidth, h = graphCanvas.clientHeight;
    const oldPos = {};
    graphNodes.forEach(n => { oldPos[n.id] = {x: n.x, y: n.y, vx: n.vx, vy: n.vy}; });

    graphNodes = data.nodes.map((n, i) => {
      const old = oldPos[n.id];
      return {
        ...n, idx: i,
        x: old ? old.x : (Math.random() - 0.5) * w * 0.6,
        y: old ? old.y : (Math.random() - 0.5) * h * 0.6,
        vx: old ? old.vx : 0, vy: old ? old.vy : 0,
      };
    });

    // Build adjacency for quick lookup
    graphNodes.forEach(n => { n.degree = 0; });
    graphEdges.forEach(e => { graphNodes[e.source].degree++; graphNodes[e.target].degree++; });

    document.getElementById('graphStats').innerHTML =
      `<span>${graphNodes.length}</span> nodes · <span>${graphEdges.length}</span> edges`;

    // Reset pan/zoom
    graphPan = {x: 0, y: 0};
    graphZoom = 1;

    runGraphSim();
  } catch(e) {
    document.getElementById('graphLoading').textContent = 'Failed: ' + e.message;
  }
}

function resizeGraph() {
  const wrap = graphCanvas.parentElement;
  const dpr = window.devicePixelRatio || 1;
  graphCanvas.width = wrap.clientWidth * dpr;
  graphCanvas.height = wrap.clientHeight * dpr;
  graphCanvas.style.width = wrap.clientWidth + 'px';
  graphCanvas.style.height = wrap.clientHeight + 'px';
  graphCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function runGraphSim() {
  let alpha = 1.0;
  const decay = 0.997;
  const minAlpha = 0.001;

  function tick() {
    if (alpha < minAlpha) { drawGraph(); return; }

    const N = graphNodes.length;
    const repulsion = 800;
    const attraction = 0.06;
    const centerGravity = 0.01;
    const damping = 0.85;
    const maxForce = 20;

    // Repulsion (all pairs)
    for (let i = 0; i < N; i++) {
      for (let j = i + 1; j < N; j++) {
        const a = graphNodes[i], b = graphNodes[j];
        let dx = a.x - b.x, dy = a.y - b.y;
        let dist = Math.sqrt(dx * dx + dy * dy) || 1;
        let force = Math.min(repulsion / (dist * dist), maxForce) * alpha;
        let fx = (dx / dist) * force, fy = (dy / dist) * force;
        a.vx += fx; a.vy += fy;
        b.vx -= fx; b.vy -= fy;
      }
    }

    // Attraction (edges)
    for (const e of graphEdges) {
      const a = graphNodes[e.source], b = graphNodes[e.target];
      let dx = b.x - a.x, dy = b.y - a.y;
      let dist = Math.sqrt(dx * dx + dy * dy) || 1;
      // Stronger attraction for higher similarity
      let strength = attraction * (e.sim * e.sim) * alpha;
      let force = dist * strength;
      let fx = (dx / dist) * force, fy = (dy / dist) * force;
      a.vx += fx; a.vy += fy;
      b.vx -= fx; b.vy -= fy;
    }

    // Center gravity
    for (const n of graphNodes) {
      n.vx -= n.x * centerGravity * alpha;
      n.vy -= n.y * centerGravity * alpha;
    }

    // Integrate
    for (const n of graphNodes) {
      if (n.idx === graphDragNode) continue;
      n.vx *= damping; n.vy *= damping;
      n.x += n.vx; n.y += n.vy;
    }

    alpha *= decay;
    drawGraph();
    graphSim = requestAnimationFrame(tick);
  }

  tick();
}

function graphToScreen(nx, ny) {
  const w = graphCanvas.clientWidth / 2;
  const h = graphCanvas.clientHeight / 2;
  return [(nx * graphZoom) + w + graphPan.x, (ny * graphZoom) + h + graphPan.y];
}

function screenToGraph(sx, sy) {
  const w = graphCanvas.clientWidth / 2;
  const h = graphCanvas.clientHeight / 2;
  return [(sx - w - graphPan.x) / graphZoom, (sy - h - graphPan.y) / graphZoom];
}

let graphHiddenCats = new Set();

function drawGraph() {
  const w = graphCanvas.clientWidth, h = graphCanvas.clientHeight;
  graphCtx.clearRect(0, 0, w, h);

  const hasQuery = graphQuerySims.length > 0;
  const topIds = new Set(graphQueryTop.map(t => t.id));
  const anim = graphAnimAlpha;

  // Draw edges
  for (const e of graphEdges) {
    const a = graphNodes[e.source], b = graphNodes[e.target];
    if (graphHiddenCats.has(a.category) || graphHiddenCats.has(b.category)) continue;
    const [ax, ay] = graphToScreen(a.x, a.y);
    const [bx, by] = graphToScreen(b.x, b.y);

    if ((ax < -50 && bx < -50) || (ax > w+50 && bx > w+50) ||
        (ay < -50 && by < -50) || (ay > h+50 && by > h+50)) continue;

    let edgeAlpha = 0.04 + (e.sim - graphThreshold) / (1 - graphThreshold) * 0.18;
    let lineW = 0.5 + (e.sim - graphThreshold) * 2;
    let color = '88,166,255';

    if (hasQuery) {
      const simA = graphQuerySims[e.source] || 0;
      const simB = graphQuerySims[e.target] || 0;
      const edgeFire = Math.max(simA, simB);

      if (edgeFire >= 0.7) {
        // Synapse fires — bright, thick, colored by intensity
        const intensity = (edgeFire - 0.7) / 0.3;
        edgeAlpha = (0.15 + intensity * 0.6) * anim;
        lineW = (1 + intensity * 3) * anim;
        // Lerp from blue to white as intensity increases
        const r = Math.round(88 + (255-88) * intensity);
        const g = Math.round(166 + (255-166) * intensity);
        color = `${r},${g},255`;
      } else {
        edgeAlpha = 0.02 * anim + 0.02 * (1 - anim);
        lineW = 0.3;
      }
    }

    graphCtx.beginPath();
    graphCtx.moveTo(ax, ay);
    graphCtx.lineTo(bx, by);
    graphCtx.strokeStyle = `rgba(${color},${edgeAlpha})`;
    graphCtx.lineWidth = lineW;
    graphCtx.stroke();
  }

  // Draw nodes
  for (let i = 0; i < graphNodes.length; i++) {
    const n = graphNodes[i];
    if (graphHiddenCats.has(n.category)) continue;
    const [px, py] = graphToScreen(n.x, n.y);
    if (px < -20 || px > w+20 || py < -20 || py > h+20) continue;

    let r = 3 + Math.sqrt(n.degree || 0) * 1.2;
    let alpha = 0.7;
    let glowR = 0;

    if (hasQuery) {
      const sim = graphQuerySims[i] || 0;
      const isTop = topIds.has(n.id);

      if (sim >= 0.7) {
        const intensity = (sim - 0.7) / 0.3;
        alpha = (0.4 + intensity * 0.6) * anim;
        r += intensity * 4 * anim;
        if (isTop) glowR = (8 + intensity * 12) * anim;
      } else {
        alpha = 0.08 + (1 - anim) * 0.6;
        r = 2.5;
      }
    }

    if (i === graphHoverNode) { r += 3; alpha = 1.0; }
    if (i === graphHighlightIdx) { r += 4; alpha = 1.0; glowR = 16; }

    // Glow for top results
    if (glowR > 0) {
      graphCtx.beginPath();
      graphCtx.arc(px, py, glowR * graphZoom, 0, Math.PI * 2);
      graphCtx.fillStyle = hexAlpha(CAT_COLORS[n.category] || '#58a6ff', 0.12);
      graphCtx.fill();
    }

    graphCtx.beginPath();
    graphCtx.arc(px, py, r * graphZoom, 0, Math.PI * 2);
    graphCtx.fillStyle = hexAlpha(CAT_COLORS[n.category] || '#8b949e', alpha);
    graphCtx.fill();

    if (i === graphHoverNode || i === graphHighlightIdx) {
      graphCtx.strokeStyle = '#c9d1d9';
      graphCtx.lineWidth = 1.5;
      graphCtx.stroke();

      for (const e of graphEdges) {
        if (e.source !== i && e.target !== i) continue;
        const other = e.source === i ? e.target : e.source;
        const on = graphNodes[other];
        if (graphHiddenCats.has(on.category)) continue;
        const [ox, oy] = graphToScreen(on.x, on.y);
        graphCtx.beginPath();
        graphCtx.moveTo(px, py);
        graphCtx.lineTo(ox, oy);
        graphCtx.strokeStyle = hexAlpha(CAT_COLORS[n.category] || '#58a6ff', 0.5);
        graphCtx.lineWidth = 1.5;
        graphCtx.stroke();
      }
    }
  }
}

function onGraphMouseMove(e) {
  const rect = graphCanvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;

  if (graphPanning) {
    graphPan.x += mx - graphLastMouse.x;
    graphPan.y += my - graphLastMouse.y;
    graphLastMouse = {x: mx, y: my};
    drawGraph();
    return;
  }

  if (graphDragNode >= 0) {
    const [gx, gy] = screenToGraph(mx, my);
    graphNodes[graphDragNode].x = gx;
    graphNodes[graphDragNode].y = gy;
    graphNodes[graphDragNode].vx = 0;
    graphNodes[graphDragNode].vy = 0;
    drawGraph();
    return;
  }

  // Hit test
  let closest = -1, closestDist = 15 / graphZoom;
  const [gmx, gmy] = screenToGraph(mx, my);
  for (let i = 0; i < graphNodes.length; i++) {
    if (graphHiddenCats.has(graphNodes[i].category)) continue;
    const d = Math.hypot(graphNodes[i].x - gmx, graphNodes[i].y - gmy);
    if (d < closestDist) { closestDist = d; closest = i; }
  }

  if (closest !== graphHoverNode) {
    graphHoverNode = closest;
    drawGraph();
  }

  const tt = document.getElementById('graphTooltip');
  if (closest >= 0) {
    const n = graphNodes[closest];
    tt.querySelector('.tt-cat').innerHTML = `<span class="cat-${n.category}">${n.category}</span>` +
      ` · ${n.degree} connection${n.degree !== 1 ? 's' : ''}`;
    tt.querySelector('.tt-content').textContent = n.content.length > 300 ? n.content.slice(0,300) + '…' : n.content;
    tt.querySelector('.tt-conf').textContent = `confidence: ${Math.round(n.confidence * 100)}%`;
    tt.style.display = 'block';
    let tx = mx + 14, ty = my - 10;
    const w = graphCanvas.clientWidth;
    const h = graphCanvas.clientHeight;
    if (tx + 370 > w) tx = mx - 374;
    if (ty + 100 > h) ty = h - 110;
    if (ty < 0) ty = 4;
    tt.style.left = tx + 'px'; tt.style.top = ty + 'px';
  } else {
    tt.style.display = 'none';
  }
}

function onGraphMouseDown(e) {
  const rect = graphCanvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;

  if (graphHoverNode >= 0) {
    graphDragNode = graphHoverNode;
    graphCanvas.style.cursor = 'grabbing';
  } else {
    graphPanning = true;
    graphLastMouse = {x: mx, y: my};
    graphCanvas.style.cursor = 'grabbing';
  }
}

function onGraphMouseUp(e) {
  graphDragNode = -1;
  graphPanning = false;
  graphCanvas.style.cursor = graphHoverNode >= 0 ? 'pointer' : 'grab';
}

function onGraphWheel(e) {
  e.preventDefault();
  const factor = e.deltaY > 0 ? 0.9 : 1.1;
  const rect = graphCanvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  const cx = graphCanvas.clientWidth / 2;
  const cy = graphCanvas.clientHeight / 2;

  // Zoom toward mouse position
  graphPan.x = mx - (mx - graphPan.x - cx) * factor - cx;
  graphPan.y = my - (my - graphPan.y - cy) * factor - cy;
  graphZoom *= factor;
  graphZoom = Math.max(0.1, Math.min(10, graphZoom));
  drawGraph();
}

function onGraphThresholdChange(val) {
  graphThreshold = parseFloat(val);
  document.getElementById('graphThresholdVal').textContent = graphThreshold.toFixed(2);
  fetchGraph();
}

function toggleGraphCat(cat) {
  if (graphHiddenCats.has(cat)) graphHiddenCats.delete(cat);
  else graphHiddenCats.add(cat);
  document.querySelectorAll('#graphLegend .map-legend-item').forEach(el => {
    el.classList.toggle('dimmed', graphHiddenCats.has(el.dataset.cat));
  });
  drawGraph();
}

// ── Graph Query ────────────────────────────────────────────────────────────────

async function runGraphQuery() {
  const input = document.getElementById('graphQueryInput');
  const query = input.value.trim();
  const results = document.getElementById('graphDrawerResults');

  if (!query) { resetGraphQuery(); return; }

  results.innerHTML = '<div class="map-drawer-empty">Embedding query…</div>';
  document.getElementById('graphResetBtn').style.display = 'block';

  const p = document.getElementById('projectSelect').value;
  try {
    const res = await fetch('/api/map/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, project: p, top_k: 10 })
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();

    graphQuerySims = data.similarities;
    graphQueryTop = data.top;
    graphHighlightIdx = -1;

    // Animate the synapse firing
    animateGraphQuery();
    renderGraphResults(data.top);
  } catch(e) {
    results.innerHTML = `<div class="map-drawer-empty" style="color:#f85149">Error: ${esc(e.message)}</div>`;
  }
}

function resetGraphQuery() {
  graphQuerySims = []; graphQueryTop = []; graphHighlightIdx = -1;
  graphQueryData = null;
  document.getElementById('graphQueryInput').value = '';
  document.getElementById('graphResetBtn').style.display = 'none';
  document.getElementById('graphDrawerResults').innerHTML =
    '<div class="map-drawer-empty">Enter a query to see synapses fire across the graph.</div>';
  drawGraph();
}

let graphAnimFrame = 0;
let graphAnimAlpha = 0;

function animateGraphQuery() {
  graphAnimFrame = 0;
  graphAnimAlpha = 0;

  function tick() {
    graphAnimFrame++;
    graphAnimAlpha = Math.min(1, graphAnimFrame / 30); // fade in over 30 frames
    drawGraph();
    if (graphAnimAlpha < 1) requestAnimationFrame(tick);
  }
  tick();
}

function renderGraphResults(top) {
  const results = document.getElementById('graphDrawerResults');
  if (!top.length) {
    results.innerHTML = '<div class="map-drawer-empty">No results found.</div>';
    return;
  }

  const aboveCount = graphQuerySims.filter(s => s >= 0.85).length;

  results.innerHTML =
    `<div class="sim-threshold">
       <span style="color:#58a6ff;font-weight:600">${aboveCount}</span> memor${aboveCount === 1 ? 'y' : 'ies'} within 0.85 cosine similarity
     </div>` +
    top.map((r, i) => {
      const simPct = Math.round(r.similarity * 100);
      const simColor = r.similarity >= 0.85 ? '#58a6ff' : r.similarity >= 0.7 ? '#d29922' : '#6e7681';
      return `<div class="map-result" data-idx="${i}"
                   onmouseenter="highlightGraphResult(${i})"
                   onmouseleave="highlightGraphResult(-1)"
                   onclick="toggleResultExpand(this)">
        <div class="map-result-head">
          <span class="map-result-sim" style="color:${simColor}">${simPct}%</span>
          <span class="map-result-cat cat-${r.category}">${r.category}</span>
          <span class="map-result-conf">${Math.round(r.confidence * 100)}% conf</span>
        </div>
        <div class="map-result-content">${esc(r.content)}</div>
      </div>`;
    }).join('');
}

function highlightGraphResult(drawerIdx) {
  if (drawerIdx < 0) {
    graphHighlightIdx = -1;
  } else {
    const targetId = graphQueryTop[drawerIdx]?.id;
    graphHighlightIdx = graphNodes.findIndex(n => n.id === targetId);
  }
  drawGraph();
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
