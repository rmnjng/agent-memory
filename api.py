#!/usr/bin/env python3
"""
Agent Memory API — recall and add over HTTP, for people without database access.
Run: python3 api.py [--port 7338]

Every request needs `Authorization: Bearer <token>` from `memory_cli.py key add`;
each key carries scopes (recall, add). The logic is memory_cli's, so a recall
here boosts and decays exactly like a local one, and embeddings come from the
embed server (web.py, AGENT_MEMORY_EMBED_URL) instead of a second model copy.

Plain HTTP: meant to be reached only over an encrypted network (Tailscale's
WireGuard on the dev box), never exposed to the internet directly.
"""

import argparse

from flask import Flask, g, jsonify, request

import memory_cli

MAX_CONTENT = 8000
MAX_TOP_K = 50

app = Flask(__name__)


def _conn():
    conn = memory_cli.get_conn(memory_cli.DEFAULT_DB_URL)
    memory_cli._ensure_migrations(conn)
    return conn


def _error(status: int, msg: str):
    return jsonify({"error": msg}), status


@app.before_request
def authenticate():
    header = request.headers.get("Authorization", "")
    token = header[7:] if header.startswith("Bearer ") else ""
    if not token:
        return _error(401, "missing bearer token")
    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute("""
            UPDATE api_keys SET last_used_at = NOW() WHERE token_hash = %s
            RETURNING name, scopes
        """, (memory_cli.hash_token(token),))
        row = cur.fetchone()
        conn.commit()
    finally:
        conn.close()
    if not row:
        return _error(401, "unknown or revoked token")
    g.key_name, g.scopes = row


def _require(scope: str):
    if scope not in g.scopes:
        return _error(403, f"this key has no '{scope}' scope")


def _body() -> dict:
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def _text(data: dict, field: str) -> str:
    value = data.get(field)
    return value.strip() if isinstance(value, str) else ""


@app.get("/v1/whoami")
def whoami():
    return jsonify({"name": g.key_name, "scopes": g.scopes})


@app.post("/v1/recall")
def recall():
    if (denied := _require("recall")):
        return denied
    data = _body()
    project, query = _text(data, "project"), _text(data, "query")
    if not project or not query:
        return _error(400, "project and query required")
    top_k = data.get("top_k", 10)
    if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= MAX_TOP_K:
        return _error(400, f"top_k must be an integer from 1 to {MAX_TOP_K}")
    category = data.get("category")
    if category is not None:
        cats = [c.strip() for c in category.split(",")] if isinstance(category, str) else [None]
        if any(c not in memory_cli.CATEGORIES for c in cats):
            return _error(400, f"category must be some of {','.join(memory_cli.CATEGORIES)}")
    conn = _conn()
    try:
        memories = memory_cli.recall_memories(conn, project, query, top_k, category)
    finally:
        conn.close()
    return jsonify({"memories": memories})


@app.post("/v1/memories")
def add():
    if (denied := _require("add")):
        return denied
    data = _body()
    project, content = _text(data, "project"), _text(data, "content")
    category = data.get("category")
    longevity = data.get("longevity", "normal")
    if not project or not content:
        return _error(400, "project and content required")
    if len(content) > MAX_CONTENT:
        return _error(400, f"content longer than {MAX_CONTENT} characters")
    if category not in memory_cli.CATEGORIES:
        return _error(400, f"category must be one of {','.join(memory_cli.CATEGORIES)}")
    if longevity not in ("normal", "high"):
        return _error(400, "longevity must be normal or high")
    conn = _conn()
    try:
        mem_id, superseded = memory_cli.add_memory(conn, project, category, content, longevity,
                                                   created_by=g.key_name)
    finally:
        conn.close()
    return jsonify({"id": mem_id, "superseded": superseded}), 201


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7338)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
