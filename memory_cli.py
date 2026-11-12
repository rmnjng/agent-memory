#!/usr/bin/env python3
"""
Agent Memory CLI — recall, decay, and manage agent memories.

Usage:
    python memory_cli.py <command> [options]

Commands:
    status      Health check
    recall      Retrieve relevant memories before a task
    revitalize  Boost decayed memories back to a minimum confidence floor
    decay       Run decay cycle (manual maintenance)
    inspect     Browse memories
    forget      Archive or delete a memory
    export      Dump project data to JSON
    import      Restore from backup
    migrate     Apply pending schema migrations
    key         Manage API keys for the memory server (api.py)

With AGENT_MEMORY_API_URL and AGENT_MEMORY_API_TOKEN set, status, recall and add
go to that server instead of the database (no psycopg2 or model needed).
"""

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional

# Optional: a remote client (AGENT_MEMORY_API_URL) needs only the standard library.
try:
    import psycopg2
    import psycopg2.extras
    from pgvector.psycopg2 import register_vector
except ImportError:
    psycopg2 = None

# ─── Configuration ───────────────────────────────────────────────────────────

DEFAULT_DB_URL = os.environ.get(
    "AGENT_MEMORY_DB_URL",
    "postgresql://agent:agent_memory_pwd@localhost:5433/agent_memory"
)
# Set both to use a memory server's API (api.py) instead of the database.
API_URL = os.environ.get("AGENT_MEMORY_API_URL", "").rstrip("/")
API_TOKEN = os.environ.get("AGENT_MEMORY_API_TOKEN", "")
API_SCOPES = ["recall", "add"]
DEFAULT_MODEL = os.environ.get(
    "AGENT_MEMORY_EMBED_MODEL",
    "nomic-ai/nomic-embed-text-v1.5"
)
EMBED_SERVER_URL = os.environ.get("AGENT_MEMORY_EMBED_URL", "http://localhost:7337")

CATEGORIES = ["convention", "gotcha", "pattern", "preference", "decision"]

# Lazy-loaded local model fallback
_model = None


# ─── Embedding ───────────────────────────────────────────────────────────────

# Generous on purpose: under CPU load a long text takes the server 3-5 s. A short
# timeout made every slow call fall back to loading a private ~850 MB model copy,
# which with parallel sessions added load and pushed the box toward OOM.
EMBED_SERVER_TIMEOUT = float(os.environ.get("AGENT_MEMORY_EMBED_TIMEOUT", "30"))


def _is_timeout(exc: Exception) -> bool:
    import socket
    import urllib.error
    if isinstance(exc, urllib.error.URLError):
        exc = exc.reason if isinstance(exc.reason, Exception) else exc
    return isinstance(exc, (TimeoutError, socket.timeout))


def _embed_via_server(text: str, mode: str = "document") -> "list[float] | None":
    """Embed via the web server. Returns None if the server is unreachable or errors
    (caller falls back to a local model); raises if it is merely slow, so a busy
    server never triggers a second model load."""
    import urllib.request
    payload = json.dumps({"text": text, "mode": mode}).encode()
    req = urllib.request.Request(
        f"{EMBED_SERVER_URL}/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=EMBED_SERVER_TIMEOUT) as resp:
            return json.loads(resp.read())["embedding"]
    except Exception as e:
        if _is_timeout(e):
            raise RuntimeError(
                f"Embed server at {EMBED_SERVER_URL} did not answer within "
                f"{EMBED_SERVER_TIMEOUT:.0f}s; not loading a local model while it is busy. Retry later."
            ) from e
        return None


def get_model():
    """Lazy-load the local sentence-transformers model (fallback only)."""
    global _model
    if _model is None:
        print("Loading embedding model (first run downloads ~270MB)...")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(DEFAULT_MODEL, trust_remote_code=True)
        print(f"Model loaded: {DEFAULT_MODEL}")
    return _model


def embed(text: str) -> list[float]:
    """Embed a document string. Uses web server if available, falls back to local model."""
    vec = _embed_via_server(text, "document")
    if vec is not None:
        return vec
    return get_model().encode(f"search_document: {text}", normalize_embeddings=True).tolist()


def embed_query(text: str) -> list[float]:
    """Embed a query string. Uses web server if available, falls back to local model."""
    vec = _embed_via_server(text, "query")
    if vec is not None:
        return vec
    return get_model().encode(f"search_query: {text}", normalize_embeddings=True).tolist()


# ─── Database ────────────────────────────────────────────────────────────────

def get_conn(db_url: str = DEFAULT_DB_URL):
    """Get a database connection with pgvector registered."""
    if psycopg2 is None:
        sys.exit("ERROR: psycopg2/pgvector not installed (pip install -r requirements.txt), "
                 "or set AGENT_MEMORY_API_URL to use a memory server.")
    conn = psycopg2.connect(db_url)
    register_vector(conn)
    return conn


# ─── API client (remote mode) ────────────────────────────────────────────────

def _api(method: str, path: str, body: Optional[dict] = None) -> dict:
    """Call the memory server's API; exits with its error message on failure."""
    import urllib.error
    import urllib.request
    if not API_TOKEN:
        sys.exit("ERROR: AGENT_MEMORY_API_URL is set but AGENT_MEMORY_API_TOKEN is not.")
    req = urllib.request.Request(
        API_URL + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=EMBED_SERVER_TIMEOUT + 30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read()).get("error", e.reason)
        except ValueError:
            msg = e.reason
        sys.exit(f"ERROR: memory server answered {e.code}: {msg}")
    except urllib.error.URLError as e:
        sys.exit(f"ERROR: memory server at {API_URL} unreachable: {e.reason}")


# ─── Schema Migrations ───────────────────────────────────────────────────────

MIGRATIONS = {
    3: """
        ALTER TABLE memories
            ADD COLUMN IF NOT EXISTS decay_multiplier REAL NOT NULL DEFAULT 1.0
                CHECK (decay_multiplier >= 0.0 AND decay_multiplier <= 1.0);
        INSERT INTO schema_version (version) VALUES (3) ON CONFLICT DO NOTHING;
    """,
    4: """
        CREATE TABLE IF NOT EXISTS config (
            key             TEXT PRIMARY KEY,
            value           TEXT NOT NULL,
            default_value   TEXT NOT NULL,
            value_type      TEXT NOT NULL CHECK (value_type IN ('float', 'int', 'json')),
            group_name      TEXT NOT NULL,
            description     TEXT NOT NULL,
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        INSERT INTO config (key, value, default_value, value_type, group_name, description) VALUES
            ('decay.passive_base',          '0.005', '0.005', 'float', 'decay',     'Base passive decay applied per recall cycle'),
            ('decay.reference_pool_size',   '150',   '150',   'int',   'decay',     'Pool size at which full base decay rate applies; larger pools decay slower'),
            ('decay.stale_threshold',       '0.3',   '0.3',   'float', 'decay',     'Confidence below which a memory becomes stale'),
            ('decay.archive_threshold',     '0.1',   '0.1',   'float', 'decay',     'Confidence below which a memory is archived'),
            ('decay.sim_floor',             '0.4',   '0.4',   'float', 'decay',     'Minimum cosine similarity for recall results; also exempts nearby memories from passive decay'),
            ('decay.category_multipliers',  '{"gotcha":0.2,"convention":0.2,"decision":0.3,"pattern":0.7,"preference":1.0}',
                                            '{"gotcha":0.2,"convention":0.2,"decision":0.3,"pattern":0.7,"preference":1.0}',
                                            'json', 'decay', 'Per-category decay rate multipliers (lower = slower decay)'),
            ('decay.longevity_values',      '{"high":0.2,"normal":1.0}',
                                            '{"high":0.2,"normal":1.0}',
                                            'json', 'decay', 'Per-longevity decay multipliers for --longevity flag'),
            ('recall.weight_similarity',    '0.5',   '0.5',   'float', 'recall',    'Scoring weight for cosine similarity (sum of all weights should be 1.0)'),
            ('recall.weight_confidence',    '0.3',   '0.3',   'float', 'recall',    'Scoring weight for confidence'),
            ('recall.weight_recency',       '0.2',   '0.2',   'float', 'recall',    'Scoring weight for recency'),
            ('recall.confidence_boost',     '0.1',   '0.1',   'float', 'recall',    'Confidence boost applied to retrieved memories'),
            ('supersede.similarity_threshold', '0.85', '0.85', 'float', 'supersede', 'Cosine similarity above which a new memory supersedes an existing one'),
            ('ui.graph_edge_threshold',     '0.75',  '0.75',  'float', 'ui',        'Default similarity threshold for graph view edges'),
            ('ui.cluster_min_size',         '3',     '3',     'int',   'ui',        'Default minimum cluster size for HDBSCAN')
        ON CONFLICT DO NOTHING;
        INSERT INTO schema_version (version) VALUES (4) ON CONFLICT DO NOTHING;
    """,
    5: """
        -- API keys for api.py. Only the token's SHA-256 is stored: the token is
        -- 32 random bytes, so a fast hash is enough (no password to brute-force).
        CREATE TABLE IF NOT EXISTS api_keys (
            id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            name            TEXT NOT NULL UNIQUE,
            token_hash      TEXT NOT NULL UNIQUE,
            scopes          TEXT[] NOT NULL,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            last_used_at    TIMESTAMPTZ
        );
        -- Which key wrote a memory through the API; NULL = written with direct DB access.
        ALTER TABLE memories ADD COLUMN IF NOT EXISTS created_by TEXT;
        INSERT INTO schema_version (version) VALUES (5) ON CONFLICT DO NOTHING;
    """,
}


def _get_schema_version(cur) -> int:
    try:
        cur.execute("SELECT MAX(version) FROM schema_version")
        row = cur.fetchone()
        return row[0] if row and row[0] else 0
    except Exception:
        return 0


def _ensure_migrations(conn):
    """Apply any pending migrations silently. Called on every connection."""
    cur = conn.cursor()
    current = _get_schema_version(cur)
    for version, sql in sorted(MIGRATIONS.items()):
        if version > current:
            cur.execute(sql)
            conn.commit()
    cur.close()


# ─── Config loader ───────────────────────────────────────────────────────────

_CONFIG_DEFAULTS = {
    "decay.passive_base":            ("0.005", "float"),
    "decay.reference_pool_size":     ("150",   "int"),
    "decay.stale_threshold":         ("0.3",   "float"),
    "decay.archive_threshold":       ("0.1",   "float"),
    "decay.sim_floor":               ("0.4",   "float"),
    "decay.category_multipliers":    ('{"gotcha":0.2,"convention":0.2,"decision":0.3,"pattern":0.7,"preference":1.0}', "json"),
    "decay.longevity_values":        ('{"high":0.2,"normal":1.0}', "json"),
    "recall.weight_similarity":      ("0.5",   "float"),
    "recall.weight_confidence":      ("0.3",   "float"),
    "recall.weight_recency":         ("0.2",   "float"),
    "recall.confidence_boost":       ("0.1",   "float"),
    "supersede.similarity_threshold": ("0.85", "float"),
    "ui.graph_edge_threshold":       ("0.75",  "float"),
    "ui.cluster_min_size":           ("3",     "int"),
}


def _parse_config_value(raw: str, vtype: str):
    if vtype == "float": return float(raw)
    if vtype == "int":   return int(raw)
    if vtype == "json":  return json.loads(raw)
    return raw


def _load_config(conn) -> dict:
    """Load config from DB, falling back to hardcoded defaults if table missing."""
    result = {}
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT key, value, value_type FROM config")
        for row in cur.fetchall():
            try:
                result[row["key"]] = _parse_config_value(row["value"], row["value_type"])
            except (ValueError, json.JSONDecodeError) as e:
                print(f"WARNING: bad config value for '{row['key']}': {e} — using default",
                      file=sys.stderr)
        cur.close()
    except psycopg2.errors.UndefinedTable:
        conn.rollback()  # clear the failed transaction
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
    for key, (default, vtype) in _CONFIG_DEFAULTS.items():
        if key not in result:
            result[key] = _parse_config_value(default, vtype)
    return result


def get_config(conn) -> dict:
    return _load_config(conn)


def _build_category_case(multipliers: dict) -> str:
    """Build a SQL CASE statement for per-category decay multipliers."""
    parts = []
    for cat in CATEGORIES:
        try:
            val = float(multipliers.get(cat, 1.0))
        except (TypeError, ValueError):
            val = 1.0
        parts.append(f"WHEN '{cat}' THEN {val}")
    return f"CASE category {' '.join(parts)} ELSE 1.0 END"


# ─── Commands ────────────────────────────────────────────────────────────────

def cmd_status(args):
    """Health check: DB, schema, model, counts."""
    print("=== Agent Memory Status ===\n")

    if API_URL:
        me = _api("GET", "/v1/whoami")
        print(f"Server:   {API_URL}")
        print(f"Key:      {me['name']} (scopes: {','.join(me['scopes'])})")
        return

    try:
        conn = get_conn(args.db_url)
        _ensure_migrations(conn)
        cur = conn.cursor()
        cur.execute("SELECT version()")
        pg_version = cur.fetchone()[0].split(",")[0]
        print(f"Database: Connected ({pg_version})")

        cur.execute("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1")
        row = cur.fetchone()
        print(f"Schema:   v{row[0]}" if row else "Schema:   NOT FOUND")

        where = ""
        params = []
        if args.project:
            where = "WHERE project = %s"
            params = [args.project]
        cur.execute(f"SELECT COUNT(*) FROM memories {where}", params)
        print(f"  memories    : {cur.fetchone()[0]} rows")

        if args.project:
            cur.execute("""
                SELECT category, status, COUNT(*) FROM memories WHERE project = %s
                GROUP BY category, status ORDER BY category, status
            """, params)
            for row in cur.fetchall():
                print(f"    [{row[0]:12s} / {row[1]:10s}]: {row[2]}")

        conn.close()
    except Exception as e:
        print(f"Database: ERROR — {e}")

    print()
    try:
        get_model()
        print(f"Model:    Loaded ({DEFAULT_MODEL})")
    except Exception as e:
        print(f"Model:    ERROR — {e}")

    print()
    import subprocess
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Status}}", "agent-memory-db"],
            capture_output=True, text=True
        )
        status = result.stdout.strip() if result.returncode == 0 else "not found"
        print(f"Docker:   agent-memory-db is {status}")
    except FileNotFoundError:
        print("Docker:   not available")


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token ≈ 4 chars."""
    return max(1, len(text) // 4)


def _apply_token_budget(results: dict, max_tokens: int) -> int:
    """
    Trim results to fit within a token budget.
    Items ranked by similarity, filled greedily from highest to lowest.
    Returns the number of items trimmed.
    """
    remaining = max_tokens
    candidates = [
        (m, m["similarity"], _estimate_tokens(m["content"]))
        for m in results["memories"]
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)

    kept = []
    trimmed = 0
    for item, _sim, tokens in candidates:
        if remaining > 0 and tokens <= remaining:
            kept.append(item)
            remaining -= tokens
        else:
            trimmed += 1

    results["memories"] = kept
    return trimmed


def cmd_recall(args):
    """Retrieve relevant memories before starting a task."""
    if API_URL:
        memories = _api("POST", "/v1/recall", {
            "project": args.project, "query": args.query,
            "top_k": args.top_k, "category": args.category,
        })["memories"]
    else:
        conn = get_conn(args.db_url)
        _ensure_migrations(conn)
        memories = recall_memories(conn, args.project, args.query, args.top_k, args.category)
        conn.close()

    results = {"memories": memories, "_trimmed": 0}
    if args.max_tokens is not None:
        results["_trimmed"] = _apply_token_budget(results, args.max_tokens)

    if args.format == "json":
        print(json.dumps(results, indent=2))
    else:
        _print_recall_markdown(results, args.query)


def recall_memories(conn, project: str, query: str, top_k: int,
                    category: Optional[str] = None) -> list[dict]:
    """Top memories for a query; boosts the retrieved ones and decays the unrelated. Commits."""
    query_vec = embed_query(query)
    cfg = get_config(conn)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    memories = []

    sim_floor = cfg["decay.sim_floor"]
    w_sim = cfg["recall.weight_similarity"]
    w_conf = cfg["recall.weight_confidence"]
    w_rec = cfg["recall.weight_recency"]
    boost = cfg["recall.confidence_boost"]
    cat_case = _build_category_case(cfg["decay.category_multipliers"])

    # Always include stale — a highly relevant stale memory should resurface.
    # The similarity floor keeps noise out instead.
    status_filter = "('active', 'stale')"

    category_filter = ""
    cat_params = []
    if category:
        cats = [c.strip() for c in category.split(",")]
        placeholders = ",".join(["%s"] * len(cats))
        category_filter = f"AND category IN ({placeholders})"
        cat_params = cats

    cur.execute(f"""
        SELECT id, category, content, confidence, status, last_referenced, created_at,
               decay_multiplier,
               1 - (content_embedding <=> %s::vector) AS similarity
        FROM memories
        WHERE project = %s AND status IN {status_filter}
          AND 1 - (content_embedding <=> %s::vector) >= %s
          {category_filter}
        ORDER BY (1 - (content_embedding <=> %s::vector)) * %s
                 + confidence * %s
                 + (1.0 / (1.0 + EXTRACT(EPOCH FROM NOW() - last_referenced) / 86400.0 * 0.1)) * %s
        DESC
        LIMIT %s
    """, [query_vec, project, query_vec, sim_floor] + cat_params +
         [query_vec, w_sim, w_conf, w_rec, top_k])

    retrieved_ids = []
    for row in cur.fetchall():
        memories.append({
            "category": row["category"],
            "content": row["content"],
            "confidence": round(row["confidence"], 3),
            "similarity": round(row["similarity"], 3),
        })
        retrieved_ids.append(row["id"])
        # Boost confidence for retrieved memories
        cur.execute(f"""
            UPDATE memories SET last_referenced = NOW(),
                   confidence = LEAST(1.0, confidence + %s)
            WHERE id = %s
        """, (boost, row["id"]))

    # Dynamic decay rate: slower when the pool is large.
    cur.execute(
        "SELECT COUNT(*) FROM memories WHERE project = %s AND status IN ('active', 'stale')",
        (project,)
    )
    active_count = cur.fetchone()["count"]
    dynamic_rate = cfg["decay.passive_base"] / max(1.0, active_count / cfg["decay.reference_pool_size"])

    archive_thresh = cfg["decay.archive_threshold"]
    stale_thresh = cfg["decay.stale_threshold"]

    # Passive decay: only affects memories BELOW the sim floor (truly unrelated to this recall).
    # Memories near the top-k boundary but above the floor are exempt — they're relevant,
    # they just didn't make the cut this time.
    # Decay is further scaled by category multiplier and the per-memory decay_multiplier.
    if retrieved_ids:
        placeholders = ",".join(["%s"] * len(retrieved_ids))
        cur.execute(f"""
            WITH decay_calc AS (
                SELECT id,
                    GREATEST(0.0,
                        confidence - %s * {cat_case} * decay_multiplier
                    ) AS new_confidence
                FROM memories
                WHERE project = %s AND status IN ('active', 'stale')
                  AND id NOT IN ({placeholders})
                  AND 1 - (content_embedding <=> %s::vector) < %s
            )
            UPDATE memories m
            SET confidence = dc.new_confidence,
                status = CASE
                    WHEN dc.new_confidence < %s THEN 'archived'
                    WHEN dc.new_confidence < %s THEN 'stale'
                    ELSE m.status
                END,
                updated_at = NOW()
            FROM decay_calc dc
            WHERE m.id = dc.id
        """, [dynamic_rate, project] + list(retrieved_ids) + [query_vec, sim_floor,
              archive_thresh, stale_thresh])
    else:
        # No retrievals — decay everything below the sim floor
        cur.execute(f"""
            WITH decay_calc AS (
                SELECT id,
                    GREATEST(0.0,
                        confidence - %s * {cat_case} * decay_multiplier
                    ) AS new_confidence
                FROM memories
                WHERE project = %s AND status IN ('active', 'stale')
                  AND 1 - (content_embedding <=> %s::vector) < %s
            )
            UPDATE memories m
            SET confidence = dc.new_confidence,
                status = CASE
                    WHEN dc.new_confidence < %s THEN 'archived'
                    WHEN dc.new_confidence < %s THEN 'stale'
                    ELSE m.status
                END,
                updated_at = NOW()
            FROM decay_calc dc
            WHERE m.id = dc.id
        """, [dynamic_rate, project, query_vec, sim_floor, archive_thresh, stale_thresh])

    conn.commit()
    return memories


def cmd_revitalize(args):
    """Boost all decayed memories back to a minimum confidence floor."""
    conn = get_conn(args.db_url)
    _ensure_migrations(conn)
    cfg = get_config(conn)
    cur = conn.cursor()

    where_clauses = ["project = %s", "status IN ('active', 'stale')", "confidence < %s"]
    params = [args.project, args.min_confidence]

    if args.category:
        cats = [c.strip() for c in args.category.split(",")]
        placeholders = ",".join(["%s"] * len(cats))
        where_clauses.append(f"category IN ({placeholders})")
        params.extend(cats)

    where = " AND ".join(where_clauses)

    if args.dry_run:
        cur.execute(f"SELECT COUNT(*) FROM memories WHERE {where}", params)
        count = cur.fetchone()[0]
        print(f"Would revitalize {count} memories to min confidence {args.min_confidence}")
        conn.close()
        return

    cur.execute(f"""
        UPDATE memories
        SET confidence = %s,
            status = CASE WHEN %s >= %s THEN 'active' ELSE status END,
            updated_at = NOW()
        WHERE {where}
    """, [args.min_confidence, args.min_confidence, cfg["decay.stale_threshold"]] + params)

    affected = cur.rowcount
    conn.commit()
    conn.close()
    print(f"Revitalized {affected} memories to confidence >= {args.min_confidence}")


def cmd_decay(args):
    """Run manual decay cycle on memories (maintenance tool)."""
    conn = get_conn(args.db_url)
    _ensure_migrations(conn)
    cfg = get_config(conn)
    cat_case = _build_category_case(cfg["decay.category_multipliers"])
    cur = conn.cursor()

    # Use DB config as defaults; CLI flags override if explicitly provided
    decay_rate = args.decay_rate if args.decay_rate is not None else cfg["decay.passive_base"]
    stale_thresh = args.stale_threshold if args.stale_threshold is not None else cfg["decay.stale_threshold"]
    archive_thresh = args.archive_threshold if args.archive_threshold is not None else cfg["decay.archive_threshold"]

    if args.dry_run:
        cur.execute(f"""
            SELECT id, category, content, confidence, decay_multiplier,
                   confidence - %s * {cat_case} * decay_multiplier AS projected_confidence
            FROM memories
            WHERE project = %s AND status IN ('active', 'stale')
            ORDER BY confidence ASC
        """, (decay_rate, args.project))

        print(f"=== Decay Dry Run ({args.project}) ===\n")
        for row in cur.fetchall():
            new_conf = max(0, row[5])
            status = "active"
            if new_conf < archive_thresh:
                status = "→ archived"
            elif new_conf < stale_thresh:
                status = "→ stale"
            print(f"  [{row[3]:.3f} → {new_conf:.3f}] {status:12s} [{row[1]:12s}] | {row[2][:70]}")
    else:
        cur.execute(f"""
            WITH decay_calc AS (
                SELECT id,
                    GREATEST(0.0,
                        confidence - %s * {cat_case} * decay_multiplier
                    ) AS new_confidence
                FROM memories
                WHERE project = %s AND status IN ('active', 'stale')
            )
            UPDATE memories m
            SET confidence = dc.new_confidence,
                status = CASE
                    WHEN dc.new_confidence < %s THEN 'archived'
                    WHEN dc.new_confidence < %s THEN 'stale'
                    ELSE m.status
                END,
                updated_at = NOW()
            FROM decay_calc dc
            WHERE m.id = dc.id
        """, (decay_rate, args.project, archive_thresh, stale_thresh))
        affected = cur.rowcount
        conn.commit()
        print(f"Decayed {affected} memories in project '{args.project}'.")

    conn.close()


def cmd_inspect(args):
    """Browse memories for a project."""
    conn = get_conn(args.db_url)
    _ensure_migrations(conn)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    where_clauses = ["project = %s"]
    params = [args.project]

    if args.status:
        where_clauses.append("status = %s")
        params.append(args.status)

    if args.category:
        where_clauses.append("category = %s")
        params.append(args.category)

    where = " AND ".join(where_clauses)
    sort_col = args.sort if args.sort else "created_at"

    cur.execute(f"""
        SELECT * FROM memories
        WHERE {where}
        ORDER BY {sort_col} DESC
        LIMIT %s
    """, params + [args.limit])

    rows = cur.fetchall()
    print(f"=== memories ({args.project}) — {len(rows)} results ===\n")
    for row in rows:
        row_clean = {k: (v.isoformat() if isinstance(v, datetime) else v)
                     for k, v in row.items()
                     if not k.endswith("_embedding") and v is not None}
        print(json.dumps(row_clean, indent=2, default=str))
        print()

    conn.close()


def cmd_add(args):
    """Add a single memory directly (no session file needed)."""
    content = sys.stdin.read().strip() if args.content == "-" else args.content
    if not content.strip():
        print("ERROR: Empty content.")
        sys.exit(1)

    if API_URL:
        res = _api("POST", "/v1/memories", {
            "project": args.project, "category": args.category,
            "content": content, "longevity": args.longevity,
        })
        mem_id, superseded = res["id"], res["superseded"]
    else:
        conn = get_conn(args.db_url)
        _ensure_migrations(conn)
        mem_id, superseded = add_memory(conn, args.project, args.category, content, args.longevity)
        conn.close()

    longevity_note = f" (longevity: {args.longevity})" if args.longevity != "normal" else ""
    print(f"Stored: {mem_id}{longevity_note}")
    if superseded:
        print(f"Superseded {superseded} similar memories")


def add_memory(conn, project: str, category: str, content: str, longevity: str = "normal",
               created_by: Optional[str] = None) -> tuple[str, int]:
    """Store one memory and supersede its near-duplicates. Commits; returns (id, superseded)."""
    cfg = get_config(conn)

    decay_multiplier = cfg["decay.longevity_values"].get(longevity, 1.0)
    supersede_thresh = cfg["supersede.similarity_threshold"]

    cur = conn.cursor()

    content_vec = embed(content)

    mem_id = str(uuid.uuid4())
    cur.execute("""
        INSERT INTO memories (id, project, category, content, content_embedding, decay_multiplier, created_by)
        VALUES (%s, %s, %s, %s, %s::vector, %s, %s)
    """, (mem_id, project, category, content, content_vec, decay_multiplier, created_by))

    # Check for near-duplicates and supersede them (cross-category)
    superseded = 0
    cur.execute("""
        SELECT id, content, 1 - (content_embedding <=> %s::vector) AS similarity
        FROM memories
        WHERE project = %s AND status = 'active' AND id != %s
        ORDER BY similarity DESC
        LIMIT 5
    """, (content_vec, project, mem_id))

    for row in cur.fetchall():
        if row[2] > supersede_thresh:
            cur.execute("""
                UPDATE memories SET status = 'superseded', superseded_by = %s, updated_at = NOW()
                WHERE id = %s
            """, (mem_id, row[0]))
            superseded += 1

    conn.commit()
    return mem_id, superseded


def cmd_forget(args):
    """Archive or hard-delete a memory."""
    conn = get_conn(args.db_url)
    _ensure_migrations(conn)
    cur = conn.cursor()

    if args.hard:
        cur.execute("DELETE FROM memories WHERE id = %s", (args.id,))
        action = "Deleted"
    else:
        cur.execute("UPDATE memories SET status = 'archived', updated_at = NOW() WHERE id = %s",
                    (args.id,))
        action = "Archived"

    if cur.rowcount == 0:
        print(f"No memory found with id {args.id}")
    else:
        print(f"{action} memory {args.id}")

    conn.commit()
    conn.close()


def hash_token(token: str) -> str:
    import hashlib
    return hashlib.sha256(token.encode()).hexdigest()


def cmd_key(args):
    """Manage API keys for api.py (direct DB access only)."""
    conn = get_conn(args.db_url)
    _ensure_migrations(conn)
    cur = conn.cursor()

    if args.key_command == "add":
        scopes = [s.strip() for s in args.scopes.split(",") if s.strip()]
        bad = [s for s in scopes if s not in API_SCOPES]
        if bad or not scopes:
            sys.exit(f"ERROR: scopes must be some of {','.join(API_SCOPES)}")
        import secrets
        token = "am_" + secrets.token_urlsafe(32)
        try:
            cur.execute("INSERT INTO api_keys (name, token_hash, scopes) VALUES (%s, %s, %s)",
                        (args.name, hash_token(token), scopes))
        except psycopg2.errors.UniqueViolation:
            sys.exit(f"ERROR: a key named '{args.name}' exists; revoke it first.")
        conn.commit()
        print(f"AGENT_MEMORY_API_TOKEN={token}")
        print(f"(key '{args.name}', scopes {','.join(scopes)} — shown once, only its hash is stored)",
              file=sys.stderr)
    elif args.key_command == "list":
        cur.execute("SELECT name, scopes, created_at, last_used_at FROM api_keys ORDER BY name")
        rows = cur.fetchall()
        if not rows:
            print("No API keys.")
        for name, scopes, created, used in rows:
            used_s = used.strftime("%Y-%m-%d %H:%M") if used else "never"
            print(f"{name:20s} {','.join(scopes):12s} created {created:%Y-%m-%d}  last used {used_s}")
    elif args.key_command == "revoke":
        cur.execute("DELETE FROM api_keys WHERE name = %s", (args.name,))
        conn.commit()
        print(f"Revoked key '{args.name}'" if cur.rowcount else f"No key named '{args.name}'")
    conn.close()


def cmd_migrate(args):
    """Apply pending schema migrations."""
    conn = get_conn(args.db_url)
    cur = conn.cursor()
    current = _get_schema_version(cur)
    print(f"Current schema version: v{current}")
    pending = [v for v in sorted(MIGRATIONS.keys()) if v > current]
    if not pending:
        print("No pending migrations.")
        conn.close()
        return
    for version in pending:
        print(f"Applying migration v{version}...")
        cur.execute(MIGRATIONS[version])
        conn.commit()
        print(f"  v{version} applied.")
    conn.close()
    print("Done.")


def cmd_export(args):
    """Export project memories to JSON."""
    conn = get_conn(args.db_url)
    _ensure_migrations(conn)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    where = "WHERE project = %s"
    params = [args.project]
    if not args.include_archived:
        where += " AND status != 'archived'"

    cur.execute(f"SELECT * FROM memories {where}", params)
    rows = cur.fetchall()
    data = {
        "project": args.project,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "memories": [
            {k: (v.isoformat() if isinstance(v, datetime) else
                 v.tolist() if hasattr(v, 'tolist') else v)
             for k, v in row.items()}
            for row in rows
        ]
    }
    conn.close()

    with open(args.output, "w") as f:
        json.dump(data, f, indent=2, default=str)

    print(f"Exported {len(data['memories'])} memories to {args.output}")


def cmd_import(args):
    """Import from a backup JSON file."""
    with open(args.input, "r") as f:
        data = json.load(f)

    project = args.project or data.get("project", "unknown")
    conn = get_conn(args.db_url)
    _ensure_migrations(conn)
    cur = conn.cursor()

    imported = 0
    for row in data.get("memories", []):
        row_id = row.get("id")
        cur.execute("SELECT 1 FROM memories WHERE id = %s", (row_id,))
        if cur.fetchone():
            continue

        if args.project:
            row["project"] = project

        row = {k: v for k, v in row.items() if not k.endswith("_embedding")}
        cols = ", ".join(row.keys())
        placeholders = ", ".join(["%s"] * len(row))
        cur.execute(f"INSERT INTO memories ({cols}) VALUES ({placeholders})", list(row.values()))
        imported += 1

    conn.commit()
    conn.close()
    print(f"Imported {imported} memories into project '{project}'")


# ─── Supersede check (internal) ──────────────────────────────────────────────

def _check_contradictions(cur, project: str, new_memories: list[dict],
                          new_id: Optional[str] = None, supersede_thresh: float = 0.85) -> int:
    """Check if new memories contradict existing ones. High similarity → supersede (cross-category)."""
    superseded = 0
    for mem in new_memories:
        vec = embed(mem["content"])
        if new_id:
            cur.execute("""
                SELECT id, content, 1 - (content_embedding <=> %s::vector) AS similarity
                FROM memories
                WHERE project = %s AND status = 'active' AND id != %s
                ORDER BY similarity DESC
                LIMIT 5
            """, (vec, project, new_id))
        else:
            cur.execute("""
                SELECT id, content, 1 - (content_embedding <=> %s::vector) AS similarity
                FROM memories
                WHERE project = %s AND status = 'active'
                ORDER BY similarity DESC
                LIMIT 5
            """, (vec, project))

        for row in cur.fetchall():
            if row[2] > supersede_thresh:
                cur.execute("""
                    UPDATE memories SET status = 'superseded', superseded_by = %s, updated_at = NOW()
                    WHERE id = %s
                """, (new_id, row[0]))
                superseded += 1

    return superseded


# ─── Output Formatting ───────────────────────────────────────────────────────

def _print_recall_markdown(results: dict, query: str):
    """Pretty-print recall results as markdown."""
    if results["memories"]:
        print(f"## Memories ({len(results['memories'])} found)\n")
        for m in results["memories"]:
            print(f"- **[{m['category']}]** (conf: {m['confidence']:.2f} | sim: {m['similarity']:.2f})")
            print(f"  {m['content']}")
            print()
    else:
        print("_No memories found. Fresh start or query needs refinement._")

    if results.get("_trimmed"):
        print(f"_⚠ {results['_trimmed']} item(s) trimmed to fit token budget._\n")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Agent Memory CLI")
    parser.add_argument("--db-url", default=DEFAULT_DB_URL, help="Database connection URL")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # status
    p_status = subparsers.add_parser("status", help="Health check")
    p_status.add_argument("--project", help="Filter by project")

    # recall
    p_recall = subparsers.add_parser("recall", help="Retrieve memories")
    p_recall.add_argument("--query", required=True, help="What you're about to work on")
    p_recall.add_argument("--project", required=True, help="Project name")
    p_recall.add_argument("--top-k", type=int, default=10, help="Max results")
    p_recall.add_argument("--max-tokens", type=int, default=None,
                          help="Token budget; trims lowest-similarity items first")
    p_recall.add_argument("--category", help="Comma-separated categories to filter")
    p_recall.add_argument("--format", default="markdown", choices=["markdown", "json"])

    # revitalize
    p_revitalize = subparsers.add_parser("revitalize",
        help="Boost decayed memories back to a minimum confidence floor")
    p_revitalize.add_argument("--project", required=True)
    p_revitalize.add_argument("--min-confidence", type=float, default=0.7,
                               help="Floor confidence to restore to (default: 0.7)")
    p_revitalize.add_argument("--category", help="Comma-separated categories to limit scope")
    p_revitalize.add_argument("--dry-run", action="store_true")

    # decay (manual maintenance)
    p_decay = subparsers.add_parser("decay", help="Run manual decay cycle")
    p_decay.add_argument("--project", required=True)
    p_decay.add_argument("--dry-run", action="store_true")
    p_decay.add_argument("--decay-rate", type=float, default=None,
                         help="Flat confidence reduction per run (default: from config DB)")
    p_decay.add_argument("--stale-threshold", type=float, default=None,
                         help="Confidence below which status becomes stale (default: from config DB)")
    p_decay.add_argument("--archive-threshold", type=float, default=None,
                         help="Confidence below which status becomes archived (default: from config DB)")

    # migrate
    subparsers.add_parser("migrate", help="Apply pending schema migrations")

    # inspect
    p_inspect = subparsers.add_parser("inspect", help="Browse memories")
    p_inspect.add_argument("--project", required=True)
    p_inspect.add_argument("--status", choices=["active", "stale", "archived", "superseded"])
    p_inspect.add_argument("--category", choices=CATEGORIES)
    p_inspect.add_argument("--limit", type=int, default=20)
    p_inspect.add_argument("--sort", default="created_at",
                           choices=["confidence", "created_at", "last_referenced"])

    # add
    p_add = subparsers.add_parser("add", help="Add a single memory directly")
    p_add.add_argument("--project", required=True, help="Project name")
    p_add.add_argument("--category", required=True, choices=CATEGORIES, help="Memory category")
    p_add.add_argument("--content", required=True,
                       help="The memory content, or '-' to read from stdin")
    p_add.add_argument("--longevity", default="normal",
                       help="Decay resistance: high (0.2x, ~thousands of sessions), normal (default)")

    # forget
    p_forget = subparsers.add_parser("forget", help="Archive/delete a memory")
    p_forget.add_argument("--id", required=True, help="Memory UUID")
    p_forget.add_argument("--hard", action="store_true", help="Permanent delete")

    # export
    p_export = subparsers.add_parser("export", help="Export project memories")
    p_export.add_argument("--project", required=True)
    p_export.add_argument("--output", required=True, help="Output JSON path")
    p_export.add_argument("--include-archived", action="store_true")

    # import
    p_import = subparsers.add_parser("import", help="Import from backup")
    p_import.add_argument("--input", required=True, help="Input JSON path")
    p_import.add_argument("--project", help="Override project name")

    # key (API keys for api.py)
    p_key = subparsers.add_parser("key", help="Manage API keys for the memory server")
    key_sub = p_key.add_subparsers(dest="key_command", required=True)
    p_key_add = key_sub.add_parser("add", help="Create a key; prints its token once")
    p_key_add.add_argument("name", help="Who the key is for; stamped on memories it adds")
    p_key_add.add_argument("--scopes", default="recall",
                           help=f"Comma-separated, some of {','.join(API_SCOPES)} (default: recall)")
    key_sub.add_parser("list", help="List keys")
    p_key_revoke = key_sub.add_parser("revoke", help="Delete a key")
    p_key_revoke.add_argument("name")

    args = parser.parse_args()

    if API_URL and args.command not in ("status", "recall", "add"):
        sys.exit(f"ERROR: '{args.command}' is not available over the API "
                 "(AGENT_MEMORY_API_URL is set); only status, recall and add are.")

    cmd_map = {
        "status": cmd_status, "recall": cmd_recall,
        "revitalize": cmd_revitalize,
        "decay": cmd_decay, "inspect": cmd_inspect, "add": cmd_add,
        "forget": cmd_forget, "export": cmd_export, "import": cmd_import,
        "migrate": cmd_migrate, "key": cmd_key,
    }
    cmd_map[args.command](args)


if __name__ == "__main__":
    main()
