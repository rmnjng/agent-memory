#!/usr/bin/env python3
"""
Agent Memory CLI — recall, decay, and manage agent memories.

Usage:
    python memory_cli.py <command> [options]

Commands:
    status      Health check
    recall      Retrieve relevant memories before a task
    decay       Run decay cycle (manual maintenance)
    inspect     Browse memories
    forget      Archive or delete a memory
    export      Dump project data to JSON
    import      Restore from backup
"""

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

# ─── Configuration ───────────────────────────────────────────────────────────

DEFAULT_DB_URL = os.environ.get(
    "AGENT_MEMORY_DB_URL",
    "postgresql://agent:agent_memory_pwd@localhost:5433/agent_memory"
)
DEFAULT_MODEL = os.environ.get(
    "AGENT_MEMORY_EMBED_MODEL",
    "nomic-ai/nomic-embed-text-v1.5"
)
EMBED_SERVER_URL = os.environ.get("AGENT_MEMORY_EMBED_URL", "http://localhost:7337")

CATEGORIES = ["convention", "gotcha", "pattern", "preference", "dependency", "optimization", "decision"]

# Passive decay applied to non-retrieved memories on each recall
PASSIVE_DECAY_PER_RECALL = 0.005
STALE_THRESHOLD = 0.3
ARCHIVE_THRESHOLD = 0.1

# Lazy-loaded local model fallback
_model = None


# ─── Embedding ───────────────────────────────────────────────────────────────

def _embed_via_server(text: str, mode: str = "document") -> "list[float] | None":
    """Try to embed via the web server. Returns None if unavailable."""
    try:
        import urllib.request
        payload = json.dumps({"text": text, "mode": mode}).encode()
        req = urllib.request.Request(
            f"{EMBED_SERVER_URL}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            return json.loads(resp.read())["embedding"]
    except Exception:
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
    conn = psycopg2.connect(db_url)
    register_vector(conn)
    return conn


# ─── Commands ────────────────────────────────────────────────────────────────

def cmd_status(args):
    """Health check: DB, schema, model, counts."""
    print("=== Agent Memory Status ===\n")

    try:
        conn = get_conn(args.db_url)
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
    query_vec = embed_query(args.query)

    conn = get_conn(args.db_url)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    results = {"memories": [], "_trimmed": 0}

    status_filter = "('active', 'stale')" if args.include_stale else "('active')"

    category_filter = ""
    params = [query_vec, args.project]
    if args.category:
        cats = [c.strip() for c in args.category.split(",")]
        placeholders = ",".join(["%s"] * len(cats))
        category_filter = f"AND category IN ({placeholders})"
        params.extend(cats)

    cur.execute(f"""
        SELECT id, category, content, confidence, status, last_referenced, created_at,
               1 - (content_embedding <=> %s::vector) AS similarity
        FROM memories
        WHERE project = %s AND status IN {status_filter} {category_filter}
        ORDER BY (1 - (content_embedding <=> %s::vector)) * 0.5
                 + confidence * 0.3
                 + (1.0 / (1.0 + EXTRACT(EPOCH FROM NOW() - last_referenced) / 86400.0 * 0.1)) * 0.2
        DESC
        LIMIT %s
    """, params + [query_vec] + [args.top_k])

    retrieved_ids = []
    for row in cur.fetchall():
        results["memories"].append({
            "category": row["category"],
            "content": row["content"],
            "confidence": round(row["confidence"], 3),
            "similarity": round(row["similarity"], 3),
        })
        retrieved_ids.append(row["id"])
        # Bump confidence for retrieved memories
        cur.execute("""
            UPDATE memories SET last_referenced = NOW(),
                   confidence = LEAST(1.0, confidence + 0.1)
            WHERE id = %s
        """, (row["id"],))

    # Passive decay: non-retrieved memories lose a tiny amount per recall
    if retrieved_ids:
        placeholders = ",".join(["%s"] * len(retrieved_ids))
        cur.execute(f"""
            UPDATE memories
            SET confidence = GREATEST(0.0, confidence - {PASSIVE_DECAY_PER_RECALL}),
                status = CASE
                    WHEN GREATEST(0.0, confidence - {PASSIVE_DECAY_PER_RECALL}) < {ARCHIVE_THRESHOLD} THEN 'archived'
                    WHEN GREATEST(0.0, confidence - {PASSIVE_DECAY_PER_RECALL}) < {STALE_THRESHOLD} THEN 'stale'
                    ELSE status
                END,
                updated_at = NOW()
            WHERE project = %s AND status IN ('active', 'stale')
              AND id NOT IN ({placeholders})
        """, [args.project] + list(retrieved_ids))
    else:
        cur.execute(f"""
            UPDATE memories
            SET confidence = GREATEST(0.0, confidence - {PASSIVE_DECAY_PER_RECALL}),
                status = CASE
                    WHEN GREATEST(0.0, confidence - {PASSIVE_DECAY_PER_RECALL}) < {ARCHIVE_THRESHOLD} THEN 'archived'
                    WHEN GREATEST(0.0, confidence - {PASSIVE_DECAY_PER_RECALL}) < {STALE_THRESHOLD} THEN 'stale'
                    ELSE status
                END,
                updated_at = NOW()
            WHERE project = %s AND status IN ('active', 'stale')
        """, [args.project])

    conn.commit()
    conn.close()

    if args.max_tokens is not None:
        results["_trimmed"] = _apply_token_budget(results, args.max_tokens)

    if args.format == "json":
        print(json.dumps(results, indent=2))
    else:
        _print_recall_markdown(results, args.query)


def cmd_decay(args):
    """Run manual decay cycle on memories (maintenance tool)."""
    conn = get_conn(args.db_url)
    cur = conn.cursor()

    if args.dry_run:
        cur.execute("""
            SELECT id, content, confidence,
                   confidence - %s AS projected_confidence
            FROM memories
            WHERE project = %s AND status IN ('active', 'stale')
            ORDER BY confidence ASC
        """, (args.decay_rate, args.project))

        print(f"=== Decay Dry Run ({args.project}) ===\n")
        for row in cur.fetchall():
            new_conf = max(0, row[3])
            status = "active"
            if new_conf < args.archive_threshold:
                status = "→ archived"
            elif new_conf < args.stale_threshold:
                status = "→ stale"
            print(f"  [{row[2]:.3f} → {new_conf:.3f}] {status:12s} | {row[1][:80]}")
    else:
        cur.execute("""
            UPDATE memories
            SET confidence = GREATEST(0.0, confidence - %s),
                status = CASE
                    WHEN GREATEST(0.0, confidence - %s) < %s THEN 'archived'
                    WHEN GREATEST(0.0, confidence - %s) < %s THEN 'stale'
                    ELSE status
                END,
                updated_at = NOW()
            WHERE project = %s AND status IN ('active', 'stale')
        """, (args.decay_rate, args.decay_rate, args.archive_threshold,
              args.decay_rate, args.stale_threshold, args.project))
        affected = cur.rowcount
        conn.commit()
        print(f"Decayed {affected} memories in project '{args.project}'.")

    conn.close()


def cmd_inspect(args):
    """Browse memories for a project."""
    conn = get_conn(args.db_url)
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

    conn = get_conn(args.db_url)
    cur = conn.cursor()

    content_vec = embed(content)

    mem_id = str(uuid.uuid4())
    cur.execute("""
        INSERT INTO memories (id, project, category, content, content_embedding)
        VALUES (%s, %s, %s, %s, %s::vector)
    """, (mem_id, args.project, args.category, content, content_vec))

    # Check for near-duplicates and supersede them
    superseded = 0
    cur.execute("""
        SELECT id, content, 1 - (content_embedding <=> %s::vector) AS similarity
        FROM memories
        WHERE project = %s AND category = %s AND status = 'active' AND id != %s
        ORDER BY similarity DESC
        LIMIT 3
    """, (content_vec, args.project, args.category, mem_id))

    for row in cur.fetchall():
        if row[2] > 0.85:
            cur.execute("""
                UPDATE memories SET status = 'superseded', updated_at = NOW()
                WHERE id = %s
            """, (row[0],))
            superseded += 1

    conn.commit()
    conn.close()

    print(f"Stored: {mem_id}")
    if superseded:
        print(f"Superseded {superseded} similar memories")


def cmd_forget(args):
    """Archive or hard-delete a memory."""
    conn = get_conn(args.db_url)
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


def cmd_export(args):
    """Export project memories to JSON."""
    conn = get_conn(args.db_url)
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


# ─── Decay (internal) ────────────────────────────────────────────────────────

def _check_contradictions(cur, project: str, new_memories: list[dict]) -> int:
    """Check if new memories contradict existing ones. Same category, high similarity → supersede."""
    superseded = 0
    for mem in new_memories:
        vec = embed(mem["content"])
        cur.execute("""
            SELECT id, content, 1 - (content_embedding <=> %s::vector) AS similarity
            FROM memories
            WHERE project = %s AND category = %s AND status = 'active'
            ORDER BY similarity DESC
            LIMIT 3
        """, (vec, project, mem["category"]))

        for row in cur.fetchall():
            if row[2] > 0.85:
                cur.execute("""
                    UPDATE memories SET status = 'superseded', updated_at = NOW()
                    WHERE id = %s
                """, (row[0],))
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
    p_recall.add_argument("--include-stale", action="store_true")
    p_recall.add_argument("--category", help="Comma-separated categories to filter")
    p_recall.add_argument("--format", default="markdown", choices=["markdown", "json"])

    # decay (manual maintenance)
    p_decay = subparsers.add_parser("decay", help="Run manual decay cycle")
    p_decay.add_argument("--project", required=True)
    p_decay.add_argument("--dry-run", action="store_true")
    p_decay.add_argument("--decay-rate", type=float, default=0.05,
                         help="Flat confidence reduction per run (default: 0.05)")
    p_decay.add_argument("--stale-threshold", type=float, default=STALE_THRESHOLD)
    p_decay.add_argument("--archive-threshold", type=float, default=ARCHIVE_THRESHOLD)

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

    args = parser.parse_args()

    cmd_map = {
        "status": cmd_status, "recall": cmd_recall,
        "decay": cmd_decay, "inspect": cmd_inspect, "add": cmd_add,
        "forget": cmd_forget, "export": cmd_export, "import": cmd_import,
    }
    cmd_map[args.command](args)


if __name__ == "__main__":
    main()
