# agent-memory

Persistent semantic memory for AI coding agents. Stores per-project memories, architectural decisions, and past errors in a local vector database. Claude (and other agents) recall relevant context before each task and save learnings when done.

## How it works

- **Add** — agent stores a memory after completing work: a convention learned, a gotcha hit, a decision made
- **Recall** — agent queries before starting a task; gets back the most semantically similar memories ranked by similarity + confidence + recency
- **Decay** — on each recall, retrieved memories gain confidence (+0.1); unrelated ones (similarity < 0.4 to the query) decay gradually. Rate is dynamic — slows automatically as the pool grows. Category matters: gotchas and conventions decay 5× slower than preferences. Per-memory longevity flags (`--longevity high|normal`) add another layer of control
- **Revitalize** — run `revitalize` to floor-raise all decayed memories back to a baseline confidence after a long sprint on an unrelated feature
- **Web UI** — browse, search, and manage memories at `localhost:7337`. Includes a UMAP scatter-plot map, a force-directed similarity graph, a creation timeline, and HDBSCAN cluster analysis. The map and graph have query drawers: type a query to embed it in real time and see which memories light up

The embedding model (`nomic-ai/nomic-embed-text-v1.5`) runs as a persistent HTTP server inside the web process, so the CLI avoids cold-loading a 270MB model on every call.

## Prerequisites

- Python 3.10+
- Docker

## Quick start

```bash
git clone https://github.com/rmnjng/agent-memory.git ~/agent-memory
cd ~/agent-memory
./install.sh
```

The installer handles everything: Python deps, Docker database setup, Claude Code integration, and optionally the systemd auto-start service. It asks for your project name and patches `~/.claude/CLAUDE.md` automatically.

**Non-interactive (for scripting or Claude Code):**
```bash
./install.sh --project myapp --yes
```

Claude will then automatically recall before tasks and remember after significant work — no prompting needed.

### Manual setup (step by step)

```bash
# 1. Install Python deps
pip install -r requirements.txt

# 2. Bootstrap the database (pulls pgvector Docker image, creates schema)
python3 bootstrap_db.py

# 3. Start the web UI + embedding server
./start.sh
# → http://localhost:7337
```

**Claude Code integration** — copy the command files, replace `YOUR_PROJECT`:
```bash
cp claude/recall.md  ~/.claude/commands/recall.md
cp claude/remember.md ~/.claude/commands/remember.md
# edit both files: replace YOUR_PROJECT with your project name
```

Add to `~/.claude/CLAUDE.md`:
```markdown
## Agent Memory

This machine runs a persistent semantic memory system for coding agents.
It stores per-project learnings (conventions, gotchas, patterns, decisions) in a local
vector database and retrieves the most relevant ones before each task via cosine similarity.

- **Web UI**: http://localhost:7337 — browse, search, edit, and manage all memories
- **CLI**: `~/agent-memory/memory_cli.py` — add / recall / inspect / forget / decay / export
- **DB**: `postgresql://agent:agent_memory_pwd@localhost:5433/agent_memory`
- **Logs** (if auto-started): `tail -f /tmp/agent-memory.log`

If the DB or web UI is unreachable, start it with: `~/agent-memory/start.sh`

## Skills

@~/.claude/commands/recall.md
@~/.claude/commands/remember.md
```

**Auto-start on login — Linux (systemd):**
```bash
cp agent-memory.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now agent-memory.service
```

**Auto-start on login — macOS (launchd):**
```bash
sed "s|__HOME__|$HOME|g" agent-memory.plist > ~/Library/LaunchAgents/com.agent-memory.plist
launchctl load ~/Library/LaunchAgents/com.agent-memory.plist
# Logs: tail -f /tmp/agent-memory.log
```

## CLI reference

```bash
# Add a memory (use heredoc to avoid shell interpolation of special chars)
python3 memory_cli.py add \
  --project myapp \
  --category gotcha \
  --content - <<'MEMORY'
The S3 bucket env var is STORAGE_BUCKET_NAME, not S3_BUCKET.
MEMORY

# Add a long-lived memory (good for core architectural decisions — decays over thousands of sessions)
python3 memory_cli.py add \
  --project myapp \
  --category decision \
  --longevity high \
  --content - <<'MEMORY'
Auth tokens are stored in httpOnly cookies, never localStorage — security requirement.
MEMORY

# Recall before a task
python3 memory_cli.py recall \
  --project myapp \
  --query "add file upload to the API" \
  --top-k 10 \
  --format markdown

# Health check
python3 memory_cli.py status --project myapp

# Browse memories
python3 memory_cli.py inspect --project myapp --status active

# Revitalize — boost all decayed memories back to a confidence floor
# (useful after a long sprint on feature X that buried feature Y memories)
python3 memory_cli.py revitalize --project myapp
python3 memory_cli.py revitalize --project myapp --dry-run          # preview count
python3 memory_cli.py revitalize --project myapp --min-confidence 0.8  # stricter floor

# Apply pending schema migrations (safe to re-run)
python3 memory_cli.py migrate

# Run decay cycle manually
python3 memory_cli.py decay --project myapp --dry-run

# Archive a memory (or hard-delete with --hard)
python3 memory_cli.py forget --id <uuid>

# Export / import
python3 memory_cli.py export --project myapp --output backup.json
python3 memory_cli.py import --input backup.json
```

### Memory categories

| Category | Use for |
|---|---|
| `convention` | Naming rules, code style, file structure |
| `gotcha` | Traps, surprises, things that broke |
| `pattern` | Recurring approaches and workflows |
| `decision` | Architectural choices that constrain future work |
| `preference` | User or team preferences |

## Architecture

```
bootstrap_db.py   — one-time setup: pulls pgvector Docker image, creates schema
memory_cli.py     — CLI: add / recall / decay / inspect / export / import / status
web.py            — Flask web UI at :7337 + POST /api/embed embedding server
                     Tabs: Memories (list), Map (UMAP 2D), Graph (force-directed),
                     Timeline (stacked bar), Clusters (HDBSCAN)
                     APIs: /api/memories, /api/map, /api/map/query, /api/graph,
                     /api/timeline, /api/clusters, /api/embed
start.sh          — starts Docker container + web.py
agent-memory.service — systemd user service for auto-start
claude/           — Claude Code command files (recall.md, remember.md)
```

**Database:** PostgreSQL 16 + pgvector, running in Docker with a named volume (`agent-memory-data`) for persistence.

**Embedding model:** `nomic-ai/nomic-embed-text-v1.5` via sentence-transformers (768-dim). Downloaded once to the HuggingFace cache (~270MB). The web server loads it once and serves embeddings over HTTP; the CLI calls this endpoint with a 2s timeout and falls back to local model load if unavailable.

**Supersedence:** When a new memory is added, any existing active memory with cosine similarity > 0.85 is automatically marked as superseded, with `superseded_by` pointing to the new memory's ID. This works across categories — a `gotcha` can supersede a `pattern` if the content is similar enough. This prevents duplicates from accumulating as knowledge evolves.

**Decay:** Recall-based, not time-based. Every recall bumps retrieved memories (+0.1 confidence, capped at 1.0). Memories with cosine similarity < 0.4 to the current query are passively decayed — memories near the top-k boundary but still relevant are explicitly exempt. The decay rate is dynamic: `base_rate / max(1, active_count / 150)`, so a large pool decays more slowly. Categories have built-in multipliers (gotcha/convention: 0.2×, decision: 0.3×, pattern: 0.7×, preference: 1.0×), and each memory has an optional `decay_multiplier` column (set via `--longevity high`) for per-memory control. There is no "permanent" — even long-lived memories eventually fade if truly abandoned, preventing stale facts from accumulating forever. A memory that's never semantically relevant to any query gradually fades to stale (< 0.3) then archived (< 0.1). Dormant projects don't decay.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `AGENT_MEMORY_DB_URL` | `postgresql://agent:agent_memory_pwd@localhost:5433/agent_memory` | Database connection |
| `AGENT_MEMORY_EMBED_MODEL` | `nomic-ai/nomic-embed-text-v1.5` | Embedding model |
| `AGENT_MEMORY_EMBED_URL` | `http://localhost:7337` | Embedding server URL |
| `PORT` | `7337` | Web server port |
