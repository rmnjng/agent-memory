# agent-memory

Project: `agent-memory`

Persistent semantic memory system for AI coding agents. Stores per-project memories (conventions, gotchas, patterns, decisions) in a local pgvector database and retrieves the most relevant ones via cosine similarity.

## Stack

- Python 3.10+, PostgreSQL + pgvector (Docker), Flask
- Embedding model: `nomic-ai/nomic-embed-text-v1.5` (served via the web process)
- CLI: `memory_cli.py` — add / recall / revitalize / inspect / forget / decay / migrate / export / import / status
- Web UI: `web.py` — localhost:7337 (UMAP scatter, force graph, timeline, HDBSCAN clusters, config)
- DB: `postgresql://agent:agent_memory_pwd@localhost:5433/agent_memory`

## Key files

- `memory_cli.py` — CLI entry point
- `web.py` — Flask web UI + embedding server
- `bootstrap_db.py` — Docker DB setup + schema
- `install.sh` — end-to-end installer
- `claude/` — recall/remember command templates for Claude Code integration
