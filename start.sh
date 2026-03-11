#!/usr/bin/env bash
set -e

CONTAINER="agent-memory-db"
WEB_PORT="${PORT:-7337}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Agent Memory ==="

# ── 1. Docker container ───────────────────────────────────────────────────────
if docker ps --filter "name=^${CONTAINER}$" --format "{{.Names}}" | grep -q "$CONTAINER"; then
  echo "DB:  container already running"
else
  if docker ps -a --filter "name=^${CONTAINER}$" --format "{{.Names}}" | grep -q "$CONTAINER"; then
    echo "DB:  starting existing container..."
    docker start "$CONTAINER" > /dev/null
  else
    echo "DB:  container not found — run bootstrap_db.py first"
    exit 1
  fi
fi

# ── 2. Wait for Postgres ──────────────────────────────────────────────────────
echo -n "DB:  waiting for postgres..."
for i in $(seq 1 15); do
  if docker exec "$CONTAINER" pg_isready -U agent > /dev/null 2>&1; then
    echo " ready"
    break
  fi
  echo -n "."
  sleep 1
done

# ── 3. Web viewer ─────────────────────────────────────────────────────────────
# Kill any previous instance on this port (fuser = Linux, lsof = macOS)
if command -v fuser &>/dev/null; then
  fuser -k "${WEB_PORT}/tcp" > /dev/null 2>&1 || true
else
  lsof -ti:"${WEB_PORT}" | xargs kill -9 > /dev/null 2>&1 || true
fi

echo "Web: starting viewer at http://localhost:${WEB_PORT}"
python3 "$SCRIPT_DIR/web.py" --port "$WEB_PORT"
