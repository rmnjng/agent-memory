#!/usr/bin/env bash
# install.sh — set up agent-memory from scratch
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="$HOME/agent-memory"
CLAUDE_COMMANDS_DIR="$HOME/.claude/commands"
CLAUDE_MD="$HOME/.claude/CLAUDE.md"

# ── Flags ─────────────────────────────────────────────────────────────────────
PROJECT_NAME=""
AUTO_YES=false

for arg in "$@"; do
  case $arg in
    --project=*) PROJECT_NAME="${arg#*=}" ;;
    --project)   shift; PROJECT_NAME="$1" ;;
    --yes|-y)    AUTO_YES=true ;;
  esac
done

ask() {
  # ask <prompt> — returns 0 (yes) or 1 (no). In --yes mode, always yes.
  if $AUTO_YES; then return 0; fi
  echo -n "  $1 [Y/n]: "
  read -r _ans
  [[ "${_ans:-Y}" =~ ^[Yy]$ ]]
}

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BOLD='\033[1m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✓${NC} $*"; }
warn() { echo -e "${YELLOW}!${NC} $*"; }
err()  { echo -e "${RED}✗${NC} $*"; }
step() { echo -e "\n${BOLD}$*${NC}"; }

echo -e "${BOLD}=== agent-memory installer ===${NC}\n"

# ── 1. Prerequisites ──────────────────────────────────────────────────────────
step "1. Checking prerequisites..."

if ! command -v python3 &>/dev/null; then
  err "python3 not found. Install Python 3.10+ first."; exit 1
fi
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
ok "Python $PY_VER"

if ! command -v pip3 &>/dev/null && ! python3 -m pip --version &>/dev/null 2>&1; then
  err "pip not found. Install pip first."; exit 1
fi
ok "pip"

if ! command -v docker &>/dev/null; then
  err "Docker not found. Install Docker first: https://docs.docker.com/get-docker/"; exit 1
fi
if ! docker info &>/dev/null 2>&1; then
  err "Docker daemon is not running. Start Docker first."; exit 1
fi
ok "Docker"

# ── 2. Install directory ──────────────────────────────────────────────────────
step "2. Install directory..."

if [ "$SCRIPT_DIR" != "$INSTALL_DIR" ]; then
  if [ -d "$INSTALL_DIR" ]; then
    warn "$INSTALL_DIR already exists — updating files"
    cp "$SCRIPT_DIR"/*.py "$SCRIPT_DIR"/*.sh "$SCRIPT_DIR"/*.txt "$INSTALL_DIR/" 2>/dev/null || true
    cp -r "$SCRIPT_DIR/claude" "$INSTALL_DIR/" 2>/dev/null || true
  else
    cp -r "$SCRIPT_DIR" "$INSTALL_DIR"
    ok "Copied to $INSTALL_DIR"
  fi
else
  ok "Already in $INSTALL_DIR"
fi

chmod +x "$INSTALL_DIR/start.sh" "$INSTALL_DIR/install.sh"

# ── 3. Python deps ────────────────────────────────────────────────────────────
step "3. Installing Python dependencies..."
python3 -m pip install -q -r "$INSTALL_DIR/requirements.txt"
ok "Dependencies installed"

# ── 4. Database ───────────────────────────────────────────────────────────────
step "4. Setting up database..."
python3 "$INSTALL_DIR/bootstrap_db.py"

# ── 5. Project name ───────────────────────────────────────────────────────────
step "5. Claude Code integration..."

if [ -z "$PROJECT_NAME" ]; then
  if $AUTO_YES; then
    PROJECT_NAME="default"
  else
    echo -n "  Project name for this machine (e.g. myapp, work): "
    read -r PROJECT_NAME
    PROJECT_NAME="${PROJECT_NAME:-default}"
  fi
fi
ok "Project: $PROJECT_NAME"

mkdir -p "$CLAUDE_COMMANDS_DIR"
sed "s/YOUR_PROJECT/$PROJECT_NAME/g" "$INSTALL_DIR/claude/recall.md"   > "$CLAUDE_COMMANDS_DIR/recall.md"
sed "s/YOUR_PROJECT/$PROJECT_NAME/g" "$INSTALL_DIR/claude/remember.md" > "$CLAUDE_COMMANDS_DIR/remember.md"
ok "Installed recall.md and remember.md to $CLAUDE_COMMANDS_DIR (project: $PROJECT_NAME)"

# ── 6. CLAUDE.md snippet ──────────────────────────────────────────────────────
SNIPPET=$(cat <<EOF

## Agent Memory

This machine runs a persistent semantic memory system for coding agents.
It stores per-project learnings (conventions, gotchas, patterns, decisions) in a local
vector database and retrieves the most relevant ones before each task via cosine similarity.

- **Web UI**: http://localhost:7337 — browse, search, edit, and manage all memories
- **CLI**: \`~/agent-memory/memory_cli.py\` — add / recall / inspect / forget / decay / export
- **DB**: \`postgresql://agent:agent_memory_pwd@localhost:5433/agent_memory\`
- **Logs** (if auto-started): \`tail -f /tmp/agent-memory.log\`

If the DB or web UI is unreachable, start it with: \`~/agent-memory/start.sh\`

## Skills

Two skills are always loaded. Follow their trigger conditions automatically.

@~/.claude/commands/recall.md
@~/.claude/commands/remember.md
EOF
)

if [ -f "$CLAUDE_MD" ] && grep -q "agent-memory" "$CLAUDE_MD" 2>/dev/null; then
  ok "CLAUDE.md already contains agent-memory config — skipping"
else
  if ask "Add agent-memory config to $CLAUDE_MD?"; then
    mkdir -p "$(dirname "$CLAUDE_MD")"
    printf '%s\n' "$SNIPPET" >> "$CLAUDE_MD"
    ok "Added snippet to $CLAUDE_MD"
  else
    echo ""
    warn "Skipped. Add this to $CLAUDE_MD manually:"
    echo "$SNIPPET"
  fi
fi

# ── 7. Auto-start on login ────────────────────────────────────────────────────
step "6. Auto-start on login..."

if command -v systemctl &>/dev/null && systemctl --user status &>/dev/null 2>&1; then
  # Linux: systemd user service
  if ask "Install systemd service (auto-start on login)?"; then
    SERVICE_DIR="$HOME/.config/systemd/user"
    mkdir -p "$SERVICE_DIR"
    cp "$INSTALL_DIR/agent-memory.service" "$SERVICE_DIR/"
    systemctl --user daemon-reload
    systemctl --user enable --now agent-memory.service
    ok "Systemd service installed and started"
  else
    ok "Skipped. Run manually: ./start.sh"
  fi
elif [[ "$(uname)" == "Darwin" ]]; then
  # macOS: LaunchAgent
  if ask "Install LaunchAgent (auto-start on login)?"; then
    PLIST_DIR="$HOME/Library/LaunchAgents"
    PLIST_DEST="$PLIST_DIR/com.agent-memory.plist"
    mkdir -p "$PLIST_DIR"
    sed "s|__HOME__|$HOME|g" "$INSTALL_DIR/agent-memory.plist" > "$PLIST_DEST"
    launchctl load "$PLIST_DEST"
    ok "LaunchAgent installed and started"
    ok "Logs: tail -f /tmp/agent-memory.log"
  else
    ok "Skipped. Run manually: ./start.sh"
  fi
else
  warn "Auto-start not supported on this platform. Run manually: ./start.sh"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo -e "\n${GREEN}${BOLD}All done!${NC}"
echo -e "  Web UI:  ${BOLD}http://localhost:7337${NC}"
echo -e "  Add a memory:  python3 ~/agent-memory/memory_cli.py add --project $PROJECT_NAME --category gotcha --content - <<'EOF'"
echo -e "  Check status:  python3 ~/agent-memory/memory_cli.py status --project $PROJECT_NAME"
