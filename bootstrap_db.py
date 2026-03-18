#!/usr/bin/env python3
"""
Bootstrap the agent memory database.
Pulls pgvector Docker image, starts container, creates schema.
"""

import argparse
import subprocess
import sys
import time

DOCKER_IMAGE = "pgvector/pgvector:pg16"
DEFAULT_CONTAINER = "agent-memory-db"
DEFAULT_PORT = 5433
DEFAULT_PASSWORD = "agent_memory_pwd"
DEFAULT_USER = "agent"
DEFAULT_DB = "agent_memory"

SCHEMA_SQL = """
-- Extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Memories: distilled insights
CREATE TABLE IF NOT EXISTS memories (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project         TEXT NOT NULL,
    category        TEXT NOT NULL CHECK (category IN (
                        'convention', 'pattern', 'preference',
                        'dependency', 'gotcha', 'optimization', 'decision'
                    )),
    content         TEXT NOT NULL,
    content_embedding VECTOR(768),
    confidence      REAL NOT NULL DEFAULT 1.0 CHECK (confidence >= 0.0 AND confidence <= 1.0),
    decay_multiplier REAL NOT NULL DEFAULT 1.0 CHECK (decay_multiplier >= 0.0 AND decay_multiplier <= 1.0),
    status          TEXT NOT NULL DEFAULT 'active' CHECK (status IN (
                        'active', 'stale', 'archived', 'superseded'
                    )),
    superseded_by   UUID REFERENCES memories(id),
    last_referenced TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_memories_project_status ON memories(project, status);
CREATE INDEX IF NOT EXISTS idx_memories_confidence ON memories(confidence DESC);

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER PRIMARY KEY,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

INSERT INTO schema_version (version) VALUES (3) ON CONFLICT DO NOTHING;
"""


def run(cmd: list[str], check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command."""
    return subprocess.run(cmd, check=check, capture_output=capture, text=True)


def docker_available() -> bool:
    try:
        run(["docker", "info"])
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def container_exists(name: str) -> bool:
    result = run(["docker", "ps", "-a", "--filter", f"name=^{name}$", "--format", "{{.Names}}"])
    return name in result.stdout.strip()


def container_running(name: str) -> bool:
    result = run(["docker", "ps", "--filter", f"name=^{name}$", "--format", "{{.Names}}"])
    return name in result.stdout.strip()


def wait_for_postgres(port: int, user: str, password: str, max_wait: int = 30) -> bool:
    """Wait for Postgres to accept connections."""
    import os
    env = os.environ.copy()
    env["PGPASSWORD"] = password

    for i in range(max_wait):
        try:
            result = subprocess.run(
                ["docker", "exec", DEFAULT_CONTAINER, "pg_isready", "-U", user],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                return True
        except Exception:
            pass
        time.sleep(1)
        if i % 5 == 0 and i > 0:
            print(f"  Still waiting for Postgres... ({i}s)")
    return False


def bootstrap(port: int, password: str, force: bool):
    print("=== Agent Memory Bootstrap ===\n")

    # Check Docker
    print("1. Checking Docker...")
    if not docker_available():
        print("   ERROR: Docker is not available. Install Docker first.")
        sys.exit(1)
    print("   Docker is available.\n")

    # Handle existing container
    if container_exists(DEFAULT_CONTAINER):
        if force:
            print(f"2. Removing existing container '{DEFAULT_CONTAINER}'...")
            if container_running(DEFAULT_CONTAINER):
                run(["docker", "stop", DEFAULT_CONTAINER])
            run(["docker", "rm", DEFAULT_CONTAINER])
            print("   Removed.\n")
        elif container_running(DEFAULT_CONTAINER):
            print(f"2. Container '{DEFAULT_CONTAINER}' already running. Use --force to recreate.\n")
        else:
            print(f"2. Starting existing container '{DEFAULT_CONTAINER}'...")
            run(["docker", "start", DEFAULT_CONTAINER])
            print("   Started.\n")
    else:
        print(f"2. Pulling image {DOCKER_IMAGE}...")
        run(["docker", "pull", DOCKER_IMAGE])
        print("   Pulled.\n")

        print(f"3. Starting container on port {port}...")
        run([
            "docker", "run", "-d",
            "--name", DEFAULT_CONTAINER,
            "-p", f"{port}:5432",
            "-e", f"POSTGRES_USER={DEFAULT_USER}",
            "-e", f"POSTGRES_PASSWORD={password}",
            "-e", f"POSTGRES_DB={DEFAULT_DB}",
            "-v", "agent-memory-data:/var/lib/postgresql/data",
            "--restart", "unless-stopped",
            DOCKER_IMAGE
        ])
        print("   Container started.\n")

    # Wait for Postgres
    print("4. Waiting for Postgres to be ready...")
    if not wait_for_postgres(port, DEFAULT_USER, password):
        print("   ERROR: Postgres did not become ready in time.")
        sys.exit(1)
    print("   Postgres is ready.\n")

    # Create schema
    print("5. Creating schema...")
    result = subprocess.run(
        [
            "docker", "exec", "-i", DEFAULT_CONTAINER,
            "psql", "-U", DEFAULT_USER, "-d", DEFAULT_DB
        ],
        input=SCHEMA_SQL,
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"   ERROR: Schema creation failed:\n{result.stderr}")
        sys.exit(1)
    print("   Schema created.\n")

    # Summary
    db_url = f"postgresql://{DEFAULT_USER}:{password}@localhost:{port}/{DEFAULT_DB}"
    print("=== Bootstrap Complete ===")
    print(f"  Container:  {DEFAULT_CONTAINER}")
    print(f"  Port:       {port}")
    print(f"  Database:   {DEFAULT_DB}")
    print(f"  Connection: {db_url}")
    print(f"\n  Export this for the CLI:")
    print(f"  export AGENT_MEMORY_DB_URL='{db_url}'")


def main():
    parser = argparse.ArgumentParser(description="Bootstrap agent memory database")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Host port (default: 5433)")
    parser.add_argument("--password", default=DEFAULT_PASSWORD, help="DB password")
    parser.add_argument("--force", action="store_true", help="Destroy and recreate container")
    args = parser.parse_args()
    bootstrap(args.port, args.password, args.force)


if __name__ == "__main__":
    main()
