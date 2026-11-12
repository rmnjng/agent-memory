"""A throwaway pgvector database (skipped without docker) and deterministic
embeddings: a bag of hashed words, so texts sharing words are similar."""
import hashlib
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psycopg2
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import bootstrap_db  # noqa: E402
import memory_cli  # noqa: E402

DIM = 768


def fake_embed(text: str) -> list[float]:
    vec = [0.0] * DIM
    for word in text.lower().split():
        vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _docker_ok() -> bool:
    return shutil.which("docker") is not None and subprocess.run(
        ["docker", "info"], capture_output=True).returncode == 0


@pytest.fixture(scope="session")
def db_url():
    if not _docker_ok():
        pytest.skip("needs docker")
    name = f"agent-memory-test-{os.getpid()}"
    subprocess.run(["docker", "run", "-d", "--name", name, "-p", "127.0.0.1::5432",
                    "-e", "POSTGRES_USER=agent", "-e", "POSTGRES_PASSWORD=x",
                    "-e", "POSTGRES_DB=agent_memory", bootstrap_db.DOCKER_IMAGE],
                   check=True, capture_output=True)
    try:
        port = subprocess.run(["docker", "port", name, "5432/tcp"], check=True,
                              capture_output=True, text=True).stdout.split(":")[-1].strip()
        url = f"postgresql://agent:x@127.0.0.1:{port}/agent_memory"
        # A raw connection: get_conn registers the vector type, which needs the schema first.
        for _ in range(60):
            try:
                conn = psycopg2.connect(url)
                break
            except psycopg2.OperationalError:
                time.sleep(1)
        else:
            pytest.fail("test database did not start")
        cur = conn.cursor()
        cur.execute(bootstrap_db.SCHEMA_SQL)
        conn.commit()
        conn.close()
        conn = memory_cli.get_conn(url)
        memory_cli._ensure_migrations(conn)
        conn.close()
        yield url
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


@pytest.fixture
def db(db_url, monkeypatch):
    """Empty memories and keys, fake embeddings, memory_cli pointed at the test DB."""
    conn = memory_cli.get_conn(db_url)
    cur = conn.cursor()
    cur.execute("TRUNCATE memories, api_keys CASCADE")
    conn.commit()
    monkeypatch.setattr(memory_cli, "DEFAULT_DB_URL", db_url)
    monkeypatch.setattr(memory_cli, "embed", fake_embed)
    monkeypatch.setattr(memory_cli, "embed_query", fake_embed)
    yield conn
    conn.close()


@pytest.fixture
def make_key(db, db_url, capsys):
    """make_key(name, scopes) -> token, through the CLI's own `key add`."""
    import argparse

    def make(name: str, scopes: str) -> str:
        memory_cli.cmd_key(argparse.Namespace(db_url=db_url, key_command="add",
                                              name=name, scopes=scopes))
        return capsys.readouterr().out.strip().split("=", 1)[1]
    return make
