"""The CLI in remote mode (AGENT_MEMORY_API_URL) against a running api.py,
with psycopg2 unimportable on the client: status, add and recall work,
everything else is refused, server errors come through readably."""
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from werkzeug.serving import make_server

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import api  # noqa: E402

CLI = Path(__file__).resolve().parent.parent / "memory_cli.py"
# Runs the CLI as a client without psycopg2 or pgvector.
NO_DB = ("import sys, runpy; sys.modules['psycopg2'] = None; sys.modules['pgvector'] = None; "
         f"sys.argv = ['memory_cli.py'] + sys.argv[1:]; runpy.run_path({str(CLI)!r}, run_name='__main__')")


@pytest.fixture
def server(db):
    srv = make_server("127.0.0.1", 0, api.app, threaded=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


@pytest.fixture
def cli(server):
    def run(token, *args, stdin=None):
        env = {**os.environ, "AGENT_MEMORY_API_URL": server, "AGENT_MEMORY_API_TOKEN": token}
        return subprocess.run([sys.executable, "-c", NO_DB, *args], input=stdin,
                              capture_output=True, text=True, env=env, timeout=60)
    return run


def test_status_add_recall(cli, make_key):
    token = make_key("anna", "recall,add")
    st = cli(token, "status")
    assert st.returncode == 0, st.stderr
    assert "anna (scopes: recall,add)" in st.stdout

    added = cli(token, "add", "--project", "p", "--category", "gotcha", "--content", "-",
                stdin="Infomaniak answers some S3 calls with a spurious $403\n")
    assert added.returncode == 0, added.stderr
    assert added.stdout.startswith("Stored: ")

    found = cli(token, "recall", "--project", "p", "--query", "Infomaniak S3 calls spurious")
    assert found.returncode == 0, found.stderr
    assert "## Memories (1 found)" in found.stdout
    assert "spurious $403" in found.stdout

    as_json = cli(token, "recall", "--project", "p", "--query", "Infomaniak S3", "--format", "json",
                  "--max-tokens", "1")
    assert as_json.returncode == 0, as_json.stderr
    assert '"_trimmed": 1' in as_json.stdout


def test_recall_only_key_gets_a_readable_403(cli, make_key):
    token = make_key("anna", "recall")
    res = cli(token, "add", "--project", "p", "--category", "gotcha", "--content", "x")
    assert res.returncode != 0
    assert "403" in res.stderr and "'add' scope" in res.stderr


def test_bad_token_is_reported(cli, db):
    res = cli("am_wrong", "status")
    assert res.returncode != 0
    assert "401" in res.stderr


@pytest.mark.parametrize("args", [["forget", "--id", "x"], ["inspect", "--project", "p"],
                                  ["key", "list"], ["decay", "--project", "p"]])
def test_other_commands_are_refused(cli, make_key, args):
    token = make_key("anna", "recall,add")
    res = cli(token, *args)
    assert res.returncode != 0
    assert "not available over the API" in res.stderr


def test_unreachable_server(db):
    env = {**os.environ, "AGENT_MEMORY_API_URL": "http://127.0.0.1:9", "AGENT_MEMORY_API_TOKEN": "am_x"}
    res = subprocess.run([sys.executable, "-c", NO_DB, "status"], capture_output=True, text=True,
                         env=env, timeout=60)
    assert res.returncode != 0
    assert "unreachable" in res.stderr
