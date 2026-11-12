"""The CLI must use the shared embed server and only load its own model when the
server is unreachable, never merely because it is slow."""
import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import memory_cli  # noqa: E402

VEC = [0.1, 0.2, 0.3]


def _serve(delay: float, status: int = 200):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            time.sleep(delay)
            body = json.dumps({"embedding": VEC} if status == 200 else {"error": "x"}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.fixture
def model_loads(monkeypatch):
    calls = []

    class FakeModel:
        def encode(self, text, normalize_embeddings=True):
            import numpy as np
            return np.array([9.0])

    def fake_get_model():
        calls.append(1)
        return FakeModel()

    monkeypatch.setattr(memory_cli, "get_model", fake_get_model)
    return calls


def _point_at(monkeypatch, url, timeout):
    monkeypatch.setattr(memory_cli, "EMBED_SERVER_URL", url)
    monkeypatch.setattr(memory_cli, "EMBED_SERVER_TIMEOUT", timeout)


def test_fast_server_is_used(monkeypatch, model_loads):
    srv = _serve(delay=0)
    _point_at(monkeypatch, f"http://127.0.0.1:{srv.server_port}", 5)
    assert memory_cli.embed("hello") == VEC
    assert memory_cli.embed_query("hello") == VEC
    assert model_loads == []
    srv.shutdown()


def test_slow_but_within_timeout_waits_instead_of_loading(monkeypatch, model_loads):
    # The old 2 s timeout fell back here; 3 s is what the server takes under load.
    srv = _serve(delay=2.5)
    _point_at(monkeypatch, f"http://127.0.0.1:{srv.server_port}", 5)
    assert memory_cli.embed("long text") == VEC
    assert model_loads == []
    srv.shutdown()


def test_timeout_raises_and_does_not_load_model(monkeypatch, model_loads):
    srv = _serve(delay=2)
    _point_at(monkeypatch, f"http://127.0.0.1:{srv.server_port}", 0.5)
    with pytest.raises(RuntimeError, match="did not answer"):
        memory_cli.embed("text")
    assert model_loads == []
    srv.shutdown()


def test_unreachable_server_falls_back_to_local_model(monkeypatch, model_loads):
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()  # nothing listens here → connection refused
    _point_at(monkeypatch, f"http://127.0.0.1:{port}", 5)
    assert memory_cli.embed("text") == [9.0]
    assert model_loads == [1]


def test_server_error_falls_back_to_local_model(monkeypatch, model_loads):
    srv = _serve(delay=0, status=500)
    _point_at(monkeypatch, f"http://127.0.0.1:{srv.server_port}", 5)
    assert memory_cli.embed("text") == [9.0]
    assert model_loads == [1]
    srv.shutdown()
