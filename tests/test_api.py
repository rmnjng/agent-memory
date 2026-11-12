"""api.py: keys and scopes decide who may recall and add; adds are stamped
with the key's name and supersede like a local add; bad input is refused."""
import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import api  # noqa: E402
import memory_cli  # noqa: E402


@pytest.fixture
def client(db):
    return api.app.test_client()


def auth(token):
    return {"Authorization": f"Bearer {token}"}


def add(client, token, content, category="gotcha", **extra):
    return client.post("/v1/memories", headers=auth(token),
                       json={"project": "p", "category": category, "content": content, **extra})


def recall(client, token, query, **extra):
    return client.post("/v1/recall", headers=auth(token), json={"project": "p", "query": query, **extra})


def test_key_add_stores_only_the_hash(db, make_key):
    token = make_key("anna", "recall")
    assert token.startswith("am_") and len(token) > 40
    cur = db.cursor()
    cur.execute("SELECT name, token_hash, scopes FROM api_keys")
    rows = cur.fetchall()
    assert rows == [("anna", memory_cli.hash_token(token), ["recall"])]
    assert token not in str(rows)


def test_key_add_refuses_unknown_scope_and_duplicate_name(db, db_url, make_key):
    with pytest.raises(SystemExit, match="scopes must be"):
        make_key("anna", "recall,delete")
    make_key("anna", "recall")
    with pytest.raises(SystemExit, match="exists"):
        make_key("anna", "add")


def test_no_unknown_or_revoked_token_is_401(client, db_url, make_key):
    assert client.get("/v1/whoami").status_code == 401
    assert client.get("/v1/whoami", headers=auth("am_nope")).status_code == 401
    token = make_key("anna", "recall")
    assert client.get("/v1/whoami", headers=auth(token)).status_code == 200
    memory_cli.cmd_key(argparse.Namespace(db_url=db_url, key_command="revoke", name="anna"))
    assert client.get("/v1/whoami", headers=auth(token)).status_code == 401


def test_recall_only_key_cannot_add(client, make_key):
    token = make_key("anna", "recall")
    assert client.get("/v1/whoami", headers=auth(token)).get_json() == {"name": "anna", "scopes": ["recall"]}
    res = add(client, token, "something")
    assert res.status_code == 403 and "add" in res.get_json()["error"]
    assert recall(client, token, "something").status_code == 200


def test_add_only_key_cannot_recall(client, make_key):
    token = make_key("bot", "add")
    assert add(client, token, "something").status_code == 201
    assert recall(client, token, "something").status_code == 403


def test_add_is_stamped_and_recalled(client, db, make_key):
    token = make_key("anna", "recall,add")
    res = add(client, token, "staging storage probes time out on s3 pub2")
    assert res.status_code == 201
    mem_id = res.get_json()["id"]
    cur = db.cursor()
    cur.execute("SELECT created_by, project, category FROM memories WHERE id = %s", (mem_id,))
    assert cur.fetchone() == ("anna", "p", "gotcha")

    found = recall(client, token, "storage probes time out").get_json()["memories"]
    assert [m["content"] for m in found] == ["staging storage probes time out on s3 pub2"]
    assert set(found[0]) == {"category", "content", "confidence", "similarity"}


def test_near_duplicate_add_supersedes(client, db, make_key):
    token = make_key("anna", "add")
    first = add(client, token, "deploy staging with ops console build and deploy").get_json()["id"]
    res = add(client, token, "deploy staging with ops console build and deploy button").get_json()
    assert res["superseded"] == 1
    cur = db.cursor()
    cur.execute("SELECT status, superseded_by::text FROM memories WHERE id = %s", (first,))
    assert cur.fetchone() == ("superseded", res["id"])


@pytest.mark.parametrize("body", [
    {"category": "gotcha", "content": ""},
    {"category": "nonsense", "content": "x"},
    {"category": "gotcha", "content": "x" * 8001},
    {"category": "gotcha", "content": "x", "longevity": "forever"},
    {"category": "gotcha", "content": "x", "project": ""},
])
def test_bad_add_is_400(client, make_key, body):
    token = make_key("anna", "add")
    res = client.post("/v1/memories", headers=auth(token), json={"project": "p", **body})
    assert res.status_code == 400


@pytest.mark.parametrize("extra", [
    {"top_k": 0}, {"top_k": 51}, {"top_k": "5"}, {"top_k": True},
    {"category": "nonsense"}, {"category": 3}, {"query": ""},
])
def test_bad_recall_is_400(client, make_key, extra):
    token = make_key("anna", "recall")
    body = {"project": "p", "query": "q", **extra}
    assert client.post("/v1/recall", headers=auth(token), json=body).status_code == 400


def test_non_json_body_is_400(client, make_key):
    token = make_key("anna", "recall")
    res = client.post("/v1/recall", headers=auth(token), data="not json")
    assert res.status_code == 400
