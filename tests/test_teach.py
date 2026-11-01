import io
import json
import time

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.services.teach import TeachError, parse_upload, validate_messages
from featherlm.data.chat_starter import INTENTS, build
from featherlm.training.checkpoint import load_model

from .test_api import LOCAL, REMOTE, env, settings  # noqa: F401  (env is a fixture)


@pytest.fixture()
def client(env):  # noqa: F811
    with TestClient(create_app(settings(env, max_training_steps=60)), client=LOCAL) as c:
        yield c


def test_starter_dataset_held_out_phrasings():
    train, val = build(corpus=None, seed=0, per_intent=20)
    seen = {json.dumps(r["messages"][0]["content"].lower().strip("?!. ")) for r in train}
    for v in val:
        if v["intent"] in INTENTS:
            q = v["messages"][0]["content"].lower().strip("?!. ")
            assert json.dumps(q) not in seen, q  # evaluation questions never appear in training
    assert len({v["intent"] for v in val}) >= len(INTENTS)


def test_validate_and_parse():
    with pytest.raises(TeachError):
        validate_messages([{"role": "user", "content": "hi"}])  # must end with assistant
    with pytest.raises(TeachError):
        validate_messages([{"role": "assistant", "content": "x"}])  # needs a user turn
    with pytest.raises(TeachError):
        validate_messages([{"role": "user", "content": " "}, {"role": "assistant", "content": "x"}])
    csv = b"question,answer\nWhat are your hours?,9 to 5\nWhere are you?,Lisbon\n"
    assert len(parse_upload("faq.csv", csv)) == 2
    txt = b"Q: Who runs the shop?\nA: Ammad.\n\nQ: Is it open on Sunday?\nA: No.\n"
    assert parse_upload("faq.txt", txt)[1][1]["content"] == "No."
    jl = (json.dumps({"prompt": "a", "completion": "b"}) + "\n" +
          json.dumps({"messages": [{"role": "user", "content": "c"}, {"role": "assistant", "content": "d"}]})).encode()
    assert len(parse_upload("x.jsonl", jl)) == 2
    for name, data in (("x.pdf", b"x"), ("x.csv", b"a,b\n1,2\n"), ("x.txt", b"nothing here"), ("x.jsonl", b"{bad")):
        with pytest.raises(TeachError):
            parse_upload(name, data)


def test_dataset_crud(client):
    assert client.post("/teach/datasets", json={"name": "shop"}).status_code == 201
    assert client.post("/teach/datasets", json={"name": "shop"}).status_code == 400  # exists
    assert client.post("/teach/datasets", json={"name": "../evil"}).status_code == 400
    assert client.post("/teach/datasets", json={"name": "starter"}).status_code == 400
    r = client.post("/teach/datasets/shop/examples", json={"question": "Hours?", "answer": "9 to 5."})
    assert r.status_code == 201
    ex_id = r.json()["id"]
    r = client.post("/teach/datasets/shop/examples", json={"messages": [
        {"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "where are you"}, {"role": "assistant", "content": "In Lisbon."}]})
    assert r.status_code == 201
    assert client.post("/teach/datasets/shop/examples", json={"question": "only a question"}).status_code == 422
    assert client.post("/teach/datasets/shop/examples", json={"messages": [
        {"role": "assistant", "content": "x"}]}).status_code == 400
    up = client.post("/teach/datasets/shop/upload",
                     files={"file": ("faq.csv", io.BytesIO(b"question,answer\nA?,B.\nC?,D.\n"), "text/csv")})
    assert up.status_code == 201 and up.json()["added"] == 2
    rows = client.get("/teach/datasets/shop/examples").json()["examples"]
    assert len(rows) == 4 and rows[-1]["id"] == ex_id  # newest first
    assert client.delete(f"/teach/datasets/shop/examples/{ex_id}").status_code == 200
    assert client.delete("/teach/datasets/shop/examples/nope").status_code == 404
    # adding to an unknown dataset creates it (used by "Teach a better answer" in chat)
    assert client.post("/teach/datasets/new-one/examples", json={"question": "q", "answer": "a"}).status_code == 201
    names = {d["name"]: d for d in client.get("/teach/datasets").json()["datasets"]}
    assert names["shop"]["examples"] == 3 and names["new-one"]["examples"] == 1
    assert client.delete("/teach/datasets/shop").status_code == 200


def test_teach_requires_admin(env):  # noqa: F811
    with TestClient(create_app(settings(env)), client=REMOTE) as c:
        assert c.get("/teach/datasets").status_code == 403
        assert c.post("/teach/datasets", json={"name": "x"}).status_code == 403
    with TestClient(create_app(settings(env, admin_key="adm")), client=REMOTE) as c:
        assert c.get("/teach/datasets", headers={"Authorization": "Bearer adm"}).status_code == 200


def test_teach_train_end_to_end(client):
    """Teach two answers, train a chat run through the API, then chat with the new model."""
    for q, a in (("What is the secret word?", "The secret word is pineapple."),
                 ("Who is the shop owner?", "The shop owner is Ammad.")):
        assert client.post("/teach/datasets/e2e/examples", json={"question": q, "answer": a}).status_code == 201
    assert client.post("/teach/train", json={"name": "e2e-run", "dataset": "empty-ds"}).status_code == 404
    base = client.get("/teach/base-models").json()["default"]
    r = client.post("/teach/train", json={"name": "e2e-run", "dataset": "e2e", "base_model": base, "steps": 40,
                                          "include_starter": False})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["taught_examples"] == 2 and body["model_id"] == "runs/e2e-run/best"
    assert client.post("/teach/train", json={"name": "e2e-run", "dataset": "e2e"}).status_code == 400  # name in use
    job_id = body["job"]["id"]
    deadline = time.time() + 600
    while time.time() < deadline:
        j = client.get(f"/training/jobs/{job_id}").json()
        if j["status"] not in ("running", "pending", "stopping"):
            break
        time.sleep(1)
    assert j["status"] == "completed", j
    assert j["kind"] == "chat"
    model = client.get("/v1/models/runs/e2e-run/best").json()["featherlm"]
    assert model["capabilities"]["chat"] is True
    chat = client.post("/v1/chat/completions", json={"model": "runs/e2e-run/best", "max_tokens": 20,
                                                     "temperature": 0,
                                                     "messages": [{"role": "user", "content": "What is the secret word?"}]})
    assert chat.status_code == 200, chat.text
    report = json.loads((client.app.state.settings.runs_dir / "e2e-run" / "finetune_report.json").read_text())
    assert report["val_loss_after"] < report["val_loss_before"]
    m, _, _ = load_model(client.app.state.settings.runs_dir / "e2e-run" / "best")
    assert m.cfg.chat_template == "featherlm-v1"
