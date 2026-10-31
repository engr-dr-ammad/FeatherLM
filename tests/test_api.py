import io
import json
import shutil
import time
import zipfile
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from api.main import create_app
from api.services.settings import Settings
from featherlm.data.dataset import CHAT_TEMPLATE
from featherlm.training.checkpoint import load_model, save_model

from .conftest import ROOT

LOCAL = ("127.0.0.1", 50000)
REMOTE = ("203.0.113.9", 50000)


@pytest.fixture(scope="module")
def env(trained_model, tmp_path_factory, synthetic_data):
    run, _ = trained_model
    root = tmp_path_factory.mktemp("server")
    models = root / "models"
    shutil.copytree(run / "final", models / "tiny")
    m, tok, _ = load_model(run / "final")
    m.cfg.chat_template = CHAT_TEMPLATE
    save_model(models / "tiny-chat", m, tok)
    (root / "configs").mkdir()
    shutil.copy(ROOT / "configs" / "smoke.yaml", root / "configs" / "smoke.yaml")
    (root / "data").mkdir()
    shutil.copytree(synthetic_data, root / "data" / "synthetic")
    (root / "results").mkdir()
    (root / "results" / "speed.json").write_text(json.dumps({"rows": [{"backend": "feather", "ttft_ms": 1.0}]}))
    return root


def settings(env, **kw):
    s = Settings()
    s.root = ROOT
    s.models_dir, s.runs_dir = env / "models", env / "runs"
    s.data_dir, s.configs_dir, s.results_dir = env / "data", env / "configs", env / "results"
    s.db_path = env / f"db-{time.time_ns()}.sqlite"
    s.max_training_steps = 30
    for k, v in kw.items():
        setattr(s, k, v)
    return s


@pytest.fixture()
def client(env):
    with TestClient(create_app(settings(env)), client=LOCAL) as c:
        yield c


def test_health_and_models(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["models_available"] == 2
    data = client.get("/v1/models").json()["data"]
    ids = {d["id"]: d for d in data}
    assert set(ids) == {"models/tiny", "models/tiny-chat"}
    assert ids["models/tiny"]["featherlm"]["capabilities"]["chat"] is False
    assert ids["models/tiny-chat"]["featherlm"]["capabilities"]["chat"] is True
    assert client.get("/v1/models/models/tiny").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_completion(client):
    r = client.post("/v1/completions", json={"model": "models/tiny", "prompt": "Ada sees", "max_tokens": 12,
                                             "temperature": 0})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["object"] == "text_completion" and j["usage"]["completion_tokens"] >= 1
    assert j["featherlm"]["tokens_per_sec"] > 0
    r2 = client.post("/v1/completions", json={"model": "models/tiny", "prompt": "Ada sees", "max_tokens": 12,
                                              "temperature": 0})
    assert r2.json()["choices"][0]["text"] == j["choices"][0]["text"]


@pytest.mark.parametrize("body", [
    {"prompt": "x", "temperature": -1}, {"prompt": "x", "top_p": 0}, {"prompt": ["a", "b"]}, {"prompt": "x", "n": 2},
    {"prompt": "x", "max_tokens": 0}, {"prompt": "x", "stop": ["s"] * 9}, {},
])
def test_completion_validation(client, body):
    assert client.post("/v1/completions", json={"model": "models/tiny", **body}).status_code == 422


def test_unknown_and_traversal_model_ids(client):
    for mid in ("models/nope", "models/../../etc", "elsewhere/tiny", "runs/../models/tiny"):
        r = client.post("/v1/completions", json={"model": mid, "prompt": "x"})
        assert r.status_code == 404, mid


def test_request_size_limit(env):
    with TestClient(create_app(settings(env, max_request_bytes=500)), client=LOCAL) as c:
        r = c.post("/v1/completions", json={"model": "models/tiny", "prompt": "x" * 1000})
        assert r.status_code == 413


def test_streaming_sse(client):
    with client.stream("POST", "/v1/completions", json={"model": "models/tiny", "prompt": "Ada", "max_tokens": 8,
                                                        "stream": True, "temperature": 0}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        events = [l[6:] for l in r.iter_lines() if l.startswith("data: ")]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    assert chunks[-1]["choices"][0]["finish_reason"] in ("length", "stop")
    assert "usage" in chunks[-1]
    streamed = "".join(c["choices"][0]["text"] for c in chunks)
    full = client.post("/v1/completions", json={"model": "models/tiny", "prompt": "Ada", "max_tokens": 8,
                                                "temperature": 0}).json()["choices"][0]["text"]
    assert streamed == full


def test_chat_requires_template(client):
    msgs = [{"role": "user", "content": "hello"}]
    assert client.post("/v1/chat/completions", json={"model": "models/tiny", "messages": msgs}).status_code == 400
    r = client.post("/v1/chat/completions", json={"model": "models/tiny-chat", "messages": msgs, "max_tokens": 5})
    assert r.status_code == 200, r.text
    assert r.json()["choices"][0]["message"]["role"] == "assistant"
    with client.stream("POST", "/v1/chat/completions", json={"model": "models/tiny-chat", "messages": msgs,
                                                             "max_tokens": 4, "stream": True}) as s:
        evs = [l[6:] for l in s.iter_lines() if l.startswith("data: ")]
    assert json.loads(evs[0])["object"] == "chat.completion.chunk" and evs[-1] == "[DONE]"
    bad = client.post("/v1/chat/completions", json={"model": "models/tiny-chat",
                                                    "messages": [{"role": "tool", "content": "x"}]})
    assert bad.status_code == 422


def test_api_key_enforced(env):
    with TestClient(create_app(settings(env, api_key="secret")), client=REMOTE) as c:
        assert c.get("/health").status_code == 200  # health stays open for orchestrators
        assert c.get("/v1/models").status_code == 401
        assert c.get("/v1/models", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert c.get("/v1/models", headers={"Authorization": "Bearer secret"}).status_code == 200
        assert c.get("/metrics").status_code == 401


def test_admin_protection(env):
    with TestClient(create_app(settings(env)), client=REMOTE) as c:  # no admin key, remote client
        assert c.get("/training/jobs").status_code == 403
        assert c.put("/settings", json={"threads": 1}).status_code == 403
    with TestClient(create_app(settings(env)), client=LOCAL) as c:  # local, but behind a proxy
        assert c.get("/training/jobs", headers={"X-Forwarded-For": "1.2.3.4"}).status_code == 403
    with TestClient(create_app(settings(env, admin_key="adm")), client=REMOTE) as c:
        assert c.get("/training/jobs").status_code == 401
        assert c.get("/training/jobs", headers={"Authorization": "Bearer adm"}).status_code == 200
    with TestClient(create_app(settings(env, enable_training_api=False)), client=LOCAL) as c:
        assert c.post("/training/jobs", json={"name": "x", "config": "smoke.yaml"}).status_code == 404


def test_training_job_lifecycle(env):
    with TestClient(create_app(settings(env)), client=LOCAL) as c:
        assert "smoke.yaml" in c.get("/training/configs").json()["configs"]
        for bad in ({"name": "../x", "config": "smoke.yaml"}, {"name": "ok", "config": "../etc.yaml"},
                    {"name": "ok", "config": "smoke.yaml", "overrides": {"output_dir": "/tmp"}},
                    {"name": "ok", "config": "smoke.yaml", "overrides": {"data.path": "../../etc"}}):
            assert c.post("/training/jobs", json=bad).status_code in (400, 422), bad
        r = c.post("/training/jobs", json={"name": "apijob", "config": "smoke.yaml",
                                           "overrides": {"training.max_steps": 10000, "data.path": "data/synthetic"}})
        assert r.status_code == 201, r.text
        job = r.json()
        assert c.post("/training/jobs", json={"name": "second", "config": "smoke.yaml"}).status_code == 400  # limit
        deadline = time.time() + 300
        while time.time() < deadline:
            j = c.get(f"/training/jobs/{job['id']}").json()
            if j["status"] not in ("running", "pending"):
                break
            time.sleep(1)
        assert j["status"] == "completed", j
        assert j["progress"]["step"] == 30  # capped by server max_training_steps
        assert c.get(f"/training/jobs/{job['id']}/metrics").json()["metrics"]
        assert c.post(f"/training/jobs/{job['id']}/stop").status_code == 400  # not running
        assert c.get("/training/jobs/doesnotexist").status_code == 404


def test_training_stop_and_resume(env):
    with TestClient(create_app(settings(env, max_training_steps=100000)), client=LOCAL) as c:
        job = c.post("/training/jobs", json={"name": "stopper", "config": "smoke.yaml",
                                             "overrides": {"training.max_steps": 100000, "training.save_interval": 10,
                                                           "data.path": "data/synthetic"}}).json()
        deadline = time.time() + 120
        while time.time() < deadline:
            j = c.get(f"/training/jobs/{job['id']}").json()
            if j["progress"].get("step", 0) >= 20:
                break
            time.sleep(0.5)
        assert c.post(f"/training/jobs/{job['id']}/stop").status_code == 200
        while time.time() < deadline + 60:
            j = c.get(f"/training/jobs/{job['id']}").json()
            if j["status"] not in ("running", "stopping"):
                break
            time.sleep(0.5)
        assert j["status"] == "stopped", j
        stopped_at = j["progress"]["step"]
        assert c.post(f"/training/jobs/{job['id']}/resume").status_code == 200
        time.sleep(4)
        c.post(f"/training/jobs/{job['id']}/stop")
        while time.time() < deadline + 120:
            j = c.get(f"/training/jobs/{job['id']}").json()
            if j["status"] not in ("running", "stopping"):
                break
            time.sleep(0.5)
        assert j["progress"]["step"] > stopped_at


def _zip(files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for k, v in files.items():
            z.writestr(k, v)
    return buf.getvalue()


def test_model_import(env, client):
    src = env / "models" / "tiny"
    good = {f: (src / f).read_bytes() for f in ("config.json", "model.safetensors", "tokenizer.json")}
    r = client.post("/models/import", data={"name": "imported"}, files={"file": ("m.zip", _zip(good))})
    assert r.status_code == 200, r.text
    assert r.json()["id"] == "models/imported"
    assert client.post("/v1/completions", json={"model": "models/imported", "prompt": "a", "max_tokens": 3}).status_code == 200
    evil = {**good, "optimizer.pt": b"\x80\x04pickle"}
    assert client.post("/models/import", data={"name": "evil"}, files={"file": ("m.zip", _zip(evil))}).status_code == 400
    trav = {**good, "../../config.json": b"{}"}
    assert client.post("/models/import", data={"name": "trav"}, files={"file": ("m.zip", _zip(trav))}).status_code == 400
    assert client.post("/models/import", data={"name": "../x"}, files={"file": ("m.zip", _zip(good))}).status_code == 400
    assert client.post("/models/import", data={"name": "notzip"}, files={"file": ("m.zip", b"hello")}).status_code == 400


def test_settings_metrics_benchmarks(client):
    assert client.put("/settings", json={"max_new_tokens": 64}).json()["max_new_tokens"] == 64
    assert client.put("/settings", json={"unknown": 1}).status_code == 422
    client.post("/v1/completions", json={"model": "models/tiny", "prompt": "a", "max_tokens": 2})
    text = client.get("/metrics").text
    assert "featherlm_generation_requests_total" in text and "process_resident_memory_bytes" in text
    assert "speed" in client.get("/benchmarks").json()["results"]
