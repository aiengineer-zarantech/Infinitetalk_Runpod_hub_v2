"""deploy/pod: pod/pod_api.py job server (no GPU, fake handler)."""
import sys
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pod"))
import pod_api  # noqa: E402

TOKEN = "t" * 24
AUTH = {"Authorization": f"Bearer {TOKEN}"}


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def make(handler, **kw):
    clock = Clock()
    store = pod_api.JobStore(handler, clock=clock, **kw)
    return store, TestClient(pod_api.create_app(store, TOKEN)), clock


def test_auth_required_everywhere_but_ping():
    store, c, _ = make(lambda job: {"video": "x"})
    assert c.get("/ping").json() == {"ok": True}
    for method, path in (("get", "/health"), ("post", "/run"), ("get", "/status/x"), ("post", "/cancel/x")):
        assert getattr(c, method)(path).status_code == 401
        assert getattr(c, method)(path, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert c.get("/health", headers=AUTH).status_code == 200


def test_run_then_status_like_runpod():
    seen = []
    store, c, clock = make(lambda job: seen.append(job) or {"video": "b64", "timing": {"windows": []}})
    r = c.post("/run", headers=AUTH, json={"input": {"prompt": "hi"}})
    job_id = r.json()["id"]
    assert r.json()["status"] == "IN_QUEUE" and job_id.startswith("pod-")
    assert c.get(f"/status/{job_id}", headers=AUTH).json()["status"] == "IN_QUEUE"
    assert c.get("/health", headers=AUTH).json()["jobs"]["inQueue"] == 1
    clock.t += 2
    store.run_next(block=False)
    st = c.get(f"/status/{job_id}", headers=AUTH).json()
    assert st["status"] == "COMPLETED" and st["output"]["video"] == "b64" and st["workerId"] == "pod"
    assert st["delayTime"] == 2000 and "executionTime" in st
    assert c.post(f"/status/{job_id}", headers=AUTH).json()["status"] == "COMPLETED"   # polled again: still there
    assert seen == [{"id": job_id, "input": {"prompt": "hi"}}]
    assert "input" not in store.jobs[job_id]                                         # payload dropped after run


def test_handler_error_dict_and_exception_are_failed():
    def handler(job):
        if job["input"].get("boom"):
            raise RuntimeError("cuda gone")
        return {"error": "ComfyUI execution_error", "details": {"node": "122"}}
    store, c, _ = make(handler)
    a = c.post("/run", headers=AUTH, json={"input": {}}).json()["id"]
    b = c.post("/run", headers=AUTH, json={"input": {"boom": True}}).json()["id"]
    store.run_next(block=False)
    store.run_next(block=False)
    sa, sb = (c.get(f"/status/{i}", headers=AUTH).json() for i in (a, b))
    assert sa["status"] == "FAILED" and sa["error"] == "ComfyUI execution_error" and sa["output"]["details"]
    assert sb["status"] == "FAILED" and "RuntimeError: cuda gone" in sb["error"]
    assert c.get("/health", headers=AUTH).json()["jobs"]["failed"] == 2


def test_unknown_job_is_200_failed_not_404():
    _, c, _ = make(lambda job: {})
    r = c.get("/status/nope", headers=AUTH)
    assert r.status_code == 200 and r.json()["status"] == "FAILED" and "not found" in r.json()["error"]


def test_queue_limit_and_cancel():
    store, c, _ = make(lambda job: {"video": "x"}, max_queue=2)
    ids = [c.post("/run", headers=AUTH, json={"input": {}}).json()["id"] for _ in range(2)]
    assert c.post("/run", headers=AUTH, json={"input": {}}).status_code == 429
    assert c.post(f"/cancel/{ids[0]}", headers=AUTH).json()["status"] == "CANCELLED"
    assert store.run_next(block=False) == ids[1]
    assert c.post(f"/cancel/{ids[1]}", headers=AUTH).json()["status"] == "COMPLETED"   # already done: unchanged


def test_bad_body_is_400():
    _, c, _ = make(lambda job: {})
    assert c.post("/run", headers=AUTH, content=b"not json").status_code == 400
    assert c.post("/run", headers=AUTH, json={"prompt": "no input key"}).status_code == 400


def test_running_job_shows_busy_worker():
    gate, entered = threading.Event(), threading.Event()

    def handler(job):
        entered.set()
        gate.wait(5)
        return {"video": "x"}
    store, c, _ = make(handler)
    job_id = c.post("/run", headers=AUTH, json={"input": {}}).json()["id"]
    t = threading.Thread(target=store.run_next, kwargs={"block": False})
    t.start()
    assert entered.wait(5)
    h = c.get("/health", headers=AUTH).json()
    assert h["workers"]["running"] == 1 and h["workers"]["idle"] == 0 and h["mode"] == "pod"
    assert c.get(f"/status/{job_id}", headers=AUTH).json()["status"] == "IN_PROGRESS"
    gate.set()
    t.join(5)
    assert c.get(f"/status/{job_id}", headers=AUTH).json()["status"] == "COMPLETED"


def test_finished_jobs_expire_after_ttl():
    store, c, clock = make(lambda job: {"video": "x"}, ttl_sec=60)
    job_id = c.post("/run", headers=AUTH, json={"input": {}}).json()["id"]
    store.run_next(block=False)
    clock.t += 61
    c.post("/run", headers=AUTH, json={"input": {}})               # submit prunes
    assert "not found" in c.get(f"/status/{job_id}", headers=AUTH).json()["error"]


def test_main_refuses_without_token(monkeypatch):
    monkeypatch.delenv("POD_API_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="POD_API_TOKEN"):
        pod_api.main()


def test_load_worker_handler_captures_without_starting(monkeypatch, tmp_path):
    (tmp_path / "handler.py").write_text(
        "import runpod\n"
        "def handler(job):\n    return {'video': 'ok'}\n"
        "runpod.serverless.start({'handler': handler, 'concurrency_modifier': lambda c: 1})\n")
    import types
    fake = types.ModuleType("runpod")
    fake.serverless = types.SimpleNamespace(start=lambda cfg: (_ for _ in ()).throw(AssertionError("started")))
    monkeypatch.setitem(sys.modules, "runpod", fake)
    monkeypatch.delitem(sys.modules, "handler", raising=False)
    h = pod_api.load_worker_handler(str(tmp_path))
    assert h({"id": "1", "input": {}}) == {"video": "ok"}
    monkeypatch.delitem(sys.modules, "handler", raising=False)
