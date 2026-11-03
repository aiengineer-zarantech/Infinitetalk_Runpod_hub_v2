"""Pod job server: the RunPod serverless job API on ONE GPU pod, so the backend only swaps the base URL.

Why not `handler.py --rp_serve_api`: that is the RunPod SDK's development simulator. Its /run only stores the
job, POST /status runs the whole job inside that one HTTP request (6+ min, longer than the RunPod proxy
timeout) and forgets it afterwards, and it has no auth. This server runs jobs in a background thread, one at
a time, keeps results for POD_JOB_TTL_SEC and checks a Bearer token. (POD_API=runpod_sim in start_pod.sh still
starts the simulator for debugging.)

Routes (same shapes as api.runpod.ai/v2/<endpoint>):
  POST /run              {"input": {...}} -> {"id", "status": "IN_QUEUE"}            (429 when the queue is full)
  GET|POST /status/{id}  -> {"id", "status", "output"?, "error"?, "delayTime", "executionTime", "workerId"}
  POST /cancel/{id}      queued -> CANCELLED; a running job cannot be stopped (status returned unchanged)
  GET /health            -> {"workers": {...}, "jobs": {...}, "mode": "pod"}
  GET /ping              -> {"ok": true}   (no auth, no data: liveness only)
Every route except /ping needs "Authorization: Bearer $POD_API_TOKEN".
An unknown job id returns 200 with status FAILED (as RunPod does), so a 404 always means "pod not reachable".
"""
import collections
import hmac
import logging
import os
import sys
import threading
import time
import uuid

logger = logging.getLogger("pod_api")

TERMINAL = ("COMPLETED", "FAILED", "CANCELLED")


def _env_int(name, default):
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


class JobStore:
    """FIFO queue + one worker thread. Thread-safe. `handler(job_dict) -> result` is the worker handler."""

    def __init__(self, handler, max_queue=10, ttl_sec=3600, max_finished=50, clock=time.time, model_info=None):
        self.handler = handler
        self.model_info = model_info      # MODEL_LOCK: () -> {"resident": bool, ...} for /health
        self.max_queue = max_queue
        self.ttl_sec = ttl_sec
        self.max_finished = max_finished
        self.clock = clock
        self.jobs = {}
        self.queue = collections.deque()
        self.lock = threading.Lock()
        self.wake = threading.Condition(self.lock)
        self.running_id = None
        self.counts = {"completed": 0, "failed": 0}
        self._stop = False
        self.thread = None

    # ---------------------------------------------------------------- API side
    def submit(self, job_input):
        with self.lock:
            self._prune()
            if len(self.queue) >= self.max_queue:
                return None
            job_id = f"pod-{uuid.uuid4()}"
            self.jobs[job_id] = {"id": job_id, "input": job_input, "status": "IN_QUEUE",
                                 "submitted": self.clock(), "started": None, "finished": None}
            self.queue.append(job_id)
            self.wake.notify()
            return job_id

    def status(self, job_id):
        with self.lock:
            rec = self.jobs.get(job_id)
            if rec is None:
                return {"id": job_id, "status": "FAILED",
                        "error": "job not found on the pod (finished too long ago, or the pod restarted)"}
            return self._public(rec)

    def cancel(self, job_id):
        with self.lock:
            rec = self.jobs.get(job_id)
            if rec is not None and rec["status"] == "IN_QUEUE":
                self.queue.remove(job_id)
                rec.update(status="CANCELLED", finished=self.clock())
                rec.pop("input", None)
        return self.status(job_id)

    def health(self):
        with self.lock:
            busy = self.running_id is not None
            return {
                "mode": "pod",
                "workers": {"idle": 0 if busy else 1, "ready": 0 if busy else 1, "running": 1 if busy else 0,
                            "initializing": 0, "throttled": 0, "unhealthy": 0},
                "jobs": {"inQueue": len(self.queue), "inProgress": 1 if busy else 0, **self.counts},
                **({"model": self._model()} if self.model_info else {}),
            }

    def _model(self):
        try:
            return self.model_info()
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    # ------------------------------------------------------------- worker side
    def start(self):
        self.thread = threading.Thread(target=self._loop, name="pod-job-worker", daemon=True)
        self.thread.start()

    def stop(self):
        with self.lock:
            self._stop = True
            self.wake.notify_all()

    def run_next(self, block=True):
        """Run one queued job. Returns its id, or None (nothing queued / stopping)."""
        with self.lock:
            while not self.queue and not self._stop and block:
                self.wake.wait(timeout=5)
            if self._stop or not self.queue:
                return None
            job_id = self.queue.popleft()
            rec = self.jobs[job_id]
            rec.update(status="IN_PROGRESS", started=self.clock())
            self.running_id = job_id
            job = {"id": job_id, "input": rec["input"]}
        logger.info(f"[POD] job {job_id} started")
        try:
            result = self.handler(job)
            if isinstance(result, dict) and result.get("error"):
                status, error, output = "FAILED", result.get("error"), result
            else:
                status, error, output = "COMPLETED", None, result
            if isinstance(result, dict) and result.get("refresh_worker"):
                logger.warning("[POD] handler asked for refresh_worker; a pod cannot replace itself, "
                               "restart the pod if the next jobs fail too")
        except Exception as e:   # the guarded handler normally returns errors as dicts
            logger.exception(f"[POD] job {job_id} raised")
            status, error, output = "FAILED", f"{type(e).__name__}: {e}", None
        with self.lock:
            rec.update(status=status, finished=self.clock(), output=output, error=error)
            rec.pop("input", None)            # the base64 audio/image is not kept after the run
            self.running_id = None
            self.counts["completed" if status == "COMPLETED" else "failed"] += 1
        logger.info(f"[POD] job {job_id} {status} in {rec['finished'] - rec['started']:.1f}s")
        return job_id

    def _loop(self):
        while not self._stop:
            self.run_next(block=True)

    # ---------------------------------------------------------------- helpers
    def _public(self, rec):
        out = {"id": rec["id"], "status": rec["status"], "workerId": "pod"}
        now = self.clock()
        start = rec["started"]
        out["delayTime"] = int(((start or now) - rec["submitted"]) * 1000)
        if start is not None:
            out["executionTime"] = int(((rec["finished"] or now) - start) * 1000)
        if rec.get("output") is not None:
            out["output"] = rec["output"]
        if rec.get("error"):
            out["error"] = rec["error"]
        return out

    def _prune(self):
        now = self.clock()
        done = [r for r in self.jobs.values() if r["status"] in TERMINAL and r["finished"] is not None]
        for r in done:
            if now - r["finished"] > self.ttl_sec:
                del self.jobs[r["id"]]
        done = sorted((r for r in self.jobs.values() if r["status"] in TERMINAL), key=lambda r: r["finished"])
        for r in done[:max(0, len(done) - self.max_finished)]:
            del self.jobs[r["id"]]


def create_app(store, token):
    from fastapi import Depends, FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse

    app = FastAPI(title="InfiniteTalk pod job API", docs_url=None, redoc_url=None, openapi_url=None)

    def auth(request: Request):
        got = request.headers.get("authorization", "")
        if not token or not hmac.compare_digest(got.encode(), f"Bearer {token}".encode()):
            raise HTTPException(status_code=401, detail="unauthorized")

    @app.get("/ping")
    def ping():
        return {"ok": True}

    @app.get("/health", dependencies=[Depends(auth)])
    def health():
        return store.health()

    @app.post("/run", dependencies=[Depends(auth)])
    async def run(request: Request):
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(status_code=400, content={"error": "body must be JSON {\"input\": {...}}"})
        if not isinstance(body, dict) or not isinstance(body.get("input"), dict):
            return JSONResponse(status_code=400, content={"error": "body must be JSON {\"input\": {...}}"})
        job_id = store.submit(body["input"])
        if job_id is None:
            return JSONResponse(status_code=429, content={"error": f"pod queue full ({store.max_queue} jobs)"})
        return {"id": job_id, "status": "IN_QUEUE"}

    @app.api_route("/status/{job_id}", methods=["GET", "POST"], dependencies=[Depends(auth)])
    def status(job_id: str):
        return store.status(job_id)

    @app.post("/cancel/{job_id}", dependencies=[Depends(auth)])
    def cancel(job_id: str):
        return store.cancel(job_id)

    return app


def load_worker_handler(worker_root="/"):
    """Import handler.py without starting the serverless loop: runpod.serverless.start is captured."""
    import runpod
    captured = {}
    runpod.serverless.start = lambda config: captured.update(config)
    if worker_root not in sys.path:
        sys.path.insert(0, worker_root)
    import handler  # noqa: F401  (module bottom calls runpod.serverless.start with the guarded handler)
    if "handler" not in captured:
        raise RuntimeError("handler.py did not call runpod.serverless.start")
    return captured["handler"]


def model_lock_info():
    """MODEL_LOCK state of this worker for /health (the UI shows "Model ready (reused)" vs "Loading model")."""
    from worker_guard import model_lock
    if not model_lock.enabled():
        return {"lock": False}
    st = model_lock.STATE
    return {"lock": True, "resident": st.signature is not None, "config": model_lock.config_hash(),
            "loads": st.loads, "reuses": st.reuses}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    token = os.environ.get("POD_API_TOKEN", "").strip()
    if len(token) < 16:
        sys.exit("[POD] POD_API_TOKEN must be set (16+ chars). Refusing to serve the GPU without auth.")
    host = os.environ.get("POD_API_HOST", "0.0.0.0")
    port = _env_int("POD_API_PORT", 8000)
    store = JobStore(load_worker_handler(os.environ.get("WORKER_ROOT") or "/"),
                     max_queue=_env_int("POD_MAX_QUEUE", 10), ttl_sec=_env_int("POD_JOB_TTL_SEC", 3600),
                     model_info=model_lock_info)
    store.start()
    import uvicorn
    logger.info(f"[POD] job API on {host}:{port} (auth on, queue {store.max_queue}, ttl {store.ttl_sec}s)")
    uvicorn.run(create_app(store, token), host=host, port=port, log_level="info", timeout_keep_alive=30)


if __name__ == "__main__":
    main()
