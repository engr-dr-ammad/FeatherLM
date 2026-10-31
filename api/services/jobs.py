"""Training jobs as managed background processes, with state in SQLite.

Each job runs `train.py` (or `finetune.py`) in its own process session, so API
requests never block on training. Stop sends SIGTERM; the trainer finishes the
current step, saves a checkpoint and exits. Resume restarts the same config and the
trainer picks up `<output_dir>/latest`.
"""
from __future__ import annotations

import json
import os
import re
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import yaml

from featherlm.training.checkpoint import safe_child

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,47}$")
CONFIG_RE = re.compile(r"^[A-Za-z0-9_-]+\.ya?ml$")
OVERRIDABLE = {
    "training.max_steps": int, "training.batch_size": int, "training.grad_accum": int, "training.lr": float,
    "training.seed": int, "training.eval_interval": int, "training.save_interval": int,
    "training.log_interval": int, "training.max_minutes": float, "data.path": str,
}


class JobError(ValueError):
    pass


class JobManager:
    def __init__(self, settings) -> None:
        self.s = settings
        self.db_path = settings.db_path
        self._procs: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path, timeout=10)
        c.row_factory = sqlite3.Row
        return c

    def _init_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, name TEXT, base_config TEXT, config_path TEXT, output_dir TEXT,
                status TEXT, pid INTEGER, created REAL, updated REAL, error TEXT, log_path TEXT, kind TEXT)""")
            c.execute("CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT)")
            # Processes do not survive a server restart's bookkeeping; mark them as interrupted.
            for row in c.execute("SELECT id, pid FROM jobs WHERE status IN ('running','stopping')").fetchall():
                if not _pid_alive(row["pid"]):
                    c.execute("UPDATE jobs SET status='interrupted', updated=? WHERE id=?", (time.time(), row["id"]))

    # ---------------- key/value settings ----------------
    def kv_get(self, key: str, default=None):
        with self._conn() as c:
            row = c.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def kv_set(self, key: str, value) -> None:
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO kv(key, value) VALUES (?, ?)", (key, json.dumps(value)))

    # ---------------- jobs ----------------
    def available_configs(self) -> list[str]:
        return sorted(p.name for p in self.s.configs_dir.glob("*.yaml") if CONFIG_RE.match(p.name))

    def _running_count(self) -> int:
        return sum(1 for j in self.list() if j["status"] in ("running", "stopping"))

    def create(self, name: str, base_config: str, overrides: dict | None = None) -> dict:
        if not NAME_RE.match(name):
            raise JobError("job name: 1-48 chars of letters, digits, '_' or '-'")
        if not CONFIG_RE.match(base_config) or not (self.s.configs_dir / base_config).is_file():
            raise JobError(f"unknown config {base_config!r}; available: {self.available_configs()}")
        raw = yaml.safe_load((self.s.configs_dir / base_config).read_text())
        for k, v in (overrides or {}).items():
            if k not in OVERRIDABLE:
                raise JobError(f"override {k!r} not allowed; allowed: {sorted(OVERRIDABLE)}")
            try:
                v = OVERRIDABLE[k](v)
            except (TypeError, ValueError) as e:
                raise JobError(f"bad value for {k}") from e
            sect, key = k.split(".")
            raw.setdefault(sect, {})[key] = v
        # Resource limits enforced server-side.
        t = raw.setdefault("training", {})
        t["max_steps"] = min(int(t.get("max_steps", 1000)), self.s.max_training_steps)
        if int(t["max_steps"]) < 1:
            raise JobError("max_steps must be >= 1")
        budget = float(t.get("memory_budget_mb") or self.s.training_memory_budget_mb)
        t["memory_budget_mb"] = min(budget, self.s.training_memory_budget_mb)
        if self.s.threads:
            t["threads"] = self.s.threads
        d = raw.setdefault("data", {})
        data_path = Path(d.get("path", "data/shakespeare"))
        rel = data_path.relative_to("data") if data_path.parts and data_path.parts[0] == "data" else data_path
        try:
            d["path"] = str(safe_child(self.s.data_dir, str(rel)))
        except ValueError as e:
            raise JobError("data.path must be inside the data directory") from e
        if raw.get("finetune", {}).get("init_from"):
            raise JobError("fine-tuning jobs are started from the CLI (finetune.py) in this version")
        with self._lock:
            if self._running_count() >= self.s.max_training_jobs:
                raise JobError(f"at most {self.s.max_training_jobs} training job(s) may run at once")
            job_id = uuid.uuid4().hex[:12]
            out_dir = safe_child(self.s.runs_dir, name)
            out_dir.mkdir(parents=True, exist_ok=True)
            raw["output_dir"] = str(out_dir)
            cfg_path = out_dir / "job_config.yaml"
            cfg_path.write_text(yaml.safe_dump(raw))
            now = time.time()
            with self._conn() as c:
                c.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                          (job_id, name, base_config, str(cfg_path), str(out_dir), "pending", None, now, now, None,
                           str(out_dir / "train.log"), "pretrain"))
            self._spawn(job_id, resume=False)
        return self.get(job_id)

    def _spawn(self, job_id: str, resume: bool) -> None:
        row = self._row(job_id)
        cmd = [sys.executable, str(self.s.root / "train.py"), "--config", row["config_path"]]
        if not resume:
            cmd.append("--no-resume")
        log = open(row["log_path"], "a")
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen(cmd, cwd=str(self.s.root), stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, env=env)
        log.close()
        self._procs[job_id] = proc
        with self._conn() as c:
            c.execute("UPDATE jobs SET status='running', pid=?, updated=?, error=NULL WHERE id=?",
                      (proc.pid, time.time(), job_id))

    def _row(self, job_id: str) -> sqlite3.Row:
        with self._conn() as c:
            row = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return row

    def _refresh(self, row: sqlite3.Row) -> dict:
        job = dict(row)
        status_file = Path(job["output_dir"]) / "status.json"
        st = {}
        if status_file.exists():
            try:
                st = json.loads(status_file.read_text())
            except json.JSONDecodeError:
                st = {}
        if job["status"] in ("running", "stopping"):
            proc = self._procs.get(job["id"])
            code = proc.poll() if proc else (None if _pid_alive(job["pid"]) else -1)
            if code is not None:
                new = st.get("state", "failed") if st.get("state") not in (None, "running") else "failed"
                if code != 0 and new not in ("stopped",):
                    new = "failed"
                    job["error"] = st.get("error") or _tail(job["log_path"])
                job["status"] = new
                with self._conn() as c:
                    c.execute("UPDATE jobs SET status=?, updated=?, error=? WHERE id=?",
                              (new, time.time(), job.get("error"), job["id"]))
                self._procs.pop(job["id"], None)
        job["progress"] = st
        job.pop("config_path", None)
        return job

    def get(self, job_id: str) -> dict:
        return self._refresh(self._row(job_id))

    def list(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM jobs ORDER BY created DESC").fetchall()
        return [self._refresh(r) for r in rows]

    def stop(self, job_id: str) -> dict:
        job = self.get(job_id)
        if job["status"] != "running":
            raise JobError(f"job is {job['status']}, not running")
        try:
            os.kill(job["pid"], signal.SIGTERM)
        except ProcessLookupError:
            pass
        with self._conn() as c:
            c.execute("UPDATE jobs SET status='stopping', updated=? WHERE id=?", (time.time(), job_id))
        return self.get(job_id)

    def resume(self, job_id: str) -> dict:
        job = self.get(job_id)
        if job["status"] in ("running", "stopping"):
            raise JobError("job is still running")
        if not (Path(job["output_dir"]) / "latest" / "trainer_state.json").exists():
            raise JobError("no checkpoint to resume from")
        with self._lock:
            if self._running_count() >= self.s.max_training_jobs:
                raise JobError(f"at most {self.s.max_training_jobs} training job(s) may run at once")
            self._spawn(job_id, resume=True)
        return self.get(job_id)

    def metrics(self, job_id: str, limit: int = 500) -> list[dict]:
        job = self.get(job_id)
        p = Path(job["output_dir"]) / "metrics.jsonl"
        if not p.exists():
            return []
        lines = p.read_text().splitlines()[-limit:]
        return [json.loads(l) for l in lines if l.strip()]

    def shutdown(self) -> None:
        """Stop child training processes gracefully on server shutdown."""
        for job_id, proc in list(self._procs.items()):
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
        deadline = time.time() + 20
        for proc in self._procs.values():
            try:
                proc.wait(timeout=max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                proc.kill()


def _pid_alive(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    # Reap zombies of our own children
    try:
        done, _ = os.waitpid(pid, os.WNOHANG)
        if done == pid:
            return False
    except ChildProcessError:
        pass
    return True


def _tail(path: str, n: int = 2000) -> str:
    try:
        return Path(path).read_text()[-n:]
    except OSError:
        return ""
