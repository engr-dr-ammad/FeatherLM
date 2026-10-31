"""Server settings, read from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _bool(v: str | None, default: bool) -> bool:
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    root: Path = field(default_factory=lambda: Path(os.getenv("FEATHERLM_ROOT", Path(__file__).resolve().parents[2])))
    models_dir: Path = Path("models")
    runs_dir: Path = Path("runs")
    data_dir: Path = Path("data")
    configs_dir: Path = Path("configs")
    results_dir: Path = Path("results")
    db_path: Path = Path("featherlm.db")
    api_key: str = ""  # required for inference endpoints when set
    admin_key: str = ""  # required for training/admin endpoints when set
    enable_training_api: bool = True
    max_request_bytes: int = 1_000_000
    max_new_tokens: int = 512
    max_prompt_chars: int = 32_000
    max_loaded_models: int = 2
    threads: int = 0
    default_model: str = ""
    max_training_jobs: int = 1
    max_training_steps: int = 20_000
    training_memory_budget_mb: float = 3000
    busy_timeout_s: float = 30.0
    cors_origins: list[str] = field(default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"])
    log_level: str = "INFO"
    serve_frontend: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls()
        env = os.getenv
        r = s.root

        def path(name: str, default: Path) -> Path:
            v = env(name)
            p = Path(v) if v else default
            return p if p.is_absolute() else (r / p)

        s.models_dir = path("FEATHERLM_MODELS_DIR", Path("models"))
        s.runs_dir = path("FEATHERLM_RUNS_DIR", Path("runs"))
        s.data_dir = path("FEATHERLM_DATA_DIR", Path("data"))
        s.configs_dir = path("FEATHERLM_CONFIGS_DIR", Path("configs"))
        s.results_dir = path("FEATHERLM_RESULTS_DIR", Path("results"))
        s.db_path = path("FEATHERLM_DB", Path("featherlm.db"))
        s.api_key = env("FEATHERLM_API_KEY", "")
        s.admin_key = env("FEATHERLM_ADMIN_KEY", "")
        s.enable_training_api = _bool(env("FEATHERLM_ENABLE_TRAINING_API"), True)
        s.max_request_bytes = int(env("FEATHERLM_MAX_REQUEST_BYTES", s.max_request_bytes))
        s.max_new_tokens = int(env("FEATHERLM_MAX_NEW_TOKENS", s.max_new_tokens))
        s.max_loaded_models = int(env("FEATHERLM_MAX_LOADED_MODELS", s.max_loaded_models))
        s.threads = int(env("FEATHERLM_THREADS", s.threads))
        s.default_model = env("FEATHERLM_DEFAULT_MODEL", "")
        s.max_training_jobs = int(env("FEATHERLM_MAX_TRAINING_JOBS", s.max_training_jobs))
        s.max_training_steps = int(env("FEATHERLM_MAX_TRAINING_STEPS", s.max_training_steps))
        s.training_memory_budget_mb = float(env("FEATHERLM_TRAINING_MEMORY_BUDGET_MB", s.training_memory_budget_mb))
        s.busy_timeout_s = float(env("FEATHERLM_BUSY_TIMEOUT_S", s.busy_timeout_s))
        if env("FEATHERLM_CORS_ORIGINS"):
            s.cors_origins = [o.strip() for o in env("FEATHERLM_CORS_ORIGINS", "").split(",") if o.strip()]
        s.log_level = env("FEATHERLM_LOG_LEVEL", s.log_level).upper()
        s.serve_frontend = _bool(env("FEATHERLM_SERVE_FRONTEND"), True)
        return s
