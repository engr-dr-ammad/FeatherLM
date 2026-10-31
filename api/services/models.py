"""Model discovery, lazy loading (bounded LRU) and safe checkpoint import."""
from __future__ import annotations

import io
import json
import re
import shutil
import threading
import zipfile
from collections import OrderedDict
from pathlib import Path

from featherlm.inference.engine import InferenceEngine
from featherlm.model.config import FeatherConfig
from featherlm.tokenizer.tokenizer import ByteBPETokenizer
from featherlm.training.checkpoint import safe_child

ALLOWED_IMPORT_FILES = {"config.json", "model.safetensors", "tokenizer.json", "metadata.json"}
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
MAX_IMPORT_BYTES = 2 * 1024**3


class ModelNotFound(KeyError):
    pass


class ModelRegistry:
    def __init__(self, settings) -> None:
        self.s = settings
        self._engines: OrderedDict[str, InferenceEngine] = OrderedDict()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def _roots(self) -> dict[str, Path]:
        return {"models": self.s.models_dir, "runs": self.s.runs_dir}

    def discover(self) -> dict[str, Path]:
        found: dict[str, Path] = {}
        for prefix, root in self._roots().items():
            if not root.exists():
                continue
            for cfg in sorted(root.glob("**/config.json")):
                d = cfg.parent
                if not (d / "model.safetensors").exists() or any(p.name.startswith(".") for p in d.parents):
                    continue
                if d.name.startswith(".") or "bench" in d.relative_to(root).parts[:1]:
                    continue
                rel = d.relative_to(root).as_posix()
                found[f"{prefix}/{rel}"] = d
        return found

    def resolve(self, model_id: str) -> Path:
        prefix, _, rel = model_id.partition("/")
        roots = self._roots()
        if prefix not in roots or not rel:
            raise ModelNotFound(model_id)
        try:
            p = safe_child(roots[prefix], rel)
        except ValueError as e:
            raise ModelNotFound(model_id) from e
        if not (p / "config.json").is_file() or not (p / "model.safetensors").is_file():
            raise ModelNotFound(model_id)
        return p

    def describe(self, model_id: str, path: Path | None = None) -> dict:
        path = path or self.resolve(model_id)
        cfg = json.loads((path / "config.json").read_text())
        meta = json.loads((path / "metadata.json").read_text()) if (path / "metadata.json").exists() else {}
        size = sum(f.stat().st_size for f in path.glob("*.safetensors"))
        params = meta.get("parameters", 0)
        # Runtime estimate: FP32 weights after dequantisation + ~350 MB runtime + small cache.
        mem_mb = params * 4 / 2**20 + 350
        siblings = sorted(p.name for p in path.parent.iterdir()
                          if p.is_dir() and (p / "config.json").exists()) if path.parent.exists() else []
        return {
            "id": model_id,
            "name": meta.get("name", cfg.get("name")),
            "architecture": cfg.get("backend"),
            "parameters": params,
            "file_size_bytes": size,
            "quantization": meta.get("quantization", "none"),
            "context_length": cfg.get("max_seq_len"),
            "vocab_size": cfg.get("vocab_size"),
            "d_model": cfg.get("d_model"),
            "n_layers": cfg.get("n_layers"),
            "chat_template": cfg.get("chat_template"),
            "capabilities": {"completions": True, "chat": cfg.get("chat_template") == "featherlm-v1"},
            "memory_estimate_mb": round(mem_mb),
            "created": meta.get("created"),
            "training": meta.get("training"),
            "checkpoints": siblings,
            "loaded": model_id in self._engines,
        }

    def list(self) -> list[dict]:
        out = []
        for mid, p in self.discover().items():
            try:
                out.append(self.describe(mid, p))
            except (OSError, json.JSONDecodeError):
                continue
        return out

    def default_id(self) -> str | None:
        if self.s.default_model:
            return self.s.default_model
        ids = list(self.discover())
        return ids[0] if ids else None

    def get(self, model_id: str | None) -> tuple[str, InferenceEngine]:
        model_id = model_id or self.default_id()
        if not model_id:
            raise ModelNotFound("no models available")
        with self._lock:
            if model_id in self._engines:
                self._engines.move_to_end(model_id)
                return model_id, self._engines[model_id]
            path = self.resolve(model_id)
            engine = InferenceEngine(path, threads=self.s.threads)
            self._engines[model_id] = engine
            while len(self._engines) > max(1, self.s.max_loaded_models):
                self._engines.popitem(last=False)
            return model_id, engine

    def unload(self, model_id: str) -> bool:
        with self._lock:
            return self._engines.pop(model_id, None) is not None

    @property
    def loaded(self) -> list[str]:
        return list(self._engines)

    # ------------------------------------------------------------------
    def import_zip(self, name: str, data: bytes) -> str:
        """Import a checkpoint zip. Only safetensors + JSON files are accepted (no pickle)."""
        if not NAME_RE.match(name):
            raise ValueError("name must be 1-64 chars of letters, digits, '.', '_' or '-'")
        if len(data) > MAX_IMPORT_BYTES:
            raise ValueError("archive too large")
        dest = safe_child(self.s.models_dir, name)
        if dest.exists():
            raise ValueError(f"model '{name}' already exists")
        try:
            zf = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile as e:
            raise ValueError("not a zip archive") from e
        members = {}
        total = 0
        for info in zf.infolist():
            if info.is_dir():
                continue
            base = Path(info.filename).name
            if base not in ALLOWED_IMPORT_FILES:
                raise ValueError(f"unexpected file in archive: {info.filename!r} (allowed: {sorted(ALLOWED_IMPORT_FILES)})")
            if base in members:
                raise ValueError(f"duplicate file {base}")
            total += info.file_size
            if total > MAX_IMPORT_BYTES:
                raise ValueError("archive expands beyond the size limit")
            members[base] = info
        missing = {"config.json", "model.safetensors", "tokenizer.json"} - set(members)
        if missing:
            raise ValueError(f"missing files: {sorted(missing)}")
        tmp = dest.with_name(f".{name}.importing")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        try:
            for base, info in members.items():
                with zf.open(info) as src, open(tmp / base, "wb") as out:
                    shutil.copyfileobj(src, out)
            # Validate: config parses, tokenizer parses, and the weights load into the model.
            FeatherConfig.load(tmp / "config.json")
            ByteBPETokenizer.load(tmp / "tokenizer.json")
            if not (tmp / "metadata.json").exists():
                (tmp / "metadata.json").write_text(json.dumps({"name": name, "quantization": "none", "imported": True}))
            from featherlm.training.checkpoint import load_model

            m, _, _ = load_model(tmp)
            meta = json.loads((tmp / "metadata.json").read_text())
            meta.setdefault("parameters", m.num_params())
            (tmp / "metadata.json").write_text(json.dumps(meta, indent=2))
            del m
            tmp.rename(dest)
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        return f"models/{name}"
