"""In-process request metrics exported in Prometheus text format."""
from __future__ import annotations

import threading

import psutil


class Metrics:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.requests = 0
        self.tokens = 0
        self.latency_sum = 0.0
        self.ttft_sum = 0.0
        self.errors = 0
        self.http_requests: dict[tuple[str, int], int] = {}

    def record(self, stats, latency_s: float) -> None:
        with self.lock:
            self.requests += 1
            self.tokens += stats.completion_tokens
            self.latency_sum += latency_s
            self.ttft_sum += stats.time_to_first_token_s

    def http(self, path: str, code: int) -> None:
        key = (path, code)
        with self.lock:
            self.http_requests[key] = self.http_requests.get(key, 0) + 1

    def prometheus(self, app) -> str:
        p = psutil.Process()
        lines = [
            "# TYPE featherlm_generation_requests_total counter",
            f"featherlm_generation_requests_total {self.requests}",
            "# TYPE featherlm_generated_tokens_total counter",
            f"featherlm_generated_tokens_total {self.tokens}",
            "# TYPE featherlm_generation_latency_seconds_sum counter",
            f"featherlm_generation_latency_seconds_sum {self.latency_sum:.6f}",
            "# TYPE featherlm_time_to_first_token_seconds_sum counter",
            f"featherlm_time_to_first_token_seconds_sum {self.ttft_sum:.6f}",
            "# TYPE featherlm_loaded_models gauge",
            f"featherlm_loaded_models {len(app.state.registry.loaded)}",
            "# TYPE process_resident_memory_bytes gauge",
            f"process_resident_memory_bytes {p.memory_info().rss}",
            "# TYPE featherlm_http_requests_total counter",
        ]
        for (path, code), n in sorted(self.http_requests.items()):
            lines.append(f'featherlm_http_requests_total{{path="{path}",code="{code}"}} {n}')
        return "\n".join(lines) + "\n"
