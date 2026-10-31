"""Request/response schemas (OpenAI-compatible where applicable)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _GenParams(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str | None = None
    max_tokens: int = Field(128, ge=1, le=4096)
    temperature: float = Field(0.8, ge=0.0, le=5.0)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    top_k: int = Field(0, ge=0, le=10000)
    repetition_penalty: float = Field(1.0, gt=0.0, le=5.0)
    stop: str | list[str] | None = None
    stream: bool = False
    seed: int | None = Field(None, ge=0, le=2**31 - 1)
    n: int = Field(1, ge=1, le=1)

    @field_validator("stop")
    @classmethod
    def _stop(cls, v):
        if v is None:
            return []
        v = [v] if isinstance(v, str) else v
        if len(v) > 8 or any(len(s) > 64 for s in v):
            raise ValueError("at most 8 stop sequences of up to 64 characters")
        return v


class CompletionRequest(_GenParams):
    prompt: str | list[str] = Field(..., description="a single prompt (list of length 1 also accepted)")

    @field_validator("prompt")
    @classmethod
    def _prompt(cls, v):
        if isinstance(v, list):
            if len(v) != 1:
                raise ValueError("exactly one prompt is supported")
            v = v[0]
        if len(v) > 32000:
            raise ValueError("prompt too long")
        return v


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(..., max_length=32000)


class ChatCompletionRequest(_GenParams):
    messages: list[ChatMessage] = Field(..., min_length=1, max_length=64)


class TrainingJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(..., min_length=1, max_length=48)
    config: str = Field("nano_cpu.yaml", max_length=64)
    overrides: dict[str, float | int | str] = Field(default_factory=dict)


class SettingsUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    threads: int | None = Field(None, ge=0, le=256)
    default_model: str | None = Field(None, max_length=256)
    max_new_tokens: int | None = Field(None, ge=1, le=4096)
    training_memory_budget_mb: float | None = Field(None, ge=256, le=1_000_000)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] | None = None
