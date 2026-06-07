from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class RulerJudgeConfig:
    """Environment-driven configuration for a local vLLM-hosted RULER judge.

    The judge endpoint is OpenAI-compatible, so it can be served by vLLM on a
    separate port from the policy model. Defaults match the intended local GPU
    layout: policy on :8000 and judge on :8001.
    """

    judge_model: str = os.getenv("RULER_JUDGE_MODEL", "local-ruler-judge")
    judge_base_url: str = os.getenv("RULER_JUDGE_BASE_URL", "http://localhost:8001/v1")
    judge_api_key: str = os.getenv("RULER_JUDGE_API_KEY", "dummy")
    timeout_seconds: int = int(os.getenv("RULER_JUDGE_TIMEOUT", "120"))
    max_tokens: int = int(os.getenv("RULER_JUDGE_MAX_TOKENS", "2048"))

