from __future__ import annotations

"""Compatibility wrapper for the repository-level vLLM-only RULER scorer.

The active implementation lives in ``src.rewards.apply_ruler_scores`` and
requires an OpenAI-compatible vLLM judge endpoint. This wrapper intentionally
provides no local scoring fallback.
"""

from src.rewards.apply_ruler_scores import *  # noqa: F401,F403
from src.rewards.apply_ruler_scores import main

if __name__ == "__main__":
    main()
