"""Compatibility wrapper for the canonical GBM benchmark implementation."""
from __future__ import annotations

from src.tools.gbm_benchmark import main, run_gbm_benchmark, run_gbm_benchmark_from_postgres

__all__ = ["main", "run_gbm_benchmark", "run_gbm_benchmark_from_postgres"]


if __name__ == "__main__":
    main()
