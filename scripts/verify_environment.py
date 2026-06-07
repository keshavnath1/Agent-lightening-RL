#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import json
import sys
from pathlib import Path

CPU_PACKAGES = [
    ("streamlit", "streamlit"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("sklearn", "scikit-learn"),
    ("pyarrow", "pyarrow"),
    ("requests", "requests"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("altair", "altair"),
]

GPU_PACKAGES = CPU_PACKAGES + [
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("peft", "peft"),
    ("trl", "trl"),
    ("datasets", "datasets"),
    ("accelerate", "accelerate"),
    ("bitsandbytes", "bitsandbytes"),
]


def check(import_name: str, dist_name: str) -> dict[str, str | bool]:
    try:
        importlib.import_module(import_name)
        ok = True
        error = ""
    except Exception as exc:
        ok = False
        error = f"{type(exc).__name__}: {exc}"
    try:
        version = md.version(dist_name)
    except Exception:
        version = "unknown"
    return {"package": dist_name, "import": import_name, "ok": ok, "version": version, "error": error}


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify CPU/GPU project environment.")
    parser.add_argument("--target", choices=["cpu", "gpu"], default="cpu")
    args = parser.parse_args()

    rows = [check(*pair) for pair in (GPU_PACKAGES if args.target == "gpu" else CPU_PACKAGES)]
    print(json.dumps({"target": args.target, "python": sys.version.split()[0], "checks": rows}, indent=2))
    missing = [r for r in rows if not r["ok"]]
    if missing:
        print("Missing or failing imports: " + ", ".join(str(r["package"]) for r in missing), file=sys.stderr)
        return 1

    repo = Path(__file__).resolve().parents[1]
    dashboard = repo / "scripts" / "demo_dashboard.py"
    if dashboard.exists():
        compile(dashboard.read_text(), str(dashboard), "exec")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
