from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path("/workspace/self-improving-ml-agent")
TASKS = ROOT / "data/real_openml/tasks.jsonl"
DEFAULT_OUTPUT = ROOT / "reports/openml_registry_payload.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def infer_problem_type(df: pd.DataFrame, target: str) -> str:
    if target not in df.columns:
        return "unknown"
    y = df[target]
    unique_count = int(y.nunique(dropna=True))
    if pd.api.types.is_numeric_dtype(y) and unique_count > 20:
        return "regression"
    if unique_count <= 2:
        return "binary_classification"
    return "multiclass_classification"


def metrics_for(problem_type: str, primary: str | None = None) -> tuple[str, list[str]]:
    if primary:
        primary_metric = primary
    elif problem_type == "regression":
        primary_metric = "rmse"
    elif problem_type == "multiclass_classification":
        primary_metric = "accuracy"
    else:
        primary_metric = "roc_auc"
    if problem_type == "regression":
        secondary = ["mae", "r2"]
    elif problem_type == "multiclass_classification":
        secondary = ["f1_macro", "precision_macro", "recall_macro", "log_loss"]
    else:
        secondary = ["f1", "precision", "recall", "log_loss"]
    return primary_metric, secondary


def safe_schema(df: pd.DataFrame, target: str) -> dict[str, Any]:
    columns: list[dict[str, Any]] = []
    for ordinal, column in enumerate(df.columns):
        series = df[column]
        columns.append(
            {
                "ordinal": ordinal,
                "name": str(column),
                "dtype": str(series.dtype),
                "role": "target" if column == target else "feature",
                "missing_count": int(series.isna().sum()),
                "missing_rate": round(float(series.isna().mean()), 6),
                "nunique": int(series.nunique(dropna=True)) if len(series) else 0,
            }
        )
    return {
        "column_count": int(len(df.columns)),
        "columns": columns,
        "feature_columns": [str(column) for column in df.columns if column != target],
        "target_column": target,
        "raw_rows_exposed": False,
    }


def safe_summary(df: pd.DataFrame, target: str) -> dict[str, Any]:
    target_counts: dict[str, int] = {}
    if target in df.columns:
        counts = df[target].value_counts(dropna=False).head(25)
        target_counts = {str(_jsonable(key)): int(value) for key, value in counts.items()}
    dtype_counts = {str(key): int(value) for key, value in df.dtypes.astype(str).value_counts().items()}
    numeric_columns = [column for column in df.select_dtypes(include="number").columns if column != target][:40]
    numeric_aggregates: dict[str, dict[str, float | None]] = {}
    for column in numeric_columns:
        series = df[column]
        clean = series.dropna()
        numeric_aggregates[str(column)] = {
            "mean": None if clean.empty else round(float(series.mean()), 6),
            "std": None if clean.empty else round(float(series.std()), 6),
            "min": None if clean.empty else round(float(series.min()), 6),
            "max": None if clean.empty else round(float(series.max()), 6),
        }
    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "target_column": target,
        "class_balance": target_counts,
        "dtype_counts": dtype_counts,
        "missing_cells": int(df.isna().sum().sum()),
        "missing_rate_total": round(float(df.isna().sum().sum() / max(1, df.shape[0] * df.shape[1])), 6),
        "numeric_aggregates": numeric_aggregates,
        "raw_rows_exposed": False,
        "summary_type": "aggregate_only",
    }


def read_tasks(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_payload(task: dict[str, Any]) -> dict[str, Any]:
    task = dict(task)
    dataset_ref = Path(task["dataset_ref"])
    target = task["target_column"]
    df = pd.read_parquet(dataset_ref)
    problem_type = task.get("problem_type") or infer_problem_type(df, target)
    primary_metric, secondary_metrics = metrics_for(problem_type, task.get("metric"))
    dataset_name = task.get("openml_name") or task.get("dataset_name") or task["task_id"]
    problem_statement = task.get("problem_statement") or f"Predict {target} for the {dataset_name} dataset."
    prediction_goal = task.get("prediction_goal") or f"Predict {target} from the available non-target attributes."
    task.update(
        {
            "source": task.get("source") or task.get("data_source") or "OpenML",
            "source_task_id": str(task.get("openml_data_id") or task.get("source_task_id") or task["task_id"]),
            "dataset_name": dataset_name,
            "problem_statement": problem_statement,
            "prediction_goal": prediction_goal,
            "problem_type": problem_type,
            "primary_metric": primary_metric,
            "secondary_metrics": secondary_metrics,
            "artifact_ref": str(dataset_ref),
            "synthetic": False,
            "registered_payload_built_at": now(),
        }
    )
    schema = safe_schema(df, target)
    summary = safe_summary(df, target)
    manifest = {
        "artifact_ref": str(dataset_ref),
        "dataset_ref": str(dataset_ref),
        "parquet_bytes": int(dataset_ref.stat().st_size),
        "approved_for_execution_layer": True,
        "raw_rows_exposed_to_llm": False,
        "source_url": task.get("openml_url"),
        "materialization_boundary": "execution_layer_only",
    }
    return {
        "task": task,
        "schema_json": schema,
        "summary_json": summary,
        "artifact_manifest": manifest,
        "raw_rows_exposed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build aggregate-only OpenML registry payloads for PostgreSQL/Prisma seeding.")
    parser.add_argument("--task-id", default="openml_31_german_credit", help="Task ID to include, or 'all'.")
    parser.add_argument("--tasks", type=Path, default=TASKS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    tasks = read_tasks(args.tasks)
    selected = tasks if args.task_id == "all" else [task for task in tasks if task["task_id"] == args.task_id]
    if not selected:
        raise SystemExit(f"No OpenML task matched {args.task_id!r}")
    payloads = [build_payload(task) for task in selected]
    result = {"built_at": now(), "task_count": len(payloads), "raw_rows_exposed": False, "payloads": payloads}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "task_count": len(payloads),
        "tasks": [p["task"]["task_id"] for p in payloads],
        "problem_types": [p["task"]["problem_type"] for p in payloads],
        "primary_metrics": [p["task"]["primary_metric"] for p in payloads],
        "raw_rows_exposed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
