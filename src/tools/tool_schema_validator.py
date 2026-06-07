from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.tools.static_tool_catalog import as_lookup, get_static_tool_catalog

UNSAFE_KEYWORDS = (
    "password",
    "secret",
    "token",
    "private_key",
    "raw_rows",
    "raw rows:",
    "df.head",
    "first 5 rows",
    "sample rows",
    "select *",
    "synthetic_dataset_rows",
    "row_payload",
)

RAW_TABLE_NAMES = {"synthetic_dataset_rows", "raw_dataset_rows", "dataset_rows"}
REGISTRY_METADATA_TOOLS = {
    "postgres_get_task_metadata",
    "postgres_get_dataset_summary",
    "postgres_get_dataset_schema",
    "postgres_get_column_profile",
    "postgres_get_target_profile",
    "PostgresTaskMetadataReader",
    "PostgresDatasetSummaryReader",
    "PostgresDatasetSchemaReader",
}
EXECUTION_LAYER_TOOLS = {"postgres_get_execution_dataset_source", "ExecutionDatasetSourceContract"}


def _schema_type_ok(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return True


def _as_text(value: Any) -> str:
    try:
        return json.dumps(value, default=str).lower()
    except TypeError:
        return str(value).lower()


def _contains_any(text: str, needles: set[str] | tuple[str, ...]) -> list[str]:
    return sorted({needle for needle in needles if needle in text})


def _call_boundary_errors(tool_name: str, call: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    arguments = call.get("arguments") or {}
    metadata = call.get("metadata") or {}
    combined_text = _as_text({"arguments": arguments, "metadata": metadata, "output_ref": call.get("output_ref"), "error": call.get("error")})
    raw_table_hits = _contains_any(combined_text, RAW_TABLE_NAMES)
    if raw_table_hits:
        errors.append(f"forbidden raw-row table reference: {', '.join(raw_table_hits)}")

    raw_flags = []
    for source_name, source in (("arguments", arguments), ("metadata", metadata)):
        if isinstance(source, dict):
            for key, value in source.items():
                key_l = str(key).lower()
                if key_l in {"raw_rows_exposed", "raw_rows_exposed_to_llm", "raw_rows_loaded_into_agent_context"} and value is True:
                    raw_flags.append(f"{source_name}.{key}=true")
    if raw_flags:
        errors.append("raw-row exposure flag set: " + ", ".join(raw_flags))

    if tool_name in REGISTRY_METADATA_TOOLS:
        if metadata.get("raw_rows_exposed") is True or arguments.get("raw_rows_exposed_to_llm") is True:
            errors.append("registry metadata tool exposed raw rows")
    if tool_name in EXECUTION_LAYER_TOOLS:
        if arguments.get("raw_rows_exposed_to_llm") is not False:
            errors.append("execution dataset source must set raw_rows_exposed_to_llm=false")
        if arguments.get("execution_only") is not True and metadata.get("execution_only") is not True:
            errors.append("execution dataset source must be marked execution_only=true")
    return errors


def validate_tool_call(tool_name: str, arguments: dict[str, Any] | None, catalog: list[dict[str, Any]] | None = None, call: dict[str, Any] | None = None) -> dict[str, Any]:
    lookup = as_lookup(catalog or get_static_tool_catalog())
    arguments = arguments or {}
    call = call or {"tool_name": tool_name, "arguments": arguments}
    if tool_name not in lookup:
        return {"valid": False, "tool_exists": False, "errors": [f"unsupported tool: {tool_name}"], "penalty": -0.20}
    tool = lookup[tool_name]
    schema = tool.get("input_schema", {}) or {}
    required = list(schema.get("required") or tool.get("required_inputs") or [])
    properties = schema.get("properties", {}) or {}
    errors: list[str] = []
    for key in required:
        if key not in arguments:
            errors.append(f"missing required arg: {key}")
    for key, value in arguments.items():
        if key not in properties and properties:
            errors.append(f"unsupported arg: {key}")
        expected = (properties.get(key) or {}).get("type")
        if expected and not _schema_type_ok(value, expected):
            errors.append(f"invalid type for {key}: expected {expected}")
        lowered = _as_text({key: value})
        if any(word in lowered for word in UNSAFE_KEYWORDS):
            # Mentioning explicit boolean guardrail keys is allowed when the value is false.
            if not (str(key).lower() in {"raw_rows_exposed", "raw_rows_exposed_to_llm", "raw_rows_loaded_into_agent_context"} and value is False):
                errors.append(f"unsafe arg content: {key}")
    errors.extend(_call_boundary_errors(tool_name, call))
    penalty = 0.0
    for error in errors:
        if error.startswith("missing required"):
            penalty -= 0.15
        elif error.startswith("unsupported arg"):
            penalty -= 0.15
        elif error.startswith("unsafe") or "raw-row" in error or "raw rows" in error:
            penalty -= 0.30
        else:
            penalty -= 0.10
    return {"valid": not errors, "tool_exists": True, "canonical_tool": tool.get("name"), "errors": errors, "penalty": round(penalty, 4)}


def validate_trajectory(traj: dict[str, Any], catalog: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    calls = []
    canonical_names: list[str] = []
    boundary_errors: list[str] = []
    registry_evidence = False
    materialization_evidence = False
    for step in traj.get("steps", []) or []:
        for call in step.get("tool_calls", []) or []:
            tool_name = call.get("tool_name", "")
            result = validate_tool_call(tool_name, call.get("arguments") or {}, catalog, call=call)
            result["tool_name"] = tool_name
            result["status"] = call.get("status")
            calls.append(result)
            canonical = result.get("canonical_tool") or tool_name
            canonical_names.append(str(canonical))
            if canonical in REGISTRY_METADATA_TOOLS or tool_name in REGISTRY_METADATA_TOOLS:
                registry_evidence = True
            if canonical in EXECUTION_LAYER_TOOLS or tool_name in EXECUTION_LAYER_TOOLS:
                materialization_evidence = True
            for err in result.get("errors", []) or []:
                if "raw-row" in err or "raw rows" in err or "synthetic_dataset_rows" in err:
                    boundary_errors.append(f"{tool_name}: {err}")
    total = len(calls)
    valid = sum(1 for c in calls if c.get("valid"))
    failures = sum(1 for c in calls if c.get("status") not in (None, "success", "ok", "completed"))
    boundary_guardrail_passed = not boundary_errors
    mcp_registry_evidence = registry_evidence and materialization_evidence
    evidence_bonus = 0.05 if mcp_registry_evidence and boundary_guardrail_passed else 0.0
    return {
        "tool_calls": total,
        "valid_tool_calls": valid,
        "argument_validity_rate": valid / max(total, 1),
        "tool_call_success_rate": 1.0 - failures / max(total, 1),
        "schema_penalty": round(sum(float(c.get("penalty", 0.0)) for c in calls) + evidence_bonus, 4),
        "call_results": calls,
        "mcp_registry_evidence": mcp_registry_evidence,
        "registry_metadata_tool_seen": registry_evidence,
        "execution_layer_materialization_seen": materialization_evidence,
        "boundary_guardrail_passed": boundary_guardrail_passed,
        "boundary_errors": boundary_errors,
        "canonical_tool_names": canonical_names,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate tool calls against MCP/static schemas and strict raw-row boundary rules.")
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    path = Path(args.trajectory)
    traj = json.loads(path.read_text(encoding="utf-8"))
    result = validate_trajectory(traj)
    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
