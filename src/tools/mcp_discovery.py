from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import requests

from src.tools.static_tool_catalog import get_static_tool_catalog


def _normalise_tool(tool: dict[str, Any]) -> dict[str, Any]:
    row = dict(tool)
    row.setdefault("aliases", [row.get("name", "UnknownTool")])
    row.setdefault("category", "uncategorized")
    row.setdefault("description", "MCP-discovered tool")
    row.setdefault("input_schema", {"type": "object", "properties": {}})
    row.setdefault("output_schema", {"type": "object", "properties": {}})
    row.setdefault("required_inputs", row.get("input_schema", {}).get("required", []))
    row.setdefault("expected_outputs", [])
    row.setdefault("risks", [])
    row.setdefault("risk_level", "unknown")
    return row


def discover_tools(mcp_url: str | None = None) -> list[dict[str, Any]]:
    """Discover MCP tools from /tools, falling back to the static project catalog."""
    fallback = get_static_tool_catalog()
    if not mcp_url:
        return fallback
    try:
        resp = requests.get(mcp_url.rstrip("/") + "/tools", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            data = data.get("tools") or data.get("result") or []
        if not isinstance(data, list) or not data:
            return fallback
        return [_normalise_tool(t) for t in data if isinstance(t, dict)]
    except Exception:
        return fallback


def write_catalog_artifacts(tools: list[dict[str, Any]], json_path: str | Path, md_path: str | Path) -> None:
    json_path = Path(json_path)
    md_path = Path(md_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"tool_count": len(tools), "tools": tools}
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# MCP Tool Catalog",
        "",
        "This catalog is the visible tool-discovery layer for the tool-learning flywheel. It is populated from a live MCP-style `/tools` endpoint when available and falls back to the project static catalog otherwise.",
        "",
        "| Tool | Category | Required Inputs | Expected Outputs | Risk |",
        "|---|---|---|---|---|",
    ]
    for tool in tools:
        lines.append(
            f"| {tool.get('name','')} | {tool.get('category','')} | {', '.join(tool.get('required_inputs', [])) or '-'} | "
            f"{', '.join(tool.get('expected_outputs', [])) or '-'} | {tool.get('risk_level','')} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover MCP tools and export a machine-readable catalog.")
    parser.add_argument("--mcp-url", default=None)
    parser.add_argument("--output", default="data/mcp/tool_catalog.json")
    parser.add_argument("--report", default="reports/mcp_tool_catalog.md")
    args = parser.parse_args()
    tools = discover_tools(args.mcp_url)
    write_catalog_artifacts(tools, args.output, args.report)
    print(json.dumps({"tool_count": len(tools), "tools": tools}, indent=2))


if __name__ == "__main__":
    main()
