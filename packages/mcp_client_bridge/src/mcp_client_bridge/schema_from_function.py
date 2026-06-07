from __future__ import annotations

import inspect
from typing import Any, Callable, get_args, get_origin, get_type_hints


def _annotation_to_schema(annotation: Any) -> dict[str, Any]:
    """Convert a Python annotation to a small JSON-schema-like fragment."""
    if annotation is inspect._empty or annotation is Any:
        return {"type": "string"}

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is None:
        mapping = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            dict: "object",
            list: "array",
            tuple: "array",
            Any: "string",
        }
        return {"type": mapping.get(annotation, "string")}

    if origin in (list, tuple, set):
        item_schema = _annotation_to_schema(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": item_schema}

    if origin is dict:
        return {"type": "object"}

    if str(origin) in {"typing.Union", "types.UnionType"} or origin is getattr(__import__("typing"), "Union", None):
        non_none = [arg for arg in args if arg is not type(None)]
        if len(non_none) == 1:
            schema = _annotation_to_schema(non_none[0])
            schema["nullable"] = True
            return schema
        return {"anyOf": [_annotation_to_schema(arg) for arg in non_none] or [{"type": "string"}]}

    return {"type": "string"}


def function_to_json_schema(func: Callable[..., Any]) -> dict[str, Any]:
    """Extract a JSON-style schema from a Python function signature.

    The result is intentionally lightweight and compatible with local tool
    catalogs. It includes the function name, docstring, input properties, and
    required parameters. The official MCP SDK still owns the runtime tool schema.
    """
    signature = inspect.signature(func)
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, parameter in signature.parameters.items():
        if name in {"self", "cls"}:
            continue
        annotation = hints.get(name, parameter.annotation)
        schema = _annotation_to_schema(annotation)
        schema["description"] = f"Argument `{name}` for `{func.__name__}`."
        if parameter.default is not inspect._empty:
            schema["default"] = parameter.default
        else:
            required.append(name)
        properties[name] = schema

    return {
        "name": func.__name__,
        "description": inspect.getdoc(func) or "",
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


def functions_to_tool_catalog(functions: list[Callable[..., Any]]) -> list[dict[str, Any]]:
    """Return JSON-style schemas for multiple tool functions."""
    return [function_to_json_schema(func) for func in functions]
