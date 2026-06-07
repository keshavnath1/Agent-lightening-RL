from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any


@dataclass
class ToolSpec:
    name: str
    description: str
    handler: Callable[..., Any]


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolSpec] = {}

    def register(self, name: str, description: str, handler: Callable[..., Any]) -> None:
        self._tools[name] = ToolSpec(name=name, description=description, handler=handler)

    def call(self, name: str, **kwargs: Any) -> Any:
        if name not in self._tools:
            raise KeyError(f'Unknown tool: {name}')
        return self._tools[name].handler(**kwargs)

    def describe(self) -> list[dict[str, str]]:
        return [{'name': t.name, 'description': t.description} for t in self._tools.values()]
