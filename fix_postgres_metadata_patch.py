from pathlib import Path
ROOT = Path('/workspace/self-improving-ml-agent')

schema_path = ROOT / 'src/telemetry/schema.py'
s = schema_path.read_text(encoding='utf-8')
if 'metadata: dict[str, Any] = field(default_factory=dict)' not in s:
    s = s.replace(
        "    error: str | None = None\n    started_at: str = field(default_factory=now_iso)",
        "    error: str | None = None\n    metadata: dict[str, Any] = field(default_factory=dict)\n    started_at: str = field(default_factory=now_iso)",
    )
    if 'metadata: dict[str, Any] = field(default_factory=dict)' not in s:
        raise SystemExit('failed to insert metadata field into ToolCallRecord')
schema_path.write_text(s, encoding='utf-8')

base_path = ROOT / 'src/agents/base.py'
b = base_path.read_text(encoding='utf-8')
if 'metadata: dict[str, Any] | None = None' not in b:
    old = "    def tool_success(self, tool_name: str, arguments: dict[str, Any], output_ref: str | None = None) -> ToolCallRecord:\n        return ToolCallRecord(tool_name=tool_name, arguments=arguments, status='success', output_ref=output_ref)\n"
    new = "    def tool_success(self, tool_name: str, arguments: dict[str, Any], output_ref: str | None = None, metadata: dict[str, Any] | None = None) -> ToolCallRecord:\n        return ToolCallRecord(tool_name=tool_name, arguments=arguments, status='success', output_ref=output_ref, metadata=metadata or {})\n"
    if old not in b:
        raise SystemExit('base.py tool_success pattern not found')
    b = b.replace(old, new)
base_path.write_text(b, encoding='utf-8')
print('fixed ToolCallRecord metadata and BaseAgent tool_success')
