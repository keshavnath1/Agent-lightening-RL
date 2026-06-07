from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / 'config' / 'settings.toml'


def load_settings(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with config_path.open('rb') as f:
        data = tomllib.load(f)
    workspace = Path(os.getenv('WORKSPACE_DIR', data['project']['workspace_dir']))
    data['runtime'] = {'workspace_dir': str(workspace)}
    return data


def project_root() -> Path:
    return Path(os.getenv('WORKSPACE_DIR', Path(__file__).resolve().parents[2]))
