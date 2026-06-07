from __future__ import annotations

from importlib import import_module
from typing import Any


def _try_import(name: str) -> tuple[bool, str | None, Any | None]:
    try:
        return True, None, import_module(name)
    except Exception as exc:
        return False, f'{name}: {exc.__class__.__name__}: {exc}', None


def check_art_available() -> dict[str, Any]:
    errors: list[str] = []
    art_ok, err, art_mod = _try_import('art')
    if err: errors.append(err)
    alg_ok, err, _ = _try_import('art.langgraph')
    if err: errors.append(err)
    weave_ok, err, _ = _try_import('weave')
    if err: errors.append(err)
    ruler_ok = False
    if art_ok:
        try:
            from src.training.art_ruler_training import get_ruler_score_group
            get_ruler_score_group()
            ruler_ok = True
        except Exception as exc:
            errors.append(f'ruler: {exc.__class__.__name__}: {exc}')
    required_attrs = ['Trajectory', 'TrajectoryGroup', 'gather_trajectory_groups', 'LocalBackend']
    for attr in required_attrs:
        if art_ok and not hasattr(art_mod, attr):
            errors.append(f'art.{attr} missing')
            art_ok = False
    return {'art': art_ok, 'art_langgraph': alg_ok, 'ruler': ruler_ok, 'weave': weave_ok, 'errors': errors}


if __name__ == '__main__':
    import json
    import sys

    status = check_art_available()
    print(json.dumps(status, indent=2, sort_keys=True))
    # A non-zero exit code is useful for shell launcher preflight checks.
    # The optional Weave dependency is not required for the core official path.
    sys.exit(0 if (status.get('art') and status.get('art_langgraph') and status.get('ruler')) else 2)
