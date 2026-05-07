"""Inject the JetsonReborn_rebar checkout onto sys.path so the existing
`broker` (rectify + pcd) and `hik` (capture) modules can be imported in-place.

Call `prepare_paths()` once near the top of any entrypoint before importing
those modules. Idempotent.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from .config import load_config

_PREPARED: bool = False


def prepare_paths(config_path: Optional[str | Path] = None) -> Path:
    """Prepend `jetson_reborn_path` to sys.path. Returns the resolved path."""
    global _PREPARED
    cfg = load_config(config_path)
    jr_path = cfg.jetson_reborn_path
    if not jr_path.exists():
        raise FileNotFoundError(
            f"jetson_reborn_path does not exist: {jr_path}. "
            f"Edit configs/calib.yaml or set $EYE2HAND_CONFIG."
        )
    s = str(jr_path)
    if s not in sys.path:
        sys.path.insert(0, s)
    _PREPARED = True
    return jr_path
