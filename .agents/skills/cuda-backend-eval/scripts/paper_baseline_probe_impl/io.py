from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from viewer_data_io import load_json as load_viewer_json

from .paths import ROOT


def load_json(path: Path) -> dict[str, Any]:
    return load_viewer_json(path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def repo_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT / path
