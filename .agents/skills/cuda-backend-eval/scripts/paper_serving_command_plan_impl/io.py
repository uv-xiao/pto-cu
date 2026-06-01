"""JSON I/O helpers for paper serving command plans."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from paper_serving_command_plan_impl.paths import VIEWER_DATA
from viewer_data_io import load_json as load_viewer_json
from viewer_data_io import write_json as write_viewer_json


def load_json(path: Path) -> dict[str, Any]:
    return load_viewer_json(path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def write_output(path: Path, payload: Any) -> None:
    if path.resolve().parent == VIEWER_DATA.resolve() or path.with_suffix("").is_dir():
        write_viewer_json(path, payload)
    else:
        write_json(path, payload)
