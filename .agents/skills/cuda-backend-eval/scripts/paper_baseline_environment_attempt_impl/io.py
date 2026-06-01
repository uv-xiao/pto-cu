"""JSON I/O helpers for paper baseline environment attempts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from paper_baseline_environment_attempt_impl.errors import fail
from paper_baseline_environment_attempt_impl.paths import VIEWER_DATA
from viewer_data_io import load_json as load_viewer_json
from viewer_data_io import write_json as write_viewer_json


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        fail(f"missing JSON file: {path}")
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {path}: {exc}")
    if not isinstance(payload, dict):
        fail(f"JSON root is not an object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def is_viewer_output(path: Path) -> bool:
    return (
        path.resolve().parent == VIEWER_DATA.resolve()
        or path.with_suffix("").is_dir()
    )


def load_viewer_output(path: Path) -> dict[str, Any]:
    if is_viewer_output(path):
        return load_viewer_json(path)
    return load_json(path)


def write_viewer_output(path: Path, payload: dict[str, Any]) -> None:
    if is_viewer_output(path):
        write_viewer_json(path, payload)
        return
    write_json(path, payload)
