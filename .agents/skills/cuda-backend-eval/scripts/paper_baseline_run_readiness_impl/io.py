from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from viewer_data_io import load_json as load_viewer_json
from viewer_data_io import write_json as write_viewer_json

from .errors import fail
from .paths import ROOT
from .paths import VIEWER_DATA


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = load_viewer_json(path)
    except FileNotFoundError:
        fail(f"missing JSON file: {path}")
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {path}: {exc}")
    except ValueError as exc:
        fail(str(exc))
    if not isinstance(data, dict):
        fail(f"JSON root is not an object: {path}")
    return data


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def write_viewer_output(path: Path, payload: dict[str, Any]) -> None:
    if path.resolve().parent == VIEWER_DATA.resolve():
        write_viewer_json(path, payload)
        return
    write_json(path, payload)


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
