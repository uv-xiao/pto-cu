"""Filesystem and process helpers for PTO serving preflight."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from pto_serving_preflight_impl.constants import ROOT, SERVING_SCAFFOLD


def fail(message: str) -> None:
    raise SystemExit(f"pto serving preflight failed: {message}")


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        fail(f"missing JSON file: {path}")
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {path}: {exc}")
    if not isinstance(data, dict):
        fail(f"JSON root is not an object: {path}")
    return data


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def text_contains(path: str, needles: list[str]) -> bool:
    full_path = ROOT / path
    if not full_path.is_file():
        return False
    text = full_path.read_text(encoding="utf-8", errors="replace")
    return all(needle in text for needle in needles)


def load_serving_scaffold() -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location(
        "persistent_qwen_serving_scaffold",
        SERVING_SCAFFOLD,
    )
    if spec is None or spec.loader is None:
        fail(f"could not load {SERVING_SCAFFOLD}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.build_scaffold()
