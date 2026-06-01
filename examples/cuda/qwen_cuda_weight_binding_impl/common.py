"""Shared constants and file helpers for Qwen CUDA weight binding."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INDEX = (
    ROOT / "tmp" / "sources" / "qwen3-8b-model-safetensors-index-d117af2f.json"
)
DEFAULT_SHARD_DIR = ROOT / "tmp" / "sources" / "qwen3-8b-safetensors"
DEFAULT_HOST_RUNTIME = (
    ROOT / "build" / "lib" / "cuda" / "onboard" / "host_schedule" / "libhost_runtime.so"
)
MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "d117af2f304f02a8647f88fe05b61cfb405a1d9e"
DEFAULT_COPY_CHUNK_BYTES = 64 * 1024 * 1024


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()

def load_python_payload(
    path: Path,
    module_name: str,
    build_name: str,
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return getattr(module, build_name)(*args, **kwargs)

