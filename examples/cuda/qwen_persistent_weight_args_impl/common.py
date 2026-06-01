"""Shared constants and file helpers for Qwen persistent weight arguments."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WEIGHT_BINDING = (
    ROOT
    / "tmp"
    / "cuda-backend"
    / "pto-serving-weight-residency-1ae913c9"
    / "qwen-cuda-weight-residency.json"
)
ABI_PATH = "src/cuda/platform/include/host/pto_cuda_persistent_device_abi.h"
TENSOR_ARG_CAPACITY = 4
MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "d117af2f304f02a8647f88fe05b61cfb405a1d9e"


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
