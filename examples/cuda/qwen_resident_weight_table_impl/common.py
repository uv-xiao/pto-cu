"""Shared Qwen resident weight table helpers."""

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
DEFAULT_WEIGHT_ARGS = (
    ROOT
    / "tmp"
    / "cuda-backend"
    / "pto-serving-weight-args-21589e81"
    / "qwen-persistent-weight-args.json"
)
DEFAULT_HOST_RUNTIME = (
    ROOT / "build" / "lib" / "cuda" / "onboard" / "host_schedule" / "libhost_runtime.so"
)
DEFAULT_COPY_CHUNK_BYTES = 64 * 1024 * 1024
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


def load_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_or_build_weight_binding(path: Path | None) -> tuple[dict[str, Any], str]:
    if path is not None:
        return load_json(path), repo_relative(path)
    if DEFAULT_WEIGHT_BINDING.is_file():
        return load_json(DEFAULT_WEIGHT_BINDING), repo_relative(DEFAULT_WEIGHT_BINDING)
    module = load_module(
        ROOT / "examples" / "cuda" / "qwen_cuda_weight_binding.py",
        "qwen_cuda_weight_binding",
    )
    return (
        module.build_weight_binding(no_cuda_probe=True),
        "generated_from_examples/cuda/qwen_cuda_weight_binding.py",
    )


def load_weight_args_path(path: Path | None) -> Path | None:
    if path is not None:
        return path
    return DEFAULT_WEIGHT_ARGS if DEFAULT_WEIGHT_ARGS.is_file() else None
