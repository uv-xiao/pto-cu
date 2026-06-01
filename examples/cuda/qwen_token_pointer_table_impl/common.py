"""Shared helpers for Qwen token pointer-table artifacts."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
TOKEN_BUFFER_SCRIPT = ROOT / "examples" / "cuda" / "qwen_cuda_token_buffer_binding.py"
DECODE_ARGS_SCRIPT = ROOT / "examples" / "cuda" / "qwen_persistent_decode_args.py"
DEFAULT_HOST_RUNTIME = (
    ROOT / "build" / "lib" / "cuda" / "onboard" / "host_schedule" / "libhost_runtime.so"
)
TOKEN_BUFFERS = ["input_ids", "attention_mask", "output_ids"]


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
