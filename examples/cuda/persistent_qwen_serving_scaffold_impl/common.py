"""Shared constants and helpers for the Qwen serving scaffold."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
VIEWER_DATA = ROOT / "evaluations" / "nvidia" / "benchmark-viewer" / "data"
TARGET_WORKLOAD_IDS = {"mpk_offline_decode", "vdcores_offline_decode"}
LIFECYCLE_PLAN = ROOT / "examples" / "cuda" / "qwen_serving_lifecycle_plan.py"
PROMPT_ACCOUNTING = ROOT / "examples" / "cuda" / "qwen_prompt_accounting.py"
RUNTIME_INPUT_BINDING = ROOT / "examples" / "cuda" / "qwen_runtime_input_binding.py"
CUDA_TOKEN_BUFFER_BINDING = ROOT / "examples" / "cuda" / "qwen_cuda_token_buffer_binding.py"
PERSISTENT_DECODE_ARGS = ROOT / "examples" / "cuda" / "qwen_persistent_decode_args.py"
TOKEN_POINTER_TABLE = ROOT / "examples" / "cuda" / "qwen_token_pointer_table.py"
WEIGHT_INVENTORY = ROOT / "examples" / "cuda" / "qwen_weight_inventory.py"
SAFETENSORS_FETCH = ROOT / "examples" / "cuda" / "qwen_safetensors_fetch.py"
SAFETENSORS_METADATA = ROOT / "examples" / "cuda" / "qwen_safetensors_metadata.py"
CUDA_WEIGHT_BINDING = ROOT / "examples" / "cuda" / "qwen_cuda_weight_binding.py"
PERSISTENT_WEIGHT_ARGS = ROOT / "examples" / "cuda" / "qwen_persistent_weight_args.py"
PERSISTENT_WEIGHT_MATERIALIZATION = ROOT / "examples" / "cuda" / "qwen_persistent_weight_materialization.py"
RESIDENT_WEIGHT_TABLE = ROOT / "examples" / "cuda" / "qwen_resident_weight_table.py"
KV_CACHE_BINDING = ROOT / "examples" / "cuda" / "qwen_kv_cache_binding.py"
DECODE_LOOP_RUNNER = ROOT / "examples" / "cuda" / "qwen_decode_loop_runner.py"
TASK_BODIES = ROOT / "examples" / "cuda" / "qwen_persistent_task_bodies.py"


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


def text_contains(path: str, needles: list[str]) -> bool:
    full_path = ROOT / path
    if not full_path.is_file():
        return False
    text = full_path.read_text(encoding="utf-8", errors="replace")
    return all(needle in text for needle in needles)


def load_python_payload(path: Path, module_name: str, build_name: str) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        return {}
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return getattr(module, build_name)()


def load_python_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
