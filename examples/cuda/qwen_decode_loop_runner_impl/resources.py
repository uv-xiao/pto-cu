"""Load Qwen resource lifecycle artifacts for the decode-loop runner."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
TOKEN_POINTER = ROOT / "examples" / "cuda" / "qwen_token_pointer_table.py"
KV_CACHE = ROOT / "examples" / "cuda" / "qwen_kv_cache_binding.py"
RESIDENT_WEIGHTS = ROOT / "examples" / "cuda" / "qwen_resident_weight_table.py"


def load_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_resources(
    *,
    mode: str,
    cache_dir: Path | None,
    token_cuda_live: bool = False,
    kv_cuda_live: bool = False,
    resident_cuda_live: bool = False,
    device: int = 0,
    host_runtime: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    token_module = load_module(TOKEN_POINTER, "qwen_token_pointer_for_runner")
    kv_module = load_module(KV_CACHE, "qwen_kv_cache_for_runner")
    resident_module = load_module(RESIDENT_WEIGHTS, "qwen_resident_weights_for_runner")
    token_kwargs: dict[str, Any] = {
        "mode": mode,
        "cache_dir": cache_dir,
        "cuda_live": token_cuda_live,
        "device": device,
    }
    if host_runtime is not None:
        token_kwargs["host_runtime"] = host_runtime
    kv_kwargs: dict[str, Any] = {"cuda_live": kv_cuda_live, "device": device}
    if host_runtime is not None:
        kv_kwargs["host_runtime"] = host_runtime
    resident_kwargs: dict[str, Any] = {
        "dry_run": not resident_cuda_live,
        "device": device,
    }
    if host_runtime is not None:
        resident_kwargs["host_runtime"] = host_runtime
    return (
        token_module.build_token_pointer_table_lifecycle(**token_kwargs),
        kv_module.build_kv_cache_lifecycle(**kv_kwargs),
        resident_module.build_resident_table_lifecycle(**resident_kwargs),
    )
