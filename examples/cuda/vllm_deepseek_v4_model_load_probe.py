#!/usr/bin/env python3
"""Bounded vLLM DeepSeek V4 Flash model-load probe."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = (
    ROOT / "tmp" / "model-artifacts" / "deepseek-ai" / "DeepSeek-V4-Flash"
)
DEFAULT_MAX_MODEL_LEN = 4096
DEFAULT_TENSOR_PARALLEL_SIZE = 2
DEFAULT_DTYPE = "bfloat16"
DEFAULT_QUANTIZATION = "deepseek_v4_fp8"
DEFAULT_KV_CACHE_DTYPE = "fp8"
DEFAULT_GPU_MEMORY_UTILIZATION = 0.82
DEFAULT_DISTRIBUTED_EXECUTOR_BACKEND = "mp"
NON_CLAIMS = [
    "not serving evidence",
    "not inference correctness evidence",
    "not tokenizer semantic correctness evidence",
    "not 256K context evidence",
    "not throughput or latency evidence",
]


def _display_path(path: Path) -> str:
    absolute_path = path if path.is_absolute() else ROOT / path
    try:
        return str(absolute_path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def classify_failure(message: str) -> str:
    lowered = message.lower()
    if "outofmemory" in lowered or "out of memory" in lowered:
        return "cuda_out_of_memory"
    if "nccl" in lowered:
        return "nccl"
    if "no such file" in lowered or "not found" in lowered or "missing" in lowered:
        return "missing_artifact"
    if "config" in lowered and (
        "unsupported" in lowered
        or "no supported" in lowered
        or "not supported" in lowered
    ):
        return "model_config"
    if "timeout" in lowered or "timed out" in lowered:
        return "timeout"
    return "unknown"


def build_llm_kwargs(
    *,
    artifact_dir: Path,
    max_model_len: int,
    tensor_parallel_size: int,
    dtype: str,
    quantization: str | None,
    kv_cache_dtype: str,
    gpu_memory_utilization: float,
    enforce_eager: bool,
    distributed_executor_backend: str,
    trust_remote_code: bool,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": str(artifact_dir),
        "tokenizer": str(artifact_dir),
        "tokenizer_mode": "deepseek_v4",
        "tensor_parallel_size": tensor_parallel_size,
        "dtype": dtype,
        "quantization": quantization,
        "kv_cache_dtype": kv_cache_dtype,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enforce_eager": enforce_eager,
        "distributed_executor_backend": distributed_executor_backend,
        "trust_remote_code": trust_remote_code,
    }
    if quantization is None:
        del kwargs["quantization"]
    return kwargs


def inspect_artifacts(artifact_dir: Path, require_artifacts: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "artifact_dir": _display_path(artifact_dir),
        "required_files": {"present": [], "missing": []},
    }
    if not artifact_dir.is_dir():
        result.update(
            {
                "status": "failed" if require_artifacts else "skipped",
                "reason": "artifact directory is missing",
            }
        )
        return result

    present = []
    missing = []
    for name in (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer_config.json",
    ):
        if (artifact_dir / name).is_file():
            present.append(name)
        else:
            missing.append(name)
    tokenizer_files = [
        name
        for name in ("tokenizer.json", "tokenizer.model")
        if (artifact_dir / name).is_file()
    ]
    if tokenizer_files:
        present.extend(tokenizer_files)
    else:
        missing.append("tokenizer.json or tokenizer.model")

    result["required_files"] = {
        "present": sorted(present),
        "missing": sorted(missing),
    }
    if missing:
        result.update(
            {
                "status": "failed" if require_artifacts else "skipped",
                "reason": "required artifact files are missing",
            }
        )
        return result

    index = _read_json(artifact_dir / "model.safetensors.index.json")
    shard_names = sorted(set(index.get("weight_map", {}).values()))
    present_shards = [
        name for name in shard_names if (artifact_dir / name).is_file()
    ]
    present_shard_set = set(present_shards)
    missing_shards = [name for name in shard_names if name not in present_shard_set]
    result.update(
        {
            "indexed_tensors": len(index.get("weight_map", {})),
            "indexed_shards": len(shard_names),
            "present_shards": len(present_shards),
            "missing_shards": len(missing_shards),
            "present_bytes": sum(
                (artifact_dir / name).stat().st_size for name in present_shards
            ),
            "index_total_size": index.get("metadata", {}).get("total_size"),
            "missing_examples": missing_shards[:5],
        }
    )
    if not shard_names:
        result.update(
            {
                "status": "failed" if require_artifacts else "skipped",
                "reason": "weight index does not list shards",
            }
        )
    elif missing_shards:
        result.update(
            {
                "status": "failed" if require_artifacts else "skipped",
                "reason": "indexed weight shards are missing",
            }
        )
    else:
        result["status"] = "passed"
    return result


def _query_nvidia_smi_memory() -> list[dict[str, str]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.free,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        return []
    if completed.returncode != 0:
        return [{"error": completed.stderr.strip()}]
    rows = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        rows.append(
            {
                "index": parts[0],
                "name": parts[1],
                "memory_used_mib": parts[2],
                "memory_free_mib": parts[3],
                "memory_total_mib": parts[4],
            }
        )
    return rows


def _runtime_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {
        "python": platform.python_version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    if importlib.util.find_spec("vllm") is not None:
        vllm = importlib.import_module("vllm")
        versions["vllm"] = getattr(vllm, "__version__", "unknown")
    if importlib.util.find_spec("torch") is not None:
        torch = importlib.import_module("torch")
        versions["torch"] = getattr(torch, "__version__", "unknown")
        versions["torch_cuda"] = getattr(getattr(torch, "version", None), "cuda", None)
        versions["torch_cuda_device_count"] = (
            torch.cuda.device_count() if torch.cuda.is_available() else 0
        )
    return versions


def _failure_payload(message: str, traceback_text: str | None = None) -> dict[str, str]:
    payload = {
        "category": classify_failure(message),
        "message": message,
    }
    if traceback_text:
        payload["traceback"] = traceback_text
    return payload


def run_probe(
    *,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    tensor_parallel_size: int = DEFAULT_TENSOR_PARALLEL_SIZE,
    dtype: str = DEFAULT_DTYPE,
    quantization: str | None = DEFAULT_QUANTIZATION,
    kv_cache_dtype: str = DEFAULT_KV_CACHE_DTYPE,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
    enforce_eager: bool = True,
    distributed_executor_backend: str = DEFAULT_DISTRIBUTED_EXECUTOR_BACKEND,
    trust_remote_code: bool = False,
    dry_run: bool = False,
    require_artifacts: bool = False,
    require_vllm: bool = False,
) -> dict[str, Any]:
    artifact_probe = inspect_artifacts(artifact_dir, require_artifacts)
    llm_kwargs = build_llm_kwargs(
        artifact_dir=artifact_dir,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        quantization=quantization,
        kv_cache_dtype=kv_cache_dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        distributed_executor_backend=distributed_executor_backend,
        trust_remote_code=trust_remote_code,
    )
    result: dict[str, Any] = {
        "status": "planned" if dry_run else "skipped",
        "artifact_probe": artifact_probe,
        "llm_kwargs": llm_kwargs,
        "load_attempted": False,
        "runtime_versions": _runtime_versions(),
        "gpu_memory_before": _query_nvidia_smi_memory(),
        "non_claims": NON_CLAIMS,
    }

    if artifact_probe["status"] != "passed":
        result["status"] = "failed" if require_artifacts else "skipped"
        result["failure"] = _failure_payload(artifact_probe.get("reason", "missing"))
        return result

    if dry_run:
        return result

    if importlib.util.find_spec("vllm") is None:
        result["status"] = "failed" if require_vllm else "skipped"
        result["failure"] = _failure_payload(
            "vLLM is not installed in the active Python environment."
        )
        return result

    result["load_attempted"] = True
    started = time.monotonic()
    try:
        llm_cls = importlib.import_module("vllm").LLM
        llm = llm_cls(**llm_kwargs)
        result["loaded_engine_class"] = type(llm).__name__
        engine = getattr(llm, "llm_engine", None)
        if engine is not None:
            result["llm_engine_class"] = type(engine).__name__
            shutdown = getattr(engine, "shutdown", None)
            if callable(shutdown):
                shutdown()
        result["status"] = "passed"
    except Exception as exc:  # pragma: no cover - exercised on remote failures.
        result["status"] = "failed"
        result["failure"] = _failure_payload(
            f"{type(exc).__name__}: {exc}",
            "".join(traceback.format_exception(exc)),
        )
    finally:
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        result["gpu_memory_after"] = _query_nvidia_smi_memory()
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="repo-relative DeepSeek-V4-Flash artifact directory",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=DEFAULT_MAX_MODEL_LEN,
        help="bounded vLLM max_model_len for the load attempt",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=DEFAULT_TENSOR_PARALLEL_SIZE,
        help="vLLM tensor_parallel_size; use with matching CUDA_VISIBLE_DEVICES",
    )
    parser.add_argument("--dtype", default=DEFAULT_DTYPE)
    parser.add_argument("--quantization", default=DEFAULT_QUANTIZATION)
    parser.add_argument("--kv-cache-dtype", default=DEFAULT_KV_CACHE_DTYPE)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
    )
    parser.add_argument(
        "--distributed-executor-backend",
        default=DEFAULT_DISTRIBUTED_EXECUTOR_BACKEND,
    )
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-artifacts", action="store_true")
    parser.add_argument("--require-vllm", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_probe(
        artifact_dir=args.artifact_dir,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        quantization=args.quantization,
        kv_cache_dtype=args.kv_cache_dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        distributed_executor_backend=args.distributed_executor_backend,
        trust_remote_code=args.trust_remote_code,
        dry_run=args.dry_run,
        require_artifacts=args.require_artifacts,
        require_vllm=args.require_vllm,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] == "failed":
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
