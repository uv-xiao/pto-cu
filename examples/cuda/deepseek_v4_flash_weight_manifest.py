#!/usr/bin/env python3
import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = (
    ROOT / "tmp" / "model-artifacts" / "deepseek-ai" / "DeepSeek-V4-Flash"
)
DEFAULT_METADATA = (
    ROOT
    / "tmp"
    / "sources"
    / "model-metadata"
    / "deepseek-ai-DeepSeek-V4-Flash.json"
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_metadata(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    return _read_json(path)


def _display_path(path: Path) -> str:
    absolute_path = path if path.is_absolute() else ROOT / path
    try:
        return str(absolute_path.relative_to(ROOT))
    except ValueError:
        pass
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _first_existing_path(path: Path) -> Path | None:
    candidate = path
    while True:
        if candidate.exists():
            return candidate
        if candidate.parent == candidate:
            return None
        candidate = candidate.parent


def _model_load_command(artifact_dir: Path) -> str:
    return (
        "PYTHONPATH=$PWD:$PWD/python .venv-vllm-probe/bin/python "
        "examples/cuda/vllm_deepseek_v4_model_load_probe.py "
        f"--artifact-dir {_display_path(artifact_dir)} "
        "--require-artifacts --require-vllm "
        "--max-model-len 4096 --tensor-parallel-size 2 "
        "--dtype bfloat16 --quantization deepseek_v4_fp8 "
        "--kv-cache-dtype fp8 --gpu-memory-utilization 0.78 "
        "--distributed-executor-backend mp --enforce-eager"
    )


def _with_preflight_fields(
    manifest: dict[str, Any],
    *,
    artifact_dir: Path,
    storage_dir: Path | None,
    storage_free_bytes: int | None,
) -> dict[str, Any]:
    capacity_path = storage_dir or _first_existing_path(artifact_dir) or artifact_dir
    free_bytes = storage_free_bytes
    if free_bytes is None and capacity_path.exists():
        free_bytes = shutil.disk_usage(capacity_path).free

    index_total_size = manifest["index_total_size"]
    if isinstance(index_total_size, int):
        required_missing_bytes = max(index_total_size - manifest["present_bytes"], 0)
    else:
        required_missing_bytes = None

    storage_has_capacity = None
    if free_bytes is not None and required_missing_bytes is not None:
        storage_has_capacity = free_bytes >= required_missing_bytes

    if manifest["status"] == "complete":
        preflight_status = "ready_for_model_load"
        next_gate = "run_model_load_probe"
    elif manifest.get("reason") == "artifact directory is missing":
        preflight_status = "blocked_missing_artifact_dir"
        next_gate = "create_artifact_directory"
    elif manifest.get("reason") == "model.safetensors.index.json is missing":
        preflight_status = "blocked_missing_index"
        next_gate = "restore_manifest_index"
    elif required_missing_bytes is None:
        preflight_status = "blocked_unknown_required_bytes"
        next_gate = "restore_index_total_size"
    elif storage_has_capacity is False:
        preflight_status = "blocked_storage_capacity"
        next_gate = "select_larger_artifact_storage"
    elif manifest["status"] == "incomplete":
        preflight_status = "needs_shard_acquisition"
        next_gate = "acquire_missing_shards"
    else:
        preflight_status = "blocked_manifest_status"
        next_gate = "inspect_manifest_status"

    manifest.update(
        {
            "required_missing_bytes": required_missing_bytes,
            "storage_dir": _display_path(capacity_path),
            "storage_free_bytes": free_bytes,
            "storage_required_bytes": required_missing_bytes,
            "storage_has_capacity": storage_has_capacity,
            "preflight_status": preflight_status,
            "next_gate": next_gate,
            "next_command": _model_load_command(artifact_dir),
        }
    )
    return manifest


def build_manifest(
    artifact_dir: Path,
    metadata_path: Path | None,
    *,
    storage_dir: Path | None = None,
    storage_free_bytes: int | None = None,
) -> dict[str, Any]:
    index_path = artifact_dir / "model.safetensors.index.json"
    if not artifact_dir.is_dir():
        return _with_preflight_fields(
            {
                "status": "missing",
                "reason": "artifact directory is missing",
                "model_id": "unknown",
                "artifact_dir": _display_path(artifact_dir),
                "index_path": _display_path(index_path),
                "indexed_tensors": 0,
                "indexed_shards": 0,
                "present_shards": 0,
                "missing_shards": 0,
                "present_bytes": 0,
                "index_total_size": None,
                "metadata_used_storage": None,
                "metadata_safetensors_total": None,
                "missing_examples": [],
                "non_claim": "not serving evidence",
            },
            artifact_dir=artifact_dir,
            storage_dir=storage_dir,
            storage_free_bytes=storage_free_bytes,
        )
    if not index_path.is_file():
        return _with_preflight_fields(
            {
                "status": "missing",
                "reason": "model.safetensors.index.json is missing",
                "model_id": "unknown",
                "artifact_dir": _display_path(artifact_dir),
                "index_path": _display_path(index_path),
                "indexed_tensors": 0,
                "indexed_shards": 0,
                "present_shards": 0,
                "missing_shards": 0,
                "present_bytes": 0,
                "index_total_size": None,
                "metadata_used_storage": None,
                "metadata_safetensors_total": None,
                "missing_examples": [],
                "non_claim": "not serving evidence",
            },
            artifact_dir=artifact_dir,
            storage_dir=storage_dir,
            storage_free_bytes=storage_free_bytes,
        )
    index = _read_json(index_path)
    metadata = _safe_metadata(metadata_path)

    shard_names = sorted(set(index.get("weight_map", {}).values()))
    present_names = [name for name in shard_names if (artifact_dir / name).is_file()]
    missing_names = [name for name in shard_names if name not in set(present_names)]
    present_bytes = sum((artifact_dir / name).stat().st_size for name in present_names)

    status = "complete" if not missing_names and shard_names else "incomplete"
    safetensors = metadata.get("safetensors", {})
    return _with_preflight_fields(
        {
            "status": status,
            "model_id": metadata.get("id") or metadata.get("modelId") or "unknown",
            "artifact_dir": _display_path(artifact_dir),
            "index_path": _display_path(index_path),
            "indexed_tensors": len(index.get("weight_map", {})),
            "indexed_shards": len(shard_names),
            "present_shards": len(present_names),
            "missing_shards": len(missing_names),
            "present_bytes": present_bytes,
            "index_total_size": index.get("metadata", {}).get("total_size"),
            "metadata_used_storage": metadata.get("usedStorage"),
            "metadata_safetensors_total": safetensors.get("total")
            if isinstance(safetensors, dict)
            else None,
            "missing_examples": missing_names[:5],
            "non_claim": "not serving evidence",
        },
        artifact_dir=artifact_dir,
        storage_dir=storage_dir,
        storage_free_bytes=storage_free_bytes,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize local DeepSeek-V4-Flash weight shard readiness."
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="Directory containing model.safetensors.index.json and shards.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help="Optional Hugging Face model metadata JSON.",
    )
    parser.add_argument(
        "--storage-dir",
        type=Path,
        default=None,
        help="Directory whose free space should be checked for missing shards.",
    )
    parser.add_argument(
        "--storage-free-bytes",
        type=int,
        default=None,
        help="Override free-byte capacity for deterministic preflight checks.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit nonzero unless every indexed shard is present.",
    )
    parser.add_argument(
        "--require-preflight",
        action="store_true",
        help="Exit nonzero unless the manifest permits the model-load gate.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_manifest(
        args.artifact_dir,
        args.metadata,
        storage_dir=args.storage_dir,
        storage_free_bytes=args.storage_free_bytes,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if args.require_complete and manifest["status"] != "complete":
        return 2
    if args.require_preflight and manifest["preflight_status"] != "ready_for_model_load":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
