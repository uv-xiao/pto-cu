#!/usr/bin/env python3
import argparse
import json
import math
import shutil
import urllib.request
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash"
DEFAULT_ARTIFACT_DIR = ROOT / "tmp" / "model-artifacts" / "deepseek-ai" / (
    "DeepSeek-V4-Flash"
)
DEFAULT_METADATA = ROOT / "tmp" / "sources" / "model-metadata" / (
    "deepseek-ai-DeepSeek-V4-Flash.json"
)
INDEX_NAME = "model.safetensors.index.json"
NON_CLAIMS = [
    "not serving evidence",
    "not model-load evidence",
    "not DeepSeek correctness evidence",
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fetch_hf_metadata(model_id: str) -> dict[str, Any]:
    url = f"https://huggingface.co/api/models/{model_id}"
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def _safe_metadata(
    path: Path | None,
    *,
    fetch_hf_metadata: bool,
    metadata_fetcher: Callable[[str], dict[str, Any]] | None,
) -> tuple[dict[str, Any], str, str | None]:
    if path is not None and path.is_file():
        return _read_json(path), "file", None
    if fetch_hf_metadata:
        fetcher = metadata_fetcher or _fetch_hf_metadata
        try:
            return fetcher(MODEL_ID), "hf_api", None
        except Exception as error:
            return {}, "hf_api_error", f"{type(error).__name__}: {error}"
    return {}, "missing", None


def _safe_int(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def _display_path(path: Path, *, include_external_name: bool = True) -> str:
    absolute_path = path if path.is_absolute() else ROOT / path
    try:
        return absolute_path.resolve().relative_to(ROOT).as_posix()
    except (OSError, ValueError):
        pass
    if not path.is_absolute():
        return path.as_posix()
    if include_external_name and path.name:
        return f"<external>/{path.name}"
    return "<external>"


def _first_existing_path(path: Path) -> Path | None:
    candidate = path
    while True:
        if candidate.exists():
            return candidate
        if candidate.parent == candidate:
            return None
        candidate = candidate.parent


def _free_bytes(path: Path, override: int | None) -> int | None:
    if override is not None:
        return override
    capacity_path = _first_existing_path(path)
    if capacity_path is None:
        return None
    return shutil.disk_usage(capacity_path).free


def _required_capacity_bytes(
    remaining_bytes: int | None,
    *,
    capacity_multiplier: float,
    reserve_bytes: int,
) -> int | None:
    if remaining_bytes is None:
        return None
    if remaining_bytes == 0:
        return 0
    return math.ceil(remaining_bytes * capacity_multiplier) + reserve_bytes


def _preflight_status(
    *,
    artifact_dir: Path,
    index_path: Path,
    indexed_bytes: int | None,
    missing_shard_count: int,
    has_required_capacity: bool | None,
) -> str:
    if has_required_capacity is False:
        return "blocked_storage_capacity"
    if not artifact_dir.is_dir():
        return "blocked_missing_artifact_dir"
    if not index_path.is_file():
        return "blocked_missing_index"
    if indexed_bytes is None:
        return "blocked_unknown_indexed_bytes"
    if missing_shard_count > 0:
        return "ready_for_shard_download"
    return "ready_for_model_load"


def build_preflight(
    artifact_dir: Path,
    metadata_path: Path | None,
    *,
    download_root: Path | None = None,
    reserve_bytes: int = 0,
    capacity_multiplier: float = 1.0,
    filesystem_free_bytes: int | None = None,
    fetch_hf_metadata: bool = False,
    metadata_fetcher: Callable[[str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    metadata, metadata_source, metadata_error = _safe_metadata(
        metadata_path,
        fetch_hf_metadata=fetch_hf_metadata,
        metadata_fetcher=metadata_fetcher,
    )
    index_path = artifact_dir / INDEX_NAME
    index = _read_json(index_path) if index_path.is_file() else {}
    weight_map = index.get("weight_map", {})
    shard_names = sorted(set(weight_map.values())) if isinstance(weight_map, dict) else []
    present_names = [name for name in shard_names if (artifact_dir / name).is_file()]
    missing_names = [name for name in shard_names if name not in set(present_names)]
    present_bytes = sum((artifact_dir / name).stat().st_size for name in present_names)

    index_metadata = index.get("metadata", {})
    indexed_bytes = (
        _safe_int(index_metadata.get("total_size"))
        if isinstance(index_metadata, dict)
        else None
    )
    metadata_storage_bytes = _safe_int(metadata.get("usedStorage"))
    safetensors = metadata.get("safetensors", {})
    metadata_safetensors_total_bytes = (
        _safe_int(safetensors.get("total")) if isinstance(safetensors, dict) else None
    )

    estimate_source_bytes = indexed_bytes
    if estimate_source_bytes is None:
        estimate_source_bytes = metadata_storage_bytes
    estimated_remaining = (
        max(estimate_source_bytes - present_bytes, 0)
        if estimate_source_bytes is not None
        else None
    )
    selected_root = download_root or artifact_dir
    free_bytes = _free_bytes(selected_root, filesystem_free_bytes)
    required_bytes = _required_capacity_bytes(
        estimated_remaining,
        capacity_multiplier=capacity_multiplier,
        reserve_bytes=reserve_bytes,
    )
    has_required_capacity = (
        free_bytes >= required_bytes
        if free_bytes is not None and required_bytes is not None
        else None
    )
    if estimated_remaining == 0 and required_bytes == 0:
        has_required_capacity = True

    manifest_complete = bool(shard_names) and not missing_names
    can_attempt_download = (
        bool(missing_names)
        and index_path.is_file()
        and indexed_bytes is not None
        and has_required_capacity is True
    )
    status = _preflight_status(
        artifact_dir=artifact_dir,
        index_path=index_path,
        indexed_bytes=indexed_bytes,
        missing_shard_count=len(missing_names),
        has_required_capacity=has_required_capacity,
    )

    return {
        "model_id": metadata.get("id") or metadata.get("modelId") or MODEL_ID,
        "artifact_dir": _display_path(artifact_dir),
        "download_root": _display_path(selected_root, include_external_name=False),
        "index_path": _display_path(index_path),
        "manifest_status": "complete" if manifest_complete else "incomplete",
        "indexed_shard_count": len(shard_names),
        "present_shard_count": len(present_names),
        "missing_shard_count": len(missing_names),
        "indexed_tensor_count": len(weight_map) if isinstance(weight_map, dict) else 0,
        "indexed_bytes": indexed_bytes,
        "present_bytes": present_bytes,
        "metadata_storage_bytes": metadata_storage_bytes,
        "metadata_safetensors_total_bytes": metadata_safetensors_total_bytes,
        "metadata_source": metadata_source,
        "metadata_error": metadata_error,
        "estimated_required_bytes_remaining": estimated_remaining,
        "filesystem_free_bytes": free_bytes,
        "capacity_multiplier": capacity_multiplier,
        "reserve_bytes": reserve_bytes,
        "required_capacity_bytes": required_bytes,
        "has_required_capacity": has_required_capacity,
        "missing_shard_examples": missing_names[:5],
        "can_attempt_download": can_attempt_download,
        "can_attempt_model_load": manifest_complete,
        "preflight_status": status,
        "non_claims": NON_CLAIMS,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dry-run DeepSeek-V4-Flash weight acquisition preflight."
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
        "--download-root",
        type=Path,
        default=None,
        help="Directory whose free space should be checked for shard download.",
    )
    parser.add_argument(
        "--reserve-bytes",
        type=int,
        default=0,
        help="Additional free-byte reserve required beyond remaining shard bytes.",
    )
    parser.add_argument(
        "--capacity-multiplier",
        type=float,
        default=1.0,
        help="Safety multiplier applied to estimated remaining shard bytes.",
    )
    parser.add_argument(
        "--filesystem-free-bytes",
        type=int,
        default=None,
        help="Override free-byte capacity for deterministic dry-run checks.",
    )
    parser.add_argument(
        "--fetch-hf-metadata",
        action="store_true",
        help="Fetch Hugging Face model metadata when --metadata is missing.",
    )
    parser.add_argument(
        "--require-capacity",
        action="store_true",
        help="Exit nonzero unless the selected root has required free bytes.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.reserve_bytes < 0:
        raise SystemExit("--reserve-bytes must be non-negative")
    if args.capacity_multiplier < 1.0:
        raise SystemExit("--capacity-multiplier must be at least 1.0")
    if args.filesystem_free_bytes is not None and args.filesystem_free_bytes < 0:
        raise SystemExit("--filesystem-free-bytes must be non-negative")

    payload = build_preflight(
        args.artifact_dir,
        args.metadata,
        download_root=args.download_root,
        reserve_bytes=args.reserve_bytes,
        capacity_multiplier=args.capacity_multiplier,
        filesystem_free_bytes=args.filesystem_free_bytes,
        fetch_hf_metadata=args.fetch_hf_metadata,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.require_capacity and payload["has_required_capacity"] is not True:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
