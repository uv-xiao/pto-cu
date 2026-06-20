#!/usr/bin/env python3
import argparse
import json
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


def build_manifest(artifact_dir: Path, metadata_path: Path | None) -> dict[str, Any]:
    index_path = artifact_dir / "model.safetensors.index.json"
    if not artifact_dir.is_dir():
        return {
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
        }
    if not index_path.is_file():
        return {
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
        }
    index = _read_json(index_path)
    metadata = _safe_metadata(metadata_path)

    shard_names = sorted(set(index.get("weight_map", {}).values()))
    present_names = [name for name in shard_names if (artifact_dir / name).is_file()]
    missing_names = [name for name in shard_names if name not in set(present_names)]
    present_bytes = sum((artifact_dir / name).stat().st_size for name in present_names)

    status = "complete" if not missing_names and shard_names else "incomplete"
    safetensors = metadata.get("safetensors", {})
    return {
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
    }


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
        "--require-complete",
        action="store_true",
        help="Exit nonzero unless every indexed shard is present.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_manifest(args.artifact_dir, args.metadata)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if args.require_complete and manifest["status"] != "complete":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
