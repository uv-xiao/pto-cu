#!/usr/bin/env python3
"""Read and write benchmark-viewer JSON, including sharded collections."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


COLLECTION_KEYS = (
    "result_records",
    "paper_baseline_execution_attempts",
    "paper_baseline_run_readiness",
)


def load_json(path: Path) -> dict[str, Any]:
    if path.is_dir():
        return load_sharded_collection(path)
    if not path.is_file() and path.suffix == ".json":
        sharded = path.with_suffix("")
        if sharded.is_dir():
            return load_sharded_collection(sharded)
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and is_sharded_manifest(data):
        return expand_manifest(path.parent, data)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return data


def write_json(path: Path, payload: Any) -> None:
    if not isinstance(payload, dict):
        path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
        return
    target = sharded_target(path, payload)
    if target is not None:
        write_sharded_collection(target, payload)
        return
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def is_sharded_manifest(data: dict[str, Any]) -> bool:
    return isinstance(data.get("collection"), str) and (
        isinstance(data.get("record_files"), list)
        or isinstance(data.get("record_files_path"), str)
    )


def expand_manifest(base: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    collection = str(manifest["collection"])
    record_files = manifest_record_files(base, manifest)
    records = []
    for relpath in record_files:
        if not isinstance(relpath, str):
            raise ValueError(f"invalid sharded record path in {base}")
        record = json.loads((base / relpath).read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            raise ValueError(f"sharded record is not an object: {base / relpath}")
        records.append(record)
    payload = {
        key: value
        for key, value in manifest.items()
        if key not in {"collection", "record_files", "record_files_path"}
    }
    payload[collection] = records
    return payload


def load_sharded_collection(path: Path) -> dict[str, Any]:
    manifest = json.loads((path / "index.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not is_sharded_manifest(manifest):
        raise ValueError(f"invalid sharded collection index: {path / 'index.json'}")
    return expand_manifest(path, manifest)


def sharded_target(path: Path, payload: dict[str, Any]) -> Path | None:
    if path.is_dir():
        return path
    if path.suffix == ".json" and path.with_suffix("").is_dir():
        return path.with_suffix("")
    if path.name == "results.json" and "result_records" in payload:
        return path.with_suffix("")
    if (
        path.name == "paper_baseline_run_readiness.json"
        and "paper_baseline_run_readiness" in payload
    ):
        return path.with_suffix("")
    return None


def write_sharded_collection(path: Path, payload: dict[str, Any]) -> None:
    collection = collection_key(payload)
    records = payload[collection]
    if not isinstance(records, list):
        raise ValueError(f"{collection} is not a list")
    records_dir = path / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    record_files = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{collection} contains a non-object record")
        relpath = f"records/{record_filename(collection, index, record)}"
        (path / relpath).write_text(
            json.dumps(record, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        record_files.append(relpath)
    for old in records_dir.glob("*.json"):
        if f"records/{old.name}" not in record_files:
            old.unlink()
    manifest = {
        key: value
        for key, value in payload.items()
        if key != collection
    }
    manifest["collection"] = collection
    if collection == "result_records":
        (path / "record_files.json").write_text(
            json.dumps(record_files, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        manifest["record_files_path"] = "record_files.json"
    else:
        manifest["record_files"] = record_files
    (path / "index.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def collection_key(payload: dict[str, Any]) -> str:
    matches = [key for key in COLLECTION_KEYS if key in payload]
    if len(matches) != 1:
        raise ValueError(f"expected one sharded collection key, got {matches}")
    return matches[0]


def manifest_record_files(base: Path, manifest: dict[str, Any]) -> list[Any]:
    if isinstance(manifest.get("record_files"), list):
        return manifest["record_files"]
    relpath = manifest.get("record_files_path")
    if not isinstance(relpath, str):
        raise ValueError(f"sharded collection missing record files: {base}")
    record_files = json.loads((base / relpath).read_text(encoding="utf-8"))
    if not isinstance(record_files, list):
        raise ValueError(f"record_files_path is not a list: {base / relpath}")
    return record_files


def record_filename(collection: str, index: int, record: dict[str, Any]) -> str:
    if "id" in record:
        return f"{record['id']}.json"
    prefix = slug(record_prefix(record))
    digest = hashlib.sha1(
        record_identity(record).encode("utf-8"),
    ).hexdigest()[:10]
    return f"{index:03d}-{prefix}-{digest}.json"


def record_prefix(record: dict[str, Any]) -> str:
    return "-".join(
        str(value)
        for value in (
            record.get("benchmark_id", "record"),
            record.get("method_id", "method"),
            record.get("hardware", {}).get("gpu", "gpu"),
        )
    )


def record_identity(record: dict[str, Any]) -> str:
    hardware = record.get("hardware", {})
    inputs = record.get("inputs", {})
    return json.dumps(
        [
            record.get("benchmark_id", ""),
            record.get("method_id", ""),
            hardware.get("gpu", ""),
            inputs.get("shape", ""),
            record.get("raw_artifact", ""),
        ],
        sort_keys=True,
    )


def slug(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return normalized[:72] or "record"
