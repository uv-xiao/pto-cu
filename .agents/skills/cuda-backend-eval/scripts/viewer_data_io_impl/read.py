from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .sidecars import expand_record_sidecars


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
        records.append(expand_record_sidecars(base, collection, record))
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
