from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .constants import COLLECTION_KEYS
from .naming import record_filename


def write_json(path: Path, payload: Any) -> None:
    sharded_path = existing_sharded_path(path)
    if sharded_path is not None and isinstance(payload, dict):
        collection = payload_collection(payload)
        if collection is not None:
            write_sharded_collection(sharded_path, payload, collection)
            if path.is_file():
                path.unlink()
            return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def existing_sharded_path(path: Path) -> Path | None:
    if path.is_dir():
        return path
    if path.suffix == ".json" and path.with_suffix("").is_dir():
        return path.with_suffix("")
    return None


def payload_collection(payload: dict[str, Any]) -> str | None:
    for key in COLLECTION_KEYS:
        if isinstance(payload.get(key), list):
            return key
    return None


def write_sharded_collection(
    path: Path,
    payload: dict[str, Any],
    collection: str,
) -> None:
    records = payload[collection]
    records_dir = path / "records"
    if records_dir.exists():
        shutil.rmtree(records_dir)
    records_dir.mkdir(parents=True, exist_ok=True)

    used: set[str] = set()
    record_files: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"sharded {collection} record is not an object")
        name = unique_record_name(collection, index, record, used)
        used.add(name)
        relpath = f"records/{name}"
        record_files.append(relpath)
        (records_dir / name).write_text(
            json.dumps(record, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )

    manifest = {
        key: value
        for key, value in payload.items()
        if key not in {collection, "collection", "record_files", "record_files_path"}
    }
    manifest["collection"] = collection
    manifest["record_files_path"] = "record_files.json"
    write_plain_json(path / "record_files.json", record_files)
    write_plain_json(path / "index.json", manifest)


def unique_record_name(
    collection: str,
    index: int,
    record: dict[str, Any],
    used: set[str],
) -> str:
    name = record_filename(collection, index, record)
    if name not in used:
        return name
    stem = name[:-5] if name.endswith(".json") else name
    return f"{stem}-{index:03d}.json"


def write_plain_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
