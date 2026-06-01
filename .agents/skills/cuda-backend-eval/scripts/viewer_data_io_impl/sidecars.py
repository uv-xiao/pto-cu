from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .constants import SIDECAR_LIST_FIELDS


def split_record_sidecars(
    base: Path,
    collection: str,
    relpath: str,
    record: dict[str, Any],
) -> dict[str, Any]:
    fields = SIDECAR_LIST_FIELDS.get(collection, ())
    if not fields:
        return record
    payload = dict(record)
    sidecar_dir = Path(relpath).with_suffix("")
    for field in fields:
        value = payload.pop(field, None)
        if value is None:
            continue
        sidecar_relpath = sidecar_dir / field
        sidecar_path = base / sidecar_relpath
        write_sidecar_list(sidecar_path, value)
        legacy_path = sidecar_path.with_suffix(".json")
        if legacy_path.is_file():
            legacy_path.unlink()
        payload[f"{field}_path"] = sidecar_relpath.as_posix()
    return payload


def write_sidecar_list(path: Path, values: Any) -> None:
    if not isinstance(values, list):
        raise ValueError(f"sidecar value is not a list: {path}")
    items_dir = path / "items"
    items_dir.mkdir(parents=True, exist_ok=True)
    item_files = []
    for index, value in enumerate(values):
        relpath = f"items/{index:03d}.json"
        (path / relpath).write_text(
            json.dumps(value, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        item_files.append(relpath)
    for old in items_dir.glob("*.json"):
        if f"items/{old.name}" not in item_files:
            old.unlink()
    (path / "index.json").write_text(
        json.dumps({"item_files": item_files}, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def expand_record_sidecars(
    base: Path,
    collection: str,
    record: dict[str, Any],
) -> dict[str, Any]:
    fields = SIDECAR_LIST_FIELDS.get(collection, ())
    if not fields:
        return record
    payload = dict(record)
    for field in fields:
        path_key = f"{field}_path"
        relpath = payload.pop(path_key, None)
        if relpath is None:
            continue
        if not isinstance(relpath, str):
            raise ValueError(f"{path_key} is not a path string")
        payload[field] = load_sidecar_list(base / relpath)
    return payload


def load_sidecar_list(path: Path) -> list[Any]:
    if path.is_file():
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError(f"sidecar file does not contain a list: {path}")
        return value
    index_path = path / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    item_files = index.get("item_files")
    if not isinstance(item_files, list):
        raise ValueError(f"sidecar index has no item_files: {index_path}")
    values = []
    for relpath in item_files:
        if not isinstance(relpath, str):
            raise ValueError(f"sidecar item path is not a string: {index_path}")
        values.append(json.loads((path / relpath).read_text(encoding="utf-8")))
    return values
