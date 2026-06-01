from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from viewer_data_io import write_json as write_viewer_json

from .errors import fail
from .paths import ROOT
from .paths import VIEWER_DATA


def load_json(path: Path) -> dict[str, Any]:
    if path.is_dir():
        return load_sharded_collection(path)
    if not path.is_file() and path.suffix == ".json":
        sharded = path.with_suffix("")
        if sharded.is_dir():
            return load_sharded_collection(sharded)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        fail(f"missing JSON file: {path}")
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {path}: {exc}")
    if not isinstance(data, dict):
        fail(f"JSON root is not an object: {path}")
    return data


def load_sharded_collection(path: Path) -> dict[str, Any]:
    index_path = path / "index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        fail(f"missing sharded collection index: {index_path}")
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {index_path}: {exc}")
    if not isinstance(index, dict):
        fail(f"sharded collection index is not an object: {index_path}")
    collection = index.get("collection")
    record_files = index.get("record_files")
    if record_files is None and isinstance(index.get("record_files_path"), str):
        record_files = json.loads(
            (path / index["record_files_path"]).read_text(encoding="utf-8")
        )
    if not isinstance(collection, str) or not collection:
        fail(f"sharded collection missing collection name: {index_path}")
    if not isinstance(record_files, list):
        fail(f"sharded collection missing record_files list: {index_path}")
    records: list[dict[str, Any]] = []
    for relpath in record_files:
        if not isinstance(relpath, str) or not relpath.endswith(".json"):
            fail(f"invalid sharded record path in {index_path}: {relpath}")
        record_path = path / relpath
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            fail(f"missing sharded record: {record_path}")
        except json.JSONDecodeError as exc:
            fail(f"invalid JSON in {record_path}: {exc}")
        if not isinstance(record, dict):
            fail(f"sharded record is not an object: {record_path}")
        records.append(expand_record_sidecars(path, record))
    payload = {
        key: value
        for key, value in index.items()
        if key not in {"collection", "record_files", "record_files_path"}
    }
    payload[collection] = records
    return payload


def expand_record_sidecars(base: Path, record: dict[str, Any]) -> dict[str, Any]:
    payload = dict(record)
    for field in (
        "current_evidence_refs",
        "missing_evidence_details",
        "paper_baseline_run_statuses",
        "paper_baseline_run_readiness_statuses",
        "execution_attempt_statuses",
        "probe_statuses",
        "next_actions",
    ):
        path_key = f"{field}_path"
        relpath = payload.pop(path_key, None)
        if relpath is None:
            continue
        if not isinstance(relpath, str):
            fail(f"{path_key} is not a string")
        payload[field] = load_sidecar_list(base / relpath)
    return payload


def load_sidecar_list(path: Path) -> list[Any]:
    if path.is_file():
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            fail(f"missing sharded sidecar: {path}")
        except json.JSONDecodeError as exc:
            fail(f"invalid JSON in {path}: {exc}")
        if not isinstance(value, list):
            fail(f"sharded sidecar is not a list: {path}")
        return value
    index_path = path / "index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        fail(f"missing sharded sidecar index: {index_path}")
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {index_path}: {exc}")
    item_files = index.get("item_files")
    if not isinstance(item_files, list):
        fail(f"sharded sidecar index missing item_files: {index_path}")
    values = []
    for relpath in item_files:
        if not isinstance(relpath, str):
            fail(f"invalid sharded sidecar item path: {index_path}")
        item_path = path / relpath
        try:
            values.append(json.loads(item_path.read_text(encoding="utf-8")))
        except FileNotFoundError:
            fail(f"missing sharded sidecar item: {item_path}")
        except json.JSONDecodeError as exc:
            fail(f"invalid JSON in {item_path}: {exc}")
    return values


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def write_output(path: Path, payload: dict[str, Any]) -> None:
    if path.resolve().parent == VIEWER_DATA.resolve() or path.with_suffix("").is_dir():
        write_viewer_json(path, payload)
        return
    write_json(path, payload)


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
