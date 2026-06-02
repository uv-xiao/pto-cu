"""Shared helpers for benchmark-viewer data validation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
VIEWER_DATA_ROOT = ROOT / "evaluations" / "nvidia" / "benchmark-viewer" / "data"
ID_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")


def fail(message: str) -> None:
    raise SystemExit(f"benchmark viewer data validation failed: {message}")


def load_json(root: Path, name: str) -> dict[str, Any]:
    path = root / "evaluations" / "nvidia" / "benchmark-viewer" / "data" / name
    if path.is_dir():
        return load_sharded_collection(path)
    if not path.is_file() and path.suffix == ".json":
        sharded = path.with_suffix("")
        if sharded.is_dir():
            return load_sharded_collection(sharded)
    if not path.is_file():
        fail(f"missing data file: {path.relative_to(root)}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {path.relative_to(root)}: {exc}")


def load_sharded_collection(path: Path) -> dict[str, Any]:
    index_path = path / "index.json"
    if not index_path.is_file():
        fail(f"missing sharded collection index: {index_path.relative_to(ROOT)}")
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {index_path.relative_to(ROOT)}: {exc}")
    if not isinstance(index, dict):
        fail(
            "sharded collection index is not an object: "
            f"{index_path.relative_to(ROOT)}"
        )
    collection = index.get("collection")
    record_files = index.get("record_files")
    if record_files is None and isinstance(index.get("record_files_path"), str):
        record_files_path = path / index["record_files_path"]
        if not record_files_path.is_file():
            fail(f"missing sharded record list: {record_files_path.relative_to(ROOT)}")
        try:
            record_files = json.loads(record_files_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            fail(f"invalid JSON in {record_files_path.relative_to(ROOT)}: {exc}")
    if not isinstance(collection, str) or not collection:
        fail(
            "sharded collection has no collection name: "
            f"{index_path.relative_to(ROOT)}"
        )
    if not isinstance(record_files, list) or not record_files:
        fail(
            "sharded collection has no record_files: "
            f"{index_path.relative_to(ROOT)}"
        )
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for relpath in record_files:
        if not isinstance(relpath, str) or not relpath.endswith(".json"):
            fail(f"invalid sharded record path in {index_path.relative_to(ROOT)}")
        if relpath in seen:
            fail(f"duplicate sharded record path: {relpath}")
        seen.add(relpath)
        record_path = path / relpath
        if not record_path.is_file():
            fail(f"missing sharded record: {record_path.relative_to(ROOT)}")
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            fail(f"invalid JSON in {record_path.relative_to(ROOT)}: {exc}")
        if not isinstance(record, dict):
            fail(f"sharded record is not an object: {record_path.relative_to(ROOT)}")
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
        except json.JSONDecodeError as exc:
            fail(f"invalid JSON in {path.relative_to(ROOT)}: {exc}")
        if not isinstance(value, list):
            fail(f"sharded sidecar is not a list: {path.relative_to(ROOT)}")
        return value
    index_path = path / "index.json"
    if not index_path.is_file():
        fail(f"missing sharded sidecar index: {index_path.relative_to(ROOT)}")
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {index_path.relative_to(ROOT)}: {exc}")
    item_files = index.get("item_files")
    if not isinstance(item_files, list):
        fail(f"sharded sidecar has no item_files: {index_path.relative_to(ROOT)}")
    values = []
    for relpath in item_files:
        if not isinstance(relpath, str):
            fail(f"invalid sharded sidecar item: {index_path.relative_to(ROOT)}")
        item_path = path / relpath
        try:
            values.append(json.loads(item_path.read_text(encoding="utf-8")))
        except FileNotFoundError:
            fail(f"missing sharded sidecar item: {item_path.relative_to(ROOT)}")
        except json.JSONDecodeError as exc:
            fail(f"invalid JSON in {item_path.relative_to(ROOT)}: {exc}")
    return values


def logical_data_path_exists(root: Path, relpath: str) -> bool:
    path = root / relpath
    if path.is_file():
        return True
    return (
        path.suffix == ".json"
        and path.with_suffix("").is_dir()
        and (path.with_suffix("") / "index.json").is_file()
    )


def logical_data_text(root: Path, relpath: str) -> str:
    path = root / relpath
    if path.is_file():
        return path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".json" and path.with_suffix("").is_dir():
        return "\n".join(
            child.read_text(encoding="utf-8", errors="replace")
            for child in sorted(path.with_suffix("").rglob("*.json"))
        )
    return ""


def require_string(record: dict[str, Any], key: str, owner: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        fail(f"{owner} has empty or missing {key}")
    return value


def require_dict(record: dict[str, Any], key: str, owner: str) -> dict[str, Any]:
    value = record.get(key)
    if not isinstance(value, dict) or not value:
        fail(f"{owner} has empty or missing {key}")
    return value


def require_list(record: dict[str, Any], key: str, owner: str) -> list[Any]:
    value = record.get(key)
    if not isinstance(value, list) or not value:
        fail(f"{owner} has empty or missing {key}")
    return value


def validate_id(identifier: str, owner: str) -> None:
    if not ID_RE.fullmatch(identifier):
        fail(f"{owner} id is not stable snake_case: {identifier}")


def check_unique_ids(records: list[dict[str, Any]], owner: str) -> set[str]:
    ids: set[str] = set()
    for record in records:
        identifier = require_string(record, "id", owner)
        validate_id(identifier, owner)
        if identifier in ids:
            fail(f"duplicate {owner} id: {identifier}")
        ids.add(identifier)
    return ids


def require_current_artifact_path(root: Path, relpath: str, owner: str) -> None:
    if not relpath.startswith("tmp/"):
        fail(f"{owner} current artifact must be under tmp/: {relpath}")
    path = root / relpath
    if not path.exists():
        fail(f"{owner} current artifact path missing: {relpath}")
    if path.is_file():
        if path.suffix != ".json":
            fail(f"{owner} current artifact file must be JSON: {relpath}")
        return
    if not path.is_dir():
        fail(f"{owner} current artifact path is not file or directory: {relpath}")
    files = [child for child in path.iterdir() if child.is_file()]
    if not files:
        fail(f"{owner} current artifact directory is empty: {relpath}")
    if not any(child.suffix == ".json" for child in files):
        fail(f"{owner} current artifact directory has no JSON evidence: {relpath}")


def load_current_json_artifact(root: Path, relpath: str, owner: str) -> dict[str, Any]:
    require_current_artifact_path(root, relpath, owner)
    path = root / relpath
    if not path.is_file() or path.suffix != ".json":
        fail(f"{owner} current artifact must be a JSON file: {relpath}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"{owner} invalid current artifact JSON {relpath}: {exc}")
    if not isinstance(data, dict):
        fail(f"{owner} current artifact JSON is not an object: {relpath}")
    return data
