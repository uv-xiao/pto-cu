"""Shared helpers for the NVIDIA review guard."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DOC_ROOT = ROOT / "docs" / "nvidia-backend"
VIEWER_ROOT = DOC_ROOT / "benchmark-viewer"
VIEWER_DATA_ROOT = ROOT / "evaluations" / "nvidia" / "benchmark-viewer" / "data"
GOAL_ROOT = ROOT / "docs" / "in_progress" / "nvidia_backend_paper_ready"
WORKFLOW_ROOT = ROOT / ".github" / "workflows"
ARCHIVED_WORKFLOW = ROOT / "docs" / "ci" / "nvidia-manual-review.workflow.yml"


def fail(message: str) -> None:
    raise SystemExit(f"nvidia review guard failed: {message}")


def require_file(path: Path) -> None:
    if not path.is_file():
        fail(f"missing file: {path.relative_to(ROOT)}")


def logical_file(path: Path) -> Path:
    if path.is_file():
        return path
    if path.suffix == ".json" and (path.with_suffix("") / "index.json").is_file():
        return path.with_suffix("") / "index.json"
    fail(f"missing file: {path.relative_to(ROOT)}")


def logical_text(path: Path) -> str:
    if path.is_file():
        return path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".json" and path.with_suffix("").is_dir():
        return "\n".join(
            child.read_text(encoding="utf-8", errors="replace")
            for child in sorted(path.with_suffix("").rglob("*.json"))
        )
    fail(f"missing file: {path.relative_to(ROOT)}")


def check_dispatch_log_structure() -> None:
    landing = GOAL_ROOT / "dispatch_log.md"
    archive_index = GOAL_ROOT / "dispatch_log" / "index.md"
    entries_root = GOAL_ROOT / "dispatch_log" / "entries"

    require_file(landing)
    require_file(archive_index)
    if not entries_root.is_dir():
        fail(f"missing directory: {entries_root.relative_to(ROOT)}")
    landing_lines = landing.read_text(encoding="utf-8").splitlines()
    if len(landing_lines) > 120:
        fail(f"{landing.relative_to(ROOT)} has {len(landing_lines)} lines")
    index_text = archive_index.read_text(encoding="utf-8")
    entry_files = sorted(entries_root.glob("*.md"))
    if not entry_files:
        fail(f"{entries_root.relative_to(ROOT)} has no entry files")
    for path in entry_files:
        rel = f"entries/{path.name}"
        if rel not in index_text:
            fail(f"{archive_index.relative_to(ROOT)} missing {rel}")
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        if line_count > 300:
            fail(f"{path.relative_to(ROOT)} has {line_count} lines")


def load_json(path: Path) -> dict:
    if path.is_dir():
        return load_sharded_json(path)
    if not path.is_file() and path.suffix == ".json" and path.with_suffix("").is_dir():
        return load_sharded_json(path.with_suffix(""))
    require_file(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {path.relative_to(ROOT)}: {exc}")


def load_sharded_json(path: Path) -> dict:
    index_path = path / "index.json"
    require_file(index_path)
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {index_path.relative_to(ROOT)}: {exc}")
    collection = index.get("collection")
    record_files = index.get("record_files")
    if record_files is None and isinstance(index.get("record_files_path"), str):
        record_files_path = path / index["record_files_path"]
        require_file(record_files_path)
        record_files = json.loads(record_files_path.read_text(encoding="utf-8"))
    if not isinstance(collection, str) or not isinstance(record_files, list):
        fail(f"invalid sharded index: {index_path.relative_to(ROOT)}")
    payload = {
        key: value
        for key, value in index.items()
        if key not in {"collection", "record_files", "record_files_path"}
    }
    records = []
    for relpath in record_files:
        record_path = path / relpath
        require_file(record_path)
        record = json.loads(record_path.read_text(encoding="utf-8"))
        records.append(expand_record_sidecars(path, record))
    payload[collection] = records
    return payload


def expand_record_sidecars(base: Path, record: dict) -> dict:
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


def load_sidecar_list(path: Path) -> list:
    if path.is_file():
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            fail(f"sidecar is not a list: {path.relative_to(ROOT)}")
        return value
    index_path = path / "index.json"
    require_file(index_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    item_files = index.get("item_files")
    if not isinstance(item_files, list):
        fail(f"sidecar index is missing item_files: {index_path.relative_to(ROOT)}")
    values = []
    for relpath in item_files:
        if not isinstance(relpath, str):
            fail(f"invalid sidecar item: {index_path.relative_to(ROOT)}")
        item_path = path / relpath
        require_file(item_path)
        values.append(json.loads(item_path.read_text(encoding="utf-8")))
    return values


def require_text(path: Path, needles: list[str]) -> None:
    require_file(path)
    text = path.read_text(encoding="utf-8")
    for needle in needles:
        if needle not in text:
            fail(f"{path.relative_to(ROOT)} missing required text: {needle}")
