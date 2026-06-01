from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .constants import COLLECTION_KEYS
from .naming import record_filename
from .sidecars import split_record_sidecars


def write_json(path: Path, payload: Any) -> None:
    if not isinstance(payload, dict):
        path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
        return
    target = sharded_target(path, payload)
    if target is not None:
        write_sharded_collection(target, payload)
        return
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


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
    if path.name == "paper_readiness_audit.json" and "claim_audits" in payload:
        return path.with_suffix("")
    if path.name == "paper_baseline_probes.json" and "paper_baseline_probes" in payload:
        return path.with_suffix("")
    if path.name == "capture_imports.json" and "capture_imports" in payload:
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
    record_stems = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{collection} contains a non-object record")
        filename = record_filename(collection, index, record)
        record_stems.add(Path(filename).stem)
        relpath = f"records/{filename}"
        record = split_record_sidecars(path, collection, relpath, record)
        (path / relpath).write_text(
            json.dumps(record, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        record_files.append(relpath)
    for old in records_dir.glob("*.json"):
        if f"records/{old.name}" not in record_files:
            old.unlink()
    for old in records_dir.iterdir():
        if old.is_dir() and old.name not in record_stems:
            shutil.rmtree(old)
    manifest = {key: value for key, value in payload.items() if key != collection}
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
