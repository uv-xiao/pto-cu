from __future__ import annotations

import hashlib
import json
import re
from typing import Any


def record_filename(collection: str, index: int, record: dict[str, Any]) -> str:
    if collection == "serving_command_plans" and "id" in record:
        return f"{slug(str(record['id']))}.json"
    if collection == "capture_imports":
        return f"{slug(capture_import_name(index, record))}.json"
    if "id" in record:
        return f"{record['id']}.json"
    prefix = slug(record_prefix(record))
    digest = hashlib.sha1(
        record_identity(record).encode("utf-8"),
    ).hexdigest()[:10]
    return f"{index:03d}-{prefix}-{digest}.json"


def capture_import_name(index: int, record: dict[str, Any]) -> str:
    return "-".join(
        str(value)
        for value in (
            f"{index:03d}",
            record.get("baseline", "baseline"),
            record.get("benchmark_id", "benchmark"),
            record.get("method_id", "method"),
            record.get("n", "n"),
            record.get("task_count", "tasks"),
        )
    )


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
