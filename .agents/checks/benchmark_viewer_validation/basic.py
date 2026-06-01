from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import *  # noqa: F403
from .evidence import *  # noqa: F403


def validate_benchmarks(data: dict[str, Any], root: Path) -> set[str]:
    records = require_list(data, "benchmarks", "benchmarks")
    benchmark_ids = check_unique_ids(records, "benchmark")
    for record in records:
        owner = f"benchmark {record['id']}"
        for key in ("title", "description", "math", "code"):
            require_string(record, key, owner)
        run = require_dict(record, "run", owner)
        require_string(run, "command", owner)
        inputs = require_dict(run, "inputs", owner)
        for key in ("shape", "dtype", "repeat_policy"):
            require_string(inputs, key, owner)
        check_evidence_refs(record, owner, root)
    return benchmark_ids


def validate_methods(data: dict[str, Any], root: Path) -> set[str]:
    records = require_list(data, "methods", "methods")
    method_ids = check_unique_ids(records, "method")
    allowed_categories = {
        "pto_runtime",
        "vendor_baseline",
        "generated_kernel_baseline",
        "framework_baseline",
        "paper_baseline",
        "diagnostic_baseline",
    }
    for record in records:
        owner = f"method {record['id']}"
        for key in ("name", "runtime_flow", "lifecycle", "launch_model"):
            require_string(record, key, owner)
        category = require_string(record, "category", owner)
        if category not in allowed_categories:
            fail(f"{owner} has invalid category: {category}")
        check_evidence_refs(record, owner, root)
    return method_ids


def validate_paper_baselines(data: dict[str, Any]) -> set[str]:
    records = require_list(data, "paper_baselines", "paper_baselines")
    baseline_ids = check_unique_ids(records, "paper baseline")
    for record in records:
        owner = f"paper baseline {record['id']}"
        for key in ("name", "paper_role", "status", "next_action"):
            require_string(record, key, owner)
        source = require_dict(record, "source", owner)
        for key in ("upstream_url", "local_tmp_path", "commit"):
            require_string(source, key, owner)
        if len(source["commit"]) != 40:
            fail(f"{owner} source commit is not pinned: {source['commit']}")
        require_list(record, "paper_baselines_to_reproduce", owner)
    return baseline_ids

