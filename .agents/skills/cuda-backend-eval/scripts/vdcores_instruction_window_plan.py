#!/usr/bin/env python3
"""Derive a shared-instruction window plan from VDCores capacity JSON."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from paper_baseline_viewer_export import ROOT, repo_relative, write_json


DEFAULT_INPUT = (
    ROOT
    / "tmp"
    / "cuda-backend"
    / "paper-baselines"
    / "vdcores"
    / "qwen3-8b-instruction-capacity-0a0392d2"
    / "instruction-capacity-n64.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "tmp"
    / "cuda-backend"
    / "paper-baselines"
    / "vdcores"
    / "qwen3-8b-shared-window-plan-0a0392d2"
    / "shared-window-plan.json"
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return payload


def positive_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or value <= 0:
        raise SystemExit(f"expected positive integer field {key!r}")
    return value


def instruction_windows(*, required: int, capacity: int) -> list[dict[str, Any]]:
    windows = []
    for start in range(0, required, capacity):
        end = min(start + capacity, required)
        windows.append(
            {
                "index": len(windows),
                "instruction_start": start,
                "instruction_end": end,
                "instruction_count": end - start,
                "capacity_ok": end - start <= capacity,
            },
        )
    return windows


def window_manifest(
    *,
    max_insts: int,
    max_cinsts: int,
    max_minsts: int,
) -> dict[str, Any]:
    compute_windows = instruction_windows(required=max_cinsts, capacity=max_insts)
    memory_windows = instruction_windows(required=max_minsts, capacity=max_insts)
    return {
        "manifest_kind": "per_sm_uniform_lower_bound",
        "scope": (
            "Uses observed max per-SM instruction pressure because the "
            "capacity artifact does not include a full per-SM histogram."
        ),
        "compute_instruction_windows": compute_windows,
        "memory_instruction_windows": memory_windows,
        "max_compute_window_instruction_count": max(
            window["instruction_count"] for window in compute_windows
        ),
        "max_memory_window_instruction_count": max(
            window["instruction_count"] for window in memory_windows
        ),
    }


def make_plan(payload: dict[str, Any], *, source: Path) -> dict[str, Any]:
    max_insts = positive_int(payload, "max_insts")
    max_cinsts = positive_int(payload, "max_cinsts_per_sm")
    max_minsts = positive_int(payload, "max_minsts_per_sm")
    num_sms = positive_int(payload, "num_sms")
    cinst_windows = math.ceil(max_cinsts / max_insts)
    minst_windows = math.ceil(max_minsts / max_insts)
    worst_windows = max(cinst_windows, minst_windows)
    overflow_cinst_sms = payload.get("overflow_cinst_sms", [])
    overflow_minst_sms = payload.get("overflow_minst_sms", [])
    if not isinstance(overflow_cinst_sms, list):
        raise SystemExit("expected overflow_cinst_sms list")
    if not isinstance(overflow_minst_sms, list):
        raise SystemExit("expected overflow_minst_sms list")
    return {
        "schema_version": 1,
        "mode": "vdcores_qwen3_8b_shared_instruction_window_plan",
        "source_artifact": repo_relative(source),
        "status": "analysis_only",
        "model": "Qwen/Qwen3-8B",
        "serving_workload_id": "vdcores_offline_decode",
        "hardware": {
            "gpu": "H200",
            "machine": "bizhaoh200",
            "compute_target": "compute_90",
        },
        "shared_instruction_capacity": {
            "instructions_per_sm": max_insts,
            "num_sms": num_sms,
        },
        "observed_instruction_pressure": {
            "max_cinsts_per_sm": max_cinsts,
            "max_minsts_per_sm": max_minsts,
            "overflow_cinst_sm_count": len(overflow_cinst_sms),
            "overflow_minst_sm_count": len(overflow_minst_sms),
        },
        "minimum_window_lower_bound": {
            "compute_instruction_windows": cinst_windows,
            "memory_instruction_windows": minst_windows,
            "worst_case_windows_per_sm": worst_windows,
        },
        "segmented_window_manifest": window_manifest(
            max_insts=max_insts,
            max_cinsts=max_cinsts,
            max_minsts=max_minsts,
        ),
        "required_runtime_change": {
            "preferred_path": "segmented_token_windowed_shared_instruction_schedule",
            "why": (
                "The global-instruction variant removes the capacity assertion "
                "but fails correctness. A paper row needs the shared-instruction "
                "runtime to execute the Qwen3-8B decode64 schedule in at least "
                f"{worst_windows} instruction windows per SM, with preserved "
                "task dependencies and correctness checks."
            ),
            "builder_requirements": [
                "split the Qwen3-8B decode64 schedule into instruction windows",
                "emit per-window instruction tables no larger than max_insts",
                "preserve token, KV-cache, and stage dependency order",
                "record window metadata in the raw benchmark artifact",
            ],
            "runtime_requirements": [
                "reload or advance shared instruction windows without a new model load",
                "keep resident tensors and scheduler state live across windows",
                "report correctness and per-window timing before viewer import",
            ],
            "pre_import_checks": [
                "every emitted compute and memory window is <= max_insts",
                "all windows execute under one model residency and KV-cache owner",
                "window dependency handoff preserves ready queues and token state",
                "raw artifact records per-window timing and correctness status",
            ],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = make_plan(load_json(args.input), source=args.input)
    write_json(args.output, plan)
    print(repo_relative(args.output))


if __name__ == "__main__":
    main()
