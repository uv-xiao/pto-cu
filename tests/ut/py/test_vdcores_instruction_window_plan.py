from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT
    / ".agents"
    / "skills"
    / "cuda-backend-eval"
    / "scripts"
    / "vdcores_instruction_window_plan.py"
)


def load_module():
    sys.path.insert(0, SCRIPT.parent.as_posix())
    spec = importlib.util.spec_from_file_location(
        "vdcores_instruction_window_plan",
        SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vdcores_instruction_window_plan_emits_bounded_manifest(tmp_path):
    module = load_module()
    payload = {
        "max_insts": 4,
        "num_sms": 2,
        "max_cinsts_per_sm": 9,
        "max_minsts_per_sm": 17,
        "overflow_cinst_sms": [0, 1],
        "overflow_minst_sms": [0, 1],
    }

    plan = module.make_plan(payload, source=tmp_path / "capacity.json")
    manifest = plan["segmented_window_manifest"]

    assert plan["minimum_window_lower_bound"] == {
        "compute_instruction_windows": 3,
        "memory_instruction_windows": 5,
        "worst_case_windows_per_sm": 5,
    }
    assert len(manifest["compute_instruction_windows"]) == 3
    assert len(manifest["memory_instruction_windows"]) == 5
    assert manifest["max_compute_window_instruction_count"] <= 4
    assert manifest["max_memory_window_instruction_count"] <= 4
    assert all(
        window["capacity_ok"]
        for window in manifest["compute_instruction_windows"]
        + manifest["memory_instruction_windows"]
    )
    assert "pre_import_checks" in plan["required_runtime_change"]


def test_vdcores_execution_attempt_records_window_manifest_status():
    import json

    data_dir = (
        ROOT
        / "docs"
        / "nvidia-backend"
        / "benchmark-viewer"
        / "data"
        / "paper_baseline_execution_attempts"
    )
    index = json.loads((data_dir / "index.json").read_text(encoding="utf-8"))
    data = {
        "paper_baseline_execution_attempts": [
            json.loads((data_dir / relpath).read_text(encoding="utf-8"))
            for relpath in index["record_files"]
        ]
    }
    attempts = {
        attempt["id"]: attempt
        for attempt in data["paper_baseline_execution_attempts"]
    }
    summary = attempts[
        "vdcores_qwen3_8b_shared_instruction_window_plan_h200"
    ]["summary"]

    assert summary["segmented_window_manifest_status"] == "generated"
    assert summary["segmented_window_manifest_kind"] == (
        "per_sm_uniform_lower_bound"
    )
    assert summary["max_compute_window_instruction_count"] == 512
    assert summary["max_memory_window_instruction_count"] == 512
    assert not summary["paper_row_importable"]
