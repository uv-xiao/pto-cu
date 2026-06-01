import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
VIEWER_ROOT = ROOT / "docs" / "nvidia-backend" / "benchmark-viewer"


def load_unit_math_live_module():
    script_path = ROOT / "examples" / "cuda" / "qwen_unit_math_live.py"
    spec = importlib.util.spec_from_file_location(
        "qwen_unit_math_live",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_unit_math_live_plan_matches_oracle_contract():
    module = load_unit_math_live_module()

    plan = module.build_unit_math_live_plan()

    assert plan["kind"] == "pto_qwen_unit_math_live_execution_plan"
    assert plan["scope"] == "single_token_hidden4_reference"
    assert plan["dag"]["task_count"] == 4
    assert plan["tasks"] == [
        "qwen_rmsnorm_input",
        "qwen_attention_qkv",
        "qwen_mlp_gate_up",
        "qwen_logits",
    ]
    assert plan["expected"]["logits"] == [
        0.186944,
        -0.237688,
        0.302409,
        -0.378139,
    ]
    assert "qwen_unit_math_cuda_live_execution_plan" in plan[
        "implemented_contracts"
    ]
    assert "cuda_live_qwen_unit_math_execution" not in plan[
        "remaining_runtime_gaps"
    ]


def test_viewer_matrix_tracks_unit_math_live_evidence():
    matrix = json.loads(
        (VIEWER_ROOT / "data" / "paper_evaluation_matrix.json").read_text(
            encoding="utf-8",
        )
    )["paper_evaluation_matrix"]
    claim = next(
        item
        for item in matrix
        if item["id"] == "llm_serving_paper_baselines"
    )

    assert any(
        ref.get("kind") == "raw_artifact"
        and ref.get("path")
        == (
            "tmp/cuda-backend/pto-serving-unit-math-live-2026-06-01/"
            "qwen-unit-math-live.json"
        )
        for ref in claim["current_evidence_refs"]
    )
    pto_gap = next(
        item
        for item in claim["missing_evidence_details"]
        if item["id"] == "pto_full_serving_qwen3_8b"
    )
    assert "live CUDA coverage for RMSNorm" in pto_gap["action"]
    assert "full Qwen decode-loop execution" in pto_gap["action"]
    assert "cuda_live execution of the Qwen unit-math" not in pto_gap["action"]


def test_unit_math_live_importer_marks_result_as_diagnostic(tmp_path):
    script_path = (
        ROOT
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
        / "pto_qwen_unit_math_viewer_import.py"
    )
    spec = importlib.util.spec_from_file_location(
        "pto_qwen_unit_math_viewer_import",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    raw_payload = {
        "kind": "pto_qwen_unit_math_live_execution",
        "status": "pass",
        "scope": "single_token_hidden4_reference",
        "model_id": "Qwen/Qwen3-8B",
        "runtime": "cuda/persistent_device",
        "dag": {"task_count": 4},
        "device": {
            "name": "NVIDIA A100 80GB PCIe",
            "arch": "compute_80",
        },
        "timing_ns": {"host_wall": 7, "device_wall": 3},
        "scheduler_counters": {
            "completed_count": 4,
            "error_count": 0,
            "scheduler_processed_count": 4,
        },
        "max_abs_error": 0.0,
    }
    record = module.build_result_record(
        raw_payload,
        raw_artifact="tmp/cuda-backend/unit-math.json",
        commit="abc1234",
    )

    assert record["benchmark_id"] == "llm_serving_decode"
    assert record["method_id"] == "pto_persistent_device"
    assert record["statistic"]["kind"] == "pto_qwen_unit_math_live"
    assert record["statistic"]["serving_coverage"] == "diagnostic_unit_math"
    assert "unit math" in record["inputs"]["shape"]

    results = {"result_records": []}
    merged = module.merge_result(results, record)
    assert merged["result_records"] == [record]
