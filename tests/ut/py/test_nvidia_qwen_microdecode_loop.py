import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
VIEWER_ROOT = ROOT / "docs" / "nvidia-backend" / "benchmark-viewer"


def load_microdecode_module():
    script_path = ROOT / "examples" / "cuda" / "qwen_persistent_microdecode_live.py"
    spec = importlib.util.spec_from_file_location(
        "qwen_persistent_microdecode_live",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_plan_tracks_repeated_decode_loop():
    module = load_microdecode_module()

    plan = module.build_live_microdecode_plan(repeat_runs=3)

    assert plan["decode_loop"] == {
        "repeat_runs": 3,
        "planned_task_executions": 9,
        "prepared_callable_reuse": "single_prepare_multiple_run_prepared",
        "reset_between_runs": ["fanin", "ready_flags", "completion_flags", "counters"],
        "carried_between_runs": ["key_cache_mutable", "value_cache_mutable"],
    }
    assert len(plan["decode_iterations"]) == 3
    assert plan["decode_iterations"][0]["expected"]["logits_out"] == [
        14.5,
        18.5,
        22.5,
        26.5,
    ]
    assert plan["decode_iterations"][1]["expected"]["logits_out"] == [
        37.5,
        45.5,
        53.5,
        61.5,
    ]
    assert plan["decode_iterations"][2]["expected"]["logits_out"] == [
        60.5,
        72.5,
        84.5,
        96.5,
    ]
    assert plan["expected"] == plan["decode_iterations"][-1]["expected"]
    assert "controlled_proxy_live_decode_loop_plan" in plan["implemented_contracts"]


def test_viewer_matrix_tracks_decode_loop_evidence():
    matrix_path = VIEWER_ROOT / "data" / "paper_evaluation_matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))[
        "paper_evaluation_matrix"
    ]
    claim = next(
        item
        for item in matrix
        if item["id"] == "llm_serving_paper_baselines"
    )

    assert any(
        ref.get("kind") == "raw_artifact"
        and ref.get("path")
        == (
            "tmp/cuda-backend/pto-serving-decode-loop-live-2026-06-01/"
            "qwen-microdecode-loop.json"
        )
        for ref in claim["current_evidence_refs"]
    )
    pto_gap = next(
        item
        for item in claim["missing_evidence_details"]
        if item["id"] == "pto_full_serving_qwen3_8b"
    )
    assert (
        "controlled proxy live repeated decode-loop execution"
        in pto_gap["action"]
    )
