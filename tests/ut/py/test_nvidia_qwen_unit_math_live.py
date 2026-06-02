import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
VIEWER_ROOT = ROOT / "evaluations" / "nvidia" / "benchmark-viewer" / "viewer"
VIEWER_DATA = ROOT / "evaluations" / "nvidia" / "benchmark-viewer" / "data"


def load_viewer_results():
    path = VIEWER_DATA / "results.json"
    return load_viewer_collection(path)["result_records"]


def load_viewer_collection(path):
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "collection" not in payload:
            return payload
        return payload
    if path.suffix == ".json" and path.with_suffix("").is_dir():
        path = path.with_suffix("")
    index = json.loads((path / "index.json").read_text(encoding="utf-8"))
    record_files = index.get("record_files")
    if record_files is None:
        record_files = json.loads(
            (path / index["record_files_path"]).read_text(encoding="utf-8")
        )
    records = [
        expand_viewer_record(
            path,
            json.loads((path / relpath).read_text(encoding="utf-8")),
        )
        for relpath in record_files
    ]
    return {
        **{
            key: value
            for key, value in index.items()
            if key not in {"collection", "record_files", "record_files_path"}
        },
        index["collection"]: records,
    }


def expand_viewer_record(base, record):
    payload = dict(record)
    for field in ("current_evidence_refs", "missing_evidence_details"):
        path_key = f"{field}_path"
        relpath = payload.pop(path_key, None)
        if relpath is not None:
            payload[field] = load_viewer_sidecar_list(base / relpath)
    return payload


def load_viewer_sidecar_list(path):
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    index = json.loads((path / "index.json").read_text(encoding="utf-8"))
    return [
        json.loads((path / relpath).read_text(encoding="utf-8"))
        for relpath in index["item_files"]
    ]


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

    plan = module.build_unit_math_live_plan(repeat_runs=3)

    assert plan["kind"] == "pto_qwen_unit_math_live_execution_plan"
    assert plan["scope"] == "single_token_hidden4_reference"
    assert plan["dag"]["task_count"] == 5
    assert plan["decode_loop"] == {
        "repeat_runs": 3,
        "planned_task_executions": 15,
        "prepared_callable_reuse": "single_prepare_multiple_run_prepared",
        "reset_between_runs": [
            "fanin",
            "ready_flags",
            "completion_flags",
            "counters",
            "unit_outputs",
        ],
        "carried_between_runs": [
            "hidden_state_from_previous_logits",
            "weight_buffers",
            "kv_cache_buffers",
        ],
    }
    assert plan["tasks"] == [
        "qwen_rmsnorm_input",
        "qwen_attention_qkv",
        "qwen_mlp_gate_up",
        "qwen_final_norm",
        "qwen_logits",
    ]
    assert plan["expected"]["final_norm"] == [
        0.943815,
        -0.927277,
        1.038197,
        -1.081823,
    ]
    assert plan["expected"]["logits"] == [
        3.208971,
        -4.080017,
        5.190983,
        -6.490936,
    ]
    assert len(plan["decode_iterations"]) == 3
    assert plan["decode_iterations"][0]["expected"] == plan["expected"]
    assert plan["decode_iterations"][1]["inputs"]["hidden"] == plan["expected"][
        "logits"
    ]
    assert "qwen_unit_math_cuda_live_execution_plan" in plan[
        "implemented_contracts"
    ]
    assert "qwen_unit_math_final_norm_live_execution" in plan[
        "implemented_contracts"
    ]
    assert "cuda_live_qwen_unit_math_execution" not in plan[
        "remaining_runtime_gaps"
    ]


def test_unit_math_live_graph_uses_normal_lowering():
    load_unit_math_live_module()
    from qwen_unit_math_live_impl.graph import make_graph_arrays
    from qwen_unit_math_live_impl.plan import CALLABLES

    names = (
        "hidden",
        "norm_weight",
        "q_weight",
        "k_weight",
        "v_weight",
        "gate_weight",
        "up_weight",
        "final_norm_weight",
        "lm_head",
        "rmsnorm",
        "context",
        "key_cache",
        "value_cache",
        "mlp",
        "final_norm",
        "logits",
    )
    graph = make_graph_arrays(
        {name: index + 100 for index, name in enumerate(names)}
    )

    assert list(graph["dependents"]) == [1, 2, 3, 4]
    assert list(graph["fanin"]) == [0, 1, 1, 1, 1]
    assert [int(task.func_id) for task in graph["tasks"]] == [
        func_id for _, func_id in CALLABLES
    ]
    assert [
        (
            int(task.dependent_begin),
            int(task.dependent_count),
            int(task.initial_fanin),
        )
        for task in graph["tasks"]
    ] == [(0, 1, 0), (1, 1, 1), (2, 1, 1), (3, 1, 1), (4, 0, 1)]
    assert graph["tasks"][4].scalar_arg_count == 2
    assert list(graph["tasks"][4].scalar_args)[:2] == [0.0, 4.0]


def test_viewer_matrix_tracks_unit_math_live_evidence():
    matrix = load_viewer_collection(
        VIEWER_DATA / "paper_evaluation_matrix.json"
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
            "tmp/cuda-backend/qwen-final-norm-rmsnorm/"
            "qwen-unit-math-live.json"
        )
        for ref in claim["current_evidence_refs"]
    )
    pto_gap = next(
        item
        for item in claim["missing_evidence_details"]
        if item["id"] == "pto_full_serving_qwen3_8b"
    )
    assert "diagnostics" in pto_gap["action"]
    assert "full Qwen numerical correctness" in pto_gap["action"]
    assert "full-serving row import" in pto_gap["action"]
    assert "full-serving row import" in pto_gap["action"]
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
        "dag": {"task_count": 5},
        "device": {
            "name": "NVIDIA A100 80GB PCIe",
            "arch": "compute_80",
        },
        "timing_ns": {"host_wall": 7, "device_wall": 3},
        "scheduler_counters": {
            "completed_count": 5,
            "error_count": 0,
            "scheduler_processed_count": 5,
        },
        "decode_loop_observations": [
            {"timing_ns": {"host_wall": 7, "device_wall": 3}},
            {"timing_ns": {"host_wall": 9, "device_wall": 4}},
            {"timing_ns": {"host_wall": 8, "device_wall": 5}},
        ],
        "decode_loop_summary": {
            "repeat_runs": 3,
            "total_completed_count": 15,
            "total_error_count": 0,
            "total_scheduler_processed_count": 15,
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
    assert record["statistic"]["sample_count"] == 3
    assert record["statistic"]["host_wall_ns"] == 8
    assert record["statistic"]["completed_count"] == 15
    assert "unit math" in record["inputs"]["shape"]
    assert "FinalRMSNorm" in record["inputs"]["shape"]

    results = {"result_records": []}
    merged = module.merge_result(results, record)
    assert merged["result_records"] == [record]


def test_viewer_results_import_repeated_unit_math_as_diagnostic():
    results = load_viewer_results()

    row = next(
        record
        for record in results
        if record["benchmark_id"] == "llm_serving_decode"
        and record["method_id"] == "pto_persistent_device"
        and record["raw_artifact"]
        == (
            "tmp/cuda-backend/qwen-final-norm-rmsnorm/"
            "qwen-unit-math-live.json"
        )
    )

    assert row["statistic"]["serving_coverage"] == "diagnostic_unit_math"
    assert row["statistic"]["sample_count"] == 3
    assert row["statistic"]["repeat_runs"] == 3
    assert row["statistic"]["completed_count"] == 15
    assert "reused across 3" in row["inputs"]["repeat_policy"]
    assert row["correctness"] == "pass"
