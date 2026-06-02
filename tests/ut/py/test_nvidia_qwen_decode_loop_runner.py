import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
VIEWER_ROOT = ROOT / "docs" / "nvidia-backend" / "benchmark-viewer"


def load_viewer_results():
    path = VIEWER_ROOT / "data" / "results"
    return load_viewer_collection(path)["result_records"]


def load_viewer_collection(path):
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


def load_decode_loop_runner_module():
    script_path = ROOT / "examples" / "cuda" / "qwen_decode_loop_runner.py"
    spec = importlib.util.spec_from_file_location(
        "qwen_decode_loop_runner",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_decode_loop_runner_tracks_cuda_live_bridge_contract():
    module = load_decode_loop_runner_module()

    runner = module.build_decode_loop_runner(mode="offline")

    bridge = runner["cuda_live_bridge_contract"]
    assert bridge["status"] == "diagnostic_bridge_ready"
    assert bridge["runtime"] == "cuda/persistent_device"
    assert bridge["source_live_artifact"] == (
        "tmp/cuda-backend/pto-serving-decode-loop-live-2026-06-01/"
        "qwen-microdecode-loop.json"
    )
    assert bridge["submission_to_live_fields"] == {
        "a": "hidden_state",
        "b": "attention_mask",
        "out": "logits_out",
        "c": "key_cache_mutable",
        "d": "value_cache_mutable",
        "tensor_args": "resident_weight_tensors",
    }
    assert bridge["decode_loop_reuse"] == {
        "prepared_callable_reuse": "single_prepare_multiple_run_prepared",
        "reset_between_runs": [
            "fanin",
            "ready_flags",
            "completion_flags",
            "counters",
        ],
        "carried_between_runs": ["key_cache_mutable", "value_cache_mutable"],
    }
    assert bridge["serving_coverage"] == "diagnostic_microdecode"
    assert bridge["remaining_gap"] == "full_qwen_decode_loop_execution"
    assert "cuda_live_resource_bridge_contract" in runner["implemented_contracts"]


def test_decode_loop_runner_attaches_unit_math_live_bridge():
    module = load_decode_loop_runner_module()

    runner = module.build_decode_loop_runner(
        mode="mock",
        unit_math_live_payload={
            "kind": "pto_qwen_unit_math_live_execution",
            "status": "pass",
            "runtime": "cuda/persistent_device",
            "decode_loop_summary": {
                "repeat_runs": 3,
                "total_completed_count": 12,
                "total_error_count": 0,
                "total_scheduler_processed_count": 12,
            },
            "max_abs_error": 0.0,
        },
    )

    bridge = runner["unit_math_live_bridge_contract"]
    assert bridge["status"] == "diagnostic_bridge_executed"
    assert bridge["runtime"] == "cuda/persistent_device"
    assert bridge["serving_coverage"] == "diagnostic_unit_math"
    assert bridge["live_summary"] == {
        "status": "pass",
        "repeat_runs": 3,
        "total_completed_count": 12,
        "total_error_count": 0,
        "max_abs_error": 0.0,
    }
    assert bridge["remaining_gap"] == "full_qwen_decode_loop_execution"
    assert "qwen_unit_math_live_bridge_contract" in runner["implemented_contracts"]


def test_decode_loop_runner_tracks_cuda_live_resource_owners(monkeypatch):
    module = load_decode_loop_runner_module()

    token_lifecycle = {
        "status": "token_pointer_table_lifecycle_ready",
        "mode": "cuda_live",
        "decode_args": {
            "workload_decode_args": [
                {
                    "workload_id": "mpk_offline_decode",
                    "scalar_fields": {"rows": 16, "cols": 1024, "inner": 64},
                    "pointer_bindings": {"a": "input_ids"},
                }
            ],
        },
    }
    kv_lifecycle = {
        "status": "kv_cache_lifecycle_ready",
        "mode": "cuda_live",
        "kv_cache_bindings": [
            {
                "workload_id": "mpk_offline_decode",
                "batch_size": 16,
                "key_cache": {"device_ptr_hex": "0x1"},
                "value_cache": {"device_ptr_hex": "0x2"},
            }
        ],
    }
    resident_lifecycle = {
        "status": "resident_weight_table_lifecycle_ready",
        "mode": "cuda_live",
        "materialized_task_count": 1,
        "bound_tensor_pointer_count": 4,
        "pointer_table": {"status": "resident_weight_pointer_table_ready"},
    }
    materialized = {
        "status": "persistent_weight_materialization_ready",
        "abi": {"task_struct": "CudaPersistentDagTask", "sizeof_bytes": 200},
        "materialized_task_descriptors": [
            {
                "id": "embedding_lookup",
                "callable": "qwen_embedding_lookup",
                "status": "ready",
                "tensor_arg_count": 1,
                "tensor_args": [
                    {
                        "arg": "tensor_args[0]",
                        "slot_id": 0,
                        "device_ptr_hex": "0x1000",
                    }
                ],
            }
        ],
        "bound_tensor_pointer_count": 4,
        "missing_pointer_count": 0,
    }

    monkeypatch.setitem(
        module.build_decode_loop_runner.__globals__,
        "build_resources",
        lambda **_kwargs: (token_lifecycle, kv_lifecycle, resident_lifecycle),
    )
    monkeypatch.setitem(
        module.build_decode_loop_runner.__globals__,
        "graph_materialization_contract",
        lambda **_kwargs: {
            "status": "resource_backed_graph_materialized",
            "runtime": "cuda/persistent_device",
            "materialization_status": materialized["status"],
            "materialized_task_count": 1,
            "bound_tensor_pointer_count": 4,
            "missing_pointer_count": 0,
            "workloads": [
                {
                    "workload_id": "mpk_offline_decode",
                    "status": "resource_backed_graph_materialized",
                    "graph_task_count": 1,
                }
            ],
            "remaining_gap": "run_prepared_resource_backed_decode_loop",
        },
    )
    runner = module.build_decode_loop_runner(
        mode="mock",
        token_cuda_live=True,
        kv_cuda_live=True,
        resident_cuda_live=True,
        submission_smoke_payload={
            "kind": "pto_qwen_submission_smoke_execution",
            "status": "pass",
            "serving_coverage": "diagnostic_qwen_descriptor_smoke",
            "func_id_sequence": list(range(7100, 7110)),
            "max_abs_error": 0.0,
            "scheduler_counters": {"completed_count": 10, "error_count": 0},
        },
    )

    assert runner["mode"] == "partial_cuda_live_submission_plan"
    assert runner["resource_lifecycle_modes"]["token_pointer_table"] == "cuda_live"
    assert runner["resource_lifecycle_modes"]["kv_cache"] == "cuda_live"
    assert runner["resource_lifecycle_modes"]["resident_weight_table"] == "cuda_live"
    assert runner["cuda_live_resource_owners"] == [
        "token_pointer_table",
        "kv_cache",
        "resident_weight_table",
    ]
    assert "cuda_live_token_pointer_table_in_runner" in runner[
        "implemented_contracts"
    ]
    assert "cuda_live_kv_cache_owner_in_runner" in runner["implemented_contracts"]
    assert "cuda_live_resident_weight_table_in_runner" in runner[
        "implemented_contracts"
    ]
    contract = runner["cuda_live_submission_descriptor_contract"]
    assert contract["status"] == "resource_backed_descriptors_ready"
    assert contract["execution_status"] == "diagnostic_descriptor_smoke_passed"
    assert contract["execution_evidence"]["completed_count"] == 10
    assert contract["remaining_gap"] == "resource_backed_full_qwen_decode_loop_execution"
    assert contract["descriptors"][0]["func_id_sequence"] == list(
        range(7100, 7110)
    )
    assert contract["descriptors"][0]["run_prepared_repetitions"] == 1024
    assert "qwen_decode_loop_submission_descriptors" in runner[
        "implemented_contracts"
    ]
    assert "qwen_decode_loop_submission_smoke_execution" in runner[
        "implemented_contracts"
    ]
    materialization = runner["resource_backed_graph_materialization"]
    assert materialization["status"] == "resource_backed_graph_materialized"
    assert materialization["workloads"][0]["graph_task_count"] == 1
    assert "qwen_resource_backed_graph_materialization" in runner[
        "implemented_contracts"
    ]


def test_viewer_matrix_tracks_decode_loop_evidence():
    matrix = load_viewer_collection(
        VIEWER_ROOT / "data" / "paper_evaluation_matrix.json"
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
        "resource-backed diagnostic execution"
        in " ".join(pto_gap.get("evidence_summary", []))
    )


def test_viewer_results_import_decode_loop_as_diagnostic_not_full_serving():
    results = load_viewer_results()

    rows = [
        record
        for record in results
        if record["benchmark_id"] == "llm_serving_decode"
        and record["method_id"] == "pto_persistent_device"
        and record["hardware"]["gpu"] == "A100"
        and record["raw_artifact"]
        == (
            "tmp/cuda-backend/pto-serving-decode-loop-live-2026-06-01/"
            "qwen-microdecode-loop.json"
        )
    ]

    assert len(rows) == 1
    row = rows[0]
    assert "Qwen/Qwen3-8B controlled proxy microdecode loop" in row["inputs"]["shape"]
    assert row["statistic"]["serving_coverage"] == "diagnostic_microdecode"
    assert row["statistic"]["repeat_runs"] == 3
    assert row["statistic"]["completed_count"] == 9
    assert row["statistic"]["error_count"] == 0
    assert row["correctness"] == "pass"


def test_preflight_does_not_promote_diagnostic_qwen_rows_to_full_serving():
    script_path = (
        ROOT
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
        / "pto_serving_preflight.py"
    )
    spec = importlib.util.spec_from_file_location(
        "pto_serving_preflight",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    full_serving = module.full_serving_qwen_rows(
        [
            {
                "benchmark_id": "llm_serving_decode",
                "method_id": "pto_persistent_device",
                "inputs": {
                    "shape": "Qwen/Qwen3-8B controlled proxy microdecode loop"
                },
                "statistic": {"serving_coverage": "diagnostic_microdecode"},
            },
            {
                "benchmark_id": "llm_serving_decode",
                "method_id": "pto_persistent_device",
                "inputs": {
                    "shape": (
                        "vdcores_offline_decode,Qwen/Qwen3-8B,batch=2,"
                        "prompt_tokens=128,decode_tokens=64"
                    )
                },
                "statistic": {
                    "serving_coverage": "full_serving",
                    "workload_id": "vdcores_offline_decode",
                    "sample_count": 3,
                    "host_wall_ns": 6_000_000_000,
                    "device_wall_ns": 5_900_000_000,
                    "end_to_end_latency_ns": 6_000_000_000,
                    "inter_token_latency_ns": 90_000_000,
                    "time_to_first_token_ns": 100_000_000,
                    "throughput_tokens_per_s": 21.3,
                    "batch_size": 2,
                    "decode_tokens": 64,
                },
                "raw_artifact": "tmp/cuda-backend/pto-full-serving/vdcores/",
                "correctness": "pass",
            },
        ]
    )

    assert len(full_serving) == 1
    assert full_serving[0]["statistic"]["serving_coverage"] == "full_serving"
