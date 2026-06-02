import importlib
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = ROOT / ".agents" / "skills" / "cuda-backend-eval" / "scripts"


def load_importer_module():
    script_path = SCRIPT_DIR / "pto_qwen_full_serving_viewer_import.py"
    spec = importlib.util.spec_from_file_location(
        "pto_qwen_full_serving_viewer_import",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_claim_status_module():
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        return importlib.import_module("paper_readiness_audit_impl.claim_status")
    finally:
        sys.modules.pop("paper_readiness_audit_impl.claim_status", None)
        sys.path.remove(str(SCRIPT_DIR))


def full_serving_raw_result(workload_id):
    return {
        "workload_id": workload_id,
        "runtime": "cuda/persistent_device",
        "serving_coverage": "full_serving",
        "hardware": {
            "gpu": "A100",
            "machine": "hina",
            "compute_target": "compute_80",
            "driver": "535.154.05",
            "cuda_toolkit": "12.4",
        },
        "inputs": {
            "batch_size": 8,
            "prompt_tokens": 128,
            "decode_tokens": 64,
            "dtype": "bfloat16",
            "repeat_policy": "three PTO full-serving decode samples",
        },
        "metrics": {
            "sample_count": 3,
            "host_wall_ns": 6_000_000_000,
            "device_wall_ns": 5_900_000_000,
            "end_to_end_latency_ns": 6_000_000_000,
            "time_to_first_token_ns": 50_000_000,
            "inter_token_latency_ns": 5_800_000,
            "throughput_tokens_per_s": 1300.0,
            "completed_requests": 8,
            "failed_requests": 0,
            "max_concurrent_requests": 8,
            "total_input_tokens": 1024,
            "total_output_tokens": 512,
        },
        "correctness": "pass",
        "correctness_details": {
            "scope": "full_qwen_numerical_correctness",
            "model_id": "Qwen/Qwen3-8B",
            "status": "pass",
            "token_match": True,
            "checked_token_count": 512,
            "max_abs_error": 0.0001,
            "tolerance": 0.001,
        },
    }


def test_full_serving_importer_builds_audit_acceptable_rows():
    importer = load_importer_module()
    claim_status = load_claim_status_module()
    records = importer.build_result_records(
        {
            "results": [
                full_serving_raw_result("mpk_offline_decode"),
                full_serving_raw_result("vdcores_offline_decode"),
            ]
        },
        raw_artifact="tmp/cuda-backend/pto-full-serving/run.json",
        commit="abc1234",
    )

    assert [record["statistic"]["workload_id"] for record in records] == [
        "mpk_offline_decode",
        "vdcores_offline_decode",
    ]
    for record in records:
        assert record["benchmark_id"] == "llm_serving_decode"
        assert record["method_id"] == "pto_persistent_device"
        assert record["correctness"] == "pass"
        assert record["statistic"]["kind"] == "pto_qwen_full_serving_capture"
        assert record["statistic"]["serving_coverage"] == "full_serving"
        assert (
            record["statistic"]["correctness_scope"]
            == "full_qwen_numerical_correctness"
        )
        assert record["statistic"]["checked_token_count"] == 512
        assert record["statistic"]["max_abs_error"] == 0.0001
        assert record["statistic"]["correctness_tolerance"] == 0.001
        assert record["correctness_details"]["token_match"] is True
        assert "Qwen/Qwen3-8B" in record["inputs"]["shape"]
        assert record["raw_artifact"].startswith("tmp/")

    current_results = claim_status.result_index({"result_records": records})
    _counts, missing = claim_status.count_evidence_refs(
        [
            {
                "kind": "viewer_result",
                "benchmark_id": "llm_serving_decode",
                "method_id": "pto_persistent_device",
                "gpu": "A100",
                "shape_contains": "Qwen/Qwen3-8B",
                "serving_coverage": "full_serving",
                "required_workload_ids": [
                    "mpk_offline_decode",
                    "vdcores_offline_decode",
                ],
            }
        ],
        current_results,
    )
    assert missing == []


def test_full_serving_importer_rejects_missing_policy_row():
    importer = load_importer_module()

    try:
        importer.build_result_records(
            {"results": [full_serving_raw_result("mpk_offline_decode")]},
            raw_artifact="tmp/cuda-backend/pto-full-serving/run.json",
            commit="abc1234",
        )
    except SystemExit as exc:
        assert "missing required workloads" in str(exc)
    else:
        raise AssertionError("missing VDCores policy row was accepted")


def test_full_serving_importer_rejects_missing_correctness_details():
    importer = load_importer_module()
    row = full_serving_raw_result("mpk_offline_decode")
    row.pop("correctness_details")

    try:
        importer.build_result_records(
            {
                "results": [
                    row,
                    full_serving_raw_result("vdcores_offline_decode"),
                ]
            },
            raw_artifact="tmp/cuda-backend/pto-full-serving/run.json",
            commit="abc1234",
        )
    except SystemExit as exc:
        assert "missing correctness_details" in str(exc)
    else:
        raise AssertionError("missing correctness_details was accepted")


def test_full_serving_importer_rejects_exceeded_correctness_tolerance():
    importer = load_importer_module()
    row = full_serving_raw_result("mpk_offline_decode")
    row["correctness_details"]["max_abs_error"] = 0.01

    try:
        importer.build_result_records(
            {
                "results": [
                    row,
                    full_serving_raw_result("vdcores_offline_decode"),
                ]
            },
            raw_artifact="tmp/cuda-backend/pto-full-serving/run.json",
            commit="abc1234",
        )
    except SystemExit as exc:
        assert "exceeds correctness_details.tolerance" in str(exc)
    else:
        raise AssertionError("exceeded correctness tolerance was accepted")


def test_full_serving_importer_rejects_diagnostic_resource_backed_rows():
    importer = load_importer_module()
    row = full_serving_raw_result("mpk_offline_decode")
    row["serving_coverage"] = "diagnostic_resource_backed_qwen_dag"

    try:
        importer.build_result_records(
            {
                "results": [
                    row,
                    full_serving_raw_result("vdcores_offline_decode"),
                ]
            },
            raw_artifact="tmp/cuda-backend/pto-full-serving/run.json",
            commit="abc1234",
        )
    except SystemExit as exc:
        assert "serving_coverage=full_serving" in str(exc)
    else:
        raise AssertionError("diagnostic resource-backed row was accepted")


def test_full_serving_importer_rejects_failed_requests():
    importer = load_importer_module()
    row = full_serving_raw_result("mpk_offline_decode")
    row["metrics"]["failed_requests"] = 1

    try:
        importer.build_result_records(
            {
                "results": [
                    row,
                    full_serving_raw_result("vdcores_offline_decode"),
                ]
            },
            raw_artifact="tmp/cuda-backend/pto-full-serving/run.json",
            commit="abc1234",
        )
    except SystemExit as exc:
        assert "failed_requests must be zero" in str(exc)
    else:
        raise AssertionError("failed full-serving request row was accepted")


def test_full_serving_importer_rejects_underchecked_decode_tokens():
    importer = load_importer_module()
    row = full_serving_raw_result("mpk_offline_decode")
    row["correctness_details"]["checked_token_count"] = 511

    try:
        importer.build_result_records(
            {
                "results": [
                    row,
                    full_serving_raw_result("vdcores_offline_decode"),
                ]
            },
            raw_artifact="tmp/cuda-backend/pto-full-serving/run.json",
            commit="abc1234",
        )
    except SystemExit as exc:
        assert "checked_token_count must cover generated tokens" in str(exc)
    else:
        raise AssertionError("underchecked full-serving row was accepted")


def test_full_serving_importer_merges_viewer_results(tmp_path):
    importer = load_importer_module()
    raw_json = tmp_path / "tmp" / "cuda-backend" / "pto-full-serving" / "raw.json"
    raw_json.parent.mkdir(parents=True)
    raw_json.write_text(
        json.dumps(
            {
                "results": [
                    full_serving_raw_result("mpk_offline_decode"),
                    full_serving_raw_result("vdcores_offline_decode"),
                ]
            }
        ),
        encoding="utf-8",
    )
    results_json = tmp_path / "results.json"
    results_json.write_text(
        json.dumps(
            {
                "snapshot": {"commit": "abc1234"},
                "headline_results": [],
                "selected_rows": [],
                "result_records": [],
            }
        ),
        encoding="utf-8",
    )
    payload = importer.load_json(raw_json)
    records = importer.build_result_records(
        payload,
        raw_artifact="tmp/cuda-backend/pto-full-serving/",
        commit="abc1234",
    )
    importer.write_viewer_json(
        results_json,
        importer.merge_results(importer.load_viewer_json(results_json), records),
    )

    updated = importer.load_viewer_json(results_json.with_suffix(""))
    assert len(updated["result_records"]) == 2
