import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CHECKS_DIR = ROOT / ".agents" / "checks"


def load_results_module():
    sys.path.insert(0, str(CHECKS_DIR))
    try:
        return importlib.import_module("benchmark_viewer_validation.results")
    finally:
        sys.modules.pop("benchmark_viewer_validation.results", None)
        sys.path.remove(str(CHECKS_DIR))


def pto_full_serving_record():
    return {
        "benchmark_id": "llm_serving_decode",
        "method_id": "pto_persistent_device",
        "commit": "abc1234",
        "hardware": {
            "gpu": "A100",
            "machine": "local",
            "compute_target": "compute_80",
            "driver": "535.154.05",
            "cuda_toolkit": "12.4",
        },
        "inputs": {
            "shape": (
                "mpk_offline_decode,Qwen/Qwen3-8B,batch=8,"
                "prompt_tokens=128,decode_tokens=64"
            ),
            "dtype": "bfloat16",
            "repeat_policy": "three full-serving samples",
        },
        "statistic": {
            "kind": "pto_qwen_full_serving_capture",
            "serving_coverage": "full_serving",
            "workload_id": "mpk_offline_decode",
            "sample_count": 3,
            "host_wall_ns": 100,
            "device_wall_ns": 90,
            "end_to_end_latency_ns": 100,
            "time_to_first_token_ns": 1,
            "inter_token_latency_ns": 2,
            "throughput_tokens_per_s": 1000.0,
            "batch_size": 8,
            "prompt_tokens": 128,
            "decode_tokens": 64,
            "completed_requests": 8,
            "failed_requests": 0,
            "total_input_tokens": 1024,
            "total_output_tokens": 512,
            "correctness_scope": "full_qwen_numerical_correctness",
            "checked_token_count": 512,
            "max_abs_error": 0.0001,
            "correctness_tolerance": 0.001,
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
        "raw_artifact": "tmp/cuda-backend/pto-full-serving-valid/",
    }


def results_payload(record):
    return {
        "snapshot": {
            "commit": "abc1234",
            "full_capture": {
                "samples": 1,
                "artifact_root": "tmp/cuda-backend/full-capture/",
            },
            "compact_capture": {
                "samples": 1,
                "artifact_root": "tmp/cuda-backend/compact-capture/",
            },
        },
        "headline_results": [{"id": "placeholder"}],
        "selected_rows": [{"id": "placeholder"}],
        "result_records": [record],
    }


def prepare_artifacts(root):
    for relative in (
        "tmp/cuda-backend/full-capture/",
        "tmp/cuda-backend/compact-capture/",
        "tmp/cuda-backend/pto-full-serving-valid/",
    ):
        artifact_dir = root / relative
        artifact_dir.mkdir(parents=True)
        (artifact_dir / "artifact.json").write_text("{}", encoding="utf-8")


def validate_record(record, tmp_path):
    results = load_results_module()
    prepare_artifacts(tmp_path)
    results.validate_results(
        results_payload(record),
        {"llm_serving_decode"},
        {"pto_persistent_device"},
        tmp_path,
    )


def test_viewer_data_validator_accepts_complete_pto_full_serving_row(tmp_path):
    validate_record(pto_full_serving_record(), tmp_path)


def test_viewer_data_validator_rejects_failed_pto_full_serving_row(tmp_path):
    record = pto_full_serving_record()
    record["statistic"]["failed_requests"] = 1

    try:
        validate_record(record, tmp_path)
    except SystemExit as exc:
        assert "failed_requests must be zero" in str(exc)
    else:
        raise AssertionError("failed PTO full-serving row was accepted")


def test_viewer_data_validator_rejects_underchecked_pto_full_serving_row(tmp_path):
    record = pto_full_serving_record()
    record["statistic"]["checked_token_count"] = 511
    record["correctness_details"]["checked_token_count"] = 511

    try:
        validate_record(record, tmp_path)
    except SystemExit as exc:
        assert "checked_token_count must cover generated tokens" in str(exc)
    else:
        raise AssertionError("underchecked PTO full-serving row was accepted")
