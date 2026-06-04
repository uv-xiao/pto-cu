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


def load_plan_history_module():
    sys.path.insert(0, str(CHECKS_DIR))
    try:
        return importlib.import_module("benchmark_viewer_validation.plan_history")
    finally:
        sys.modules.pop("benchmark_viewer_validation.plan_history", None)
        sys.path.remove(str(CHECKS_DIR))


def plan_history_payload():
    return {
        "schema_version": 1,
        "generated_at": "2026-06-04",
        "latest_reviewed_commit": "abc1234",
        "summary": {
            "current_focus": "Qwen full-serving correctness",
            "recent_pattern": "Recent work is test-heavy.",
            "reflection": (
                "Too much time went to guardrails; next work must run a "
                "benchmark model farther."
            ),
            "test_strategy": (
                "Avoid sparse row-by-row tests and prefer large integrated "
                "tests for benchmark-model execution."
            ),
        },
        "work_focus": [
            {
                "window": "latest 12 commits",
                "feature_or_runtime": 2,
                "tests_or_guardrails": 3,
                "viewer_or_docs": 1,
                "notes": "Test-heavy window.",
            }
        ],
        "recent_slices": [
            {
                "commit": "abc1234",
                "title": "fix: run qwen benchmark farther",
                "focus": "feature_or_runtime",
                "reflection": "Benchmark model progress.",
            }
        ],
        "reflection_log": [
            {
                "date": "2026-06-04",
                "trigger": "before another reporting-only test",
                "finding": "Spent too much time on non-feature work.",
                "decision": "Return to benchmark-model execution.",
            }
        ],
        "next_reflection_check": {
            "cadence": "Before another artifact-specific pytest",
            "question": "Does this change make a benchmark model run farther?",
            "preferred_action_if_reporting_only": "Keep the assertion broad.",
        },
    }


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
            "comparison_scope": "model_equivalent_decode",
            "model_equivalent_ready": True,
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
            "model_equivalent_ready": True,
            "comparison_scope": "model_equivalent_decode",
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


def test_viewer_data_validator_rejects_diagnostic_comparison_scope(tmp_path):
    record = pto_full_serving_record()
    record["correctness_details"]["comparison_scope"] = (
        "diagnostic_decode_without_prompt_prefill"
    )
    record["correctness_details"]["model_equivalent_ready"] = False

    try:
        validate_record(record, tmp_path)
    except SystemExit as exc:
        assert "model_equivalent_ready" in str(exc)
    else:
        raise AssertionError("diagnostic comparison scope was accepted")


def test_plan_history_validator_requires_recent_checkout_commit():
    plan_history = load_plan_history_module()
    payload = plan_history_payload()

    try:
        plan_history.validate_plan_history(
            payload,
            allowed_latest_commits={"def5678"},
        )
    except SystemExit as exc:
        assert "latest_reviewed_commit" in str(exc)
    else:
        raise AssertionError("stale plan history commit was accepted")


def test_plan_history_validator_rejects_verbose_recent_slices():
    plan_history = load_plan_history_module()
    payload = plan_history_payload()
    payload["recent_slices"] = [
        {
            "commit": f"abc123{i}",
            "title": f"test: sparse detail {i}",
            "focus": "tests_or_guardrails",
            "reflection": "Reporting-only detail.",
        }
        for i in range(9)
    ]

    try:
        plan_history.validate_plan_history(
            payload,
            allowed_latest_commits={"abc1234"},
        )
    except SystemExit as exc:
        assert "recent_slices must stay brief" in str(exc)
    else:
        raise AssertionError("verbose plan history slices were accepted")


def test_plan_history_validator_accepts_feature_dominant_runtime_window():
    plan_history = load_plan_history_module()
    payload = plan_history_payload()
    payload["work_focus"][0]["feature_or_runtime"] = 7
    payload["work_focus"][0]["tests_or_guardrails"] = 5
    payload["work_focus"][0]["viewer_or_docs"] = 0

    plan_history.validate_plan_history(
        payload,
        allowed_latest_commits={"abc1234"},
    )
