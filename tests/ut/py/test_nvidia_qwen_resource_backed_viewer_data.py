import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def load_viewer_results() -> list[dict]:
    path = (
        ROOT
        / "docs"
        / "nvidia-backend"
        / "benchmark-viewer"
        / "data"
        / "results"
    )
    index = json.loads((path / "index.json").read_text(encoding="utf-8"))
    record_files = index.get("record_files")
    if record_files is None:
        record_files = json.loads(
            (path / index["record_files_path"]).read_text(encoding="utf-8")
        )
    return [
        json.loads((path / relpath).read_text(encoding="utf-8"))
        for relpath in record_files
    ]


def resource_backed_rows() -> list[dict]:
    return [
        record
        for record in load_viewer_results()
        if record["statistic"].get("serving_coverage")
        == "diagnostic_resource_backed_qwen_dag"
    ]


def test_viewer_results_include_resource_backed_diagnostic_rows():
    rows = resource_backed_rows()

    assert {row["statistic"]["workload_id"] for row in rows} == {
        "mpk_offline_decode",
        "vdcores_offline_decode",
    }
    assert any(row["statistic"].get("repeat_runs") == 3 for row in rows)
    assert all(
        row["statistic"].get(
            "last_completed_count",
            row["statistic"]["completed_count"],
        )
        == 255
        for row in rows
    )
    assert all(
        row["statistic"]["completed_count"] in {255, 510, 765, 16320, 261120}
        for row in rows
    )
    assert any(
        row["statistic"].get("execution_mode") == "bounded_decode_steps"
        and row["statistic"].get("executed_decode_steps") == 2
        and row["statistic"]["completed_count"] == 510
        for row in rows
    )
    assert any(
        row["statistic"].get("decode_feedback_status")
        == "diagnostic_token_feedback_applied"
        and row["statistic"].get("decode_feedback_applied_step_count") == 2
        for row in rows
    )
    assert any(
        row["statistic"].get("decode_feedback_status")
        == "device_token_feedback_observed"
        and row["statistic"].get("decode_feedback_applied_step_count") == 2
        for row in rows
    )
    long_decode_rows = [
        row["statistic"]
        for row in rows
        if row["statistic"].get("executed_decode_steps") == 64
    ]
    assert {row["workload_id"] for row in long_decode_rows} == {
        "mpk_offline_decode",
        "vdcores_offline_decode",
    }
    assert all(
        row.get("logits_check_policy") == "final_step"
        and row.get("logits_checked_step_count") == 1
        and row.get("logits_deferred_step_count") == 63
        for row in long_decode_rows
    )
    full_mpk_rows = [
        row["statistic"]
        for row in rows
        if row["statistic"].get("workload_id") == "mpk_offline_decode"
        and row["statistic"].get("executed_decode_steps") == 1024
    ]
    assert len(full_mpk_rows) == 1
    assert full_mpk_rows[0]["completed_count"] == 261120
    assert full_mpk_rows[0]["logits_deferred_step_count"] == 1023
    assert full_mpk_rows[0]["serving_coverage"] == (
        "diagnostic_resource_backed_qwen_dag"
    )
    assert full_mpk_rows[0]["decode_feedback_status"] == (
        "device_token_feedback_observed"
    )
    unit_numeric_rows = [
        row["statistic"]
        for row in rows
        if row["statistic"].get("numeric_task_mode") == "unit_math"
    ]
    rmsnorm_scale_rows = [
        row
        for row in unit_numeric_rows
        if row.get("numeric_task_scope")
        == "resource_backed_unit_math_weighted_elementwise_branches"
    ]
    assert len(rmsnorm_scale_rows) == 1
    assert rmsnorm_scale_rows[0]["workload_id"] == "vdcores_offline_decode"
    assert rmsnorm_scale_rows[0]["numeric_ready_callable_count"] == 8
    assert rmsnorm_scale_rows[0]["external_scale_contract_count"] == 1
    assert rmsnorm_scale_rows[0]["weighted_elementwise_callable_count"] == 5
    assert all(row["statistic"]["error_count"] == 0 for row in rows)
    assert any(row["correctness"] == "pass" for row in rows)
    assert any(
        row["correctness"] == "pass"
        and row["statistic"].get("logits_coverage") == "full_logits_buffer_checked"
        and row["statistic"].get("diagnostic_logits_reference_checked_count")
        == row["statistic"].get("logits_buffer_element_count")
        for row in rows
    )
    assert any(
        row["correctness"] == "fail"
        and row["statistic"].get("diagnostic_logits_reference_status") == "fail"
        for row in rows
    )
    assert all(
        row["statistic"].get("logits_coverage")
        in {
            "partial_logits_not_full_vocab",
            "full_logits_buffer_prefix_sampled",
            "full_logits_buffer_checked",
            None,
        }
        for row in rows
    )
