import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def load_viewer_results() -> list[dict]:
    data_root = (
        ROOT
        / "evaluations"
        / "nvidia"
        / "benchmark-viewer"
        / "data"
    )
    json_path = data_root / "results.json"
    if json_path.is_file():
        return json.loads(json_path.read_text(encoding="utf-8"))["result_records"]
    path = data_root / "results"
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
    bounded_projection_rows = [
        row
        for row in rows
        if row["raw_artifact"].startswith(
            "tmp/cuda-backend/qwen-projection-bounded/",
        )
    ]
    first_layer_logits_rows = [
        row
        for row in rows
        if row["raw_artifact"].startswith(
            "tmp/cuda-backend/qwen-full-task-coverage/",
        )
    ]

    assert {row["statistic"]["workload_id"] for row in rows} == {
        "mpk_offline_decode",
        "vdcores_offline_decode",
    }
    assert len(bounded_projection_rows) == 2
    assert len(first_layer_logits_rows) == 2
    assert all(
        row["statistic"]["completed_count"] == 255
        for row in bounded_projection_rows
    )
    assert all(
        row["statistic"].get("task_selection") == "first_layer_with_logits"
        and row["statistic"].get("task_coverage_count") == 10
        and row["statistic"].get("task_func_id_sequence") == list(range(7100, 7110))
        and row["statistic"].get("last_completed_count") == 10
        and row["statistic"].get("logits_checked_step_count") == 1
        and row["statistic"].get("decode_feedback_status")
        == "device_token_feedback_observed"
        and row["correctness"] == "pass"
        for row in first_layer_logits_rows
    )
    assert all(
        row["statistic"].get("repeat_runs") == 1
        and row["statistic"].get("execution_mode") == "repeat_submissions"
        for row in bounded_projection_rows
    )
    assert all(row["statistic"]["error_count"] == 0 for row in bounded_projection_rows)
    assert all(
        row["correctness"] == "pass"
        and row["statistic"].get("logits_coverage")
        == "partial_logits_not_full_vocab"
        and row["statistic"].get("diagnostic_logits_reference_status") == "pass"
        and row["statistic"].get("diagnostic_logits_reference_checked_count") > 0
        for row in bounded_projection_rows
    )
