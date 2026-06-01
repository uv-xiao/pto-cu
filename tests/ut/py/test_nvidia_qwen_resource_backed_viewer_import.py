import json
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def load_importer_module():
    script_path = (
        ROOT
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
        / "pto_qwen_resource_backed_viewer_import.py"
    )
    spec = importlib.util.spec_from_file_location(
        "pto_qwen_resource_backed_viewer_import",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_resource_backed_importer_emits_diagnostic_rows():
    module = load_importer_module()
    raw_payload = {
        "resource_backed_execution": {
            "status": "pass",
            "device": {"arch": "compute_80"},
            "context_policy": "one_cuda_context_for_all_resource_owners",
            "workloads": [
                {
                    "workload_id": "mpk_offline_decode",
                    "status": "pass",
                    "run_prepared_status": 0,
                    "repeat_runs": 3,
                    "graph_task_count": 255,
                    "timing_ns": {"host_wall": 11, "device_wall": 7},
                    "scheduler_counters": {
                        "completed_count": 255,
                        "error_count": 0,
                        "scheduler_processed_count": 255,
                    },
                    "total_completed_count": 765,
                    "total_error_count": 0,
                },
                {
                    "workload_id": "vdcores_offline_decode",
                    "status": "pass",
                    "run_prepared_status": 0,
                    "graph_task_count": 255,
                    "timing_ns": {"host_wall": 13, "device_wall": 9},
                    "scheduler_counters": {
                        "completed_count": 255,
                        "error_count": 0,
                        "scheduler_processed_count": 255,
                    },
                },
            ],
        },
    }

    records = module.build_result_records(
        raw_payload,
        raw_artifact="tmp/cuda-backend/resource-backed.json",
        commit="abc1234",
    )

    assert [item["statistic"]["workload_id"] for item in records] == [
        "mpk_offline_decode",
        "vdcores_offline_decode",
    ]
    for record in records:
        assert record["benchmark_id"] == "llm_serving_decode"
        assert record["method_id"] == "pto_persistent_device"
        assert record["statistic"]["kind"] == "pto_qwen_resource_backed_execution"
        assert record["statistic"]["serving_coverage"] == (
            "diagnostic_resource_backed_qwen_dag"
        )
        assert record["statistic"]["last_completed_count"] == 255
        assert record["statistic"]["completed_count"] in {255, 765}
        assert record["statistic"]["error_count"] == 0
        assert record["statistic"]["repeat_runs"] in {1, 3}
        assert record["statistic"]["sample_count"] in {1, 3}
        assert record["correctness"] == "pass"
        assert "resource-backed diagnostic" in record["inputs"]["shape"]
        assert "prepared callable reused" in record["inputs"]["repeat_policy"]


def test_resource_backed_importer_adds_matrix_ref():
    module = load_importer_module()
    matrix = {
        "paper_evaluation_matrix": [
            {
                "id": "llm_serving_paper_baselines",
                "current_evidence_refs": [],
                "missing_evidence_details": [
                    {
                        "id": "pto_full_serving_qwen3_8b",
                        "action": (
                            "diagnostic proxy, unit-math, and descriptor-smoke "
                            "viewer_result_imports are present."
                        ),
                    }
                ],
            }
        ]
    }

    updated = module.ensure_matrix_ref(
        matrix,
        raw_artifact="tmp/cuda-backend/resource-backed-repeat.json",
    )
    claim = updated["paper_evaluation_matrix"][0]

    assert claim["current_evidence_refs"][0] == (
        {
            "kind": "viewer_result",
            "benchmark_id": "llm_serving_decode",
            "method_id": "pto_persistent_device",
            "gpu": "A100",
            "shape_contains": "Qwen/Qwen3-8B resource-backed diagnostic",
            "serving_coverage": "diagnostic_resource_backed_qwen_dag",
        }
    )
    assert claim["current_evidence_refs"][1]["path"] == (
        "tmp/cuda-backend/resource-backed-repeat.json"
    )
    assert "repeat_runs" in claim["current_evidence_refs"][1]["symbols"]
    assert "repeated resource-backed execution viewer_result_imports" in claim[
        "missing_evidence_details"
    ][0]["action"]


def test_viewer_results_include_resource_backed_diagnostic_rows():
    results = json.loads(
        (
            ROOT
            / "docs"
            / "nvidia-backend"
            / "benchmark-viewer"
            / "data"
            / "results.json"
        ).read_text(encoding="utf-8")
    )["result_records"]

    rows = [
        record
        for record in results
        if record["statistic"].get("serving_coverage")
        == "diagnostic_resource_backed_qwen_dag"
    ]

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
    assert all(row["statistic"]["completed_count"] in {255, 765} for row in rows)
    assert all(row["statistic"]["error_count"] == 0 for row in rows)
    assert all(row["correctness"] == "pass" for row in rows)
