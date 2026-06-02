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
            "repeat_policy": {
                "task_selection": "first_layer_with_logits",
                "logits_active_cols_policy": {
                    "mode": "explicit_active_cols",
                    "requested_active_cols": 4096,
                    "applied_scalar1_values": [4096],
                },
            },
            "task_coverage": {
                "task_count": 10,
                "func_id_sequence": list(range(7100, 7110)),
                "callables": [
                    "qwen_embedding_lookup",
                    "qwen_rmsnorm_input",
                    "qwen_attention_qkv",
                    "qwen_attention_qk_norm",
                    "qwen_attention_o",
                    "qwen_rmsnorm_post_attention",
                    "qwen_mlp_gate_up",
                    "qwen_mlp_down",
                    "qwen_final_norm",
                    "qwen_logits",
                ],
            },
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
                    "execution_mode": "bounded_decode_steps",
                    "planned_decode_steps": 1024,
                    "executed_decode_steps": 3,
                    "decode_step_limit": 3,
                    "logits_check_policy": "final_step",
                    "logits_check_summary": {
                        "checked_step_count": 1,
                        "deferred_step_count": 2,
                    },
                    "numeric_task_mode": {
                        "mode": "unit_math",
                        "scope": (
                            "resource_backed_unit_math_weighted_elementwise_branches"
                        ),
                        "numeric_ready_callables": [
                            "qwen_rmsnorm_input",
                            "qwen_attention_qkv",
                            "qwen_attention_qk_norm",
                            "qwen_attention_o",
                            "qwen_rmsnorm_post_attention",
                            "qwen_mlp_gate_up",
                            "qwen_mlp_down",
                            "qwen_final_norm",
                        ],
                        "external_scale_contracts": [
                            {
                                "callable": "qwen_rmsnorm_input",
                                "scale_arg": "scalar_args[1]",
                                "scale": 1.0,
                            },
                        ],
                        "full_reduction_contracts": [],
                        "weighted_elementwise_callables": [
                            "qwen_attention_qk_norm",
                            "qwen_attention_o",
                            "qwen_rmsnorm_post_attention",
                            "qwen_mlp_down",
                            "qwen_final_norm",
                        ],
                    },
                    "decode_feedback": {
                        "status": "diagnostic_token_feedback_applied",
                        "applied_step_count": 3,
                    },
                    "logits_summary": {
                        "coverage": "full_logits_buffer_prefix_sampled",
                        "written_element_count": 65536,
                        "logits_buffer_elements": 2430976,
                        "sampled_element_count": 65536,
                        "topk": [{"token_id": 17, "logit": 2.5}],
                        "diagnostic_reference": {
                            "status": "pass",
                            "checked_element_count": 65536,
                            "checked_row_count": 16,
                            "max_abs_error": 0.0,
                        },
                    },
                    "logits_summary_stable": True,
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
                    "logits_summary": {
                        "coverage": "full_logits_buffer_prefix_sampled",
                        "written_element_count": 20480,
                        "logits_buffer_elements": 759680,
                        "sampled_element_count": 20480,
                        "topk": [{"token_id": 9, "logit": 1.25}],
                        "diagnostic_reference": {
                            "status": "pass",
                            "checked_element_count": 20480,
                            "max_abs_error": 0.0,
                        },
                    },
                    "logits_summary_stable": True,
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
        assert record["statistic"]["execution_mode"] in {
            "repeat_submissions",
            "bounded_decode_steps",
        }
        assert record["statistic"]["decode_feedback_status"] in {
            "diagnostic_token_feedback_applied",
            "device_token_feedback_observed",
            "not_recorded",
        }
        if record["statistic"]["decode_feedback_status"] != "not_recorded":
            assert record["statistic"]["decode_feedback_scope"] == (
                "single_sequence_row0_greedy_argmax"
            )
        assert record["statistic"]["logits_check_policy"] in {
            "every_step",
            "final_step",
        }
        assert record["statistic"]["logits_active_cols_policy"] == {
            "mode": "explicit_active_cols",
            "requested_active_cols": 4096,
            "applied_scalar1_values": [4096],
        }
        if record["statistic"]["workload_id"] == "mpk_offline_decode":
            assert record["statistic"]["logits_check_policy"] == "final_step"
            assert record["statistic"]["logits_checked_step_count"] == 1
            assert record["statistic"]["logits_deferred_step_count"] == 2
            assert record["statistic"]["numeric_task_mode"] == "unit_math"
            assert record["statistic"]["numeric_ready_callable_count"] == 8
            assert record["statistic"]["external_scale_contract_count"] == 1
            assert record["statistic"]["full_reduction_contract_count"] == 0
            assert record["statistic"]["weighted_elementwise_callable_count"] == 5
        assert record["statistic"]["task_selection"] == "first_layer_with_logits"
        assert record["statistic"]["task_coverage_count"] == 10
        assert record["statistic"]["task_func_id_sequence"] == list(range(7100, 7110))
        assert record["statistic"]["task_callable_sequence"][-2:] == [
            "qwen_final_norm",
            "qwen_logits",
        ]
        assert record["correctness"] == "pass"
        assert "resource-backed diagnostic" in record["inputs"]["shape"]
        assert "prepared callable reused" in record["inputs"]["repeat_policy"]
        assert record["statistic"]["logits_coverage"] == (
            "full_logits_buffer_prefix_sampled"
        )
        assert record["statistic"]["logits_written_element_count"] > 0
        assert record["statistic"]["logits_buffer_element_count"] > (
            record["statistic"]["logits_written_element_count"]
        )
        assert isinstance(record["statistic"]["sampled_token_id"], int)
        assert record["statistic"]["logits_summary_stable"] is True
        assert record["statistic"]["diagnostic_logits_reference_status"] == "pass"
        assert (
            record["statistic"]["diagnostic_logits_reference_checked_count"] > 0
        )
        if record["statistic"]["workload_id"] == "mpk_offline_decode":
            assert (
                record["statistic"][
                    "diagnostic_logits_reference_checked_row_count"
                ]
                == 16
            )
        assert (
            record["statistic"]["diagnostic_logits_reference_max_abs_error"] == 0.0
        )


def test_resource_backed_importer_fails_failed_diagnostic_reference():
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
                    "graph_task_count": 255,
                    "timing_ns": {"host_wall": 11, "device_wall": 7},
                    "scheduler_counters": {
                        "completed_count": 255,
                        "error_count": 0,
                        "scheduler_processed_count": 255,
                    },
                    "logits_summary": {
                        "coverage": "full_logits_buffer_prefix_sampled",
                        "written_element_count": 2430976,
                        "logits_buffer_elements": 2430976,
                        "sampled_element_count": 65536,
                        "topk": [{"token_id": 17, "logit": 2.5}],
                        "diagnostic_reference": {
                            "status": "fail",
                            "checked_element_count": 65536,
                            "max_abs_error": 0.001,
                        },
                    },
                },
            ],
        },
    }

    records = module.build_result_records(
        raw_payload,
        raw_artifact="tmp/cuda-backend/resource-backed-ref.json",
        commit="abc1234",
    )

    assert records[0]["correctness"] == "fail"
    assert (
        records[0]["statistic"]["diagnostic_logits_reference_status"] == "fail"
    )


def test_resource_backed_importer_counts_full_rmsnorm_contracts():
    module = load_importer_module()
    workload = {
        "workload_id": "mpk_offline_decode",
        "status": "pass",
        "run_prepared_status": 0,
        "graph_task_count": 255,
        "timing_ns": {"host_wall": 11, "device_wall": 7},
        "scheduler_counters": {
            "completed_count": 255,
            "error_count": 0,
            "scheduler_processed_count": 255,
        },
        "numeric_task_mode": {
            "mode": "unit_math_full_rmsnorm",
            "scope": "resource_backed_unit_math_full_rmsnorm_reduction",
            "numeric_ready_callables": ["qwen_rmsnorm_input"],
            "external_scale_contracts": [],
            "full_reduction_contracts": [
                {
                    "callable": "qwen_rmsnorm_input",
                    "shape": "hidden_size=4096",
                    "scope": "resource_backed_full_rmsnorm_reduction",
                },
            ],
            "weighted_elementwise_callables": [],
        },
        "logits_summary": {
            "coverage": "full_logits_buffer_prefix_sampled",
            "written_element_count": 1,
            "logits_buffer_elements": 2,
            "sampled_element_count": 1,
            "topk": [{"token_id": 0, "logit": 0.0}],
            "diagnostic_reference": {"status": "pass"},
        },
    }
    records = module.build_result_records(
        {
            "resource_backed_execution": {
                "device": {"arch": "compute_80"},
                "context_policy": "one_cuda_context_for_all_resource_owners",
                "workloads": [workload],
            },
        },
        raw_artifact="tmp/cuda-backend/full-rmsnorm.json",
        commit="abc1234",
    )
    statistic = records[0]["statistic"]

    assert statistic["numeric_task_mode"] == "unit_math_full_rmsnorm"
    assert statistic["external_scale_contract_count"] == 0
    assert statistic["full_reduction_contract_count"] == 1


def test_resource_backed_importer_adds_matrix_ref():
    module = load_importer_module()
    matrix = {
        "paper_evaluation_matrix": [
            {
                "id": "llm_serving_paper_baselines",
                "current_evidence_refs": [
                    {
                        "kind": "raw_artifact",
                        "path": "tmp/cuda-backend/resource-backed-repeat.json",
                        "symbols": ["repeat_runs"],
                    },
                ],
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
    assert "partial_logits_not_full_vocab" in claim[
        "current_evidence_refs"
    ][1]["symbols"]
    assert "diagnostic_qwen_tiled_vocab_projection" in claim[
        "current_evidence_refs"
    ][1]["symbols"]
    assert "final_step_logits_check_policy" in claim[
        "current_evidence_refs"
    ][1]["symbols"]
    updated_every_step = module.ensure_matrix_ref(
        matrix,
        raw_artifact="tmp/cuda-backend/resource-backed-every-step.json",
        logits_check_policy="every_step",
    )
    every_step_symbols = next(
        item["symbols"]
        for item in updated_every_step["paper_evaluation_matrix"][0][
            "current_evidence_refs"
        ]
        if item.get("path") == "tmp/cuda-backend/resource-backed-every-step.json"
    )
    assert "every_step_logits_check_policy" in every_step_symbols
    assert "final_step_logits_check_policy" not in every_step_symbols
    assert "qwen_resource_backed_full_rmsnorm_reduction" in claim[
        "current_evidence_refs"
    ][1]["symbols"]
    assert (
        len(
            [
                item
                for item in claim["current_evidence_refs"]
                if item.get("path") == "tmp/cuda-backend/resource-backed-repeat.json"
            ]
        )
        == 1
    )
    assert "repeated resource-backed execution viewer_result_imports" in claim[
        "missing_evidence_details"
    ][0]["action"]


def test_resource_backed_matrix_symbols_follow_execution_payload():
    module = load_importer_module()
    execution = {
        "status": "pass",
        "repeat_policy": {"logits_check_policy": "final_step"},
        "decode_step_execution": {"status": "bounded_decode_steps_executed"},
        "workloads": [
            {
                "numeric_task_mode": {
                    "mode": "unit_math",
                    "external_scale_contracts": [{}],
                    "weighted_elementwise_callables": ["qwen_logits"],
                },
                "decode_feedback": {
                    "status": "device_token_feedback_observed",
                },
                "logits_summary": {
                    "coverage": "full_logits_buffer_checked",
                    "diagnostic_reference": {
                        "scope": "diagnostic_qwen_tiled_vocab_projection",
                    },
                },
                "logits_summary_stable": True,
            },
        ],
    }

    symbols = set(module.raw_symbols_from_execution(execution))

    assert {
        "full_logits_buffer_checked",
        "diagnostic_qwen_tiled_vocab_projection",
        "qwen_device_decode_token_feedback",
        "qwen_resource_backed_external_rmsnorm_scale",
    } <= symbols
    assert {
        "partial_logits_not_full_vocab",
        "full_logits_buffer_prefix_sampled",
        "qwen_resource_backed_full_rmsnorm_reduction",
    }.isdisjoint(symbols)
