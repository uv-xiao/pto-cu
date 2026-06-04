import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def load_decode_loop_runner_module():
    script_path = ROOT / "examples" / "cuda" / "qwen_decode_loop_runner.py"
    spec = importlib.util.spec_from_file_location(
        "qwen_decode_loop_runner_single_context_test",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_resource_graph_module():
    sys.path.insert(0, str(ROOT / "examples" / "cuda"))
    script_path = (
        ROOT
        / "examples"
        / "cuda"
        / "qwen_decode_loop_runner_impl"
        / "resource_graph.py"
    )
    spec = importlib.util.spec_from_file_location(
        "qwen_resource_graph_test",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_resource_execution_policy_module():
    sys.path.insert(0, str(ROOT / "examples" / "cuda"))
    script_path = (
        ROOT
        / "examples"
        / "cuda"
        / "qwen_decode_loop_runner_impl"
        / "resource_execution_policy.py"
    )
    spec = importlib.util.spec_from_file_location(
        "qwen_resource_execution_policy_test",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeSingleContextSession:
    def __init__(self):
        self.closed = False
        self.graph_seen_before_close = False
        self.workspace = {
            "status": "activation_workspace_lifecycle_ready",
            "mode": "cuda_live",
            "pointer_table": {
                "status": "activation_workspace_pointer_table_ready",
                "pointer_count": 2,
            },
        }

    def token_lifecycle(self):
        return {
            "status": "token_pointer_table_lifecycle_ready",
            "mode": "cuda_live",
            "decode_args": {
                "workload_decode_args": [
                    {
                        "workload_id": "mpk_offline_decode",
                        "scalar_fields": {"rows": 16, "cols": 4, "inner": 64},
                        "pointer_bindings": [
                            {
                                "field": "a",
                                "buffer": "input_ids",
                                "device_ptr_hex": "0x1000",
                                "byte_count": 64,
                            },
                        ],
                    },
                ],
            },
        }

    def kv_lifecycle(self):
        return {
            "status": "kv_cache_lifecycle_ready",
            "mode": "cuda_live",
            "kv_cache_bindings": [
                {
                    "workload_id": "mpk_offline_decode",
                    "batch_size": 16,
                    "sequence_capacity_tokens": 128,
                    "key_cache": {"device_ptr_hex": "0x2000"},
                    "value_cache": {"device_ptr_hex": "0x3000"},
                },
            ],
        }

    def resident_lifecycle(self):
        return {
            "status": "resident_weight_table_lifecycle_ready",
            "mode": "cuda_live",
            "materialized_task_count": 1,
            "bound_tensor_pointer_count": 1,
            "pointer_table": {"status": "resident_weight_pointer_table_ready"},
        }

    def open_activation_workspace(self, *, plans, graph_task_count, descriptors=None):
        assert plans[0]["workload_id"] == "mpk_offline_decode"
        assert graph_task_count == 1
        assert descriptors == []
        return self.workspace

    def close(self):
        self.closed = True
        return {
            "status": "single_context_session_closed",
            "freed_pointer_count": 4,
            "freed_by_group": {
                "token_pointer_table": 1,
                "kv_cache": 1,
                "resident_weight_table": 1,
                "activation_workspace": 1,
            },
        }

    def closed_table(self, table, group):
        assert self.closed
        return {
            **table,
            "status": table["status"].replace("_ready", "_closed"),
            "freed_pointer_count": 1,
        }


def test_single_context_session_materializes_graph_before_close(monkeypatch):
    module = load_decode_loop_runner_module()
    session = FakeSingleContextSession()

    def fake_graph_materialization(**kwargs):
        session.graph_seen_before_close = not session.closed
        return {
            "status": "resource_backed_graph_materialized",
            "runtime": "cuda/persistent_device",
            "workloads": [
                {
                    "workload_id": "mpk_offline_decode",
                    "launch_packet_preflight": {
                        "status": "resource_backed_launch_packet_workspace_bound",
                    },
                },
            ],
        }

    monkeypatch.setitem(
        module.build_decode_loop_runner.__globals__,
        "open_single_context_live_session",
        lambda **_kwargs: session,
    )
    monkeypatch.setitem(
        module.build_decode_loop_runner.__globals__,
        "graph_materialization_contract",
        fake_graph_materialization,
    )

    runner = module.build_decode_loop_runner(
        mode="mock",
        single_context_live_session=True,
    )

    assert session.closed
    assert session.graph_seen_before_close
    assert runner["single_context_live_session"] == {
        "status": "single_context_launch_packet_session_ready",
        "runtime": "cuda/persistent_device",
        "context_policy": "one_cuda_context_for_all_resource_owners",
        "graph_materialized_before_close": True,
        "launch_packet_bound_before_close": True,
        "close_summary": {
            "status": "single_context_session_closed",
            "freed_pointer_count": 4,
            "freed_by_group": {
                "token_pointer_table": 1,
                "kv_cache": 1,
                "resident_weight_table": 1,
                "activation_workspace": 1,
            },
        },
    }
    assert "single_context_live_resource_session" in runner["implemented_contracts"]
    assert (
        runner["activation_workspace_lifecycle"]["pointer_table"]["status"]
        == "activation_workspace_pointer_table_closed"
    )


def test_resource_backed_smoke_runs_before_single_context_close(monkeypatch):
    module = load_decode_loop_runner_module()
    session = FakeSingleContextSession()
    execution_seen_before_close = {"value": False}
    opened_workload_ids = {}

    def fake_open_single_context_live_session(**kwargs):
        opened_workload_ids["value"] = kwargs.get("workload_ids")
        opened_workload_ids["batch_size"] = kwargs.get("batch_size")
        return session

    monkeypatch.setitem(
        module.build_decode_loop_runner.__globals__,
        "open_single_context_live_session",
        fake_open_single_context_live_session,
    )
    monkeypatch.setitem(
        module.build_decode_loop_runner.__globals__,
        "graph_materialization_contract",
        lambda **_kwargs: {
            "status": "resource_backed_graph_materialized",
            "runtime": "cuda/persistent_device",
            "workloads": [
                {
                    "workload_id": "mpk_offline_decode",
                    "launch_packet_preflight": {
                        "status": "resource_backed_launch_packet_workspace_bound",
                    },
                },
            ],
        },
    )

    def fake_resource_backed_execution(**kwargs):
        execution_seen_before_close["value"] = not session.closed
        assert kwargs["session"] is session
        assert kwargs["plans"][0]["workload_id"] == "mpk_offline_decode"
        assert kwargs["activation_workspace"] is session.workspace
        assert kwargs["repeat_runs"] == 4
        assert kwargs["decode_step_limit"] is None
        assert kwargs["max_task_count"] == 8
        assert kwargs["worker_blocks"] == 4
        return {
            "status": "pass",
            "serving_coverage": "diagnostic_resource_backed_qwen_dag",
            "workloads": [
                {
                    "workload_id": "mpk_offline_decode",
                    "status": "pass",
                    "graph_task_count": 1,
                },
            ],
        }

    monkeypatch.setitem(
        module.build_decode_loop_runner.__globals__,
        "run_resource_backed_execution",
        fake_resource_backed_execution,
    )

    runner = module.build_decode_loop_runner(
        mode="mock",
        single_context_live_session=True,
        run_resource_backed_smoke=True,
        resource_backed_workloads=["mpk_offline_decode"],
        resource_backed_batch_size=1,
        resource_backed_repeat_runs=4,
        resource_backed_max_tasks=8,
        resource_backed_worker_blocks=4,
    )

    assert session.closed
    assert opened_workload_ids["value"] == ["mpk_offline_decode"]
    assert opened_workload_ids["batch_size"] == 1
    assert execution_seen_before_close["value"]
    assert runner["resource_backed_execution"]["status"] == "pass"
    assert "qwen_resource_backed_diagnostic_execution" in runner[
        "implemented_contracts"
    ]


def test_resource_backed_decode_step_limit_runs_before_close(monkeypatch):
    module = load_decode_loop_runner_module()
    session = FakeSingleContextSession()

    monkeypatch.setitem(
        module.build_decode_loop_runner.__globals__,
        "open_single_context_live_session",
        lambda **_kwargs: session,
    )
    monkeypatch.setitem(
        module.build_decode_loop_runner.__globals__,
        "graph_materialization_contract",
        lambda **_kwargs: {
            "status": "resource_backed_graph_materialized",
            "runtime": "cuda/persistent_device",
            "workloads": [
                {
                    "workload_id": "mpk_offline_decode",
                    "launch_packet_preflight": {
                        "status": "resource_backed_launch_packet_workspace_bound",
                    },
                },
            ],
        },
    )

    def fake_resource_backed_execution(**kwargs):
        assert not session.closed
        assert kwargs["decode_step_limit"] == 2
        return {
            "status": "pass",
            "decode_step_execution": {
                "status": "policy_length_decode_steps_executed",
                "total_planned_decode_steps": 4,
                "total_executed_decode_steps": 4,
                "policy_length_complete": True,
            },
            "implemented_contracts": [
                "qwen_resource_backed_diagnostic_execution",
                "qwen_resource_backed_decode_step_execution",
                "qwen_resource_backed_policy_length_decode_execution",
            ],
            "serving_coverage": "diagnostic_resource_backed_qwen_dag",
            "workloads": [],
        }

    monkeypatch.setitem(
        module.build_decode_loop_runner.__globals__,
        "run_resource_backed_execution",
        fake_resource_backed_execution,
    )

    runner = module.build_decode_loop_runner(
        mode="mock",
        single_context_live_session=True,
        run_resource_backed_smoke=True,
        resource_backed_decode_steps=2,
    )

    assert session.closed
    assert "qwen_resource_backed_decode_step_execution" in runner[
        "implemented_contracts"
    ]
    assert "qwen_resource_backed_policy_length_decode_execution" in runner[
        "implemented_contracts"
    ]
    assert "full_cuda_live_decode_loop_execution" not in runner[
        "remaining_runtime_gaps"
    ]


def test_policy_length_decode_execution_is_explicit():
    policy = load_resource_execution_policy_module()
    workload_results = [
        {
            "workload_id": "mpk_offline_decode",
            "planned_decode_steps": 1024,
            "executed_decode_steps": 1024,
        },
        {
            "workload_id": "vdcores_offline_decode",
            "planned_decode_steps": 64,
            "executed_decode_steps": 64,
        },
    ]

    summary = policy.decode_step_execution_summary(
        workload_results,
        decode_step_limit=1024,
    )
    contracts = policy.implemented_contracts(
        1024,
        policy_length_complete=summary["policy_length_complete"],
    )

    assert summary["status"] == "policy_length_decode_steps_executed"
    assert summary["total_planned_decode_steps"] == 1088
    assert summary["total_executed_decode_steps"] == 1088
    assert summary["policy_length_complete"] is True
    assert "qwen_resource_backed_decode_step_execution" in contracts
    assert "qwen_resource_backed_policy_length_decode_execution" in contracts


def test_partial_decode_execution_remains_bounded_not_policy_length():
    policy = load_resource_execution_policy_module()

    summary = policy.decode_step_execution_summary(
        [
            {
                "workload_id": "mpk_offline_decode",
                "planned_decode_steps": 1024,
                "executed_decode_steps": 2,
            }
        ],
        decode_step_limit=2,
    )
    contracts = policy.implemented_contracts(
        2,
        policy_length_complete=summary["policy_length_complete"],
    )

    assert summary["status"] == "bounded_decode_steps_executed"
    assert summary["policy_length_complete"] is False
    assert "qwen_resource_backed_decode_step_execution" in contracts
    assert "qwen_resource_backed_policy_length_decode_execution" not in contracts


def test_resource_backed_logits_summary_marks_partial_vocab_coverage():
    module = load_resource_graph_module()

    summary = module.summarize_logits_values(
        [0.0, 3.5, -1.0, 2.0],
        logits_buffer_elements=12,
        written_element_count=4,
        top_k=2,
    )

    assert summary["status"] == "partial_logits_sampled"
    assert summary["coverage"] == "partial_logits_not_full_vocab"
    assert summary["full_buffer_sampled"] is False
    assert summary["sampled_element_count"] == 4
    assert summary["written_element_count"] == 4
    assert summary["logits_buffer_elements"] == 12
    assert summary["finite_count"] == 4
    assert summary["nonzero_count"] == 3
    assert summary["topk"] == [
        {"token_id": 1, "logit": 3.5},
        {"token_id": 3, "logit": 2.0},
    ]


def test_active_logits_written_elements_uses_diagnostic_window():
    module = load_resource_graph_module()

    class FinalTask:
        n = 16
        cols = 16
        scalar1 = 4.0

    written = module.active_logits_written_elements(
        FinalTask(),
        {"logits_buffer": {"element_count": 32}},
    )

    assert written == 4


def test_active_logits_sample_extent_preserves_row_strided_window():
    module = load_resource_graph_module()

    class FinalTask:
        n = 32
        cols = 16
        scalar1 = 4.0

    extent = module.active_logits_sample_extent(
        FinalTask(),
        {"logits_buffer": {"element_count": 32}},
    )

    assert module.active_logits_written_elements(
        FinalTask(),
        {"logits_buffer": {"element_count": 32}},
    ) == 8
    assert extent == 20


def test_resource_backed_logits_summary_marks_full_buffer_written_prefix_sampled():
    module = load_resource_graph_module()

    summary = module.summarize_logits_values(
        [0.0, 3.5, -1.0, 2.0],
        logits_buffer_elements=4,
        written_element_count=4,
        diagnostic_reference={
            "status": "pass",
            "scope": "diagnostic_qwen_tiled_vocab_projection",
            "checked_element_count": 4,
            "max_abs_error": 0.0,
        },
        top_k=1,
    )

    assert summary["coverage"] == "full_logits_buffer_checked"
    assert summary["full_buffer_sampled"] is True
    assert summary["topk"] == [{"token_id": 1, "logit": 3.5}]
    assert summary["diagnostic_reference"]["status"] == "pass"


def test_resource_backed_logits_summary_reports_row0_token_ids():
    module = load_resource_graph_module()

    summary = module.summarize_logits_values(
        [0.0, 3.5, 1.0, 2.0, 9.0, 8.0, 7.0, 6.0],
        logits_buffer_elements=8,
        written_element_count=8,
        vocab_cols=4,
        top_k=2,
    )

    assert summary["coverage"] == "full_logits_buffer_checked"
    assert summary["topk"] == [
        {"token_id": 1, "logit": 3.5},
        {"token_id": 3, "logit": 2.0},
    ]


def test_activation_finiteness_summary_reports_row_local_nonfinite_column():
    module = load_resource_graph_module()

    summary = module.summarize_activation_row_values(
        [1.0, 2.0, float("nan"), 4.0],
        row_index=1,
        row_width=4,
    )

    assert summary == {
        "row_index": 1,
        "row_width": 4,
        "sampled_element_count": 4,
        "finite_count": 3,
        "nan_count": 1,
        "posinf_count": 0,
        "neginf_count": 0,
        "nonfinite_count": 1,
        "first_nonfinite_column": 2,
        "first_nonfinite_index": 6,
        "max_abs_finite": 4.0,
        "value_sample": [1.0, 2.0, "nan", 4.0],
    }


def test_activation_finiteness_summary_reports_selected_columns():
    module = load_resource_graph_module()

    summary = module.summarize_activation_row_values(
        [10.0, 20.0, 30.0, 40.0],
        row_index=2,
        row_width=4,
        selected_columns=[3, 1, 9],
    )

    assert summary["selected_columns"] == [
        {"column": 3, "index": 11, "value": 40.0},
        {"column": 1, "index": 9, "value": 20.0},
        {"column": 9, "index": 17, "value": "out_of_sample"},
    ]


def test_activation_finiteness_summary_can_dump_row_values():
    module = load_resource_graph_module()

    summary = module.summarize_activation_row_values(
        [1.25, float("nan"), float("inf"), -2.5],
        row_index=0,
        row_width=4,
        include_values=True,
    )

    assert summary["row_values"] == [1.25, "nan", "inf", -2.5]


def test_activation_summary_indices_include_trailing_activation_task():
    module = load_resource_graph_module()

    assert module.activation_summary_task_indices(
        3,
        [
            {"id": "input_norm", "callable": "qwen_input_norm"},
            {"id": "mlp_down", "callable": "qwen_mlp_down"},
            {"id": "logits", "callable": "qwen_logits"},
        ],
    ) == [0, 1]

    assert module.activation_summary_task_indices(
        2,
        [
            {"id": "input_norm", "callable": "qwen_input_norm"},
            {"id": "mlp_down", "callable": "qwen_mlp_down"},
        ],
    ) == [0, 1]


def test_diagnostic_logits_reference_compares_tiled_vocab_projection():
    module = load_resource_graph_module()

    reference = module.diagnostic_logits_projection_values(
        hidden=[1.0, 2.0, 10.0, 20.0],
        lm_head=[1.0, 0.0, 2.0, 0.0],
        count=4,
        cols=2,
        hidden_width=2,
        hidden_stride=2,
        weight_stride=2,
    )
    comparison = module.compare_logits_reference(
        [1.0, 2.0, 10.0, 20.0],
        reference,
    )

    assert reference == [1.0, 2.0, 10.0, 20.0]
    assert comparison["status"] == "pass"
    assert comparison["scope"] == "diagnostic_qwen_tiled_vocab_projection"
    assert comparison["max_abs_error"] == 0.0


def test_diagnostic_logits_projection_checks_batch_rows():
    module = load_resource_graph_module()

    reference = module.diagnostic_logits_projection_values(
        hidden=[1.0, 2.0, 10.0, 20.0],
        lm_head=[1.0, 0.0, 2.0, 0.0],
        count=4,
        cols=2,
        hidden_width=2,
        hidden_stride=2,
        weight_stride=2,
    )
    comparison = module.compare_logits_reference(
        [1.0, 2.0, 10.0, 20.0],
        reference,
    )

    assert comparison["status"] == "pass"
    assert comparison["checked_element_count"] == 4


def test_diagnostic_logits_projection_accumulates_like_float32_kernel():
    module = load_resource_graph_module()

    reference = module.diagnostic_logits_projection_values(
        hidden=[16777216.0, 1.0],
        lm_head=[1.0, 1.0],
        count=1,
        cols=1,
        hidden_width=2,
        hidden_stride=2,
        weight_stride=2,
    )

    assert reference == [16777216.0]


def test_diagnostic_logits_projection_matches_bf16_output_boundary():
    module = load_resource_graph_module()

    reference = module.diagnostic_logits_projection_values(
        hidden=[1.0, 0.00390625],
        lm_head=[1.0, 1.0],
        count=1,
        cols=1,
        hidden_width=2,
        hidden_stride=2,
        weight_stride=2,
    )

    assert reference == [1.0]


def test_diagnostic_logits_reference_reports_mismatch_indices():
    module = load_resource_graph_module()
    values = [0.0] * 301
    values[100] = 10.0
    values[200] = 1.25
    values[300] = -3.0

    comparison = module.compare_logits_reference(
        values,
        [10.0, 1.0, -3.125],
        checked_indices=[100, 200, 300],
        tolerance=0.01,
    )

    assert comparison["status"] == "fail"
    assert comparison["mismatch_count"] == 2
    assert comparison["first_mismatch_index"] == 200
    assert comparison["first_mismatch_error"] == 0.25
    assert comparison["first_mismatch_value"] == 1.25
    assert comparison["first_mismatch_expected"] == 1.0
    assert comparison["max_error_index"] == 200
    assert comparison["max_error_allowed_error"] == 0.01


def test_diagnostic_logits_reference_samples_large_vocab_rows():
    module = load_resource_graph_module()

    indices = module.diagnostic_logits_reference_indices(
        value_count=2_430_976,
        cols=151_936,
        hidden_width=4096,
        weight_stride=4096,
        max_weight_elements=1_000_000,
        max_checked_elements=16,
    )

    assert indices == [row * 151_936 for row in range(16)]
    assert module.diagnostic_logits_reference_row_count(
        checked_indices=indices,
        cols=151_936,
    ) == 16


def test_diagnostic_logits_reference_indices_stay_inside_active_window():
    module = load_resource_graph_module()

    indices = module.diagnostic_logits_reference_indices(
        value_count=20,
        cols=16,
        active_cols=4,
        hidden_width=2,
        weight_stride=2,
        max_weight_elements=32,
        max_checked_elements=32,
    )

    assert indices == [0, 16, 1, 17, 2, 18, 3, 19]


def test_diagnostic_logits_reference_decodes_bfloat16_weights():
    module = load_resource_graph_module()

    assert module.tensor_arg_values_to_f32([0x3F80, 0x4000], dtype_code=6) == [
        1.0,
        2.0,
    ]


def test_diagnostic_logits_fallback_reference_matches_task_branch():
    module = load_resource_graph_module()

    reference = module.diagnostic_logits_fallback_values(
        hidden=[2.0, 3.0, 5.0],
        lm_head=[0.5, 0.25, 0.125, 2.0],
        indices=[0, 1, 2, 3, 4],
    )

    assert reference == [1.0, 0.75, 0.625, 4.0, 1.5]
