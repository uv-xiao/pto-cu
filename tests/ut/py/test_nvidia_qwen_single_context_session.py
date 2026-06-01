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

    def open_activation_workspace(self, *, plans, graph_task_count):
        assert plans[0]["workload_id"] == "mpk_offline_decode"
        assert graph_task_count == 1
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
        execution_seen_before_close["value"] = not session.closed
        assert kwargs["session"] is session
        assert kwargs["plans"][0]["workload_id"] == "mpk_offline_decode"
        assert kwargs["activation_workspace"] is session.workspace
        assert kwargs["repeat_runs"] == 4
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
        resource_backed_repeat_runs=4,
    )

    assert session.closed
    assert execution_seen_before_close["value"]
    assert runner["resource_backed_execution"]["status"] == "pass"
    assert "qwen_resource_backed_diagnostic_execution" in runner[
        "implemented_contracts"
    ]


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


def test_resource_backed_logits_summary_marks_full_buffer_written_prefix_sampled():
    module = load_resource_graph_module()

    summary = module.summarize_logits_values(
        [0.0, 3.5, -1.0, 2.0],
        logits_buffer_elements=4,
        written_element_count=4,
        diagnostic_reference={
            "status": "pass",
            "scope": "diagnostic_qwen_logits_formula",
            "checked_element_count": 4,
            "max_abs_error": 0.0,
        },
        top_k=1,
    )

    assert summary["coverage"] == "full_logits_buffer_checked"
    assert summary["full_buffer_sampled"] is True
    assert summary["topk"] == [{"token_id": 1, "logit": 3.5}]
    assert summary["diagnostic_reference"]["status"] == "pass"


def test_diagnostic_logits_reference_compares_sampled_formula():
    module = load_resource_graph_module()

    reference = module.diagnostic_logits_reference_values(
        hidden=[2.0, -3.0],
        lm_head=[0.5, 2.0, 4.0, 8.0],
        count=4,
    )
    comparison = module.compare_logits_reference(
        [1.0, -6.0, 8.0, -24.0],
        reference,
    )

    assert reference == [1.0, -6.0, 8.0, -24.0]
    assert comparison["status"] == "pass"
    assert comparison["scope"] == "diagnostic_qwen_logits_formula"
    assert comparison["max_abs_error"] == 0.0


def test_diagnostic_logits_formula_checks_more_than_hidden_extent():
    module = load_resource_graph_module()

    comparison = module.compare_logits_formula(
        [1.0, -6.0, 8.0, -24.0, 1.0, -6.0],
        hidden=[2.0, -3.0],
        lm_head=[0.5, 2.0, 4.0, 8.0],
    )

    assert comparison["status"] == "pass"
    assert comparison["checked_element_count"] == 6
