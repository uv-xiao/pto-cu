import ctypes
import sys
from pathlib import Path

from simpler_setup.cuda_callable_compiler import CudaPersistentDagTask


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples" / "cuda"))

from qwen_decode_loop_runner_impl.launch_preflight import (  # noqa: E402
    build_host_task_packet,
    keyed_fields,
    launch_packet_preflight,
    set_decode_step_index,
    set_decode_step_state,
)
from qwen_decode_loop_runner_impl.activation_workspace import (  # noqa: E402
    workspace_plan,
)
from qwen_decode_loop_runner_impl.workspace_pointers import (  # noqa: E402
    initialize_kv_page_table,
    initialize_rope_tables,
    refresh_rope_tables_for_decode_position,
    rope_table_values,
)
from qwen_decode_loop_runner_impl.resource_backed_results import (  # noqa: E402
    build_execution_result,
    dynamic_rope_refresh_ready,
    prompt_prefill_summary,
)
from qwen_decode_loop_runner_impl.logits_active_cols import (  # noqa: E402
    apply_logits_active_cols_override,
    apply_projection_active_cols_override,
)
from qwen_decode_loop_runner_impl.resource_backed_execution import (  # noqa: E402
    prompt_prefill_descriptors,
    prompt_readout_descriptors,
    select_task_descriptors,
)
from qwen_decode_loop_runner_impl.resource_graph import (  # noqa: E402
    MaterializedGraph,
    activation_buffer_index_for_packet_task,
)
from qwen_decode_loop_runner_impl.graph_materialization import task_summary  # noqa: E402
from qwen_decode_loop_runner_impl.launch_helpers import (  # noqa: E402
    numeric_task_mode_summary,
    required_activation_buffer_count,
)
from qwen_kv_cache_binding_impl.zeroing import (  # noqa: E402
    zero_device_allocation,
)
from qwen_persistent_weight_materialization_impl.materializer import (  # noqa: E402
    materialized_descriptor,
)
from qwen_persistent_weight_materialization_impl.loaders import (  # noqa: E402
    weight_args_shape_fields_ready,
)
from qwen_persistent_weight_args_impl.descriptors import (  # noqa: E402
    build_task_descriptors,
)
from qwen_persistent_weight_args_impl.shape_contract import (  # noqa: E402
    QwenTaskShape,
)


def test_launch_packet_preflight_packs_resource_backed_task_records():
    descriptors = [
        {
            "callable": "qwen_embedding_lookup",
            "tensor_args": [
                {
                    "arg": "tensor_args[0]",
                    "slot_id": 1,
                    "device_ptr_hex": "0x1000",
                }
            ],
        },
        {
            "callable": "qwen_logits",
            "tensor_args": [
                {
                    "arg": "tensor_args[0]",
                    "slot_id": 398,
                    "device_ptr_hex": "0x2000",
                }
            ],
        },
    ]
    plan = {
        "workload_id": "mpk_offline_decode",
        "token_pointer_fields": [
            {"field": "a", "device_ptr_hex": "0x3000"},
            {"field": "b", "device_ptr_hex": "0x4000"},
            {"field": "out", "device_ptr_hex": "0x5000"},
        ],
        "kv_pointer_fields": {
            "c": {"device_ptr_hex": "0x6000"},
            "d": {"device_ptr_hex": "0x7000"},
        },
    }

    preflight = launch_packet_preflight(plan=plan, descriptors=descriptors)

    assert preflight["status"] == "resource_backed_launch_packet_preflight_ready"
    assert preflight["execution_status"] == "not_launched"
    assert preflight["task_count"] == 2
    assert preflight["host_task_packet_bytes"] == 2 * ctypes.sizeof(
        CudaPersistentDagTask,
    )
    assert preflight["missing_runtime_buffers"] == [
        {
            "buffer": "intermediate_activation_buffers",
            "required_count": 1,
            "status": "not_allocated",
        },
        {
            "buffer": "float_logits_or_sampling_output",
            "required_count": 1,
            "status": "not_allocated",
        },
    ]
    assert "intermediate_activation_buffers_not_allocated" in preflight[
        "launch_blockers"
    ]


def test_graph_task_summary_preserves_layer_index():
    summary = task_summary(
        {
            "id": "layer_7_attention_qkv",
            "callable": "qwen_attention_qkv",
            "layer_index": 7,
            "tensor_arg_count": 0,
            "tensor_args": [],
        }
    )

    assert summary["layer_index"] == 7


def test_launch_packet_preflight_binds_activation_workspace():
    descriptors = [
        {"callable": "qwen_embedding_lookup", "tensor_args": []},
        {"callable": "qwen_logits", "tensor_args": []},
    ]
    plan = {
        "workload_id": "mpk_offline_decode",
        "token_pointer_fields": [
            {"field": "a", "device_ptr_hex": "0x3000"},
            {"field": "b", "device_ptr_hex": "0x4000"},
            {"field": "out", "device_ptr_hex": "0x5000"},
        ],
        "kv_pointer_fields": {
            "c": {"device_ptr_hex": "0x6000"},
            "d": {"device_ptr_hex": "0x7000"},
        },
    }
    workspace = {
        "status": "activation_workspace_lifecycle_ready",
        "pointer_table": {
            "mode": "cuda_live",
            "pointer_sets": [
                {
                    "workload_id": "mpk_offline_decode",
                    "activation_buffers": [
                        {
                            "device_ptr_hex": "0x8000",
                            "element_count": 4096,
                        }
                    ],
                    "logits_buffer": {
                        "device_ptr_hex": "0x9000",
                        "element_count": 151936,
                    },
                    "runtime_buffers": {
                        "rope_cos_table": {
                            "device_ptr_hex": "0xa000",
                            "element_count": 64,
                        },
                        "rope_sin_table": {
                            "device_ptr_hex": "0xb000",
                            "element_count": 64,
                        },
                        "kv_page_table": {
                            "device_ptr_hex": "0xc000",
                            "element_count": 1,
                        },
                    },
                    "total_byte_count": 623616,
                }
            ],
        },
    }

    preflight = launch_packet_preflight(
        plan=plan,
        descriptors=descriptors,
        activation_workspace=workspace,
    )

    assert preflight["status"] == "resource_backed_launch_packet_workspace_bound"
    assert preflight["missing_runtime_buffers"] == []
    assert preflight["workspace_pointer_policy"]["status"] == "workspace_bound"
    assert preflight["workspace_pointer_policy"]["runtime_buffers"] == {
        "rope_cos_table": "0xa000",
        "rope_sin_table": "0xb000",
        "kv_page_table": "0xc000",
    }
    assert preflight["remaining_gap"] == "run_prepared_resource_backed_decode_loop"


def test_launch_packet_uses_full_logits_extent_for_final_logits_task():
    descriptors = [
        {"callable": "qwen_embedding_lookup", "tensor_args": []},
        {"callable": "qwen_logits", "tensor_args": []},
    ]
    token_fields = keyed_fields(
        [
            {"field": "a", "device_ptr_hex": "0x3000"},
            {"field": "b", "device_ptr_hex": "0x4000"},
            {"field": "out", "device_ptr_hex": "0x5000"},
        ],
    )
    workspace = {
        "activation_buffers": [
            {
                "device_ptr_hex": "0x8000",
                "element_count": 4096,
            }
        ],
        "logits_buffer": {
            "device_ptr_hex": "0x9000",
            "element_count": 151936,
        },
        "total_byte_count": 623616,
    }

    packet = build_host_task_packet(
        descriptors=descriptors,
        token_fields=token_fields,
        kv_fields={
            "c": {"device_ptr_hex": "0x6000"},
            "d": {"device_ptr_hex": "0x7000"},
        },
        workspace=workspace,
    )

    assert packet is not None
    assert packet[0].n == 4096
    assert packet[0].scalar_arg_count == 0
    assert packet[1].n == 151936
    assert packet[1].scalar_arg_count == 3
    assert list(packet[1].scalar_args)[:3] == [0.0, 4096.0, 151936.0]
    assert packet[1].tensor_args[2] == 0x3000
    assert packet[1].tensor_args[3] == 0x5000
    assert packet[1].tensor_arg_count == 4

    set_decode_step_index(packet, 7)
    assert packet[1].scalar_arg_count == 4
    assert packet[1].scalar_args[3] == 7.0

    set_decode_step_state(packet, step_index=8, decode_position=136)
    assert packet[0].scalar_arg_count == 3
    assert packet[0].scalar_args[2] == 136.0
    assert packet[1].scalar_arg_count == 4
    assert packet[1].scalar_args[2] == 136.0
    assert packet[1].scalar_args[3] == 8.0


def test_decode_step_state_extends_attention_o_kv_window():
    task_t = CudaPersistentDagTask * 2
    packet = task_t(
        CudaPersistentDagTask(func_id=7104, inner=128),
        CudaPersistentDagTask(func_id=7109, inner=151936),
    )

    set_decode_step_state(packet, step_index=3, decode_position=130)

    assert packet[0].scalar_arg_count == 3
    assert packet[0].scalar_args[2] == 130.0
    assert packet[0].inner == 131
    assert packet[1].scalar_arg_count == 4
    assert packet[1].scalar_args[2] == 130.0
    assert packet[1].scalar_args[3] == 3.0
    assert packet[1].inner == 151936


def test_workspace_plan_sizes_activation_buffers_from_descriptor_outputs():
    plan = {
        "workload_id": "mpk_offline_decode",
        "max_batch_size": 2,
        "first_decode_position": 4,
    }
    model_shape = {
        "head_dim": 4,
        "hidden_size": 4,
        "rope_theta": 100.0,
        "vocab_size": 16,
    }
    descriptors = [
        {"callable": "qwen_rmsnorm_input", "task_shape_fields": {"cols": 4}},
        {"callable": "qwen_attention_qkv", "task_shape_fields": {"cols": 8}},
        {"callable": "qwen_logits", "task_shape_fields": {"cols": 16}},
    ]

    workspace = workspace_plan(
        plan=plan,
        graph_task_count=len(descriptors),
        model_shape=model_shape,
        descriptors=descriptors,
    )

    assert workspace["activation_buffer_element_counts"] == [8, 16]
    assert workspace["activation_buffer_byte_counts"] == [32, 64]
    assert workspace["activation_buffer_elements"] == 16
    assert workspace["rope_table_count"] == 2
    assert workspace["rope_table_elements"] == 2
    assert workspace["rope_base_position"] == 4
    assert workspace["rope_theta"] == 100.0
    assert workspace["rope_table_policy"] == (
        "position_correct_for_first_decode_position"
    )
    assert workspace["kv_page_table_count"] == 1
    assert workspace["kv_page_size_tokens"] == 16
    assert workspace["kv_page_table_elements"] == 1
    assert workspace["kv_page_table_bytes"] == 4
    assert workspace["kv_page_table_policy"] == "identity_logical_to_physical_pages"
    assert workspace["total_buffer_count"] == 6
    assert workspace["total_byte_count"] == 32 + 64 + 2 * 16 * 4 + 2 * 2 * 4 + 4


def test_workspace_plan_keeps_terminal_non_readout_activation_buffer():
    plan = {
        "workload_id": "mpk_offline_decode",
        "max_batch_size": 2,
        "first_decode_position": 4,
    }
    model_shape = {
        "head_dim": 4,
        "hidden_size": 4,
        "rope_theta": 100.0,
        "vocab_size": 16,
    }
    descriptors = [
        {"callable": "qwen_embedding_lookup", "task_shape_fields": {"cols": 4}},
        {"callable": "qwen_mlp_down", "task_shape_fields": {"cols": 4}},
    ]

    workspace = workspace_plan(
        plan=plan,
        graph_task_count=len(descriptors),
        model_shape=model_shape,
        descriptors=descriptors,
    )

    assert required_activation_buffer_count(descriptors) == 2
    assert workspace["activation_buffer_count"] == 2
    assert workspace["activation_buffer_element_counts"] == [8, 8]


def test_launch_packet_binds_runtime_rope_table_tensor_args():
    descriptors = [
        {
            "callable": "qwen_attention_qk_norm",
            "tensor_args": [
                {"arg": "tensor_args[0]", "device_ptr_hex": "0x1000"},
                {"arg": "tensor_args[1]", "device_ptr_hex": "0x2000"},
                {
                    "arg": "tensor_args[2]",
                    "tensor": "rope_cos_table",
                    "status": "requires_live_pointer",
                    "device_ptr_source": "runtime_buffers.rope_cos_table",
                },
                {
                    "arg": "tensor_args[3]",
                    "tensor": "rope_sin_table",
                    "status": "requires_live_pointer",
                    "device_ptr_source": "runtime_buffers.rope_sin_table",
                },
                {
                    "arg": "tensor_args[4]",
                    "tensor": "kv_page_table",
                    "status": "runtime_generated_tensor",
                    "device_ptr_source": "runtime_buffers.kv_page_table",
                },
            ],
        }
    ]
    token_fields = keyed_fields(
        [
            {"field": "a", "device_ptr_hex": "0x3000"},
            {"field": "b", "device_ptr_hex": "0x4000"},
            {"field": "out", "device_ptr_hex": "0x5000"},
        ],
    )
    workspace = {
        "activation_buffers": [{"device_ptr_hex": "0x8000", "element_count": 128}],
        "logits_buffer": {"device_ptr_hex": "0x9000", "element_count": 16},
        "runtime_buffers": {
            "rope_cos_table": {"device_ptr_hex": "0xa000", "element_count": 64},
            "rope_sin_table": {"device_ptr_hex": "0xb000", "element_count": 64},
            "kv_page_table": {"device_ptr_hex": "0xc000", "element_count": 1},
        },
        "total_byte_count": 0,
    }

    packet = build_host_task_packet(
        descriptors=descriptors,
        token_fields=token_fields,
        kv_fields={
            "c": {"device_ptr_hex": "0x6000"},
            "d": {"device_ptr_hex": "0x7000"},
        },
        workspace=workspace,
    )

    assert packet is not None
    assert packet[0].tensor_args[0] == 0x1000
    assert packet[0].tensor_args[1] == 0x2000
    assert packet[0].tensor_args[2] == 0xA000
    assert packet[0].tensor_args[3] == 0xB000
    assert packet[0].tensor_args[4] == 0xC000
    assert packet[0].tensor_arg_count == 5


def test_launch_packet_binds_runtime_kv_page_table_tensor_arg():
    descriptors = [
        {
            "callable": "qwen_attention_o",
            "tensor_args": [
                {"arg": "tensor_args[0]", "device_ptr_hex": "0x1000"},
                {
                    "arg": "tensor_args[1]",
                    "tensor": "kv_page_table",
                    "status": "requires_live_pointer",
                    "device_ptr_source": "runtime_buffers.kv_page_table",
                },
            ],
        }
    ]
    token_fields = keyed_fields(
        [
            {"field": "a", "device_ptr_hex": "0x3000"},
            {"field": "b", "device_ptr_hex": "0x4000"},
            {"field": "out", "device_ptr_hex": "0x5000"},
        ],
    )
    workspace = {
        "activation_buffers": [{"device_ptr_hex": "0x8000", "element_count": 128}],
        "logits_buffer": {"device_ptr_hex": "0x9000", "element_count": 16},
        "runtime_buffers": {
            "kv_page_table": {"device_ptr_hex": "0xc000", "element_count": 1},
        },
        "total_byte_count": 0,
    }

    packet = build_host_task_packet(
        descriptors=descriptors,
        token_fields=token_fields,
        kv_fields={
            "c": {"device_ptr_hex": "0x6000"},
            "d": {"device_ptr_hex": "0x7000"},
        },
        workspace=workspace,
    )

    assert packet is not None
    assert packet[0].tensor_args[0] == 0x1000
    assert packet[0].tensor_args[1] == 0xC000
    assert packet[0].tensor_arg_count == 2


def test_rope_table_values_use_qwen_position_formula():
    tables = rope_table_values(
        {
            "rope_base_position": 4,
            "rope_table_elements": 2,
            "rope_theta": 100.0,
        },
    )

    assert [round(value, 6) for value in tables["rope_cos_table"]] == [
        -0.653644,
        0.921061,
    ]
    assert [round(value, 6) for value in tables["rope_sin_table"]] == [
        -0.756802,
        0.389418,
    ]


def test_materialized_graph_uses_configured_scheduler_blocks():
    class FakeRuntime:
        def __init__(self):
            self.next_ptr = 0x100000
            self.memory = {}

        def device_malloc_ctx(self, ctx, size):
            del ctx
            self.next_ptr += 0x1000
            self.memory[self.next_ptr] = bytearray(int(size))
            return self.next_ptr

        def copy_to_device_ctx(self, ctx, device_ptr, host_ptr, size):
            del ctx
            self.memory[int(device_ptr)][: int(size)] = ctypes.string_at(
                host_ptr,
                int(size),
            )
            return 0

        def copy_from_device_ctx(self, ctx, host_ptr, device_ptr, size):
            del ctx
            ctypes.memmove(host_ptr, bytes(self.memory[int(device_ptr)]), int(size))
            return 0

    class FakeSession:
        def __init__(self):
            self.runtime = FakeRuntime()
            self.ctx = object()
            self.allocations = []

    packet = build_host_task_packet(
        descriptors=[
            {"callable": "qwen_embedding_lookup", "tensor_args": []},
            {"callable": "qwen_logits", "tensor_args": []},
        ],
        token_fields={
            "a": {"device_ptr_hex": "0x1000"},
            "b": {"device_ptr_hex": "0x2000"},
            "out": {"device_ptr_hex": "0x3000"},
        },
        kv_fields={
            "c": {"device_ptr_hex": "0x4000"},
            "d": {"device_ptr_hex": "0x5000"},
        },
        workspace=None,
    )
    graph = MaterializedGraph(
        FakeSession(),
        packet,
        scheduler_blocks=3,
        block_dim=128,
    )

    state = graph.make_state()

    assert graph.block_dim == 128
    assert graph.scheduler_blocks == 3
    assert state.scheduler_blocks == 3
    assert len(graph.read_counters()["scheduler_processed_by_block"]) == 3


def test_kv_cache_zero_initialization_uses_chunked_device_copy():
    class FakeRuntime:
        def __init__(self):
            self.copied = []

        def copy_to_device_ctx(self, ctx, device_ptr, host_ptr, size):
            del ctx
            self.copied.append(
                (
                    int(device_ptr),
                    ctypes.string_at(host_ptr, int(size)),
                )
            )
            return 0

    runtime = FakeRuntime()

    zero_device_allocation(
        runtime=runtime,
        ctx=object(),
        ptr=0x1000,
        byte_count=10,
        chunk_bytes=4,
    )

    assert runtime.copied == [
        (0x1000, b"\x00" * 4),
        (0x1004, b"\x00" * 4),
        (0x1008, b"\x00" * 2),
    ]


def test_live_workspace_initializes_rope_tables_from_position_policy():
    class FakeRuntime:
        def __init__(self):
            self.copied = {}

        def copy_to_device_ctx(self, ctx, device_ptr, host_ptr, size):
            del ctx
            count = int(size) // ctypes.sizeof(ctypes.c_float)
            host_t = ctypes.c_float * count
            values = ctypes.cast(host_ptr, ctypes.POINTER(host_t)).contents
            self.copied[int(device_ptr.value)] = list(values)
            return 0

    runtime = FakeRuntime()
    rope_tables = [
        {
            "role": "rope_cos_table",
            "device_ptr": 0xA000,
            "element_count": 2,
        },
        {
            "role": "rope_sin_table",
            "device_ptr": 0xB000,
            "element_count": 2,
        },
    ]

    plan = {
        "rope_base_position": 4,
        "rope_table_elements": 2,
        "rope_table_policy": "position_correct_for_first_decode_position",
        "rope_theta": 100.0,
    }

    initialize_rope_tables(runtime, object(), rope_tables, plan=plan)

    assert [round(value, 6) for value in runtime.copied[0xA000]] == [
        -0.653644,
        0.921061,
    ]
    assert [round(value, 6) for value in runtime.copied[0xB000]] == [
        -0.756802,
        0.389418,
    ]
    assert [item["initialization"] for item in rope_tables] == [
        "position_correct_for_first_decode_position",
        "position_correct_for_first_decode_position",
    ]
    assert [item["base_position"] for item in rope_tables] == [4, 4]


def test_live_workspace_initializes_identity_kv_page_table():
    class FakeRuntime:
        def __init__(self):
            self.copied = {}

        def copy_to_device_ctx(self, ctx, device_ptr, host_ptr, size):
            del ctx
            count = int(size) // ctypes.sizeof(ctypes.c_uint32)
            host_t = ctypes.c_uint32 * count
            values = ctypes.cast(host_ptr, ctypes.POINTER(host_t)).contents
            self.copied[int(device_ptr.value)] = list(values)
            return 0

    runtime = FakeRuntime()
    kv_page_table = {
        "role": "kv_page_table",
        "device_ptr": 0xC000,
        "element_count": 3,
    }
    plan = {
        "kv_page_table_elements": 3,
        "kv_page_table_policy": "identity_logical_to_physical_pages",
        "kv_page_size_tokens": 16,
    }

    initialize_kv_page_table(runtime, object(), kv_page_table, plan=plan)

    assert runtime.copied[0xC000] == [0, 1, 2]
    assert kv_page_table["initialization"] == "identity_logical_to_physical_pages"
    assert kv_page_table["kv_page_size_tokens"] == 16


def test_refresh_rope_tables_for_decode_position_updates_runtime_buffers():
    class FakeRuntime:
        def __init__(self):
            self.copied = {}

        def copy_to_device_ctx(self, ctx, device_ptr, host_ptr, size):
            del ctx
            count = int(size) // ctypes.sizeof(ctypes.c_float)
            host_t = ctypes.c_float * count
            values = ctypes.cast(host_ptr, ctypes.POINTER(host_t)).contents
            self.copied[int(device_ptr.value)] = list(values)
            return 0

    workspace = {
        "runtime_buffers": {
            "rope_cos_table": {
                "role": "rope_cos_table",
                "device_ptr_hex": "0xa000",
                "element_count": 2,
                "rope_theta": 100.0,
            },
            "rope_sin_table": {
                "role": "rope_sin_table",
                "device_ptr_hex": "0xb000",
                "element_count": 2,
                "rope_theta": 100.0,
            },
        }
    }
    runtime = FakeRuntime()

    refresh = refresh_rope_tables_for_decode_position(
        runtime,
        object(),
        workspace,
        decode_position=5,
    )

    assert refresh["status"] == "refreshed"
    assert refresh["policy"] == "position_correct_for_decode_step"
    assert refresh["decode_position"] == 5
    assert [round(value, 6) for value in runtime.copied[0xA000]] == [
        0.283662,
        0.877583,
    ]
    assert [round(value, 6) for value in runtime.copied[0xB000]] == [
        -0.958924,
        0.479426,
    ]
    assert workspace["runtime_buffers"]["rope_cos_table"]["base_position"] == 5


def test_dynamic_rope_refresh_contract_requires_decode_step_refreshes():
    assert dynamic_rope_refresh_ready(
        [
            {
                "repeat_results": [
                    {
                        "decode_step_index": 0,
                        "rope_table_refresh": {
                            "status": "refreshed",
                            "policy": "position_correct_for_decode_step",
                        },
                    },
                    {
                        "decode_step_index": 1,
                        "rope_table_refresh": {
                            "status": "refreshed",
                            "policy": "position_correct_for_decode_step",
                        },
                    },
                ]
            }
        ]
    )
    assert not dynamic_rope_refresh_ready(
        [
            {
                "repeat_results": [
                    {
                        "decode_step_index": None,
                        "rope_table_refresh": {"status": "refreshed"},
                    }
                ]
            }
        ]
    )


def test_launch_packet_uses_per_task_activation_output_extent():
    descriptors = [
        {"callable": "qwen_rmsnorm_input", "tensor_args": []},
        {"callable": "qwen_attention_qkv", "tensor_args": []},
        {"callable": "qwen_logits", "tensor_args": []},
    ]
    token_fields = keyed_fields(
        [
            {"field": "a", "device_ptr_hex": "0x3000"},
            {"field": "b", "device_ptr_hex": "0x4000"},
            {"field": "out", "device_ptr_hex": "0x5000"},
        ],
    )
    workspace = {
        "activation_buffers": [
            {"device_ptr_hex": "0x8000", "element_count": 8},
            {"device_ptr_hex": "0x9000", "element_count": 16},
        ],
        "logits_buffer": {"device_ptr_hex": "0xa000", "element_count": 32},
        "total_byte_count": 224,
    }

    packet = build_host_task_packet(
        descriptors=descriptors,
        token_fields=token_fields,
        kv_fields={
            "c": {"device_ptr_hex": "0x6000"},
            "d": {"device_ptr_hex": "0x7000"},
        },
        workspace=workspace,
    )

    assert packet is not None
    assert packet[0].n == 8
    assert packet[1].n == 16
    assert packet[2].n == 32
    assert list(packet[2].scalar_args)[:3] == [0.0, 16.0, 32.0]


def test_launch_packet_binds_post_attention_residual_source():
    descriptors = [
        {
            "id": "layer_0_input_norm",
            "callable": "qwen_rmsnorm_input",
            "tensor_args": [],
        },
        {
            "id": "layer_0_attention_qkv",
            "callable": "qwen_attention_qkv",
            "tensor_args": [],
        },
        {
            "id": "layer_0_attention_qk_norm",
            "callable": "qwen_attention_qk_norm",
            "tensor_args": [],
        },
        {
            "id": "layer_0_attention_o",
            "callable": "qwen_attention_o",
            "tensor_args": [],
        },
        {
            "id": "layer_0_post_attention_norm",
            "callable": "qwen_rmsnorm_post_attention",
            "tensor_args": [],
        },
        {
            "id": "layer_0_mlp_gate_up",
            "callable": "qwen_mlp_gate_up",
            "tensor_args": [],
        },
    ]
    token_fields = keyed_fields(
        [
            {"field": "a", "device_ptr_hex": "0x3000"},
            {"field": "b", "device_ptr_hex": "0x4000"},
            {"field": "out", "device_ptr_hex": "0x5000"},
        ],
    )
    workspace = {
        "activation_buffers": [
            {"device_ptr_hex": "0x8000", "element_count": 8},
            {"device_ptr_hex": "0x9000", "element_count": 16},
            {"device_ptr_hex": "0xa000", "element_count": 16},
            {"device_ptr_hex": "0xb000", "element_count": 8},
            {"device_ptr_hex": "0xc000", "element_count": 8},
            {"device_ptr_hex": "0xd000", "element_count": 16},
        ],
        "logits_buffer": {"device_ptr_hex": "0xe000", "element_count": 32},
        "total_byte_count": 304,
    }

    packet = build_host_task_packet(
        descriptors=descriptors,
        token_fields=token_fields,
        kv_fields={
            "c": {"device_ptr_hex": "0x6000"},
            "d": {"device_ptr_hex": "0x7000"},
        },
        workspace=workspace,
    )

    assert packet is not None
    assert packet[4].a == 0xB000
    assert packet[4].b == 0x3000


def test_launch_packet_binds_mlp_down_residual_source():
    descriptors = [
        {
            "id": "embedding_lookup",
            "callable": "qwen_embedding_lookup",
            "tensor_args": [],
        },
        {
            "id": "layer_0_input_norm",
            "callable": "qwen_rmsnorm_input",
            "tensor_args": [],
        },
        {
            "id": "layer_0_attention_qkv",
            "callable": "qwen_attention_qkv",
            "tensor_args": [],
        },
        {
            "id": "layer_0_attention_qk_norm",
            "callable": "qwen_attention_qk_norm",
            "tensor_args": [],
        },
        {
            "id": "layer_0_attention_o",
            "callable": "qwen_attention_o",
            "tensor_args": [],
        },
        {
            "id": "layer_0_post_attention_norm",
            "callable": "qwen_rmsnorm_post_attention",
            "tensor_args": [],
        },
        {
            "id": "layer_0_mlp_gate_up",
            "callable": "qwen_mlp_gate_up",
            "tensor_args": [],
        },
        {
            "id": "layer_0_mlp_down",
            "callable": "qwen_mlp_down",
            "tensor_args": [],
        },
        {"id": "final_norm", "callable": "qwen_final_norm", "tensor_args": []},
    ]
    token_fields = keyed_fields(
        [
            {"field": "a", "device_ptr_hex": "0x3000"},
            {"field": "b", "device_ptr_hex": "0x4000"},
            {"field": "out", "device_ptr_hex": "0x5000"},
        ],
    )
    workspace = {
        "activation_buffers": [
            {"device_ptr_hex": "0x8000", "element_count": 8},
            {"device_ptr_hex": "0x9000", "element_count": 16},
            {"device_ptr_hex": "0xa000", "element_count": 16},
            {"device_ptr_hex": "0xb000", "element_count": 8},
            {"device_ptr_hex": "0xc000", "element_count": 8},
            {"device_ptr_hex": "0xd000", "element_count": 16},
            {"device_ptr_hex": "0xe000", "element_count": 8},
            {"device_ptr_hex": "0xf000", "element_count": 8},
        ],
        "logits_buffer": {"device_ptr_hex": "0x10000", "element_count": 32},
        "total_byte_count": 368,
    }

    packet = build_host_task_packet(
        descriptors=descriptors,
        token_fields=token_fields,
        kv_fields={
            "c": {"device_ptr_hex": "0x6000"},
            "d": {"device_ptr_hex": "0x7000"},
        },
        workspace=workspace,
    )

    assert packet is not None
    assert packet[7].a == 0xE000
    assert packet[7].b == 0xC000
    assert packet[7].tensor_args[1] == 0x8000
    assert packet[7].tensor_arg_count == 2


def test_launch_packet_carries_cuda_task_shape_fields():
    descriptors = [
        {"callable": "qwen_rmsnorm_input", "tensor_args": []},
        {
            "callable": "qwen_attention_qkv",
            "tensor_args": [
                {
                    "arg": "tensor_args[0]",
                    "slot_id": 2,
                    "device_ptr_hex": "0xa000",
                },
                {
                    "arg": "tensor_args[1]",
                    "slot_id": 3,
                    "device_ptr_hex": "0xb000",
                },
            ],
            "tensor_arg_metadata": [
                {"arg": "tensor_args[0]", "dtype": "bfloat16"},
                {"arg": "tensor_args[1]", "dtype": "float32"},
            ],
            "task_shape_fields": {
                "rows": 16,
                "cols": 4096,
                "inner": 4096,
                "lda": 4096,
                "ldb": 4096,
                "ldc": 4096,
                "scalar0": 1.25,
                "scalar1": 1024,
                "a_batch_stride": 4096,
            },
        },
        {"callable": "qwen_logits", "tensor_args": []},
    ]
    token_fields = keyed_fields(
        [
            {"field": "a", "device_ptr_hex": "0x3000"},
            {"field": "b", "device_ptr_hex": "0x4000"},
            {"field": "out", "device_ptr_hex": "0x5000"},
        ],
    )

    packet = build_host_task_packet(
        descriptors=descriptors,
        token_fields=token_fields,
        kv_fields={
            "c": {"device_ptr_hex": "0x6000"},
            "d": {"device_ptr_hex": "0x7000"},
        },
        task_shape_defaults={"rows": 2, "cols": 64, "inner": 128},
    )

    assert packet is not None
    assert packet[0].rows == 2
    assert packet[0].cols == 64
    assert packet[0].inner == 128
    assert packet[1].rows == 16
    assert packet[1].cols == 4096
    assert packet[1].inner == 4096
    assert packet[1].lda == 4096
    assert packet[1].ldb == 4096
    assert packet[1].ldc == 4096
    assert packet[1].scalar0 == 1.25
    assert packet[1].scalar1 == 1024.0
    assert packet[1].a_batch_stride == 4096
    assert list(packet[1].tensor_arg_dtypes)[:2] == [6, 0]


def test_materialized_weight_descriptor_preserves_task_shape_fields():
    descriptor = {
        "id": "layer_0_attention_qkv",
        "callable": "qwen_attention_qkv",
        "phase": "per_layer_decode",
        "layer_index": 3,
        "tensor_args": [
            {
                "arg": "tensor_args[0]",
                "slot_id": 7,
                "tensor": "model.layers.0.self_attn.q_proj.weight",
            }
        ],
        "task_shape_fields": {"rows": 16, "cols": 4096, "inner": 4096},
        "tensor_arg_metadata": [
            {
                "arg": "tensor_args[0]",
                "slot_id": 7,
                "tensor": "model.layers.0.self_attn.q_proj.weight",
                "dtype": "bfloat16",
                "shape": [4096, 4096],
            }
        ],
    }

    materialized = materialized_descriptor(
        descriptor=descriptor,
        bindings={7: {"size_bytes": 1024}},
        pointers={
            7: {
                "tensor": "model.layers.0.self_attn.q_proj.weight",
                "device_ptr": 0x1000,
                "device_ptr_hex": "0x1000",
                "size_bytes": 1024,
            }
        },
        pointer_table_ready=True,
    )

    assert materialized["status"] == "ready"
    assert materialized["layer_index"] == 3
    assert materialized["task_shape_fields"] == {
        "rows": 16,
        "cols": 4096,
        "inner": 4096,
    }
    assert materialized["tensor_arg_metadata"][0]["dtype"] == "bfloat16"
    assert materialized["tensor_arg_metadata"][0]["shape"] == [4096, 4096]


def test_weight_args_loader_rejects_shape_field_stale_artifact():
    assert not weight_args_shape_fields_ready(
        {
            "task_arg_descriptors": [
                {"id": "final_norm", "callable": "qwen_final_norm"},
                {"id": "logits", "callable": "qwen_logits"},
            ],
        }
    )
    assert not weight_args_shape_fields_ready(
        {
            "task_arg_descriptors": [
                {
                    "id": "logits",
                    "callable": "qwen_logits",
                    "task_shape_fields": {"cols": 16, "inner": 4},
                },
            ],
        }
    )
    assert weight_args_shape_fields_ready(
        {
            "task_arg_descriptors": [
                {
                    "id": "logits",
                    "callable": "qwen_logits",
                    "task_shape_fields": {
                        "cols": 16,
                        "inner": 4,
                        "scalar1": 1024,
                    },
                },
            ],
        }
    )


def test_logits_active_cols_override_preserves_original_descriptors():
    descriptors = [
        {
            "id": "layer_0_mlp_down",
            "callable": "qwen_mlp_down",
            "task_shape_fields": {"cols": 4, "scalar1": 1024},
        },
        {
            "id": "logits",
            "callable": "qwen_logits",
            "task_shape_fields": {"cols": 16, "inner": 4, "scalar1": 1024},
        },
    ]

    updated, policy = apply_logits_active_cols_override(descriptors, "full")

    assert policy == {
        "mode": "full_descriptor_cols",
        "requested_active_cols": "full",
        "applied_scalar1_values": [16],
    }
    assert descriptors[1]["task_shape_fields"]["scalar1"] == 1024
    assert updated[0] is descriptors[0]
    assert updated[1]["task_shape_fields"] == {
        "cols": 16,
        "inner": 4,
        "scalar1": 16,
    }


def test_logits_active_cols_override_accepts_explicit_window():
    descriptors = [
        {
            "id": "logits",
            "callable": "qwen_logits",
            "task_shape_fields": {"cols": 151936, "scalar1": 1024},
        },
    ]

    updated, policy = apply_logits_active_cols_override(descriptors, 4096)

    assert policy == {
        "mode": "explicit_active_cols",
        "requested_active_cols": 4096,
        "applied_scalar1_values": [4096],
    }
    assert updated[0]["task_shape_fields"]["scalar1"] == 4096


def test_projection_active_cols_override_targets_only_projection_callables():
    descriptors = [
        {
            "id": "layer_0_attention_qkv",
            "callable": "qwen_attention_qkv",
            "task_shape_fields": {"cols": 6144, "scalar1": 1024},
        },
        {
            "id": "layer_0_attention_o",
            "callable": "qwen_attention_o",
            "task_shape_fields": {"cols": 4096, "scalar1": 16},
        },
        {
            "id": "layer_0_mlp_gate_up",
            "callable": "qwen_mlp_gate_up",
            "task_shape_fields": {"cols": 12288, "scalar1": 1024},
        },
        {
            "id": "layer_0_mlp_down",
            "callable": "qwen_mlp_down",
            "task_shape_fields": {"cols": 4096, "scalar1": 1024},
        },
        {
            "id": "logits",
            "callable": "qwen_logits",
            "task_shape_fields": {"cols": 151936, "scalar1": 1024},
        },
    ]

    updated, policy = apply_projection_active_cols_override(descriptors, "full")

    assert policy == {
        "mode": "full_descriptor_cols",
        "requested_active_cols": "full",
        "applied_scalar1_values": [
            {
                "callable": "qwen_attention_qkv",
                "id": "layer_0_attention_qkv",
                "field": "scalar1",
                "value": 6144,
            },
            {
                "callable": "qwen_attention_o",
                "id": "layer_0_attention_o",
                "field": "attention_o_projection_input_count",
                "value": 4096,
            },
            {
                "callable": "qwen_mlp_gate_up",
                "id": "layer_0_mlp_gate_up",
                "field": "scalar1",
                "value": 12288,
            },
            {
                "callable": "qwen_mlp_down",
                "id": "layer_0_mlp_down",
                "field": "scalar1",
                "value": 4096,
            },
        ],
    }
    assert descriptors[0]["task_shape_fields"]["scalar1"] == 1024
    assert updated[0]["task_shape_fields"]["scalar1"] == 6144
    assert updated[1]["task_shape_fields"]["scalar1"] == 16
    assert updated[1]["attention_o_projection_input_count"] == 4096
    assert updated[2]["task_shape_fields"]["scalar1"] == 12288
    assert updated[3]["task_shape_fields"]["scalar1"] == 4096
    assert updated[4] is descriptors[4]


def test_qwen_weight_descriptors_emit_callable_shape_fields():
    bindings = {
        "model.embed_tokens.weight": {"slot_id": 0},
        "model.layers.0.input_layernorm.weight": {"slot_id": 1},
        "model.layers.0.self_attn.q_proj.weight": {
            "slot_id": 2,
            "dtype": "bfloat16",
            "shape": [4, 4],
            "size_bytes": 32,
        },
        "model.layers.0.self_attn.k_proj.weight": {"slot_id": 3},
        "model.layers.0.self_attn.v_proj.weight": {"slot_id": 4},
        "model.layers.0.self_attn.q_norm.weight": {"slot_id": 5},
        "model.layers.0.self_attn.k_norm.weight": {"slot_id": 6},
        "model.layers.0.self_attn.o_proj.weight": {"slot_id": 7},
        "model.layers.0.post_attention_layernorm.weight": {"slot_id": 8},
        "model.layers.0.mlp.gate_proj.weight": {"slot_id": 9},
        "model.layers.0.mlp.up_proj.weight": {"slot_id": 10},
        "model.layers.0.mlp.down_proj.weight": {"slot_id": 11},
        "model.norm.weight": {"slot_id": 12},
        "lm_head.weight": {"slot_id": 13},
    }
    model_shape = QwenTaskShape(
        hidden_size=4,
        intermediate_size=8,
        vocab_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
    )

    descriptors = {
        item["id"]: item
        for item in build_task_descriptors(
            bindings=bindings,
            num_hidden_layers=1,
            model_shape=model_shape,
        )
    }

    assert descriptors["layer_0_attention_qkv"]["task_shape_fields"] == {
        "cols": 8,
        "inner": 4,
        "lda": 4,
        "ldb": 4,
        "ldc": 8,
        "scalar0": 16,
        "scalar1": 1024,
    }
    assert descriptors["layer_0_attention_qkv"]["tensor_args"][3] == {
        "arg": "tensor_args[3]",
        "tensor": "kv_page_table",
        "role": "kv_page_table",
        "status": "runtime_generated_tensor",
        "device_ptr_source": "runtime_buffers.kv_page_table",
    }
    assert descriptors["layer_0_attention_qkv"]["layer_index"] == 0
    qkv_metadata = descriptors["layer_0_attention_qkv"]["tensor_arg_metadata"][0]
    assert qkv_metadata["dtype"] == "bfloat16"
    assert qkv_metadata["shape"] == [4, 4]
    assert qkv_metadata["size_bytes"] == 32
    assert descriptors["layer_0_attention_qk_norm"]["task_shape_fields"] == {
        "cols": 6,
        "inner": 2,
        "lda": 2,
        "ldb": 1,
        "ldc": 6,
        "a_batch_stride": 8,
        "scalar0": 16,
    }
    assert descriptors["layer_0_attention_qk_norm"]["tensor_args"][4] == {
        "arg": "tensor_args[4]",
        "tensor": "kv_page_table",
        "role": "kv_page_table",
        "status": "runtime_generated_tensor",
        "device_ptr_source": "runtime_buffers.kv_page_table",
    }
    assert descriptors["layer_0_attention_o"]["task_shape_fields"] == {
        "cols": 4,
        "inner": 2,
        "lda": 2,
        "ldb": 1,
        "ldc": 4,
        "a_batch_stride": 6,
        "scalar0": 16.0,
        "scalar1": 16.0,
    }
    assert descriptors["layer_0_attention_o"]["tensor_args"][1] == {
        "arg": "tensor_args[1]",
        "tensor": "kv_page_table",
        "role": "kv_page_table",
        "status": "runtime_generated_tensor",
        "device_ptr_source": "runtime_buffers.kv_page_table",
    }
    assert descriptors["layer_0_mlp_down"]["task_shape_fields"] == {
        "cols": 4,
        "inner": 8,
        "lda": 8,
        "ldb": 8,
        "ldc": 4,
        "scalar1": 1024,
    }
    assert descriptors["layer_0_mlp_down"]["tensor_args"][1] == {
        "arg": "tensor_args[1]",
        "tensor": "mlp_residual",
        "role": "mlp_residual",
        "status": "runtime_generated_tensor",
        "device_ptr_source": "runtime_buffers.mlp_residual",
    }
    assert descriptors["layer_0_mlp_gate_up"]["task_shape_fields"] == {
        "cols": 8,
        "inner": 4,
        "lda": 4,
        "ldb": 4,
        "ldc": 8,
        "scalar1": 1024,
    }
    assert descriptors["logits"]["task_shape_fields"] == {
        "cols": 16,
        "inner": 4,
        "lda": 4,
        "ldb": 4,
        "ldc": 16,
        "scalar0": 256,
        "scalar1": 1024,
    }


def test_launch_packet_carries_layer_index_for_kv_tasks():
    descriptors = [
        {
            "id": "layer_2_attention_qkv",
            "callable": "qwen_attention_qkv",
            "layer_index": 2,
            "tensor_args": [],
        },
        {
            "id": "layer_2_attention_qk_norm",
            "callable": "qwen_attention_qk_norm",
            "layer_index": 2,
            "tensor_args": [],
        },
        {
            "id": "layer_2_attention_o",
            "callable": "qwen_attention_o",
            "layer_index": 2,
            "tensor_args": [],
            "attention_o_projection_input_count": 4096,
        },
        {"id": "logits", "callable": "qwen_logits", "tensor_args": []},
    ]
    token_fields = keyed_fields(
        [
            {"field": "a", "device_ptr_hex": "0x3000"},
            {"field": "b", "device_ptr_hex": "0x4000"},
            {"field": "out", "device_ptr_hex": "0x5000"},
        ],
    )

    packet = build_host_task_packet(
        descriptors=descriptors,
        token_fields=token_fields,
        kv_fields={
            "c": {"device_ptr_hex": "0x6000"},
            "d": {"device_ptr_hex": "0x7000"},
        },
        numeric_task_mode="unit_math_full_rmsnorm",
    )

    assert packet is not None
    assert packet[0].scalar_args[3] == 2.0
    assert packet[0].scalar_arg_count == 4
    assert packet[1].scalar_args[3] == 2.0
    assert packet[1].scalar_arg_count == 4
    assert packet[2].scalar_args[3] == 2.0
    assert packet[2].scalar_arg_count == 4
    assert packet[2].scalar_args[1] == 4096.0

    set_decode_step_state(packet, step_index=5, decode_position=17)

    assert packet[0].scalar_args[2] == 17.0
    assert packet[0].scalar_args[3] == 2.0
    assert packet[2].scalar_args[2] == 17.0
    assert packet[2].scalar_args[3] == 2.0
    assert packet[3].scalar_args[3] == 5.0


def test_launch_packet_marks_unit_math_numeric_ready_tasks():
    descriptors = [
        {"callable": "qwen_rmsnorm_input", "tensor_args": []},
        {"callable": "qwen_attention_qkv", "tensor_args": []},
        {"callable": "qwen_attention_qk_norm", "tensor_args": []},
        {"callable": "qwen_attention_o", "tensor_args": []},
        {"callable": "qwen_rmsnorm_post_attention", "tensor_args": []},
        {"callable": "qwen_mlp_down", "tensor_args": []},
        {"callable": "qwen_final_norm", "tensor_args": []},
        {"callable": "qwen_logits", "tensor_args": []},
    ]
    token_fields = keyed_fields(
        [
            {"field": "a", "device_ptr_hex": "0x3000"},
            {"field": "b", "device_ptr_hex": "0x4000"},
            {"field": "out", "device_ptr_hex": "0x5000"},
        ],
    )

    packet = build_host_task_packet(
        descriptors=descriptors,
        token_fields=token_fields,
        kv_fields={
            "c": {"device_ptr_hex": "0x6000"},
            "d": {"device_ptr_hex": "0x7000"},
        },
        numeric_task_mode="unit_math",
    )

    assert packet is not None
    assert packet[0].scalar_arg_count == 2
    assert list(packet[0].scalar_args)[:2] == [1.0, 1.0]
    assert packet[1].scalar_arg_count == 1
    assert packet[1].scalar_args[0] == 1.0
    assert packet[2].scalar_arg_count == 1
    assert packet[2].scalar_args[0] == 1.0
    assert packet[3].scalar_arg_count == 2
    assert list(packet[3].scalar_args)[:2] == [1.0, 64.0]
    assert packet[4].scalar_arg_count == 2
    assert list(packet[4].scalar_args)[:2] == [1.0, 1.0]
    assert packet[5].scalar_arg_count == 1
    assert packet[5].scalar_args[0] == 1.0
    assert packet[6].scalar_arg_count == 2
    assert list(packet[6].scalar_args)[:2] == [1.0, 1.0]
    assert packet[7].scalar_arg_count == 3
    assert list(packet[7].scalar_args)[:3] == [0.0, 1.0, 1.0]


def test_resource_backed_execution_reports_task_coverage():
    class FakeSession:
        device = 0

    class FakeArtifact:
        cache_key = "fake"
        cache_hit = False
        source_path = "tmp/fake.cu"
        ptx_path = "tmp/fake.ptx"
        entry_name = "pto_persistent_dag_f32_executor"
        source_kind = "generated-dispatch"

    class FakePrepared:
        artifact = FakeArtifact()

    callables = [
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
    ]
    descriptors = [{"callable": callable_name} for callable_name in callables]

    result = build_execution_result(
        session=FakeSession(),
        arch="compute_80",
        prepared=FakePrepared(),
        descriptors=descriptors,
        workload_results=[
            {
                "workload_id": "mpk_offline_decode",
                "status": "pass",
                "planned_decode_steps": 1,
                "executed_decode_steps": 1,
                "decode_step_limit": 1,
                "repeat_results": [],
                "logits_summary_stable": True,
            },
        ],
        repeat_runs=1,
        decode_step_limit=1,
        workload_ids=None,
        max_task_count=10,
        task_selection="first_layer_with_logits",
        layer_count=None,
        scheduler_blocks=1,
        worker_blocks=8,
        grid_dim=9,
        logits_check_policy="final_step",
        logits_active_cols_policy={"mode": "descriptor_default"},
        projection_active_cols_policy={"mode": "descriptor_default"},
        numeric_task_mode="unit_math_full_rmsnorm",
        prefill_prompt=False,
        repo_relative=lambda path: str(path),
    )

    assert result["task_coverage"] == {
        "task_count": 10,
        "func_id_sequence": list(range(7100, 7110)),
        "callables": callables,
    }
    assert result["repeat_policy"]["projection_active_cols_policy"] == {
        "mode": "descriptor_default",
    }


def test_prompt_prefill_summary_reports_executed_prompt_positions():
    summary = prompt_prefill_summary(
        plan={"active_prompt_tokens": 3, "first_decode_position": 64},
        prefill_packet_len=8,
        prefill_task_policy="omit_final_norm_and_logits_readout",
        prefill_results=[
            {
                "status": "pass",
                "scheduler_counters": {"completed_count": 10, "error_count": 0},
            },
            {
                "status": "pass",
                "scheduler_counters": {"completed_count": 10, "error_count": 0},
            },
            {
                "status": "pass",
                "scheduler_counters": {"completed_count": 10, "error_count": 0},
            },
        ],
    )

    assert summary == {
        "status": "prompt_prefill_executed",
        "expected_prompt_positions": 3,
        "executed_prompt_positions": 3,
        "first_decode_position": 64,
        "graph_task_count": 8,
        "task_policy": "omit_final_norm_and_logits_readout",
        "total_completed_count": 30,
        "total_error_count": 0,
    }


def test_resource_backed_first_layer_logits_selector_keeps_final_tasks():
    descriptors = [
        {"id": "embedding_lookup", "callable": "qwen_embedding_lookup"},
        {"id": "layer_0_input_norm", "callable": "qwen_rmsnorm_input"},
        {"id": "layer_0_attention_qkv", "callable": "qwen_attention_qkv"},
        {"id": "layer_0_attention_qk_norm", "callable": "qwen_attention_qk_norm"},
        {"id": "layer_0_attention_o", "callable": "qwen_attention_o"},
        {
            "id": "layer_0_post_attention_norm",
            "callable": "qwen_rmsnorm_post_attention",
        },
        {"id": "layer_0_mlp_gate_up", "callable": "qwen_mlp_gate_up"},
        {"id": "layer_0_mlp_down", "callable": "qwen_mlp_down"},
        {"id": "layer_1_input_norm", "callable": "qwen_rmsnorm_input"},
        {"id": "final_norm", "callable": "qwen_final_norm"},
        {"id": "logits", "callable": "qwen_logits"},
    ]

    selected = select_task_descriptors(
        descriptors,
        max_task_count=10,
        task_selection="first_layer_with_logits",
    )

    assert [item["id"] for item in selected] == [
        "embedding_lookup",
        "layer_0_input_norm",
        "layer_0_attention_qkv",
        "layer_0_attention_qk_norm",
        "layer_0_attention_o",
        "layer_0_post_attention_norm",
        "layer_0_mlp_gate_up",
        "layer_0_mlp_down",
        "final_norm",
        "logits",
    ]


def test_resource_backed_layer_prefix_selector_keeps_complete_layers_and_logits():
    descriptors = [
        {"id": "embedding_lookup", "callable": "qwen_embedding_lookup"},
        {"id": "layer_0_input_norm", "callable": "qwen_rmsnorm_input"},
        {"id": "layer_0_attention_qkv", "callable": "qwen_attention_qkv"},
        {"id": "layer_0_attention_qk_norm", "callable": "qwen_attention_qk_norm"},
        {"id": "layer_0_attention_o", "callable": "qwen_attention_o"},
        {
            "id": "layer_0_post_attention_norm",
            "callable": "qwen_rmsnorm_post_attention",
        },
        {"id": "layer_0_mlp_gate_up", "callable": "qwen_mlp_gate_up"},
        {"id": "layer_0_mlp_down", "callable": "qwen_mlp_down"},
        {"id": "layer_1_input_norm", "callable": "qwen_rmsnorm_input"},
        {"id": "layer_1_attention_qkv", "callable": "qwen_attention_qkv"},
        {"id": "layer_1_attention_qk_norm", "callable": "qwen_attention_qk_norm"},
        {"id": "layer_1_attention_o", "callable": "qwen_attention_o"},
        {
            "id": "layer_1_post_attention_norm",
            "callable": "qwen_rmsnorm_post_attention",
        },
        {"id": "layer_1_mlp_gate_up", "callable": "qwen_mlp_gate_up"},
        {"id": "layer_1_mlp_down", "callable": "qwen_mlp_down"},
        {"id": "layer_2_input_norm", "callable": "qwen_rmsnorm_input"},
        {"id": "final_norm", "callable": "qwen_final_norm"},
        {"id": "logits", "callable": "qwen_logits"},
    ]

    selected = select_task_descriptors(
        descriptors,
        max_task_count=None,
        task_selection="layer_prefix_with_logits",
        layer_count=2,
    )

    assert [item["id"] for item in selected] == [
        "embedding_lookup",
        "layer_0_input_norm",
        "layer_0_attention_qkv",
        "layer_0_attention_qk_norm",
        "layer_0_attention_o",
        "layer_0_post_attention_norm",
        "layer_0_mlp_gate_up",
        "layer_0_mlp_down",
        "layer_1_input_norm",
        "layer_1_attention_qkv",
        "layer_1_attention_qk_norm",
        "layer_1_attention_o",
        "layer_1_post_attention_norm",
        "layer_1_mlp_gate_up",
        "layer_1_mlp_down",
        "final_norm",
        "logits",
    ]


def test_prompt_prefill_descriptors_omit_final_readout_tasks():
    descriptors = [
        {"id": "embedding_lookup", "callable": "qwen_embedding_lookup"},
        {"id": "layer_0_input_norm", "callable": "qwen_rmsnorm_input"},
        {"id": "layer_0_mlp_down", "callable": "qwen_mlp_down"},
        {"id": "final_norm", "callable": "qwen_final_norm"},
        {"id": "logits", "callable": "qwen_logits"},
    ]

    prefill = prompt_prefill_descriptors(descriptors)

    assert [item["id"] for item in prefill] == [
        "embedding_lookup",
        "layer_0_input_norm",
        "layer_0_mlp_down",
    ]


def test_prompt_readout_descriptors_keep_final_norm_and_logits():
    descriptors = [
        {"id": "embedding_lookup", "callable": "qwen_embedding_lookup"},
        {"id": "layer_0_mlp_down", "callable": "qwen_mlp_down"},
        {"id": "final_norm", "callable": "qwen_final_norm"},
        {"id": "logits", "callable": "qwen_logits"},
    ]

    readout = prompt_readout_descriptors(descriptors)

    assert [item["id"] for item in readout] == ["final_norm", "logits"]


def test_readout_packet_uses_offset_activation_as_first_input():
    descriptors = [
        {"id": "final_norm", "callable": "qwen_final_norm", "tensor_args": []},
        {"id": "logits", "callable": "qwen_logits", "tensor_args": []},
    ]
    token_fields = keyed_fields(
        [
            {"field": "a", "device_ptr_hex": "0x1000"},
            {"field": "b", "device_ptr_hex": "0x2000"},
            {"field": "out", "device_ptr_hex": "0x3000"},
        ],
    )
    workspace = {
        "activation_buffers": [
            {"device_ptr_hex": "0x4000", "element_count": 16},
            {"device_ptr_hex": "0x5000", "element_count": 16},
            {"device_ptr_hex": "0x6000", "element_count": 16},
            {"device_ptr_hex": "0x7000", "element_count": 16},
        ],
        "logits_buffer": {"device_ptr_hex": "0x8000", "element_count": 32},
    }

    packet = build_host_task_packet(
        descriptors=descriptors,
        token_fields=token_fields,
        kv_fields={
            "c": {"device_ptr_hex": "0x9000"},
            "d": {"device_ptr_hex": "0xa000"},
        },
        workspace=workspace,
        numeric_task_mode="unit_math_full_rmsnorm",
        packet_index_offset=3,
    )

    assert packet is not None
    assert packet[0].a == 0x6000
    assert packet[0].out == 0x7000
    assert packet[1].a == 0x7000
    assert packet[1].out == 0x8000


def test_readout_activation_sampling_uses_packet_offset():
    assert activation_buffer_index_for_packet_task(
        task_index=0,
        packet_index_offset=57,
    ) == 57
    assert activation_buffer_index_for_packet_task(
        task_index=1,
        packet_index_offset=57,
    ) == 58


def test_prefill_packet_keeps_last_non_logits_output_in_activation_chain():
    descriptors = [
        {"id": "embedding_lookup", "callable": "qwen_embedding_lookup"},
        {"id": "layer_0_mlp_down", "callable": "qwen_mlp_down"},
    ]
    token_fields = keyed_fields(
        [
            {"field": "a", "device_ptr_hex": "0x1000"},
            {"field": "b", "device_ptr_hex": "0x2000"},
            {"field": "out", "device_ptr_hex": "0x3000"},
        ],
    )
    workspace = {
        "activation_buffers": [
            {"device_ptr_hex": "0x4000", "element_count": 16},
            {"device_ptr_hex": "0x5000", "element_count": 16},
        ],
        "logits_buffer": {"device_ptr_hex": "0x8000", "element_count": 32},
    }

    packet = build_host_task_packet(
        descriptors=descriptors,
        token_fields=token_fields,
        kv_fields={
            "c": {"device_ptr_hex": "0x9000"},
            "d": {"device_ptr_hex": "0xa000"},
        },
        workspace=workspace,
        numeric_task_mode="unit_math_full_rmsnorm",
    )

    assert packet is not None
    assert packet[1].out == 0x5000


def test_launch_packet_can_select_full_rmsnorm_reduction_branch():
    descriptors = [
        {"callable": "qwen_rmsnorm_input", "tensor_args": []},
        {"callable": "qwen_attention_qkv", "tensor_args": []},
        {"callable": "qwen_final_norm", "tensor_args": []},
        {"callable": "qwen_logits", "tensor_args": []},
    ]
    token_fields = keyed_fields(
        [
            {"field": "a", "device_ptr_hex": "0x3000"},
            {"field": "b", "device_ptr_hex": "0x4000"},
            {"field": "out", "device_ptr_hex": "0x5000"},
        ],
    )

    packet = build_host_task_packet(
        descriptors=descriptors,
        token_fields=token_fields,
        kv_fields={
            "c": {"device_ptr_hex": "0x6000"},
            "d": {"device_ptr_hex": "0x7000"},
        },
        numeric_task_mode="unit_math_full_rmsnorm",
    )

    assert packet is not None
    assert packet[0].scalar_arg_count == 2
    assert list(packet[0].scalar_args)[:2] == [1.0, 0.0]
    assert packet[1].scalar_arg_count == 1
    assert packet[1].scalar_args[0] == 1.0
    assert packet[2].scalar_arg_count == 2
    assert list(packet[2].scalar_args)[:2] == [1.0, 0.0]
    set_decode_step_state(packet, step_index=0, decode_position=128)
    assert packet[0].scalar_arg_count == 3
    assert list(packet[0].scalar_args)[:3] == [1.0, 0.0, 128.0]
    summary = numeric_task_mode_summary("unit_math_full_rmsnorm")
    assert summary["scope"] == "resource_backed_unit_math_full_rmsnorm_reduction"
    assert summary["external_scale_contracts"] == []
    assert summary["full_reduction_contracts"] == [
        {
            "callable": "qwen_rmsnorm_input",
            "scalar_arg_count": 2,
            "scope": "resource_backed_full_rmsnorm_reduction",
            "threading": "block",
        },
        {
            "callable": "qwen_rmsnorm_post_attention",
            "scalar_arg_count": 2,
            "scope": "resource_backed_full_rmsnorm_reduction",
            "threading": "block",
        },
        {
            "callable": "qwen_final_norm",
            "scalar_arg_count": 2,
            "scope": "resource_backed_full_rmsnorm_reduction",
            "threading": "block",
        },
    ]


def test_full_rmsnorm_mode_uses_full_hidden_vector_extent():
    descriptors = [
        {"callable": "qwen_rmsnorm_input", "tensor_args": []},
        {"callable": "qwen_attention_qkv", "tensor_args": []},
        {"callable": "qwen_logits", "tensor_args": []},
    ]
    token_fields = keyed_fields(
        [
            {"field": "a", "device_ptr_hex": "0x3000"},
            {"field": "b", "device_ptr_hex": "0x4000"},
            {"field": "out", "device_ptr_hex": "0x5000"},
        ],
    )
    workspace = {
        "activation_buffers": [
            {
                "device_ptr_hex": "0x8000",
                "element_count": 65536,
            },
            {
                "device_ptr_hex": "0x9000",
                "element_count": 65536,
            },
        ],
        "logits_buffer": {
            "device_ptr_hex": "0xa000",
            "element_count": 2430976,
        },
        "total_byte_count": 0,
    }

    packet = build_host_task_packet(
        descriptors=descriptors,
        token_fields=token_fields,
        kv_fields={
            "c": {"device_ptr_hex": "0x6000"},
            "d": {"device_ptr_hex": "0x7000"},
        },
        workspace=workspace,
        numeric_task_mode="unit_math_full_rmsnorm",
    )

    assert packet is not None
    assert packet[0].n == 65536
    assert packet[1].n == 65536
    assert list(packet[2].scalar_args)[:3] == [0.0, 65536.0, 2430976.0]
