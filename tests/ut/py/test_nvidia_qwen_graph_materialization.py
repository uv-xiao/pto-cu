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
)
from qwen_decode_loop_runner_impl.launch_helpers import (  # noqa: E402
    numeric_task_mode_summary,
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
    for index in range(2, 7):
        assert packet[index].scalar_arg_count == 1
        assert packet[index].scalar_args[0] == 1.0
    assert packet[7].scalar_arg_count == 3
    assert list(packet[7].scalar_args)[:3] == [0.0, 1.0, 1.0]


def test_launch_packet_can_select_full_rmsnorm_reduction_branch():
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
    assert packet[0].scalar_arg_count == 1
    assert list(packet[0].scalar_args)[:2] == [1.0, 0.0]
    assert packet[1].scalar_arg_count == 1
    assert packet[1].scalar_args[0] == 1.0
    summary = numeric_task_mode_summary("unit_math_full_rmsnorm")
    assert summary["scope"] == "resource_backed_unit_math_full_rmsnorm_reduction"
    assert summary["external_scale_contracts"] == []
    assert summary["full_reduction_contracts"] == [
        {
            "callable": "qwen_rmsnorm_input",
            "scalar_arg_count": 1,
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
