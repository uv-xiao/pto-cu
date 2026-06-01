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
