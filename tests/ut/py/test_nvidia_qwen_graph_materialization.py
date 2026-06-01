import ctypes
import sys
from pathlib import Path

from simpler_setup.cuda_callable_compiler import CudaPersistentDagTask


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples" / "cuda"))

from qwen_decode_loop_runner_impl.launch_preflight import (  # noqa: E402
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
