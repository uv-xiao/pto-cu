import ctypes
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def load_decode_feedback_module():
    sys.path.insert(0, str(ROOT / "examples" / "cuda"))
    script_path = (
        ROOT
        / "examples"
        / "cuda"
        / "qwen_decode_loop_runner_impl"
        / "decode_feedback.py"
    )
    spec = importlib.util.spec_from_file_location(
        "qwen_decode_feedback_test",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeRuntime:
    def __init__(self):
        self.memory = bytearray(128)

    def copy_to_device_ctx(self, ctx, ptr, src, size):
        del ctx
        value = ctypes.cast(src, ctypes.POINTER(ctypes.c_int32)).contents.value
        self.memory[ptr : ptr + size] = int(value).to_bytes(
            size,
            "little",
            signed=True,
        )
        return 0

    def copy_from_device_ctx(self, ctx, dst, ptr, size):
        del ctx
        value = int.from_bytes(self.memory[ptr : ptr + size], "little", signed=True)
        ctypes.cast(dst, ctypes.POINTER(ctypes.c_int32)).contents.value = value
        return 0


class FakeSession:
    def __init__(self):
        self.ctx = object()
        self.runtime = FakeRuntime()


def test_decode_feedback_commits_sampled_token_to_output_and_next_input():
    module = load_decode_feedback_module()
    session = FakeSession()

    result = module.apply_decode_feedback(
        session=session,
        token_fields={
            "a": {"device_ptr_hex": "0x0"},
            "out": {"device_ptr_hex": "0x40"},
        },
        decode_step_index=2,
        logits_summary={"topk": [{"token_id": 17, "logit": 2.5}]},
    )

    assert result == {
        "status": "feedback_applied",
        "sampled_token_id": 17,
        "output_ids_index": 2,
        "output_ids_value": 17,
        "next_input_index": 0,
        "next_input_value": 17,
        "policy": "host_commits_diagnostic_sampled_token_for_next_step",
    }
    assert module.feedback_summary([{"decode_feedback": result}]) == {
        "status": "diagnostic_token_feedback_applied",
        "applied_step_count": 1,
        "step_count": 1,
        "sampled_token_ids": [17],
        "policy": "host_commits_diagnostic_sampled_token_for_next_step",
    }
