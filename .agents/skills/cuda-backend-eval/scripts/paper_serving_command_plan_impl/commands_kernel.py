"""Command builders for persistent-kernel and kernel-family baselines."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

from paper_serving_command_plan_impl.paths import path_from_cwd
from paper_serving_command_plan_impl.shell import shell_join


def mpk_commands(
    *,
    model: str,
    policy: dict[str, Any],
    batch_size: int,
    out_dir: str,
) -> list[dict[str, str]]:
    decode_tokens = policy["decode_policy"]["decode_tokens"]
    prompt = policy["prompt_policy"]["prompt_text"]
    native_json = f"{out_dir}/mpk-native-batch{batch_size}.json"
    persistent_json = f"{out_dir}/mpk-persistent-batch{batch_size}.json"
    native_save = path_from_cwd(native_json, "tmp/baselines/mirage-mpk")
    persistent_save = path_from_cwd(
        persistent_json,
        "tmp/baselines/mirage-mpk",
    )
    common = [
        "python",
        "demo/qwen3/demo.py",
        "--model",
        model,
        "--prompt",
        prompt,
        "--max-new-tokens",
        str(decode_tokens),
        "--max-num-batched-requests",
        str(batch_size),
        "--max-num-batched-tokens",
        str(batch_size),
        "--ignore-eos",
        "--temperature",
        "0",
    ]
    return [
        {
            "kind": "native_demo",
            "command": "cd tmp/baselines/mirage-mpk && "
            + shell_join([*common, "--save-tokens", native_save]),
            "raw_artifact": native_json,
        },
        {
            "kind": "persistent_demo",
            "command": "cd tmp/baselines/mirage-mpk && "
            + shell_join(
                [*common, "--use-mirage", "--save-tokens", persistent_save]
            ),
            "raw_artifact": persistent_json,
        },
    ]


def vdcores_commands(
    *,
    model: str,
    policy: dict[str, Any],
    batch_size: int,
    out_dir: str,
) -> list[dict[str, str]]:
    del model
    decode_tokens = policy["decode_policy"]["decode_tokens"]
    raw_log = f"{out_dir}/vdcores-decode-batch{batch_size}.log"
    raw_log_path = path_from_cwd(raw_log, "tmp/baselines/vdcores")
    raw_log_dir = PurePosixPath(raw_log_path).parent.as_posix()
    command = shell_join(
        [
            "python",
            "app/python/qwen3/sched.py",
            "--hf-cache-dir",
            "<shared-hf-cache>/hub",
            "-N",
            str(decode_tokens),
            "--bench",
            str(batch_size),
        ]
    )
    return [
        {
            "kind": "decode_benchmark",
            "command": (
                "cd tmp/baselines/vdcores && "
                + shell_join(["mkdir", "-p", raw_log_dir])
                + " && HF_TOKEN= HF_HUB_OFFLINE=1 "
                "TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 "
            )
            + command
            + " 2>&1 | "
            + shell_join(["tee", raw_log_path]),
            "raw_artifact": raw_log,
            "note": (
                "Current VDCores Qwen3-8B path fixes part of the internal "
                "batch policy; batch_size remains the paper policy target "
                "until a full serving harness records actual scheduled "
                "request count. The raw artifact is a captured log until "
                "VDCores exposes machine-readable benchmark output."
            ),
        }
    ]


def thunderkittens_commands(
    *,
    model: str,
    policy: dict[str, Any],
    batch_size: int,
    out_dir: str,
) -> list[dict[str, str]]:
    del model
    prompt_tokens = policy["prompt_policy"]["target_prompt_tokens"]
    decode_tokens = policy["decode_policy"]["decode_tokens"]
    raw_json = f"{out_dir}/thunderkittens-mha-batch{batch_size}.json"
    kernel_sequence_tokens = max(int(decode_tokens), 256)
    shape = f"{batch_size},1,{kernel_sequence_tokens},64"
    command = shell_join(
        [
            ".venv/bin/python",
            ".agents/skills/cuda-backend-eval/scripts/"
            "thunderkittens_mha_capture.py",
            "--baseline-dir",
            "tmp/baselines/thunderkittens/kernels/attention/mha_h100",
            "--output",
            raw_json,
            "--machine",
            "<h200-host>",
            "--pto-commit",
            "<pto-commit>",
            "--cuda-toolkit",
            "12.8",
            "--paper-baseline-run-id",
            "thunderkittens_decode_attention_tile",
            "--benchmark-id",
            "llm_serving_decode",
            "--serving-workload-id",
            policy["id"],
            "--prompt-tokens",
            str(prompt_tokens),
            "--decode-tokens",
            str(decode_tokens),
            "--shape",
            shape,
            "--warmup",
            "5",
            "--repeats",
            "20",
            "--causal",
        ]
    )
    return [
        {
            "kind": "decode_attention_tile",
            "command": f"PYTHONPATH=$PWD:$PWD/python {command}",
            "raw_artifact": raw_json,
            "note": (
                "ThunderKittens is a serving-family kernel baseline here; "
                "the H100 MHA wrapper pads the 64-token decode policy to "
                "n=256 so the kernel launch grid is nonzero while preserving "
                "prompt/decode metadata for the VDCores policy batch ladder."
            ),
        }
    ]
