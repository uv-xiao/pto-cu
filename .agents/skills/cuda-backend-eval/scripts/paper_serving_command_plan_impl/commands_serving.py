"""Command builders for serving framework baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from paper_serving_command_plan_impl.commands_sglang import sglang_commands
from paper_serving_command_plan_impl.shell import shell_join


def vllm_commands(
    *,
    model: str,
    policy: dict[str, Any],
    batch_size: int,
    out_dir: str,
) -> list[dict[str, str]]:
    prompt_tokens = policy["prompt_policy"]["target_prompt_tokens"]
    decode_tokens = policy["decode_policy"]["decode_tokens"]
    max_model_len = prompt_tokens + decode_tokens
    serve_json = f"{out_dir}/vllm-serve-batch{batch_size}.json"
    throughput_json = f"{out_dir}/vllm-throughput-batch{batch_size}.json"
    return [
        {
            "kind": "server",
            "command": shell_join(
                [
                    "vllm",
                    "serve",
                    model,
                    "--port",
                    "8000",
                    "--served-model-name",
                    model,
                    "--max-model-len",
                    str(max_model_len),
                ]
            ),
        },
        {
            "kind": "online_serving",
            "command": shell_join(
                [
                    "vllm",
                    "bench",
                    "serve",
                    "--backend",
                    "vllm",
                    "--model",
                    model,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "8000",
                    "--dataset-name",
                    "random",
                    "--input-len",
                    str(prompt_tokens),
                    "--output-len",
                    str(decode_tokens),
                    "--num-prompts",
                    str(batch_size),
                    "--max-concurrency",
                    str(batch_size),
                    "--request-rate",
                    "inf",
                    "--ignore-eos",
                    "--temperature",
                    "0",
                    "--save-result",
                    "--result-dir",
                    out_dir,
                    "--result-filename",
                    Path(serve_json).name,
                ]
            ),
            "raw_artifact": serve_json,
        },
        {
            "kind": "offline_throughput",
            "command": shell_join(
                [
                    "vllm",
                    "bench",
                    "throughput",
                    "--model",
                    model,
                    "--dataset-name",
                    "random",
                    "--input-len",
                    str(prompt_tokens),
                    "--output-len",
                    str(decode_tokens),
                    "--num-prompts",
                    str(batch_size),
                    "--num-warmups",
                    "1",
                    "--output-json",
                    throughput_json,
                ]
            ),
            "raw_artifact": throughput_json,
        },
    ]
