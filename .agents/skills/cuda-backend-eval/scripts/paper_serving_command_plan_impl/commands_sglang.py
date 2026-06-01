"""Command builder for the SGLang serving baseline."""

from __future__ import annotations

from typing import Any

from paper_serving_command_plan_impl.shell import shell_join


def sglang_commands(
    *,
    model: str,
    policy: dict[str, Any],
    batch_size: int,
    out_dir: str,
) -> list[dict[str, str]]:
    prompt_tokens = policy["prompt_policy"]["target_prompt_tokens"]
    decode_tokens = policy["decode_policy"]["decode_tokens"]
    max_context = max(prompt_tokens + decode_tokens, 256)
    serving_json = f"{out_dir}/sglang-serving-batch{batch_size}.jsonl"
    offline_json = f"{out_dir}/sglang-offline-batch{batch_size}.json"
    one_batch_json = f"{out_dir}/sglang-one-batch{batch_size}.json"
    source_env = "PYTHONPATH=$PWD/tmp/baselines/sglang/python:$PYTHONPATH"
    return [
        _sglang_server(model, max_context, source_env),
        _sglang_online(
            model,
            batch_size,
            prompt_tokens,
            decode_tokens,
            source_env,
            serving_json,
        ),
        _sglang_offline(
            model,
            batch_size,
            prompt_tokens,
            decode_tokens,
            max_context,
            source_env,
            offline_json,
        ),
        _sglang_one_batch(
            model,
            batch_size,
            prompt_tokens,
            decode_tokens,
            max_context,
            source_env,
            one_batch_json,
        ),
    ]


def _sglang_server(
    model: str,
    max_context: int,
    source_env: str,
) -> dict[str, str]:
    return {
        "kind": "server",
        "command": shell_join(
            [
                "env",
                source_env,
                "python",
                "-m",
                "sglang.launch_server",
                "--model-path",
                model,
                "--port",
                "30000",
                "--context-length",
                str(max_context),
                "--disable-piecewise-cuda-graph",
            ]
        ),
    }


def _sglang_online(
    model: str,
    batch_size: int,
    prompt_tokens: int,
    decode_tokens: int,
    source_env: str,
    serving_json: str,
) -> dict[str, str]:
    return {
        "kind": "online_serving",
        "command": shell_join(
            [
                "env",
                source_env,
                "python",
                "-m",
                "sglang.bench_serving",
                "--backend",
                "sglang",
                "--model",
                model,
                "--host",
                "127.0.0.1",
                "--port",
                "30000",
                "--dataset-name",
                "random-ids",
                "--tokenize-prompt",
                "--random-input-len",
                str(prompt_tokens),
                "--random-output-len",
                str(decode_tokens),
                "--random-range-ratio",
                "1.0",
                "--num-prompts",
                str(batch_size),
                "--max-concurrency",
                str(batch_size),
                "--request-rate",
                "inf",
                "--output-file",
                serving_json,
            ]
        ),
        "raw_artifact": serving_json,
    }


def _sglang_offline(
    model: str,
    batch_size: int,
    prompt_tokens: int,
    decode_tokens: int,
    max_context: int,
    source_env: str,
    offline_json: str,
) -> dict[str, str]:
    return {
        "kind": "offline_throughput",
        "command": shell_join(
            [
                "env",
                source_env,
                "python",
                "-m",
                "sglang.bench_offline_throughput",
                "--model-path",
                model,
                "--context-length",
                str(max(max_context, 384)),
                "--disable-piecewise-cuda-graph",
                "--dataset-name",
                "random",
                "--random-input-len",
                str(prompt_tokens),
                "--random-output-len",
                str(decode_tokens),
                "--random-range-ratio",
                "1.0",
                "--num-prompts",
                str(batch_size),
                "--skip-warmup",
                "--result-filename",
                offline_json,
            ]
        ),
        "raw_artifact": offline_json,
    }


def _sglang_one_batch(
    model: str,
    batch_size: int,
    prompt_tokens: int,
    decode_tokens: int,
    max_context: int,
    source_env: str,
    one_batch_json: str,
) -> dict[str, str]:
    return {
        "kind": "one_batch",
        "command": shell_join(
            [
                "env",
                source_env,
                "python",
                "-m",
                "sglang.bench_one_batch",
                "--model-path",
                model,
                "--context-length",
                str(max_context),
                "--disable-piecewise-cuda-graph",
                "--disable-cuda-graph",
                "--batch-size",
                str(batch_size),
                "--input-len",
                str(prompt_tokens),
                "--output-len",
                str(decode_tokens),
                "--result-filename",
                one_batch_json,
            ]
        ),
        "raw_artifact": one_batch_json,
    }
