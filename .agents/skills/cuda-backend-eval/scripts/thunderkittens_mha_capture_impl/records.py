"""Raw benchmark-result records for ThunderKittens MHA captures."""

from __future__ import annotations

from typing import Any


def normalized_gpu_name(gpu_metadata: dict[str, str]) -> str:
    gpu = gpu_metadata["gpu"]
    return "H200" if "H200" in gpu else gpu


def build_raw_result_record(
    *,
    paper_baseline_run_id: str,
    benchmark_id: str,
    machine: str,
    cuda_toolkit: str,
    clock_policy: str,
    gpu_metadata: dict[str, str],
    shape: dict[str, Any],
    latency: dict[str, Any],
    correctness: dict[str, Any],
    serving_workload_id: str = "",
    prompt_tokens: int = 0,
    decode_tokens: int = 0,
) -> dict[str, Any]:
    base_shape = (
        "mha_h100,"
        f"b={shape['b']},h={shape['h']},n={shape['n']},"
        f"d={shape['d']},causal={shape['causal']}"
    )
    metrics: dict[str, Any] = {
        "kind": "paper_baseline_capture",
        "sample_count": latency["sample_count"],
        "host_wall_ns": 0,
        "device_wall_ns": int(latency["p50_ns"]),
    }
    if serving_workload_id:
        if prompt_tokens <= 0 or decode_tokens <= 0:
            raise ValueError("serving rows require prompt and decode tokens")
        elapsed_ns = int(latency["p50_ns"])
        decoded_tokens = int(shape["b"]) * decode_tokens
        metrics = {
            "kind": "paper_baseline_serving_tile_capture",
            "serving_coverage": "controlled_attention_tile_proxy",
            "sample_count": latency["sample_count"],
            "host_wall_ns": elapsed_ns,
            "device_wall_ns": elapsed_ns,
            "end_to_end_latency_ns": elapsed_ns,
            "time_to_first_token_ns": elapsed_ns,
            "inter_token_latency_ns": elapsed_ns // decode_tokens,
            "throughput_tokens_per_s": int(decoded_tokens * 1_000_000_000 / elapsed_ns),
            "batch_size": int(shape["b"]),
            "prompt_tokens": prompt_tokens,
            "decode_tokens": decode_tokens,
        }
        base_shape = (
            f"{serving_workload_id},{base_shape},"
            f"prompt_tokens={prompt_tokens},decode_tokens={decode_tokens}"
        )
    return {
        "paper_baseline_run_id": paper_baseline_run_id,
        "benchmark_id": benchmark_id,
        "hardware": {
            "gpu": normalized_gpu_name(gpu_metadata),
            "machine": machine,
            "compute_target": gpu_metadata["compute_target"],
            "driver": gpu_metadata["driver"],
            "cuda_toolkit": cuda_toolkit,
            "clock_policy": clock_policy,
        },
        "inputs": {
            "shape": base_shape,
            "dtype": shape["dtype"],
            "repeat_policy": (
                f"{latency['warmup']} warmup, {latency['repeats']} timed "
                "CUDA-event repeats"
            ),
        },
        "metrics": metrics,
        "correctness": correctness["status"],
    }
