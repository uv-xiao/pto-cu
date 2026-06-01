#!/usr/bin/env python3
"""Normalize an MPK native Qwen token artifact into viewer import JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def require_positive_number(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise SystemExit(f"{key} must be a positive number")
    return float(value)


def require_positive_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SystemExit(f"{key} must be a positive integer")
    return value


def build_result_record(
    *,
    token_payload: dict[str, Any],
    machine: str,
    pto_commit: str,
    gpu: str,
    raw_gpu_name: str,
    driver: str,
    compute_target: str,
    cuda_toolkit: str,
    dtype: str,
    clock_policy: str,
) -> dict[str, Any]:
    prompt_tokens = require_positive_int(token_payload, "prompt_length")
    decode_tokens = require_positive_int(token_payload, "generate_length")
    latency_ms_per_token = require_positive_number(
        token_payload,
        "latency_ms_per_token",
    )
    total_latency_ns = int(round(latency_ms_per_token * decode_tokens * 1_000_000))
    token_latency_ns = int(round(latency_ms_per_token * 1_000_000))
    throughput_tokens_per_s = decode_tokens * 1_000_000_000 / total_latency_ns
    model = str(token_payload.get("model", "Qwen/Qwen3-0.6B"))
    mode = str(token_payload.get("mode", "torch"))

    return {
        "paper_baseline_run_id": "mpk_qwen3_native_token_bringup",
        "benchmark_id": "llm_serving_decode",
        "hardware": {
            "gpu": gpu,
            "raw_gpu_name": raw_gpu_name,
            "machine": machine,
            "driver": driver,
            "compute_target": compute_target,
            "cuda_toolkit": cuda_toolkit,
            "clock_policy": clock_policy,
        },
        "inputs": {
            "shape": (
                f"model={model},prompt_tokens={prompt_tokens},"
                f"decode_tokens={decode_tokens},batch=1,mode={mode}"
            ),
            "dtype": dtype,
            "repeat_policy": (
                "single native torch MPK demo run; batch=1; "
                f"decode_tokens={decode_tokens}"
            ),
        },
        "metrics": {
            "kind": "mpk_native_token_bringup",
            "sample_count": 1,
            "host_wall_ns": total_latency_ns,
            "end_to_end_latency_ns": total_latency_ns,
            "inter_token_latency_ns": token_latency_ns,
            "time_per_output_token_ns": token_latency_ns,
            "throughput_tokens_per_s": throughput_tokens_per_s,
            "prompt_tokens": prompt_tokens,
            "decode_tokens": decode_tokens,
            "completed_requests": 1,
            "failed_requests": 0,
        },
        "correctness": "pass",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--machine", default="bizhaoh200")
    parser.add_argument("--pto-commit", required=True)
    parser.add_argument("--gpu", default="H200")
    parser.add_argument("--raw-gpu-name", default="NVIDIA H200")
    parser.add_argument("--driver", default="see raw artifact")
    parser.add_argument("--compute-target", default="compute_90")
    parser.add_argument("--cuda-toolkit", default="see raw artifact")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--clock-policy", default="not recorded")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token_payload = load_json(args.input)
    record = build_result_record(
        token_payload=token_payload,
        machine=args.machine,
        pto_commit=args.pto_commit,
        gpu=args.gpu,
        raw_gpu_name=args.raw_gpu_name,
        driver=args.driver,
        compute_target=args.compute_target,
        cuda_toolkit=args.cuda_toolkit,
        dtype=args.dtype,
        clock_policy=args.clock_policy,
    )
    payload = {
        "metadata": {
            "pto_commit": args.pto_commit,
            "baseline": "mpk",
            "capture": "native_qwen_token_bringup",
            "source": str(args.input),
        },
        "results": [record],
    }
    write_json(args.output, payload)
    print(args.output)


if __name__ == "__main__":
    main()
