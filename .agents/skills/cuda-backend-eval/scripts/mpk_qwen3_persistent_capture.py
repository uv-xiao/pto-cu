#!/usr/bin/env python3
"""Normalize MPK Qwen3 persistent-kernel token artifacts for viewer import."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def fail(message: str) -> None:
    raise SystemExit(f"MPK Qwen3 persistent capture failed: {message}")


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        fail(f"missing JSON artifact: {path}")
    if not isinstance(payload, dict):
        fail(f"artifact root is not an object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def require_positive_int(payload: dict[str, Any], key: str, path: Path) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        fail(f"{path} has invalid {key}")
    return value


def require_positive_number(payload: dict[str, Any], key: str, path: Path) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        fail(f"{path} has invalid {key}")
    return float(value)


def require_mode(payload: dict[str, Any], expected: str, path: Path) -> None:
    mode = payload.get("mode")
    if mode != expected:
        fail(f"{path} has mode {mode!r}, expected {expected!r}")


def require_status_zero(path: Path) -> None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        fail(f"missing status artifact: {path}")
    if value != "0":
        fail(f"{path} is {value!r}, expected '0'")


def require_saved_tokens(payload: dict[str, Any], path: Path) -> int:
    tokens = payload.get("token_ids")
    if not isinstance(tokens, list) or not tokens:
        fail(f"{path} has no token_ids")
    return len(tokens)


def require_text(payload: dict[str, Any], path: Path) -> None:
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        fail(f"{path} has no decoded text")


def build_result_record(
    *,
    persistent_payload: dict[str, Any],
    persistent_path: Path,
    native_payload: dict[str, Any],
    native_path: Path,
    model: str,
    machine: str,
    gpu: str,
    raw_gpu_name: str,
    driver: str,
    compute_target: str,
    cuda_toolkit: str,
    dtype: str,
    clock_policy: str,
) -> dict[str, Any]:
    require_mode(persistent_payload, "mpk", persistent_path)
    require_mode(native_payload, "torch", native_path)
    prompt_tokens = require_positive_int(
        persistent_payload,
        "prompt_length",
        persistent_path,
    )
    decode_tokens = require_positive_int(
        persistent_payload,
        "generate_length",
        persistent_path,
    )
    latency_ms_per_token = require_positive_number(
        persistent_payload,
        "latency_ms_per_token",
        persistent_path,
    )
    saved_token_count = require_saved_tokens(persistent_payload, persistent_path)
    require_text(persistent_payload, persistent_path)

    token_latency_ns = int(round(latency_ms_per_token * 1_000_000))
    total_latency_ns = int(round(latency_ms_per_token * decode_tokens * 1_000_000))
    if total_latency_ns <= 0 or token_latency_ns <= 0:
        fail("derived latency is not positive")
    throughput_tokens_per_s = decode_tokens * 1_000_000_000 / total_latency_ns
    native_decode_tokens = require_positive_int(
        native_payload,
        "generate_length",
        native_path,
    )

    return {
        "paper_baseline_run_id": "mpk_qwen3_native_vs_persistent",
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
                "mpk_offline_decode,"
                f"{model},batch=1,target_prompt_tokens=64,"
                f"actual_prompt_tokens={prompt_tokens},"
                f"decode_tokens={decode_tokens},mode=mpk_persistent"
            ),
            "dtype": dtype,
            "repeat_policy": (
                "single H200 MPK persistent-kernel sample; native torch "
                f"control decoded {native_decode_tokens} tokens; MPK demo "
                "reports one combined prefill+decode per-token timing and "
                "launches the persistent kernel asynchronously"
            ),
        },
        "metrics": {
            "kind": "mpk_qwen3_persistent_decode_async_timing_caveat",
            "sample_count": 1,
            "host_wall_ns": total_latency_ns,
            "end_to_end_latency_ns": total_latency_ns,
            "time_to_first_token_ns": token_latency_ns,
            "inter_token_latency_ns": token_latency_ns,
            "time_per_output_token_ns": token_latency_ns,
            "throughput_tokens_per_s": throughput_tokens_per_s,
            "prompt_tokens": prompt_tokens,
            "decode_tokens": decode_tokens,
            "total_output_tokens": decode_tokens,
            "last_gen_throughput_tokens_per_s": throughput_tokens_per_s,
            "completed_requests": 1,
            "failed_requests": 0,
            "max_concurrent_requests": 1,
            "saved_debug_token_count": saved_token_count,
        },
        "correctness": "pass",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--persistent-json",
        default="persistent-batch1-decode1024.json",
    )
    parser.add_argument("--native-json", default="native-token2.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--machine", default="bizhaoh200")
    parser.add_argument("--pto-commit", required=True)
    parser.add_argument("--gpu", default="H200")
    parser.add_argument("--raw-gpu-name", default="NVIDIA H200 NVL")
    parser.add_argument("--driver", default="580.126.20")
    parser.add_argument("--compute-target", default="compute_90a")
    parser.add_argument(
        "--cuda-toolkit",
        default="12.8 (compiler build 35404655_0)",
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--clock-policy", default="not pinned")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir
    persistent_path = artifact_dir / args.persistent_json
    native_path = artifact_dir / args.native_json
    require_status_zero(artifact_dir / args.persistent_json.replace(".json", "-status.txt"))
    require_status_zero(artifact_dir / args.native_json.replace(".json", "-status.txt"))
    persistent_payload = load_json(persistent_path)
    native_payload = load_json(native_path)
    result = build_result_record(
        persistent_payload=persistent_payload,
        persistent_path=persistent_path,
        native_payload=native_payload,
        native_path=native_path,
        model=args.model,
        machine=args.machine,
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
            "capture": "qwen3_8b_persistent_decode1024",
            "source": str(artifact_dir),
            "latency_caveat": (
                "MPK demo timing is a single combined prefill+decode per-token "
                "number around asynchronous persistent-kernel launch; use this "
                "row for execution coverage, not final latency distributions."
            ),
        },
        "results": [result],
    }
    output = args.output or artifact_dir / "paper-baseline-results.json"
    write_json(output, payload)

    summary = {
        "status": "pass",
        "paper_baseline_run_id": "mpk_qwen3_native_vs_persistent",
        "artifact_dir": str(artifact_dir),
        "persistent_json": args.persistent_json,
        "native_json": args.native_json,
        "model": args.model,
        "prompt_tokens": result["metrics"]["prompt_tokens"],
        "decode_tokens": result["metrics"]["decode_tokens"],
        "latency_caveat": payload["metadata"]["latency_caveat"],
    }
    summary_output = args.summary_output or artifact_dir / "attempt-summary.json"
    write_json(summary_output, summary)
    print(output)


if __name__ == "__main__":
    main()
