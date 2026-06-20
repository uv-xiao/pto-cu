#!/usr/bin/env python3
"""Two-request vLLM chat 256K synthetic needle exact-repeat probe."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BASE_PROBE_PATH = (
    ROOT / "examples" / "cuda" / "vllm_deepseek_v4_chat_256k_needle_exact_probe.py"
)
DEFAULT_PORT = 28_152
DEFAULT_EXPECTED_ANSWER = "PTO_CHAT_NEEDLE_256K_REPEAT_OK_28152"
DEFAULT_REPEAT_COUNT = 2
DEFAULT_STOP_SEQUENCES = ("\n```",)


def _load_base_probe():
    spec = importlib.util.spec_from_file_location(
        "vllm_deepseek_v4_chat_256k_needle_exact_probe",
        BASE_PROBE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"unable to load {BASE_PROBE_PATH}")
    spec.loader.exec_module(module)
    return module


base_probe = _load_base_probe()
health_probe = base_probe.health_probe

build_chat_needle_request = base_probe.build_chat_needle_request
send_chat_needle_request = base_probe.send_chat_needle_request
send_chat_needle_repeat_requests = base_probe.send_chat_needle_repeat_requests
aggregate_chat_repeat_attempts = base_probe.aggregate_chat_repeat_attempts
review_safe_request = base_probe.review_safe_request


def _default_stop_sequences(stop_sequences: list[str] | None) -> list[str]:
    if stop_sequences is None:
        return list(DEFAULT_STOP_SEQUENCES)
    return stop_sequences


def build_planned_chat_needle_request(**kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("expected_answer", DEFAULT_EXPECTED_ANSWER)
    kwargs["stop_sequences"] = _default_stop_sequences(kwargs.get("stop_sequences"))
    return base_probe.build_planned_chat_needle_request(**kwargs)


def run_probe(**kwargs: Any) -> dict[str, Any]:
    if kwargs.get("port") is None:
        kwargs["port"] = DEFAULT_PORT
    if kwargs.get("server_log") is None:
        kwargs["server_log"] = (
            ROOT
            / "tmp"
            / "vllm-chat-256k-needle-repeat-probe"
            / f"server-{kwargs['port']}.log"
        )
    kwargs.setdefault("expected_answer", DEFAULT_EXPECTED_ANSWER)
    kwargs["stop_sequences"] = _default_stop_sequences(kwargs.get("stop_sequences"))
    kwargs.setdefault("repeat_count", DEFAULT_REPEAT_COUNT)
    return base_probe.run_probe(**kwargs)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=health_probe.DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--vllm-bin", type=Path, default=health_probe.DEFAULT_VLLM_BIN)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--server-log", type=Path, default=None)
    parser.add_argument("--served-model-name", default=health_probe.DEFAULT_SERVED_MODEL_NAME)
    parser.add_argument("--max-model-len", type=int, default=base_probe.DEFAULT_MAX_MODEL_LEN)
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=health_probe.DEFAULT_TENSOR_PARALLEL_SIZE,
    )
    parser.add_argument("--dtype", default=health_probe.DEFAULT_DTYPE)
    parser.add_argument("--quantization", default=health_probe.DEFAULT_QUANTIZATION)
    parser.add_argument("--kv-cache-dtype", default=health_probe.DEFAULT_KV_CACHE_DTYPE)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=health_probe.DEFAULT_GPU_MEMORY_UTILIZATION,
    )
    parser.add_argument(
        "--distributed-executor-backend",
        default=health_probe.DEFAULT_DISTRIBUTED_EXECUTOR_BACKEND,
    )
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=health_probe.DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=health_probe.DEFAULT_POLL_INTERVAL_SECONDS,
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=base_probe.DEFAULT_REQUEST_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--terminate-timeout-seconds",
        type=float,
        default=health_probe.DEFAULT_TERMINATE_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--target-prompt-tokens",
        type=int,
        default=base_probe.DEFAULT_TARGET_PROMPT_TOKENS,
    )
    parser.add_argument("--max-tokens", type=int, default=base_probe.DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=base_probe.DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=base_probe.DEFAULT_TOP_P)
    parser.add_argument("--seed", type=int, default=base_probe.DEFAULT_SEED)
    parser.add_argument("--expected-answer", default=DEFAULT_EXPECTED_ANSWER)
    parser.add_argument(
        "--stop-sequence",
        action="append",
        dest="stop_sequences",
        default=None,
    )
    parser.add_argument(
        "--needle-position",
        choices=base_probe.NEEDLE_POSITIONS,
        default=base_probe.DEFAULT_NEEDLE_POSITION,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_probe(
        artifact_dir=args.artifact_dir,
        vllm_bin=args.vllm_bin,
        port=args.port,
        server_log=args.server_log,
        served_model_name=args.served_model_name,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        quantization=args.quantization,
        kv_cache_dtype=args.kv_cache_dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        distributed_executor_backend=args.distributed_executor_backend,
        enforce_eager=args.enforce_eager,
        trust_remote_code=args.trust_remote_code,
        timeout_seconds=args.timeout_seconds,
        poll_interval_seconds=args.poll_interval_seconds,
        request_timeout_seconds=args.request_timeout_seconds,
        terminate_timeout_seconds=args.terminate_timeout_seconds,
        target_prompt_tokens=args.target_prompt_tokens,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        expected_answer=args.expected_answer,
        stop_sequences=args.stop_sequences,
        needle_position=args.needle_position,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] == "failed":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
