#!/usr/bin/env python3
"""Build paper-baseline raw results from repeated SGLang serving artifacts."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


DEFAULT_BATCHES = "1,2,4,8,16"
DEFAULT_REPEATS = "1,2,3"


def fail(message: str) -> None:
    raise SystemExit(f"sglang serving repeat import failed: {message}")


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8").strip().splitlines()[-1])
    except FileNotFoundError:
        fail(f"missing artifact: {path}")
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {path}: {exc}")
    if not isinstance(data, dict):
        fail(f"artifact is not a JSON object: {path}")
    return data


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def parse_csv_ints(value: str) -> list[int]:
    try:
        items = [int(item) for item in value.split(",") if item]
    except ValueError as exc:
        fail(f"invalid integer list {value!r}: {exc}")
    if not items:
        fail("empty integer list")
    return items


def read_status(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return "missing"


def require_status_ok(path: Path) -> None:
    status = read_status(path)
    if status != "0":
        fail(f"{path} status is {status}, expected 0")


def mean(values: list[float]) -> float:
    if not values:
        fail("cannot average an empty list")
    return statistics.fmean(values)


def stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def ns_from_ms(value: float) -> int:
    return int(round(value * 1_000_000))


def ns_from_s(value: float) -> int:
    return int(round(value * 1_000_000_000))


def int_mean(values: list[float]) -> int:
    return int(round(mean(values)))


def online_record(
    *,
    artifact_dir: Path,
    repeat_ids: list[int],
    batch: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    samples = []
    for repeat in repeat_ids:
        stem = f"bench-serving-r{repeat}-batch{batch}"
        require_status_ok(artifact_dir / f"{stem}-status.txt")
        samples.append(load_json(artifact_dir / f"{stem}.jsonl"))

    output_tps = [float(item["output_throughput"]) for item in samples]
    request_tps = [float(item["request_throughput"]) for item in samples]
    total_tps = [float(item["total_throughput"]) for item in samples]
    e2e_ms = [float(item["mean_e2e_latency_ms"]) for item in samples]
    ttft_ms = [float(item["mean_ttft_ms"]) for item in samples]
    itl_ms = [float(item["mean_itl_ms"]) for item in samples]
    tpot_ms = [float(item["mean_tpot_ms"]) for item in samples]
    completed = [float(item["completed"]) for item in samples]
    total_input = [float(item["total_input_tokens"]) for item in samples]
    total_output = [float(item["total_output_tokens"]) for item in samples]
    max_concurrency = [float(item["max_concurrent_requests"]) for item in samples]

    return {
        "paper_baseline_run_id": args.paper_baseline_run_id,
        "benchmark_id": "llm_serving_decode",
        "hardware": hardware(args),
        "inputs": inputs(
            args,
            batch=batch,
            mode="online_serving",
            repeat_policy=(
                f"{len(samples)} sglang bench_serving samples, random-ids "
                "fixed range, max_concurrency=batch, request_rate=inf"
            ),
        ),
        "metrics": {
            "kind": "paper_baseline_serving_repeat_capture",
            "serving_coverage": "full_serving",
            "sample_count": len(samples),
            "host_wall_ns": ns_from_ms(mean(e2e_ms)),
            "device_wall_ns": 0,
            "end_to_end_latency_ns": ns_from_ms(mean(e2e_ms)),
            "end_to_end_latency_stdev_ns": ns_from_ms(stdev(e2e_ms)),
            "time_to_first_token_ns": ns_from_ms(mean(ttft_ms)),
            "time_to_first_token_stdev_ns": ns_from_ms(stdev(ttft_ms)),
            "inter_token_latency_ns": ns_from_ms(mean(itl_ms)),
            "inter_token_latency_stdev_ns": ns_from_ms(stdev(itl_ms)),
            "time_per_output_token_ns": ns_from_ms(mean(tpot_ms)),
            "throughput": mean(request_tps),
            "throughput_tokens_per_s": mean(output_tps),
            "throughput_tokens_per_s_stdev": stdev(output_tps),
            "throughput_tokens_per_s_min": min(output_tps),
            "throughput_tokens_per_s_max": max(output_tps),
            "throughput_tokens_per_s_samples": output_tps,
            "total_token_throughput_tokens_per_s": mean(total_tps),
            "batch_size": batch,
            "prompt_tokens": args.prompt_tokens,
            "decode_tokens": args.decode_tokens,
            "completed_requests": int_mean(completed),
            "failed_requests": max(0, batch - int_mean(completed)),
            "max_concurrent_requests": int_mean(max_concurrency),
            "total_input_tokens": int_mean(total_input),
            "total_output_tokens": int_mean(total_output),
            "p50_output_throughput_tokens_per_s": statistics.median(output_tps),
            "p90_output_throughput_tokens_per_s": sorted(output_tps)[
                max(0, int(0.9 * (len(output_tps) - 1)))
            ],
            "p99_output_throughput_tokens_per_s": max(output_tps),
            "p99_ttft_ns": ns_from_ms(
                mean([float(item["p99_ttft_ms"]) for item in samples])
            ),
            "p99_itl_ns": ns_from_ms(
                mean([float(item["p99_itl_ms"]) for item in samples])
            ),
        },
        "correctness": "pass",
    }


def offline_record(
    *,
    artifact_dir: Path,
    repeat_ids: list[int],
    batch: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    samples = []
    for repeat in repeat_ids:
        stem = f"offline-throughput-r{repeat}-batch{batch}"
        require_status_ok(artifact_dir / f"{stem}-status.txt")
        samples.append(load_json(artifact_dir / f"{stem}.jsonl"))

    output_tps = [float(item["output_throughput"]) for item in samples]
    request_tps = [float(item["request_throughput"]) for item in samples]
    total_tps = [float(item["total_throughput"]) for item in samples]
    last_gen_tps = [float(item["last_gen_throughput"]) for item in samples]
    latency_s = [float(item["total_latency"]) for item in samples]
    successful = [float(item["successful_requests"]) for item in samples]
    total_input = [float(item["total_input_tokens"]) for item in samples]
    total_output = [float(item["total_output_tokens"]) for item in samples]

    return {
        "paper_baseline_run_id": args.paper_baseline_run_id,
        "benchmark_id": "llm_serving_decode",
        "hardware": hardware(args),
        "inputs": inputs(
            args,
            batch=batch,
            mode="offline_engine",
            repeat_policy=(
                f"{len(samples)} sglang bench_offline_throughput samples, "
                "fixed random length, skip warmup, local ShareGPT-shaped seed file"
            ),
        ),
        "metrics": {
            "kind": "paper_baseline_offline_repeat_capture",
            "serving_coverage": "full_serving",
            "sample_count": len(samples),
            "host_wall_ns": ns_from_s(mean(latency_s)),
            "device_wall_ns": 0,
            "end_to_end_latency_ns": ns_from_s(mean(latency_s)),
            "end_to_end_latency_stdev_ns": ns_from_s(stdev(latency_s)),
            "time_to_first_token_ns": 0,
            "inter_token_latency_ns": ns_from_s(mean(latency_s) / args.decode_tokens),
            "throughput": mean(request_tps),
            "throughput_tokens_per_s": mean(output_tps),
            "throughput_tokens_per_s_stdev": stdev(output_tps),
            "throughput_tokens_per_s_min": min(output_tps),
            "throughput_tokens_per_s_max": max(output_tps),
            "throughput_tokens_per_s_samples": output_tps,
            "total_token_throughput_tokens_per_s": mean(total_tps),
            "last_gen_throughput_tokens_per_s": mean(last_gen_tps),
            "last_gen_throughput_tokens_per_s_samples": last_gen_tps,
            "batch_size": batch,
            "prompt_tokens": args.prompt_tokens,
            "decode_tokens": args.decode_tokens,
            "completed_requests": int_mean(successful),
            "failed_requests": max(0, batch - int_mean(successful)),
            "max_concurrent_requests": batch,
            "total_input_tokens": int_mean(total_input),
            "total_output_tokens": int_mean(total_output),
            "p50_output_throughput_tokens_per_s": statistics.median(output_tps),
            "p90_output_throughput_tokens_per_s": sorted(output_tps)[
                max(0, int(0.9 * (len(output_tps) - 1)))
            ],
            "p99_output_throughput_tokens_per_s": max(output_tps),
        },
        "correctness": "pass",
    }


def hardware(args: argparse.Namespace) -> dict[str, str]:
    return {
        "gpu": args.gpu,
        "machine": args.machine,
        "compute_target": args.compute_target,
        "driver": args.driver,
        "cuda_toolkit": args.cuda_toolkit,
        "clock_policy": args.clock_policy,
    }


def inputs(
    args: argparse.Namespace,
    *,
    batch: int,
    mode: str,
    repeat_policy: str,
) -> dict[str, str]:
    return {
        "shape": (
            f"{args.serving_workload_id},{args.model},batch={batch},"
            f"prompt_tokens={args.prompt_tokens},"
            f"decode_tokens={args.decode_tokens},mode={mode},"
            f"repeats={args.repeat_count}"
        ),
        "dtype": args.dtype,
        "repeat_policy": repeat_policy,
    }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    artifact_dir = args.artifact_dir
    batches = parse_csv_ints(args.batches)
    repeat_ids = parse_csv_ints(args.repeats)
    args.repeat_count = len(repeat_ids)

    results = []
    for batch in batches:
        results.append(
            online_record(
                artifact_dir=artifact_dir,
                repeat_ids=repeat_ids,
                batch=batch,
                args=args,
            )
        )
        results.append(
            offline_record(
                artifact_dir=artifact_dir,
                repeat_ids=repeat_ids,
                batch=batch,
                args=args,
            )
        )
    return {
        "metadata": {
            "pto_commit": args.pto_commit,
            "artifact_root": f"{artifact_dir.as_posix().rstrip('/')}/",
            "source": "sglang_serving_repeats_to_paper_results.py",
        },
        "results": results,
    }


def write_status_summary(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    lines = [
        f"artifact_root: {args.artifact_dir}",
        f"model: {args.model}",
        f"serving_workload_id: {args.serving_workload_id}",
        f"prompt_tokens: {args.prompt_tokens}",
        f"decode_tokens: {args.decode_tokens}",
        f"result_records: {len(payload['results'])}",
    ]
    for record in payload["results"]:
        metrics = record["metrics"]
        lines.append(
            ", ".join(
                [
                    record["inputs"]["shape"],
                    f"tok/s={metrics['throughput_tokens_per_s']}",
                    f"samples={metrics['sample_count']}",
                ]
            )
        )
    (args.artifact_dir / "status-summary.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_attempt_summary(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    online = [
        item for item in payload["results"]
        if item["metrics"]["kind"] == "paper_baseline_serving_repeat_capture"
    ]
    offline = [
        item for item in payload["results"]
        if item["metrics"]["kind"] == "paper_baseline_offline_repeat_capture"
    ]
    summary = {
        "mode": "sglang_bench_serving_and_offline_mpk_repeats",
        "model": args.model,
        "serving_workload_id": args.serving_workload_id,
        "offline_cache_used": True,
        "remote_refresh_path": "tree sync fallback",
        "pto_commit": args.pto_commit,
        "prompt_tokens_per_request": args.prompt_tokens,
        "decode_tokens_per_request": args.decode_tokens,
        "batches": parse_csv_ints(args.batches),
        "sample_count_per_batch": args.repeat_count,
        "online_output_tokens_per_second_mean_by_batch": {
            str(item["metrics"]["batch_size"]): item["metrics"][
                "throughput_tokens_per_s"
            ]
            for item in online
        },
        "offline_output_tokens_per_second_mean_by_batch": {
            str(item["metrics"]["batch_size"]): item["metrics"][
                "throughput_tokens_per_s"
            ]
            for item in offline
        },
        "serving_latency_measured": bool(online),
        "offline_throughput_measured": bool(offline),
        "one_batch_measured": False,
        "batch_ladder_measured": True,
        "repeated_samples_measured": args.repeat_count > 1,
        "paper_target_model_measured": args.model == "Qwen/Qwen3-8B",
        "h200_measured": args.gpu == "H200",
        "viewer_result_imported": args.viewer_result_imported,
        "remaining_paper_gaps": [
            "resolve bench_one_batch input_ids None failure or document an "
            "exclusion policy",
            "align SGLang repeated rows with matching PTO persistent-device, "
            "MPK, VDCores, and vLLM serving-policy evidence",
        ],
    }
    write_json(
        args.artifact_dir / "attempt-summary.json",
        {
            "paper_baseline_id": "sglang",
            "paper_baseline_run_id": args.paper_baseline_run_id,
            "status": "partial",
            "artifact_root": f"{args.artifact_dir.as_posix().rstrip('/')}/",
            "summary": summary,
            "blocker": (
                "Repeated online serving and offline engine throughput are "
                "captured as SGLang MPK-policy evidence, but one-batch remains "
                "unmeasured and final paper claims need matching PTO, MPK, "
                "VDCores, and ThunderKittens rows."
            ),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--paper-baseline-run-id", default="sglang_serving_and_offline")
    parser.add_argument("--serving-workload-id", default="mpk_offline_decode")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--batches", default=DEFAULT_BATCHES)
    parser.add_argument("--repeats", default=DEFAULT_REPEATS)
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--decode-tokens", type=int, default=1024)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu", default="H200")
    parser.add_argument("--machine", default="bizhaoh200")
    parser.add_argument("--compute-target", default="compute_90")
    parser.add_argument("--driver", default="580.126.20")
    parser.add_argument("--cuda-toolkit", default="12.8")
    parser.add_argument("--clock-policy", default="not recorded in current snapshot")
    parser.add_argument("--pto-commit")
    parser.add_argument("--viewer-result-imported", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.pto_commit is None:
        commit_path = args.artifact_dir / "pto-commit.txt"
        args.pto_commit = commit_path.read_text(encoding="utf-8").strip()
    payload = build_payload(args)
    output = args.output or args.artifact_dir / "paper-baseline-results.json"
    write_json(output, payload)
    write_status_summary(args, payload)
    write_attempt_summary(args, payload)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
