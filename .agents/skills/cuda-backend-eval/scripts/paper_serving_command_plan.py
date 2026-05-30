#!/usr/bin/env python3
"""Generate paper-baseline serving commands from viewer workload policies."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
VIEWER_DATA = ROOT / "docs" / "nvidia-backend" / "benchmark-viewer" / "data"
DEFAULT_SERVING = VIEWER_DATA / "serving_workloads.json"
DEFAULT_RUNS = VIEWER_DATA / "paper_baseline_runs.json"


def fail(message: str) -> None:
    raise SystemExit(f"paper serving command plan failed: {message}")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip()


def shell_join(parts: list[str]) -> str:
    def quote(part: str) -> str:
        if not part:
            return "''"
        safe = set(
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789_+-=.,:/@%"
        )
        if all(char in safe for char in part):
            return part
        return "'" + part.replace("'", "'\"'\"'") + "'"

    return " ".join(quote(str(part)) for part in parts)


def selected_model(workload: dict[str, Any], tier: str) -> str:
    key = f"{tier}_model"
    model = workload["model_policy"].get(key)
    if not model:
        fail(f"{workload['id']} has no model tier {tier!r}")
    return str(model)


def artifact_dir(root: str, baseline_id: str, policy_id: str) -> str:
    return f"{root.rstrip('/')}/{baseline_id}/{policy_id}"


def path_from_cwd(repo_relative_path: str, cwd: str) -> str:
    return os.path.relpath(ROOT / repo_relative_path, ROOT / cwd)


def plan_id(run_id: str, policy_id: str, batch_size: int) -> str:
    return f"{run_id}:{policy_id}:batch{batch_size}"


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


def sglang_commands(
    *,
    model: str,
    policy: dict[str, Any],
    batch_size: int,
    out_dir: str,
) -> list[dict[str, str]]:
    prompt_tokens = policy["prompt_policy"]["target_prompt_tokens"]
    decode_tokens = policy["decode_policy"]["decode_tokens"]
    max_context = prompt_tokens + decode_tokens
    serving_json = f"{out_dir}/sglang-serving-batch{batch_size}.jsonl"
    offline_json = f"{out_dir}/sglang-offline-batch{batch_size}.json"
    one_batch_json = f"{out_dir}/sglang-one-batch{batch_size}.json"
    return [
        {
            "kind": "server",
            "command": shell_join(
                [
                    "python",
                    "-m",
                    "sglang.launch_server",
                    "--model-path",
                    model,
                    "--port",
                    "30000",
                    "--context-length",
                    str(max_context),
                ]
            ),
        },
        {
            "kind": "online_serving",
            "command": shell_join(
                [
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
                    "random",
                    "--random-input-len",
                    str(prompt_tokens),
                    "--random-output-len",
                    str(decode_tokens),
                    "--random-range-ratio",
                    "0",
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
        },
        {
            "kind": "offline_throughput",
            "command": shell_join(
                [
                    "python",
                    "-m",
                    "sglang.bench_offline_throughput",
                    "--model-path",
                    model,
                    "--dataset-name",
                    "random",
                    "--random-input-len",
                    str(prompt_tokens),
                    "--random-output-len",
                    str(decode_tokens),
                    "--random-range-ratio",
                    "0",
                    "--num-prompts",
                    str(batch_size),
                    "--result-filename",
                    offline_json,
                ]
            ),
            "raw_artifact": offline_json,
        },
        {
            "kind": "one_batch",
            "command": shell_join(
                [
                    "python",
                    "-m",
                    "sglang.bench_one_batch",
                    "--model-path",
                    model,
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
        },
    ]


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
    prompt = policy["prompt_policy"]["prompt_text"]
    raw_json = f"{out_dir}/vdcores-decode-batch{batch_size}.json"
    command = shell_join(
        [
            "python",
            "app/python/llama3/sched.py",
            "--prompt",
            prompt,
            "--max-decode-steps",
            str(decode_tokens),
            "-N",
            str(decode_tokens),
            "--bench",
            "1",
        ]
    )
    return [
        {
            "kind": "decode_benchmark",
            "command": "cd tmp/baselines/vdcores && " + command,
            "raw_artifact": raw_json,
            "note": (
                "Current VDCores demo path is Llama-only and has no batch-size "
                "CLI; batch_size is retained for policy tracking."
            ),
        }
    ]


COMMAND_BUILDERS = {
    "mpk": mpk_commands,
    "vdcores": vdcores_commands,
    "vllm": vllm_commands,
    "sglang": sglang_commands,
}


def build_plan(
    serving: dict[str, Any],
    runs: dict[str, Any],
    *,
    artifact_root: str,
    model_tier: str,
) -> dict[str, Any]:
    workloads = {item["id"]: item for item in serving["serving_workloads"]}
    records = []
    for run in runs["paper_baseline_runs"]:
        baseline_id = run["paper_baseline_id"]
        builder = COMMAND_BUILDERS.get(baseline_id)
        if builder is None:
            continue
        for policy_id in run.get("serving_workload_ids", []):
            policy = workloads.get(policy_id)
            if policy is None:
                fail(f"{run['id']} references unknown serving workload {policy_id}")
            model = selected_model(policy, model_tier)
            out_dir = artifact_dir(artifact_root, baseline_id, policy_id)
            for batch_size in policy["decode_policy"]["batch_sizes"]:
                commands = builder(
                    model=model,
                    policy=policy,
                    batch_size=int(batch_size),
                    out_dir=out_dir,
                )
                records.append(
                    {
                        "id": plan_id(run["id"], policy_id, int(batch_size)),
                        "paper_baseline_run_id": run["id"],
                        "paper_baseline_id": baseline_id,
                        "serving_workload_id": policy_id,
                        "model_tier": model_tier,
                        "model": model,
                        "batch_size": int(batch_size),
                        "prompt_tokens": policy["prompt_policy"][
                            "target_prompt_tokens"
                        ],
                        "decode_tokens": policy["decode_policy"]["decode_tokens"],
                        "traffic_mode": policy["decode_policy"]["traffic_mode"],
                        "commands": commands,
                    }
                )
    return {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "pto_commit": git_commit(),
            "artifact_root": artifact_root,
            "model_tier": model_tier,
            "source_files": [
                str(DEFAULT_SERVING.relative_to(ROOT)),
                str(DEFAULT_RUNS.relative_to(ROOT)),
            ],
        },
        "serving_command_plans": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serving-workloads", type=Path, default=DEFAULT_SERVING)
    parser.add_argument("--baseline-runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument(
        "--artifact-root",
        default="tmp/cuda-backend/paper-baselines/serving-runs",
    )
    parser.add_argument(
        "--model-tier",
        choices=["primary", "bringup", "fallback"],
        default="primary",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = build_plan(
        load_json(args.serving_workloads),
        load_json(args.baseline_runs),
        artifact_root=args.artifact_root,
        model_tier=args.model_tier,
    )
    write_json(args.output, plan)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
