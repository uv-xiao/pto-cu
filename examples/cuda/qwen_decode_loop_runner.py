#!/usr/bin/env python3
"""Emit Qwen persistent decode-loop runner integration evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from qwen_decode_loop_runner_impl.lifecycle import (  # noqa: E402
    build_decode_loop_runner,
    repo_relative,
    write_json,
)


EVIDENCE_SYMBOLS = [
    "pto_qwen_decode_loop_runner",
    "decode_loop_owner_lifetime_order",
    "persistent_dag_submission_plan",
    "output_token_accounting_plan",
    "cuda_live_resource_bridge_contract",
    "qwen_unit_math_live_bridge_contract",
    "cuda_live_token_pointer_table_in_runner",
    "cuda_live_kv_cache_owner_in_runner",
    "cuda_live_resident_weight_table_in_runner",
    "cuda_live_activation_workspace_in_runner",
    "qwen_decode_loop_submission_descriptors",
    "qwen_decode_loop_submission_smoke_execution",
    "qwen_resource_backed_graph_materialization",
    "qwen_resource_backed_launch_packet_preflight",
    "qwen_activation_workspace_launch_packet_binding",
    "qwen_rope_table_launch_packet_binding",
    "qwen_kv_page_table_launch_packet_binding",
    "qwen_position_rope_table_population",
    "single_context_live_resource_session",
    "qwen_resource_backed_diagnostic_execution",
    "qwen_resource_backed_decode_step_execution",
    "qwen_resource_backed_policy_length_decode_execution",
    "qwen_diagnostic_decode_token_feedback",
    "qwen_device_decode_token_feedback",
    "qwen_resource_backed_unit_numeric_task_mode",
    "qwen_resource_backed_external_rmsnorm_scale",
    "qwen_resource_backed_full_rmsnorm_reduction",
    "qwen_resource_backed_model_equivalent_numeric_path",
    "qwen_resource_backed_weighted_elementwise_branches",
    "qwen_dynamic_rope_table_refresh",
    "qwen_resource_backed_projection_active_cols_override",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["offline", "download", "mock"],
        default="offline",
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--unit-math-live-json", type=Path)
    parser.add_argument("--run-unit-math-live", action="store_true")
    parser.add_argument("--token-cuda-live", action="store_true")
    parser.add_argument("--kv-cuda-live", action="store_true")
    parser.add_argument("--resident-cuda-live", action="store_true")
    parser.add_argument("--workspace-cuda-live", action="store_true")
    parser.add_argument("--single-context-live-session", action="store_true")
    parser.add_argument("--run-submission-smoke", action="store_true")
    parser.add_argument("--run-resource-backed-smoke", action="store_true")
    parser.add_argument("--resource-backed-repeat-runs", type=int, default=1)
    parser.add_argument("--resource-backed-decode-steps", type=int)
    parser.add_argument("--resource-backed-workload", action="append")
    parser.add_argument("--resource-backed-max-tasks", type=int)
    parser.add_argument(
        "--resource-backed-task-selection",
        choices=["prefix", "first_layer_with_logits", "layer_prefix_with_logits"],
        default="prefix",
    )
    parser.add_argument(
        "--resource-backed-layer-count",
        type=int,
        help=(
            "When task selection is layer_prefix_with_logits, include this many "
            "complete decoder layers plus embedding, final_norm, and logits."
        ),
    )
    parser.add_argument("--resource-backed-worker-blocks", type=int, default=1)
    parser.add_argument(
        "--resource-backed-logits-check-policy",
        choices=["every_step", "final_step"],
        default="every_step",
    )
    parser.add_argument(
        "--resource-backed-logits-active-cols",
        help=(
            "Override qwen_logits active vocab columns; use a positive integer "
            "or 'full'. Defaults to descriptor scalar1."
        ),
    )
    parser.add_argument(
        "--resource-backed-projection-active-cols",
        help=(
            "Override qwen_attention_qkv and MLP projection active columns; "
            "use a positive integer or 'full'. Defaults to descriptor scalar1."
        ),
    )
    parser.add_argument(
        "--resource-backed-numeric-task-mode",
        choices=[
            "diagnostic",
            "unit_math",
            "unit_math_full_rmsnorm",
            "model_equivalent",
        ],
        default="diagnostic",
    )
    parser.add_argument(
        "--resource-backed-prefill-prompt",
        action="store_true",
        help="Replay active prompt token positions into the live KV cache before decode.",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--arch", default="compute_80")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--host-runtime", type=Path)
    parser.add_argument("--build-runtime", action="store_true")
    parser.add_argument("--repeat-runs", type=int, default=3)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def load_unit_math_payload(args: argparse.Namespace) -> dict | None:
    if args.unit_math_live_json:
        return json.loads(args.unit_math_live_json.read_text(encoding="utf-8"))
    if not args.run_unit_math_live:
        return None
    from qwen_unit_math_live_impl.runner import run_unit_math_live

    return run_unit_math_live(
        device=args.device,
        arch=args.arch,
        cache_root=args.cache_root,
        build_runtime=args.build_runtime,
        repeat_runs=args.repeat_runs,
    )


def load_submission_smoke_payload(args: argparse.Namespace) -> dict | None:
    if not args.run_submission_smoke:
        return None
    from qwen_decode_loop_runner_impl.submission_smoke import run_submission_smoke

    return run_submission_smoke(
        device=args.device,
        arch=args.arch,
        cache_root=args.cache_root,
        build_runtime=args.build_runtime,
    )


def main() -> None:
    args = parse_args()
    payload = build_decode_loop_runner(
        mode=args.mode,
        cache_dir=args.cache_dir,
        unit_math_live_payload=load_unit_math_payload(args),
        token_cuda_live=args.token_cuda_live,
        kv_cuda_live=args.kv_cuda_live,
        resident_cuda_live=args.resident_cuda_live,
        device=args.device,
        host_runtime=args.host_runtime,
        submission_smoke_payload=load_submission_smoke_payload(args),
        workspace_cuda_live=args.workspace_cuda_live,
        single_context_live_session=args.single_context_live_session,
        run_resource_backed_smoke=args.run_resource_backed_smoke,
        resource_backed_repeat_runs=args.resource_backed_repeat_runs,
        resource_backed_decode_steps=args.resource_backed_decode_steps,
        resource_backed_workloads=args.resource_backed_workload,
        resource_backed_max_tasks=args.resource_backed_max_tasks,
        resource_backed_task_selection=args.resource_backed_task_selection,
        resource_backed_layer_count=args.resource_backed_layer_count,
        resource_backed_worker_blocks=args.resource_backed_worker_blocks,
        resource_backed_logits_check_policy=(
            args.resource_backed_logits_check_policy
        ),
        resource_backed_logits_active_cols=args.resource_backed_logits_active_cols,
        resource_backed_projection_active_cols=(
            args.resource_backed_projection_active_cols
        ),
        resource_backed_numeric_task_mode=args.resource_backed_numeric_task_mode,
        resource_backed_prefill_prompt=args.resource_backed_prefill_prompt,
        arch=args.arch,
    )
    if args.output_json:
        write_json(args.output_json, payload)
        print(repo_relative(args.output_json))
    else:
        print(json.dumps(payload, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
