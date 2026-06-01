#!/usr/bin/env python3
"""Generate paper-baseline serving commands from viewer workload policies."""

from __future__ import annotations

import argparse
from pathlib import Path

from paper_serving_command_plan_impl.builder import COMMAND_BUILDERS
from paper_serving_command_plan_impl.builder import build_plan
from paper_serving_command_plan_impl.builder import filter_commands_for_run
from paper_serving_command_plan_impl.commands_kernel import mpk_commands
from paper_serving_command_plan_impl.commands_kernel import thunderkittens_commands
from paper_serving_command_plan_impl.commands_kernel import vdcores_commands
from paper_serving_command_plan_impl.commands_serving import sglang_commands
from paper_serving_command_plan_impl.commands_serving import vllm_commands
from paper_serving_command_plan_impl.errors import fail
from paper_serving_command_plan_impl.io import load_json
from paper_serving_command_plan_impl.io import write_json
from paper_serving_command_plan_impl.io import write_output
from paper_serving_command_plan_impl.paths import DEFAULT_RUNS
from paper_serving_command_plan_impl.paths import DEFAULT_SERVING
from paper_serving_command_plan_impl.paths import artifact_dir
from paper_serving_command_plan_impl.paths import path_from_cwd
from paper_serving_command_plan_impl.plan_ids import plan_id
from paper_serving_command_plan_impl.plan_ids import selected_model
from paper_serving_command_plan_impl.shell import shell_join
from paper_serving_command_plan_impl.vcs import git_commit


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
    write_output(args.output, plan)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
