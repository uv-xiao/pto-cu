#!/usr/bin/env python3
"""Run and record bounded paper-baseline environment setup attempts."""

from __future__ import annotations

import argparse
from pathlib import Path

from paper_baseline_environment_attempt_impl.attempt import append_viewer_attempts
from paper_baseline_environment_attempt_impl.attempt import build_attempt
from paper_baseline_environment_attempt_impl.errors import fail
from paper_baseline_environment_attempt_impl.io import is_viewer_output
from paper_baseline_environment_attempt_impl.io import load_json
from paper_baseline_environment_attempt_impl.io import load_viewer_output
from paper_baseline_environment_attempt_impl.io import write_json
from paper_baseline_environment_attempt_impl.io import write_viewer_output
from paper_baseline_environment_attempt_impl.paths import DEFAULT_OUTPUT_ROOT
from paper_baseline_environment_attempt_impl.paths import DEFAULT_PLANS
from paper_baseline_environment_attempt_impl.paths import DEFAULT_VIEWER_OUTPUT
from paper_baseline_environment_attempt_impl.paths import repo_relative
from paper_baseline_environment_attempt_impl.plans import plan_for_baseline
from paper_baseline_environment_attempt_impl.runner import command_is_allowed
from paper_baseline_environment_attempt_impl.runner import run_step
from paper_baseline_environment_attempt_impl.vcs import git_commit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", type=Path, default=DEFAULT_PLANS)
    parser.add_argument("--baseline", required=True, choices=["vllm", "sglang"])
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--viewer-output", type=Path, default=DEFAULT_VIEWER_OUTPUT)
    parser.add_argument("--commit", default=None)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--start-step", type=int, default=1)
    parser.add_argument("--attempt-id-suffix", default="")
    parser.add_argument("--append-viewer", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commit = args.commit or git_commit()
    output_root = args.output_root
    if output_root is None:
        output_root = DEFAULT_OUTPUT_ROOT / f"{args.baseline}-{commit}"
    payload = build_attempt(
        plans=load_json(args.plans),
        baseline_id=args.baseline,
        output_root=output_root,
        commit=commit,
        max_steps=args.max_steps,
        start_step=args.start_step,
        timeout_seconds=args.timeout_seconds,
        attempt_id_suffix=args.attempt_id_suffix,
    )
    if args.append_viewer:
        payload = append_viewer_attempts(args.viewer_output, payload)
    write_viewer_output(args.viewer_output, payload)
    print(f"wrote {repo_relative(output_root / 'environment-attempt.json')}")


if __name__ == "__main__":
    main()
