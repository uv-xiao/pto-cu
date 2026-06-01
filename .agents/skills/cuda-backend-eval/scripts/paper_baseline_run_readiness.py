#!/usr/bin/env python3
"""Probe paper-baseline run readiness without executing long benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

from paper_baseline_run_readiness_impl.builder import build_run_readiness
from paper_baseline_run_readiness_impl.git_utils import git_commit
from paper_baseline_run_readiness_impl.io import load_json
from paper_baseline_run_readiness_impl.io import repo_relative
from paper_baseline_run_readiness_impl.io import write_json
from paper_baseline_run_readiness_impl.io import write_viewer_output
from paper_baseline_run_readiness_impl.paths import DEFAULT_BASELINES
from paper_baseline_run_readiness_impl.paths import DEFAULT_ENV_PLANS
from paper_baseline_run_readiness_impl.paths import DEFAULT_OUTPUT_ROOT
from paper_baseline_run_readiness_impl.paths import DEFAULT_PROBES
from paper_baseline_run_readiness_impl.paths import DEFAULT_RUNS
from paper_baseline_run_readiness_impl.paths import DEFAULT_VIEWER_OUTPUT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--probes", type=Path, default=DEFAULT_PROBES)
    parser.add_argument("--env-plans", type=Path, default=DEFAULT_ENV_PLANS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--viewer-output", type=Path, default=DEFAULT_VIEWER_OUTPUT)
    parser.add_argument("--commit", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commit = args.commit or git_commit()
    payload = build_run_readiness(
        baselines=load_json(args.baselines),
        runs=load_json(args.runs),
        probes=load_json(args.probes),
        env_plans=load_json(args.env_plans),
        output_root=args.output_root,
        commit=commit,
    )
    write_json(args.output_root / "run-readiness.json", payload)
    write_viewer_output(args.viewer_output, payload)
    print(f"wrote {repo_relative(args.output_root / 'run-readiness.json')}")


if __name__ == "__main__":
    main()
