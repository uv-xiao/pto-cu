#!/usr/bin/env python3
"""Collect safe readiness probes for paper baseline source checkouts."""

from __future__ import annotations

import argparse
from pathlib import Path

from paper_baseline_probe_impl.artifact import collect_probe
from paper_baseline_probe_impl.artifact import collect_probe_artifact
from paper_baseline_probe_impl.checks import collect_check
from paper_baseline_probe_impl.commands import git_commit
from paper_baseline_probe_impl.commands import nvidia_smi
from paper_baseline_probe_impl.commands import nvcc_version
from paper_baseline_probe_impl.commands import run_command
from paper_baseline_probe_impl.io import load_json
from paper_baseline_probe_impl.io import write_json
from paper_baseline_probe_impl.paths import DEFAULT_BASELINES
from paper_baseline_probe_impl.paths import DEFAULT_PROBES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--probes", type=Path, default=DEFAULT_PROBES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--artifact-root",
        default="tmp/cuda-backend/paper-baselines/probes/",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = collect_probe_artifact(
        load_json(args.baselines),
        load_json(args.probes),
        artifact_root=args.artifact_root,
    )
    write_json(args.output, artifact)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
