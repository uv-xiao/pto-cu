#!/usr/bin/env python3
"""Materialize isolated environment plans for paper serving baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

from paper_baseline_environment_plan_impl.builder import build_environment_plan
from paper_baseline_environment_plan_impl.builder import build_environment_plans
from paper_baseline_environment_plan_impl.dependency import dependency_evidence
from paper_baseline_environment_plan_impl.dependency import fallback_toml_dependencies
from paper_baseline_environment_plan_impl.dependency import package_name
from paper_baseline_environment_plan_impl.dependency import requirements_dependencies
from paper_baseline_environment_plan_impl.dependency import strip_inline_comment
from paper_baseline_environment_plan_impl.dependency import toml_dependencies
from paper_baseline_environment_plan_impl.errors import fail
from paper_baseline_environment_plan_impl.io import load_json
from paper_baseline_environment_plan_impl.io import write_json
from paper_baseline_environment_plan_impl.paths import DEFAULT_BASELINES
from paper_baseline_environment_plan_impl.paths import DEFAULT_OUTPUT_ROOT
from paper_baseline_environment_plan_impl.paths import DEFAULT_VIEWER_OUTPUT
from paper_baseline_environment_plan_impl.paths import repo_relative
from paper_baseline_environment_plan_impl.specs import ENVIRONMENT_SPECS
from paper_baseline_environment_plan_impl.vcs import git_commit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--viewer-output", type=Path, default=DEFAULT_VIEWER_OUTPUT)
    parser.add_argument("--commit", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commit = args.commit or git_commit()
    output_root = args.output_root
    payload = build_environment_plans(
        baselines=load_json(args.baselines),
        output_root=output_root,
        commit=commit,
    )
    write_json(output_root / "environment-plans.json", payload)
    write_json(args.viewer_output, payload)
    print(f"wrote {repo_relative(output_root / 'environment-plans.json')}")


if __name__ == "__main__":
    main()
