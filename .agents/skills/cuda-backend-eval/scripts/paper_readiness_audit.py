#!/usr/bin/env python3
"""Generate paper-readiness audit data from benchmark-viewer JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from paper_readiness_audit_impl.builder import build_readiness_audit
from paper_readiness_audit_impl.io import load_json
from paper_readiness_audit_impl.io import repo_relative
from paper_readiness_audit_impl.io import write_output
from paper_readiness_audit_impl.paths import DEFAULT_ATTEMPTS
from paper_readiness_audit_impl.paths import DEFAULT_MATRIX
from paper_readiness_audit_impl.paths import DEFAULT_PROBES
from paper_readiness_audit_impl.paths import DEFAULT_RESULTS
from paper_readiness_audit_impl.paths import DEFAULT_RUN_READINESS
from paper_readiness_audit_impl.paths import DEFAULT_RUNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--probes", type=Path, default=DEFAULT_PROBES)
    parser.add_argument("--run-readiness", type=Path, default=DEFAULT_RUN_READINESS)
    parser.add_argument("--execution-attempts", type=Path, default=DEFAULT_ATTEMPTS)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = build_readiness_audit(
        matrix=load_json(args.matrix),
        runs=load_json(args.runs),
        probes=load_json(args.probes),
        run_readiness=load_json(args.run_readiness),
        execution_attempts=load_json(args.execution_attempts),
        results=load_json(args.results),
    )
    if args.output:
        write_output(args.output, audit)
        print(f"wrote {repo_relative(args.output)}")
    else:
        print(json.dumps(audit, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
