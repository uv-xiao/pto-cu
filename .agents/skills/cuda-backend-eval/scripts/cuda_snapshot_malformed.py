#!/usr/bin/env python3
"""Capture malformed live C++ snapshot lowering coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cuda_persistent_smoke_impl.snapshot_malformed import run_cpp_snapshot_malformed_cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    rendered = json.dumps(
        run_cpp_snapshot_malformed_cases(n=args.n),
        indent=2,
        sort_keys=True,
    )
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
