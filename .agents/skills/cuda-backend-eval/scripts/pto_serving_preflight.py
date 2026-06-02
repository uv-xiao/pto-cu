#!/usr/bin/env python3
"""Capture current PTO persistent-device full-serving readiness."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pto_serving_preflight_impl.build import build_preflight  # noqa: E402
from pto_serving_preflight_impl.constants import DEFAULT_OUTPUT  # noqa: E402
from pto_serving_preflight_impl.io_helpers import (  # noqa: E402
    repo_relative,
    write_json,
)
from pto_serving_preflight_impl.rows import (  # noqa: E402
    full_serving_qwen_row_status,
    full_serving_qwen_rows,
    row_workload_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_preflight()
    write_json(args.output, payload)
    print(repo_relative(args.output))


if __name__ == "__main__":
    main()
