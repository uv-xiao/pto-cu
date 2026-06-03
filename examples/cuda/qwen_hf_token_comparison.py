#!/usr/bin/env python3
"""Compare PTO Qwen resource-backed logits with a Hugging Face reference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from qwen_decode_loop_runner_impl.hf_comparison import (  # noqa: E402
    build_hf_token_comparison,
    load_json,
    write_json,
)


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pto_artifact", type=Path)
    parser.add_argument("hf_reference", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison = build_hf_token_comparison(
        load_json(args.pto_artifact),
        load_json(args.hf_reference),
        pto_artifact=repo_relative(args.pto_artifact),
        hf_reference=repo_relative(args.hf_reference),
    )
    if args.output_json:
        write_json(args.output_json, comparison)
        print(repo_relative(args.output_json))
    else:
        print(json.dumps(comparison, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
