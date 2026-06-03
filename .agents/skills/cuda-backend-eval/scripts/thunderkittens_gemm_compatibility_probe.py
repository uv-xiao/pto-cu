#!/usr/bin/env python3
"""Inspect ThunderKittens GEMM entrypoints against Qwen tensor target tiles."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, NamedTuple


BF16_GEMM_REL = "kernels/gemm/bf16_h100/bf16_h100_gemm.cu"
INT8_GEMM_REL = "kernels/gemm/int8_h100/int8_h100_gemm.cu"


class TensorTileTarget(NamedTuple):
    id: str
    rows: int
    cols: int
    inner: int


DEFAULT_TARGETS = [
    TensorTileTarget(
        id="qwen_attention_projection_tile",
        rows=16,
        cols=64,
        inner=128,
    ),
    TensorTileTarget(
        id="qwen_mlp_projection_tile",
        rows=16,
        cols=64,
        inner=256,
    ),
]


def _read_text(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing ThunderKittens source: {path}")
    return path.read_text(encoding="utf-8")


def _parse_int(pattern: str, source: str, field: str) -> int:
    match = re.search(pattern, source, re.MULTILINE)
    if match is None:
        raise ValueError(f"could not parse {field}")
    return int(match.group(1))


def _target_payload(target: TensorTileTarget) -> dict[str, Any]:
    return {
        "id": target.id,
        "tensor_tile": {
            "rows": target.rows,
            "cols": target.cols,
            "inner": target.inner,
        },
    }


def _bf16_reason(
    target: TensorTileTarget,
    *,
    base_rows: int,
    base_cols: int,
    block_rows: int,
    block_cols: int,
) -> str:
    reasons = []
    if target.rows < base_rows:
        reasons.append(
            f"target rows {target.rows} are smaller than BF16 base tile rows "
            f"{base_rows}"
        )
    if target.cols < base_cols:
        reasons.append(
            f"target cols {target.cols} are smaller than BF16 base tile cols "
            f"{base_cols}"
        )
    if target.rows < block_rows:
        reasons.append(
            f"target rows {target.rows} are smaller than default output block "
            f"rows {block_rows}"
        )
    if target.cols < block_cols:
        reasons.append(
            f"target cols {target.cols} are smaller than default output block "
            f"cols {block_cols}"
        )
    if target.inner % base_cols != 0:
        reasons.append(
            f"target inner {target.inner} is not divisible by BF16 K tile "
            f"{base_cols}"
        )
    if not reasons:
        return "target is compatible with the parsed BF16 GEMM entrypoint"
    return "; ".join(reasons)


def analyze_bf16_h100_source(
    source: str,
    targets: list[TensorTileTarget],
) -> dict[str, Any]:
    base_match = re.search(r"base_tile\s*=\s*st_bf<\s*(\d+)\s*,\s*(\d+)\s*>", source)
    if base_match is None:
        raise ValueError("could not parse BF16 base tile")
    base_rows = int(base_match.group(1))
    base_cols = int(base_match.group(2))
    m_block = _parse_int(r"_M_BLOCK\s*=\s*(\d+)", source, "BF16 M_BLOCK")
    n_block = _parse_int(r"_N_BLOCK\s*=\s*(\d+)", source, "BF16 N_BLOCK")
    block_rows = base_rows * m_block
    block_cols = base_cols * n_block
    compatibility = []
    for target in targets:
        exact = (
            target.rows >= block_rows
            and target.cols >= block_cols
            and target.rows % block_rows == 0
            and target.cols % block_cols == 0
            and target.inner % base_cols == 0
        )
        compatibility.append(
            {
                "target_id": target.id,
                "exact_target_compatible": exact,
                "reason": _bf16_reason(
                    target,
                    base_rows=base_rows,
                    base_cols=base_cols,
                    block_rows=block_rows,
                    block_cols=block_cols,
                ),
            }
        )
    return {
        "entrypoint_id": "bf16_h100_gemm",
        "source_path": BF16_GEMM_REL,
        "dtype": "bf16",
        "base_tile": {"rows": base_rows, "cols": base_cols},
        "default_m_block": m_block,
        "default_n_block": n_block,
        "default_output_block": {"rows": block_rows, "cols": block_cols},
        "comparability_scope": "same_dtype_family_but_entrypoint_tile_mismatch",
        "target_compatibility": compatibility,
    }


def analyze_int8_h100_source(
    source: str,
    targets: list[TensorTileTarget],
) -> dict[str, Any]:
    mb = _parse_int(r"_Mb\s*==\s*(\d+)", source, "INT8 Mb static assertion")
    nb_min = _parse_int(r"_Nb\s*>=\s*(\d+)", source, "INT8 Nb minimum")
    nb_max = _parse_int(r"_Nb\s*<=\s*(\d+)", source, "INT8 Nb maximum")
    nb_multiple = _parse_int(r"_Nb\s*%\s*(\d+)\s*==\s*0", source, "INT8 Nb multiple")
    kb_min = _parse_int(r"_Kb\s*>=\s*(\d+)", source, "INT8 Kb minimum")
    kb_multiple = _parse_int(r"_Kb\s*%\s*(\d+)\s*==\s*0", source, "INT8 Kb multiple")
    compatibility = []
    for target in targets:
        compatibility.append(
            {
                "target_id": target.id,
                "exact_target_compatible": False,
                "reason": (
                    "current Qwen tensor target claim is float/tensor-core "
                    f"comparator evidence, while this entrypoint is INT8 and "
                    f"requires Mb={mb}"
                ),
            }
        )
    return {
        "entrypoint_id": "int8_h100_gemm",
        "source_path": INT8_GEMM_REL,
        "dtype": "int8",
        "required_mb": mb,
        "nb_range": {"min": nb_min, "max": nb_max, "multiple": nb_multiple},
        "kb_constraints": {"min": kb_min, "multiple": kb_multiple},
        "comparability_scope": "dtype_mismatch_for_current_qwen_float_tensor_claim",
        "target_compatibility": compatibility,
    }


def build_compatibility_report(
    *,
    baseline_dir: str,
    bf16_source: str,
    int8_source: str,
    targets: list[TensorTileTarget],
) -> dict[str, Any]:
    entrypoints = [
        analyze_bf16_h100_source(bf16_source, targets),
        analyze_int8_h100_source(int8_source, targets),
    ]
    any_exact = any(
        item["exact_target_compatible"]
        for entrypoint in entrypoints
        for item in entrypoint["target_compatibility"]
    )
    status = (
        "source_entrypoint_supports_at_least_one_qwen_tile"
        if any_exact
        else "exact_qwen_tile_not_supported_by_current_gemm_entrypoints"
    )
    return {
        "schema_version": 1,
        "probe_id": "thunderkittens_qwen_gemm_compatibility",
        "baseline": "thunderkittens",
        "baseline_dir": baseline_dir,
        "status": status,
        "targets": [_target_payload(target) for target in targets],
        "entrypoints": entrypoints,
        "recommended_next_actions": [
            "capture a source-compatible ThunderKittens BF16 GEMM shape and label it as scaled comparator evidence",
            "or add a reviewed local wrapper experiment outside the upstream checkout for exact Qwen tiles",
            "or record a policy exception before closing the same-GEMM-tile comparator gap",
        ],
    }


def parse_target(value: str) -> TensorTileTarget:
    parts = value.split("=")
    if len(parts) != 2 or not parts[0]:
        raise argparse.ArgumentTypeError("target must be id=rows,cols,inner")
    dims = parts[1].split(",")
    if len(dims) != 3:
        raise argparse.ArgumentTypeError("target dimensions must be rows,cols,inner")
    try:
        rows, cols, inner = (int(item) for item in dims)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("target dimensions must be integers") from exc
    if min(rows, cols, inner) <= 0:
        raise argparse.ArgumentTypeError("target dimensions must be positive")
    return TensorTileTarget(parts[0], rows, cols, inner)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-dir",
        default="tmp/baselines/thunderkittens",
        help="ThunderKittens checkout root to inspect",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for compatibility JSON output",
    )
    parser.add_argument(
        "--target",
        action="append",
        type=parse_target,
        default=[],
        help="Target tile as id=rows,cols,inner; defaults to Qwen target tiles",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = Path(args.baseline_dir)
    targets = args.target or DEFAULT_TARGETS
    report = build_compatibility_report(
        baseline_dir=args.baseline_dir.rstrip("/") + "/",
        bf16_source=_read_text(baseline / BF16_GEMM_REL),
        int8_source=_read_text(baseline / INT8_GEMM_REL),
        targets=targets,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
