#!/usr/bin/env python3
"""Skip-safe correctness harness for generated Triton/Gluon FlashAttention."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import shlex
import sys
from pathlib import Path
from typing import Callable

from simpler_setup.gluon_gen import GluonKernelArtifact
from simpler_setup.kernel_compiler import KernelCompiler

DEFAULT_OUTPUT_DIR = Path("tmp/gluon-flashattention-local")
FLASHATTENTION_TILE = (32, 32, 32)
FLASHATTENTION_REFERENCE = "softmax((q @ k.T) * scale) @ v"
FLASHATTENTION_CAUSAL_REFERENCE = (
    "softmax(masked_fill((q @ k.T) * scale, key_index > query_index, -inf)) @ v"
)
FLASHATTENTION_CAUSAL_DECODE_REFERENCE = (
    "softmax(masked_fill((q @ k.T) * scale, "
    "key_index > query_index + (seqlen_k - seqlen_q), -inf)) @ v"
)
FLASHATTENTION_SWEEP_CASES = [
    {
        "name": "existing_32x32x32",
        "tile_shape": (32, 32, 32),
        "seed": 0,
        "provenance": "existing 32x32x32 correctness fixture",
    },
    {
        "name": "q16_k64_head_dim64",
        "tile_shape": (16, 64, 64),
        "seed": 1,
        "provenance": (
            "common serving attention head dimension; selected after "
            "32x32x64 failed H200 correctness"
        ),
    },
]
FLASHATTENTION_CAUSAL_PREFILL_SWEEP_CASES = [
    {
        "name": "prefill_16x16x64",
        "tile_shape": (16, 16, 64),
        "seed": 2,
        "provenance": "bounded same-length multi-query causal prefill fixture",
    },
    {
        "name": "prefill_32x32x64",
        "tile_shape": (32, 32, 64),
        "seed": 3,
        "provenance": "bounded same-length multi-query causal prefill H200 gate",
    },
]
FLASHATTENTION_CAUSAL_APPEND_SWEEP_CASES = [
    {
        "name": "append_4x32x64",
        "tile_shape": (4, 32, 64),
        "seed": 4,
        "provenance": "bounded multi-query causal append fixture",
    },
    {
        "name": "append_8x32x64",
        "tile_shape": (8, 32, 64),
        "seed": 5,
        "provenance": "bounded multi-query causal append H200 gate",
    },
]
FLASHATTENTION_CAUSAL_DECODE_SWEEP_CASES = [
    {
        "name": "decode_1x16x64",
        "tile_shape": (1, 16, 64),
        "seed": 6,
        "provenance": "bounded single-query causal decode fixture",
    },
    {
        "name": "decode_1x32x64",
        "tile_shape": (1, 32, 64),
        "seed": 7,
        "provenance": "bounded single-query causal decode H200 gate",
    },
]


def build_flashattention_artifact(
    *,
    output_dir: str | Path | None = None,
    arch: str = "compute_90",
    tile_shape: tuple[int, int, int] = FLASHATTENTION_TILE,
) -> GluonKernelArtifact:
    return KernelCompiler(platform="cuda").generate_gluon_kernel(
        "flashattention_fwd_f32",
        output_dir=DEFAULT_OUTPUT_DIR if output_dir is None else output_dir,
        arch=arch,
        tile_shape=tile_shape,
    )


def flashattention_skip_reason() -> str | None:
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            import torch
    except Exception as exc:  # pragma: no cover - depends on local environment
        return f"torch import failed: {_clean_text(str(exc))}"

    if not torch.cuda.is_available():
        return "torch.cuda is not available"

    try:
        import triton  # noqa: F401
        from triton.experimental import gluon  # noqa: F401
        from triton.experimental.gluon import language as gl
    except Exception as exc:  # pragma: no cover - depends on local environment
        return f"triton Gluon import failed: {_clean_text(str(exc))}"

    for required in ("dot_fma", "exp", "max", "sum", "where"):
        if not hasattr(gl, required):
            return f"triton Gluon import failed: missing gl.{required}"

    return None


def load_generated_module(source_path: str | Path):
    source_path = Path(source_path)
    spec = importlib.util.spec_from_file_location(
        f"pto_generated_{source_path.stem.replace('.', '_')}",
        source_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"could not load generated Gluon source: {_relative_path(source_path)}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_flashattention_correctness(
    *,
    output_dir: str | Path | None = None,
    arch: str = "compute_90",
    tile_shape: tuple[int, int, int] = FLASHATTENTION_TILE,
    atol: float = 1e-3,
    rtol: float = 1e-2,
    seed: int = 0,
    causal: bool = False,
    attention_variant: str = "standard",
    sequence_boundary: str = "fixed",
    kv_cache_boundary: str = "none",
    skip_reason: Callable[[], str | None] | None = None,
) -> dict:
    resolved_output_dir = DEFAULT_OUTPUT_DIR if output_dir is None else Path(output_dir)
    if resolved_output_dir.is_absolute():
        raise ValueError("--output-dir must be repo-relative")
    _validate_attention_variant(attention_variant)
    _validate_sequence_boundary(sequence_boundary)
    _validate_kv_cache_boundary(kv_cache_boundary)

    if attention_variant != "standard":
        return _unsupported_attention_variant_result(
            arch=arch,
            tile_shape=tile_shape,
            atol=atol,
            rtol=rtol,
            causal=causal,
            attention_variant=attention_variant,
        )

    if sequence_boundary != "fixed":
        return _unsupported_sequence_boundary_result(
            arch=arch,
            tile_shape=tile_shape,
            atol=atol,
            rtol=rtol,
            causal=causal,
            sequence_boundary=sequence_boundary,
        )

    if kv_cache_boundary != "none":
        return _unsupported_kv_cache_boundary_result(
            arch=arch,
            tile_shape=tile_shape,
            atol=atol,
            rtol=rtol,
            causal=causal,
            kv_cache_boundary=kv_cache_boundary,
        )

    artifact = build_flashattention_artifact(
        output_dir=resolved_output_dir,
        arch=arch,
        tile_shape=tile_shape,
    )
    seqlen_q, seqlen_k, head_dim = artifact.tile_shape
    phase = _phase_for(tile_shape=artifact.tile_shape, causal=causal)
    result = {
        "schema_version": 1,
        "kernel_name": "flashattention_fwd_f32",
        "phase": phase,
        "artifact": _artifact_payload(artifact),
        "shape": {
            "seqlen_q": seqlen_q,
            "seqlen_k": seqlen_k,
            "head_dim": head_dim,
        },
        "causal": causal,
        "reference": _reference_for(phase=phase, causal=causal),
        "tolerance": {"atol": atol, "rtol": rtol},
    }

    skip_check = flashattention_skip_reason if skip_reason is None else skip_reason
    reason = skip_check()
    if reason is not None:
        return {**result, "status": "skipped", "reason": _clean_text(reason)}

    try:
        import torch

        module = load_generated_module(artifact.source_path)
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed)
        q = torch.randn(
            (seqlen_q, head_dim),
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        k = torch.randn(
            (seqlen_k, head_dim),
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        v = torch.randn(
            (seqlen_k, head_dim),
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        out = torch.empty((seqlen_q, head_dim), device="cuda", dtype=torch.float32)

        scale = head_dim**-0.5
        module.run_flashattention_fwd_f32(
            q,
            k,
            v,
            out,
            scale=scale,
            causal=causal,
            num_warps=4,
        )
        torch.cuda.synchronize()

        scores = (q @ k.T) * scale
        if causal:
            query_index = torch.arange(seqlen_q, device="cuda")[:, None]
            key_index = torch.arange(seqlen_k, device="cuda")[None, :]
            if phase in ("decode", "append"):
                query_index = query_index + (seqlen_k - seqlen_q)
            scores = scores.masked_fill(key_index > query_index, float("-inf"))
        expected = torch.softmax(scores, dim=-1) @ v
        torch.cuda.synchronize()
        max_abs_error = float((out - expected).abs().max().item())
        passed = bool(torch.allclose(out, expected, atol=atol, rtol=rtol))
        return {
            **result,
            "status": "passed" if passed else "failed",
            "max_abs_error": max_abs_error,
        }
    except Exception as exc:
        return {
            **result,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": _clean_text(str(exc)),
        }


def run_flashattention_sweep(
    *,
    output_dir: str | Path | None = None,
    arch: str = "compute_90",
    atol: float = 1e-3,
    rtol: float = 1e-2,
    causal: bool = False,
    causal_sweep_phase: str = "prefill",
    cases: list[dict] | None = None,
    skip_reason: Callable[[], str | None] | None = None,
) -> dict:
    resolved_output_dir = DEFAULT_OUTPUT_DIR if output_dir is None else Path(output_dir)
    if resolved_output_dir.is_absolute():
        raise ValueError("--output-dir must be repo-relative")
    _validate_causal_sweep_phase(causal_sweep_phase)

    if cases is None:
        if not causal:
            case_specs = FLASHATTENTION_SWEEP_CASES
        elif causal_sweep_phase == "decode":
            case_specs = FLASHATTENTION_CAUSAL_DECODE_SWEEP_CASES
        elif causal_sweep_phase == "append":
            case_specs = FLASHATTENTION_CAUSAL_APPEND_SWEEP_CASES
        else:
            case_specs = FLASHATTENTION_CAUSAL_PREFILL_SWEEP_CASES
    else:
        case_specs = cases
    case_results = []
    counts = {"passed": 0, "failed": 0, "skipped": 0}

    for index, case in enumerate(case_specs):
        case_name = str(case["name"])
        tile_shape = tuple(int(value) for value in case["tile_shape"])
        provenance = str(case["provenance"])
        try:
            result = run_flashattention_correctness(
                output_dir=resolved_output_dir / case_name,
                arch=arch,
                tile_shape=tile_shape,
                atol=atol,
                rtol=rtol,
                seed=int(case["seed"]),
                causal=causal,
                skip_reason=skip_reason,
            )
            status = result["status"]
            counts[status] += 1
            case_results.append(
                _sweep_case_payload(
                    index=index,
                    case_name=case_name,
                    provenance=provenance,
                    result=result,
                )
            )
        except Exception as exc:
            counts["failed"] += 1
            seqlen_q, seqlen_k, head_dim = tile_shape
            phase = _phase_for(tile_shape=tile_shape, causal=causal)
            case_results.append(
                {
                    "case_index": index,
                    "case_name": case_name,
                    "shape": {
                        "seqlen_q": seqlen_q,
                        "seqlen_k": seqlen_k,
                        "head_dim": head_dim,
                    },
                    "provenance": provenance,
                    "phase": phase,
                    "causal": causal,
                    "reference": _reference_for(phase=phase, causal=causal),
                    "tolerance": {"atol": atol, "rtol": rtol},
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": _clean_text(str(exc)),
                }
            )

    aggregate_status = "passed"
    if counts["failed"]:
        aggregate_status = "failed"
    elif counts["skipped"]:
        aggregate_status = "skipped"

    return {
        "schema_version": 1,
        "kernel_name": "flashattention_fwd_f32",
        "status": aggregate_status,
        "case_count": len(case_results),
        "passed_cases": counts["passed"],
        "failed_cases": counts["failed"],
        "skipped_cases": counts["skipped"],
        "cases": case_results,
    }


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    require_cuda = False

    try:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
        parser.add_argument("--arch", default="compute_90")
        parser.add_argument("--atol", type=float, default=1e-3)
        parser.add_argument("--rtol", type=float, default=1e-2)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument(
            "--tile-shape",
            help="single-case tile shape as MxNxD, for example 32x32x64",
        )
        parser.add_argument(
            "--sweep",
            action="store_true",
            help="run the fixed review sweep instead of the single default case",
        )
        parser.add_argument(
            "--causal",
            action="store_true",
            help="apply a lower-triangular causal mask for single-case or sweep runs",
        )
        parser.add_argument(
            "--causal-sweep-phase",
            choices=("prefill", "decode", "append"),
            default="prefill",
            help=(
                "select causal sweep cases; default preserves the bounded "
                "same-length prefill sweep"
            ),
        )
        parser.add_argument(
            "--kv-cache-boundary",
            choices=("none", "paged", "ragged"),
            default="none",
            help=(
                "report an explicit unsupported KV-cache boundary for the "
                "single-case run"
            ),
        )
        parser.add_argument(
            "--sequence-boundary",
            choices=("fixed", "varlen"),
            default="fixed",
            help=(
                "report an explicit unsupported sequence boundary for the "
                "single-case run"
            ),
        )
        parser.add_argument(
            "--attention-variant",
            choices=("standard", "mla", "cascade", "sparse", "pod"),
            default="standard",
            help=(
                "report an explicit unsupported attention variant boundary "
                "for the single-case run"
            ),
        )
        parser.add_argument(
            "--require-cuda",
            action="store_true",
            help="return a non-zero status when dependencies or CUDA are unavailable",
        )
        args = parser.parse_args(raw_args)
        require_cuda = args.require_cuda

        if args.sweep:
            if args.kv_cache_boundary != "none":
                raise ValueError("--kv-cache-boundary is only supported without --sweep")
            if args.sequence_boundary != "fixed":
                raise ValueError("--sequence-boundary is only supported without --sweep")
            if args.attention_variant != "standard":
                raise ValueError("--attention-variant is only supported without --sweep")
            result = run_flashattention_sweep(
                output_dir=args.output_dir,
                arch=args.arch,
                atol=args.atol,
                rtol=args.rtol,
                causal=args.causal,
                causal_sweep_phase=args.causal_sweep_phase,
            )
        else:
            result = run_flashattention_correctness(
                output_dir=args.output_dir,
                arch=args.arch,
                tile_shape=_parse_tile_shape(args.tile_shape),
                atol=args.atol,
                rtol=args.rtol,
                seed=args.seed,
                causal=args.causal,
                attention_variant=args.attention_variant,
                sequence_boundary=args.sequence_boundary,
                kv_cache_boundary=args.kv_cache_boundary,
            )
    except Exception as exc:
        result = {
            "schema_version": 1,
            "kernel_name": "flashattention_fwd_f32",
            "status": "failed",
            "command": _display_command(raw_args),
            "error_type": type(exc).__name__,
            "error": _clean_text(str(exc)),
        }

    print(json.dumps(result, indent=2, sort_keys=True))

    if result["status"] == "failed":
        return 1
    if result["status"] == "skipped" and require_cuda:
        return 2
    return 0


def _artifact_payload(artifact: GluonKernelArtifact) -> dict:
    return {
        "arch": artifact.arch,
        "compiler_role": artifact.compiler_role,
        "manifest_path": _relative_path(artifact.manifest_path),
        "source_path": _relative_path(artifact.source_path),
        "source_sha256": artifact.source_sha256,
        "tile_shape": list(artifact.tile_shape),
    }


def _sweep_case_payload(
    *,
    index: int,
    case_name: str,
    provenance: str,
    result: dict,
) -> dict:
    payload = {
        "case_index": index,
        "case_name": case_name,
        "shape": result["shape"],
        "phase": result["phase"],
        "provenance": provenance,
        "reference": result["reference"],
        "tolerance": result["tolerance"],
        "causal": result["causal"],
        "status": result["status"],
        "artifact": result["artifact"],
    }
    if "reason" in result:
        payload["reason"] = result["reason"]
    if "max_abs_error" in result:
        payload["max_abs_error"] = result["max_abs_error"]
    if "error_type" in result:
        payload["error_type"] = result["error_type"]
    if "error" in result:
        payload["error"] = _clean_text(str(result["error"]))
    return payload


def _relative_path(path: str | Path) -> str:
    path = Path(path)
    if path.is_absolute():
        try:
            path = path.relative_to(Path.cwd())
        except ValueError:
            path = Path(path.name)
    return path.as_posix()


def _clean_text(text: str) -> str:
    cwd = Path.cwd().as_posix()
    home = Path.home().as_posix()
    return text.replace(cwd, ".").replace(home, "~")


def _parse_tile_shape(raw_tile_shape: str | None) -> tuple[int, int, int]:
    if raw_tile_shape is None:
        return FLASHATTENTION_TILE
    try:
        parts = tuple(int(part) for part in raw_tile_shape.lower().split("x"))
    except ValueError as exc:
        raise ValueError("--tile-shape must use MxNxD positive integers") from exc
    if len(parts) != 3 or any(part <= 0 for part in parts):
        raise ValueError("--tile-shape must use MxNxD positive integers")
    return parts


def _validate_kv_cache_boundary(kv_cache_boundary: str) -> None:
    if kv_cache_boundary not in ("none", "paged", "ragged"):
        raise ValueError("--kv-cache-boundary must be one of: none, paged, ragged")


def _validate_sequence_boundary(sequence_boundary: str) -> None:
    if sequence_boundary not in ("fixed", "varlen"):
        raise ValueError("--sequence-boundary must be one of: fixed, varlen")


def _validate_attention_variant(attention_variant: str) -> None:
    if attention_variant not in ("standard", "mla", "cascade", "sparse", "pod"):
        raise ValueError(
            "--attention-variant must be one of: standard, mla, cascade, sparse, pod"
        )


def _validate_causal_sweep_phase(causal_sweep_phase: str) -> None:
    if causal_sweep_phase not in ("prefill", "decode", "append"):
        raise ValueError(
            "--causal-sweep-phase must be one of: prefill, decode, append"
        )


def _unsupported_attention_variant_result(
    *,
    arch: str,
    tile_shape: tuple[int, int, int],
    atol: float,
    rtol: float,
    causal: bool,
    attention_variant: str,
) -> dict:
    seqlen_q, seqlen_k, head_dim = tile_shape
    phase = _phase_for(tile_shape=tile_shape, causal=causal)
    boundary_kind = {
        "mla": "mla_attention",
        "cascade": "cascade_attention",
        "sparse": "sparse_attention",
        "pod": "pod_attention",
    }[attention_variant]
    boundary_label = {
        "mla": "MLA attention",
        "cascade": "Cascade Attention",
        "sparse": "Sparse Attention",
        "pod": "POD-Attention",
    }[attention_variant]
    return {
        "schema_version": 1,
        "kernel_name": "flashattention_fwd_f32",
        "status": "skipped",
        "phase": phase,
        "arch": arch,
        "shape": {
            "seqlen_q": seqlen_q,
            "seqlen_k": seqlen_k,
            "head_dim": head_dim,
        },
        "causal": causal,
        "attention_variant": attention_variant,
        "reference": _reference_for(phase=phase, causal=causal),
        "tolerance": {"atol": atol, "rtol": rtol},
        "unsupported_boundary": {
            "kind": boundary_kind,
            "operator": "flashattention_fwd_f32",
            "boundary": attention_variant,
            "status": "unsupported",
        },
        "reason": (
            f"Gluon FlashAttention {boundary_label} boundary is unsupported; "
            "this is unsupported-boundary evidence only"
        ),
    }


def _unsupported_sequence_boundary_result(
    *,
    arch: str,
    tile_shape: tuple[int, int, int],
    atol: float,
    rtol: float,
    causal: bool,
    sequence_boundary: str,
) -> dict:
    seqlen_q, seqlen_k, head_dim = tile_shape
    phase = _phase_for(tile_shape=tile_shape, causal=causal)
    return {
        "schema_version": 1,
        "kernel_name": "flashattention_fwd_f32",
        "status": "skipped",
        "phase": phase,
        "arch": arch,
        "shape": {
            "seqlen_q": seqlen_q,
            "seqlen_k": seqlen_k,
            "head_dim": head_dim,
        },
        "causal": causal,
        "sequence_boundary": sequence_boundary,
        "reference": _reference_for(phase=phase, causal=causal),
        "tolerance": {"atol": atol, "rtol": rtol},
        "unsupported_boundary": {
            "kind": "varlen_attention",
            "operator": "flashattention_fwd_f32",
            "boundary": sequence_boundary,
            "status": "unsupported",
        },
        "reason": (
            "Gluon FlashAttention varlen attention boundary is unsupported; "
            "this is unsupported-boundary evidence only"
        ),
    }


def _unsupported_kv_cache_boundary_result(
    *,
    arch: str,
    tile_shape: tuple[int, int, int],
    atol: float,
    rtol: float,
    causal: bool,
    kv_cache_boundary: str,
) -> dict:
    seqlen_q, seqlen_k, head_dim = tile_shape
    phase = _phase_for(tile_shape=tile_shape, causal=causal)
    return {
        "schema_version": 1,
        "kernel_name": "flashattention_fwd_f32",
        "status": "skipped",
        "phase": phase,
        "arch": arch,
        "shape": {
            "seqlen_q": seqlen_q,
            "seqlen_k": seqlen_k,
            "head_dim": head_dim,
        },
        "causal": causal,
        "kv_cache_boundary": kv_cache_boundary,
        "reference": _reference_for(phase=phase, causal=causal),
        "tolerance": {"atol": atol, "rtol": rtol},
        "unsupported_boundary": {
            "kind": f"{kv_cache_boundary}_kv_cache",
            "operator": "flashattention_fwd_f32",
            "boundary": kv_cache_boundary,
            "status": "unsupported",
        },
        "reason": (
            f"Gluon FlashAttention {kv_cache_boundary} KV-cache boundary is "
            "unsupported; this is unsupported-boundary evidence only"
        ),
    }


def _phase_for(*, tile_shape: tuple[int, int, int], causal: bool) -> str:
    seqlen_q, seqlen_k, _head_dim = tile_shape
    if not causal:
        return "single_tile"
    if seqlen_q == seqlen_k:
        return "prefill"
    if seqlen_q == 1 and seqlen_k > seqlen_q:
        return "decode"
    if 1 < seqlen_q < seqlen_k:
        return "append"
    return "single_tile"


def _reference_for(*, phase: str, causal: bool) -> str:
    if not causal:
        return FLASHATTENTION_REFERENCE
    if phase in ("decode", "append"):
        return FLASHATTENTION_CAUSAL_DECODE_REFERENCE
    return FLASHATTENTION_CAUSAL_REFERENCE


def _display_command(raw_args: list[str]) -> str:
    safe_args = []
    for arg in raw_args:
        if "=" in arg:
            option, value = arg.split("=", 1)
            path = Path(value)
            safe_args.append(f"{option}={path.name}" if path.is_absolute() else arg)
        else:
            path = Path(arg)
            safe_args.append(path.name if path.is_absolute() else arg)
    command = "examples/cuda/gluon_flashattention_fwd.py"
    if safe_args:
        command = f"{command} {shlex.join(safe_args)}"
    return command


if __name__ == "__main__":
    raise SystemExit(main())
