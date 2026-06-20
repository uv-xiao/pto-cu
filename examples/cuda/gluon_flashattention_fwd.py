#!/usr/bin/env python3
"""Skip-safe correctness harness for generated Triton/Gluon FlashAttention."""

from __future__ import annotations

import argparse
import importlib.util
import json
import traceback
from pathlib import Path
from typing import Callable

from simpler_setup.gluon_gen import GluonKernelArtifact
from simpler_setup.kernel_compiler import KernelCompiler

FLASHATTENTION_TILE = (32, 32, 32)


def build_flashattention_artifact(
    *,
    output_dir: str | Path | None = None,
    arch: str = "compute_90",
) -> GluonKernelArtifact:
    return KernelCompiler(platform="cuda").generate_gluon_kernel(
        "flashattention_fwd_f32",
        output_dir=output_dir,
        arch=arch,
        tile_shape=FLASHATTENTION_TILE,
    )


def flashattention_skip_reason() -> str | None:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on local environment
        return f"torch import failed: {exc}"

    if not torch.cuda.is_available():
        return "torch.cuda is not available"

    try:
        import triton  # noqa: F401
        from triton.experimental import gluon  # noqa: F401
        from triton.experimental.gluon import language as gl
    except Exception as exc:  # pragma: no cover - depends on local environment
        return f"triton Gluon import failed: {exc}"

    for required in ("dot_fma", "exp", "max", "sum"):
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
        raise RuntimeError(f"could not load generated Gluon source: {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_flashattention_correctness(
    *,
    output_dir: str | Path | None = None,
    arch: str = "compute_90",
    atol: float = 1e-3,
    rtol: float = 1e-2,
    seed: int = 0,
    skip_reason: Callable[[], str | None] | None = None,
) -> dict:
    artifact = build_flashattention_artifact(output_dir=output_dir, arch=arch)
    seqlen_q, seqlen_k, head_dim = artifact.tile_shape
    result = {
        "kernel_name": "flashattention_fwd_f32",
        "artifact": _artifact_payload(artifact),
        "shape": {
            "seqlen_q": seqlen_q,
            "seqlen_k": seqlen_k,
            "head_dim": head_dim,
        },
        "tolerance": {"atol": atol, "rtol": rtol},
    }

    skip_check = flashattention_skip_reason if skip_reason is None else skip_reason
    reason = skip_check()
    if reason is not None:
        return {**result, "status": "skipped", "reason": reason}

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
    module.run_flashattention_fwd_f32(q, k, v, out, scale=scale, num_warps=4)
    torch.cuda.synchronize()

    expected = torch.softmax((q @ k.T) * scale, dim=-1) @ v
    torch.cuda.synchronize()
    max_abs_error = float((out - expected).abs().max().item())
    passed = bool(torch.allclose(out, expected, atol=atol, rtol=rtol))
    return {
        **result,
        "status": "passed" if passed else "failed",
        "max_abs_error": max_abs_error,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--arch", default="compute_90")
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="return a non-zero status when dependencies or CUDA are unavailable",
    )
    args = parser.parse_args(argv)

    try:
        result = run_flashattention_correctness(
            output_dir=args.output_dir,
            arch=args.arch,
            atol=args.atol,
            rtol=args.rtol,
            seed=args.seed,
        )
    except Exception as exc:
        result = {
            "kernel_name": "flashattention_fwd_f32",
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }

    print(json.dumps(result, indent=2, sort_keys=True))

    if result["status"] == "failed":
        return 1
    if result["status"] == "skipped" and args.require_cuda:
        return 2
    return 0


def _artifact_payload(artifact: GluonKernelArtifact) -> dict:
    return {
        "arch": artifact.arch,
        "compiler_role": artifact.compiler_role,
        "manifest_path": str(artifact.manifest_path),
        "source_path": str(artifact.source_path),
        "source_sha256": artifact.source_sha256,
        "tile_shape": list(artifact.tile_shape),
    }


if __name__ == "__main__":
    raise SystemExit(main())
