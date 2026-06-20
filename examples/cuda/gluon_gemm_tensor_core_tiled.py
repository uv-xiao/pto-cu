#!/usr/bin/env python3
"""Skip-safe correctness harness for tiled Triton/Gluon WGMMA GEMM."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Callable

from simpler_setup.gluon_gen import GluonKernelArtifact
from simpler_setup.kernel_compiler import KernelCompiler

TILED_TENSOR_CORE_TILE = (64, 128, 32)


def build_tiled_tensor_core_artifact(
    *,
    output_dir: str | Path | None = None,
    arch: str = "compute_90",
) -> GluonKernelArtifact:
    return KernelCompiler(platform="cuda").generate_gluon_kernel(
        "gemm_tensor_core_tiled_f16_f32",
        output_dir=output_dir,
        arch=arch,
        tile_shape=TILED_TENSOR_CORE_TILE,
    )


def tensor_core_skip_reason() -> str | None:
    tensor_core = _load_sibling("gluon_gemm_tensor_core.py")
    return tensor_core.tensor_core_skip_reason()


def load_generated_module(source_path: str | Path):
    tensor_core = _load_sibling("gluon_gemm_tensor_core.py")
    return tensor_core.load_generated_module(source_path)


def run_tiled_tensor_core_correctness(
    *,
    output_dir: str | Path | None = None,
    arch: str = "compute_90",
    m: int = 256,
    n: int = 256,
    k: int = 64,
    atol: float = 1e-3,
    rtol: float = 1e-1,
    seed: int = 0,
    skip_reason: Callable[[], str | None] | None = None,
) -> dict:
    artifact = build_tiled_tensor_core_artifact(output_dir=output_dir, arch=arch)
    result = {
        "kernel_name": "gemm_tensor_core_tiled_f16_f32",
        "artifact": _artifact_payload(artifact),
        "shape": {"m": m, "n": n, "k": k},
        "tolerance": {"atol": atol, "rtol": rtol},
    }

    block_m, block_n, block_k = artifact.tile_shape
    if k % block_k != 0:
        raise ValueError(f"expected k divisible by {block_k}, got {k}")
    if m % block_m != 0 or n % block_n != 0:
        raise ValueError(
            f"expected m,n divisible by tile ({block_m}, {block_n}), got ({m}, {n})"
        )

    skip_check = tensor_core_skip_reason if skip_reason is None else skip_reason
    reason = skip_check()
    if reason is not None:
        return {**result, "status": "skipped", "reason": reason}

    import torch

    module = load_generated_module(artifact.source_path)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    a = torch.randn((m, k), device="cuda", dtype=torch.float16, generator=generator)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16, generator=generator)
    c = torch.zeros((m, n), device="cuda", dtype=torch.float32)
    d = torch.empty_like(c)

    module.run_gemm_tensor_core_tiled_f16_f32(
        a,
        b,
        c,
        d,
        instr_shape_n=16,
        num_warps=4,
    )
    torch.cuda.synchronize()

    expected = a @ b + c
    torch.cuda.synchronize()
    max_abs_error = float((d - expected).abs().max().item())
    passed = bool(torch.allclose(d, expected, atol=atol, rtol=rtol))
    return {
        **result,
        "status": "passed" if passed else "failed",
        "max_abs_error": max_abs_error,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--arch", default="compute_90")
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="return a non-zero status when dependencies or CUDA are unavailable",
    )
    args = parser.parse_args(argv)

    try:
        result = run_tiled_tensor_core_correctness(
            output_dir=args.output_dir,
            arch=args.arch,
            m=args.m,
            n=args.n,
            k=args.k,
            atol=args.atol,
            rtol=args.rtol,
            seed=args.seed,
        )
    except Exception as exc:
        result = {
            "kernel_name": "gemm_tensor_core_tiled_f16_f32",
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
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


def _load_sibling(filename: str):
    source_path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(source_path.stem, source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    raise SystemExit(main())
