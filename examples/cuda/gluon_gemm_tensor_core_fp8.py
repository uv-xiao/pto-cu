#!/usr/bin/env python3
"""Structured FP8 boundary harness for generated Gluon WGMMA GEMM."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Callable

from simpler_setup.gluon_gen import GluonKernelArtifact
from simpler_setup.kernel_compiler import KernelCompiler


KERNEL_NAME = "gemm_tensor_core_tiled_fp8e4nv_f32"
DEFAULT_OUTPUT_DIR = Path("tmp/gluon-tensor-core-fp8-local")
TILED_TENSOR_CORE_TILE = (64, 128, 32)
FP8_DTYPE_BOUNDARY = {
    "a": "torch.float8_e4m3fn / gl.float8e4nv",
    "b": "torch.float8_e4m3fn / gl.float8e4nv",
    "accumulator": "float32",
    "out": "float32",
}
UNSUPPORTED_BOUNDARY = {
    "kind": "gluon_fp8_wgmma_compile",
    "expected_failure": "Triton/Gluon may reject FP8 WGMMA type or shape lowering",
}


def build_fp8_tensor_core_artifact(
    *,
    output_dir: str | Path | None = None,
    arch: str = "compute_90",
) -> GluonKernelArtifact:
    return KernelCompiler(platform="cuda").generate_gluon_kernel(
        KERNEL_NAME,
        output_dir=DEFAULT_OUTPUT_DIR if output_dir is None else output_dir,
        arch=arch,
        tile_shape=TILED_TENSOR_CORE_TILE,
    )


def tensor_core_skip_reason() -> str | None:
    tensor_core = _load_sibling("gluon_gemm_tensor_core.py")
    return tensor_core.tensor_core_skip_reason()


def load_generated_module(source_path: str | Path):
    tensor_core = _load_sibling("gluon_gemm_tensor_core.py")
    return tensor_core.load_generated_module(source_path)


def probe_fp8_dtypes() -> dict:
    result = {
        "status": "failed",
        "gl_dtype": "float8e4nv",
        "torch_dtype": "float8_e4m3fn",
        "gl_fp8_attrs": [],
        "torch_fp8_attrs": [],
    }

    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on local env
        return {**result, "reason": f"torch import failed: {_clean_text(str(exc))}"}

    result["torch_fp8_attrs"] = sorted(
        name for name in dir(torch) if "float8" in name.lower() or "fp8" in name.lower()
    )
    if not hasattr(torch, "float8_e4m3fn"):
        return {**result, "reason": "missing torch.float8_e4m3fn"}

    try:
        from triton.experimental.gluon import language as gl
    except Exception as exc:  # pragma: no cover - depends on local env
        return {
            **result,
            "reason": f"triton Gluon language import failed: {_clean_text(str(exc))}",
        }

    result["gl_fp8_attrs"] = sorted(
        name for name in dir(gl) if "float8" in name.lower() or "fp8" in name.lower()
    )
    if not hasattr(gl, "float8e4nv"):
        return {**result, "reason": "missing gl.float8e4nv"}

    gl_dtype = getattr(gl, "float8e4nv")
    result["gl_primitive_bitwidth"] = getattr(gl_dtype, "primitive_bitwidth", None)
    if result["gl_primitive_bitwidth"] != 8:
        return {
            **result,
            "reason": f"expected gl.float8e4nv primitive_bitwidth=8, "
            f"got {result['gl_primitive_bitwidth']}",
        }
    return {**result, "status": "passed"}


def run_fp8_tensor_core_boundary(
    *,
    output_dir: str | Path | None = None,
    arch: str = "compute_90",
    m: int = 64,
    n: int = 128,
    k: int = 32,
    atol: float = 1.0,
    rtol: float = 0.2,
    seed: int = 0,
    device: int = 0,
    skip_reason: Callable[[], str | None] | None = None,
    dtype_probe: Callable[[], dict] | None = None,
) -> dict:
    resolved_output_dir = DEFAULT_OUTPUT_DIR if output_dir is None else Path(output_dir)
    if resolved_output_dir.is_absolute():
        raise ValueError("--output-dir must be repo-relative")

    artifact = build_fp8_tensor_core_artifact(output_dir=resolved_output_dir, arch=arch)
    result = {
        "schema_version": 1,
        "kernel_name": KERNEL_NAME,
        "artifact": _artifact_payload(artifact),
        "dtype_boundary": FP8_DTYPE_BOUNDARY,
        "shape": {"m": m, "n": n, "k": k},
        "tolerance": {"atol": atol, "rtol": rtol},
        "unsupported_boundary": UNSUPPORTED_BOUNDARY,
    }

    _validate_shape(m=m, n=n, k=k, tile_shape=artifact.tile_shape)
    probe = probe_fp8_dtypes if dtype_probe is None else dtype_probe
    dtype_status = probe()
    result["fp8_dtype_probe"] = dtype_status

    skip_check = tensor_core_skip_reason if skip_reason is None else skip_reason
    reason = skip_check()
    if reason is not None:
        return {**result, "status": "skipped", "reason": _clean_text(reason)}
    if dtype_status["status"] != "passed":
        return {
            **result,
            "status": "skipped",
            "reason": _clean_text(str(dtype_status.get("reason", "missing FP8 dtype API"))),
        }

    try:
        import torch

        torch.cuda.set_device(device)
        cuda_device = torch.device("cuda", device)
        module = load_generated_module(artifact.source_path)
        generator = torch.Generator(device=cuda_device)
        generator.manual_seed(seed)
        a = torch.randn((m, k), device=cuda_device, dtype=torch.float32, generator=generator)
        b = torch.randn((k, n), device=cuda_device, dtype=torch.float32, generator=generator)
        a = a.to(torch.float8_e4m3fn)
        b = b.to(torch.float8_e4m3fn)
        c = torch.zeros((m, n), device=cuda_device, dtype=torch.float32)
        d = torch.empty_like(c)

        module.run_gemm_tensor_core_tiled_fp8e4nv_f32(
            a,
            b,
            c,
            d,
            instr_shape_n=16,
            num_warps=4,
        )
        torch.cuda.synchronize(device)

        expected = torch.mm(a.float(), b.float()) + c
        torch.cuda.synchronize(device)
        max_abs_error = float((d - expected).abs().max().item())
        passed = bool(torch.allclose(d, expected, atol=atol, rtol=rtol))
        return {
            **result,
            "status": "passed" if passed else "failed",
            "max_abs_error": max_abs_error,
        }
    except Exception as exc:  # noqa: BLE001 - boundary probes must emit JSON.
        return {
            **result,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": _clean_text(str(exc)),
        }


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    require_cuda = False

    try:
        parser = JsonArgumentParser(description=__doc__)
        parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
        parser.add_argument("--arch", default="compute_90")
        parser.add_argument("--m", type=int, default=64)
        parser.add_argument("--n", type=int, default=128)
        parser.add_argument("--k", type=int, default=32)
        parser.add_argument("--atol", type=float, default=1.0)
        parser.add_argument("--rtol", type=float, default=0.2)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--device", type=int, default=0)
        parser.add_argument(
            "--require-cuda",
            action="store_true",
            help="return a non-zero status when FP8 WGMMA is unavailable",
        )
        args = parser.parse_args(raw_args)
        require_cuda = args.require_cuda

        result = run_fp8_tensor_core_boundary(
            output_dir=args.output_dir,
            arch=args.arch,
            m=args.m,
            n=args.n,
            k=args.k,
            atol=args.atol,
            rtol=args.rtol,
            seed=args.seed,
            device=args.device,
        )
    except Exception as exc:
        result = {
            "schema_version": 1,
            "kernel_name": KERNEL_NAME,
            "status": "failed",
            "command": _display_command(raw_args),
            "unsupported_boundary": UNSUPPORTED_BOUNDARY,
            "error_type": type(exc).__name__,
            "error": _clean_text(str(exc)),
        }

    print(json.dumps(result, indent=2, sort_keys=True))

    if result["status"] == "failed":
        return 1
    if result["status"] == "skipped" and require_cuda:
        return 2
    return 0


def _validate_shape(*, m: int, n: int, k: int, tile_shape: tuple[int, int, int]) -> None:
    block_m, block_n, block_k = tile_shape
    if m <= 0 or n <= 0 or k <= 0:
        raise ValueError("m, n, and k must be positive")
    if k % block_k != 0:
        raise ValueError(f"expected k divisible by {block_k}, got {k}")
    if m % block_m != 0 or n % block_n != 0:
        raise ValueError(
            f"expected m,n divisible by tile ({block_m}, {block_n}), got ({m}, {n})"
        )


def _artifact_payload(artifact: GluonKernelArtifact) -> dict:
    return {
        "arch": artifact.arch,
        "compiler_role": artifact.compiler_role,
        "manifest_path": _relative_path(artifact.manifest_path),
        "source_path": _relative_path(artifact.source_path),
        "source_sha256": artifact.source_sha256,
        "tile_shape": list(artifact.tile_shape),
    }


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


def _display_command(raw_args: list[str]) -> str:
    return " ".join(["gluon_gemm_tensor_core_fp8.py", *raw_args])


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
