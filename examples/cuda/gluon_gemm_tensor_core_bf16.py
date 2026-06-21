#!/usr/bin/env python3
"""Skip-safe correctness harness for generated Gluon BF16 WGMMA GEMM."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
import sys
from pathlib import Path
from typing import Callable

from simpler_setup.gluon_gen import GluonKernelArtifact
from simpler_setup.kernel_compiler import KernelCompiler


KERNEL_NAME = "gemm_tensor_core_tiled_bf16_f32"
DEFAULT_OUTPUT_DIR = Path("tmp/gluon-tensor-core-bf16-local")
TILED_TENSOR_CORE_TILE = (64, 128, 32)
BF16_DTYPE_BOUNDARY = {
    "a": "bfloat16",
    "b": "bfloat16",
    "accumulator": "float32",
    "out": "float32",
}
BF16_TENSOR_CORE_CASES = [
    {
        "name": "smoke_tile",
        "m": 64,
        "n": 128,
        "k": 32,
        "seed": 0,
        "provenance": "existing tensor-core tiled smoke boundary",
    },
    {
        "name": "deepseek_v4_flash_hidden_size",
        "m": 64,
        "n": 128,
        "k": 7168,
        "seed": 0,
        "provenance": "DeepSeek-V4-Flash config hidden_size=7168",
    },
]


def build_bf16_tensor_core_artifact(
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


def run_bf16_tensor_core_correctness(
    *,
    output_dir: str | Path | None = None,
    arch: str = "compute_90",
    m: int = 64,
    n: int = 128,
    k: int = 7168,
    atol: float = 1.0,
    rtol: float = 0.2,
    seed: int = 0,
    device: int = 0,
    skip_reason: Callable[[], str | None] | None = None,
) -> dict:
    resolved_output_dir = DEFAULT_OUTPUT_DIR if output_dir is None else Path(output_dir)
    if resolved_output_dir.is_absolute():
        raise ValueError("--output-dir must be repo-relative")

    artifact = build_bf16_tensor_core_artifact(output_dir=resolved_output_dir, arch=arch)
    result = {
        "schema_version": 1,
        "kernel_name": KERNEL_NAME,
        "artifact": _artifact_payload(artifact),
        "dtype_boundary": BF16_DTYPE_BOUNDARY,
        "shape": {"m": m, "n": n, "k": k},
        "tolerance": {"atol": atol, "rtol": rtol},
    }

    _validate_shape(m=m, n=n, k=k, tile_shape=artifact.tile_shape)

    skip_check = tensor_core_skip_reason if skip_reason is None else skip_reason
    reason = skip_check()
    if reason is not None:
        return {**result, "status": "skipped", "reason": _clean_text(reason)}

    import torch

    torch.cuda.set_device(device)
    cuda_device = torch.device("cuda", device)
    module = load_generated_module(artifact.source_path)
    generator = torch.Generator(device=cuda_device)
    generator.manual_seed(seed)
    a = torch.randn((m, k), device=cuda_device, dtype=torch.bfloat16, generator=generator)
    b = torch.randn((k, n), device=cuda_device, dtype=torch.bfloat16, generator=generator)
    c = torch.zeros((m, n), device=cuda_device, dtype=torch.float32)
    d = torch.empty_like(c)

    module.run_gemm_tensor_core_tiled_bf16_f32(
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


def run_bf16_tensor_core_sweep(
    *,
    output_dir: str | Path | None = None,
    arch: str = "compute_90",
    atol: float = 1.0,
    rtol: float = 0.2,
    device: int = 0,
    cases: list[dict] | None = None,
    skip_reason: Callable[[], str | None] | None = None,
) -> dict:
    resolved_output_dir = DEFAULT_OUTPUT_DIR if output_dir is None else Path(output_dir)
    if resolved_output_dir.is_absolute():
        raise ValueError("--output-dir must be repo-relative")

    case_specs = BF16_TENSOR_CORE_CASES if cases is None else cases
    for case in case_specs:
        _validate_shape(
            m=int(case["m"]),
            n=int(case["n"]),
            k=int(case["k"]),
            tile_shape=TILED_TENSOR_CORE_TILE,
        )

    counts = {"passed": 0, "failed": 0, "skipped": 0}
    case_results = []
    for index, case in enumerate(case_specs):
        case_name = str(case["name"])
        result = run_bf16_tensor_core_correctness(
            output_dir=resolved_output_dir / case_name,
            arch=arch,
            m=int(case["m"]),
            n=int(case["n"]),
            k=int(case["k"]),
            atol=atol,
            rtol=rtol,
            seed=int(case["seed"]),
            device=device,
            skip_reason=skip_reason,
        )
        counts[result["status"]] += 1
        case_results.append(
            _sweep_case_payload(
                index=index,
                case_name=case_name,
                provenance=str(case["provenance"]),
                result=result,
            )
        )

    aggregate_status = "passed"
    if counts["failed"]:
        aggregate_status = "failed"
    elif counts["skipped"]:
        aggregate_status = "skipped"

    return {
        "schema_version": 1,
        "kernel_name": KERNEL_NAME,
        "status": aggregate_status,
        "case_count": len(case_results),
        "passed_cases": counts["passed"],
        "failed_cases": counts["failed"],
        "skipped_cases": counts["skipped"],
        "cases": case_results,
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
        parser.add_argument("--k", type=int, default=7168)
        parser.add_argument("--atol", type=float, default=1.0)
        parser.add_argument("--rtol", type=float, default=0.2)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--device", type=int, default=0)
        parser.add_argument(
            "--sweep",
            action="store_true",
            help="run the fixed review sweep instead of the single default case",
        )
        parser.add_argument(
            "--require-cuda",
            action="store_true",
            help="return a non-zero status when dependencies or CUDA are unavailable",
        )
        args = parser.parse_args(raw_args)
        require_cuda = args.require_cuda

        if args.sweep:
            result = run_bf16_tensor_core_sweep(
                output_dir=args.output_dir,
                arch=args.arch,
                atol=args.atol,
                rtol=args.rtol,
                device=args.device,
            )
        else:
            result = run_bf16_tensor_core_correctness(
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
        "kernel_name": result["kernel_name"],
        "dtype_boundary": result["dtype_boundary"],
        "shape": result["shape"],
        "provenance": provenance,
        "tolerance": result["tolerance"],
        "status": result["status"],
        "artifact": result["artifact"],
    }
    if "reason" in result:
        payload["reason"] = result["reason"]
    if "max_abs_error" in result:
        payload["max_abs_error"] = result["max_abs_error"]
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


def _display_command(raw_args: list[str]) -> str:
    safe_args = []
    for arg in raw_args:
        path = Path(arg)
        safe_args.append(path.name if path.is_absolute() else arg)
    command = "examples/cuda/gluon_gemm_tensor_core_bf16.py"
    if safe_args:
        command = f"{command} {shlex.join(safe_args)}"
    return command


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
