#!/usr/bin/env python3
"""Skip-safe correctness harness for a generated Gluon MoE expert affine."""

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


DEFAULT_OUTPUT_DIR = Path("tmp/gluon-moe-expert-local")
MOE_EXPERT_SWEEP_CASES = [
    {"name": "n16_baseline", "n": 16, "scale_a": 1.25, "scale_b": 0.5, "seed": 0},
    {"name": "n31_signed", "n": 31, "scale_a": -0.75, "scale_b": 2.0, "seed": 7},
    {"name": "n256_b_only", "n": 256, "scale_a": 0.0, "scale_b": -1.0, "seed": 13},
    {"name": "n4096_mixed", "n": 4096, "scale_a": 1.5, "scale_b": -0.25, "seed": 23},
]


def build_moe_expert_artifact(
    *,
    output_dir: str | Path | None = None,
    arch: str = "compute_90",
) -> GluonKernelArtifact:
    return KernelCompiler(platform="cuda").generate_gluon_kernel(
        "moe_expert_affine_f32",
        output_dir=DEFAULT_OUTPUT_DIR if output_dir is None else output_dir,
        arch=arch,
        tile_shape=(1, 1, 1),
    )


def moe_expert_skip_reason() -> str | None:
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
        from triton.experimental.gluon import language as gl  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on local environment
        return f"triton Gluon import failed: {_clean_text(str(exc))}"

    return None


def load_generated_module(source_path: str | Path):
    source_path = Path(source_path)
    spec = importlib.util.spec_from_file_location(
        f"pto_generated_{source_path.stem.replace('.', '_')}",
        source_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load generated Gluon source: {_relative_path(source_path)}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_moe_expert_correctness(
    *,
    output_dir: str | Path | None = None,
    arch: str = "compute_90",
    n: int = 4096,
    scale_a: float = 1.25,
    scale_b: float = 0.5,
    atol: float = 1e-6,
    rtol: float = 1e-6,
    seed: int = 0,
    skip_reason: Callable[[], str | None] | None = None,
) -> dict:
    resolved_output_dir = DEFAULT_OUTPUT_DIR if output_dir is None else Path(output_dir)
    if resolved_output_dir.is_absolute():
        raise ValueError("--output-dir must be repo-relative")

    artifact = build_moe_expert_artifact(output_dir=resolved_output_dir, arch=arch)
    result = {
        "schema_version": 1,
        "kernel_name": "moe_expert_affine_f32",
        "artifact": _artifact_payload(artifact),
        "shape": {"n": n},
        "scalars": {"scale_a": scale_a, "scale_b": scale_b},
        "tolerance": {"atol": atol, "rtol": rtol},
    }

    skip_check = moe_expert_skip_reason if skip_reason is None else skip_reason
    reason = skip_check()
    if reason is not None:
        return {**result, "status": "skipped", "reason": _clean_text(reason)}

    import torch

    module = load_generated_module(artifact.source_path)

    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    a = torch.randn((n,), device="cuda", dtype=torch.float32, generator=generator)
    b = torch.randn((n,), device="cuda", dtype=torch.float32, generator=generator)
    out = torch.empty((n,), device="cuda", dtype=torch.float32)

    module.run_moe_expert_affine_f32(
        a,
        b,
        out,
        scale_a=scale_a,
        scale_b=scale_b,
    )
    torch.cuda.synchronize()

    expected = scale_a * a + scale_b * b
    torch.cuda.synchronize()
    max_abs_error = float((out - expected).abs().max().item())
    passed = bool(torch.allclose(out, expected, atol=atol, rtol=rtol))
    return {
        **result,
        "status": "passed" if passed else "failed",
        "max_abs_error": max_abs_error,
    }


def run_moe_expert_sweep(
    *,
    output_dir: str | Path | None = None,
    arch: str = "compute_90",
    atol: float = 1e-6,
    rtol: float = 1e-6,
    cases: list[dict] | None = None,
    skip_reason: Callable[[], str | None] | None = None,
) -> dict:
    resolved_output_dir = DEFAULT_OUTPUT_DIR if output_dir is None else Path(output_dir)
    if resolved_output_dir.is_absolute():
        raise ValueError("--output-dir must be repo-relative")

    case_specs = MOE_EXPERT_SWEEP_CASES if cases is None else cases
    case_results = []
    counts = {"passed": 0, "failed": 0, "skipped": 0}

    for index, case in enumerate(case_specs):
        case_name = str(case["name"])
        try:
            result = run_moe_expert_correctness(
                output_dir=resolved_output_dir / case_name,
                arch=arch,
                n=int(case["n"]),
                scale_a=float(case["scale_a"]),
                scale_b=float(case["scale_b"]),
                atol=atol,
                rtol=rtol,
                seed=int(case["seed"]),
                skip_reason=skip_reason,
            )
            status = result["status"]
            counts[status] += 1
            case_results.append(_sweep_case_payload(index, case_name, result))
        except Exception as exc:
            counts["failed"] += 1
            case_results.append(
                {
                    "case_index": index,
                    "case_name": case_name,
                    "shape": {"n": int(case["n"])},
                    "scalars": {
                        "scale_a": float(case["scale_a"]),
                        "scale_b": float(case["scale_b"]),
                    },
                    "tolerance": {"atol": atol, "rtol": rtol},
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": _clean_text(str(exc)),
                }
            )

    case_count = len(case_results)
    aggregate_status = "passed"
    if counts["failed"]:
        aggregate_status = "failed"
    elif counts["skipped"]:
        aggregate_status = "skipped"

    return {
        "schema_version": 1,
        "kernel_name": "moe_expert_affine_f32",
        "status": aggregate_status,
        "case_count": case_count,
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
        parser.add_argument("--n", type=int, default=4096)
        parser.add_argument("--scale-a", type=float, default=1.25)
        parser.add_argument("--scale-b", type=float, default=0.5)
        parser.add_argument("--atol", type=float, default=1e-6)
        parser.add_argument("--rtol", type=float, default=1e-6)
        parser.add_argument("--seed", type=int, default=0)
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
            result = run_moe_expert_sweep(
                output_dir=args.output_dir,
                arch=args.arch,
                atol=args.atol,
                rtol=args.rtol,
            )
        else:
            result = run_moe_expert_correctness(
                output_dir=args.output_dir,
                arch=args.arch,
                n=args.n,
                scale_a=args.scale_a,
                scale_b=args.scale_b,
                atol=args.atol,
                rtol=args.rtol,
                seed=args.seed,
            )
    except Exception as exc:
        result = {
            "schema_version": 1,
            "kernel_name": "moe_expert_affine_f32",
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


def _sweep_case_payload(index: int, case_name: str, result: dict) -> dict:
    payload = {
        "case_index": index,
        "case_name": case_name,
        "shape": result["shape"],
        "scalars": result["scalars"],
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
    command = "examples/cuda/gluon_moe_expert_affine.py"
    if safe_args:
        command = f"{command} {shlex.join(safe_args)}"
    return command


if __name__ == "__main__":
    raise SystemExit(main())
