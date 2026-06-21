#!/usr/bin/env python3
"""Skip-safe correctness harness for generated Gluon FP32 gated SiLU."""

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


DEFAULT_OUTPUT_DIR = Path("tmp/gluon-gated-silu-local")
GATED_SILU_REFERENCE = "out = value * gate / (1.0 + exp(-gate))"
GATED_SILU_SWEEP_CASES = [
    {
        "name": "existing_smoke",
        "n": 32,
        "provenance": "existing smoke correctness fixture",
    },
    {
        "name": "deepseek_v4_flash_moe_inter_dim",
        "n": 2048,
        "provenance": (
            "tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash/"
            "inference/config.json moe_inter_dim: 2048, swiglu_limit: 10.0"
        ),
    },
]


def build_gated_silu_artifact(
    *,
    output_dir: str | Path | None = None,
    arch: str = "compute_90",
) -> GluonKernelArtifact:
    return KernelCompiler(platform="cuda").generate_gluon_kernel(
        "gated_silu_f32",
        output_dir=DEFAULT_OUTPUT_DIR if output_dir is None else output_dir,
        arch=arch,
        tile_shape=(1, 1, 1),
    )


def gated_silu_skip_reason() -> str | None:
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

    if not hasattr(gl, "exp"):
        return "triton Gluon import failed: missing gl.exp"

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


def run_gated_silu_correctness(
    *,
    output_dir: str | Path | None = None,
    arch: str = "compute_90",
    n: int = 32,
    atol: float = 1.0e-5,
    rtol: float = 1.0e-5,
    device: int = 0,
    skip_reason: Callable[[], str | None] | None = None,
) -> dict:
    resolved_output_dir = DEFAULT_OUTPUT_DIR if output_dir is None else Path(output_dir)
    if resolved_output_dir.is_absolute():
        raise ValueError("--output-dir must be repo-relative")
    if n <= 0:
        raise ValueError("--n must be positive")

    artifact = build_gated_silu_artifact(output_dir=resolved_output_dir, arch=arch)
    result = {
        "schema_version": 1,
        "kernel_name": "gated_silu_f32",
        "artifact": _artifact_payload(artifact),
        "shape": {"n": n},
        "reference": GATED_SILU_REFERENCE,
        "tolerance": {"atol": atol, "rtol": rtol},
    }

    skip_check = gated_silu_skip_reason if skip_reason is None else skip_reason
    reason = skip_check()
    if reason is not None:
        return {**result, "status": "skipped", "reason": _clean_text(reason)}

    import torch

    torch.cuda.set_device(device)
    cuda_device = torch.device("cuda", device)
    module = load_generated_module(artifact.source_path)

    gate = torch.linspace(-4.0, 4.0, n, device=cuda_device, dtype=torch.float32)
    value = torch.linspace(0.25, 2.25, n, device=cuda_device, dtype=torch.float32)
    out = torch.empty((n,), device=cuda_device, dtype=torch.float32)

    module.run_gated_silu_f32(gate, value, out)
    torch.cuda.synchronize(device)

    expected = value * gate / (1.0 + torch.exp(-gate))
    torch.cuda.synchronize(device)
    max_abs_error = float((out - expected).abs().max().item())
    passed = bool(torch.allclose(out, expected, atol=atol, rtol=rtol))
    return {
        **result,
        "status": "passed" if passed else "failed",
        "max_abs_error": max_abs_error,
    }


def run_gated_silu_sweep(
    *,
    output_dir: str | Path | None = None,
    arch: str = "compute_90",
    atol: float = 1.0e-5,
    rtol: float = 1.0e-5,
    device: int = 0,
    cases: list[dict] | None = None,
    skip_reason: Callable[[], str | None] | None = None,
) -> dict:
    resolved_output_dir = DEFAULT_OUTPUT_DIR if output_dir is None else Path(output_dir)
    if resolved_output_dir.is_absolute():
        raise ValueError("--output-dir must be repo-relative")

    case_specs = GATED_SILU_SWEEP_CASES if cases is None else cases
    case_results = []
    counts = {"passed": 0, "failed": 0, "skipped": 0}

    for index, case in enumerate(case_specs):
        case_name = str(case["name"])
        n = int(case["n"])
        provenance = str(case["provenance"])
        try:
            result = run_gated_silu_correctness(
                output_dir=resolved_output_dir / case_name,
                arch=arch,
                n=n,
                atol=atol,
                rtol=rtol,
                device=device,
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
            case_results.append(
                {
                    "case_index": index,
                    "case_name": case_name,
                    "shape": {"n": n},
                    "provenance": provenance,
                    "reference": GATED_SILU_REFERENCE,
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
        "kernel_name": "gated_silu_f32",
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
        parser.add_argument("--n", type=int, default=32)
        parser.add_argument("--atol", type=float, default=1.0e-5)
        parser.add_argument("--rtol", type=float, default=1.0e-5)
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
            result = run_gated_silu_sweep(
                output_dir=args.output_dir,
                arch=args.arch,
                atol=args.atol,
                rtol=args.rtol,
                device=args.device,
            )
        else:
            result = run_gated_silu_correctness(
                output_dir=args.output_dir,
                arch=args.arch,
                n=args.n,
                atol=args.atol,
                rtol=args.rtol,
                device=args.device,
            )
    except Exception as exc:
        result = {
            "schema_version": 1,
            "kernel_name": "gated_silu_f32",
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
        "provenance": provenance,
        "reference": result["reference"],
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
        if "=" in arg:
            option, value = arg.split("=", 1)
            path = Path(value)
            safe_args.append(f"{option}={path.name}" if path.is_absolute() else arg)
        else:
            path = Path(arg)
            safe_args.append(path.name if path.is_absolute() else arg)
    command = "examples/cuda/gluon_gated_silu_f32.py"
    if safe_args:
        command = f"{command} {shlex.join(safe_args)}"
    return command


if __name__ == "__main__":
    raise SystemExit(main())
