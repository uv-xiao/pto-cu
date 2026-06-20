#!/usr/bin/env python3
"""H200 microbenchmark harness for generated Triton/Gluon kernels."""

from __future__ import annotations

import argparse
import contextlib
import io
import importlib.util
import json
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


NON_CLAIMS = [
    "microbenchmark timings are not serving throughput",
    "skipped runs are not H200 performance evidence",
    "results cover only the listed shapes and dtypes",
]


@dataclass(frozen=True)
class BenchmarkSpec:
    kernel_name: str
    shape: dict[str, int]
    dtype: dict[str, str]
    tolerance: dict[str, float]
    runner: Callable[[Path, int, int, int], dict]


def h200_skip_reason() -> str | None:
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            import torch
    except Exception as exc:  # pragma: no cover - depends on local environment
        return f"torch import failed: {_clean_text(str(exc))}"

    if not torch.cuda.is_available():
        return "torch.cuda is not available"

    capability = torch.cuda.get_device_capability()
    if capability[0] != 9:
        return f"expected H200 capability 9.x, got {capability}"

    device_name = torch.cuda.get_device_name(0)
    if "H200" not in device_name:
        return f"expected H200 GPU, got {device_name}"

    return None


def run_benchmarks(
    *,
    output_dir: Path,
    arch: str,
    warmup: int,
    iterations: int,
    seed: int,
    command: str,
) -> dict:
    if output_dir.is_absolute():
        raise ValueError("--output-dir must be repo-relative")

    output_dir.mkdir(parents=True, exist_ok=True)
    hardware_skip = h200_skip_reason()
    benchmarks = [
        _run_or_skip(spec, output_dir, arch, warmup, iterations, seed, hardware_skip)
        for spec in _benchmark_specs()
    ]

    statuses = {entry["status"] for entry in benchmarks}
    if statuses == {"passed"}:
        status = "passed"
    elif "failed" in statuses:
        status = "failed"
    elif statuses == {"skipped"}:
        status = "skipped"
    else:
        status = "mixed"

    return {
        "schema_version": 1,
        "status": status,
        "machine_class": "H200",
        "command": command,
        "non_claims": NON_CLAIMS,
        "benchmarks": benchmarks,
    }


def _run_or_skip(
    spec: BenchmarkSpec,
    output_dir: Path,
    arch: str,
    warmup: int,
    iterations: int,
    seed: int,
    hardware_skip: str | None,
) -> dict:
    if hardware_skip is not None:
        artifact = _build_artifact(spec.kernel_name, output_dir, arch)
        return _skip_payload(spec, artifact, warmup, iterations, hardware_skip)

    try:
        return spec.runner(output_dir, warmup, iterations, seed)
    except SkipKernel as exc:
        artifact = _build_artifact(spec.kernel_name, output_dir, arch)
        return _skip_payload(spec, artifact, warmup, iterations, exc.reason)
    except Exception as exc:  # pragma: no cover - exercised on H200 failures
        return {
            **_base_payload(spec, None),
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": _clean_text(str(exc)),
            "correctness": {"measured": False, "status": "failed"},
            "timing": {
                "measured": False,
                "warmup": warmup,
                "iterations": iterations,
            },
        }


def _benchmark_specs() -> list[BenchmarkSpec]:
    return [
        BenchmarkSpec(
            kernel_name="gemm_f32",
            shape={"m": 128, "n": 128, "k": 64},
            dtype={"a": "float32", "b": "float32", "out": "float32"},
            tolerance={"atol": 1e-4, "rtol": 1e-4},
            runner=_run_gemm_f32,
        ),
        BenchmarkSpec(
            kernel_name="gemm_tensor_core_f16_f32",
            shape={"m": 64, "n": 32, "k": 32},
            dtype={"a": "float16", "b": "float16", "accumulator": "float32", "out": "float32"},
            tolerance={"atol": 1e-3, "rtol": 1e-1},
            runner=_run_tensor_core_gemm,
        ),
        BenchmarkSpec(
            kernel_name="gemm_tensor_core_tiled_f16_f32",
            shape={"m": 256, "n": 256, "k": 64},
            dtype={"a": "float16", "b": "float16", "accumulator": "float32", "out": "float32"},
            tolerance={"atol": 1e-3, "rtol": 1e-1},
            runner=_run_tiled_tensor_core_gemm,
        ),
        BenchmarkSpec(
            kernel_name="flashattention_fwd_f32",
            shape={"seqlen_q": 32, "seqlen_k": 32, "head_dim": 32},
            dtype={"q": "float32", "k": "float32", "v": "float32", "out": "float32"},
            tolerance={"atol": 1e-3, "rtol": 1e-2},
            runner=_run_flashattention,
        ),
    ]


def _run_gemm_f32(output_dir: Path, warmup: int, iterations: int, seed: int) -> dict:
    example = _load_sibling("gluon_gemm_f32.py")
    reason = example.gluon_gemm_skip_reason()
    if reason is not None:
        raise SkipKernel(reason)

    import torch

    artifact = example.build_gemm_artifact(output_dir=output_dir, arch="compute_90")
    module = example.load_generated_module(artifact.source_path)
    spec = _spec_by_name("gemm_f32")
    m, n, k = spec.shape["m"], spec.shape["n"], spec.shape["k"]

    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    a = torch.randn((m, k), device="cuda", dtype=torch.float32, generator=generator)
    b = torch.randn((k, n), device="cuda", dtype=torch.float32, generator=generator)
    c = torch.empty((m, n), device="cuda", dtype=torch.float32)

    def launch() -> None:
        module.gemm_f32_kernel[(m, n)](a, b, c, m, n, k, num_warps=4)

    launch()
    torch.cuda.synchronize()
    expected = a @ b
    torch.cuda.synchronize()
    return _passed_payload(
        spec,
        artifact,
        c,
        expected,
        _time_cuda(launch, warmup, iterations),
        _time_cuda(lambda: torch.mm(a, b), warmup, iterations),
    )


def _run_tensor_core_gemm(output_dir: Path, warmup: int, iterations: int, seed: int) -> dict:
    example = _load_sibling("gluon_gemm_tensor_core.py")
    reason = example.tensor_core_skip_reason()
    if reason is not None:
        raise SkipKernel(reason)

    import torch

    artifact = example.build_tensor_core_artifact(output_dir=output_dir, arch="compute_90")
    module = example.load_generated_module(artifact.source_path)
    spec = _spec_by_name("gemm_tensor_core_f16_f32")
    m, n, k = spec.shape["m"], spec.shape["n"], spec.shape["k"]

    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    a = torch.randn((m, k), device="cuda", dtype=torch.float16, generator=generator)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16, generator=generator)
    c = torch.zeros((m, n), device="cuda", dtype=torch.float32)
    d = torch.empty_like(c)

    def launch() -> None:
        module.run_gemm_tensor_core_f16_f32(a, b, c, d, instr_shape_n=16, num_warps=4)

    launch()
    torch.cuda.synchronize()
    expected = a @ b + c
    torch.cuda.synchronize()
    return _passed_payload(
        spec,
        artifact,
        d,
        expected,
        _time_cuda(launch, warmup, iterations),
        _time_cuda(lambda: torch.mm(a, b) + c, warmup, iterations),
    )


def _run_tiled_tensor_core_gemm(output_dir: Path, warmup: int, iterations: int, seed: int) -> dict:
    example = _load_sibling("gluon_gemm_tensor_core_tiled.py")
    reason = example.tensor_core_skip_reason()
    if reason is not None:
        raise SkipKernel(reason)

    import torch

    artifact = example.build_tiled_tensor_core_artifact(output_dir=output_dir, arch="compute_90")
    module = example.load_generated_module(artifact.source_path)
    spec = _spec_by_name("gemm_tensor_core_tiled_f16_f32")
    m, n, k = spec.shape["m"], spec.shape["n"], spec.shape["k"]

    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    a = torch.randn((m, k), device="cuda", dtype=torch.float16, generator=generator)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16, generator=generator)
    c = torch.zeros((m, n), device="cuda", dtype=torch.float32)
    d = torch.empty_like(c)

    def launch() -> None:
        module.run_gemm_tensor_core_tiled_f16_f32(
            a,
            b,
            c,
            d,
            instr_shape_n=16,
            num_warps=4,
        )

    launch()
    torch.cuda.synchronize()
    expected = a @ b + c
    torch.cuda.synchronize()
    return _passed_payload(
        spec,
        artifact,
        d,
        expected,
        _time_cuda(launch, warmup, iterations),
        _time_cuda(lambda: torch.mm(a, b) + c, warmup, iterations),
    )


def _run_flashattention(output_dir: Path, warmup: int, iterations: int, seed: int) -> dict:
    example = _load_sibling("gluon_flashattention_fwd.py")
    reason = example.flashattention_skip_reason()
    if reason is not None:
        raise SkipKernel(reason)

    import torch

    artifact = example.build_flashattention_artifact(output_dir=output_dir, arch="compute_90")
    module = example.load_generated_module(artifact.source_path)
    spec = _spec_by_name("flashattention_fwd_f32")
    seqlen_q = spec.shape["seqlen_q"]
    seqlen_k = spec.shape["seqlen_k"]
    head_dim = spec.shape["head_dim"]

    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    q = torch.randn((seqlen_q, head_dim), device="cuda", dtype=torch.float32, generator=generator)
    k = torch.randn((seqlen_k, head_dim), device="cuda", dtype=torch.float32, generator=generator)
    v = torch.randn((seqlen_k, head_dim), device="cuda", dtype=torch.float32, generator=generator)
    out = torch.empty((seqlen_q, head_dim), device="cuda", dtype=torch.float32)
    scale = head_dim**-0.5

    def launch() -> None:
        module.run_flashattention_fwd_f32(q, k, v, out, scale=scale, num_warps=4)

    def reference():
        return torch.softmax((q @ k.T) * scale, dim=-1) @ v

    launch()
    torch.cuda.synchronize()
    expected = reference()
    torch.cuda.synchronize()
    return _passed_payload(
        spec,
        artifact,
        out,
        expected,
        _time_cuda(launch, warmup, iterations),
        _time_cuda(reference, warmup, iterations),
    )


def _passed_payload(
    spec: BenchmarkSpec,
    artifact,
    actual,
    expected,
    kernel_timing: dict,
    reference_timing: dict,
) -> dict:
    import torch

    max_abs_error = float((actual - expected).abs().max().item())
    passed = bool(
        torch.allclose(
            actual,
            expected,
            atol=spec.tolerance["atol"],
            rtol=spec.tolerance["rtol"],
        )
    )
    return {
        **_base_payload(spec, artifact),
        "status": "passed" if passed else "failed",
        "correctness": {
            "measured": True,
            "status": "passed" if passed else "failed",
            "max_abs_error": max_abs_error,
        },
        "timing": {
            "measured": True,
            "generated_kernel": kernel_timing,
            "pytorch_reference": reference_timing,
        },
    }


def _skip_payload(
    spec: BenchmarkSpec,
    artifact,
    warmup: int,
    iterations: int,
    reason: str,
) -> dict:
    return {
        **_base_payload(spec, artifact),
        "status": "skipped",
        "reason": _clean_text(reason),
        "correctness": {"measured": False, "status": "skipped"},
        "timing": {
            "measured": False,
            "warmup": warmup,
            "iterations": iterations,
        },
    }


def _base_payload(spec: BenchmarkSpec, artifact) -> dict:
    payload = {
        "kernel_name": spec.kernel_name,
        "machine_class": "H200",
        "shape": spec.shape,
        "dtype": spec.dtype,
        "tolerance": spec.tolerance,
    }
    if artifact is not None:
        payload["artifact"] = _artifact_payload(artifact)
    return payload


def _artifact_payload(artifact) -> dict:
    return {
        "arch": artifact.arch,
        "compiler_role": artifact.compiler_role,
        "manifest_path": _relative_path(artifact.manifest_path),
        "source_path": _relative_path(artifact.source_path),
        "source_sha256": artifact.source_sha256,
        "tile_shape": list(artifact.tile_shape),
    }


def _time_cuda(fn: Callable[[], object], warmup: int, iterations: int) -> dict:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))

    return {
        "unit": "ms",
        "warmup": warmup,
        "iterations": iterations,
        "mean": sum(samples) / len(samples),
        "min": min(samples),
        "max": max(samples),
    }


def _build_artifact(kernel_name: str, output_dir: Path, arch: str):
    if kernel_name == "gemm_f32":
        return _load_sibling("gluon_gemm_f32.py").build_gemm_artifact(
            output_dir=output_dir,
            arch=arch,
        )
    if kernel_name == "gemm_tensor_core_f16_f32":
        return _load_sibling("gluon_gemm_tensor_core.py").build_tensor_core_artifact(
            output_dir=output_dir,
            arch=arch,
        )
    if kernel_name == "gemm_tensor_core_tiled_f16_f32":
        return _load_sibling("gluon_gemm_tensor_core_tiled.py").build_tiled_tensor_core_artifact(
            output_dir=output_dir,
            arch=arch,
        )
    if kernel_name == "flashattention_fwd_f32":
        return _load_sibling("gluon_flashattention_fwd.py").build_flashattention_artifact(
            output_dir=output_dir,
            arch=arch,
        )
    raise AssertionError(f"unknown benchmark kernel: {kernel_name}")


def _spec_by_name(kernel_name: str) -> BenchmarkSpec:
    for spec in _benchmark_specs():
        if spec.kernel_name == kernel_name:
            return spec
    raise AssertionError(f"unknown benchmark kernel: {kernel_name}")


def _load_sibling(filename: str):
    source_path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(source_path.stem, source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {source_path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
        if path.is_absolute():
            safe_args.append(path.name)
        else:
            safe_args.append(arg)
    command = "examples/cuda/gluon_benchmark.py"
    if safe_args:
        command = f"{command} {shlex.join(safe_args)}"
    return command


class SkipKernel(Exception):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    command = _display_command(raw_args)
    require_cuda = False

    try:
        parser = JsonArgumentParser(description=__doc__)
        parser.add_argument("--output-dir", type=Path, default=Path("tmp/gluon-performance-h200"))
        parser.add_argument("--arch", default="compute_90")
        parser.add_argument("--warmup", type=int, default=5)
        parser.add_argument("--iterations", type=int, default=20)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument(
            "--require-cuda",
            action="store_true",
            help="return a non-zero status when any benchmark skips",
        )
        args = parser.parse_args(raw_args)
        require_cuda = args.require_cuda
        result = run_benchmarks(
            output_dir=args.output_dir,
            arch=args.arch,
            warmup=args.warmup,
            iterations=args.iterations,
            seed=args.seed,
            command=command,
        )
    except Exception as exc:
        result = {
            "schema_version": 1,
            "status": "failed",
            "machine_class": "H200",
            "command": command,
            "non_claims": NON_CLAIMS,
            "error_type": type(exc).__name__,
            "error": _clean_text(str(exc)),
        }

    print(json.dumps(result, indent=2, sort_keys=True))

    if result["status"] == "failed":
        return 1
    if result["status"] in {"skipped", "mixed"} and require_cuda:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
