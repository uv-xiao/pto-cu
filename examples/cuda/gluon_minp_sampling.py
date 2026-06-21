#!/usr/bin/env python3
"""Skip-safe correctness harness for generated Gluon min-p sampling."""

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


DEFAULT_OUTPUT_DIR = Path("tmp/gluon-minp-sampling-local")
DEFAULT_PROBABILITIES = [
    [0.05, 0.30, 0.10, 0.20, 0.05, 0.15, 0.05, 0.10],
    [0.25, 0.05, 0.25, 0.05, 0.10, 0.05, 0.15, 0.10],
]
NON_CLAIMS = [
    "not FlashInfer integration evidence",
    "not vLLM or simpler-nv kernel integration evidence",
    "not DeepSeek serving correctness evidence",
    "not generated-text or tokenizer-semantics evidence",
    "not throughput or latency evidence",
]


def build_minp_sampling_artifact(
    *,
    output_dir: str | Path | None = None,
    arch: str = "compute_90",
) -> GluonKernelArtifact:
    return KernelCompiler(platform="cuda").generate_gluon_kernel(
        "minp_sampling_f32",
        output_dir=DEFAULT_OUTPUT_DIR if output_dir is None else output_dir,
        arch=arch,
        tile_shape=(1, 1, 1),
    )


def minp_sampling_skip_reason() -> str | None:
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


def compute_minp_cpu_golden(
    probabilities: list[list[float]],
    *,
    min_p: float,
    max_k: int,
) -> dict:
    if min_p <= 0.0 or min_p > 1.0:
        raise ValueError("--min-p must satisfy 0 < min_p <= 1")
    if max_k <= 0:
        raise ValueError("--max-k must be positive")
    if not probabilities or not probabilities[0]:
        raise ValueError("probabilities must be a non-empty rank-2 list")

    vocab = len(probabilities[0])
    if max_k > vocab:
        raise ValueError("--max-k must be less than or equal to vocab")

    values = []
    indices = []
    selected_counts = []
    for row in probabilities:
        if len(row) != vocab:
            raise ValueError("all probability rows must have the same vocab length")
        row_sum = sum(float(item) for item in row)
        if abs(row_sum - 1.0) > 1e-6:
            raise ValueError("probability rows must sum to one")

        threshold = min_p * max(float(item) for item in row)
        selected = [
            token_id
            for token_id in range(vocab)
            if float(row[token_id]) >= threshold
        ]
        ordered = sorted(selected, key=lambda token_id: (-row[token_id], token_id))
        row_indices = [int(token_id) for token_id in ordered[:max_k]]
        row_values = [round(float(row[token_id]), 7) for token_id in row_indices]
        selected_count = len(row_indices)

        while len(row_values) < max_k:
            row_values.append(0.0)
            row_indices.append(-1)

        values.append(row_values)
        indices.append(row_indices)
        selected_counts.append(selected_count)

    return {
        "values": values,
        "indices": indices,
        "selected_counts": selected_counts,
    }


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


def run_minp_sampling_gpu(
    probabilities: list[list[float]],
    *,
    min_p: float,
    max_k: int,
    artifact: GluonKernelArtifact,
    device: int = 0,
) -> dict:
    import torch

    torch.cuda.set_device(device)
    cuda_device = torch.device("cuda", device)
    module = load_generated_module(artifact.source_path)

    probabilities_tensor = torch.tensor(
        probabilities,
        device=cuda_device,
        dtype=torch.float32,
    )
    top_values = torch.empty((len(probabilities), max_k), device=cuda_device, dtype=torch.float32)
    top_indices = torch.empty((len(probabilities), max_k), device=cuda_device, dtype=torch.int64)
    selected_counts = torch.empty((len(probabilities),), device=cuda_device, dtype=torch.int64)

    module.run_minp_sampling_f32(
        probabilities_tensor,
        top_values,
        top_indices,
        selected_counts,
        min_p=min_p,
        max_k=max_k,
    )
    torch.cuda.synchronize(device)

    return {
        "values": _round_nested_float_list(top_values.cpu().tolist()),
        "indices": [[int(item) for item in row] for row in top_indices.cpu().tolist()],
        "selected_counts": [int(item) for item in selected_counts.cpu().tolist()],
    }


def run_minp_sampling_correctness(
    *,
    output_dir: str | Path | None = None,
    arch: str = "compute_90",
    rows: int = 2,
    vocab: int = 8,
    max_k: int = 5,
    min_p: float = 0.5,
    device: int = 0,
    skip_reason: Callable[[], str | None] | None = None,
    gpu_runner: Callable[..., dict] | None = None,
) -> dict:
    resolved_output_dir = DEFAULT_OUTPUT_DIR if output_dir is None else Path(output_dir)
    if resolved_output_dir.is_absolute():
        raise ValueError("--output-dir must be repo-relative")
    if rows != 2 or vocab != 8:
        raise ValueError("this review gate currently supports rows=2 and vocab=8")

    probabilities = [row[:] for row in DEFAULT_PROBABILITIES]
    cpu_golden = compute_minp_cpu_golden(probabilities, min_p=min_p, max_k=max_k)
    artifact = build_minp_sampling_artifact(output_dir=resolved_output_dir, arch=arch)
    result = {
        "schema_version": 1,
        "kernel_name": "minp_sampling_f32",
        "artifact": _artifact_payload(artifact),
        "shape": {"rows": rows, "vocab": vocab, "max_k": max_k},
        "dtype": "float32",
        "request": {
            "sampling_operator": "min-p",
            "probabilities_already_normalized": True,
            "min_p": min_p,
            "threshold_rule": "probability >= min_p * row_max_probability",
            "deterministic": True,
            "tie_break": "lower token id first",
        },
        "cpu_golden": cpu_golden,
        "non_claims": NON_CLAIMS,
    }

    skip_check = minp_sampling_skip_reason if skip_reason is None else skip_reason
    reason = skip_check()
    if reason is not None:
        return {**result, "status": "skipped", "reason": _clean_text(reason)}

    runner = run_minp_sampling_gpu if gpu_runner is None else gpu_runner
    gpu_result = runner(probabilities, min_p=min_p, max_k=max_k, artifact=artifact, device=device)
    validation = _validate_gpu_result(cpu_golden, gpu_result)
    passed = (
        validation["values_match"]
        and validation["indices_match"]
        and validation["selected_counts_match"]
    )
    return {
        **result,
        "status": "passed" if passed else "failed",
        "gpu_result": gpu_result,
        "validation": validation,
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
        parser.add_argument("--rows", type=int, default=2)
        parser.add_argument("--vocab", type=int, default=8)
        parser.add_argument("--max-k", type=int, default=5)
        parser.add_argument("--min-p", type=float, default=0.5)
        parser.add_argument("--device", type=int, default=0)
        parser.add_argument(
            "--require-cuda",
            action="store_true",
            help="return a non-zero status when dependencies or CUDA are unavailable",
        )
        args = parser.parse_args(raw_args)
        require_cuda = args.require_cuda

        result = run_minp_sampling_correctness(
            output_dir=args.output_dir,
            arch=args.arch,
            rows=args.rows,
            vocab=args.vocab,
            max_k=args.max_k,
            min_p=args.min_p,
            device=args.device,
        )
    except Exception as exc:
        result = {
            "schema_version": 1,
            "kernel_name": "minp_sampling_f32",
            "status": "failed",
            "command": _display_command(raw_args),
            "error_type": type(exc).__name__,
            "error": _clean_text(str(exc)),
            "non_claims": NON_CLAIMS,
        }

    print(json.dumps(result, indent=2, sort_keys=True))

    if result["status"] == "failed":
        return 1
    if result["status"] == "skipped" and require_cuda:
        return 2
    return 0


def _validate_gpu_result(cpu_golden: dict, gpu_result: dict) -> dict:
    max_abs_error = 0.0
    expected_values = cpu_golden["values"]
    actual_values = gpu_result.get("values", [])
    if not isinstance(actual_values, list):
        actual_values = []
    values_match = _nested_lengths_match(expected_values, actual_values)
    for expected_row, actual_row in zip(expected_values, actual_values):
        if not isinstance(actual_row, list):
            continue
        for expected, actual in zip(expected_row, actual_row):
            error = abs(float(actual) - float(expected))
            max_abs_error = max(max_abs_error, error)
            if error != 0.0:
                values_match = False

    expected_indices = cpu_golden["indices"]
    actual_indices = gpu_result.get("indices", [])
    indices_match = (
        isinstance(actual_indices, list)
        and _nested_lengths_match(expected_indices, actual_indices)
        and actual_indices == expected_indices
    )

    expected_counts = cpu_golden["selected_counts"]
    actual_counts = gpu_result.get("selected_counts", [])
    selected_counts_match = (
        isinstance(actual_counts, list)
        and len(actual_counts) == len(expected_counts)
        and actual_counts == expected_counts
    )

    return {
        "values_match": values_match,
        "indices_match": indices_match,
        "selected_counts_match": selected_counts_match,
        "max_abs_error": max_abs_error,
    }


def _nested_lengths_match(expected: list[list[float]], actual: list[list[float]]) -> bool:
    if len(actual) != len(expected):
        return False
    return all(
        isinstance(actual_row, list) and len(actual_row) == len(expected_row)
        for expected_row, actual_row in zip(expected, actual)
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


def _round_nested_float_list(values: list[list[float]]) -> list[list[float]]:
    return [[round(float(item), 7) for item in row] for row in values]


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
    command = "examples/cuda/gluon_minp_sampling.py"
    if safe_args:
        command = f"{command} {shlex.join(safe_args)}"
    return command


if __name__ == "__main__":
    raise SystemExit(main())
