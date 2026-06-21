#!/usr/bin/env python3
"""Skip-safe correctness harness for generated Gluon top-k sampling."""

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


DEFAULT_OUTPUT_DIR = Path("tmp/gluon-topk-sampling-local")
DEFAULT_LOGITS = [
    [0.1, 0.9, 0.2, 0.9, -0.4, 0.7, 0.3, 0.8],
    [-1.0, -0.5, 0.0, 0.25, 0.25, 0.1, -0.2, 0.9],
]
SHAPE_FIXTURES = {
    (2, 8): DEFAULT_LOGITS,
    (3, 16): [
        [
            0.1,
            -0.2,
            1.5,
            1.5,
            -3.0,
            0.7,
            0.7,
            2.0,
            -0.5,
            1.2,
            0.0,
            2.0,
            -1.0,
            0.4,
            1.2,
            -2.0,
        ],
        [
            -1.0,
            -0.1,
            -0.1,
            -0.1,
            -2.5,
            3.0,
            2.0,
            3.0,
            0.0,
            2.0,
            -4.0,
            1.5,
            1.5,
            -0.2,
            0.8,
            0.8,
        ],
        [
            0.0,
            -0.5,
            0.0,
            4.0,
            4.0,
            -1.0,
            3.5,
            3.5,
            2.2,
            -3.0,
            2.2,
            1.1,
            1.1,
            1.1,
            -0.1,
            3.5,
        ],
    ],
}
NON_CLAIMS = [
    "not FlashInfer integration evidence",
    "not vLLM or simpler-nv kernel integration evidence",
    "not DeepSeek serving correctness evidence",
    "not generated-text or tokenizer-semantics evidence",
    "not throughput or latency evidence",
]


def build_topk_sampling_artifact(
    *,
    output_dir: str | Path | None = None,
    arch: str = "compute_90",
) -> GluonKernelArtifact:
    return KernelCompiler(platform="cuda").generate_gluon_kernel(
        "topk_sampling_f32",
        output_dir=DEFAULT_OUTPUT_DIR if output_dir is None else output_dir,
        arch=arch,
        tile_shape=(1, 1, 1),
    )


def topk_sampling_skip_reason() -> str | None:
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


def compute_topk_cpu_golden(logits: list[list[float]], *, k: int) -> dict:
    if k <= 0:
        raise ValueError("--k must be positive")
    if not logits or not logits[0]:
        raise ValueError("logits must be a non-empty rank-2 list")

    vocab = len(logits[0])
    if k > vocab:
        raise ValueError("--k must be less than or equal to vocab")

    values = []
    indices = []
    for row in logits:
        if len(row) != vocab:
            raise ValueError("all logits rows must have the same vocab length")
        ordered = sorted(range(vocab), key=lambda token_id: (-row[token_id], token_id))
        top_indices = ordered[:k]
        indices.append(top_indices)
        values.append([float(row[token_id]) for token_id in top_indices])
    return {"values": values, "indices": indices}


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


def run_topk_sampling_gpu(
    logits: list[list[float]],
    *,
    k: int,
    artifact: GluonKernelArtifact,
    device: int = 0,
) -> dict:
    import torch

    torch.cuda.set_device(device)
    cuda_device = torch.device("cuda", device)
    module = load_generated_module(artifact.source_path)

    logits_tensor = torch.tensor(logits, device=cuda_device, dtype=torch.float32)
    top_values = torch.empty((len(logits), k), device=cuda_device, dtype=torch.float32)
    top_indices = torch.empty((len(logits), k), device=cuda_device, dtype=torch.int64)

    module.run_topk_sampling_f32(logits_tensor, top_values, top_indices, k=k)
    torch.cuda.synchronize(device)

    return {
        "values": _round_nested_float_list(top_values.cpu().tolist()),
        "indices": [[int(item) for item in row] for row in top_indices.cpu().tolist()],
    }


def run_topk_sampling_correctness(
    *,
    output_dir: str | Path | None = None,
    arch: str = "compute_90",
    rows: int = 2,
    vocab: int = 8,
    k: int = 3,
    device: int = 0,
    skip_reason: Callable[[], str | None] | None = None,
    gpu_runner: Callable[..., dict] | None = None,
) -> dict:
    resolved_output_dir = DEFAULT_OUTPUT_DIR if output_dir is None else Path(output_dir)
    if resolved_output_dir.is_absolute():
        raise ValueError("--output-dir must be repo-relative")

    logits = _logits_for_shape(rows=rows, vocab=vocab)
    cpu_golden = compute_topk_cpu_golden(logits, k=k)
    artifact = build_topk_sampling_artifact(output_dir=resolved_output_dir, arch=arch)
    result = {
        "schema_version": 1,
        "kernel_name": "topk_sampling_f32",
        "artifact": _artifact_payload(artifact),
        "shape": {"rows": rows, "vocab": vocab, "k": k},
        "dtype": "float32",
        "request": {
            "sampling_operator": "top-k",
            "deterministic": True,
            "tie_break": "lower token id first",
        },
        "cpu_golden": cpu_golden,
        "non_claims": NON_CLAIMS,
    }

    skip_check = topk_sampling_skip_reason if skip_reason is None else skip_reason
    reason = skip_check()
    if reason is not None:
        return {**result, "status": "skipped", "reason": _clean_text(reason)}

    runner = run_topk_sampling_gpu if gpu_runner is None else gpu_runner
    gpu_result = runner(logits, k=k, artifact=artifact, device=device)
    validation = _validate_gpu_result(cpu_golden, gpu_result)
    passed = validation["values_match"] and validation["indices_match"]
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
        parser.add_argument("--k", type=int, default=3)
        parser.add_argument("--device", type=int, default=0)
        parser.add_argument(
            "--require-cuda",
            action="store_true",
            help="return a non-zero status when dependencies or CUDA are unavailable",
        )
        args = parser.parse_args(raw_args)
        require_cuda = args.require_cuda

        result = run_topk_sampling_correctness(
            output_dir=args.output_dir,
            arch=args.arch,
            rows=args.rows,
            vocab=args.vocab,
            k=args.k,
            device=args.device,
        )
    except Exception as exc:
        result = {
            "schema_version": 1,
            "kernel_name": "topk_sampling_f32",
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
    values_shape_match = _nested_shape_matches(
        cpu_golden["values"],
        gpu_result.get("values"),
    )
    indices_shape_match = _nested_shape_matches(
        cpu_golden["indices"],
        gpu_result.get("indices"),
    )

    max_abs_error = 0.0
    values_match = values_shape_match
    if values_shape_match:
        for expected_row, actual_row in zip(cpu_golden["values"], gpu_result["values"]):
            for expected, actual in zip(expected_row, actual_row):
                error = abs(float(actual) - float(expected))
                max_abs_error = max(max_abs_error, error)
                if error != 0.0:
                    values_match = False

    indices_match = (
        indices_shape_match
        and gpu_result.get("indices") == cpu_golden["indices"]
    )
    return {
        "values_shape_match": values_shape_match,
        "indices_shape_match": indices_shape_match,
        "values_match": values_match,
        "indices_match": indices_match,
        "max_abs_error": max_abs_error,
    }


def _logits_for_shape(*, rows: int, vocab: int) -> list[list[float]]:
    try:
        logits = SHAPE_FIXTURES[(rows, vocab)]
    except KeyError as exc:
        supported = ", ".join(
            f"rows={fixture_rows}, vocab={fixture_vocab}"
            for fixture_rows, fixture_vocab in sorted(SHAPE_FIXTURES)
        )
        raise ValueError(
            f"unsupported deterministic Top-K fixture rows={rows}, vocab={vocab}; "
            f"supported fixtures: {supported}"
        ) from exc
    return [row[:] for row in logits]


def _nested_shape_matches(expected: object, actual: object) -> bool:
    if not isinstance(expected, list) or not isinstance(actual, list):
        return False
    if len(expected) != len(actual):
        return False
    for expected_row, actual_row in zip(expected, actual):
        if not isinstance(expected_row, list) or not isinstance(actual_row, list):
            return False
        if len(expected_row) != len(actual_row):
            return False
    return True


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
    command = "examples/cuda/gluon_topk_sampling.py"
    if safe_args:
        command = f"{command} {shlex.join(safe_args)}"
    return command


if __name__ == "__main__":
    raise SystemExit(main())
