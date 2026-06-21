#!/usr/bin/env python3
"""Skip-safe correctness harness for generated Gluon speculative decoding."""

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


DEFAULT_OUTPUT_DIR = Path("tmp/gluon-speculative-decoding-local")
DEFAULT_DRAFT_TOKEN_IDS = [
    [10, 11, 12, 13],
    [20, 21, 22, 23],
]
DEFAULT_DRAFT_PROBABILITIES = [
    [0.50, 0.40, 0.20, 0.10],
    [0.50, 0.50, 0.40, 0.20],
]
DEFAULT_TARGET_PROBABILITIES = [
    [0.50, 0.40, 0.30, 0.20],
    [0.50, 0.10, 0.80, 0.40],
]
DEFAULT_THRESHOLDS = [
    [1.00, 0.90, 0.80, 0.70],
    [0.80, 0.30, 0.50, 0.50],
]
NON_CLAIMS = [
    "not FlashInfer integration evidence",
    "not vLLM or simpler-nv kernel integration evidence",
    "not DeepSeek serving correctness evidence",
    "not generated-text or tokenizer-semantics evidence",
    "not throughput or latency evidence",
]
ACCEPTANCE_RULE = (
    "accept while threshold <= min(1.0, target_probability / "
    "draft_probability); stop at first reject per row"
)


def build_speculative_accept_artifact(
    *,
    output_dir: str | Path | None = None,
    arch: str = "compute_90",
) -> GluonKernelArtifact:
    return KernelCompiler(platform="cuda").generate_gluon_kernel(
        "speculative_accept_f32",
        output_dir=DEFAULT_OUTPUT_DIR if output_dir is None else output_dir,
        arch=arch,
        tile_shape=(1, 1, 1),
    )


def speculative_decoding_skip_reason() -> str | None:
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


def compute_speculative_accept_cpu_golden(
    *,
    draft_token_ids: list[list[int]],
    draft_probabilities: list[list[float]],
    target_probabilities: list[list[float]],
    thresholds: list[list[float]],
) -> dict:
    rows, max_draft = _validate_fixture_shapes(
        draft_token_ids,
        draft_probabilities,
        target_probabilities,
        thresholds,
    )

    accepted_token_ids = []
    accept_mask = []
    accepted_counts = []
    for row in range(rows):
        row_token_ids = []
        row_mask = []
        accepted_count = 0
        accepting = True
        for pos in range(max_draft):
            draft_probability = float(draft_probabilities[row][pos])
            if draft_probability <= 0.0:
                raise ValueError("draft probabilities must be positive")
            target_probability = float(target_probabilities[row][pos])
            threshold = float(thresholds[row][pos])
            if threshold < 0.0 or threshold > 1.0:
                raise ValueError("thresholds must satisfy 0 <= threshold <= 1")

            accept_probability = min(1.0, target_probability / draft_probability)
            should_accept = accepting and threshold <= accept_probability
            if should_accept:
                row_token_ids.append(int(draft_token_ids[row][pos]))
                row_mask.append(1)
                accepted_count += 1
            else:
                row_token_ids.append(-1)
                row_mask.append(0)
                accepting = False

        accepted_token_ids.append(row_token_ids)
        accept_mask.append(row_mask)
        accepted_counts.append(accepted_count)

    return {
        "accepted_token_ids": accepted_token_ids,
        "accept_mask": accept_mask,
        "accepted_counts": accepted_counts,
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


def run_speculative_decoding_gpu(
    draft_token_ids: list[list[int]],
    draft_probabilities: list[list[float]],
    target_probabilities: list[list[float]],
    thresholds: list[list[float]],
    *,
    artifact: GluonKernelArtifact,
    device: int = 0,
) -> dict:
    import torch

    torch.cuda.set_device(device)
    cuda_device = torch.device("cuda", device)
    module = load_generated_module(artifact.source_path)

    draft_token_ids_tensor = torch.tensor(
        draft_token_ids,
        device=cuda_device,
        dtype=torch.int64,
    )
    draft_probabilities_tensor = torch.tensor(
        draft_probabilities,
        device=cuda_device,
        dtype=torch.float32,
    )
    target_probabilities_tensor = torch.tensor(
        target_probabilities,
        device=cuda_device,
        dtype=torch.float32,
    )
    thresholds_tensor = torch.tensor(thresholds, device=cuda_device, dtype=torch.float32)
    accepted_token_ids = torch.empty_like(draft_token_ids_tensor)
    accept_mask = torch.empty_like(draft_token_ids_tensor)
    accepted_counts = torch.empty((len(draft_token_ids),), device=cuda_device, dtype=torch.int64)

    module.run_speculative_accept_f32(
        draft_token_ids_tensor,
        draft_probabilities_tensor,
        target_probabilities_tensor,
        thresholds_tensor,
        accepted_token_ids,
        accept_mask,
        accepted_counts,
        max_draft=len(draft_token_ids[0]),
    )
    torch.cuda.synchronize(device)

    return {
        "accepted_token_ids": [
            [int(item) for item in row] for row in accepted_token_ids.cpu().tolist()
        ],
        "accept_mask": [[int(item) for item in row] for row in accept_mask.cpu().tolist()],
        "accepted_counts": [int(item) for item in accepted_counts.cpu().tolist()],
    }


def run_speculative_decoding_correctness(
    *,
    output_dir: str | Path | None = None,
    arch: str = "compute_90",
    rows: int = 2,
    max_draft: int = 4,
    device: int = 0,
    skip_reason: Callable[[], str | None] | None = None,
    gpu_runner: Callable[..., dict] | None = None,
) -> dict:
    resolved_output_dir = DEFAULT_OUTPUT_DIR if output_dir is None else Path(output_dir)
    if resolved_output_dir.is_absolute():
        raise ValueError("--output-dir must be repo-relative")
    if rows != 2 or max_draft != 4:
        raise ValueError("this review gate currently supports rows=2 and max_draft=4")

    draft_token_ids = [row[:] for row in DEFAULT_DRAFT_TOKEN_IDS]
    draft_probabilities = [row[:] for row in DEFAULT_DRAFT_PROBABILITIES]
    target_probabilities = [row[:] for row in DEFAULT_TARGET_PROBABILITIES]
    thresholds = [row[:] for row in DEFAULT_THRESHOLDS]
    cpu_golden = compute_speculative_accept_cpu_golden(
        draft_token_ids=draft_token_ids,
        draft_probabilities=draft_probabilities,
        target_probabilities=target_probabilities,
        thresholds=thresholds,
    )
    artifact = build_speculative_accept_artifact(output_dir=resolved_output_dir, arch=arch)
    result = {
        "schema_version": 1,
        "kernel_name": "speculative_accept_f32",
        "artifact": _artifact_payload(artifact),
        "shape": {"rows": rows, "max_draft": max_draft},
        "dtype": {
            "draft_token_ids": "int64",
            "draft_probabilities": "float32",
            "target_probabilities": "float32",
            "thresholds": "float32",
            "accepted_token_ids": "int64",
            "accept_mask": "int64",
            "accepted_counts": "int64",
        },
        "request": {
            "sampling_operator": "speculative-decoding-accept-reject",
            "acceptance_rule": ACCEPTANCE_RULE,
            "deterministic": True,
            "model_stack": "none",
        },
        "fixture": {
            "draft_token_ids": draft_token_ids,
            "draft_probabilities": draft_probabilities,
            "target_probabilities": target_probabilities,
            "thresholds": thresholds,
        },
        "cpu_golden": cpu_golden,
        "non_claims": NON_CLAIMS,
    }

    skip_check = speculative_decoding_skip_reason if skip_reason is None else skip_reason
    reason = skip_check()
    if reason is not None:
        return {**result, "status": "skipped", "reason": _clean_text(reason)}

    runner = run_speculative_decoding_gpu if gpu_runner is None else gpu_runner
    gpu_result = runner(
        draft_token_ids,
        draft_probabilities,
        target_probabilities,
        thresholds,
        artifact=artifact,
        device=device,
    )
    validation = _validate_gpu_result(cpu_golden, gpu_result)
    passed = (
        validation["accepted_token_ids_match"]
        and validation["accept_mask_match"]
        and validation["accepted_counts_match"]
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
        parser.add_argument("--max-draft", type=int, default=4)
        parser.add_argument("--device", type=int, default=0)
        parser.add_argument(
            "--require-cuda",
            action="store_true",
            help="return a non-zero status when dependencies or CUDA are unavailable",
        )
        args = parser.parse_args(raw_args)
        require_cuda = args.require_cuda

        result = run_speculative_decoding_correctness(
            output_dir=args.output_dir,
            arch=args.arch,
            rows=args.rows,
            max_draft=args.max_draft,
            device=args.device,
        )
    except Exception as exc:
        result = {
            "schema_version": 1,
            "kernel_name": "speculative_accept_f32",
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
    expected_token_ids = cpu_golden["accepted_token_ids"]
    actual_token_ids = gpu_result.get("accepted_token_ids", [])
    accepted_token_ids_match = (
        isinstance(actual_token_ids, list)
        and _nested_lengths_match(expected_token_ids, actual_token_ids)
        and actual_token_ids == expected_token_ids
    )

    expected_mask = cpu_golden["accept_mask"]
    actual_mask = gpu_result.get("accept_mask", [])
    accept_mask_match = (
        isinstance(actual_mask, list)
        and _nested_lengths_match(expected_mask, actual_mask)
        and actual_mask == expected_mask
    )

    expected_counts = cpu_golden["accepted_counts"]
    actual_counts = gpu_result.get("accepted_counts", [])
    accepted_counts_match = (
        isinstance(actual_counts, list)
        and len(actual_counts) == len(expected_counts)
        and actual_counts == expected_counts
    )

    return {
        "accepted_token_ids_match": accepted_token_ids_match,
        "accept_mask_match": accept_mask_match,
        "accepted_counts_match": accepted_counts_match,
    }


def _validate_fixture_shapes(
    draft_token_ids: list[list[int]],
    draft_probabilities: list[list[float]],
    target_probabilities: list[list[float]],
    thresholds: list[list[float]],
) -> tuple[int, int]:
    if not draft_token_ids or not draft_token_ids[0]:
        raise ValueError("draft_token_ids must be a non-empty rank-2 list")

    rows = len(draft_token_ids)
    max_draft = len(draft_token_ids[0])
    expected_shape = (rows, max_draft)
    for name, values in (
        ("draft_token_ids", draft_token_ids),
        ("draft_probabilities", draft_probabilities),
        ("target_probabilities", target_probabilities),
        ("thresholds", thresholds),
    ):
        if len(values) != rows:
            raise ValueError(f"{name} must have {rows} rows")
        for row in values:
            if len(row) != max_draft:
                raise ValueError(f"{name} must have shape {expected_shape}")

    return expected_shape


def _nested_lengths_match(expected: list[list[int]], actual: list[list[int]]) -> bool:
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
    command = "examples/cuda/gluon_speculative_decoding.py"
    if safe_args:
        command = f"{command} {shlex.join(safe_args)}"
    return command


if __name__ == "__main__":
    raise SystemExit(main())
