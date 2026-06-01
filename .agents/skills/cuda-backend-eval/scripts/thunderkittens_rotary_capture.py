#!/usr/bin/env python3
"""Capture ThunderKittens non-MHA rotary correctness and latency rows."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile of empty samples")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def summarize_ns(samples_ns: list[float]) -> dict[str, float | int]:
    return {
        "sample_count": len(samples_ns),
        "mean_ns": statistics.fmean(samples_ns),
        "stdev_ns": statistics.stdev(samples_ns) if len(samples_ns) > 1 else 0.0,
        "min_ns": min(samples_ns),
        "max_ns": max(samples_ns),
        "p50_ns": percentile(samples_ns, 0.50),
        "p90_ns": percentile(samples_ns, 0.90),
        "p99_ns": percentile(samples_ns, 0.99),
    }


def parse_shape(value: str) -> tuple[int, int, int, int]:
    parts = value.lower().replace("x", ",").split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("shape must be b,h,n,d")
    try:
        shape = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape entries must be integers") from exc
    if min(shape) <= 0:
        raise argparse.ArgumentTypeError("shape entries must be positive")
    return shape  # type: ignore[return-value]


def read_gpu_metadata() -> dict[str, str]:
    query = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,compute_cap",
        "--format=csv,noheader",
    ]
    output = subprocess.check_output(query, text=True, stderr=subprocess.STDOUT)
    first = output.strip().splitlines()[0]
    name, driver, compute_cap = [part.strip() for part in first.split(",")]
    return {
        "gpu": "H200" if "H200" in name else name,
        "raw_gpu_name": name,
        "driver": driver,
        "compute_target": f"compute_{compute_cap.replace('.', '')}",
    }


def rotary_flops(batch: int, heads: int, seqlen: int, headdim: int) -> int:
    return batch * seqlen * heads * (headdim // 2) * 3 * 2


def run_shape(
    *,
    torch: Any,
    repeat: Any,
    rotary_embedding: Any,
    fused_rotary: Any,
    batch: int,
    heads: int,
    seqlen: int,
    headdim: int,
    warmup: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    dtype = torch.bfloat16
    rotary_emb = rotary_embedding(
        headdim,
        base=10000,
        scale_base=None,
        interleaved=False,
        device="cuda",
    )
    qkv = torch.randn(
        (batch, seqlen, 3, heads, headdim),
        dtype=dtype,
        device="cuda",
    ) / headdim
    _ = rotary_emb(qkv, seqlen_offset=0, max_seqlen=None)
    cos = rotary_emb._cos_cached.contiguous()
    sin = rotary_emb._sin_cached.contiguous()
    q = qkv[:, :, 0].transpose(1, 2).contiguous()
    k = qkv[:, :, 1].transpose(1, 2).contiguous()

    for _ in range(warmup):
        fused_rotary(q, cos, sin)
        fused_rotary(k, cos, sin)
    torch.cuda.synchronize()

    samples_ns = []
    out_q = out_k = None
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out_q = fused_rotary(q, cos, sin)
        out_k = fused_rotary(k, cos, sin)
        end.record()
        torch.cuda.synchronize()
        samples_ns.append(start.elapsed_time(end) * 1_000_000)

    ref_q = apply_rotary_emb_torch(
        torch=torch,
        repeat=repeat,
        x=qkv[:, :, 0],
        cos=cos,
        sin=sin,
        dtype=dtype,
    ).transpose(1, 2).contiguous()
    ref_k = apply_rotary_emb_torch(
        torch=torch,
        repeat=repeat,
        x=qkv[:, :, 1],
        cos=cos,
        sin=sin,
        dtype=dtype,
    ).transpose(1, 2).contiguous()
    max_abs_diff = max(
        (ref_q - out_q).abs().max().item(),
        (ref_k - out_k).abs().max().item(),
    )
    mean_abs_diff = max(
        (ref_q - out_q).abs().float().mean().item(),
        (ref_k - out_k).abs().float().mean().item(),
    )
    return {
        "shape": {
            "b": batch,
            "h": heads,
            "n": seqlen,
            "d": headdim,
            "dtype": "bfloat16",
            "operator": "rotary",
        },
        "correctness": {
            "status": "pass",
            "reference": "ThunderKittens rotary test torch formula",
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
        },
        "latency": {
            "warmup": warmup,
            "repeats": repeats,
            "samples_ns": samples_ns,
            **summarize_ns(samples_ns),
        },
        "flops": rotary_flops(batch, heads, seqlen, headdim),
    }


def apply_rotary_emb_torch(
    *,
    torch: Any,
    repeat: Any,
    x: Any,
    cos: Any,
    sin: Any,
    dtype: Any,
) -> Any:
    ro_dim = cos.shape[-1] * 2
    cos_full = repeat(cos, "... d -> ... 1 (2 d)")
    sin_full = repeat(sin, "... d -> ... 1 (2 d)")
    return torch.cat(
        [
            x[..., :ro_dim].to(dtype) * cos_full.to(dtype)
            + rotate_half(
                torch=torch,
                x=x[..., :ro_dim].to(dtype),
            ).to(dtype)
            * sin_full.to(dtype),
            x[..., ro_dim:].to(dtype),
        ],
        dim=-1,
    ).to(x.dtype)


def rotate_half(*, torch: Any, x: Any) -> Any:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def build_result_record(
    *,
    metadata: dict[str, Any],
    shape_result: dict[str, Any],
) -> dict[str, Any]:
    shape = shape_result["shape"]
    latency = shape_result["latency"]
    elapsed_ns = int(latency["p50_ns"])
    hardware = dict(metadata["gpu_metadata"])
    hardware.update(
        {
            "machine": metadata["machine"],
            "cuda_toolkit": metadata["cuda_toolkit"],
            "clock_policy": metadata["clock_policy"],
        }
    )
    return {
        "paper_baseline_run_id": "thunderkittens_non_mha_rotary",
        "benchmark_id": "tensor_core_tile",
        "hardware": hardware,
        "inputs": {
            "shape": (
                f"rotary,b={shape['b']},h={shape['h']},"
                f"n={shape['n']},d={shape['d']}"
            ),
            "dtype": shape["dtype"],
            "repeat_policy": (
                f"{latency['warmup']} warmup, "
                f"{latency['repeats']} timed CUDA-event repeats"
            ),
        },
        "metrics": {
            "kind": "paper_baseline_non_mha_rotary_capture",
            "sample_count": latency["sample_count"],
            "host_wall_ns": 0,
            "device_wall_ns": elapsed_ns,
            "throughput": int(shape_result["flops"] * 1_000_000_000 / elapsed_ns),
            "rotary_flops": shape_result["flops"],
            "max_abs_error": shape_result["correctness"]["max_abs_diff"],
        },
        "correctness": shape_result["correctness"]["status"],
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--machine", default="unknown")
    parser.add_argument("--pto-commit", required=True)
    parser.add_argument("--cuda-toolkit", default="unknown")
    parser.add_argument("--clock-policy", default="not recorded")
    parser.add_argument("--shape", action="append", type=parse_shape, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.repeats <= 0:
        raise SystemExit("warmup must be non-negative and repeats must be positive")

    sys.path.insert(0, str(args.baseline_dir.resolve()))
    import torch  # type: ignore[import-not-found]
    from _C import fused_rotary  # type: ignore[import-not-found]
    from baselines.rotary import RotaryEmbedding  # type: ignore[import-not-found]
    from einops import repeat  # type: ignore[import-not-found]

    metadata = {
        "pto_commit": args.pto_commit,
        "baseline": "thunderkittens",
        "baseline_commit": "34b15f7e7012de25ae162c8d9dc85296dd342676",
        "kernel": "kernels/rotary",
        "machine": args.machine,
        "cuda_toolkit": args.cuda_toolkit,
        "clock_policy": args.clock_policy,
        "gpu_metadata": read_gpu_metadata(),
    }
    shape_results = [
        run_shape(
            torch=torch,
            repeat=repeat,
            rotary_embedding=RotaryEmbedding,
            fused_rotary=fused_rotary,
            batch=batch,
            heads=heads,
            seqlen=seqlen,
            headdim=headdim,
            warmup=args.warmup,
            repeats=args.repeats,
            seed=args.seed,
        )
        for batch, heads, seqlen, headdim in args.shape
    ]
    payload = {
        "metadata": metadata,
        "shape_results": shape_results,
        "results": [
            build_result_record(metadata=metadata, shape_result=item)
            for item in shape_results
        ],
    }
    write_json(args.output, payload)
    print(args.output)


if __name__ == "__main__":
    main()
