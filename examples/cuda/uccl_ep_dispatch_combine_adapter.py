#!/usr/bin/env python3
"""UCCL-EP dispatch/combine smoke through the private PTO descriptor boundary."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from simpler_setup.cuda_comm import (
    UcclEpDispatchCombineDescriptor,
    create_cuda_comm_host_plan,
    create_uccl_ep_dispatch_combine_descriptor,
)

OPERATION = "ep_dispatch_combine"
TRANSPORT = "ep"
_UCCL_EP_BENCH_ENV = "UCCL_EP_BENCH_DIR"


def parse_device_ids(value: str) -> tuple[int, ...]:
    device_ids = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(device_ids) < 2:
        raise argparse.ArgumentTypeError("expected at least two comma-separated device ids")
    if len(set(device_ids)) != len(device_ids):
        raise argparse.ArgumentTypeError("device ids must be unique")
    return device_ids


def resolve_uccl_ep_bench_dir(bench_dir: str | os.PathLike[str] | None = None) -> Path | None:
    candidates: list[Path] = []
    if bench_dir is not None:
        candidates.append(Path(bench_dir))
    env_value = os.environ.get(_UCCL_EP_BENCH_ENV)
    if env_value:
        candidates.append(Path(env_value))
    candidates.append(
        Path(__file__).resolve().parents[2]
        / "tmp"
        / "sources"
        / "repos"
        / "communication"
        / "uccl"
        / "ep"
        / "bench"
    )

    for candidate in candidates:
        if (candidate / "buffer.py").is_file() and (candidate / "utils.py").is_file():
            return candidate
    return None


def uccl_ep_skip_reason(*, bench_dir: str | os.PathLike[str] | None = None) -> str | None:
    if importlib.util.find_spec("uccl") is None:
        return "UCCL-EP dependencies unavailable: ModuleNotFoundError: No module named 'uccl'"

    try:
        import torch  # noqa: PLC0415
        import torch.distributed as dist  # noqa: PLC0415
        import uccl.ep  # noqa: F401, PLC0415
    except Exception as exc:  # pragma: no cover - depends on optional packages
        return f"UCCL-EP dependencies unavailable: {type(exc).__name__}: {exc}"

    if resolve_uccl_ep_bench_dir(bench_dir) is None:
        return (
            "UCCL-EP bench buffer.py unavailable; set "
            f"{_UCCL_EP_BENCH_ENV}=/path/to/uccl/ep/bench"
        )
    world_size = os.environ.get("WORLD_SIZE")
    if world_size is None:
        return "requires torchrun with WORLD_SIZE set"
    if not dist.is_available():
        return "torch.distributed is unavailable"
    if not torch.cuda.is_available():
        return "CUDA is unavailable"
    return None


def run_uccl_ep_dispatch_combine_adapter(
    *,
    device_ids: tuple[int, ...] = (0, 1),
    num_tokens: int = 64,
    hidden: int = 128,
    num_topk: int = 4,
    num_experts: int = 16,
    input_dtype: str = "bf16",
    repeats: int = 1,
    bench_dir: str | os.PathLike[str] | None = None,
    skip_reason: Callable[[], str | None] | None = None,
) -> dict:
    if len(device_ids) < 2:
        raise ValueError("UCCL-EP adapter smoke expects at least two device ids")
    if repeats <= 0:
        raise ValueError("UCCL-EP adapter repeats must be positive")

    host_plan = create_cuda_comm_host_plan(backend="uccl", device_ids=device_ids)
    descriptor = create_uccl_ep_dispatch_combine_descriptor(
        host_plan.capability,
        num_tokens=num_tokens,
        hidden=hidden,
        num_topk=num_topk,
        num_experts=num_experts,
        input_dtype=input_dtype,
    )
    resolved_bench_dir = resolve_uccl_ep_bench_dir(bench_dir)
    result = {
        "backend": "uccl",
        "transport": TRANSPORT,
        "operation": OPERATION,
        "world_size": len(device_ids),
        "device_ids": list(device_ids),
        "capability": host_plan.capability.as_dict(),
        "launch_plans": [plan.as_dict() for plan in host_plan.launch_plans],
        "descriptor": descriptor.as_dict(),
        "repeats": int(repeats),
        "uccl_ep_bench_dir": None if resolved_bench_dir is None else str(resolved_bench_dir),
    }

    check = (lambda: uccl_ep_skip_reason(bench_dir=bench_dir)) if skip_reason is None else skip_reason
    reason = check()
    if reason is not None:
        return {**result, "status": "skipped", "reason": reason}

    if resolved_bench_dir is None:
        raise RuntimeError("UCCL-EP bench directory disappeared after skip check")
    return _run_distributed_dispatch_combine(
        result,
        host_plan,
        descriptor,
        resolved_bench_dir,
        repeats=int(repeats),
    )


def _run_distributed_dispatch_combine(
    result: dict,
    host_plan,
    descriptor,
    bench_dir: Path,
    *,
    repeats: int,
) -> dict:
    import torch  # noqa: PLC0415
    import torch.distributed as dist  # noqa: PLC0415

    Buffer, Config, utils = _load_uccl_ep_bench_modules(bench_dir)

    created_process_group = False
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        created_process_group = True

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != host_plan.world_size:
        raise RuntimeError(
            f"UCCL-EP adapter smoke expects world_size={host_plan.world_size}, got {world_size}"
        )

    launch_plan = host_plan.launch_plan_for_worker(rank)
    torch.cuda.set_device(launch_plan.device_id)
    os.environ["LOCAL_RANK"] = str(launch_plan.device_id)
    buffer = Buffer(
        dist.group.WORLD,
        int(2e9),
        0,
        low_latency_mode=False,
        num_qps_per_rank=1,
        explicitly_destroy=True,
        is_intranode=True,
    )
    torch.manual_seed(rank)

    try:
        repeat_results = tuple(
            _run_rank_dispatch_combine(
                torch,
                dist,
                Buffer,
                Config,
                utils,
                buffer,
                descriptor,
                rank=rank,
                device_id=launch_plan.device_id,
                repeat_index=repeat_index,
            )
            for repeat_index in range(repeats)
        )
        rank_result = _summarize_rank_repeats(
            rank,
            launch_plan.device_id,
            descriptor.input_dtype,
            repeat_results,
        )
        gathered = [None] * world_size if rank == 0 else None
        dist.gather_object(rank_result, object_gather_list=gathered, dst=0)
        if rank == 0:
            assert gathered is not None
            passed = all(item["passed"] for item in gathered)
            return {
                **result,
                "status": "passed" if passed else "failed",
                "rank_results": gathered,
            }
        return {**result, "status": "passed", "rank": rank, "_suppress_output": True}
    finally:
        buffer.destroy()
        dist.barrier()
        if created_process_group:
            dist.destroy_process_group()


def _run_rank_dispatch_combine(
    torch,
    dist,
    Buffer,
    Config,
    utils,
    buffer,
    descriptor: UcclEpDispatchCombineDescriptor,
    *,
    rank: int,
    device_id: int,
    repeat_index: int,
) -> dict:
    num_sms = 24 if torch.version.cuda else 64
    config = Config(num_sms, 8, 256)
    x = torch.ones(
        (descriptor.num_tokens, descriptor.hidden),
        dtype=torch.bfloat16,
        device=f"cuda:{device_id}",
    ) * (rank + 1)
    topk_idx = _build_topk_idx(torch, descriptor, device_id=device_id)
    topk_weights = torch.ones(
        (descriptor.num_tokens, descriptor.num_topk),
        dtype=torch.float32,
        device=f"cuda:{device_id}",
    ) * (rank + 1)
    (
        num_tokens_per_rank,
        _,
        num_tokens_per_expert,
        is_token_in_rank,
        _,
    ) = buffer.get_dispatch_layout(topk_idx, descriptor.num_experts)
    global_num_tokens_per_rank = num_tokens_per_rank.clone()
    dist.all_reduce(global_num_tokens_per_rank)
    global_num_tokens_per_expert = num_tokens_per_expert.clone()
    dist.all_reduce(global_num_tokens_per_expert)

    dispatch_x: Any = x
    if descriptor.input_dtype == "fp8":
        if descriptor.hidden % 128 != 0:
            raise RuntimeError("FP8 UCCL-EP dispatch requires hidden divisible by 128")
        if not Buffer.is_sm90_compiled():
            raise RuntimeError("FP8 UCCL-EP dispatch requires an SM90 build")
        dispatch_x = utils.per_token_cast_to_fp8(x)
        dispatch_x = (dispatch_x[0], dispatch_x[1].T.contiguous().T)

    (
        recv_x,
        recv_topk_idx,
        recv_topk_weights,
        recv_num_tokens_per_expert_list,
        handle,
        _,
    ) = buffer.dispatch(
        x=dispatch_x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_tokens_per_rank=num_tokens_per_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        config=config,
    )
    if descriptor.input_dtype == "fp8":
        recv_x = utils.per_token_cast_back(*recv_x)

    combined_x, combined_topk_weights, _ = buffer.combine(
        x=recv_x,
        handle=handle,
        topk_weights=recv_topk_weights,
        config=config,
    )
    torch.cuda.synchronize(device_id)
    routed_rank_count = is_token_in_rank.sum(dim=1).clamp_min(1).unsqueeze(1)
    max_abs_error = float((combined_x.float() / routed_rank_count - x.float()).abs().max().item())
    topk_weight_error = float((combined_topk_weights - topk_weights).abs().max().item())
    expected_recv_tokens = int(global_num_tokens_per_rank[rank].item())
    actual_recv_tokens = int(recv_x.size(0))
    local_expert_counts = global_num_tokens_per_expert.view(buffer.group_size, -1)[rank].tolist()
    passed = (
        max_abs_error < 5e-6
        and topk_weight_error < 1e-6
        and actual_recv_tokens == expected_recv_tokens
        and recv_topk_idx.size(0) == actual_recv_tokens
        and recv_topk_weights.size(0) == actual_recv_tokens
        and recv_num_tokens_per_expert_list == local_expert_counts
    )
    return {
        "rank": rank,
        "device_id": device_id,
        "repeat_index": repeat_index,
        "input_dtype": descriptor.input_dtype,
        "recv_tokens": actual_recv_tokens,
        "expected_total_sent_tokens": expected_recv_tokens,
        "max_abs_error": max_abs_error,
        "topk_weight_error": topk_weight_error,
        "passed": passed,
    }


def _summarize_rank_repeats(
    rank: int,
    device_id: int,
    input_dtype: str,
    repeat_results: tuple[dict, ...],
) -> dict:
    return {
        "rank": rank,
        "device_id": device_id,
        "input_dtype": input_dtype,
        "repeats": list(repeat_results),
        "repeat_count": len(repeat_results),
        "recv_tokens": [item["recv_tokens"] for item in repeat_results],
        "expected_total_sent_tokens": [item["expected_total_sent_tokens"] for item in repeat_results],
        "max_abs_error": max(item["max_abs_error"] for item in repeat_results),
        "topk_weight_error": max(item["topk_weight_error"] for item in repeat_results),
        "passed": all(item["passed"] for item in repeat_results),
    }


def _build_topk_idx(torch, descriptor: UcclEpDispatchCombineDescriptor, *, device_id: int):
    token_offsets = torch.arange(
        descriptor.num_tokens,
        dtype=torch.int64,
        device=f"cuda:{device_id}",
    ).unsqueeze(1)
    topk_offsets = torch.arange(
        descriptor.num_topk,
        dtype=torch.int64,
        device=f"cuda:{device_id}",
    ).unsqueeze(0)
    return (token_offsets + topk_offsets) % descriptor.num_experts


def _load_uccl_ep_bench_modules(bench_dir: Path):
    bench_dir_str = str(bench_dir)
    if bench_dir_str not in sys.path:
        sys.path.insert(0, bench_dir_str)
    import utils  # noqa: PLC0415
    from buffer import Buffer  # noqa: PLC0415
    from uccl.ep import Config  # noqa: PLC0415

    return Buffer, Config, utils


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-ids", type=parse_device_ids, default=(0, 1))
    parser.add_argument("--num-tokens", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--num-topk", type=int, default=4)
    parser.add_argument("--num-experts", type=int, default=16)
    parser.add_argument("--input-dtype", choices=("bf16", "fp8"), default="bf16")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--uccl-ep-bench-dir", default=None)
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="return a non-zero status when dependencies, torchrun, or CUDA are unavailable",
    )
    args = parser.parse_args(argv)

    result = run_uccl_ep_dispatch_combine_adapter(
        device_ids=args.device_ids,
        num_tokens=args.num_tokens,
        hidden=args.hidden,
        num_topk=args.num_topk,
        num_experts=args.num_experts,
        input_dtype=args.input_dtype,
        repeats=args.repeats,
        bench_dir=args.uccl_ep_bench_dir,
    )
    suppress = result.pop("_suppress_output", False)
    if not suppress:
        print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] == "failed":
        return 1
    if result["status"] == "skipped" and args.require_cuda:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
