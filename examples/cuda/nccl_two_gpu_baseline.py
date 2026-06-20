#!/usr/bin/env python3
"""Skip-safe two-GPU NCCL baseline for CUDA communication evidence."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import socket
import time
from queue import Empty
from typing import Callable

from simpler_setup.cuda_comm import (
    CudaCommRuntimeRegistry,
    create_cuda_comm_host_plan,
)

OPERATIONS = ["all_reduce", "reduce_scatter", "all_gather", "send_recv"]
PROCESS_TIMEOUT_S = 120


def parse_device_ids(value: str) -> tuple[int, ...]:
    device_ids = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(device_ids) != 2:
        raise argparse.ArgumentTypeError("expected exactly two comma-separated device ids")
    if len(set(device_ids)) != len(device_ids):
        raise argparse.ArgumentTypeError("device ids must be unique")
    return device_ids


def nccl_skip_reason(min_gpus: int = 2) -> str | None:
    try:
        import torch  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - depends on local env
        return f"torch import failed: {exc}"

    if not torch.cuda.is_available():
        return "torch.cuda is not available"
    if torch.cuda.device_count() < min_gpus:
        return f"need at least {min_gpus} CUDA devices, found {torch.cuda.device_count()}"
    if not torch.distributed.is_available():
        return "torch.distributed is not available"
    if not torch.distributed.is_nccl_available():
        return "torch.distributed NCCL is not available"
    return None


def run_nccl_baseline(
    *,
    device_ids: tuple[int, int] = (0, 1),
    tensor_numel: int = 1024,
    skip_reason: Callable[[int], str | None] | None = None,
) -> dict:
    host_plan = create_cuda_comm_host_plan(backend="nccl", device_ids=device_ids)
    result = {
        "backend": "nccl",
        "world_size": len(device_ids),
        "device_ids": list(device_ids),
        "capability": host_plan.capability.as_dict(),
        "launch_plans": [plan.as_dict() for plan in host_plan.launch_plans],
        "tensor_numel": tensor_numel,
        "operations": OPERATIONS,
    }
    if len(device_ids) != 2:
        raise ValueError("NCCL baseline currently expects exactly two device ids")
    if tensor_numel <= 0:
        raise ValueError("tensor_numel must be positive")

    skip_check = nccl_skip_reason if skip_reason is None else skip_reason
    reason = skip_check(len(device_ids))
    if reason is not None:
        return {**result, "status": "skipped", "reason": reason}

    init_method = f"tcp://127.0.0.1:{_free_tcp_port()}"
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    processes = [
        ctx.Process(
            target=_rank_worker,
            args=(rank, device_ids, tensor_numel, init_method, queue),
        )
        for rank in range(len(device_ids))
    ]

    start = time.time()
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=PROCESS_TIMEOUT_S)
    timed_out = [process.pid for process in processes if process.is_alive()]
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=5)
    elapsed_s = time.time() - start

    rank_results = []
    while True:
        try:
            rank_results.append(queue.get_nowait())
        except Empty:
            break
    rank_results.sort(key=lambda item: item.get("rank", -1))

    failures = [item for item in rank_results if item.get("status") != "passed"]
    failed_processes = [process.exitcode for process in processes if process.exitcode != 0]
    if timed_out or failures or failed_processes or len(rank_results) != len(device_ids):
        return {
            **result,
            "status": "failed",
            "elapsed_s": elapsed_s,
            "timed_out_pids": timed_out,
            "rank_results": rank_results,
            "process_exitcodes": [process.exitcode for process in processes],
        }

    return {
        **result,
        "status": "passed",
        "elapsed_s": elapsed_s,
        "rank_results": rank_results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-ids", type=parse_device_ids, default=(0, 1))
    parser.add_argument("--tensor-numel", type=int, default=1024)
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="return a non-zero status when dependencies or CUDA are unavailable",
    )
    args = parser.parse_args(argv)

    result = run_nccl_baseline(device_ids=args.device_ids, tensor_numel=args.tensor_numel)
    print(json.dumps(result, indent=2, sort_keys=True))

    if result["status"] == "failed":
        return 1
    if result["status"] == "skipped" and args.require_cuda:
        return 2
    return 0


def _rank_worker(
    rank: int,
    device_ids: tuple[int, ...],
    tensor_numel: int,
    init_method: str,
    queue,
) -> None:
    registry = None
    runtime = None
    world_size = len(device_ids)
    try:
        import torch  # noqa: PLC0415

        host_plan = create_cuda_comm_host_plan(backend="nccl", device_ids=device_ids)
        plan = host_plan.launch_plan_for_worker(rank)
        registry = CudaCommRuntimeRegistry()
        runtime = plan.acquire_runtime(registry, init_method=init_method)
        device = torch.device(f"cuda:{plan.device_id}")

        base = torch.arange(tensor_numel, device=device, dtype=torch.float32)
        reduced = base + float(rank)
        runtime.all_reduce(reduced)
        expected_reduced = world_size * base + sum(range(world_size))
        all_reduce_ok = bool(torch.allclose(reduced, expected_reduced))

        scatter_input = torch.cat(
            [base + float(rank) + (10.0 * float(chunk)) for chunk in range(world_size)]
        )
        scattered = torch.empty_like(base)
        runtime.reduce_scatter(scattered, scatter_input)
        expected_scattered = world_size * (base + (10.0 * float(rank))) + sum(
            range(world_size)
        )
        reduce_scatter_ok = bool(torch.allclose(scattered, expected_scattered))

        gathered = runtime.all_gather(torch.tensor([float(rank)], device=device))
        gathered_values = [float(item.cpu().item()) for item in gathered]
        all_gather_ok = gathered_values == [float(item) for item in range(world_size)]

        if rank == 0:
            send_tensor = torch.tensor([17.0], device=device)
            recv_tensor = torch.empty(1, device=device)
            runtime.send(send_tensor, dst=1)
            runtime.recv(recv_tensor, src=1)
            send_recv_value = float(recv_tensor.cpu().item())
            send_recv_ok = send_recv_value == 23.0
        else:
            recv_tensor = torch.empty(1, device=device)
            send_tensor = torch.tensor([23.0], device=device)
            runtime.recv(recv_tensor, src=0)
            runtime.send(send_tensor, dst=0)
            send_recv_value = float(recv_tensor.cpu().item())
            send_recv_ok = send_recv_value == 17.0

        torch.cuda.synchronize(device)
        queue.put(
            {
                "rank": rank,
                "device_id": plan.device_id,
                "launch_plan": plan.as_dict(),
                "status": (
                    "passed"
                    if all_reduce_ok and reduce_scatter_ok and all_gather_ok and send_recv_ok
                    else "failed"
                ),
                "all_reduce": {
                    "passed": all_reduce_ok,
                    "checksum": float(reduced.sum().cpu().item()),
                },
                "reduce_scatter": {
                    "passed": reduce_scatter_ok,
                    "checksum": float(scattered.sum().cpu().item()),
                },
                "all_gather": {
                    "passed": all_gather_ok,
                    "values": gathered_values,
                },
                "send_recv": {
                    "passed": send_recv_ok,
                    "received": send_recv_value,
                },
            }
        )
    except Exception as exc:  # pragma: no cover - exercised by hardware failures
        queue.put(
            {
                "rank": rank,
                "device_id": device_ids[rank],
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        raise
    finally:
        if registry is not None and runtime is not None:
            registry.release(runtime.runtime_id)


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


if __name__ == "__main__":
    raise SystemExit(main())
