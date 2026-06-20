#!/usr/bin/env python3
"""Two-GPU NCCL baseline through PTO Worker CTRL_COMM_OP transport."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import struct
import subprocess
import sys
import time
from collections.abc import Callable
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any

from simpler_setup.cuda_comm import create_cuda_comm_host_plan

OPERATIONS = ["all_reduce", "reduce_scatter", "all_gather", "send_recv"]
_NCCL_LIBRARY_ENV = "PTO_CUDA_NCCL_LIBRARY"


class _HostFloatBuffer:
    _registry: dict[int, _HostFloatBuffer] = {}

    def __init__(self, count: int):
        if count <= 0:
            raise ValueError("host float buffer count must be positive")
        self.count = int(count)
        self.shm = SharedMemory(create=True, size=self.count * 4)
        self.address = ctypes.addressof(ctypes.c_char.from_buffer(self.shm.buf))
        self._registry[self.address] = self

    @classmethod
    def from_address(cls, address: int) -> _HostFloatBuffer:
        try:
            return cls._registry[int(address)]
        except KeyError as exc:
            raise KeyError(f"unknown host buffer address 0x{int(address):x}") from exc

    def write(self, values) -> None:
        data = [float(item) for item in values]
        if len(data) > self.count:
            raise ValueError(f"too many values for host buffer: {len(data)} > {self.count}")
        for idx, value in enumerate(data):
            struct.pack_into("<f", self.shm.buf, idx * 4, value)

    def read(self, count: int | None = None) -> list[float]:
        n = self.count if count is None else int(count)
        if n > self.count:
            raise ValueError(f"read count {n} exceeds host buffer count {self.count}")
        return [struct.unpack_from("<f", self.shm.buf, idx * 4)[0] for idx in range(n)]

    def close(self) -> None:
        self._registry.pop(self.address, None)
        self.shm.close()
        self.shm.unlink()


def parse_device_ids(value: str) -> tuple[int, ...]:
    device_ids = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(device_ids) != 2:
        raise argparse.ArgumentTypeError("expected exactly two comma-separated device ids")
    if len(set(device_ids)) != len(device_ids):
        raise argparse.ArgumentTypeError("device ids must be unique")
    return device_ids


def cuda_worker_skip_reason(min_gpus: int = 2) -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=15,
        )
    except Exception as exc:  # pragma: no cover - depends on local host tools
        return f"nvidia-smi failed: {type(exc).__name__}: {exc}"
    if result.returncode != 0:
        return f"nvidia-smi failed with code {result.returncode}: {result.stdout.strip()}"
    gpus = [line for line in result.stdout.splitlines() if line.strip()]
    if len(gpus) < min_gpus:
        return f"need at least {min_gpus} CUDA devices, found {len(gpus)}"
    return None


def discover_bundled_nccl_library(search_paths: list[str | Path] | None = None) -> str | None:
    roots = [Path(item) for item in (sys.path if search_paths is None else search_paths) if item]
    candidates: list[Path] = []
    for root in roots:
        candidates.extend((root / "nvidia" / "nccl" / "lib").glob("libnccl.so*"))
    for candidate in sorted(candidates, key=lambda path: (path.name != "libnccl.so.2", str(path))):
        if candidate.is_file():
            return str(candidate)
    return None


def configure_nccl_library_env(search_paths: list[str | Path] | None = None) -> str | None:
    existing = os.environ.get(_NCCL_LIBRARY_ENV)
    if existing:
        return existing
    discovered = discover_bundled_nccl_library(search_paths)
    if discovered is not None:
        os.environ[_NCCL_LIBRARY_ENV] = discovered
    return discovered


def run_worker_control_ops(
    *,
    device_ids: tuple[int, int] = (0, 1),
    tensor_numel: int = 1024,
    build: bool = False,
    skip_reason: Callable[[int], str | None] | None = None,
    worker_factory: type | Callable[..., Any] | None = None,
) -> dict:
    if len(device_ids) != 2:
        raise ValueError("worker-control NCCL baseline currently expects exactly two device ids")
    if tensor_numel <= 0:
        raise ValueError("tensor_numel must be positive")

    host_plan = create_cuda_comm_host_plan(backend="nccl", device_ids=device_ids)
    result = {
        "backend": "nccl",
        "transport": "worker_control",
        "world_size": len(device_ids),
        "device_ids": list(device_ids),
        "capability": host_plan.capability.as_dict(),
        "launch_plans": [plan.as_dict() for plan in host_plan.launch_plans],
        "tensor_numel": int(tensor_numel),
        "operations": OPERATIONS,
    }

    check = cuda_worker_skip_reason if skip_reason is None else skip_reason
    reason = check(len(device_ids))
    if reason is not None:
        return {**result, "status": "skipped", "reason": reason}
    nccl_library = configure_nccl_library_env()
    if nccl_library is not None:
        result["nccl_library"] = nccl_library

    from simpler.task_interface import CallConfig  # noqa: PLC0415
    from simpler.worker import (  # noqa: PLC0415
        _COMM_OP_ALL_GATHER_F32,
        _COMM_OP_ALL_REDUCE_F32,
        _COMM_OP_REDUCE_SCATTER_F32,
        _COMM_OP_SEND_RECV_F32,
        Worker,
    )

    worker_cls = Worker if worker_factory is None else worker_factory
    buffers = _make_operation_buffers(tensor_numel)
    worker = worker_cls(
        level=3,
        platform="cuda",
        runtime="host_schedule",
        device_ids=list(device_ids),
        num_sub_workers=0,
        build=bool(build),
    )
    allocations: list[tuple[int, int]] = []
    started = time.time()
    try:
        worker.init()

        def orch_fn(orch, _args, _cfg):
            _stage_all_reduce(
                worker,
                orch,
                buffers,
                allocations,
                tensor_numel,
                _COMM_OP_ALL_REDUCE_F32,
            )
            _stage_reduce_scatter(
                worker,
                orch,
                buffers,
                allocations,
                tensor_numel,
                _COMM_OP_REDUCE_SCATTER_F32,
            )
            _stage_all_gather(
                worker,
                orch,
                buffers,
                allocations,
                tensor_numel,
                _COMM_OP_ALL_GATHER_F32,
            )
            _stage_send_recv(
                worker,
                orch,
                buffers,
                allocations,
                _COMM_OP_SEND_RECV_F32,
            )
            for rank, ptr in reversed(allocations):
                orch.free(worker_id=rank, ptr=ptr)
            allocations.clear()

        worker.run(orch_fn, args=None, config=CallConfig())
        checks = _validate_operation_outputs(buffers, tensor_numel)
        status = "passed" if all(item["passed"] for item in checks.values()) else "failed"
        return {
            **result,
            "status": status,
            "elapsed_s": time.time() - started,
            **checks,
        }
    except Exception as exc:  # pragma: no cover - exercised by hardware failures
        return {
            **result,
            "status": "failed",
            "elapsed_s": time.time() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        try:
            for rank, ptr in reversed(allocations):
                worker._orch.free(rank, ptr)  # noqa: SLF001 -- cleanup after failed orch_fn staging
        except Exception:
            pass
        try:
            worker.close()
        finally:
            _close_operation_buffers(buffers)


def _make_operation_buffers(tensor_numel: int) -> dict[str, dict[int, _HostFloatBuffer]]:
    n = int(tensor_numel)
    buffers = {
        "all_reduce_send": {rank: _HostFloatBuffer(n) for rank in (0, 1)},
        "all_reduce_recv": {rank: _HostFloatBuffer(n) for rank in (0, 1)},
        "reduce_scatter_send": {rank: _HostFloatBuffer(n * 2) for rank in (0, 1)},
        "reduce_scatter_recv": {rank: _HostFloatBuffer(n) for rank in (0, 1)},
        "all_gather_send": {rank: _HostFloatBuffer(n) for rank in (0, 1)},
        "all_gather_recv": {rank: _HostFloatBuffer(n * 2) for rank in (0, 1)},
        "send_recv_send": {rank: _HostFloatBuffer(1) for rank in (0, 1)},
        "send_recv_recv": {rank: _HostFloatBuffer(1) for rank in (0, 1)},
    }

    base = [float(idx) for idx in range(n)]
    for rank in (0, 1):
        buffers["all_reduce_send"][rank].write(value + float(rank) for value in base)
        scatter_values = []
        for chunk in (0, 1):
            scatter_values.extend(value + float(rank) + 10.0 * float(chunk) for value in base)
        buffers["reduce_scatter_send"][rank].write(scatter_values)
        buffers["all_gather_send"][rank].write(float(rank) + 0.25 * float(idx) for idx in range(n))
    buffers["send_recv_send"][0].write([17.0])
    buffers["send_recv_send"][1].write([23.0])
    return buffers


def _close_operation_buffers(buffers: dict[str, dict[int, _HostFloatBuffer]]) -> None:
    for rank_buffers in buffers.values():
        for buffer in rank_buffers.values():
            buffer.close()


def _stage_all_reduce(worker, orch, buffers, allocations, count: int, op_code: int) -> None:
    send_ptrs = _copy_inputs_to_device(orch, buffers["all_reduce_send"], allocations, count)
    recv_ptrs = _alloc_outputs(orch, buffers["all_reduce_recv"], allocations, count)
    worker._dispatch_control_comm_op(
        workers=(0, 1),
        op_code=op_code,
        send_ptrs=send_ptrs,
        recv_ptrs=recv_ptrs,
        counts=count,
        op_name="nccl_worker_control_all_reduce",
    )
    _copy_outputs_from_device(orch, buffers["all_reduce_recv"], recv_ptrs, count)


def _stage_reduce_scatter(worker, orch, buffers, allocations, count: int, op_code: int) -> None:
    send_ptrs = _copy_inputs_to_device(orch, buffers["reduce_scatter_send"], allocations, count * 2)
    recv_ptrs = _alloc_outputs(orch, buffers["reduce_scatter_recv"], allocations, count)
    worker._dispatch_control_comm_op(
        workers=(0, 1),
        op_code=op_code,
        send_ptrs=send_ptrs,
        recv_ptrs=recv_ptrs,
        counts=count,
        op_name="nccl_worker_control_reduce_scatter",
    )
    _copy_outputs_from_device(orch, buffers["reduce_scatter_recv"], recv_ptrs, count)


def _stage_all_gather(worker, orch, buffers, allocations, count: int, op_code: int) -> None:
    send_ptrs = _copy_inputs_to_device(orch, buffers["all_gather_send"], allocations, count)
    recv_ptrs = _alloc_outputs(orch, buffers["all_gather_recv"], allocations, count * 2)
    worker._dispatch_control_comm_op(
        workers=(0, 1),
        op_code=op_code,
        send_ptrs=send_ptrs,
        recv_ptrs=recv_ptrs,
        counts=count,
        op_name="nccl_worker_control_all_gather",
    )
    _copy_outputs_from_device(orch, buffers["all_gather_recv"], recv_ptrs, count * 2)


def _stage_send_recv(worker, orch, buffers, allocations, op_code: int) -> None:
    send_ptrs = _copy_inputs_to_device(orch, buffers["send_recv_send"], allocations, 1)
    recv_ptrs = _alloc_outputs(orch, buffers["send_recv_recv"], allocations, 1)
    worker._dispatch_control_comm_op(
        workers=(0, 1),
        op_code=op_code,
        send_ptrs=send_ptrs,
        recv_ptrs=recv_ptrs,
        counts=1,
        dst_ranks={0: 1, 1: 0},
        src_ranks={0: 1, 1: 0},
        op_name="nccl_worker_control_send_recv",
    )
    _copy_outputs_from_device(orch, buffers["send_recv_recv"], recv_ptrs, 1)


def _copy_inputs_to_device(orch, host_buffers, allocations, count: int) -> dict[int, int]:
    ptrs = {}
    for rank, host in host_buffers.items():
        ptr = orch.malloc(worker_id=rank, size=count * 4)
        allocations.append((rank, ptr))
        orch.copy_to(worker_id=rank, dst=ptr, src=host.address, size=count * 4)
        ptrs[rank] = ptr
    return ptrs


def _alloc_outputs(orch, host_buffers, allocations, count: int) -> dict[int, int]:
    ptrs = {}
    for rank in host_buffers:
        ptr = orch.malloc(worker_id=rank, size=count * 4)
        allocations.append((rank, ptr))
        ptrs[rank] = ptr
    return ptrs


def _copy_outputs_from_device(orch, host_buffers, ptrs: dict[int, int], count: int) -> None:
    for rank, host in host_buffers.items():
        orch.copy_from(worker_id=rank, dst=host.address, src=ptrs[rank], size=count * 4)


def _validate_operation_outputs(buffers, tensor_numel: int) -> dict[str, dict]:
    n = int(tensor_numel)
    expected_all_reduce = [_expected_all_reduce_f32(idx) for idx in range(n)]
    expected_scatter = {
        rank: [_expected_reduce_scatter_f32(dst_rank=rank, idx=idx) for idx in range(n)]
        for rank in (0, 1)
    }
    expected_gather = []
    for rank in (0, 1):
        expected_gather.extend(_expected_all_gather_f32(src_rank=rank, idx=idx) for idx in range(n))
    expected_send_recv = {0: [23.0], 1: [17.0]}

    all_reduce_actual = {rank: buffers["all_reduce_recv"][rank].read(n) for rank in (0, 1)}
    reduce_scatter_actual = {rank: buffers["reduce_scatter_recv"][rank].read(n) for rank in (0, 1)}
    all_gather_actual = {rank: buffers["all_gather_recv"][rank].read(n * 2) for rank in (0, 1)}
    send_recv_actual = {rank: buffers["send_recv_recv"][rank].read(1) for rank in (0, 1)}

    return {
        "all_reduce": _rank_check(all_reduce_actual, {0: expected_all_reduce, 1: expected_all_reduce}),
        "reduce_scatter": _rank_check(reduce_scatter_actual, expected_scatter),
        "all_gather": _rank_check(all_gather_actual, {0: expected_gather, 1: expected_gather}),
        "send_recv": _rank_check(send_recv_actual, expected_send_recv),
    }


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def _expected_all_reduce_f32(idx: int) -> float:
    return _f32(_f32(float(idx)) + _f32(float(idx) + 1.0))


def _expected_reduce_scatter_f32(*, dst_rank: int, idx: int) -> float:
    chunk = float(dst_rank)
    return _f32(_f32(float(idx) + 10.0 * chunk) + _f32(float(idx) + 1.0 + 10.0 * chunk))


def _expected_all_gather_f32(*, src_rank: int, idx: int) -> float:
    return _f32(float(src_rank) + 0.25 * float(idx))


def _rank_check(actual: dict[int, list[float]], expected: dict[int, list[float]]) -> dict:
    per_rank = {}
    passed = True
    max_abs_error = 0.0
    for rank in sorted(expected):
        rank_error = _max_abs_error(actual[rank], expected[rank])
        rank_passed = rank_error <= 1e-5
        passed = passed and rank_passed
        max_abs_error = max(max_abs_error, rank_error)
        per_rank[str(rank)] = {
            "passed": rank_passed,
            "checksum": float(sum(actual[rank])),
            "max_abs_error": rank_error,
        }
    return {"passed": passed, "max_abs_error": max_abs_error, "per_rank": per_rank}


def _max_abs_error(actual: list[float], expected: list[float]) -> float:
    if len(actual) != len(expected):
        return float("inf")
    return max((abs(a - b) for a, b in zip(actual, expected)), default=0.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-ids", type=parse_device_ids, default=(0, 1))
    parser.add_argument("--tensor-numel", type=int, default=1024)
    parser.add_argument("--build", action="store_true", help="rebuild CUDA runtime before running")
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="return a non-zero status when dependencies or CUDA are unavailable",
    )
    args = parser.parse_args(argv)

    result = run_worker_control_ops(
        device_ids=args.device_ids,
        tensor_numel=args.tensor_numel,
        build=args.build,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] == "failed":
        return 1
    if result["status"] == "skipped" and args.require_cuda:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
