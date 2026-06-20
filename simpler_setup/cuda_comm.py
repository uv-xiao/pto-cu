# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Internal CUDA communication capability descriptors.

This module is intentionally part of ``simpler_setup`` rather than the stable
``simpler`` package. It gives CUDA runtime bring-up code an opaque capability
shape for mock and NCCL communicator lifecycle tests without exposing transport
objects through public task APIs.
"""

from __future__ import annotations

import struct
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import ModuleType
from typing import Any

_SUPPORTED_BACKENDS = {"mock", "nccl"}
_BACKEND_CODES = {"mock": 0, "nccl": 1}
_DEVICE_DESCRIPTOR_STRUCT = struct.Struct("<IIIII")


class CudaCommOp(str, Enum):
    ALL_REDUCE = "all_reduce"
    REDUCE_SCATTER = "reduce_scatter"
    ALL_GATHER = "all_gather"
    SEND_RECV = "send_recv"


_BASELINE_COLLECTIVE_OPS = (
    CudaCommOp.ALL_REDUCE,
    CudaCommOp.REDUCE_SCATTER,
    CudaCommOp.ALL_GATHER,
    CudaCommOp.SEND_RECV,
)
_DEFAULT_BACKEND_OPS = {
    "mock": _BASELINE_COLLECTIVE_OPS,
    "nccl": _BASELINE_COLLECTIVE_OPS,
}


@dataclass(frozen=True)
class CudaCommRank:
    rank: int
    device_id: int


@dataclass(frozen=True)
class CudaCommCapability:
    backend: str
    ranks: tuple[CudaCommRank, ...]
    capability_id: str
    operations: tuple[CudaCommOp, ...]

    def __post_init__(self) -> None:
        if not self.ranks:
            raise ValueError("CUDA communication capability needs at least one rank")

        rank_ids = [item.rank for item in self.ranks]
        device_ids = [item.device_id for item in self.ranks]
        if sorted(rank_ids) != list(range(len(self.ranks))):
            raise ValueError("rank mapping must cover contiguous ranks starting at zero")
        if len(set(device_ids)) != len(device_ids):
            raise ValueError("rank mapping must use unique device ids")

    @property
    def world_size(self) -> int:
        return len(self.ranks)

    def rank_to_device(self) -> dict[int, int]:
        return {item.rank: item.device_id for item in self.ranks}

    def supports(self, op: CudaCommOp | str) -> bool:
        normalized = _normalize_op(op)
        return normalized in self.operations

    def as_dict(self) -> dict:
        return {
            "backend": self.backend,
            "capability_id": self.capability_id,
            "world_size": self.world_size,
            "rank_to_device": {str(rank): device_id for rank, device_id in self.rank_to_device().items()},
            "operations": [op.value for op in self.operations],
        }


def create_cuda_comm_capability(*, backend: str, device_ids: Sequence[int]) -> CudaCommCapability:
    if backend not in _SUPPORTED_BACKENDS:
        raise ValueError(f"unsupported CUDA communication backend: {backend}")
    device_tuple = tuple(int(device_id) for device_id in device_ids)
    ranks = tuple(CudaCommRank(rank=rank, device_id=device_id) for rank, device_id in enumerate(device_tuple))
    capability_id = f"{backend}:" + ",".join(
        f"rank{rank}->cuda{device_id}" for rank, device_id in enumerate(device_tuple)
    )
    return CudaCommCapability(
        backend=backend,
        ranks=ranks,
        capability_id=capability_id,
        operations=_DEFAULT_BACKEND_OPS[backend],
    )


def create_mock_cuda_comm_capability(*, device_ids: Sequence[int]) -> CudaCommCapability:
    return create_cuda_comm_capability(backend="mock", device_ids=device_ids)


@dataclass(frozen=True)
class CudaCommDeviceDescriptor:
    backend: str
    backend_code: int
    rank: int
    device_id: int
    world_size: int
    capability_crc32: int

    def as_dict(self) -> dict:
        return {
            "backend": self.backend,
            "backend_code": self.backend_code,
            "rank": self.rank,
            "device_id": self.device_id,
            "world_size": self.world_size,
            "capability_crc32": self.capability_crc32,
        }

    def to_bytes(self) -> bytes:
        return _DEVICE_DESCRIPTOR_STRUCT.pack(
            self.backend_code,
            self.rank,
            self.device_id,
            self.world_size,
            self.capability_crc32,
        )


@dataclass(frozen=True)
class CudaCommLaunchPlan:
    capability: CudaCommCapability
    rank: int
    device_id: int
    runtime_id: str

    @property
    def backend(self) -> str:
        return self.capability.backend

    @property
    def world_size(self) -> int:
        return self.capability.world_size

    def as_dict(self) -> dict:
        return {
            "backend": self.backend,
            "capability_id": self.capability.capability_id,
            "runtime_id": self.runtime_id,
            "rank": self.rank,
            "device_id": self.device_id,
            "world_size": self.world_size,
        }

    def device_descriptor(self) -> CudaCommDeviceDescriptor:
        return CudaCommDeviceDescriptor(
            backend=self.backend,
            backend_code=_BACKEND_CODES[self.backend],
            rank=self.rank,
            device_id=self.device_id,
            world_size=self.world_size,
            capability_crc32=zlib.crc32(self.capability.capability_id.encode("utf-8")),
        )

    def acquire_runtime(
        self,
        registry: CudaCommRuntimeRegistry,
        *,
        init_method: str | None = None,
        torch_module: ModuleType | Any | None = None,
        dist_module: ModuleType | Any | None = None,
    ) -> MockCudaCommRuntime | TorchNcclCudaCommRuntime:
        if self.backend == "nccl":
            return registry.acquire(
                self.capability,
                rank=self.rank,
                init_method=init_method,
                torch_module=torch_module,
                dist_module=dist_module,
            )
        return registry.acquire(self.capability)


def create_cuda_comm_launch_plan(capability: CudaCommCapability, *, rank: int) -> CudaCommLaunchPlan:
    rank_to_device = capability.rank_to_device()
    if rank not in rank_to_device:
        raise ValueError(f"unknown rank {rank} for CUDA communication capability")
    return CudaCommLaunchPlan(
        capability=capability,
        rank=int(rank),
        device_id=rank_to_device[rank],
        runtime_id=_runtime_id(capability, rank if capability.backend == "nccl" else None),
    )


@dataclass(frozen=True)
class CudaCommHostPlan:
    capability: CudaCommCapability
    launch_plans: tuple[CudaCommLaunchPlan, ...]

    @property
    def backend(self) -> str:
        return self.capability.backend

    @property
    def world_size(self) -> int:
        return self.capability.world_size

    @property
    def device_ids(self) -> tuple[int, ...]:
        rank_to_device = self.capability.rank_to_device()
        return tuple(rank_to_device[rank] for rank in range(self.world_size))

    def runtime_ids(self) -> tuple[str, ...]:
        return tuple(plan.runtime_id for plan in self.launch_plans)

    def launch_plan_for_worker(self, worker_index: int) -> CudaCommLaunchPlan:
        if worker_index < 0 or worker_index >= len(self.launch_plans):
            raise ValueError(f"worker index {worker_index} outside CUDA communication launch plan range")
        return self.launch_plans[worker_index]

    def as_dict(self) -> dict:
        return {
            "backend": self.backend,
            "world_size": self.world_size,
            "device_ids": list(self.device_ids),
            "capability": self.capability.as_dict(),
            "launch_plans": [plan.as_dict() for plan in self.launch_plans],
        }


def create_cuda_comm_host_plan(*, backend: str, device_ids: Sequence[int]) -> CudaCommHostPlan:
    capability = create_cuda_comm_capability(backend=backend, device_ids=device_ids)
    launch_plans = tuple(create_cuda_comm_launch_plan(capability, rank=rank) for rank in range(capability.world_size))
    return CudaCommHostPlan(capability=capability, launch_plans=launch_plans)


class MockCudaCommRuntime:
    """Pure-Python communication runtime used to test the CUDA boundary shape."""

    def __init__(self, capability: CudaCommCapability):
        if capability.backend != "mock":
            raise ValueError("MockCudaCommRuntime requires a mock capability")
        self.capability = capability

    def all_reduce(self, rank_buffers: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
        buffers = self._validate_rank_buffers(rank_buffers)
        reduced = tuple(sum(values) for values in zip(*buffers))
        return tuple(reduced for _ in range(self.capability.world_size))

    def reduce_scatter(self, rank_buffers: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
        buffers = self._validate_rank_buffers(rank_buffers)
        total_len = len(buffers[0])
        if total_len % self.capability.world_size != 0:
            raise ValueError("reduce_scatter input length must be divisible by world_size")

        reduced = tuple(sum(values) for values in zip(*buffers))
        chunk_size = total_len // self.capability.world_size
        return tuple(tuple(reduced[start : start + chunk_size]) for start in range(0, total_len, chunk_size))

    def all_gather(self, rank_buffers: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
        buffers = self._validate_rank_buffers(rank_buffers)
        gathered = tuple(value for buffer in buffers for value in buffer)
        return tuple(gathered for _ in range(self.capability.world_size))

    def send_recv(self, sends: Mapping[tuple[int, int], Sequence[float]]) -> dict[int, tuple[float, ...]]:
        rank_to_device = self.capability.rank_to_device()
        received: dict[int, tuple[float, ...]] = {}
        for (src_rank, dst_rank), payload in sends.items():
            if src_rank not in rank_to_device:
                raise ValueError(f"unknown source rank {src_rank}")
            if dst_rank not in rank_to_device:
                raise ValueError(f"unknown destination rank {dst_rank}")
            received[dst_rank] = tuple(float(item) for item in payload)
        return received

    def _validate_rank_buffers(self, rank_buffers: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
        if len(rank_buffers) != self.capability.world_size:
            raise ValueError("rank buffer count must match capability world_size")

        buffers = tuple(tuple(float(item) for item in buffer) for buffer in rank_buffers)
        lengths = {len(buffer) for buffer in buffers}
        if len(lengths) != 1:
            raise ValueError("rank buffers must have the same length")
        return buffers


class TorchNcclCudaCommRuntime:
    """Thin torch.distributed NCCL runtime owned by a local rank process."""

    def __init__(
        self,
        capability: CudaCommCapability,
        *,
        rank: int,
        init_method: str,
        torch_module: ModuleType | Any | None = None,
        dist_module: ModuleType | Any | None = None,
    ) -> None:
        if capability.backend != "nccl":
            raise ValueError("TorchNcclCudaCommRuntime requires an nccl capability")
        rank_to_device = capability.rank_to_device()
        if rank not in rank_to_device:
            raise ValueError(f"unknown NCCL rank {rank}")
        if not init_method:
            raise ValueError("NCCL init_method is required")

        self.capability = capability
        self.rank = int(rank)
        self.device_id = rank_to_device[self.rank]
        self.runtime_id = _runtime_id(capability, self.rank)
        self._torch = _import_torch() if torch_module is None else torch_module
        self._dist = _import_torch_distributed() if dist_module is None else dist_module

        self._torch.cuda.set_device(self.device_id)
        self._dist.init_process_group(
            backend="nccl",
            init_method=init_method,
            rank=self.rank,
            world_size=capability.world_size,
        )

    def close(self) -> None:
        if self._dist.is_available() and self._dist.is_initialized():
            self._dist.destroy_process_group()

    def all_reduce(self, tensor):
        self._dist.all_reduce(tensor, op=self._dist.ReduceOp.SUM)
        return tensor

    def reduce_scatter(self, output, tensor):
        if hasattr(self._dist, "reduce_scatter_tensor"):
            self._dist.reduce_scatter_tensor(output, tensor, op=self._dist.ReduceOp.SUM)
        else:
            scatter_chunks = list(tensor.chunk(self.capability.world_size))
            self._dist.reduce_scatter(output, scatter_chunks, op=self._dist.ReduceOp.SUM)
        return output

    def all_gather(self, tensor):
        gathered = [self._torch.empty_like(tensor) for _ in range(self.capability.world_size)]
        self._dist.all_gather(gathered, tensor)
        return tuple(gathered)

    def send(self, tensor, *, dst: int) -> None:
        self._dist.send(tensor, dst=dst)

    def recv(self, tensor, *, src: int) -> None:
        self._dist.recv(tensor, src=src)


class CudaCommRuntimeRegistry:
    """Runtime-private cache for CUDA communication capability state."""

    def __init__(self) -> None:
        self._runtimes: dict[str, MockCudaCommRuntime | TorchNcclCudaCommRuntime] = {}

    def acquire(
        self,
        capability: CudaCommCapability,
        *,
        rank: int | None = None,
        init_method: str | None = None,
        torch_module: ModuleType | Any | None = None,
        dist_module: ModuleType | Any | None = None,
    ) -> MockCudaCommRuntime | TorchNcclCudaCommRuntime:
        runtime_id = _runtime_id(capability, rank) if capability.backend == "nccl" else capability.capability_id
        cached = self._runtimes.get(runtime_id)
        if cached is not None:
            return cached

        if capability.backend == "mock":
            runtime = MockCudaCommRuntime(capability)
        elif capability.backend == "nccl":
            if rank is None:
                raise ValueError("NCCL runtime acquisition requires rank")
            if init_method is None:
                raise ValueError("NCCL runtime acquisition requires init_method")
            runtime = TorchNcclCudaCommRuntime(
                capability,
                rank=rank,
                init_method=init_method,
                torch_module=torch_module,
                dist_module=dist_module,
            )
        else:
            raise NotImplementedError(f"{capability.backend} communicator lifecycle is not implemented")

        self._runtimes[runtime_id] = runtime
        return runtime

    def get(self, runtime_id: str) -> MockCudaCommRuntime | TorchNcclCudaCommRuntime:
        try:
            return self._runtimes[runtime_id]
        except KeyError as exc:
            raise KeyError(f"unknown CUDA communication capability: {runtime_id}") from exc

    def release(self, runtime_id: str) -> None:
        runtime = self._runtimes.pop(runtime_id, None)
        if hasattr(runtime, "close"):
            runtime.close()

    def active_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._runtimes))


def _normalize_op(op: CudaCommOp | str) -> CudaCommOp:
    if isinstance(op, CudaCommOp):
        return op
    try:
        return CudaCommOp(op)
    except ValueError as exc:
        raise ValueError(f"unsupported CUDA communication operation: {op}") from exc


def _runtime_id(capability: CudaCommCapability, rank: int | None) -> str:
    if rank is None:
        return capability.capability_id
    return f"{capability.capability_id}/local_rank{rank}"


def _import_torch() -> ModuleType:
    import torch  # noqa: PLC0415

    return torch


def _import_torch_distributed() -> ModuleType:
    import torch.distributed as dist  # noqa: PLC0415

    return dist
