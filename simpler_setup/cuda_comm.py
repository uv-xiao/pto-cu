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
objects through public task APIs. UCCL helpers in this module are Python-side
adapter evidence only; they do not define a CUDA host-runtime UCCL ABI.
"""

from __future__ import annotations

import struct
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import ModuleType
from typing import Any

_SUPPORTED_BACKENDS = {"mock", "nccl", "uccl"}
_BACKEND_CODES = {"mock": 0, "nccl": 1, "uccl": 2}
_DEVICE_DESCRIPTOR_STRUCT = struct.Struct("<IIIII")


class CudaCommOp(str, Enum):
    ALL_REDUCE = "all_reduce"
    REDUCE_SCATTER = "reduce_scatter"
    ALL_GATHER = "all_gather"
    SEND_RECV = "send_recv"
    P2P_WRITE_IPC = "p2p_write_ipc"
    EP_DISPATCH_COMBINE = "ep_dispatch_combine"


_BASELINE_COLLECTIVE_OPS = (
    CudaCommOp.ALL_REDUCE,
    CudaCommOp.REDUCE_SCATTER,
    CudaCommOp.ALL_GATHER,
    CudaCommOp.SEND_RECV,
)
_DEFAULT_BACKEND_OPS = {
    "mock": _BASELINE_COLLECTIVE_OPS,
    "nccl": _BASELINE_COLLECTIVE_OPS,
    "uccl": (CudaCommOp.P2P_WRITE_IPC, CudaCommOp.EP_DISPATCH_COMBINE),
}
_UCCL_TRANSPORT_P2P_IPC = "p2p_ipc"
_UCCL_EP_INPUT_DTYPES = {"bf16", "fp8"}


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
        uccl_transport: str | None = None,
        p2p_module: ModuleType | Any | None = None,
        endpoint: Any | None = None,
    ) -> MockCudaCommRuntime | TorchNcclCudaCommRuntime | UcclP2PCudaCommRuntime:
        if self.backend == "nccl":
            return registry.acquire(
                self.capability,
                rank=self.rank,
                init_method=init_method,
                torch_module=torch_module,
                dist_module=dist_module,
            )
        if self.backend == "uccl":
            return registry.acquire(
                self.capability,
                rank=self.rank,
                uccl_transport=uccl_transport,
                p2p_module=p2p_module,
                endpoint=endpoint,
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
        runtime_id=_runtime_id(
            capability,
            rank if capability.backend in {"nccl", "uccl"} else None,
        ),
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


@dataclass(frozen=True)
class UcclP2PWriteIpcDescriptor:
    src_rank: int
    dst_rank: int
    nbytes: int

    def as_dict(self) -> dict:
        return {
            "operation": CudaCommOp.P2P_WRITE_IPC.value,
            "src_rank": self.src_rank,
            "dst_rank": self.dst_rank,
            "nbytes": self.nbytes,
        }


def create_uccl_p2p_write_ipc_descriptor(
    capability: CudaCommCapability,
    *,
    src_rank: int,
    dst_rank: int,
    nbytes: int,
) -> UcclP2PWriteIpcDescriptor:
    if capability.backend != "uccl":
        raise ValueError("UCCL P2P IPC descriptors require a uccl capability")
    rank_to_device = capability.rank_to_device()
    if src_rank not in rank_to_device:
        raise ValueError(f"unknown UCCL source rank {src_rank}")
    if dst_rank not in rank_to_device:
        raise ValueError(f"unknown UCCL destination rank {dst_rank}")
    if src_rank == dst_rank:
        raise ValueError("UCCL P2P IPC source and destination ranks must be distinct")
    if nbytes <= 0:
        raise ValueError("UCCL P2P IPC nbytes must be positive")
    return UcclP2PWriteIpcDescriptor(
        src_rank=int(src_rank),
        dst_rank=int(dst_rank),
        nbytes=int(nbytes),
    )


@dataclass(frozen=True)
class UcclEpDispatchCombineDescriptor:
    world_size: int
    num_tokens: int
    hidden: int
    num_topk: int
    num_experts: int
    input_dtype: str
    include_topk_weights: bool = True

    @property
    def experts_per_rank(self) -> int:
        return self.num_experts // self.world_size

    def metadata_shapes(self) -> dict[str, list[int]]:
        return {
            "topk_idx": [self.num_tokens, self.num_topk],
            "topk_weights": [self.num_tokens, self.num_topk],
            "num_tokens_per_rank": [self.world_size],
            "is_token_in_rank": [self.num_tokens, self.world_size],
            "num_tokens_per_expert": [self.num_experts],
        }

    def as_dict(self) -> dict:
        return {
            "operation": CudaCommOp.EP_DISPATCH_COMBINE.value,
            "world_size": self.world_size,
            "num_tokens": self.num_tokens,
            "hidden": self.hidden,
            "num_topk": self.num_topk,
            "num_experts": self.num_experts,
            "experts_per_rank": self.experts_per_rank,
            "input_dtype": self.input_dtype,
            "include_topk_weights": self.include_topk_weights,
            "metadata_shapes": self.metadata_shapes(),
        }


def create_uccl_ep_dispatch_combine_descriptor(
    capability: CudaCommCapability,
    *,
    num_tokens: int,
    hidden: int,
    num_topk: int,
    num_experts: int,
    input_dtype: str,
    include_topk_weights: bool = True,
) -> UcclEpDispatchCombineDescriptor:
    if capability.backend != "uccl":
        raise ValueError("UCCL EP dispatch/combine descriptors require a uccl capability")
    if not capability.supports(CudaCommOp.EP_DISPATCH_COMBINE):
        raise ValueError("uccl capability does not support ep_dispatch_combine")
    if num_tokens <= 0 or hidden <= 0 or num_topk <= 0 or num_experts <= 0:
        raise ValueError("UCCL EP dispatch/combine dimensions must be positive")
    if num_topk > num_experts:
        raise ValueError("UCCL EP num_topk must be less than or equal to num_experts")
    if num_experts % capability.world_size != 0:
        raise ValueError("UCCL EP num_experts must be divisible by world_size")
    normalized_dtype = input_dtype.lower()
    if normalized_dtype not in _UCCL_EP_INPUT_DTYPES:
        raise ValueError("UCCL EP input_dtype must be one of: bf16, fp8")
    return UcclEpDispatchCombineDescriptor(
        world_size=capability.world_size,
        num_tokens=int(num_tokens),
        hidden=int(hidden),
        num_topk=int(num_topk),
        num_experts=int(num_experts),
        input_dtype=normalized_dtype,
        include_topk_weights=bool(include_topk_weights),
    )


class UcclP2PCudaCommRuntime:
    """Internal UCCL-P2P IPC adapter for one local rank."""

    def __init__(
        self,
        capability: CudaCommCapability,
        *,
        rank: int,
        p2p_module: ModuleType | Any | None = None,
        endpoint: Any | None = None,
    ) -> None:
        if capability.backend != "uccl":
            raise ValueError("UcclP2PCudaCommRuntime requires a uccl capability")
        rank_to_device = capability.rank_to_device()
        if rank not in rank_to_device:
            raise ValueError(f"unknown UCCL rank {rank}")

        self.capability = capability
        self.rank = int(rank)
        self.device_id = rank_to_device[self.rank]
        self.runtime_id = _runtime_id(capability, self.rank)
        if endpoint is not None:
            self._endpoint = endpoint
        else:
            p2p = _import_uccl_p2p() if p2p_module is None else p2p_module
            self._endpoint = p2p.Endpoint(self.device_id)

    def close(self) -> None:
        if hasattr(self._endpoint, "close"):
            self._endpoint.close()

    def get_metadata(self) -> bytes:
        return bytes(self._endpoint.get_metadata())

    def accept_local(self, *, peer_rank: int, peer_address: str | None = None) -> int:
        expected_peer_device = self._device_for_rank(peer_rank)
        ok, remote_peer, conn_id = self._endpoint.accept_local()
        if not ok:
            raise RuntimeError("UCCL P2P accept_local failed")
        if peer_address is not None:
            if str(remote_peer) != peer_address:
                raise ValueError(
                    f"UCCL P2P peer rank {peer_rank} resolved to {peer_address}, "
                    f"but accept_local returned {remote_peer}"
                )
        elif not isinstance(remote_peer, str) and int(remote_peer) != expected_peer_device:
            raise ValueError(
                f"UCCL P2P peer rank {peer_rank} resolved to cuda{expected_peer_device}, "
                f"but accept_local returned cuda{remote_peer}"
            )
        return int(conn_id)

    def connect_local(self, *, peer_rank: int, peer_address: str | None = None) -> int:
        peer_device = self._device_for_rank(peer_rank)
        peer_target = peer_device if peer_address is None else peer_address
        ok, conn_id = self._endpoint.connect_local(peer_target)
        if not ok:
            raise RuntimeError(f"UCCL P2P connect_local failed for peer rank {peer_rank}")
        return int(conn_id)

    def advertise_write_ipc(
        self,
        descriptor: UcclP2PWriteIpcDescriptor,
        *,
        conn_id: int,
        dst_ptr: int,
    ) -> bytes:
        self._validate_descriptor(descriptor)
        if self.rank != descriptor.dst_rank:
            raise ValueError("UCCL P2P write_ipc advertise must run on the destination rank")
        ok, info_blob = self._endpoint.advertise_ipc(
            int(conn_id),
            int(dst_ptr),
            descriptor.nbytes,
        )
        if not ok:
            raise RuntimeError("UCCL P2P advertise_ipc failed")
        return bytes(info_blob)

    def write_ipc(
        self,
        descriptor: UcclP2PWriteIpcDescriptor,
        *,
        conn_id: int,
        src_ptr: int,
        info_blob: bytes,
    ) -> None:
        self._validate_descriptor(descriptor)
        if self.rank != descriptor.src_rank:
            raise ValueError("UCCL P2P write_ipc must run on the source rank")
        ok = self._endpoint.write_ipc(
            int(conn_id),
            int(src_ptr),
            descriptor.nbytes,
            bytes(info_blob),
        )
        if not ok:
            raise RuntimeError("UCCL P2P write_ipc failed")

    def _device_for_rank(self, rank: int) -> int:
        rank_to_device = self.capability.rank_to_device()
        if rank not in rank_to_device:
            raise ValueError(f"unknown UCCL peer rank {rank}")
        return rank_to_device[rank]

    def _validate_descriptor(self, descriptor: UcclP2PWriteIpcDescriptor) -> None:
        create_uccl_p2p_write_ipc_descriptor(
            self.capability,
            src_rank=descriptor.src_rank,
            dst_rank=descriptor.dst_rank,
            nbytes=descriptor.nbytes,
        )


class CudaCommRuntimeRegistry:
    """Runtime-private cache for CUDA communication capability state."""

    def __init__(self) -> None:
        self._runtimes: dict[
            str,
            MockCudaCommRuntime | TorchNcclCudaCommRuntime | UcclP2PCudaCommRuntime,
        ] = {}

    def acquire(
        self,
        capability: CudaCommCapability,
        *,
        rank: int | None = None,
        init_method: str | None = None,
        torch_module: ModuleType | Any | None = None,
        dist_module: ModuleType | Any | None = None,
        uccl_transport: str | None = None,
        p2p_module: ModuleType | Any | None = None,
        endpoint: Any | None = None,
    ) -> MockCudaCommRuntime | TorchNcclCudaCommRuntime | UcclP2PCudaCommRuntime:
        runtime_id = (
            _runtime_id(capability, rank)
            if capability.backend in {"nccl", "uccl"}
            else capability.capability_id
        )
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
        elif capability.backend == "uccl":
            if rank is None:
                raise ValueError("UCCL runtime acquisition requires rank")
            if uccl_transport != _UCCL_TRANSPORT_P2P_IPC:
                raise NotImplementedError("UCCL runtime acquisition requires uccl_transport='p2p_ipc'")
            runtime = UcclP2PCudaCommRuntime(
                capability,
                rank=rank,
                p2p_module=p2p_module,
                endpoint=endpoint,
            )
        else:
            raise NotImplementedError(f"{capability.backend} communicator lifecycle is not implemented yet")

        self._runtimes[runtime_id] = runtime
        return runtime

    def get(self, runtime_id: str) -> MockCudaCommRuntime | TorchNcclCudaCommRuntime | UcclP2PCudaCommRuntime:
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


def _import_uccl_p2p() -> ModuleType:
    _import_torch()
    from uccl import p2p  # noqa: PLC0415

    return p2p
