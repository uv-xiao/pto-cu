#!/usr/bin/env python3
"""Two-rank UCCL-P2P IPC smoke through the private PTO CUDA comm adapter."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from collections.abc import Callable

from simpler_setup.cuda_comm import (
    CudaCommRuntimeRegistry,
    create_cuda_comm_host_plan,
    create_uccl_p2p_write_ipc_descriptor,
)

OPERATION = "p2p_write_ipc"
TRANSPORT = "p2p_ipc"


def parse_device_ids(value: str) -> tuple[int, ...]:
    device_ids = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(device_ids) != 2:
        raise argparse.ArgumentTypeError("expected exactly two comma-separated device ids")
    if len(set(device_ids)) != len(device_ids):
        raise argparse.ArgumentTypeError("device ids must be unique")
    return device_ids


def uccl_p2p_skip_reason() -> str | None:
    if importlib.util.find_spec("uccl") is None:
        return "UCCL-P2P dependencies unavailable: ModuleNotFoundError: No module named 'uccl'"

    try:
        import torch  # noqa: PLC0415
        import torch.distributed as dist  # noqa: PLC0415
        from uccl import p2p  # noqa: F401, PLC0415
    except Exception as exc:  # pragma: no cover - depends on optional packages
        return f"UCCL-P2P dependencies unavailable: {type(exc).__name__}: {exc}"

    world_size = os.environ.get("WORLD_SIZE")
    if world_size != "2":
        return "requires torchrun with WORLD_SIZE=2"
    if not dist.is_available():
        return "torch.distributed is unavailable"
    if not torch.cuda.is_available():
        return "CUDA is unavailable"
    if torch.cuda.device_count() < 2:
        return f"need at least 2 visible CUDA devices, found {torch.cuda.device_count()}"
    return None


def run_uccl_p2p_ipc_adapter(
    *,
    device_ids: tuple[int, int] = (0, 1),
    nbytes: int = 1024,
    skip_reason: Callable[[], str | None] | None = None,
) -> dict:
    if len(device_ids) != 2:
        raise ValueError("UCCL-P2P IPC adapter smoke expects exactly two device ids")
    if nbytes <= 0:
        raise ValueError("nbytes must be positive")

    host_plan = create_cuda_comm_host_plan(backend="uccl", device_ids=device_ids)
    descriptor = create_uccl_p2p_write_ipc_descriptor(
        host_plan.capability,
        src_rank=0,
        dst_rank=1,
        nbytes=nbytes,
    )
    result = {
        "backend": "uccl",
        "transport": TRANSPORT,
        "operation": OPERATION,
        "world_size": len(device_ids),
        "device_ids": list(device_ids),
        "capability": host_plan.capability.as_dict(),
        "launch_plans": [plan.as_dict() for plan in host_plan.launch_plans],
        "descriptor": descriptor.as_dict(),
        "nbytes": int(nbytes),
    }

    check = uccl_p2p_skip_reason if skip_reason is None else skip_reason
    reason = check()
    if reason is not None:
        return {**result, "status": "skipped", "reason": reason}

    return _run_distributed_write_ipc(result, host_plan, descriptor)


def _run_distributed_write_ipc(result: dict, host_plan, descriptor) -> dict:
    import torch  # noqa: PLC0415
    import torch.distributed as dist  # noqa: PLC0415
    from uccl import p2p  # noqa: PLC0415

    created_process_group = False
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
        created_process_group = True

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise RuntimeError(f"UCCL-P2P IPC adapter smoke expects world_size=2, got {world_size}")

    launch_plan = host_plan.launch_plan_for_worker(rank)
    torch.cuda.set_device(launch_plan.device_id)
    registry = CudaCommRuntimeRegistry()
    runtime = registry.acquire(
        host_plan.capability,
        rank=rank,
        uccl_transport=TRANSPORT,
        p2p_module=p2p,
    )
    remote_metadata = _exchange_bytes(dist, torch, runtime.get_metadata(), rank=rank)
    _, _, remote_peer_address = p2p.Endpoint.parse_metadata(remote_metadata)
    dist.barrier()

    try:
        if rank == descriptor.dst_rank:
            rank_result = _run_write_ipc_server(
                torch,
                dist,
                runtime,
                descriptor,
                launch_plan.device_id,
                peer_address=remote_peer_address,
            )
        elif rank == descriptor.src_rank:
            rank_result = _run_write_ipc_client(
                torch,
                dist,
                runtime,
                descriptor,
                launch_plan.device_id,
                peer_address=remote_peer_address,
            )
        else:  # pragma: no cover - guarded by two-rank descriptor construction
            raise RuntimeError(f"unexpected rank {rank}")

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
        registry.release(runtime.runtime_id)
        if created_process_group:
            dist.destroy_process_group()


def _run_write_ipc_server(torch, dist, runtime, descriptor, device_id: int, *, peer_address: str) -> dict:
    conn_id = runtime.accept_local(peer_rank=descriptor.src_rank, peer_address=peer_address)
    dst = torch.zeros(descriptor.nbytes, dtype=torch.uint8, device=f"cuda:{device_id}")
    info_blob = runtime.advertise_write_ipc(descriptor, conn_id=conn_id, dst_ptr=dst.data_ptr())
    _send_bytes(dist, torch, info_blob, dst=descriptor.src_rank)
    _recv_int(dist, torch, src=descriptor.src_rank)
    torch.cuda.synchronize(device_id)
    passed = bool(torch.all(dst == 7).item())
    return {
        "rank": descriptor.dst_rank,
        "role": "server",
        "conn_id": conn_id,
        "nbytes": descriptor.nbytes,
        "passed": passed,
    }


def _run_write_ipc_client(torch, dist, runtime, descriptor, device_id: int, *, peer_address: str) -> dict:
    conn_id = runtime.connect_local(peer_rank=descriptor.dst_rank, peer_address=peer_address)
    src = torch.full((descriptor.nbytes,), 7, dtype=torch.uint8, device=f"cuda:{device_id}")
    info_blob = _recv_bytes(dist, torch, src=descriptor.dst_rank)
    runtime.write_ipc(descriptor, conn_id=conn_id, src_ptr=src.data_ptr(), info_blob=info_blob)
    torch.cuda.synchronize(device_id)
    _send_int(dist, torch, 1, dst=descriptor.dst_rank)
    return {
        "rank": descriptor.src_rank,
        "role": "client",
        "conn_id": conn_id,
        "nbytes": descriptor.nbytes,
        "passed": True,
    }


def _send_int(dist, torch, value: int, *, dst: int) -> None:
    dist.send(torch.tensor([int(value)], dtype=torch.int64), dst=dst)


def _recv_int(dist, torch, *, src: int) -> int:
    value = torch.empty(1, dtype=torch.int64)
    dist.recv(value, src=src)
    return int(value.item())


def _send_bytes(dist, torch, payload: bytes, *, dst: int) -> None:
    data = bytes(payload)
    _send_int(dist, torch, len(data), dst=dst)
    if data:
        dist.send(torch.tensor(list(data), dtype=torch.uint8), dst=dst)


def _recv_bytes(dist, torch, *, src: int) -> bytes:
    nbytes = _recv_int(dist, torch, src=src)
    if nbytes == 0:
        return b""
    payload = torch.empty(nbytes, dtype=torch.uint8)
    dist.recv(payload, src=src)
    return bytes(payload.tolist())


def _exchange_bytes(dist, torch, payload: bytes, *, rank: int) -> bytes:
    if rank == 0:
        _send_bytes(dist, torch, payload, dst=1)
        return _recv_bytes(dist, torch, src=1)
    remote = _recv_bytes(dist, torch, src=0)
    _send_bytes(dist, torch, payload, dst=0)
    return remote


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-ids", type=parse_device_ids, default=(0, 1))
    parser.add_argument("--nbytes", type=int, default=1024)
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="return a non-zero status when dependencies, torchrun, or CUDA are unavailable",
    )
    args = parser.parse_args(argv)

    result = run_uccl_p2p_ipc_adapter(device_ids=args.device_ids, nbytes=args.nbytes)
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
