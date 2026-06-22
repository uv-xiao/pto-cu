"""Tests for internal CUDA communication capability descriptors."""

from __future__ import annotations

import importlib.util
import struct
import sys
import zlib
from pathlib import Path

import pytest
import simpler
from simpler import worker as worker_mod

from simpler_setup.cuda_comm import (
    CudaCommDeviceDescriptor,
    CudaCommHostPlan,
    CudaCommLaunchPlan,
    CudaCommOp,
    CudaCommRuntimeRegistry,
    MockCudaCommRuntime,
    TorchNcclCudaCommRuntime,
    UcclEpDispatchCombineDescriptor,
    UcclP2PCudaCommRuntime,
    UcclP2PWriteIpcDescriptor,
    create_cuda_comm_capability,
    create_cuda_comm_host_plan,
    create_cuda_comm_launch_plan,
    create_mock_cuda_comm_capability,
    create_uccl_ep_dispatch_combine_descriptor,
    create_uccl_p2p_write_ipc_descriptor,
)

ROOT = Path(__file__).resolve().parents[3]


def _load_nccl_worker_control_example():
    example = ROOT / "examples" / "cuda" / "nccl_worker_control_ops.py"
    assert example.is_file(), f"missing {example.relative_to(ROOT)}"
    spec = importlib.util.spec_from_file_location("nccl_worker_control_ops_example", example)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_uccl_p2p_adapter_example():
    example = ROOT / "examples" / "cuda" / "uccl_p2p_ipc_adapter.py"
    assert example.is_file(), f"missing {example.relative_to(ROOT)}"
    spec = importlib.util.spec_from_file_location("uccl_p2p_ipc_adapter_example", example)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_uccl_ep_adapter_example():
    example = ROOT / "examples" / "cuda" / "uccl_ep_dispatch_combine_adapter.py"
    assert example.is_file(), f"missing {example.relative_to(ROOT)}"
    spec = importlib.util.spec_from_file_location("uccl_ep_dispatch_combine_adapter_example", example)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_persistent_moe_example():
    example = ROOT / "examples" / "cuda" / "persistent_moe_dispatch_combine.py"
    assert example.is_file(), f"missing {example.relative_to(ROOT)}"
    spec = importlib.util.spec_from_file_location("persistent_moe_dispatch_combine_example", example)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cuda_comm_capability_is_internal_and_opaque():
    capability = create_mock_cuda_comm_capability(device_ids=(2, 5))

    assert capability.backend == "mock"
    assert capability.world_size == 2
    assert capability.rank_to_device() == {0: 2, 1: 5}
    assert capability.supports(CudaCommOp.ALL_REDUCE)
    assert capability.supports("reduce_scatter")
    assert capability.capability_id.startswith("mock:")
    assert "nccl" not in capability.capability_id.lower()
    assert "uccl" not in capability.capability_id.lower()
    assert not hasattr(simpler, "CudaCommCapability")


def test_cuda_comm_capability_supports_nccl_without_transport_objects():
    capability = create_cuda_comm_capability(backend="nccl", device_ids=(0, 1))

    assert capability.backend == "nccl"
    assert capability.world_size == 2
    assert capability.rank_to_device() == {0: 0, 1: 1}
    assert capability.supports(CudaCommOp.REDUCE_SCATTER)
    assert capability.as_dict() == {
        "backend": "nccl",
        "capability_id": "nccl:rank0->cuda0,rank1->cuda1",
        "world_size": 2,
        "rank_to_device": {"0": 0, "1": 1},
        "operations": ["all_reduce", "reduce_scatter", "all_gather", "send_recv"],
    }
    assert not hasattr(simpler, "TorchNcclCudaCommRuntime")


def test_cuda_comm_capability_supports_uccl_p2p_without_transport_objects():
    capability = create_cuda_comm_capability(backend="uccl", device_ids=(6, 7))

    assert capability.backend == "uccl"
    assert capability.world_size == 2
    assert capability.rank_to_device() == {0: 6, 1: 7}
    assert capability.supports(CudaCommOp.P2P_WRITE_IPC)
    assert capability.supports(CudaCommOp.EP_DISPATCH_COMBINE)
    assert not capability.supports(CudaCommOp.ALL_REDUCE)
    assert capability.as_dict() == {
        "backend": "uccl",
        "capability_id": "uccl:rank0->cuda6,rank1->cuda7",
        "world_size": 2,
        "rank_to_device": {"0": 6, "1": 7},
        "operations": ["p2p_write_ipc", "ep_dispatch_combine"],
    }
    assert not hasattr(simpler, "UcclP2PCudaCommRuntime")


def test_cuda_comm_launch_plan_resolves_local_rank_without_transport_objects():
    capability = create_cuda_comm_capability(backend="nccl", device_ids=(2, 7))
    plan = create_cuda_comm_launch_plan(capability, rank=1)

    assert isinstance(plan, CudaCommLaunchPlan)
    assert plan.capability is capability
    assert plan.backend == "nccl"
    assert plan.rank == 1
    assert plan.device_id == 7
    assert plan.world_size == 2
    assert plan.runtime_id == "nccl:rank0->cuda2,rank1->cuda7/local_rank1"
    assert plan.as_dict() == {
        "backend": "nccl",
        "capability_id": "nccl:rank0->cuda2,rank1->cuda7",
        "runtime_id": "nccl:rank0->cuda2,rank1->cuda7/local_rank1",
        "rank": 1,
        "device_id": 7,
        "world_size": 2,
    }
    assert not hasattr(simpler, "CudaCommLaunchPlan")


def test_cuda_comm_launch_plan_resolves_uccl_local_rank_runtime_id():
    capability = create_cuda_comm_capability(backend="uccl", device_ids=(6, 7))
    plan = create_cuda_comm_launch_plan(capability, rank=1)

    assert plan.backend == "uccl"
    assert plan.rank == 1
    assert plan.device_id == 7
    assert plan.world_size == 2
    assert plan.runtime_id == "uccl:rank0->cuda6,rank1->cuda7/local_rank1"
    assert plan.device_descriptor().as_dict()["backend_code"] == 2


def test_cuda_comm_launch_plan_rejects_unknown_rank():
    capability = create_cuda_comm_capability(backend="nccl", device_ids=(0, 1))

    with pytest.raises(ValueError, match="unknown rank"):
        create_cuda_comm_launch_plan(capability, rank=2)


def test_cuda_comm_launch_plan_exports_compact_device_descriptor():
    capability = create_cuda_comm_capability(backend="nccl", device_ids=(2, 7))
    plan = create_cuda_comm_launch_plan(capability, rank=1)

    descriptor = plan.device_descriptor()
    expected_crc = zlib.crc32(capability.capability_id.encode("utf-8"))

    assert isinstance(descriptor, CudaCommDeviceDescriptor)
    assert descriptor.as_dict() == {
        "backend": "nccl",
        "backend_code": 1,
        "rank": 1,
        "device_id": 7,
        "world_size": 2,
        "capability_crc32": expected_crc,
    }
    assert descriptor.to_bytes() == struct.pack("<IIIII", 1, 1, 7, 2, expected_crc)
    assert len(descriptor.to_bytes()) == 20
    assert not hasattr(simpler, "CudaCommDeviceDescriptor")


def test_cuda_comm_host_plan_maps_worker_indices_to_launch_plans():
    host_plan = create_cuda_comm_host_plan(backend="nccl", device_ids=(2, 7))

    assert isinstance(host_plan, CudaCommHostPlan)
    assert host_plan.backend == "nccl"
    assert host_plan.world_size == 2
    assert host_plan.device_ids == (2, 7)
    assert host_plan.capability.capability_id == "nccl:rank0->cuda2,rank1->cuda7"
    assert host_plan.runtime_ids() == (
        "nccl:rank0->cuda2,rank1->cuda7/local_rank0",
        "nccl:rank0->cuda2,rank1->cuda7/local_rank1",
    )
    assert host_plan.launch_plan_for_worker(0).rank == 0
    assert host_plan.launch_plan_for_worker(0).device_id == 2
    assert host_plan.launch_plan_for_worker(1).rank == 1
    assert host_plan.launch_plan_for_worker(1).device_id == 7
    assert host_plan.as_dict() == {
        "backend": "nccl",
        "world_size": 2,
        "device_ids": [2, 7],
        "capability": {
            "backend": "nccl",
            "capability_id": "nccl:rank0->cuda2,rank1->cuda7",
            "world_size": 2,
            "rank_to_device": {"0": 2, "1": 7},
            "operations": ["all_reduce", "reduce_scatter", "all_gather", "send_recv"],
        },
        "launch_plans": [
            {
                "backend": "nccl",
                "capability_id": "nccl:rank0->cuda2,rank1->cuda7",
                "runtime_id": "nccl:rank0->cuda2,rank1->cuda7/local_rank0",
                "rank": 0,
                "device_id": 2,
                "world_size": 2,
            },
            {
                "backend": "nccl",
                "capability_id": "nccl:rank0->cuda2,rank1->cuda7",
                "runtime_id": "nccl:rank0->cuda2,rank1->cuda7/local_rank1",
                "rank": 1,
                "device_id": 7,
                "world_size": 2,
            },
        ],
    }
    assert not hasattr(simpler, "CudaCommHostPlan")


def test_cuda_comm_host_plan_rejects_unknown_worker_index():
    host_plan = create_cuda_comm_host_plan(backend="nccl", device_ids=(0, 1))

    with pytest.raises(ValueError, match="worker index"):
        host_plan.launch_plan_for_worker(2)


def test_cuda_comm_capability_rejects_ambiguous_rank_mapping():
    with pytest.raises(ValueError, match="unique"):
        create_mock_cuda_comm_capability(device_ids=(0, 0))

    with pytest.raises(ValueError, match="at least one"):
        create_mock_cuda_comm_capability(device_ids=())

    with pytest.raises(ValueError, match="unsupported"):
        create_cuda_comm_capability(backend="ray", device_ids=(0, 1))


def test_nccl_worker_control_ops_example_is_skip_safe():
    module = _load_nccl_worker_control_example()

    result = module.run_worker_control_ops(
        device_ids=(0, 1),
        tensor_numel=4,
        skip_reason=lambda _min_gpus: "no test GPU",
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "no test GPU"
    assert result["transport"] == "worker_control"
    assert result["operations"] == ["all_reduce", "reduce_scatter", "all_gather", "send_recv"]


def test_nccl_worker_control_ops_example_discovers_bundled_nccl(monkeypatch, tmp_path):
    module = _load_nccl_worker_control_example()
    nccl_lib = tmp_path / "nvidia" / "nccl" / "lib" / "libnccl.so.2"
    nccl_lib.parent.mkdir(parents=True)
    nccl_lib.write_bytes(b"fake nccl")
    monkeypatch.delenv(module._NCCL_LIBRARY_ENV, raising=False)

    configured = module.configure_nccl_library_env(search_paths=[tmp_path])

    assert configured == str(nccl_lib)
    assert module.os.environ[module._NCCL_LIBRARY_ENV] == str(nccl_lib)


def test_nccl_worker_control_ops_example_respects_explicit_nccl_library(monkeypatch, tmp_path):
    module = _load_nccl_worker_control_example()
    explicit = tmp_path / "explicit-libnccl.so.2"
    monkeypatch.setenv(module._NCCL_LIBRARY_ENV, str(explicit))

    assert module.configure_nccl_library_env(search_paths=[]) == str(explicit)


def test_uccl_p2p_ipc_adapter_example_is_skip_safe():
    module = _load_uccl_p2p_adapter_example()

    result = module.run_uccl_p2p_ipc_adapter(
        device_ids=(0, 1),
        nbytes=1024,
        skip_reason=lambda: "no test UCCL",
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "no test UCCL"
    assert result["backend"] == "uccl"
    assert result["transport"] == "p2p_ipc"
    assert result["operation"] == "p2p_write_ipc"
    assert result["capability"]["operations"] == ["p2p_write_ipc", "ep_dispatch_combine"]
    assert result["descriptor"] == {
        "operation": "p2p_write_ipc",
        "src_rank": 0,
        "dst_rank": 1,
        "nbytes": 1024,
    }


def test_uccl_p2p_ipc_adapter_cli_returns_nonzero_when_cuda_required(monkeypatch, capsys):
    module = _load_uccl_p2p_adapter_example()
    monkeypatch.setattr(module, "uccl_p2p_skip_reason", lambda: "no test UCCL")

    assert module.main([]) == 0
    assert module.main(["--require-cuda"]) == 2
    captured = capsys.readouterr()
    assert '"status": "skipped"' in captured.out
    assert "no test UCCL" in captured.out


def test_uccl_ep_dispatch_combine_adapter_example_is_skip_safe():
    module = _load_uccl_ep_adapter_example()

    result = module.run_uccl_ep_dispatch_combine_adapter(
        device_ids=(0, 1),
        num_tokens=64,
        hidden=128,
        num_topk=4,
        num_experts=16,
        input_dtype="bf16",
        repeats=3,
        skip_reason=lambda: "no test UCCL-EP",
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "no test UCCL-EP"
    assert result["backend"] == "uccl"
    assert result["transport"] == "ep"
    assert result["operation"] == "ep_dispatch_combine"
    assert result["repeats"] == 3
    assert result["capability"]["operations"] == ["p2p_write_ipc", "ep_dispatch_combine"]
    assert result["descriptor"] == {
        "operation": "ep_dispatch_combine",
        "world_size": 2,
        "num_tokens": 64,
        "hidden": 128,
        "num_topk": 4,
        "num_experts": 16,
        "experts_per_rank": 8,
        "input_dtype": "bf16",
        "include_topk_weights": True,
        "metadata_shapes": {
            "topk_idx": [64, 4],
            "topk_weights": [64, 4],
            "num_tokens_per_rank": [2],
            "is_token_in_rank": [64, 2],
            "num_tokens_per_expert": [16],
        },
    }


def test_uccl_ep_dispatch_combine_adapter_cli_returns_nonzero_when_cuda_required(monkeypatch, capsys):
    module = _load_uccl_ep_adapter_example()
    monkeypatch.setattr(module, "uccl_ep_skip_reason", lambda **_kwargs: "no test UCCL-EP")

    assert module.main([]) == 0
    assert module.main(["--require-cuda"]) == 2
    captured = capsys.readouterr()
    assert '"status": "skipped"' in captured.out
    assert "no test UCCL-EP" in captured.out


def test_uccl_ep_dispatch_combine_adapter_validates_repeats():
    module = _load_uccl_ep_adapter_example()

    with pytest.raises(ValueError, match="repeats"):
        module.run_uccl_ep_dispatch_combine_adapter(
            device_ids=(0, 1),
            repeats=0,
            skip_reason=lambda: "no test UCCL-EP",
        )


def test_persistent_moe_uccl_ep_fused_boundary_reports_unsupported():
    module = _load_persistent_moe_example()
    source_digests = {
        "dispatch_source_sha256": "dispatch",
        "gluon_expert_bridge_sha256": "bridge",
        "task_body_func12_sha256": "bridge",
    }

    def fake_moe_runner(**_kwargs):
        return {
            "status": "passed",
            "device_ids": [6, 7],
            "evidence_scope": "same-node-two-device-baseline",
            "validation": {
                "all_devices_passed": True,
                "completed_count_is_5": True,
                "scheduler_errors_zero": True,
                "fanin_remaining_zero": True,
                "source_digests_match": True,
                "bridge_metadata_match": True,
            },
            "source_digests": source_digests,
            "per_device_results": [
                {
                    "device": 6,
                    "max_abs_error": 0.0,
                    "device_scheduler_errors": {"count": 0, "code": 0, "task_id": 0},
                },
                {
                    "device": 7,
                    "max_abs_error": 0.0,
                    "device_scheduler_errors": {"count": 0, "code": 0, "task_id": 0},
                },
            ],
        }

    def fake_uccl_ep_runner(**_kwargs):
        return {
            "status": "passed",
            "backend": "uccl",
            "transport": "ep",
            "operation": "ep_dispatch_combine",
            "device_ids": [6, 7],
            "capability": {"capability_id": "uccl:rank0->cuda6,rank1->cuda7"},
            "descriptor": {
                "num_tokens": 64,
                "hidden": 1024,
                "num_topk": 4,
                "num_experts": 16,
                "input_dtype": "bf16",
            },
            "rank_results": [
                {
                    "rank": 0,
                    "passed": True,
                    "max_abs_error": 0.0,
                    "topk_weight_error": 0.0,
                },
                {
                    "rank": 1,
                    "passed": True,
                    "max_abs_error": 0.0,
                    "topk_weight_error": 0.0,
                },
            ],
        }

    result = module.run_persistent_moe_uccl_ep_fused_boundary(
        device_ids=(6, 7),
        tensor_numel=1024,
        moe_runner=fake_moe_runner,
        uccl_ep_runner=fake_uccl_ep_runner,
    )

    assert result["status"] == "unsupported"
    assert result["fused_boundary_scope"] == (
        "reduced-fused-cross-gpu-expert-parallel-moe-boundary"
    )
    assert result["handoff_scope"] == "persistent-moe-plus-uccl-ep-adapter"
    assert result["boundary_validation"]["handoff_passed"] is True
    assert result["boundary_validation"]["actual_fused_cross_gpu_execution"] is False
    assert result["boundary_validation"]["structured_unsupported_boundary"] is True
    assert "persistent_device_uccl_ep_runtime_fusion" in result["missing_boundaries"]
    assert "non-evidence" in result["evidence_statement"]


def test_nccl_worker_control_ops_example_drives_ctrl_comm_op_with_worker_memory():
    module = _load_nccl_worker_control_example()
    created_workers = []

    class FakeOrchestrator:
        def __init__(self):
            self.device_memory = {}
            self.next_ptr = 0x1000

        def malloc(self, worker_id: int, size: int) -> int:
            ptr = self.next_ptr + worker_id * 0x100000
            self.next_ptr += 0x1000
            self.device_memory[ptr] = [0.0] * (size // 4)
            return ptr

        def copy_to(self, worker_id: int, dst: int, src: int, size: int) -> None:
            self.device_memory[dst] = module._HostFloatBuffer.from_address(src).read(size // 4)

        def copy_from(self, worker_id: int, dst: int, src: int, size: int) -> None:
            module._HostFloatBuffer.from_address(dst).write(self.device_memory[src][: size // 4])

        def free(self, worker_id: int, ptr: int) -> None:
            self.device_memory.pop(ptr, None)

    class FakeWorker:
        def __init__(self, **config):
            self.config = config
            self.orch = FakeOrchestrator()
            self.dispatches = []
            created_workers.append(self)

        def init(self) -> None:
            self.initialized = True

        def run(self, fn, args=None, config=None) -> None:
            fn(self.orch, args, config)

        def close(self) -> None:
            self.closed = True

        def _dispatch_control_comm_op(self, **kwargs) -> None:
            self.dispatches.append(kwargs)
            workers = tuple(kwargs["workers"])
            op_code = kwargs["op_code"]
            send_ptrs = kwargs["send_ptrs"]
            recv_ptrs = kwargs["recv_ptrs"]
            counts = kwargs["counts"]
            count_for = counts if isinstance(counts, dict) else {worker: counts for worker in workers}

            if op_code == worker_mod._COMM_OP_ALL_REDUCE_F32:
                count = count_for[workers[0]]
                reduced = [sum(self.orch.device_memory[send_ptrs[worker]][idx] for worker in workers) for idx in range(count)]
                for worker in workers:
                    self.orch.device_memory[recv_ptrs[worker]][:count] = reduced
            elif op_code == worker_mod._COMM_OP_REDUCE_SCATTER_F32:
                count = count_for[workers[0]]
                for worker in workers:
                    offset = worker * count
                    self.orch.device_memory[recv_ptrs[worker]][:count] = [
                        sum(self.orch.device_memory[send_ptrs[src]][offset + idx] for src in workers)
                        for idx in range(count)
                    ]
            elif op_code == worker_mod._COMM_OP_ALL_GATHER_F32:
                count = count_for[workers[0]]
                gathered = []
                for worker in workers:
                    gathered.extend(self.orch.device_memory[send_ptrs[worker]][:count])
                for worker in workers:
                    self.orch.device_memory[recv_ptrs[worker]][: len(gathered)] = gathered
            elif op_code == worker_mod._COMM_OP_SEND_RECV_F32:
                dst_ranks = kwargs["dst_ranks"]
                src_ranks = kwargs["src_ranks"]
                for worker in workers:
                    count = count_for[worker]
                    assert dst_ranks[src_ranks[worker]] == worker
                    self.orch.device_memory[recv_ptrs[worker]][:count] = self.orch.device_memory[
                        send_ptrs[src_ranks[worker]]
                    ][:count]
            else:  # pragma: no cover - defensive branch for future op additions
                raise AssertionError(f"unexpected op_code {op_code}")

    result = module.run_worker_control_ops(
        device_ids=(0, 1),
        tensor_numel=4,
        skip_reason=lambda _min_gpus: None,
        worker_factory=FakeWorker,
    )

    assert result["status"] == "passed"
    assert result["transport"] == "worker_control"
    assert result["all_reduce"]["passed"] is True
    assert result["reduce_scatter"]["passed"] is True
    assert result["all_gather"]["passed"] is True
    assert result["send_recv"]["passed"] is True
    assert [item["op_code"] for item in created_workers[0].dispatches] == [
        worker_mod._COMM_OP_ALL_REDUCE_F32,
        worker_mod._COMM_OP_REDUCE_SCATTER_F32,
        worker_mod._COMM_OP_ALL_GATHER_F32,
        worker_mod._COMM_OP_SEND_RECV_F32,
    ]
    assert created_workers[0].config["device_ids"] == [0, 1]


def test_nccl_worker_control_expected_values_track_float32_storage():
    module = _load_nccl_worker_control_example()

    assert module._expected_all_reduce_f32(16777217) == 33554432.0
    assert module._expected_reduce_scatter_f32(dst_rank=0, idx=16777217) == 33554432.0
    assert module._expected_reduce_scatter_f32(dst_rank=1, idx=16777217) == 33554456.0
    assert module._expected_all_gather_f32(src_rank=0, idx=29360127) == 7340032.0
    assert module._expected_all_gather_f32(src_rank=1, idx=29360127) == 7340033.0


def test_mock_cuda_comm_runtime_matches_baseline_collective_shapes():
    runtime = MockCudaCommRuntime(create_mock_cuda_comm_capability(device_ids=(0, 1)))

    all_reduce = runtime.all_reduce(([1.0, 2.0], [10.0, 20.0]))
    assert all_reduce == ((11.0, 22.0), (11.0, 22.0))

    reduce_scatter = runtime.reduce_scatter(([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]))
    assert reduce_scatter == ((11.0, 22.0), (33.0, 44.0))

    all_gather = runtime.all_gather(([0.0], [1.0]))
    assert all_gather == ((0.0, 1.0), (0.0, 1.0))

    send_recv = runtime.send_recv({(0, 1): [23.0], (1, 0): [17.0]})
    assert send_recv == {0: (17.0,), 1: (23.0,)}


def test_mock_cuda_comm_runtime_validates_world_size_and_shapes():
    runtime = MockCudaCommRuntime(create_mock_cuda_comm_capability(device_ids=(0, 1)))

    with pytest.raises(ValueError, match="world_size"):
        runtime.all_reduce(([1.0],))

    with pytest.raises(ValueError, match="same length"):
        runtime.all_reduce(([1.0], [2.0, 3.0]))

    with pytest.raises(ValueError, match="divisible"):
        runtime.reduce_scatter(([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]))

    with pytest.raises(ValueError, match="destination rank"):
        runtime.send_recv({(0, 2): [1.0]})


def test_uccl_p2p_write_ipc_descriptor_validates_rank_and_shape():
    capability = create_cuda_comm_capability(backend="uccl", device_ids=(6, 7))

    descriptor = create_uccl_p2p_write_ipc_descriptor(
        capability,
        src_rank=1,
        dst_rank=0,
        nbytes=4096,
    )

    assert isinstance(descriptor, UcclP2PWriteIpcDescriptor)
    assert descriptor.as_dict() == {
        "operation": "p2p_write_ipc",
        "src_rank": 1,
        "dst_rank": 0,
        "nbytes": 4096,
    }

    with pytest.raises(ValueError, match="uccl"):
        create_uccl_p2p_write_ipc_descriptor(
            create_mock_cuda_comm_capability(device_ids=(0, 1)),
            src_rank=1,
            dst_rank=0,
            nbytes=4096,
        )
    with pytest.raises(ValueError, match="distinct"):
        create_uccl_p2p_write_ipc_descriptor(capability, src_rank=0, dst_rank=0, nbytes=4096)
    with pytest.raises(ValueError, match="nbytes"):
        create_uccl_p2p_write_ipc_descriptor(capability, src_rank=1, dst_rank=0, nbytes=0)
    with pytest.raises(ValueError, match="unknown"):
        create_uccl_p2p_write_ipc_descriptor(capability, src_rank=2, dst_rank=0, nbytes=4096)


def test_uccl_ep_dispatch_combine_descriptor_records_moe_metadata():
    capability = create_cuda_comm_capability(backend="uccl", device_ids=(6, 7))

    descriptor = create_uccl_ep_dispatch_combine_descriptor(
        capability,
        num_tokens=4096,
        hidden=7168,
        num_topk=8,
        num_experts=256,
        input_dtype="bf16",
    )

    assert isinstance(descriptor, UcclEpDispatchCombineDescriptor)
    assert descriptor.experts_per_rank == 128
    assert descriptor.as_dict() == {
        "operation": "ep_dispatch_combine",
        "world_size": 2,
        "num_tokens": 4096,
        "hidden": 7168,
        "num_topk": 8,
        "num_experts": 256,
        "experts_per_rank": 128,
        "input_dtype": "bf16",
        "include_topk_weights": True,
        "metadata_shapes": {
            "topk_idx": [4096, 8],
            "topk_weights": [4096, 8],
            "num_tokens_per_rank": [2],
            "is_token_in_rank": [4096, 2],
            "num_tokens_per_expert": [256],
        },
    }
    assert not hasattr(simpler, "UcclEpDispatchCombineDescriptor")


def test_uccl_ep_dispatch_combine_descriptor_validates_shape_and_backend():
    capability = create_cuda_comm_capability(backend="uccl", device_ids=(6, 7))

    with pytest.raises(ValueError, match="uccl"):
        create_uccl_ep_dispatch_combine_descriptor(
            create_mock_cuda_comm_capability(device_ids=(0, 1)),
            num_tokens=4096,
            hidden=7168,
            num_topk=8,
            num_experts=256,
            input_dtype="bf16",
        )
    with pytest.raises(ValueError, match="positive"):
        create_uccl_ep_dispatch_combine_descriptor(
            capability,
            num_tokens=0,
            hidden=7168,
            num_topk=8,
            num_experts=256,
            input_dtype="bf16",
        )
    with pytest.raises(ValueError, match="divisible"):
        create_uccl_ep_dispatch_combine_descriptor(
            capability,
            num_tokens=4096,
            hidden=7168,
            num_topk=8,
            num_experts=255,
            input_dtype="bf16",
        )
    with pytest.raises(ValueError, match="num_topk"):
        create_uccl_ep_dispatch_combine_descriptor(
            capability,
            num_tokens=4096,
            hidden=7168,
            num_topk=257,
            num_experts=256,
            input_dtype="bf16",
        )
    with pytest.raises(ValueError, match="input_dtype"):
        create_uccl_ep_dispatch_combine_descriptor(
            capability,
            num_tokens=4096,
            hidden=7168,
            num_topk=8,
            num_experts=256,
            input_dtype="float32",
        )


def test_cuda_comm_runtime_registry_owns_mock_lifecycle():
    registry = CudaCommRuntimeRegistry()
    capability = create_mock_cuda_comm_capability(device_ids=(0, 1))

    first = registry.acquire(capability)
    second = registry.acquire(capability)

    assert first is second
    assert registry.active_ids() == (capability.capability_id,)
    assert first.all_gather(([2.0], [5.0])) == ((2.0, 5.0), (2.0, 5.0))

    registry.release(capability.capability_id)
    assert registry.active_ids() == ()
    with pytest.raises(KeyError, match=capability.capability_id):
        registry.get(capability.capability_id)


def test_cuda_comm_runtime_registry_acquires_nccl_runtime_with_rank_lifecycle():
    registry = CudaCommRuntimeRegistry()
    capability = create_cuda_comm_capability(backend="nccl", device_ids=(0, 1))
    fake_torch = _FakeTorch()
    fake_dist = _FakeDist()

    runtime = registry.acquire(
        capability,
        rank=1,
        init_method="tcp://127.0.0.1:12345",
        torch_module=fake_torch,
        dist_module=fake_dist,
    )

    assert isinstance(runtime, TorchNcclCudaCommRuntime)
    assert runtime.rank == 1
    assert runtime.device_id == 1
    assert fake_torch.cuda.device_ids == [1]
    assert fake_dist.init_calls == [
        {
            "backend": "nccl",
            "init_method": "tcp://127.0.0.1:12345",
            "rank": 1,
            "world_size": 2,
        }
    ]
    assert (
        registry.acquire(
            capability,
            rank=1,
            init_method="tcp://127.0.0.1:12345",
            torch_module=fake_torch,
            dist_module=fake_dist,
        )
        is runtime
    )
    assert registry.active_ids() == ("nccl:rank0->cuda0,rank1->cuda1/local_rank1",)

    registry.release(runtime.runtime_id)

    assert fake_dist.destroy_calls == 1
    assert registry.active_ids() == ()


def test_cuda_comm_runtime_registry_acquires_uccl_p2p_runtime_with_rank_lifecycle():
    registry = CudaCommRuntimeRegistry()
    capability = create_cuda_comm_capability(backend="uccl", device_ids=(6, 7))
    fake_p2p = _FakeUcclP2P()

    runtime = registry.acquire(
        capability,
        rank=1,
        uccl_transport="p2p_ipc",
        p2p_module=fake_p2p,
    )

    assert isinstance(runtime, UcclP2PCudaCommRuntime)
    assert runtime.rank == 1
    assert runtime.device_id == 7
    assert runtime.runtime_id == "uccl:rank0->cuda6,rank1->cuda7/local_rank1"
    assert fake_p2p.endpoint_device_ids == [7]
    assert registry.acquire(
        capability,
        rank=1,
        uccl_transport="p2p_ipc",
        p2p_module=fake_p2p,
    ) is runtime
    assert registry.active_ids() == ("uccl:rank0->cuda6,rank1->cuda7/local_rank1",)

    registry.release(runtime.runtime_id)

    assert fake_p2p.endpoints[0].closed is True
    assert registry.active_ids() == ()


def test_uccl_p2p_runtime_wraps_endpoint_ipc_primitives():
    capability = create_cuda_comm_capability(backend="uccl", device_ids=(6, 7))
    descriptor = create_uccl_p2p_write_ipc_descriptor(
        capability,
        src_rank=1,
        dst_rank=0,
        nbytes=4096,
    )
    server_endpoint = _FakeUcclEndpoint(local_gpu_idx=6)
    server_endpoint.accept_local_result = (True, "0000:be:00.0", 41)
    client_endpoint = _FakeUcclEndpoint(local_gpu_idx=7)
    server = UcclP2PCudaCommRuntime(capability, rank=0, endpoint=server_endpoint)
    client = UcclP2PCudaCommRuntime(capability, rank=1, endpoint=client_endpoint)

    assert server.accept_local(peer_rank=1, peer_address="0000:be:00.0") == 41
    assert client.connect_local(peer_rank=0, peer_address="0000:bd:00.0") == 106
    info_blob = server.advertise_write_ipc(descriptor, conn_id=41, dst_ptr=0x2000)
    client.write_ipc(descriptor, conn_id=106, src_ptr=0x1000, info_blob=info_blob)

    assert server_endpoint.calls == [
        ("accept_local",),
        ("advertise_ipc", 41, 0x2000, 4096),
    ]
    assert client_endpoint.calls == [
        ("connect_local", "0000:bd:00.0"),
        ("write_ipc", 106, 0x1000, 4096, b"ipc-info"),
    ]

    with pytest.raises(ValueError, match="destination rank"):
        client.advertise_write_ipc(descriptor, conn_id=106, dst_ptr=0x2000)
    with pytest.raises(ValueError, match="source rank"):
        server.write_ipc(descriptor, conn_id=41, src_ptr=0x1000, info_blob=info_blob)


def test_cuda_comm_launch_plan_acquires_registry_runtime():
    registry = CudaCommRuntimeRegistry()
    capability = create_cuda_comm_capability(backend="nccl", device_ids=(0, 1))
    plan = create_cuda_comm_launch_plan(capability, rank=0)
    fake_torch = _FakeTorch()
    fake_dist = _FakeDist()

    runtime = plan.acquire_runtime(
        registry,
        init_method="tcp://127.0.0.1:12345",
        torch_module=fake_torch,
        dist_module=fake_dist,
    )

    assert isinstance(runtime, TorchNcclCudaCommRuntime)
    assert runtime.rank == plan.rank
    assert runtime.device_id == plan.device_id
    assert runtime.runtime_id == plan.runtime_id
    assert registry.active_ids() == (plan.runtime_id,)

    registry.release(plan.runtime_id)


def test_nccl_runtime_forwards_collective_operations():
    capability = create_cuda_comm_capability(backend="nccl", device_ids=(0, 1))
    fake_dist = _FakeDist()
    runtime = TorchNcclCudaCommRuntime(
        capability,
        rank=0,
        init_method="tcp://127.0.0.1:12345",
        torch_module=_FakeTorch(),
        dist_module=fake_dist,
    )

    tensor = _FakeTensor("input")
    output = _FakeTensor("output")

    assert runtime.all_reduce(tensor) is tensor
    assert runtime.reduce_scatter(output, tensor) is output
    gathered = runtime.all_gather(tensor)
    runtime.send(tensor, dst=1)
    runtime.recv(output, src=1)

    assert len(gathered) == 2
    assert fake_dist.calls == [
        ("all_reduce", tensor, "sum"),
        ("reduce_scatter_tensor", output, tensor, "sum"),
        ("all_gather", tuple(gathered), tensor),
        ("send", tensor, 1),
        ("recv", output, 1),
    ]


def test_cuda_comm_runtime_registry_rejects_uccl_without_p2p_transport_selection():
    registry = CudaCommRuntimeRegistry()
    capability = create_cuda_comm_capability(backend="uccl", device_ids=(0, 1))

    with pytest.raises(ValueError, match="rank"):
        registry.acquire(capability, uccl_transport="p2p_ipc")

    with pytest.raises(NotImplementedError, match="p2p_ipc"):
        registry.acquire(capability, rank=0)


class _FakeCuda:
    def __init__(self) -> None:
        self.device_ids: list[int] = []

    def set_device(self, device_id: int) -> None:
        self.device_ids.append(device_id)


class _FakeTorch:
    def __init__(self) -> None:
        self.cuda = _FakeCuda()

    def empty_like(self, tensor):
        return _FakeTensor(f"empty_like:{tensor.name}")


class _FakeDist:
    class ReduceOp:
        SUM = "sum"

    def __init__(self) -> None:
        self.init_calls: list[dict] = []
        self.destroy_calls = 0
        self.calls: list[tuple] = []
        self._initialized = False

    def init_process_group(self, *, backend: str, init_method: str, rank: int, world_size: int) -> None:
        self.init_calls.append(
            {
                "backend": backend,
                "init_method": init_method,
                "rank": rank,
                "world_size": world_size,
            }
        )
        self._initialized = True

    def is_available(self) -> bool:
        return True

    def is_initialized(self) -> bool:
        return self._initialized

    def destroy_process_group(self) -> None:
        self.destroy_calls += 1
        self._initialized = False

    def all_reduce(self, tensor, *, op) -> None:
        self.calls.append(("all_reduce", tensor, op))

    def reduce_scatter_tensor(self, output, tensor, *, op) -> None:
        self.calls.append(("reduce_scatter_tensor", output, tensor, op))

    def all_gather(self, gathered, tensor) -> None:
        self.calls.append(("all_gather", tuple(gathered), tensor))

    def send(self, tensor, *, dst: int) -> None:
        self.calls.append(("send", tensor, dst))

    def recv(self, tensor, *, src: int) -> None:
        self.calls.append(("recv", tensor, src))


class _FakeTensor:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeUcclP2P:
    def __init__(self) -> None:
        self.endpoint_device_ids: list[int] = []
        self.endpoints: list[_FakeUcclEndpoint] = []

    def Endpoint(self, device_id: int):
        self.endpoint_device_ids.append(device_id)
        endpoint = _FakeUcclEndpoint(device_id)
        self.endpoints.append(endpoint)
        return endpoint


class _FakeUcclEndpoint:
    def __init__(self, local_gpu_idx: int) -> None:
        self.local_gpu_idx = local_gpu_idx
        self.calls: list[tuple] = []
        self.closed = False
        self.accept_local_result = (True, 6 if local_gpu_idx == 7 else 7, 106)

    def close(self) -> None:
        self.closed = True

    def get_metadata(self) -> bytes:
        self.calls.append(("get_metadata",))
        return f"metadata:{self.local_gpu_idx}".encode("ascii")

    def accept_local(self):
        self.calls.append(("accept_local",))
        return self.accept_local_result

    def connect_local(self, peer_target):
        self.calls.append(("connect_local", peer_target))
        return True, 106

    def advertise_ipc(self, conn_id: int, dst_ptr: int, nbytes: int):
        self.calls.append(("advertise_ipc", conn_id, dst_ptr, nbytes))
        return True, b"ipc-info"

    def write_ipc(self, conn_id: int, src_ptr: int, nbytes: int, info_blob: bytes):
        self.calls.append(("write_ipc", conn_id, src_ptr, nbytes, info_blob))
        return True
