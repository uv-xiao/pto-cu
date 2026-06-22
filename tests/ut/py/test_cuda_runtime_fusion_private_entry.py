import subprocess
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _compile_and_run(tmp_path: Path, source: str) -> str:
    source_path = tmp_path / "runtime_fusion_entry_test.cpp"
    binary_path = tmp_path / "runtime_fusion_entry_test"
    source_path.write_text(textwrap.dedent(source), encoding="utf-8")

    compile_result = subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-I",
            str(ROOT / "src" / "cuda" / "platform" / "include"),
            "-o",
            str(binary_path),
            str(source_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert compile_result.returncode == 0, compile_result.stdout

    run_result = subprocess.run(
        [str(binary_path)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert run_result.returncode == 0, run_result.stdout
    return run_result.stdout


def test_private_runtime_fusion_entry_reports_missing_runtime_surfaces(tmp_path):
    output = _compile_and_run(
        tmp_path,
        """
        #include "host/pto_cuda_runtime_fusion_abi.h"

        #include <cassert>
        #include <cstdint>
        #include <iostream>

        int main() {
            PtoCudaCommDeviceDescriptor descriptor = {
                PTO_CUDA_COMM_BACKEND_NCCL, 1U, 7U, 2U, 0x1234U
            };
            PtoCudaRuntimeFusionRequest request = {};
            request.version = PTO_CUDA_RUNTIME_FUSION_REQUEST_VERSION;
            request.callable_id = 42;
            request.chip_storage_task_args = reinterpret_cast<const void *>(0x10);
            request.chip_storage_task_args_size = 4096U;
            request.persistent_graph_descriptor = reinterpret_cast<const void *>(0x20);
            request.comm_descriptor = &descriptor;
            request.uccl_ep_capability_metadata = reinterpret_cast<const void *>(0x30);
            request.output_sink = reinterpret_cast<void *>(0x40);

            PtoCudaRuntimeFusionResult result = {};
            int rc = persistent_device_uccl_ep_runtime_fusion_entry(&request, &result);

            assert(rc == 0);
            assert(result.status == PTO_CUDA_RUNTIME_FUSION_STATUS_UNSUPPORTED);
            assert(result.actual_fused_cross_gpu_execution == 0U);
            assert(result.callable_id == 42);
            assert(result.rank == 1U);
            assert(result.device_id == 7U);
            assert(result.world_size == 2U);
            assert(
                (result.failure_fields & PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_COORDINATOR) != 0U
            );
            assert(
                (result.failure_fields & PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_DESCRIPTOR_ALLOCATOR) != 0U
            );
            assert(
                (result.failure_fields & PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_UCCL_EP_RUNTIME) != 0U
            );
            assert(
                (result.failure_fields & PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_VALIDATION_POLICY) != 0U
            );
            assert(
                (result.failure_fields & PTO_CUDA_RUNTIME_FUSION_FAILURE_UNSUPPORTED_BOUNDARY) != 0U
            );
            std::cout << pto_cuda_runtime_fusion_status_name(result.status) << "\\n";
            std::cout << result.reason << "\\n";
            return 0;
        }
        """,
    )

    assert "unsupported" in output
    assert "persistent_device_uccl_ep_runtime_fusion" in output


def test_private_runtime_fusion_entry_rejects_forbidden_pass_evidence(tmp_path):
    output = _compile_and_run(
        tmp_path,
        """
        #include "host/pto_cuda_runtime_fusion_abi.h"

        #include <cassert>
        #include <iostream>

        int main() {
            PtoCudaCommDeviceDescriptor descriptor = {
                PTO_CUDA_COMM_BACKEND_NCCL, 0U, 6U, 2U, 0x4567U
            };
            PtoCudaRuntimeFusionRequest request = {};
            request.version = PTO_CUDA_RUNTIME_FUSION_REQUEST_VERSION;
            request.callable_id = 5;
            request.chip_storage_task_args = reinterpret_cast<const void *>(0x10);
            request.chip_storage_task_args_size = 4096U;
            request.persistent_graph_descriptor = reinterpret_cast<const void *>(0x20);
            request.comm_descriptor = &descriptor;
            request.uccl_ep_capability_metadata = reinterpret_cast<const void *>(0x30);
            request.coordinator = reinterpret_cast<const void *>(0x40);
            request.descriptor_allocator = reinterpret_cast<const void *>(0x50);
            request.uccl_ep_runtime = reinterpret_cast<const void *>(0x60);
            request.validation_policy = reinterpret_cast<const void *>(0x70);
            request.output_sink = reinterpret_cast<void *>(0x80);
            request.pass_evidence_source =
                PTO_CUDA_RUNTIME_FUSION_EVIDENCE_ADAPTER_PROVENANCE;

            PtoCudaRuntimeFusionResult result = {};
            int rc = persistent_device_uccl_ep_runtime_fusion_entry(&request, &result);

            assert(rc == 0);
            assert(result.status == PTO_CUDA_RUNTIME_FUSION_STATUS_FAILED);
            assert(result.actual_fused_cross_gpu_execution == 0U);
            assert(
                (result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_FABRICATED_OR_UNTRUSTED_PASS_EVIDENCE) != 0U
            );
            std::cout << pto_cuda_runtime_fusion_status_name(result.status) << "\\n";
            std::cout << pto_cuda_runtime_fusion_failure_name(
                PTO_CUDA_RUNTIME_FUSION_FAILURE_FABRICATED_OR_UNTRUSTED_PASS_EVIDENCE
            ) << "\\n";
            return 0;
        }
        """,
    )

    assert "failed" in output
    assert "fabricated_or_untrusted_pass_evidence" in output


def test_private_runtime_fusion_entry_keeps_pass_unreachable_without_evidence(tmp_path):
    _compile_and_run(
        tmp_path,
        """
        #include "host/pto_cuda_runtime_fusion_abi.h"

        #include <cassert>

        int main() {
            PtoCudaCommDeviceDescriptor descriptor = {
                PTO_CUDA_COMM_BACKEND_NCCL, 0U, 6U, 2U, 0x789AU
            };
            PtoCudaRuntimeFusionRequest request = {};
            request.version = PTO_CUDA_RUNTIME_FUSION_REQUEST_VERSION;
            request.callable_id = 9;
            request.chip_storage_task_args = reinterpret_cast<const void *>(0x10);
            request.chip_storage_task_args_size = 4096U;
            request.persistent_graph_descriptor = reinterpret_cast<const void *>(0x20);
            request.comm_descriptor = &descriptor;
            request.uccl_ep_capability_metadata = reinterpret_cast<const void *>(0x30);
            request.coordinator = reinterpret_cast<const void *>(0x40);
            request.descriptor_allocator = reinterpret_cast<const void *>(0x50);
            request.uccl_ep_runtime = reinterpret_cast<const void *>(0x60);
            request.validation_policy = reinterpret_cast<const void *>(0x70);
            request.output_sink = reinterpret_cast<void *>(0x80);

            PtoCudaRuntimeFusionResult result = {};
            int rc = persistent_device_uccl_ep_runtime_fusion_entry(&request, &result);

            assert(rc == 0);
            assert(result.status == PTO_CUDA_RUNTIME_FUSION_STATUS_UNSUPPORTED);
            assert(result.status != PTO_CUDA_RUNTIME_FUSION_STATUS_PASSED);
            assert(result.actual_fused_cross_gpu_execution == 0U);
            assert(
                (result.failure_fields & PTO_CUDA_RUNTIME_FUSION_FAILURE_UNSUPPORTED_BOUNDARY) != 0U
            );
            return 0;
        }
        """,
    )


def test_cuda_host_runtime_hooks_private_entry_without_public_api_expansion():
    host_runtime = (
        ROOT / "src" / "cuda" / "platform" / "onboard" / "host" / "pto_runtime_c_api.cpp"
    ).read_text(encoding="utf-8")
    common_abi = (ROOT / "src" / "common" / "worker" / "pto_runtime_c_api.h").read_text(
        encoding="utf-8"
    )

    assert "persistent_device_uccl_ep_runtime_fusion_entry" in host_runtime
    assert "PtoCudaRuntimeFusionRequest" in host_runtime
    assert "PtoCudaRuntimeFusionResult" in host_runtime
    assert "persistent_device_uccl_ep_runtime_fusion_entry" not in common_abi
    assert "PtoCudaRuntimeFusionRequest" not in common_abi
