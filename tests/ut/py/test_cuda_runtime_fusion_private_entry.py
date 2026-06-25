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
            "-I",
            str(ROOT / "src" / "common"),
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
            ChipStorageTaskArgs chip_storage = {};
            PtoCudaCommDeviceDescriptor descriptor = {
                PTO_CUDA_COMM_BACKEND_NCCL, 1U, 7U, 2U, 0x1234U
            };
            PtoCudaRuntimeFusionRequest request = {};
            request.version = PTO_CUDA_RUNTIME_FUSION_REQUEST_VERSION;
            request.callable_id = 42;
            request.chip_storage_task_args = &chip_storage;
            request.chip_storage_task_args_size = sizeof(chip_storage);
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
            ChipStorageTaskArgs chip_storage = {};
            PtoCudaCommDeviceDescriptor descriptor = {
                PTO_CUDA_COMM_BACKEND_NCCL, 0U, 6U, 2U, 0x4567U
            };
            PtoCudaRuntimeFusionRequest request = {};
            request.version = PTO_CUDA_RUNTIME_FUSION_REQUEST_VERSION;
            request.callable_id = 5;
            request.chip_storage_task_args = &chip_storage;
            request.chip_storage_task_args_size = sizeof(chip_storage);
            request.persistent_graph_descriptor = reinterpret_cast<const void *>(0x20);
            request.comm_descriptor = &descriptor;
            request.uccl_ep_capability_metadata = reinterpret_cast<const void *>(0x30);
            request.coordinator = reinterpret_cast<const void *>(0x40);
            request.descriptor_allocator = reinterpret_cast<const void *>(0x50);
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
            ChipStorageTaskArgs chip_storage = {};
            PtoCudaCommDeviceDescriptor descriptor = {
                PTO_CUDA_COMM_BACKEND_NCCL, 0U, 6U, 2U, 0x789AU
            };
            PtoCudaRuntimeFusionRequest request = {};
            request.version = PTO_CUDA_RUNTIME_FUSION_REQUEST_VERSION;
            request.callable_id = 9;
            request.chip_storage_task_args = &chip_storage;
            request.chip_storage_task_args_size = sizeof(chip_storage);
            request.persistent_graph_descriptor = reinterpret_cast<const void *>(0x20);
            request.comm_descriptor = &descriptor;
            request.uccl_ep_capability_metadata = reinterpret_cast<const void *>(0x30);
            request.coordinator = reinterpret_cast<const void *>(0x40);
            request.descriptor_allocator = reinterpret_cast<const void *>(0x50);
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


def test_private_runtime_fusion_entry_accepts_private_runtime_path_scaffold_but_stays_unsupported(
    tmp_path,
):
    _compile_and_run(
        tmp_path,
        """
        #include "host/pto_cuda_runtime_fusion_abi.h"

        #include <cassert>

        int main() {
            ChipStorageTaskArgs chip_storage = {};
            PtoCudaCommDeviceDescriptor descriptor = {
                PTO_CUDA_COMM_BACKEND_NCCL, 0U, 6U, 2U, 0xA11CEU
            };
            PtoCudaUcclEpRuntimeDescriptorView dispatch_view = {};
            dispatch_view.version = PTO_CUDA_UCCL_EP_RUNTIME_DESCRIPTOR_VIEW_VERSION;
            dispatch_view.invocation_id = 77U;
            dispatch_view.persistent_graph_descriptor = reinterpret_cast<const void *>(0x20);
            dispatch_view.capability_crc32 = descriptor.capability_crc32;
            dispatch_view.rank = descriptor.rank;
            dispatch_view.device_id = descriptor.device_id;
            dispatch_view.world_size = descriptor.world_size;
            dispatch_view.descriptor_vocabulary =
                PTO_CUDA_RUNTIME_FUSION_DESCRIPTOR_VOCABULARY_DISPATCH;
            dispatch_view.shared_token = 0xCAFEU;
            dispatch_view.source =
                PTO_CUDA_UCCL_EP_RUNTIME_PATH_SOURCE_COORDINATOR_OWNED;

            PtoCudaUcclEpRuntimeDescriptorView combine_view = dispatch_view;
            combine_view.descriptor_vocabulary =
                PTO_CUDA_RUNTIME_FUSION_DESCRIPTOR_VOCABULARY_COMBINE;

            PtoCudaUcclEpRuntimePath runtime_path = {};
            runtime_path.version = PTO_CUDA_UCCL_EP_RUNTIME_PATH_VERSION;
            runtime_path.transport_mode = PTO_CUDA_UCCL_EP_TRANSPORT_MODE_EP;
            runtime_path.dispatch_descriptor = &dispatch_view;
            runtime_path.combine_descriptor = &combine_view;

            PtoCudaRuntimeFusionRequest request = {};
            request.version = PTO_CUDA_RUNTIME_FUSION_REQUEST_VERSION;
            request.callable_id = 9;
            request.invocation_id = 77U;
            request.chip_storage_task_args = &chip_storage;
            request.chip_storage_task_args_size = sizeof(chip_storage);
            request.persistent_graph_descriptor = reinterpret_cast<const void *>(0x20);
            request.comm_descriptor = &descriptor;
            request.uccl_ep_capability_metadata = reinterpret_cast<const void *>(0x30);
            request.uccl_ep_runtime = &runtime_path;
            request.validation_policy = reinterpret_cast<const void *>(0x40);
            request.output_sink = reinterpret_cast<void *>(0x50);

            PtoCudaRuntimeFusionResult result = {};
            int rc = persistent_device_uccl_ep_runtime_fusion_entry(&request, &result);

            assert(rc == 0);
            assert(result.status == PTO_CUDA_RUNTIME_FUSION_STATUS_UNSUPPORTED);
            assert(result.status != PTO_CUDA_RUNTIME_FUSION_STATUS_PASSED);
            assert(result.actual_fused_cross_gpu_execution == 0U);
            assert(
                (result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_COORDINATOR) != 0U
            );
            assert(
                (result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_DESCRIPTOR_ALLOCATOR) != 0U
            );
            assert(
                (result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_UCCL_EP_RUNTIME) == 0U
            );
            assert(
                (result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_DESCRIPTOR_TOKEN_MISMATCH) == 0U
            );
            return 0;
        }
        """,
    )


def test_private_runtime_fusion_entry_rejects_public_runtime_path_fields(tmp_path):
    output = _compile_and_run(
        tmp_path,
        """
        #include "host/pto_cuda_runtime_fusion_abi.h"

        #include <cassert>
        #include <iostream>

        int main() {
            ChipStorageTaskArgs chip_storage = {};
            PtoCudaCommDeviceDescriptor descriptor = {
                PTO_CUDA_COMM_BACKEND_NCCL, 0U, 6U, 2U, 0xA11CEU
            };
            PtoCudaUcclEpRuntimeDescriptorView dispatch_view = {};
            dispatch_view.version = PTO_CUDA_UCCL_EP_RUNTIME_DESCRIPTOR_VIEW_VERSION;
            dispatch_view.invocation_id = 11U;
            dispatch_view.persistent_graph_descriptor = reinterpret_cast<const void *>(0x20);
            dispatch_view.capability_crc32 = descriptor.capability_crc32;
            dispatch_view.rank = descriptor.rank;
            dispatch_view.device_id = descriptor.device_id;
            dispatch_view.world_size = descriptor.world_size;
            dispatch_view.descriptor_vocabulary =
                PTO_CUDA_RUNTIME_FUSION_DESCRIPTOR_VOCABULARY_DISPATCH;
            dispatch_view.shared_token = 0xBEEFU;
            dispatch_view.source = PTO_CUDA_UCCL_EP_RUNTIME_PATH_SOURCE_PUBLIC_API;

            PtoCudaUcclEpRuntimeDescriptorView combine_view = dispatch_view;
            combine_view.descriptor_vocabulary =
                PTO_CUDA_RUNTIME_FUSION_DESCRIPTOR_VOCABULARY_COMBINE;

            PtoCudaUcclEpRuntimePath runtime_path = {};
            runtime_path.version = PTO_CUDA_UCCL_EP_RUNTIME_PATH_VERSION;
            runtime_path.transport_mode = PTO_CUDA_UCCL_EP_TRANSPORT_MODE_EP;
            runtime_path.dispatch_descriptor = &dispatch_view;
            runtime_path.combine_descriptor = &combine_view;

            PtoCudaRuntimeFusionRequest request = {};
            request.version = PTO_CUDA_RUNTIME_FUSION_REQUEST_VERSION;
            request.invocation_id = 11U;
            request.chip_storage_task_args = &chip_storage;
            request.chip_storage_task_args_size = sizeof(chip_storage);
            request.persistent_graph_descriptor = reinterpret_cast<const void *>(0x20);
            request.comm_descriptor = &descriptor;
            request.uccl_ep_capability_metadata = reinterpret_cast<const void *>(0x30);
            request.coordinator = reinterpret_cast<const void *>(0x40);
            request.descriptor_allocator = reinterpret_cast<const void *>(0x50);
            request.uccl_ep_runtime = &runtime_path;
            request.validation_policy = reinterpret_cast<const void *>(0x60);
            request.output_sink = reinterpret_cast<void *>(0x70);

            PtoCudaRuntimeFusionResult result = {};
            int rc = persistent_device_uccl_ep_runtime_fusion_entry(&request, &result);

            assert(rc == 0);
            assert(result.status == PTO_CUDA_RUNTIME_FUSION_STATUS_FAILED);
            assert(result.actual_fused_cross_gpu_execution == 0U);
            assert(
                (result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_PUBLIC_API_RUNTIME_PATH) != 0U
            );
            assert(
                (result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_FABRICATED_OR_UNTRUSTED_PASS_EVIDENCE) != 0U
            );
            std::cout << pto_cuda_runtime_fusion_failure_name(
                PTO_CUDA_RUNTIME_FUSION_FAILURE_PUBLIC_API_RUNTIME_PATH
            ) << "\\n";
            return 0;
        }
        """,
    )

    assert "public_api_runtime_path" in output


def test_private_runtime_fusion_entry_fails_runtime_path_validation_mismatches(
    tmp_path,
):
    output = _compile_and_run(
        tmp_path,
        """
        #include "host/pto_cuda_runtime_fusion_abi.h"

        #include <cassert>
        #include <iostream>

        int main() {
            ChipStorageTaskArgs chip_storage = {};
            PtoCudaCommDeviceDescriptor descriptor = {
                PTO_CUDA_COMM_BACKEND_NCCL, 0U, 6U, 2U, 0xA11CEU
            };
            PtoCudaUcclEpRuntimeDescriptorView dispatch_view = {};
            dispatch_view.version = PTO_CUDA_UCCL_EP_RUNTIME_DESCRIPTOR_VIEW_VERSION - 1U;
            dispatch_view.invocation_id = 12U;
            dispatch_view.persistent_graph_descriptor = reinterpret_cast<const void *>(0x99);
            dispatch_view.capability_crc32 = descriptor.capability_crc32 + 1U;
            dispatch_view.rank = 1U;
            dispatch_view.device_id = descriptor.device_id + 1U;
            dispatch_view.world_size = descriptor.world_size;
            dispatch_view.descriptor_vocabulary =
                PTO_CUDA_RUNTIME_FUSION_DESCRIPTOR_VOCABULARY_COMBINE;
            dispatch_view.shared_token = 0x111U;
            dispatch_view.source =
                PTO_CUDA_UCCL_EP_RUNTIME_PATH_SOURCE_COORDINATOR_OWNED;

            PtoCudaUcclEpRuntimeDescriptorView combine_view = dispatch_view;
            combine_view.version = PTO_CUDA_UCCL_EP_RUNTIME_DESCRIPTOR_VIEW_VERSION;
            combine_view.invocation_id = 13U;
            combine_view.persistent_graph_descriptor = reinterpret_cast<const void *>(0x20);
            combine_view.capability_crc32 = descriptor.capability_crc32;
            combine_view.rank = descriptor.rank;
            combine_view.device_id = descriptor.device_id;
            combine_view.descriptor_vocabulary =
                PTO_CUDA_RUNTIME_FUSION_DESCRIPTOR_VOCABULARY_DISPATCH;
            combine_view.shared_token = 0x222U;

            PtoCudaUcclEpRuntimePath runtime_path = {};
            runtime_path.version = PTO_CUDA_UCCL_EP_RUNTIME_PATH_VERSION;
            runtime_path.transport_mode = PTO_CUDA_UCCL_EP_TRANSPORT_MODE_UNKNOWN;
            runtime_path.dispatch_descriptor = &dispatch_view;
            runtime_path.combine_descriptor = &combine_view;

            PtoCudaRuntimeFusionRequest request = {};
            request.version = PTO_CUDA_RUNTIME_FUSION_REQUEST_VERSION;
            request.invocation_id = 12U;
            request.chip_storage_task_args = &chip_storage;
            request.chip_storage_task_args_size = sizeof(chip_storage);
            request.persistent_graph_descriptor = reinterpret_cast<const void *>(0x20);
            request.comm_descriptor = &descriptor;
            request.uccl_ep_capability_metadata = reinterpret_cast<const void *>(0x30);
            request.coordinator = reinterpret_cast<const void *>(0x40);
            request.descriptor_allocator = reinterpret_cast<const void *>(0x50);
            request.uccl_ep_runtime = &runtime_path;
            request.validation_policy = reinterpret_cast<const void *>(0x60);
            request.output_sink = reinterpret_cast<void *>(0x70);

            PtoCudaRuntimeFusionResult result = {};
            int rc = persistent_device_uccl_ep_runtime_fusion_entry(&request, &result);

            assert(rc == 0);
            assert(result.status == PTO_CUDA_RUNTIME_FUSION_STATUS_FAILED);
            assert(
                (result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_STALE_DESCRIPTOR_VIEW) != 0U
            );
            assert(
                (result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_DESCRIPTOR_TOKEN_MISMATCH) != 0U
            );
            assert(
                (result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_RANK_DEVICE_MISMATCH) != 0U
            );
            assert(
                (result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_TRANSPORT_MODE_MISMATCH) != 0U
            );
            assert(
                (result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_DESCRIPTOR_VOCABULARY_MISMATCH) != 0U
            );
            std::cout << result.reason << "\\n";
            return 0;
        }
        """,
    )

    assert "runtime path validation failed" in output


def test_private_descriptor_allocation_builds_runtime_path_but_stays_unsupported(
    tmp_path,
):
    _compile_and_run(
        tmp_path,
        """
        #include "host/pto_cuda_runtime_fusion_abi.h"

        #include <cassert>
        #include <cstddef>
        #include <cstdint>

        int main() {
            ChipStorageTaskArgs chip_storage = {};
            PtoCudaCommDeviceDescriptor descriptor = {
                PTO_CUDA_COMM_BACKEND_NCCL, 1U, 7U, 2U, 0xD15C0U
            };

            PtoCudaRuntimeFusionRequest request = {};
            request.version = PTO_CUDA_RUNTIME_FUSION_REQUEST_VERSION;
            request.callable_id = 19;
            request.invocation_id = 31337U;
            request.chip_storage_task_args = &chip_storage;
            request.chip_storage_task_args_size = sizeof(chip_storage);
            request.persistent_graph_descriptor = reinterpret_cast<const void *>(0x20);
            request.comm_descriptor = &descriptor;
            request.uccl_ep_capability_metadata = reinterpret_cast<const void *>(0x30);
            request.validation_policy = reinterpret_cast<const void *>(0x40);
            request.output_sink = reinterpret_cast<void *>(0x50);

            alignas(PtoCudaUcclEpDeviceDescriptorBuffer) unsigned char device_storage[
                sizeof(PtoCudaUcclEpDeviceDescriptorBuffer)
            ] = {};
            PtoCudaUcclEpDescriptorAllocation allocation = {};

            int allocation_rc = pto_cuda_runtime_fusion_allocate_uccl_ep_descriptors(
                &request,
                &allocation,
                device_storage,
                sizeof(device_storage),
                0x5A5A5A5AU
            );

            assert(allocation_rc == 0);
            assert(allocation.host_control.version ==
                   PTO_CUDA_UCCL_EP_DESCRIPTOR_ALLOCATION_VERSION);
            assert(allocation.host_control.invocation_id == request.invocation_id);
            assert(allocation.host_control.persistent_graph_descriptor ==
                   request.persistent_graph_descriptor);
            assert(allocation.host_control.shared_token == 0x5A5A5A5AU);
            assert(allocation.host_control.runtime_owned == 1U);
            assert(allocation.host_control.dispatch_descriptor_offset == 0U);
            assert(allocation.host_control.combine_descriptor_offset >
                   allocation.host_control.dispatch_descriptor_offset);
            assert(allocation.device_buffer ==
                   reinterpret_cast<PtoCudaUcclEpDeviceDescriptorBuffer *>(
                       device_storage
                   ));
            assert(allocation.runtime_path.dispatch_descriptor ==
                   &allocation.dispatch_descriptor);
            assert(allocation.runtime_path.combine_descriptor ==
                   &allocation.combine_descriptor);
            assert(allocation.dispatch_descriptor.invocation_id ==
                   request.invocation_id);
            assert(allocation.combine_descriptor.invocation_id ==
                   request.invocation_id);
            assert(allocation.dispatch_descriptor.shared_token ==
                   allocation.combine_descriptor.shared_token);
            assert(allocation.dispatch_descriptor.descriptor_vocabulary ==
                   PTO_CUDA_RUNTIME_FUSION_DESCRIPTOR_VOCABULARY_DISPATCH);
            assert(allocation.combine_descriptor.descriptor_vocabulary ==
                   PTO_CUDA_RUNTIME_FUSION_DESCRIPTOR_VOCABULARY_COMBINE);

            request.descriptor_allocator = &allocation;
            request.uccl_ep_runtime = &allocation.runtime_path;

            PtoCudaRuntimeFusionResult result = {};
            int rc = persistent_device_uccl_ep_runtime_fusion_entry(&request, &result);

            assert(rc == 0);
            assert(result.status == PTO_CUDA_RUNTIME_FUSION_STATUS_UNSUPPORTED);
            assert(result.actual_fused_cross_gpu_execution == 0U);
            assert(
                (result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_DESCRIPTOR_ALLOCATOR) == 0U
            );
            assert(
                (result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_UCCL_EP_RUNTIME) == 0U
            );
            assert(
                (result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_COORDINATOR) != 0U
            );
            assert(
                (result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_UNSUPPORTED_BOUNDARY) != 0U
            );
            return 0;
        }
        """,
    )


def test_private_coordinator_scaffold_owns_runtime_path_for_one_invocation(
    tmp_path,
):
    _compile_and_run(
        tmp_path,
        """
        #include "host/pto_cuda_runtime_fusion_abi.h"

        #include <cassert>
        #include <cstdint>

        int main() {
            ChipStorageTaskArgs chip_storage = {};
            PtoCudaCommDeviceDescriptor descriptor = {
                PTO_CUDA_COMM_BACKEND_NCCL, 1U, 7U, 2U, 0xC001D0U
            };

            PtoCudaRuntimeFusionRequest request = {};
            request.version = PTO_CUDA_RUNTIME_FUSION_REQUEST_VERSION;
            request.callable_id = 23;
            request.invocation_id = 4242U;
            request.chip_storage_task_args = &chip_storage;
            request.chip_storage_task_args_size = sizeof(chip_storage);
            request.persistent_graph_descriptor = reinterpret_cast<const void *>(0x20);
            request.comm_descriptor = &descriptor;
            request.uccl_ep_capability_metadata = reinterpret_cast<const void *>(0x30);
            request.validation_policy = reinterpret_cast<const void *>(0x40);

            PtoCudaRuntimeFusionResult output_sink = {};
            PtoCudaUcclEpDeviceDescriptorBuffer device_storage = {};
            PtoCudaRuntimeFusionCoordinator coordinator = {};

            int coordinator_rc = pto_cuda_runtime_fusion_prepare_private_coordinator(
                &request,
                &coordinator,
                &device_storage,
                sizeof(device_storage),
                &output_sink
            );

            assert(coordinator_rc == 0);
            assert(coordinator.version == PTO_CUDA_RUNTIME_FUSION_COORDINATOR_VERSION);
            assert(coordinator.invocation_id == request.invocation_id);
            assert(coordinator.output_sink == &output_sink);
            assert(
                coordinator.descriptor_allocation.runtime_path.dispatch_descriptor ==
                &coordinator.descriptor_allocation.dispatch_descriptor
            );

            request.coordinator = &coordinator;
            request.descriptor_allocator = &coordinator.descriptor_allocation;
            request.uccl_ep_runtime = &coordinator.descriptor_allocation.runtime_path;
            request.output_sink = coordinator.output_sink;

            PtoCudaRuntimeFusionResult result = {};
            int rc = persistent_device_uccl_ep_runtime_fusion_entry(&request, &result);

            assert(rc == 0);
            assert(result.status == PTO_CUDA_RUNTIME_FUSION_STATUS_UNSUPPORTED);
            assert(result.status != PTO_CUDA_RUNTIME_FUSION_STATUS_PASSED);
            assert(result.actual_fused_cross_gpu_execution == 0U);
            assert(output_sink.status == result.status);
            assert(output_sink.failure_fields == result.failure_fields);
            assert(
                (result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_COORDINATOR) == 0U
            );
            assert(
                (result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_DESCRIPTOR_ALLOCATOR) == 0U
            );
            assert(
                (result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_UCCL_EP_RUNTIME) == 0U
            );
            assert(
                (result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_UNSUPPORTED_BOUNDARY) != 0U
            );
            return 0;
        }
        """,
    )


def test_private_runtime_dispatch_scaffold_status_gate_is_coordinator_owned(
    tmp_path,
):
    _compile_and_run(
        tmp_path,
        """
        #include "host/pto_cuda_runtime_fusion_abi.h"

        #include <cassert>
        #include <cstdint>

        int main() {
            ChipStorageTaskArgs chip_storage = {};
            PtoCudaCommDeviceDescriptor descriptor = {
                PTO_CUDA_COMM_BACKEND_NCCL, 1U, 7U, 2U, 0xD15A7CU
            };

            PtoCudaRuntimeFusionRequest request = {};
            request.version = PTO_CUDA_RUNTIME_FUSION_REQUEST_VERSION;
            request.callable_id = 29;
            request.invocation_id = 5150U;
            request.chip_storage_task_args = &chip_storage;
            request.chip_storage_task_args_size = sizeof(chip_storage);
            request.persistent_graph_descriptor = reinterpret_cast<const void *>(0x20);
            request.comm_descriptor = &descriptor;
            request.uccl_ep_capability_metadata = reinterpret_cast<const void *>(0x30);
            request.validation_policy = reinterpret_cast<const void *>(0x40);

            PtoCudaRuntimeFusionResult output_sink = {};
            PtoCudaUcclEpDeviceDescriptorBuffer device_storage = {};
            PtoCudaRuntimeFusionCoordinator coordinator = {};

            int coordinator_rc = pto_cuda_runtime_fusion_prepare_private_coordinator(
                &request,
                &coordinator,
                &device_storage,
                sizeof(device_storage),
                &output_sink
            );
            assert(coordinator_rc == 0);
            assert(
                coordinator.runtime_dispatch_scaffold_status.version ==
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_SCAFFOLD_STATUS_VERSION
            );

            request.coordinator = &coordinator;
            request.descriptor_allocator = &coordinator.descriptor_allocation;
            request.uccl_ep_runtime = &coordinator.descriptor_allocation.runtime_path;
            request.output_sink = coordinator.output_sink;

            PtoCudaUcclEpRuntimeDispatchScaffoldStatus saved_gate =
                coordinator.runtime_dispatch_scaffold_status;
            coordinator.runtime_dispatch_scaffold_status = {};

            PtoCudaRuntimeFusionResult missing_gate_result = {};
            int missing_gate_rc =
                persistent_device_uccl_ep_runtime_fusion_entry(
                    &request, &missing_gate_result
                );

            assert(missing_gate_rc == 0);
            assert(missing_gate_result.status == PTO_CUDA_RUNTIME_FUSION_STATUS_FAILED);
            assert(missing_gate_result.actual_fused_cross_gpu_execution == 0U);
            assert(
                (missing_gate_result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_RUNTIME_DISPATCH_SCAFFOLD) != 0U
            );
            assert(output_sink.status == missing_gate_result.status);
            assert(output_sink.failure_fields == missing_gate_result.failure_fields);

            coordinator.runtime_dispatch_scaffold_status = saved_gate;
            int gate_rc =
                pto_cuda_runtime_fusion_prepare_runtime_dispatch_scaffold_status(
                    &request, &coordinator
                );
            assert(gate_rc == 0);
            assert(coordinator.runtime_dispatch_scaffold_status.dispatch_eligible == 1U);
            assert(
                coordinator.runtime_dispatch_scaffold_status.status ==
                PTO_CUDA_RUNTIME_FUSION_STATUS_UNSUPPORTED
            );

            PtoCudaRuntimeFusionResult result = {};
            int rc = persistent_device_uccl_ep_runtime_fusion_entry(&request, &result);

            assert(rc == 0);
            assert(result.status == PTO_CUDA_RUNTIME_FUSION_STATUS_UNSUPPORTED);
            assert(result.status != PTO_CUDA_RUNTIME_FUSION_STATUS_PASSED);
            assert(result.actual_fused_cross_gpu_execution == 0U);
            assert(
                (result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_RUNTIME_DISPATCH_SCAFFOLD) == 0U
            );
            assert(
                (result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_UNSUPPORTED_BOUNDARY) != 0U
            );
            assert(output_sink.status == result.status);
            assert(output_sink.failure_fields == result.failure_fields);
            return 0;
        }
        """,
    )


def test_private_runtime_dispatch_request_handoff_scaffold_status_is_coordinator_owned(
    tmp_path,
):
    _compile_and_run(
        tmp_path,
        """
        #include "host/pto_cuda_runtime_fusion_abi.h"

        #include <cassert>
        #include <cstdint>

        int main() {
            ChipStorageTaskArgs chip_storage = {};
            PtoCudaCommDeviceDescriptor descriptor = {
                PTO_CUDA_COMM_BACKEND_NCCL, 1U, 7U, 2U, 0xD15A7DU
            };

            PtoCudaRuntimeFusionRequest request = {};
            request.version = PTO_CUDA_RUNTIME_FUSION_REQUEST_VERSION;
            request.callable_id = 31;
            request.invocation_id = 5151U;
            request.chip_storage_task_args = &chip_storage;
            request.chip_storage_task_args_size = sizeof(chip_storage);
            request.persistent_graph_descriptor = reinterpret_cast<const void *>(0x20);
            request.comm_descriptor = &descriptor;
            request.uccl_ep_capability_metadata = reinterpret_cast<const void *>(0x30);
            request.validation_policy = reinterpret_cast<const void *>(0x40);

            PtoCudaRuntimeFusionResult output_sink = {};
            PtoCudaUcclEpDeviceDescriptorBuffer device_storage = {};
            PtoCudaRuntimeFusionCoordinator coordinator = {};

            int coordinator_rc = pto_cuda_runtime_fusion_prepare_private_coordinator(
                &request,
                &coordinator,
                &device_storage,
                sizeof(device_storage),
                &output_sink
            );
            assert(coordinator_rc == 0);
            assert(
                coordinator.runtime_dispatch_request_handoff_scaffold_status.version ==
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_REQUEST_HANDOFF_SCAFFOLD_STATUS_VERSION
            );

            request.coordinator = &coordinator;
            request.descriptor_allocator = &coordinator.descriptor_allocation;
            request.uccl_ep_runtime = &coordinator.descriptor_allocation.runtime_path;
            request.output_sink = coordinator.output_sink;

            PtoCudaUcclEpRuntimeDispatchRequestHandoffScaffoldStatus saved_handoff =
                coordinator.runtime_dispatch_request_handoff_scaffold_status;
            coordinator.runtime_dispatch_request_handoff_scaffold_status = {};

            PtoCudaRuntimeFusionResult missing_driver_result = {};
            int missing_driver_rc =
                persistent_device_uccl_ep_runtime_fusion_entry(
                    &request, &missing_driver_result
                );

            assert(missing_driver_rc == 0);
            assert(missing_driver_result.status == PTO_CUDA_RUNTIME_FUSION_STATUS_FAILED);
            assert(missing_driver_result.actual_fused_cross_gpu_execution == 0U);
            assert(
                (missing_driver_result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_RUNTIME_DISPATCH_HANDOFF_DRIVER) != 0U
            );
            assert(output_sink.status == missing_driver_result.status);
            assert(output_sink.failure_fields == missing_driver_result.failure_fields);

            coordinator.runtime_dispatch_request_handoff_scaffold_status =
                saved_handoff;
            int handoff_rc =
                pto_cuda_runtime_fusion_prepare_runtime_dispatch_request_handoff_scaffold_status(
                    &request, &coordinator
                );
            assert(handoff_rc == 0);
            assert(
                coordinator.runtime_dispatch_request_handoff_scaffold_status.
                    request_owner == &coordinator
            );
            assert(
                coordinator.runtime_dispatch_request_handoff_scaffold_status.
                    driver_state == &coordinator.runtime_dispatch_request_handoff_driver_state
            );
            assert(
                coordinator.runtime_dispatch_request_handoff_scaffold_status.status ==
                PTO_CUDA_RUNTIME_FUSION_STATUS_UNSUPPORTED
            );

            PtoCudaRuntimeFusionResult result = {};
            int rc = persistent_device_uccl_ep_runtime_fusion_entry(&request, &result);

            assert(rc == 0);
            assert(result.status == PTO_CUDA_RUNTIME_FUSION_STATUS_UNSUPPORTED);
            assert(result.status != PTO_CUDA_RUNTIME_FUSION_STATUS_PASSED);
            assert(result.actual_fused_cross_gpu_execution == 0U);
            assert(
                (result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_RUNTIME_DISPATCH_HANDOFF_DRIVER) == 0U
            );
            assert(
                (result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_UNSUPPORTED_BOUNDARY) != 0U
            );
            assert(output_sink.status == result.status);
            assert(output_sink.failure_fields == result.failure_fields);
            return 0;
        }
        """,
    )


def test_private_runtime_dispatch_driver_scaffold_status_is_driver_owned(
    tmp_path,
):
    output = _compile_and_run(
        tmp_path,
        """
        #include "host/pto_cuda_runtime_fusion_abi.h"

        #include <cassert>
        #include <cstdint>
        #include <iostream>

        int main() {
            ChipStorageTaskArgs chip_storage = {};
            PtoCudaCommDeviceDescriptor descriptor = {
                PTO_CUDA_COMM_BACKEND_NCCL, 1U, 7U, 2U, 0xD21E7U
            };

            PtoCudaRuntimeFusionRequest request = {};
            request.version = PTO_CUDA_RUNTIME_FUSION_REQUEST_VERSION;
            request.callable_id = 37;
            request.invocation_id = 6161U;
            request.chip_storage_task_args = &chip_storage;
            request.chip_storage_task_args_size = sizeof(chip_storage);
            request.persistent_graph_descriptor = reinterpret_cast<const void *>(0x20);
            request.comm_descriptor = &descriptor;
            request.uccl_ep_capability_metadata = reinterpret_cast<const void *>(0x30);
            request.validation_policy = reinterpret_cast<const void *>(0x40);

            PtoCudaRuntimeFusionResult output_sink = {};
            PtoCudaUcclEpDeviceDescriptorBuffer device_storage = {};
            PtoCudaRuntimeFusionCoordinator coordinator = {};

            int coordinator_rc = pto_cuda_runtime_fusion_prepare_private_coordinator(
                &request,
                &coordinator,
                &device_storage,
                sizeof(device_storage),
                &output_sink
            );
            assert(coordinator_rc == 0);
            assert(
                coordinator.runtime_dispatch_driver_scaffold_status.version ==
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_SCAFFOLD_STATUS_VERSION
            );

            request.coordinator = &coordinator;
            request.descriptor_allocator = &coordinator.descriptor_allocation;
            request.uccl_ep_runtime = &coordinator.descriptor_allocation.runtime_path;
            request.output_sink = coordinator.output_sink;

            PtoCudaUcclEpRuntimeDispatchDriverScaffoldStatus saved_driver =
                coordinator.runtime_dispatch_driver_scaffold_status;

            coordinator.runtime_dispatch_driver_scaffold_status.invocation_id =
                request.invocation_id + 1U;
            coordinator.runtime_dispatch_driver_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_STATUS_OWNER_MISMATCH;

            PtoCudaRuntimeFusionResult mismatch_result = {};
            int mismatch_rc =
                persistent_device_uccl_ep_runtime_fusion_entry(
                    &request, &mismatch_result
                );

            assert(mismatch_rc == 0);
            assert(mismatch_result.status == PTO_CUDA_RUNTIME_FUSION_STATUS_FAILED);
            assert(mismatch_result.actual_fused_cross_gpu_execution == 0U);
            assert(
                (mismatch_result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH) != 0U
            );
            assert(
                (mismatch_result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_INVOCATION_MISMATCH) != 0U
            );
            assert(output_sink.status == mismatch_result.status);
            assert(output_sink.failure_fields == mismatch_result.failure_fields);
            std::cout << pto_cuda_uccl_ep_runtime_dispatch_driver_status_name(
                coordinator.runtime_dispatch_driver_scaffold_status.status
            ) << "\\n";
            std::cout << pto_cuda_runtime_fusion_failure_name(
                PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH
            ) << "\\n";
            std::cout << mismatch_result.reason << "\\n";

            coordinator.runtime_dispatch_driver_scaffold_status = saved_driver;
            int driver_rc =
                pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_scaffold_status(
                    &request, &coordinator
                );
            assert(driver_rc == 0);
            assert(
                coordinator.runtime_dispatch_driver_scaffold_status.status ==
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_STATUS_UNSUPPORTED_BOUNDARY
            );

            PtoCudaRuntimeFusionResult result = {};
            int rc = persistent_device_uccl_ep_runtime_fusion_entry(&request, &result);

            assert(rc == 0);
            assert(result.status == PTO_CUDA_RUNTIME_FUSION_STATUS_UNSUPPORTED);
            assert(result.status != PTO_CUDA_RUNTIME_FUSION_STATUS_PASSED);
            assert(result.actual_fused_cross_gpu_execution == 0U);
            assert(
                (result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH) == 0U
            );
            assert(
                (result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_UNSUPPORTED_BOUNDARY) != 0U
            );
            assert(output_sink.status == result.status);
            assert(output_sink.failure_fields == result.failure_fields);
            return 0;
        }
        """,
    )

    assert "driver_owner_mismatch" in output
    assert "private UCCL-EP runtime dispatch driver scaffold/status" in output


def test_private_runtime_dispatch_driver_backend_scaffold_status_is_driver_owned(
    tmp_path,
):
    output = _compile_and_run(
        tmp_path,
        """
        #include "host/pto_cuda_runtime_fusion_abi.h"

        #include <cassert>
        #include <cstdint>
        #include <iostream>

        int main() {
            ChipStorageTaskArgs chip_storage = {};
            PtoCudaCommDeviceDescriptor descriptor = {
                PTO_CUDA_COMM_BACKEND_NCCL, 1U, 7U, 2U, 0xBACE1D7U
            };

            PtoCudaRuntimeFusionRequest request = {};
            request.version = PTO_CUDA_RUNTIME_FUSION_REQUEST_VERSION;
            request.callable_id = 43;
            request.invocation_id = 7171U;
            request.chip_storage_task_args = &chip_storage;
            request.chip_storage_task_args_size = sizeof(chip_storage);
            request.persistent_graph_descriptor = reinterpret_cast<const void *>(0x20);
            request.comm_descriptor = &descriptor;
            request.uccl_ep_capability_metadata = reinterpret_cast<const void *>(0x30);
            request.validation_policy = reinterpret_cast<const void *>(0x40);

            PtoCudaRuntimeFusionResult output_sink = {};
            PtoCudaUcclEpDeviceDescriptorBuffer device_storage = {};
            PtoCudaRuntimeFusionCoordinator coordinator = {};

            int coordinator_rc = pto_cuda_runtime_fusion_prepare_private_coordinator(
                &request,
                &coordinator,
                &device_storage,
                sizeof(device_storage),
                &output_sink
            );
            assert(coordinator_rc == 0);
            assert(
                coordinator.runtime_dispatch_driver_backend_scaffold_status.version ==
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_SCAFFOLD_STATUS_VERSION
            );

            request.coordinator = &coordinator;
            request.descriptor_allocator = &coordinator.descriptor_allocation;
            request.uccl_ep_runtime = &coordinator.descriptor_allocation.runtime_path;
            request.output_sink = coordinator.output_sink;

            PtoCudaUcclEpRuntimeDispatchDriverBackendScaffoldStatus saved_backend =
                coordinator.runtime_dispatch_driver_backend_scaffold_status;

            coordinator.runtime_dispatch_driver_backend_scaffold_status.invocation_id =
                request.invocation_id + 1U;
            coordinator.runtime_dispatch_driver_backend_scaffold_status.backend_owner =
                reinterpret_cast<const void *>(0xBAD0U);
            coordinator.runtime_dispatch_driver_backend_scaffold_status.shared_token += 1U;
            coordinator.runtime_dispatch_driver_backend_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_STATUS_OWNER_MISMATCH;

            PtoCudaRuntimeFusionResult mismatch_result = {};
            int mismatch_rc =
                persistent_device_uccl_ep_runtime_fusion_entry(
                    &request, &mismatch_result
                );

            assert(mismatch_rc == 0);
            assert(mismatch_result.status == PTO_CUDA_RUNTIME_FUSION_STATUS_FAILED);
            assert(mismatch_result.actual_fused_cross_gpu_execution == 0U);
            assert(
                (mismatch_result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_SCAFFOLD) != 0U
            );
            assert(
                (mismatch_result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH) != 0U
            );
            assert(
                (mismatch_result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_INVOCATION_MISMATCH) != 0U
            );
            assert(
                (mismatch_result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_DESCRIPTOR_TOKEN_MISMATCH) != 0U
            );
            assert(output_sink.status == mismatch_result.status);
            assert(output_sink.failure_fields == mismatch_result.failure_fields);
            std::cout << pto_cuda_uccl_ep_runtime_dispatch_driver_backend_status_name(
                coordinator.runtime_dispatch_driver_backend_scaffold_status.status
            ) << "\\n";
            std::cout << pto_cuda_runtime_fusion_failure_name(
                PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_SCAFFOLD
            ) << "\\n";
            std::cout << mismatch_result.reason << "\\n";

            coordinator.runtime_dispatch_driver_backend_scaffold_status = saved_backend;
            int backend_rc =
                pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_scaffold_status(
                    &request, &coordinator
                );
            assert(backend_rc == 0);
            assert(
                coordinator.runtime_dispatch_driver_backend_scaffold_status.status ==
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_STATUS_UNSUPPORTED_BOUNDARY
            );

            PtoCudaRuntimeFusionResult result = {};
            int rc = persistent_device_uccl_ep_runtime_fusion_entry(&request, &result);

            assert(rc == 0);
            assert(result.status == PTO_CUDA_RUNTIME_FUSION_STATUS_UNSUPPORTED);
            assert(result.status != PTO_CUDA_RUNTIME_FUSION_STATUS_PASSED);
            assert(result.actual_fused_cross_gpu_execution == 0U);
            assert(
                (result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_SCAFFOLD) == 0U
            );
            assert(
                (result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH) == 0U
            );
            assert(
                (result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_UNSUPPORTED_BOUNDARY) != 0U
            );
            assert(output_sink.status == result.status);
            assert(output_sink.failure_fields == result.failure_fields);
            return 0;
        }
        """,
    )

    assert "driver_backend_owner_mismatch" in output
    assert "driver_backend_scaffold_status" in output
    assert "private UCCL-EP runtime dispatch driver backend scaffold/status" in output


def test_private_runtime_dispatch_driver_backend_request_scaffold_status_is_backend_owned(
    tmp_path,
):
    output = _compile_and_run(
        tmp_path,
        """
        #include "host/pto_cuda_runtime_fusion_abi.h"

        #include <cassert>
        #include <cstdint>
        #include <iostream>

        int main() {
            ChipStorageTaskArgs chip_storage = {};
            PtoCudaCommDeviceDescriptor descriptor = {
                PTO_CUDA_COMM_BACKEND_NCCL, 1U, 7U, 2U, 0xBEAD123U
            };

            PtoCudaRuntimeFusionRequest request = {};
            request.version = PTO_CUDA_RUNTIME_FUSION_REQUEST_VERSION;
            request.callable_id = 47;
            request.invocation_id = 8181U;
            request.chip_storage_task_args = &chip_storage;
            request.chip_storage_task_args_size = sizeof(chip_storage);
            request.persistent_graph_descriptor = reinterpret_cast<const void *>(0x20);
            request.comm_descriptor = &descriptor;
            request.uccl_ep_capability_metadata = reinterpret_cast<const void *>(0x30);
            request.validation_policy = reinterpret_cast<const void *>(0x40);

            PtoCudaRuntimeFusionResult output_sink = {};
            PtoCudaUcclEpDeviceDescriptorBuffer device_storage = {};
            PtoCudaRuntimeFusionCoordinator coordinator = {};

            int coordinator_rc = pto_cuda_runtime_fusion_prepare_private_coordinator(
                &request,
                &coordinator,
                &device_storage,
                sizeof(device_storage),
                &output_sink
            );
            assert(coordinator_rc == 0);
            assert(
                coordinator.runtime_dispatch_driver_backend_request_scaffold_status.version ==
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_REQUEST_SCAFFOLD_STATUS_VERSION
            );

            request.coordinator = &coordinator;
            request.descriptor_allocator = &coordinator.descriptor_allocation;
            request.uccl_ep_runtime = &coordinator.descriptor_allocation.runtime_path;
            request.output_sink = coordinator.output_sink;

            PtoCudaUcclEpRuntimeDispatchDriverBackendRequestScaffoldStatus saved_request =
                coordinator.runtime_dispatch_driver_backend_request_scaffold_status;

            coordinator.runtime_dispatch_driver_backend_request_scaffold_status.invocation_id =
                request.invocation_id + 1U;
            coordinator.runtime_dispatch_driver_backend_request_scaffold_status.request_owner =
                reinterpret_cast<const void *>(0xBAD0U);
            coordinator.runtime_dispatch_driver_backend_request_scaffold_status.shared_token += 1U;
            coordinator.runtime_dispatch_driver_backend_request_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_REQUEST_STATUS_OWNER_MISMATCH;

            PtoCudaRuntimeFusionResult mismatch_result = {};
            int mismatch_rc =
                persistent_device_uccl_ep_runtime_fusion_entry(
                    &request, &mismatch_result
                );

            assert(mismatch_rc == 0);
            assert(mismatch_result.status == PTO_CUDA_RUNTIME_FUSION_STATUS_FAILED);
            assert(mismatch_result.actual_fused_cross_gpu_execution == 0U);
            assert(
                (mismatch_result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_REQUEST_SCAFFOLD) != 0U
            );
            assert(
                (mismatch_result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH) != 0U
            );
            assert(
                (mismatch_result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_INVOCATION_MISMATCH) != 0U
            );
            assert(
                (mismatch_result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_DESCRIPTOR_TOKEN_MISMATCH) != 0U
            );
            assert(output_sink.status == mismatch_result.status);
            assert(output_sink.failure_fields == mismatch_result.failure_fields);
            std::cout << pto_cuda_uccl_ep_runtime_dispatch_driver_backend_request_status_name(
                coordinator.runtime_dispatch_driver_backend_request_scaffold_status.status
            ) << "\\n";
            std::cout << pto_cuda_runtime_fusion_failure_name(
                PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_REQUEST_SCAFFOLD
            ) << "\\n";
            std::cout << mismatch_result.reason << "\\n";

            coordinator.runtime_dispatch_driver_backend_request_scaffold_status =
                saved_request;
            int backend_request_rc =
                pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_request_scaffold_status(
                    &request, &coordinator
                );
            assert(backend_request_rc == 0);
            assert(
                coordinator.runtime_dispatch_driver_backend_request_scaffold_status.status ==
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_REQUEST_STATUS_UNSUPPORTED_BOUNDARY
            );

            PtoCudaRuntimeFusionResult result = {};
            int rc = persistent_device_uccl_ep_runtime_fusion_entry(&request, &result);

            assert(rc == 0);
            assert(result.status == PTO_CUDA_RUNTIME_FUSION_STATUS_UNSUPPORTED);
            assert(result.status != PTO_CUDA_RUNTIME_FUSION_STATUS_PASSED);
            assert(result.actual_fused_cross_gpu_execution == 0U);
            assert(
                (result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_REQUEST_SCAFFOLD) == 0U
            );
            assert(
                (result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH) == 0U
            );
            assert(
                (result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_UNSUPPORTED_BOUNDARY) != 0U
            );
            assert(output_sink.status == result.status);
            assert(output_sink.failure_fields == result.failure_fields);
            return 0;
        }
        """,
    )

    assert "driver_backend_request_owner_mismatch" in output
    assert "driver_backend_request_scaffold_status" in output
    assert (
        "private UCCL-EP runtime dispatch driver backend request scaffold/status"
        in output
    )


def test_private_runtime_dispatch_driver_backend_dispatch_request_scaffold_status_is_backend_owned(
    tmp_path,
):
    output = _compile_and_run(
        tmp_path,
        """
        #include "host/pto_cuda_runtime_fusion_abi.h"

        #include <cassert>
        #include <cstdint>
        #include <iostream>

        int main() {
            ChipStorageTaskArgs chip_storage = {};
            PtoCudaCommDeviceDescriptor descriptor = {
                PTO_CUDA_COMM_BACKEND_NCCL, 1U, 7U, 2U, 0xD15A7CU
            };

            PtoCudaRuntimeFusionRequest request = {};
            request.version = PTO_CUDA_RUNTIME_FUSION_REQUEST_VERSION;
            request.callable_id = 48;
            request.invocation_id = 9191U;
            request.chip_storage_task_args = &chip_storage;
            request.chip_storage_task_args_size = sizeof(chip_storage);
            request.persistent_graph_descriptor = reinterpret_cast<const void *>(0x20);
            request.comm_descriptor = &descriptor;
            request.uccl_ep_capability_metadata = reinterpret_cast<const void *>(0x30);
            request.validation_policy = reinterpret_cast<const void *>(0x40);

            PtoCudaRuntimeFusionResult output_sink = {};
            PtoCudaUcclEpDeviceDescriptorBuffer device_storage = {};
            PtoCudaRuntimeFusionCoordinator coordinator = {};

            int coordinator_rc = pto_cuda_runtime_fusion_prepare_private_coordinator(
                &request,
                &coordinator,
                &device_storage,
                sizeof(device_storage),
                &output_sink
            );
            assert(coordinator_rc == 0);
            assert(
                coordinator.runtime_dispatch_driver_backend_dispatch_request_scaffold_status.version ==
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_DISPATCH_REQUEST_SCAFFOLD_STATUS_VERSION
            );

            request.coordinator = &coordinator;
            request.descriptor_allocator = &coordinator.descriptor_allocation;
            request.uccl_ep_runtime = &coordinator.descriptor_allocation.runtime_path;
            request.output_sink = coordinator.output_sink;

            PtoCudaUcclEpRuntimeDispatchDriverBackendDispatchRequestScaffoldStatus
                saved_dispatch_request =
                    coordinator.runtime_dispatch_driver_backend_dispatch_request_scaffold_status;

            coordinator.runtime_dispatch_driver_backend_dispatch_request_scaffold_status.
                invocation_id = request.invocation_id + 1U;
            coordinator.runtime_dispatch_driver_backend_dispatch_request_scaffold_status.
                request_owner = reinterpret_cast<const void *>(0xBAD0U);
            coordinator.runtime_dispatch_driver_backend_dispatch_request_scaffold_status.
                shared_token += 1U;
            coordinator.runtime_dispatch_driver_backend_dispatch_request_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_DISPATCH_REQUEST_STATUS_OWNER_MISMATCH;

            PtoCudaRuntimeFusionResult mismatch_result = {};
            int mismatch_rc =
                persistent_device_uccl_ep_runtime_fusion_entry(
                    &request, &mismatch_result
                );

            assert(mismatch_rc == 0);
            assert(mismatch_result.status == PTO_CUDA_RUNTIME_FUSION_STATUS_FAILED);
            assert(mismatch_result.actual_fused_cross_gpu_execution == 0U);
            assert(
                (mismatch_result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_DISPATCH_REQUEST_SCAFFOLD) != 0U
            );
            assert(
                (mismatch_result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH) != 0U
            );
            assert(
                (mismatch_result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_INVOCATION_MISMATCH) != 0U
            );
            assert(
                (mismatch_result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_DESCRIPTOR_TOKEN_MISMATCH) != 0U
            );
            assert(output_sink.status == mismatch_result.status);
            assert(output_sink.failure_fields == mismatch_result.failure_fields);
            std::cout << pto_cuda_uccl_ep_runtime_dispatch_driver_backend_dispatch_request_status_name(
                coordinator.runtime_dispatch_driver_backend_dispatch_request_scaffold_status.status
            ) << "\\n";
            std::cout << pto_cuda_runtime_fusion_failure_name(
                PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_DISPATCH_REQUEST_SCAFFOLD
            ) << "\\n";
            std::cout << mismatch_result.reason << "\\n";

            coordinator.runtime_dispatch_driver_backend_dispatch_request_scaffold_status =
                saved_dispatch_request;
            int dispatch_request_rc =
                pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_dispatch_request_scaffold_status(
                    &request, &coordinator
                );
            assert(dispatch_request_rc == 0);
            assert(
                coordinator.runtime_dispatch_driver_backend_dispatch_request_scaffold_status.status ==
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_DISPATCH_REQUEST_STATUS_UNSUPPORTED_BOUNDARY
            );

            PtoCudaRuntimeFusionResult result = {};
            int rc = persistent_device_uccl_ep_runtime_fusion_entry(&request, &result);

            assert(rc == 0);
            assert(result.status == PTO_CUDA_RUNTIME_FUSION_STATUS_UNSUPPORTED);
            assert(result.status != PTO_CUDA_RUNTIME_FUSION_STATUS_PASSED);
            assert(result.actual_fused_cross_gpu_execution == 0U);
            assert(
                (result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_DISPATCH_REQUEST_SCAFFOLD) == 0U
            );
            assert(
                (result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH) == 0U
            );
            assert(
                (result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_UNSUPPORTED_BOUNDARY) != 0U
            );
            assert(output_sink.status == result.status);
            assert(output_sink.failure_fields == result.failure_fields);
            return 0;
        }
        """,
    )

    assert "driver_backend_dispatch_request_owner_mismatch" in output
    assert "driver_backend_dispatch_request_scaffold_status" in output
    assert (
        "private UCCL-EP runtime dispatch driver backend dispatch request "
        "scaffold/status" in output
    )


def test_private_runtime_dispatch_driver_backend_combine_request_scaffold_status_is_backend_owned(
    tmp_path,
):
    output = _compile_and_run(
        tmp_path,
        """
        #include "host/pto_cuda_runtime_fusion_abi.h"

        #include <cassert>
        #include <cstdint>
        #include <iostream>

        int main() {
            ChipStorageTaskArgs chip_storage = {};
            PtoCudaCommDeviceDescriptor descriptor = {
                PTO_CUDA_COMM_BACKEND_NCCL, 1U, 7U, 2U, 0xC09B17EU
            };

            PtoCudaRuntimeFusionRequest request = {};
            request.version = PTO_CUDA_RUNTIME_FUSION_REQUEST_VERSION;
            request.callable_id = 49;
            request.invocation_id = 9292U;
            request.chip_storage_task_args = &chip_storage;
            request.chip_storage_task_args_size = sizeof(chip_storage);
            request.persistent_graph_descriptor = reinterpret_cast<const void *>(0x20);
            request.comm_descriptor = &descriptor;
            request.uccl_ep_capability_metadata = reinterpret_cast<const void *>(0x30);
            request.validation_policy = reinterpret_cast<const void *>(0x40);

            PtoCudaRuntimeFusionResult output_sink = {};
            PtoCudaUcclEpDeviceDescriptorBuffer device_storage = {};
            PtoCudaRuntimeFusionCoordinator coordinator = {};

            int coordinator_rc = pto_cuda_runtime_fusion_prepare_private_coordinator(
                &request,
                &coordinator,
                &device_storage,
                sizeof(device_storage),
                &output_sink
            );
            assert(coordinator_rc == 0);
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_request_scaffold_status.version ==
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_REQUEST_SCAFFOLD_STATUS_VERSION
            );
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_request_scaffold_status.
                    dispatch_request_status ==
                &coordinator.runtime_dispatch_driver_backend_dispatch_request_scaffold_status
            );

            request.coordinator = &coordinator;
            request.descriptor_allocator = &coordinator.descriptor_allocation;
            request.uccl_ep_runtime = &coordinator.descriptor_allocation.runtime_path;
            request.output_sink = coordinator.output_sink;

            PtoCudaUcclEpRuntimeDispatchDriverBackendCombineRequestScaffoldStatus
                saved_combine_request =
                    coordinator.runtime_dispatch_driver_backend_combine_request_scaffold_status;

            coordinator.runtime_dispatch_driver_backend_combine_request_scaffold_status.
                invocation_id = request.invocation_id + 1U;
            coordinator.runtime_dispatch_driver_backend_combine_request_scaffold_status.
                request_owner = reinterpret_cast<const void *>(0xBAD0U);
            coordinator.runtime_dispatch_driver_backend_combine_request_scaffold_status.
                dispatch_request_status = nullptr;
            coordinator.runtime_dispatch_driver_backend_combine_request_scaffold_status.
                shared_token += 1U;
            coordinator.runtime_dispatch_driver_backend_combine_request_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_REQUEST_STATUS_OWNER_MISMATCH;

            PtoCudaRuntimeFusionResult mismatch_result = {};
            int mismatch_rc =
                persistent_device_uccl_ep_runtime_fusion_entry(
                    &request, &mismatch_result
                );

            assert(mismatch_rc == 0);
            assert(mismatch_result.status == PTO_CUDA_RUNTIME_FUSION_STATUS_FAILED);
            assert(mismatch_result.actual_fused_cross_gpu_execution == 0U);
            assert(
                (mismatch_result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_REQUEST_SCAFFOLD) != 0U
            );
            assert(
                (mismatch_result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH) != 0U
            );
            assert(
                (mismatch_result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_INVOCATION_MISMATCH) != 0U
            );
            assert(
                (mismatch_result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_DESCRIPTOR_TOKEN_MISMATCH) != 0U
            );
            assert(output_sink.status == mismatch_result.status);
            assert(output_sink.failure_fields == mismatch_result.failure_fields);
            std::cout << pto_cuda_uccl_ep_runtime_dispatch_driver_backend_combine_request_status_name(
                coordinator.runtime_dispatch_driver_backend_combine_request_scaffold_status.status
            ) << "\\n";
            std::cout << pto_cuda_runtime_fusion_failure_name(
                PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_REQUEST_SCAFFOLD
            ) << "\\n";
            std::cout << mismatch_result.reason << "\\n";

            coordinator.runtime_dispatch_driver_backend_combine_request_scaffold_status =
                saved_combine_request;
            int combine_request_rc =
                pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_combine_request_scaffold_status(
                    &request, &coordinator
                );
            assert(combine_request_rc == 0);
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_request_scaffold_status.
                    dispatch_request_status ==
                &coordinator.runtime_dispatch_driver_backend_dispatch_request_scaffold_status
            );
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_request_scaffold_status.status ==
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_REQUEST_STATUS_UNSUPPORTED_BOUNDARY
            );

            PtoCudaRuntimeFusionResult result = {};
            int rc = persistent_device_uccl_ep_runtime_fusion_entry(&request, &result);

            assert(rc == 0);
            assert(result.status == PTO_CUDA_RUNTIME_FUSION_STATUS_UNSUPPORTED);
            assert(result.status != PTO_CUDA_RUNTIME_FUSION_STATUS_PASSED);
            assert(result.actual_fused_cross_gpu_execution == 0U);
            assert(
                (result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_REQUEST_SCAFFOLD) == 0U
            );
            assert(
                (result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH) == 0U
            );
            assert(
                (result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_UNSUPPORTED_BOUNDARY) != 0U
            );
            assert(output_sink.status == result.status);
            assert(output_sink.failure_fields == result.failure_fields);
            return 0;
        }
        """,
    )

    assert "driver_backend_combine_request_owner_mismatch" in output
    assert "driver_backend_combine_request_scaffold_status" in output
    assert (
        "private UCCL-EP runtime dispatch driver backend combine request "
        "scaffold/status" in output
    )


def test_private_runtime_dispatch_driver_backend_combine_payload_scaffold_status_is_backend_owned(
    tmp_path,
):
    output = _compile_and_run(
        tmp_path,
        """
        #include "host/pto_cuda_runtime_fusion_abi.h"

        #include <cassert>
        #include <cstdint>
        #include <iostream>

        int main() {
            ChipStorageTaskArgs chip_storage = {};
            PtoCudaCommDeviceDescriptor descriptor = {
                PTO_CUDA_COMM_BACKEND_NCCL, 1U, 7U, 2U, 0xC0A1815EU
            };

            PtoCudaRuntimeFusionRequest request = {};
            request.version = PTO_CUDA_RUNTIME_FUSION_REQUEST_VERSION;
            request.callable_id = 53;
            request.invocation_id = 9393U;
            request.chip_storage_task_args = &chip_storage;
            request.chip_storage_task_args_size = sizeof(chip_storage);
            request.persistent_graph_descriptor = reinterpret_cast<const void *>(0x20);
            request.comm_descriptor = &descriptor;
            request.uccl_ep_capability_metadata = reinterpret_cast<const void *>(0x30);
            request.validation_policy = reinterpret_cast<const void *>(0x40);

            PtoCudaRuntimeFusionResult output_sink = {};
            PtoCudaUcclEpDeviceDescriptorBuffer device_storage = {};
            PtoCudaRuntimeFusionCoordinator coordinator = {};

            int coordinator_rc = pto_cuda_runtime_fusion_prepare_private_coordinator(
                &request,
                &coordinator,
                &device_storage,
                sizeof(device_storage),
                &output_sink
            );
            assert(coordinator_rc == 0);
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_scaffold_status.version ==
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD_STATUS_VERSION
            );
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_scaffold_status.
                    combine_request_status ==
                &coordinator.runtime_dispatch_driver_backend_combine_request_scaffold_status
            );

            request.coordinator = &coordinator;
            request.descriptor_allocator = &coordinator.descriptor_allocation;
            request.uccl_ep_runtime = &coordinator.descriptor_allocation.runtime_path;
            request.output_sink = coordinator.output_sink;

            PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadScaffoldStatus
                saved_combine_payload =
                    coordinator.runtime_dispatch_driver_backend_combine_payload_scaffold_status;

            coordinator.runtime_dispatch_driver_backend_combine_payload_scaffold_status.
                invocation_id = request.invocation_id + 1U;
            coordinator.runtime_dispatch_driver_backend_combine_payload_scaffold_status.
                payload_owner = reinterpret_cast<const void *>(0xBAD0U);
            coordinator.runtime_dispatch_driver_backend_combine_payload_scaffold_status.
                combine_request_status = nullptr;
            coordinator.runtime_dispatch_driver_backend_combine_payload_scaffold_status.
                shared_token += 1U;
            coordinator.runtime_dispatch_driver_backend_combine_payload_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_STATUS_OWNER_MISMATCH;

            PtoCudaRuntimeFusionResult mismatch_result = {};
            int mismatch_rc =
                persistent_device_uccl_ep_runtime_fusion_entry(
                    &request, &mismatch_result
                );

            assert(mismatch_rc == 0);
            assert(mismatch_result.status == PTO_CUDA_RUNTIME_FUSION_STATUS_FAILED);
            assert(mismatch_result.actual_fused_cross_gpu_execution == 0U);
            assert(
                (mismatch_result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD) != 0U
            );
            assert(
                (mismatch_result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH) != 0U
            );
            assert(
                (mismatch_result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_INVOCATION_MISMATCH) != 0U
            );
            assert(
                (mismatch_result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_DESCRIPTOR_TOKEN_MISMATCH) != 0U
            );
            assert(output_sink.status == mismatch_result.status);
            assert(output_sink.failure_fields == mismatch_result.failure_fields);
            std::cout << pto_cuda_uccl_ep_runtime_dispatch_driver_backend_combine_payload_status_name(
                coordinator.runtime_dispatch_driver_backend_combine_payload_scaffold_status.status
            ) << "\\n";
            std::cout << pto_cuda_runtime_fusion_failure_name(
                PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD
            ) << "\\n";
            std::cout << mismatch_result.reason << "\\n";

            coordinator.runtime_dispatch_driver_backend_combine_payload_scaffold_status =
                saved_combine_payload;
            int combine_payload_rc =
                pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_combine_payload_scaffold_status(
                    &request, &coordinator
                );
            assert(combine_payload_rc == 0);
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_scaffold_status.
                    combine_request_status ==
                &coordinator.runtime_dispatch_driver_backend_combine_request_scaffold_status
            );
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_scaffold_status.status ==
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_STATUS_UNSUPPORTED_BOUNDARY
            );

            PtoCudaRuntimeFusionResult result = {};
            int rc = persistent_device_uccl_ep_runtime_fusion_entry(&request, &result);

            assert(rc == 0);
            assert(result.status == PTO_CUDA_RUNTIME_FUSION_STATUS_UNSUPPORTED);
            assert(result.status != PTO_CUDA_RUNTIME_FUSION_STATUS_PASSED);
            assert(result.actual_fused_cross_gpu_execution == 0U);
            assert(
                (result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD) == 0U
            );
            assert(
                (result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH) == 0U
            );
            assert(
                (result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_UNSUPPORTED_BOUNDARY) != 0U
            );
            assert(output_sink.status == result.status);
            assert(output_sink.failure_fields == result.failure_fields);
            return 0;
        }
        """,
    )

    assert "driver_backend_combine_payload_owner_mismatch" in output
    assert "driver_backend_combine_payload_scaffold_status" in output
    assert (
        "private UCCL-EP runtime dispatch driver backend combine payload "
        "scaffold/status" in output
    )


def test_private_runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status_is_backend_owned(
    tmp_path,
):
    output = _compile_and_run(
        tmp_path,
        """
        #include "host/pto_cuda_runtime_fusion_abi.h"

        #include <cassert>
        #include <cstdint>
        #include <iostream>

        int main() {
            ChipStorageTaskArgs chip_storage = {};
            PtoCudaCommDeviceDescriptor descriptor = {
                PTO_CUDA_COMM_BACKEND_NCCL, 1U, 7U, 2U, 0xC0A1815EU
            };

            PtoCudaRuntimeFusionRequest request = {};
            request.version = PTO_CUDA_RUNTIME_FUSION_REQUEST_VERSION;
            request.callable_id = 53;
            request.invocation_id = 9494U;
            request.chip_storage_task_args = &chip_storage;
            request.chip_storage_task_args_size = sizeof(chip_storage);
            request.persistent_graph_descriptor = reinterpret_cast<const void *>(0x20);
            request.comm_descriptor = &descriptor;
            request.uccl_ep_capability_metadata = reinterpret_cast<const void *>(0x30);
            request.validation_policy = reinterpret_cast<const void *>(0x40);

            PtoCudaRuntimeFusionResult output_sink = {};
            PtoCudaUcclEpDeviceDescriptorBuffer device_storage = {};
            PtoCudaRuntimeFusionCoordinator coordinator = {};

            int coordinator_rc = pto_cuda_runtime_fusion_prepare_private_coordinator(
                &request,
                &coordinator,
                &device_storage,
                sizeof(device_storage),
                &output_sink
            );
            assert(coordinator_rc == 0);
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.
                    version ==
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_SCAFFOLD_STATUS_VERSION
            );
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.
                    backend_request_status ==
                &coordinator.runtime_dispatch_driver_backend_request_scaffold_status
            );
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.
                    dispatch_request_status ==
                &coordinator.runtime_dispatch_driver_backend_dispatch_request_scaffold_status
            );
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.
                    combine_request_status ==
                &coordinator.runtime_dispatch_driver_backend_combine_request_scaffold_status
            );
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.
                    combine_payload_status ==
                &coordinator.runtime_dispatch_driver_backend_combine_payload_scaffold_status
            );

            request.coordinator = &coordinator;
            request.descriptor_allocator = &coordinator.descriptor_allocation;
            request.uccl_ep_runtime = &coordinator.descriptor_allocation.runtime_path;
            request.output_sink = coordinator.output_sink;

            auto saved_transfer =
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status;

            auto expect_failed = [&]() {
                PtoCudaRuntimeFusionResult mismatch_result = {};
                int mismatch_rc =
                    persistent_device_uccl_ep_runtime_fusion_entry(
                        &request, &mismatch_result
                    );

                assert(mismatch_rc == 0);
                assert(mismatch_result.status == PTO_CUDA_RUNTIME_FUSION_STATUS_FAILED);
                assert(mismatch_result.actual_fused_cross_gpu_execution == 0U);
                assert(
                    (mismatch_result.failure_fields &
                     PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD) != 0U
                );
                assert(output_sink.status == mismatch_result.status);
                assert(output_sink.failure_fields == mismatch_result.failure_fields);
                std::cout << mismatch_result.reason << "\\n";
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status =
                    saved_transfer;
                output_sink = {};
            };

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.
                transfer_owner = reinterpret_cast<const void *>(0xBAD0U);
            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_OWNER_MISMATCH;
            expect_failed();

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.
                invocation_id = request.invocation_id + 1U;
            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_INVOCATION_MISMATCH;
            expect_failed();

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.
                combine_payload_status = nullptr;
            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_PAYLOAD_SCAFFOLD_MISMATCH;
            expect_failed();

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.
                shared_token += 1U;
            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_DESCRIPTOR_TOKEN_MISMATCH;
            expect_failed();

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.
                rank += 1U;
            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_RANK_DEVICE_MISMATCH;
            expect_failed();

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.
                status_sink = nullptr;
            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_STATUS_SINK_MISMATCH;
            expect_failed();

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_PUBLIC_API_SOURCED_STATE;
            expect_failed();

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_PROVENANCE_SOURCED_STATE;
            expect_failed();

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_FABRICATED_PASS_EVIDENCE;
            expect_failed();

            int transfer_rc =
                pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status(
                    &request, &coordinator
                );
            assert(transfer_rc == 0);
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.
                    combine_payload_status ==
                &coordinator.runtime_dispatch_driver_backend_combine_payload_scaffold_status
            );
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.status ==
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_UNSUPPORTED_BOUNDARY
            );

            PtoCudaRuntimeFusionResult result = {};
            int rc = persistent_device_uccl_ep_runtime_fusion_entry(&request, &result);

            assert(rc == 0);
            assert(result.status == PTO_CUDA_RUNTIME_FUSION_STATUS_UNSUPPORTED);
            assert(result.status != PTO_CUDA_RUNTIME_FUSION_STATUS_PASSED);
            assert(result.actual_fused_cross_gpu_execution == 0U);
            assert(
                (result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD) == 0U
            );
            assert(
                (result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_UNSUPPORTED_BOUNDARY) != 0U
            );
            assert(output_sink.status == result.status);
            assert(output_sink.failure_fields == result.failure_fields);
            std::cout
                << pto_cuda_uccl_ep_runtime_dispatch_driver_backend_combine_payload_transfer_status_name(
                       coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.status
                   )
                << "\\n";
            std::cout << pto_cuda_runtime_fusion_failure_name(
                PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD
            ) << "\\n";
            std::cout << result.reason << "\\n";
            return 0;
        }
        """,
    )

    assert "driver_backend_combine_payload_transfer_map_unsupported_boundary" in output
    assert "driver_backend_combine_payload_scaffold_status" in output
    assert (
        "private UCCL-EP runtime dispatch driver backend combine payload "
        "scaffold/status" in output
    )


def test_private_runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status_is_backend_owned(
    tmp_path,
):
    output = _compile_and_run(
        tmp_path,
        """
        #include "host/pto_cuda_runtime_fusion_abi.h"

        #include <cassert>
        #include <cstdint>
        #include <iostream>

        int main() {
            ChipStorageTaskArgs chip_storage = {};
            PtoCudaCommDeviceDescriptor descriptor = {
                PTO_CUDA_COMM_BACKEND_NCCL, 1U, 7U, 2U, 0xC0A1815EU
            };

            PtoCudaRuntimeFusionRequest request = {};
            request.version = PTO_CUDA_RUNTIME_FUSION_REQUEST_VERSION;
            request.callable_id = 53;
            request.invocation_id = 9696U;
            request.chip_storage_task_args = &chip_storage;
            request.chip_storage_task_args_size = sizeof(chip_storage);
            request.persistent_graph_descriptor = reinterpret_cast<const void *>(0x20);
            request.comm_descriptor = &descriptor;
            request.uccl_ep_capability_metadata = reinterpret_cast<const void *>(0x30);
            request.validation_policy = reinterpret_cast<const void *>(0x40);

            PtoCudaRuntimeFusionResult output_sink = {};
            PtoCudaUcclEpDeviceDescriptorBuffer device_storage = {};
            PtoCudaRuntimeFusionCoordinator coordinator = {};

            int coordinator_rc = pto_cuda_runtime_fusion_prepare_private_coordinator(
                &request,
                &coordinator,
                &device_storage,
                sizeof(device_storage),
                &output_sink
            );
            assert(coordinator_rc == 0);
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.
                    version ==
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_SCAFFOLD_STATUS_VERSION
            );
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.
                    backend_request_status ==
                &coordinator.runtime_dispatch_driver_backend_request_scaffold_status
            );
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.
                    dispatch_request_status ==
                &coordinator.runtime_dispatch_driver_backend_dispatch_request_scaffold_status
            );
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.
                    combine_request_status ==
                &coordinator.runtime_dispatch_driver_backend_combine_request_scaffold_status
            );
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.
                    combine_payload_status ==
                &coordinator.runtime_dispatch_driver_backend_combine_payload_scaffold_status
            );
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.
                    transfer_status ==
                &coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status
            );
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.
                    completion_sink == &output_sink
            );

            request.coordinator = &coordinator;
            request.descriptor_allocator = &coordinator.descriptor_allocation;
            request.uccl_ep_runtime = &coordinator.descriptor_allocation.runtime_path;
            request.output_sink = coordinator.output_sink;

            auto saved_completion =
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status;

            auto expect_failed = [&]() {
                PtoCudaRuntimeFusionResult mismatch_result = {};
                int mismatch_rc =
                    persistent_device_uccl_ep_runtime_fusion_entry(
                        &request, &mismatch_result
                    );

                assert(mismatch_rc == 0);
                assert(mismatch_result.status == PTO_CUDA_RUNTIME_FUSION_STATUS_FAILED);
                assert(mismatch_result.actual_fused_cross_gpu_execution == 0U);
                assert(
                    (mismatch_result.failure_fields &
                     PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD) != 0U
                );
                assert(output_sink.status == mismatch_result.status);
                assert(output_sink.failure_fields == mismatch_result.failure_fields);
                std::cout << mismatch_result.reason << "\\n";
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status =
                    saved_completion;
                output_sink = {};
            };

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.
                completion_owner = reinterpret_cast<const void *>(0xBAD0U);
            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_STATUS_OWNER_MISMATCH;
            expect_failed();

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.
                invocation_id = request.invocation_id + 1U;
            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_STATUS_INVOCATION_MISMATCH;
            expect_failed();

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.
                transfer_status = nullptr;
            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_STATUS_TRANSFER_SCAFFOLD_MISMATCH;
            expect_failed();

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.
                shared_token += 1U;
            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_STATUS_DESCRIPTOR_TOKEN_MISMATCH;
            expect_failed();

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.
                rank += 1U;
            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_STATUS_RANK_DEVICE_MISMATCH;
            expect_failed();

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.
                status_sink = nullptr;
            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_STATUS_STATUS_SINK_MISMATCH;
            expect_failed();

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.
                completion_sink = nullptr;
            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_STATUS_STATUS_SINK_MISMATCH;
            expect_failed();

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_STATUS_PUBLIC_API_SOURCED_STATE;
            expect_failed();

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_STATUS_PROVENANCE_SOURCED_STATE;
            expect_failed();

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_STATUS_FABRICATED_PASS_EVIDENCE;
            expect_failed();

            int completion_rc =
                pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status(
                    &request, &coordinator
                );
            assert(completion_rc == 0);
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.
                    transfer_status ==
                &coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status
            );
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.status ==
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_STATUS_UNSUPPORTED_BOUNDARY
            );

            PtoCudaRuntimeFusionResult result = {};
            int rc = persistent_device_uccl_ep_runtime_fusion_entry(&request, &result);

            assert(rc == 0);
            assert(result.status == PTO_CUDA_RUNTIME_FUSION_STATUS_UNSUPPORTED);
            assert(result.status != PTO_CUDA_RUNTIME_FUSION_STATUS_PASSED);
            assert(result.actual_fused_cross_gpu_execution == 0U);
            assert(
                (result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD) == 0U
            );
            assert(
                (result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_UNSUPPORTED_BOUNDARY) != 0U
            );
            assert(output_sink.status == result.status);
            assert(output_sink.failure_fields == result.failure_fields);
            std::cout
                << pto_cuda_uccl_ep_runtime_dispatch_driver_backend_combine_payload_transfer_completion_status_name(
                       coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.status
                   )
                << "\\n";
            std::cout << pto_cuda_runtime_fusion_failure_name(
                PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD
            ) << "\\n";
            std::cout << result.reason << "\\n";
            return 0;
        }
        """,
    )

    assert "driver_backend_combine_payload_transfer_completion_map_unsupported_boundary" in output
    assert "driver_backend_combine_payload_scaffold_status" in output
    assert (
        "private UCCL-EP runtime dispatch driver backend combine payload "
        "scaffold/status" in output
    )


def test_private_runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status_is_backend_owned(
    tmp_path,
):
    output = _compile_and_run(
        tmp_path,
        """
        #include "host/pto_cuda_runtime_fusion_abi.h"

        #include <cassert>
        #include <cstdint>
        #include <iostream>

        int main() {
            ChipStorageTaskArgs chip_storage = {};
            PtoCudaCommDeviceDescriptor descriptor = {
                PTO_CUDA_COMM_BACKEND_NCCL, 1U, 7U, 2U, 0xC0A1815FU
            };

            PtoCudaRuntimeFusionRequest request = {};
            request.version = PTO_CUDA_RUNTIME_FUSION_REQUEST_VERSION;
            request.callable_id = 54;
            request.invocation_id = 9797U;
            request.chip_storage_task_args = &chip_storage;
            request.chip_storage_task_args_size = sizeof(chip_storage);
            request.persistent_graph_descriptor = reinterpret_cast<const void *>(0x20);
            request.comm_descriptor = &descriptor;
            request.uccl_ep_capability_metadata = reinterpret_cast<const void *>(0x30);
            request.validation_policy = reinterpret_cast<const void *>(0x40);

            PtoCudaRuntimeFusionResult output_sink = {};
            PtoCudaUcclEpDeviceDescriptorBuffer device_storage = {};
            PtoCudaRuntimeFusionCoordinator coordinator = {};

            int coordinator_rc = pto_cuda_runtime_fusion_prepare_private_coordinator(
                &request,
                &coordinator,
                &device_storage,
                sizeof(device_storage),
                &output_sink
            );
            assert(coordinator_rc == 0);
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.
                    version ==
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_SCAFFOLD_STATUS_VERSION
            );
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.
                    backend_request_status ==
                &coordinator.runtime_dispatch_driver_backend_request_scaffold_status
            );
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.
                    dispatch_request_status ==
                &coordinator.runtime_dispatch_driver_backend_dispatch_request_scaffold_status
            );
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.
                    combine_request_status ==
                &coordinator.runtime_dispatch_driver_backend_combine_request_scaffold_status
            );
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.
                    combine_payload_status ==
                &coordinator.runtime_dispatch_driver_backend_combine_payload_scaffold_status
            );
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.
                    transfer_status ==
                &coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status
            );
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.
                    completion_status ==
                &coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status
            );
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.
                    handoff_sink == &output_sink
            );

            request.coordinator = &coordinator;
            request.descriptor_allocator = &coordinator.descriptor_allocation;
            request.uccl_ep_runtime = &coordinator.descriptor_allocation.runtime_path;
            request.output_sink = coordinator.output_sink;

            auto saved_handoff =
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status;

            auto expect_failed = [&]() {
                PtoCudaRuntimeFusionResult mismatch_result = {};
                int mismatch_rc =
                    persistent_device_uccl_ep_runtime_fusion_entry(
                        &request, &mismatch_result
                    );

                assert(mismatch_rc == 0);
                assert(mismatch_result.status == PTO_CUDA_RUNTIME_FUSION_STATUS_FAILED);
                assert(mismatch_result.actual_fused_cross_gpu_execution == 0U);
                assert(
                    (mismatch_result.failure_fields &
                     PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD) != 0U
                );
                assert(output_sink.status == mismatch_result.status);
                assert(output_sink.failure_fields == mismatch_result.failure_fields);
                std::cout << mismatch_result.reason << "\\n";
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status =
                    saved_handoff;
                output_sink = {};
            };

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.
                handoff_owner = reinterpret_cast<const void *>(0xBAD0U);
            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_OWNER_MISMATCH;
            expect_failed();

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.
                invocation_id = request.invocation_id + 1U;
            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_INVOCATION_MISMATCH;
            expect_failed();

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.
                completion_status = nullptr;
            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_COMPLETION_SCAFFOLD_MISMATCH;
            expect_failed();

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.
                transfer_status = nullptr;
            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_TRANSFER_SCAFFOLD_MISMATCH;
            expect_failed();

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.
                shared_token += 1U;
            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_DESCRIPTOR_TOKEN_MISMATCH;
            expect_failed();

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.
                rank += 1U;
            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_RANK_DEVICE_MISMATCH;
            expect_failed();

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.
                status_sink = nullptr;
            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_STATUS_SINK_MISMATCH;
            expect_failed();

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.
                handoff_sink = nullptr;
            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_STATUS_SINK_MISMATCH;
            expect_failed();

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_PUBLIC_API_SOURCED_STATE;
            expect_failed();

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_PROVENANCE_SOURCED_STATE;
            expect_failed();

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_FABRICATED_PASS_EVIDENCE;
            expect_failed();

            int handoff_rc =
                pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status(
                    &request, &coordinator
                );
            assert(handoff_rc == 0);
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.
                    completion_status ==
                &coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status
            );
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.status ==
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_UNSUPPORTED_BOUNDARY
            );

            PtoCudaRuntimeFusionResult result = {};
            int rc = persistent_device_uccl_ep_runtime_fusion_entry(&request, &result);

            assert(rc == 0);
            assert(result.status == PTO_CUDA_RUNTIME_FUSION_STATUS_UNSUPPORTED);
            assert(result.status != PTO_CUDA_RUNTIME_FUSION_STATUS_PASSED);
            assert(result.actual_fused_cross_gpu_execution == 0U);
            assert(
                (result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD) == 0U
            );
            assert(
                (result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_UNSUPPORTED_BOUNDARY) != 0U
            );
            assert(output_sink.status == result.status);
            assert(output_sink.failure_fields == result.failure_fields);
            std::cout
                << pto_cuda_uccl_ep_runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_status_name(
                       coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.status
                   )
                << "\\n";
            std::cout << pto_cuda_runtime_fusion_failure_name(
                PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD
            ) << "\\n";
            std::cout << result.reason << "\\n";
            return 0;
        }
        """,
    )

    assert "driver_backend_combine_payload_transfer_completion_handoff_map_unsupported_boundary" in output
    assert "driver_backend_combine_payload_scaffold_status" in output
    assert (
        "private UCCL-EP runtime dispatch driver backend combine payload "
        "scaffold/status" in output
    )


def test_private_runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status_is_backend_owned(
    tmp_path,
):
    output = _compile_and_run(
        tmp_path,
        """
        #include "host/pto_cuda_runtime_fusion_abi.h"

        #include <cassert>
        #include <cstdint>
        #include <iostream>

        int main() {
            ChipStorageTaskArgs chip_storage = {};
            PtoCudaCommDeviceDescriptor descriptor = {
                PTO_CUDA_COMM_BACKEND_NCCL, 1U, 7U, 2U, 0xC0A1815FU
            };

            PtoCudaRuntimeFusionRequest request = {};
            request.version = PTO_CUDA_RUNTIME_FUSION_REQUEST_VERSION;
            request.callable_id = 55;
            request.invocation_id = 9898U;
            request.chip_storage_task_args = &chip_storage;
            request.chip_storage_task_args_size = sizeof(chip_storage);
            request.persistent_graph_descriptor = reinterpret_cast<const void *>(0x20);
            request.comm_descriptor = &descriptor;
            request.uccl_ep_capability_metadata = reinterpret_cast<const void *>(0x30);
            request.validation_policy = reinterpret_cast<const void *>(0x40);

            PtoCudaRuntimeFusionResult output_sink = {};
            PtoCudaUcclEpDeviceDescriptorBuffer device_storage = {};
            PtoCudaRuntimeFusionCoordinator coordinator = {};

            int coordinator_rc = pto_cuda_runtime_fusion_prepare_private_coordinator(
                &request,
                &coordinator,
                &device_storage,
                sizeof(device_storage),
                &output_sink
            );
            assert(coordinator_rc == 0);
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.
                    version ==
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_SCAFFOLD_STATUS_VERSION
            );
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.
                    backend_request_status ==
                &coordinator.runtime_dispatch_driver_backend_request_scaffold_status
            );
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.
                    dispatch_request_status ==
                &coordinator.runtime_dispatch_driver_backend_dispatch_request_scaffold_status
            );
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.
                    combine_request_status ==
                &coordinator.runtime_dispatch_driver_backend_combine_request_scaffold_status
            );
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.
                    combine_payload_status ==
                &coordinator.runtime_dispatch_driver_backend_combine_payload_scaffold_status
            );
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.
                    transfer_status ==
                &coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status
            );
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.
                    completion_status ==
                &coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status
            );
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.
                    handoff_status ==
                &coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status
            );
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.
                    result_sink == &output_sink
            );

            request.coordinator = &coordinator;
            request.descriptor_allocator = &coordinator.descriptor_allocation;
            request.uccl_ep_runtime = &coordinator.descriptor_allocation.runtime_path;
            request.output_sink = coordinator.output_sink;

            auto saved_result =
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status;

            auto expect_failed = [&]() {
                PtoCudaRuntimeFusionResult mismatch_result = {};
                int mismatch_rc =
                    persistent_device_uccl_ep_runtime_fusion_entry(
                        &request, &mismatch_result
                    );

                assert(mismatch_rc == 0);
                assert(mismatch_result.status == PTO_CUDA_RUNTIME_FUSION_STATUS_FAILED);
                assert(mismatch_result.actual_fused_cross_gpu_execution == 0U);
                assert(
                    (mismatch_result.failure_fields &
                     PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD) != 0U
                );
                assert(output_sink.status == mismatch_result.status);
                assert(output_sink.failure_fields == mismatch_result.failure_fields);
                std::cout << mismatch_result.reason << "\\n";
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status =
                    saved_result;
                output_sink = {};
            };

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.
                result_owner = reinterpret_cast<const void *>(0xBAD0U);
            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_OWNER_MISMATCH;
            expect_failed();

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.
                invocation_id = request.invocation_id + 1U;
            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_INVOCATION_MISMATCH;
            expect_failed();

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.
                handoff_status = nullptr;
            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_HANDOFF_SCAFFOLD_MISMATCH;
            expect_failed();

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.
                completion_status = nullptr;
            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_COMPLETION_SCAFFOLD_MISMATCH;
            expect_failed();

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.
                transfer_status = nullptr;
            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_TRANSFER_SCAFFOLD_MISMATCH;
            expect_failed();

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.
                shared_token += 1U;
            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_DESCRIPTOR_TOKEN_MISMATCH;
            expect_failed();

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.
                rank += 1U;
            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_RANK_DEVICE_MISMATCH;
            expect_failed();

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.
                status_sink = nullptr;
            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_STATUS_SINK_MISMATCH;
            expect_failed();

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.
                result_sink = nullptr;
            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_RESULT_SINK_MISMATCH;
            expect_failed();

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_PUBLIC_API_SOURCED_STATE;
            expect_failed();

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_PROVENANCE_SOURCED_STATE;
            expect_failed();

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_FABRICATED_PASS_EVIDENCE;
            expect_failed();

            int result_rc =
                pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status(
                    &request, &coordinator
                );
            assert(result_rc == 0);
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.
                    handoff_status ==
                &coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status
            );
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.status ==
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_UNSUPPORTED_BOUNDARY
            );

            PtoCudaRuntimeFusionResult result = {};
            int rc = persistent_device_uccl_ep_runtime_fusion_entry(&request, &result);

            assert(rc == 0);
            assert(result.status == PTO_CUDA_RUNTIME_FUSION_STATUS_UNSUPPORTED);
            assert(result.status != PTO_CUDA_RUNTIME_FUSION_STATUS_PASSED);
            assert(result.actual_fused_cross_gpu_execution == 0U);
            assert(
                (result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD) == 0U
            );
            assert(
                (result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_UNSUPPORTED_BOUNDARY) != 0U
            );
            assert(output_sink.status == result.status);
            assert(output_sink.failure_fields == result.failure_fields);
            std::cout
                << pto_cuda_uccl_ep_runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_status_name(
                       coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.status
                   )
                << "\\n";
            std::cout << pto_cuda_runtime_fusion_failure_name(
                PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD
            ) << "\\n";
            std::cout << result.reason << "\\n";
            return 0;
        }
        """,
    )

    assert "driver_backend_combine_payload_transfer_completion_handoff_result_map_unsupported_boundary" in output
    assert "driver_backend_combine_payload_scaffold_status" in output
    assert (
        "private UCCL-EP runtime dispatch driver backend combine payload "
        "scaffold/status" in output
    )


def test_private_runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_transport_scaffold_status_is_backend_owned(
    tmp_path,
):
    output = _compile_and_run(
        tmp_path,
        """
        #include "host/pto_cuda_runtime_fusion_abi.h"

        #include <cassert>
        #include <cstdint>
        #include <iostream>

        int main() {
            ChipStorageTaskArgs chip_storage = {};
            PtoCudaCommDeviceDescriptor descriptor = {
                PTO_CUDA_COMM_BACKEND_NCCL, 1U, 7U, 2U, 0xC0A1815FU
            };

            PtoCudaRuntimeFusionRequest request = {};
            request.version = PTO_CUDA_RUNTIME_FUSION_REQUEST_VERSION;
            request.callable_id = 56;
            request.invocation_id = 9999U;
            request.chip_storage_task_args = &chip_storage;
            request.chip_storage_task_args_size = sizeof(chip_storage);
            request.persistent_graph_descriptor = reinterpret_cast<const void *>(0x20);
            request.comm_descriptor = &descriptor;
            request.uccl_ep_capability_metadata = reinterpret_cast<const void *>(0x30);
            request.validation_policy = reinterpret_cast<const void *>(0x40);

            PtoCudaRuntimeFusionResult output_sink = {};
            PtoCudaUcclEpDeviceDescriptorBuffer device_storage = {};
            PtoCudaRuntimeFusionCoordinator coordinator = {};

            int coordinator_rc = pto_cuda_runtime_fusion_prepare_private_coordinator(
                &request,
                &coordinator,
                &device_storage,
                sizeof(device_storage),
                &output_sink
            );
            assert(coordinator_rc == 0);
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_transport_scaffold_status.
                    version ==
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_TRANSPORT_SCAFFOLD_STATUS_VERSION
            );
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_transport_scaffold_status.
                    backend_request_status ==
                &coordinator.runtime_dispatch_driver_backend_request_scaffold_status
            );
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_transport_scaffold_status.
                    dispatch_request_status ==
                &coordinator.runtime_dispatch_driver_backend_dispatch_request_scaffold_status
            );
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_transport_scaffold_status.
                    combine_request_status ==
                &coordinator.runtime_dispatch_driver_backend_combine_request_scaffold_status
            );
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_transport_scaffold_status.
                    combine_payload_status ==
                &coordinator.runtime_dispatch_driver_backend_combine_payload_scaffold_status
            );
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_transport_scaffold_status.
                    transfer_status ==
                &coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status
            );
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_transport_scaffold_status.
                    completion_status ==
                &coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status
            );
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_transport_scaffold_status.
                    handoff_status ==
                &coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status
            );
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_transport_scaffold_status.
                    handoff_result_status ==
                &coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status
            );
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_transport_scaffold_status.
                    result_sink == &output_sink
            );
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_transport_scaffold_status.
                    transport_sink == &output_sink
            );
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_transport_scaffold_status.
                    transport_handle ==
                &coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status
            );

            request.coordinator = &coordinator;
            request.descriptor_allocator = &coordinator.descriptor_allocation;
            request.uccl_ep_runtime = &coordinator.descriptor_allocation.runtime_path;
            request.output_sink = coordinator.output_sink;

            auto saved_transport =
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_transport_scaffold_status;

            auto expect_failed = [&]() {
                PtoCudaRuntimeFusionResult mismatch_result = {};
                int mismatch_rc =
                    persistent_device_uccl_ep_runtime_fusion_entry(
                        &request, &mismatch_result
                    );

                assert(mismatch_rc == 0);
                assert(mismatch_result.status == PTO_CUDA_RUNTIME_FUSION_STATUS_FAILED);
                assert(mismatch_result.actual_fused_cross_gpu_execution == 0U);
                assert(
                    (mismatch_result.failure_fields &
                     PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD) != 0U
                );
                assert(output_sink.status == mismatch_result.status);
                assert(output_sink.failure_fields == mismatch_result.failure_fields);
                std::cout << mismatch_result.reason << "\\n";
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_transport_scaffold_status =
                    saved_transport;
                output_sink = {};
            };

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_transport_scaffold_status.
                transport_owner = reinterpret_cast<const void *>(0xBAD0U);
            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_transport_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_TRANSPORT_STATUS_OWNER_MISMATCH;
            expect_failed();

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_transport_scaffold_status.
                invocation_id = request.invocation_id + 1U;
            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_transport_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_TRANSPORT_STATUS_INVOCATION_MISMATCH;
            expect_failed();

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_transport_scaffold_status.
                handoff_result_status = nullptr;
            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_transport_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_TRANSPORT_STATUS_HANDOFF_RESULT_SCAFFOLD_MISMATCH;
            expect_failed();

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_transport_scaffold_status.
                shared_token += 1U;
            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_transport_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_TRANSPORT_STATUS_DESCRIPTOR_TOKEN_MISMATCH;
            expect_failed();

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_transport_scaffold_status.
                world_size += 1U;
            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_transport_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_TRANSPORT_STATUS_RANK_DEVICE_WORLD_SIZE_MISMATCH;
            expect_failed();

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_transport_scaffold_status.
                status_sink = nullptr;
            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_transport_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_TRANSPORT_STATUS_STATUS_SINK_MISMATCH;
            expect_failed();

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_transport_scaffold_status.
                result_sink = nullptr;
            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_transport_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_TRANSPORT_STATUS_RESULT_SINK_MISMATCH;
            expect_failed();

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_transport_scaffold_status.
                transport_sink = nullptr;
            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_transport_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_TRANSPORT_STATUS_TRANSPORT_SINK_MISMATCH;
            expect_failed();

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_transport_scaffold_status.
                transport_handle = nullptr;
            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_transport_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_TRANSPORT_STATUS_TRANSPORT_HANDLE_MISMATCH;
            expect_failed();

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_transport_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_TRANSPORT_STATUS_PUBLIC_API_SOURCED_STATE;
            expect_failed();

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_transport_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_TRANSPORT_STATUS_PROVENANCE_SOURCED_STATE;
            expect_failed();

            coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_transport_scaffold_status.status =
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_TRANSPORT_STATUS_FABRICATED_PASS_EVIDENCE;
            expect_failed();

            int transport_rc =
                pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_transport_scaffold_status(
                    &request, &coordinator
                );
            assert(transport_rc == 0);
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_transport_scaffold_status.
                    handoff_result_status ==
                &coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status
            );
            assert(
                coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_transport_scaffold_status.status ==
                PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_TRANSPORT_STATUS_UNSUPPORTED_BOUNDARY
            );

            PtoCudaRuntimeFusionResult result = {};
            int rc = persistent_device_uccl_ep_runtime_fusion_entry(&request, &result);

            assert(rc == 0);
            assert(result.status == PTO_CUDA_RUNTIME_FUSION_STATUS_UNSUPPORTED);
            assert(result.status != PTO_CUDA_RUNTIME_FUSION_STATUS_PASSED);
            assert(result.actual_fused_cross_gpu_execution == 0U);
            assert(
                (result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD) == 0U
            );
            assert(
                (result.failure_fields &
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_UNSUPPORTED_BOUNDARY) != 0U
            );
            assert(output_sink.status == result.status);
            assert(output_sink.failure_fields == result.failure_fields);
            std::cout
                << pto_cuda_uccl_ep_runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_transport_status_name(
                       coordinator.runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_transport_scaffold_status.status
                   )
                << "\\n";
            std::cout << pto_cuda_runtime_fusion_failure_name(
                PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD
            ) << "\\n";
            std::cout << result.reason << "\\n";
            return 0;
        }
        """,
    )

    assert "driver_backend_combine_payload_transfer_completion_handoff_result_transport_map_unsupported_boundary" in output
    assert "driver_backend_combine_payload_scaffold_status" in output
    assert (
        "private UCCL-EP runtime dispatch driver backend combine payload "
        "scaffold/status" in output
    )


def test_private_runtime_fusion_request_envelope_keeps_chip_storage_typed_and_separate(
    tmp_path,
):
    _compile_and_run(
        tmp_path,
        """
        #include "host/pto_cuda_persistent_device_abi.h"
        #include "host/pto_cuda_private_run_envelope.h"
        #include "host/pto_cuda_runtime_fusion_abi.h"

        #include <cassert>
        #include <type_traits>

        int main() {
            static_assert(
                std::is_same<
                    decltype(PtoCudaRuntimeFusionRequest{}.chip_storage_task_args),
                    const ChipStorageTaskArgs *
                >::value,
                "runtime fusion request must carry a typed ChipStorageTaskArgs pointer"
            );

            ChipStorageTaskArgs chip_storage = {};
            PtoCudaPersistentDagArgs persistent_dag_args = {};
            PtoCudaPrivateRunArgsEnvelope envelope = {};
            envelope.version = PTO_CUDA_PRIVATE_RUN_ENVELOPE_VERSION;
            envelope.runtime_task_args = &persistent_dag_args;
            envelope.runtime_task_args_size = sizeof(persistent_dag_args);
            envelope.chip_storage_task_args = &chip_storage;
            envelope.chip_storage_task_args_size = sizeof(chip_storage);

            PtoCudaRuntimeFusionRequest request = {};
            request.version = PTO_CUDA_RUNTIME_FUSION_REQUEST_VERSION;
            request.callable_id = 17;
            request.chip_storage_task_args = envelope.chip_storage_task_args;
            request.chip_storage_task_args_size = envelope.chip_storage_task_args_size;
            request.persistent_graph_descriptor =
                static_cast<const PtoCudaPersistentDagArgs *>(
                    envelope.runtime_task_args
                )->state;

            assert(request.chip_storage_task_args == &chip_storage);
            assert(request.chip_storage_task_args !=
                   reinterpret_cast<const ChipStorageTaskArgs *>(&persistent_dag_args));
            assert(request.chip_storage_task_args_size == sizeof(ChipStorageTaskArgs));
            assert(envelope.runtime_task_args == &persistent_dag_args);
            assert(envelope.runtime_task_args_size == sizeof(PtoCudaPersistentDagArgs));
            return 0;
        }
        """,
    )


def test_private_runtime_fusion_envelope_validates_same_invocation_dag_only(
    tmp_path,
):
    _compile_and_run(
        tmp_path,
        """
        #include "host/pto_cuda_host_schedule_abi.h"
        #include "host/pto_cuda_persistent_device_abi.h"
        #include "host/pto_cuda_private_run_envelope.h"

        #include <cassert>
        #include <cstdint>

        int main() {
            ChipStorageTaskArgs chip_storage = {};
            PtoCudaPersistentDagState dag_state = {};
            PtoCudaPersistentDagArgs persistent_dag_args = {&dag_state};

            PtoCudaPrivateRunArgsEnvelope envelope = {};
            int rc = pto_cuda_private_run_envelope_init(
                &envelope,
                31,
                9001,
                &persistent_dag_args,
                sizeof(persistent_dag_args),
                &chip_storage,
                sizeof(chip_storage)
            );
            assert(rc == 0);
            assert(envelope.callable_id == 31);
            assert(envelope.invocation_id == 9001);

            assert(
                pto_cuda_private_run_envelope_validate(
                    &envelope,
                    31,
                    9001,
                    PTO_CUDA_PERSISTENT_OP_DAG_F32_RING,
                    sizeof(PtoCudaPersistentDagArgs)
                ) == PTO_CUDA_PRIVATE_RUN_ENVELOPE_OK
            );
            assert(
                pto_cuda_private_run_envelope_validate(
                    &envelope,
                    32,
                    9001,
                    PTO_CUDA_PERSISTENT_OP_DAG_F32_RING,
                    sizeof(PtoCudaPersistentDagArgs)
                ) == PTO_CUDA_PRIVATE_RUN_ENVELOPE_CALLABLE_MISMATCH
            );
            assert(
                pto_cuda_private_run_envelope_validate(
                    &envelope,
                    31,
                    9002,
                    PTO_CUDA_PERSISTENT_OP_DAG_F32_RING,
                    sizeof(PtoCudaPersistentDagArgs)
                ) == PTO_CUDA_PRIVATE_RUN_ENVELOPE_CROSS_INVOCATION
            );
            assert(
                pto_cuda_private_run_envelope_validate(
                    &envelope,
                    31,
                    9001,
                    PTO_CUDA_HOST_OP_VECTOR_ADD_F32,
                    sizeof(PtoCudaPersistentDagArgs)
                ) == PTO_CUDA_PRIVATE_RUN_ENVELOPE_CALLABLE_TYPE_MISMATCH
            );
            return 0;
        }
        """,
    )


def test_private_runtime_fusion_envelope_rejects_null_wrong_size_and_stale(
    tmp_path,
):
    _compile_and_run(
        tmp_path,
        """
        #include "host/pto_cuda_persistent_device_abi.h"
        #include "host/pto_cuda_private_run_envelope.h"

        #include <cassert>

        int main() {
            ChipStorageTaskArgs chip_storage = {};
            PtoCudaPersistentDagState dag_state = {};
            PtoCudaPersistentDagArgs persistent_dag_args = {&dag_state};

            PtoCudaPrivateRunArgsEnvelope envelope = {};
            assert(
                pto_cuda_private_run_envelope_init(
                    nullptr,
                    1,
                    2,
                    &persistent_dag_args,
                    sizeof(persistent_dag_args),
                    &chip_storage,
                    sizeof(chip_storage)
                ) == PTO_CUDA_PRIVATE_RUN_ENVELOPE_NULL_POINTER
            );
            assert(
                pto_cuda_private_run_envelope_init(
                    &envelope,
                    1,
                    2,
                    nullptr,
                    sizeof(persistent_dag_args),
                    &chip_storage,
                    sizeof(chip_storage)
                ) == PTO_CUDA_PRIVATE_RUN_ENVELOPE_NULL_POINTER
            );
            assert(
                pto_cuda_private_run_envelope_init(
                    &envelope,
                    1,
                    2,
                    &persistent_dag_args,
                    sizeof(persistent_dag_args) - 1,
                    &chip_storage,
                    sizeof(chip_storage)
                ) == PTO_CUDA_PRIVATE_RUN_ENVELOPE_WRONG_RUNTIME_ARGS_SIZE
            );
            assert(
                pto_cuda_private_run_envelope_init(
                    &envelope,
                    1,
                    2,
                    &persistent_dag_args,
                    sizeof(persistent_dag_args),
                    &chip_storage,
                    sizeof(chip_storage) - 1
                ) == PTO_CUDA_PRIVATE_RUN_ENVELOPE_WRONG_CHIP_STORAGE_SIZE
            );

            assert(
                pto_cuda_private_run_envelope_init(
                    &envelope,
                    1,
                    2,
                    &persistent_dag_args,
                    sizeof(persistent_dag_args),
                    &chip_storage,
                    sizeof(chip_storage)
                ) == PTO_CUDA_PRIVATE_RUN_ENVELOPE_OK
            );
            envelope.version = PTO_CUDA_PRIVATE_RUN_ENVELOPE_VERSION - 1;
            assert(
                pto_cuda_private_run_envelope_validate(
                    &envelope,
                    1,
                    2,
                    PTO_CUDA_PERSISTENT_OP_DAG_F32_RING,
                    sizeof(PtoCudaPersistentDagArgs)
                ) == PTO_CUDA_PRIVATE_RUN_ENVELOPE_STALE
            );
            return 0;
        }
        """,
    )


def test_cuda_host_runtime_hooks_private_entry_without_public_api_expansion():
    host_runtime = (
        ROOT / "src" / "cuda" / "platform" / "onboard" / "host" / "pto_runtime_c_api.cpp"
    ).read_text(encoding="utf-8")
    chip_worker = (ROOT / "src" / "common" / "worker" / "chip_worker.cpp").read_text(
        encoding="utf-8"
    )
    common_abi = (ROOT / "src" / "common" / "worker" / "pto_runtime_c_api.h").read_text(
        encoding="utf-8"
    )

    assert "persistent_device_uccl_ep_runtime_fusion_entry" in host_runtime
    assert "PtoCudaRuntimeFusionRequest" in host_runtime
    assert "PtoCudaRuntimeFusionResult" in host_runtime
    assert "PtoCudaPrivateRunArgsEnvelope" in host_runtime
    assert "run_prepared_with_cuda_private_args" in host_runtime
    assert "run_prepared_with_cuda_private_args" in chip_worker
    assert "request.chip_storage_task_args = args" not in host_runtime
    assert "request.chip_storage_task_args = envelope->chip_storage_task_args" in host_runtime
    assert "persistent_device_uccl_ep_runtime_fusion_entry" not in common_abi
    assert "PtoCudaRuntimeFusionRequest" not in common_abi
    assert "PtoCudaPrivateRunArgsEnvelope" not in common_abi


def test_chip_worker_builds_private_handoff_without_inventing_runtime_args():
    chip_worker = (ROOT / "src" / "common" / "worker" / "chip_worker.cpp").read_text(
        encoding="utf-8"
    )

    assert "envelope.runtime_task_args = args;" not in chip_worker
    assert "envelope.runtime_task_args_size = sizeof(*args);" not in chip_worker
    assert "envelope.chip_storage_task_args = args;" in chip_worker
    assert "envelope.chip_storage_task_args_size = sizeof(*args);" in chip_worker
    assert "run_prepared_with_cuda_private_args_fn_" in chip_worker
    assert "ChipWorker cannot build CUDA private run envelope" not in chip_worker


def test_cuda_host_runtime_validates_private_handoff_after_callable_resolution():
    host_runtime = (
        ROOT / "src" / "cuda" / "platform" / "onboard" / "host" / "pto_runtime_c_api.cpp"
    ).read_text(encoding="utf-8")

    assert "PreparedCallable &prepared = it->second" in host_runtime
    assert "pto_cuda_private_run_envelope_validate" in host_runtime
    assert "prepared.op" in host_runtime
    assert "sizeof(PtoCudaPersistentDagArgs)" in host_runtime
    assert "PTO_CUDA_PRIVATE_RUN_ENVELOPE_CALLABLE_TYPE_MISMATCH" in host_runtime
