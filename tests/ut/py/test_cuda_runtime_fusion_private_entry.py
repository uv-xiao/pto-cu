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
