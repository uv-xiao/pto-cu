/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#ifndef SRC_CUDA_PLATFORM_INCLUDE_HOST_PTO_CUDA_RUNTIME_FUSION_ABI_H_
#define SRC_CUDA_PLATFORM_INCLUDE_HOST_PTO_CUDA_RUNTIME_FUSION_ABI_H_

#include "pto_cuda_comm_descriptor_abi.h"
#include "task_interface/task_args.h"

#include <stddef.h>
#include <stdint.h>

static const uint32_t PTO_CUDA_RUNTIME_FUSION_REQUEST_VERSION = 1;
static const uint32_t PTO_CUDA_RUNTIME_FUSION_RESULT_VERSION = 1;
static const uint32_t PTO_CUDA_UCCL_EP_RUNTIME_PATH_VERSION = 1;
static const uint32_t PTO_CUDA_UCCL_EP_RUNTIME_DESCRIPTOR_VIEW_VERSION = 1;
static const uint32_t PTO_CUDA_UCCL_EP_DESCRIPTOR_ALLOCATION_VERSION = 1;
static const uint32_t PTO_CUDA_UCCL_EP_DEVICE_DESCRIPTOR_BUFFER_VERSION = 1;
static const uint32_t PTO_CUDA_RUNTIME_FUSION_COORDINATOR_VERSION = 1;
static const uint32_t PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_SCAFFOLD_STATUS_VERSION = 1;
static const uint32_t PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_REQUEST_HANDOFF_SCAFFOLD_STATUS_VERSION = 1;
static const uint32_t PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_HANDOFF_DRIVER_STATE_VERSION = 1;

enum PtoCudaRuntimeFusionStatus : uint32_t {
    PTO_CUDA_RUNTIME_FUSION_STATUS_UNSUPPORTED = 1,
    PTO_CUDA_RUNTIME_FUSION_STATUS_SETUP_FAILED = 2,
    PTO_CUDA_RUNTIME_FUSION_STATUS_FAILED = 3,
    PTO_CUDA_RUNTIME_FUSION_STATUS_PASSED = 4,
};

enum PtoCudaRuntimeFusionEvidenceSource : uint32_t {
    PTO_CUDA_RUNTIME_FUSION_EVIDENCE_NONE = 0,
    PTO_CUDA_RUNTIME_FUSION_EVIDENCE_COORDINATOR_RESULT = 1,
    PTO_CUDA_RUNTIME_FUSION_EVIDENCE_ADAPTER_PROVENANCE = 2,
    PTO_CUDA_RUNTIME_FUSION_EVIDENCE_EXAMPLE_JSON = 3,
    PTO_CUDA_RUNTIME_FUSION_EVIDENCE_HANDOFF_METADATA = 4,
    PTO_CUDA_RUNTIME_FUSION_EVIDENCE_PUBLIC_TASK_ARGS = 5,
    PTO_CUDA_RUNTIME_FUSION_EVIDENCE_PUBLIC_CALL_CONFIG = 6,
    PTO_CUDA_RUNTIME_FUSION_EVIDENCE_PAYLOAD_PROVENANCE = 7,
};

enum PtoCudaRuntimeFusionFailure : uint32_t {
    PTO_CUDA_RUNTIME_FUSION_FAILURE_UNSUPPORTED_BOUNDARY = 1U << 0U,
    PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_COORDINATOR = 1U << 1U,
    PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_DESCRIPTOR_ALLOCATOR = 1U << 2U,
    PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_UCCL_EP_RUNTIME = 1U << 3U,
    PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_VALIDATION_POLICY = 1U << 4U,
    PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_CHIP_STORAGE_TASK_ARGS = 1U << 5U,
    PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_PERSISTENT_GRAPH_DESCRIPTOR = 1U << 6U,
    PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_RANK_DEVICE_METADATA = 1U << 7U,
    PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_UCCL_EP_CAPABILITY = 1U << 8U,
    PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_OUTPUT_SINK = 1U << 9U,
    PTO_CUDA_RUNTIME_FUSION_FAILURE_FABRICATED_OR_UNTRUSTED_PASS_EVIDENCE = 1U << 10U,
    PTO_CUDA_RUNTIME_FUSION_FAILURE_STALE_DESCRIPTOR_VIEW = 1U << 11U,
    PTO_CUDA_RUNTIME_FUSION_FAILURE_DESCRIPTOR_TOKEN_MISMATCH = 1U << 12U,
    PTO_CUDA_RUNTIME_FUSION_FAILURE_RANK_DEVICE_MISMATCH = 1U << 13U,
    PTO_CUDA_RUNTIME_FUSION_FAILURE_TRANSPORT_MODE_MISMATCH = 1U << 14U,
    PTO_CUDA_RUNTIME_FUSION_FAILURE_DESCRIPTOR_VOCABULARY_MISMATCH = 1U << 15U,
    PTO_CUDA_RUNTIME_FUSION_FAILURE_PUBLIC_API_RUNTIME_PATH = 1U << 16U,
    PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_RUNTIME_DISPATCH_SCAFFOLD = 1U << 17U,
    PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_RUNTIME_DISPATCH_HANDOFF_DRIVER = 1U << 18U,
};

enum PtoCudaUcclEpRuntimePathSource : uint32_t {
    PTO_CUDA_UCCL_EP_RUNTIME_PATH_SOURCE_COORDINATOR_OWNED = 1,
    PTO_CUDA_UCCL_EP_RUNTIME_PATH_SOURCE_PUBLIC_API = 2,
    PTO_CUDA_UCCL_EP_RUNTIME_PATH_SOURCE_EXAMPLE_JSON = 3,
    PTO_CUDA_UCCL_EP_RUNTIME_PATH_SOURCE_ADAPTER_PROVENANCE = 4,
    PTO_CUDA_UCCL_EP_RUNTIME_PATH_SOURCE_HANDOFF_METADATA = 5,
};

enum PtoCudaUcclEpTransportMode : uint32_t {
    PTO_CUDA_UCCL_EP_TRANSPORT_MODE_UNKNOWN = 0,
    PTO_CUDA_UCCL_EP_TRANSPORT_MODE_EP = 1,
};

enum PtoCudaRuntimeFusionDescriptorVocabulary : uint32_t {
    PTO_CUDA_RUNTIME_FUSION_DESCRIPTOR_VOCABULARY_DISPATCH = 1U << 0U,
    PTO_CUDA_RUNTIME_FUSION_DESCRIPTOR_VOCABULARY_COMBINE = 1U << 1U,
};

struct PtoCudaUcclEpRuntimeDescriptorView {
    uint32_t version;
    uint64_t invocation_id;
    const void *persistent_graph_descriptor;
    uint32_t capability_crc32;
    uint32_t rank;
    uint32_t device_id;
    uint32_t world_size;
    uint32_t descriptor_vocabulary;
    uint64_t shared_token;
    uint32_t source;
};

struct PtoCudaUcclEpRuntimePath {
    uint32_t version;
    uint32_t transport_mode;
    const PtoCudaUcclEpRuntimeDescriptorView *dispatch_descriptor;
    const PtoCudaUcclEpRuntimeDescriptorView *combine_descriptor;
};

struct PtoCudaUcclEpDeviceDescriptor {
    uint32_t version;
    uint32_t descriptor_vocabulary;
    uint64_t invocation_id;
    const void *persistent_graph_descriptor;
    uint32_t capability_crc32;
    uint32_t rank;
    uint32_t device_id;
    uint32_t world_size;
    uint64_t shared_token;
};

struct PtoCudaUcclEpDeviceDescriptorBuffer {
    PtoCudaUcclEpDeviceDescriptor dispatch_descriptor;
    PtoCudaUcclEpDeviceDescriptor combine_descriptor;
};

struct PtoCudaUcclEpDescriptorHostControl {
    uint32_t version;
    uint32_t runtime_owned;
    uint64_t invocation_id;
    const void *persistent_graph_descriptor;
    uint32_t capability_crc32;
    uint32_t rank;
    uint32_t device_id;
    uint32_t world_size;
    uint32_t descriptor_vocabulary;
    uint32_t allocation_state;
    uint64_t shared_token;
    size_t device_buffer_size;
    size_t dispatch_descriptor_offset;
    size_t combine_descriptor_offset;
};

struct PtoCudaUcclEpDescriptorAllocation {
    PtoCudaUcclEpDescriptorHostControl host_control;
    PtoCudaUcclEpDeviceDescriptorBuffer *device_buffer;
    PtoCudaUcclEpRuntimeDescriptorView dispatch_descriptor;
    PtoCudaUcclEpRuntimeDescriptorView combine_descriptor;
    PtoCudaUcclEpRuntimePath runtime_path;
};

struct PtoCudaRuntimeFusionResult;

struct PtoCudaUcclEpRuntimeDispatchScaffoldStatus {
    uint32_t version;
    uint64_t invocation_id;
    const PtoCudaUcclEpRuntimePath *runtime_path;
    uint32_t dispatch_eligible;
    uint32_t status;
    uint32_t failure_fields;
};

struct PtoCudaUcclEpRuntimeDispatchHandoffDriverState {
    uint32_t version;
    uint64_t invocation_id;
    const void *request_owner;
    const PtoCudaUcclEpRuntimeDispatchScaffoldStatus *runtime_dispatch_scaffold_status;
    const PtoCudaUcclEpRuntimePath *runtime_path;
    uint32_t status;
    uint32_t failure_fields;
};

struct PtoCudaUcclEpRuntimeDispatchRequestHandoffScaffoldStatus {
    uint32_t version;
    uint64_t invocation_id;
    const void *request_owner;
    const PtoCudaUcclEpRuntimeDispatchScaffoldStatus *runtime_dispatch_scaffold_status;
    const PtoCudaUcclEpRuntimeDispatchHandoffDriverState *driver_state;
    uint32_t handoff_eligible;
    uint32_t status;
    uint32_t failure_fields;
};

struct PtoCudaRuntimeFusionCoordinator {
    uint32_t version;
    uint64_t invocation_id;
    PtoCudaUcclEpDescriptorAllocation descriptor_allocation;
    PtoCudaUcclEpRuntimeDispatchScaffoldStatus runtime_dispatch_scaffold_status;
    PtoCudaUcclEpRuntimeDispatchHandoffDriverState runtime_dispatch_request_handoff_driver_state;
    PtoCudaUcclEpRuntimeDispatchRequestHandoffScaffoldStatus
        runtime_dispatch_request_handoff_scaffold_status;
    PtoCudaRuntimeFusionResult *output_sink;
    uint32_t status;
    uint32_t failure_fields;
};

struct PtoCudaRuntimeFusionRequest {
    uint32_t version;
    int32_t callable_id;
    uint64_t invocation_id;
    const ChipStorageTaskArgs *chip_storage_task_args;
    size_t chip_storage_task_args_size;
    const void *persistent_graph_descriptor;
    const PtoCudaCommDeviceDescriptor *comm_descriptor;
    const void *uccl_ep_capability_metadata;
    const void *coordinator;
    const void *descriptor_allocator;
    const void *uccl_ep_runtime;
    const void *validation_policy;
    void *output_sink;
    uint32_t pass_evidence_source;
};

struct PtoCudaRuntimeFusionResult {
    uint32_t version;
    uint32_t status;
    uint32_t actual_fused_cross_gpu_execution;
    uint32_t failure_fields;
    int32_t callable_id;
    uint32_t rank;
    uint32_t device_id;
    uint32_t world_size;
    uint32_t capability_crc32;
    const char *producer;
    const char *entry_name;
    const char *reason;
};

inline int pto_cuda_runtime_fusion_evidence_source_is_forbidden(uint32_t source) {
    return source == PTO_CUDA_RUNTIME_FUSION_EVIDENCE_ADAPTER_PROVENANCE ||
           source == PTO_CUDA_RUNTIME_FUSION_EVIDENCE_EXAMPLE_JSON ||
           source == PTO_CUDA_RUNTIME_FUSION_EVIDENCE_HANDOFF_METADATA ||
           source == PTO_CUDA_RUNTIME_FUSION_EVIDENCE_PUBLIC_TASK_ARGS ||
           source == PTO_CUDA_RUNTIME_FUSION_EVIDENCE_PUBLIC_CALL_CONFIG ||
           source == PTO_CUDA_RUNTIME_FUSION_EVIDENCE_PAYLOAD_PROVENANCE;
}

inline const char *pto_cuda_runtime_fusion_status_name(uint32_t status) {
    switch (status) {
        case PTO_CUDA_RUNTIME_FUSION_STATUS_UNSUPPORTED:
            return "unsupported";
        case PTO_CUDA_RUNTIME_FUSION_STATUS_SETUP_FAILED:
            return "setup_failed";
        case PTO_CUDA_RUNTIME_FUSION_STATUS_FAILED:
            return "failed";
        case PTO_CUDA_RUNTIME_FUSION_STATUS_PASSED:
            return "passed";
        default:
            return "unknown";
    }
}

inline const char *pto_cuda_runtime_fusion_failure_name(uint32_t failure) {
    switch (failure) {
        case PTO_CUDA_RUNTIME_FUSION_FAILURE_UNSUPPORTED_BOUNDARY:
            return "unsupported_boundary";
        case PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_COORDINATOR:
            return "missing_coordinator";
        case PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_DESCRIPTOR_ALLOCATOR:
            return "missing_descriptor_allocator";
        case PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_UCCL_EP_RUNTIME:
            return "missing_uccl_ep_runtime";
        case PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_VALIDATION_POLICY:
            return "missing_validation_policy";
        case PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_CHIP_STORAGE_TASK_ARGS:
            return "missing_chip_storage_task_args";
        case PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_PERSISTENT_GRAPH_DESCRIPTOR:
            return "missing_persistent_graph_descriptor";
        case PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_RANK_DEVICE_METADATA:
            return "missing_rank_device_metadata";
        case PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_UCCL_EP_CAPABILITY:
            return "missing_uccl_ep_capability";
        case PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_OUTPUT_SINK:
            return "missing_output_sink";
        case PTO_CUDA_RUNTIME_FUSION_FAILURE_FABRICATED_OR_UNTRUSTED_PASS_EVIDENCE:
            return "fabricated_or_untrusted_pass_evidence";
        case PTO_CUDA_RUNTIME_FUSION_FAILURE_STALE_DESCRIPTOR_VIEW:
            return "stale_descriptor_view";
        case PTO_CUDA_RUNTIME_FUSION_FAILURE_DESCRIPTOR_TOKEN_MISMATCH:
            return "descriptor_token_mismatch";
        case PTO_CUDA_RUNTIME_FUSION_FAILURE_RANK_DEVICE_MISMATCH:
            return "rank_device_mismatch";
        case PTO_CUDA_RUNTIME_FUSION_FAILURE_TRANSPORT_MODE_MISMATCH:
            return "transport_mode_mismatch";
        case PTO_CUDA_RUNTIME_FUSION_FAILURE_DESCRIPTOR_VOCABULARY_MISMATCH:
            return "descriptor_vocabulary_mismatch";
        case PTO_CUDA_RUNTIME_FUSION_FAILURE_PUBLIC_API_RUNTIME_PATH:
            return "public_api_runtime_path";
        case PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_RUNTIME_DISPATCH_SCAFFOLD:
            return "missing_runtime_dispatch_scaffold";
        case PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_RUNTIME_DISPATCH_HANDOFF_DRIVER:
            return "missing_runtime_dispatch_handoff_driver";
        default:
            return "unknown_failure";
    }
}

inline int pto_cuda_uccl_ep_runtime_path_source_is_forbidden(uint32_t source) {
    return source != PTO_CUDA_UCCL_EP_RUNTIME_PATH_SOURCE_COORDINATOR_OWNED;
}

inline int pto_cuda_runtime_fusion_allocate_uccl_ep_descriptors(
    const PtoCudaRuntimeFusionRequest *request, PtoCudaUcclEpDescriptorAllocation *allocation, void *device_buffer,
    size_t device_buffer_size, uint64_t shared_token
) {
    if (request == nullptr || allocation == nullptr || device_buffer == nullptr ||
        request->version != PTO_CUDA_RUNTIME_FUSION_REQUEST_VERSION || request->comm_descriptor == nullptr ||
        request->persistent_graph_descriptor == nullptr || shared_token == 0U ||
        device_buffer_size < sizeof(PtoCudaUcclEpDeviceDescriptorBuffer)) {
        return -1;
    }

    PtoCudaUcclEpDeviceDescriptorBuffer *typed_buffer =
        static_cast<PtoCudaUcclEpDeviceDescriptorBuffer *>(device_buffer);
    *allocation = {};
    *typed_buffer = {};

    typed_buffer->dispatch_descriptor.version = PTO_CUDA_UCCL_EP_DEVICE_DESCRIPTOR_BUFFER_VERSION;
    typed_buffer->dispatch_descriptor.descriptor_vocabulary =
        PTO_CUDA_RUNTIME_FUSION_DESCRIPTOR_VOCABULARY_DISPATCH;
    typed_buffer->dispatch_descriptor.invocation_id = request->invocation_id;
    typed_buffer->dispatch_descriptor.persistent_graph_descriptor = request->persistent_graph_descriptor;
    typed_buffer->dispatch_descriptor.capability_crc32 = request->comm_descriptor->capability_crc32;
    typed_buffer->dispatch_descriptor.rank = request->comm_descriptor->rank;
    typed_buffer->dispatch_descriptor.device_id = request->comm_descriptor->device_id;
    typed_buffer->dispatch_descriptor.world_size = request->comm_descriptor->world_size;
    typed_buffer->dispatch_descriptor.shared_token = shared_token;

    typed_buffer->combine_descriptor = typed_buffer->dispatch_descriptor;
    typed_buffer->combine_descriptor.descriptor_vocabulary =
        PTO_CUDA_RUNTIME_FUSION_DESCRIPTOR_VOCABULARY_COMBINE;

    allocation->host_control.version = PTO_CUDA_UCCL_EP_DESCRIPTOR_ALLOCATION_VERSION;
    allocation->host_control.runtime_owned = 1U;
    allocation->host_control.invocation_id = request->invocation_id;
    allocation->host_control.persistent_graph_descriptor = request->persistent_graph_descriptor;
    allocation->host_control.capability_crc32 = request->comm_descriptor->capability_crc32;
    allocation->host_control.rank = request->comm_descriptor->rank;
    allocation->host_control.device_id = request->comm_descriptor->device_id;
    allocation->host_control.world_size = request->comm_descriptor->world_size;
    allocation->host_control.descriptor_vocabulary =
        PTO_CUDA_RUNTIME_FUSION_DESCRIPTOR_VOCABULARY_DISPATCH |
        PTO_CUDA_RUNTIME_FUSION_DESCRIPTOR_VOCABULARY_COMBINE;
    allocation->host_control.allocation_state = 1U;
    allocation->host_control.shared_token = shared_token;
    allocation->host_control.device_buffer_size = device_buffer_size;
    allocation->host_control.dispatch_descriptor_offset =
        offsetof(PtoCudaUcclEpDeviceDescriptorBuffer, dispatch_descriptor);
    allocation->host_control.combine_descriptor_offset =
        offsetof(PtoCudaUcclEpDeviceDescriptorBuffer, combine_descriptor);
    allocation->device_buffer = typed_buffer;

    allocation->dispatch_descriptor.version = PTO_CUDA_UCCL_EP_RUNTIME_DESCRIPTOR_VIEW_VERSION;
    allocation->dispatch_descriptor.invocation_id = request->invocation_id;
    allocation->dispatch_descriptor.persistent_graph_descriptor = request->persistent_graph_descriptor;
    allocation->dispatch_descriptor.capability_crc32 = request->comm_descriptor->capability_crc32;
    allocation->dispatch_descriptor.rank = request->comm_descriptor->rank;
    allocation->dispatch_descriptor.device_id = request->comm_descriptor->device_id;
    allocation->dispatch_descriptor.world_size = request->comm_descriptor->world_size;
    allocation->dispatch_descriptor.descriptor_vocabulary =
        PTO_CUDA_RUNTIME_FUSION_DESCRIPTOR_VOCABULARY_DISPATCH;
    allocation->dispatch_descriptor.shared_token = shared_token;
    allocation->dispatch_descriptor.source = PTO_CUDA_UCCL_EP_RUNTIME_PATH_SOURCE_COORDINATOR_OWNED;

    allocation->combine_descriptor = allocation->dispatch_descriptor;
    allocation->combine_descriptor.descriptor_vocabulary =
        PTO_CUDA_RUNTIME_FUSION_DESCRIPTOR_VOCABULARY_COMBINE;

    allocation->runtime_path.version = PTO_CUDA_UCCL_EP_RUNTIME_PATH_VERSION;
    allocation->runtime_path.transport_mode = PTO_CUDA_UCCL_EP_TRANSPORT_MODE_EP;
    allocation->runtime_path.dispatch_descriptor = &allocation->dispatch_descriptor;
    allocation->runtime_path.combine_descriptor = &allocation->combine_descriptor;
    return 0;
}

inline uint32_t pto_cuda_runtime_fusion_validate_descriptor_view(
    const PtoCudaRuntimeFusionRequest *request, const PtoCudaUcclEpRuntimeDescriptorView *view,
    uint32_t expected_vocabulary
) {
    if (view == nullptr) {
        return PTO_CUDA_RUNTIME_FUSION_FAILURE_STALE_DESCRIPTOR_VIEW;
    }

    uint32_t failures = 0;
    if (view->version != PTO_CUDA_UCCL_EP_RUNTIME_DESCRIPTOR_VIEW_VERSION ||
        view->invocation_id != request->invocation_id ||
        view->persistent_graph_descriptor != request->persistent_graph_descriptor) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_STALE_DESCRIPTOR_VIEW;
    }
    if (view->descriptor_vocabulary != expected_vocabulary) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DESCRIPTOR_VOCABULARY_MISMATCH;
    }
    if (pto_cuda_uccl_ep_runtime_path_source_is_forbidden(view->source)) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_PUBLIC_API_RUNTIME_PATH |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_FABRICATED_OR_UNTRUSTED_PASS_EVIDENCE;
    }

    if (request->comm_descriptor != nullptr &&
        (view->capability_crc32 != request->comm_descriptor->capability_crc32 ||
         view->rank != request->comm_descriptor->rank || view->device_id != request->comm_descriptor->device_id ||
         view->world_size != request->comm_descriptor->world_size)) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_RANK_DEVICE_MISMATCH;
    }
    return failures;
}

inline uint32_t pto_cuda_runtime_fusion_validate_uccl_ep_runtime_path(
    const PtoCudaRuntimeFusionRequest *request, const PtoCudaUcclEpRuntimePath *runtime_path
) {
    if (runtime_path == nullptr) {
        return PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_UCCL_EP_RUNTIME;
    }

    uint32_t failures = 0;
    if (runtime_path->version != PTO_CUDA_UCCL_EP_RUNTIME_PATH_VERSION) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_STALE_DESCRIPTOR_VIEW;
    }
    if (runtime_path->transport_mode != PTO_CUDA_UCCL_EP_TRANSPORT_MODE_EP) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_TRANSPORT_MODE_MISMATCH;
    }

    const PtoCudaUcclEpRuntimeDescriptorView *dispatch = runtime_path->dispatch_descriptor;
    const PtoCudaUcclEpRuntimeDescriptorView *combine = runtime_path->combine_descriptor;
    failures |= pto_cuda_runtime_fusion_validate_descriptor_view(
        request, dispatch, PTO_CUDA_RUNTIME_FUSION_DESCRIPTOR_VOCABULARY_DISPATCH
    );
    failures |= pto_cuda_runtime_fusion_validate_descriptor_view(
        request, combine, PTO_CUDA_RUNTIME_FUSION_DESCRIPTOR_VOCABULARY_COMBINE
    );

    if (dispatch == nullptr || combine == nullptr || dispatch->shared_token == 0U ||
        dispatch->shared_token != combine->shared_token) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DESCRIPTOR_TOKEN_MISMATCH;
    }
    return failures;
}

inline int pto_cuda_runtime_fusion_failure_is_runtime_path_failed(uint32_t failures) {
    const uint32_t runtime_path_failed_mask =
        PTO_CUDA_RUNTIME_FUSION_FAILURE_STALE_DESCRIPTOR_VIEW |
        PTO_CUDA_RUNTIME_FUSION_FAILURE_DESCRIPTOR_TOKEN_MISMATCH |
        PTO_CUDA_RUNTIME_FUSION_FAILURE_RANK_DEVICE_MISMATCH |
        PTO_CUDA_RUNTIME_FUSION_FAILURE_TRANSPORT_MODE_MISMATCH |
        PTO_CUDA_RUNTIME_FUSION_FAILURE_DESCRIPTOR_VOCABULARY_MISMATCH |
        PTO_CUDA_RUNTIME_FUSION_FAILURE_PUBLIC_API_RUNTIME_PATH |
        PTO_CUDA_RUNTIME_FUSION_FAILURE_FABRICATED_OR_UNTRUSTED_PASS_EVIDENCE;
    return (failures & runtime_path_failed_mask) != 0U;
}

inline int pto_cuda_runtime_fusion_prepare_private_coordinator(
    const PtoCudaRuntimeFusionRequest *request, PtoCudaRuntimeFusionCoordinator *coordinator, void *device_buffer,
    size_t device_buffer_size, PtoCudaRuntimeFusionResult *output_sink
) {
    if (request == nullptr || coordinator == nullptr || output_sink == nullptr ||
        request->version != PTO_CUDA_RUNTIME_FUSION_REQUEST_VERSION || request->invocation_id == 0U) {
        return -1;
    }

    PtoCudaUcclEpDescriptorAllocation allocation = {};
    if (pto_cuda_runtime_fusion_allocate_uccl_ep_descriptors(
            request, &allocation, device_buffer, device_buffer_size, request->invocation_id
        ) != 0) {
        return -1;
    }

    *coordinator = {};
    coordinator->version = PTO_CUDA_RUNTIME_FUSION_COORDINATOR_VERSION;
    coordinator->invocation_id = request->invocation_id;
    coordinator->descriptor_allocation = allocation;
    coordinator->descriptor_allocation.runtime_path.dispatch_descriptor =
        &coordinator->descriptor_allocation.dispatch_descriptor;
    coordinator->descriptor_allocation.runtime_path.combine_descriptor =
        &coordinator->descriptor_allocation.combine_descriptor;
    coordinator->runtime_dispatch_scaffold_status.version =
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_SCAFFOLD_STATUS_VERSION;
    coordinator->runtime_dispatch_scaffold_status.invocation_id = request->invocation_id;
    coordinator->runtime_dispatch_scaffold_status.runtime_path =
        &coordinator->descriptor_allocation.runtime_path;
    coordinator->runtime_dispatch_scaffold_status.dispatch_eligible = 1U;
    coordinator->runtime_dispatch_scaffold_status.status = PTO_CUDA_RUNTIME_FUSION_STATUS_UNSUPPORTED;
    coordinator->runtime_dispatch_scaffold_status.failure_fields =
        PTO_CUDA_RUNTIME_FUSION_FAILURE_UNSUPPORTED_BOUNDARY;
    coordinator->runtime_dispatch_request_handoff_driver_state.version =
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_HANDOFF_DRIVER_STATE_VERSION;
    coordinator->runtime_dispatch_request_handoff_driver_state.invocation_id = request->invocation_id;
    coordinator->runtime_dispatch_request_handoff_driver_state.request_owner = coordinator;
    coordinator->runtime_dispatch_request_handoff_driver_state.runtime_dispatch_scaffold_status =
        &coordinator->runtime_dispatch_scaffold_status;
    coordinator->runtime_dispatch_request_handoff_driver_state.runtime_path =
        &coordinator->descriptor_allocation.runtime_path;
    coordinator->runtime_dispatch_request_handoff_driver_state.status =
        PTO_CUDA_RUNTIME_FUSION_STATUS_UNSUPPORTED;
    coordinator->runtime_dispatch_request_handoff_driver_state.failure_fields =
        PTO_CUDA_RUNTIME_FUSION_FAILURE_UNSUPPORTED_BOUNDARY;
    coordinator->runtime_dispatch_request_handoff_scaffold_status.version =
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_REQUEST_HANDOFF_SCAFFOLD_STATUS_VERSION;
    coordinator->runtime_dispatch_request_handoff_scaffold_status.invocation_id =
        request->invocation_id;
    coordinator->runtime_dispatch_request_handoff_scaffold_status.request_owner = coordinator;
    coordinator->runtime_dispatch_request_handoff_scaffold_status.runtime_dispatch_scaffold_status =
        &coordinator->runtime_dispatch_scaffold_status;
    coordinator->runtime_dispatch_request_handoff_scaffold_status.driver_state =
        &coordinator->runtime_dispatch_request_handoff_driver_state;
    coordinator->runtime_dispatch_request_handoff_scaffold_status.handoff_eligible = 1U;
    coordinator->runtime_dispatch_request_handoff_scaffold_status.status =
        PTO_CUDA_RUNTIME_FUSION_STATUS_UNSUPPORTED;
    coordinator->runtime_dispatch_request_handoff_scaffold_status.failure_fields =
        PTO_CUDA_RUNTIME_FUSION_FAILURE_UNSUPPORTED_BOUNDARY;
    coordinator->output_sink = output_sink;
    coordinator->status = PTO_CUDA_RUNTIME_FUSION_STATUS_UNSUPPORTED;
    coordinator->failure_fields = PTO_CUDA_RUNTIME_FUSION_FAILURE_UNSUPPORTED_BOUNDARY;
    return 0;
}

inline int pto_cuda_runtime_fusion_request_has_private_coordinator_shape(
    const PtoCudaRuntimeFusionRequest *request
) {
    if (request == nullptr || request->coordinator == nullptr || request->descriptor_allocator == nullptr ||
        request->uccl_ep_runtime == nullptr) {
        return 0;
    }

    const uintptr_t coordinator_base = reinterpret_cast<uintptr_t>(request->coordinator);
    const uintptr_t descriptor_allocation =
        coordinator_base + offsetof(PtoCudaRuntimeFusionCoordinator, descriptor_allocation);
    const uintptr_t runtime_path =
        descriptor_allocation + offsetof(PtoCudaUcclEpDescriptorAllocation, runtime_path);
    return reinterpret_cast<uintptr_t>(request->descriptor_allocator) == descriptor_allocation &&
           reinterpret_cast<uintptr_t>(request->uccl_ep_runtime) == runtime_path;
}

inline uint32_t pto_cuda_runtime_fusion_validate_private_coordinator(
    const PtoCudaRuntimeFusionRequest *request, const PtoCudaRuntimeFusionCoordinator *coordinator
) {
    if (request == nullptr) {
        return PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_COORDINATOR;
    }
    if (coordinator == nullptr) {
        return PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_COORDINATOR;
    }

    uint32_t failures = 0;
    if (coordinator->version != PTO_CUDA_RUNTIME_FUSION_COORDINATOR_VERSION ||
        coordinator->invocation_id != request->invocation_id) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_STALE_DESCRIPTOR_VIEW;
    }
    if (coordinator->output_sink == nullptr || coordinator->output_sink != request->output_sink) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_OUTPUT_SINK;
    }
    if (request->descriptor_allocator != &coordinator->descriptor_allocation ||
        request->uccl_ep_runtime != &coordinator->descriptor_allocation.runtime_path) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_STALE_DESCRIPTOR_VIEW;
    }
    failures |= pto_cuda_runtime_fusion_validate_uccl_ep_runtime_path(
        request, &coordinator->descriptor_allocation.runtime_path
    );
    const PtoCudaUcclEpRuntimeDispatchScaffoldStatus *dispatch_status =
        &coordinator->runtime_dispatch_scaffold_status;
    if (dispatch_status->version != PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_SCAFFOLD_STATUS_VERSION ||
        dispatch_status->invocation_id != request->invocation_id ||
        dispatch_status->runtime_path != &coordinator->descriptor_allocation.runtime_path ||
        dispatch_status->dispatch_eligible == 0U) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_RUNTIME_DISPATCH_SCAFFOLD;
    }
    if (dispatch_status->status == PTO_CUDA_RUNTIME_FUSION_STATUS_PASSED) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_FABRICATED_OR_UNTRUSTED_PASS_EVIDENCE;
    }
    const PtoCudaUcclEpRuntimeDispatchRequestHandoffScaffoldStatus *handoff_status =
        &coordinator->runtime_dispatch_request_handoff_scaffold_status;
    const PtoCudaUcclEpRuntimeDispatchHandoffDriverState *driver_state =
        handoff_status->driver_state;
    if (handoff_status->version !=
            PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_REQUEST_HANDOFF_SCAFFOLD_STATUS_VERSION ||
        handoff_status->invocation_id != request->invocation_id ||
        handoff_status->request_owner != coordinator ||
        handoff_status->runtime_dispatch_scaffold_status !=
            &coordinator->runtime_dispatch_scaffold_status ||
        handoff_status->handoff_eligible == 0U || driver_state == nullptr) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_RUNTIME_DISPATCH_HANDOFF_DRIVER;
    } else if (driver_state != &coordinator->runtime_dispatch_request_handoff_driver_state ||
               driver_state->version !=
                   PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_HANDOFF_DRIVER_STATE_VERSION ||
               driver_state->invocation_id != request->invocation_id ||
               driver_state->request_owner != coordinator ||
               driver_state->runtime_dispatch_scaffold_status !=
                   &coordinator->runtime_dispatch_scaffold_status ||
               driver_state->runtime_path != &coordinator->descriptor_allocation.runtime_path) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_RUNTIME_DISPATCH_HANDOFF_DRIVER;
    }
    if (handoff_status->status == PTO_CUDA_RUNTIME_FUSION_STATUS_PASSED ||
        (driver_state != nullptr && driver_state->status == PTO_CUDA_RUNTIME_FUSION_STATUS_PASSED)) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_FABRICATED_OR_UNTRUSTED_PASS_EVIDENCE;
    }
    return failures;
}

inline int pto_cuda_runtime_fusion_prepare_runtime_dispatch_scaffold_status(
    const PtoCudaRuntimeFusionRequest *request, PtoCudaRuntimeFusionCoordinator *coordinator
) {
    if (request == nullptr || coordinator == nullptr ||
        coordinator->version != PTO_CUDA_RUNTIME_FUSION_COORDINATOR_VERSION ||
        coordinator->invocation_id != request->invocation_id) {
        return -1;
    }

    coordinator->runtime_dispatch_scaffold_status = {};
    coordinator->runtime_dispatch_scaffold_status.version =
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_SCAFFOLD_STATUS_VERSION;
    coordinator->runtime_dispatch_scaffold_status.invocation_id = request->invocation_id;
    coordinator->runtime_dispatch_scaffold_status.runtime_path =
        &coordinator->descriptor_allocation.runtime_path;
    coordinator->runtime_dispatch_scaffold_status.dispatch_eligible = 1U;
    coordinator->runtime_dispatch_scaffold_status.status = PTO_CUDA_RUNTIME_FUSION_STATUS_UNSUPPORTED;
    coordinator->runtime_dispatch_scaffold_status.failure_fields =
        PTO_CUDA_RUNTIME_FUSION_FAILURE_UNSUPPORTED_BOUNDARY;
    return 0;
}

inline int pto_cuda_runtime_fusion_prepare_runtime_dispatch_request_handoff_scaffold_status(
    const PtoCudaRuntimeFusionRequest *request, PtoCudaRuntimeFusionCoordinator *coordinator
) {
    if (request == nullptr || coordinator == nullptr ||
        coordinator->version != PTO_CUDA_RUNTIME_FUSION_COORDINATOR_VERSION ||
        coordinator->invocation_id != request->invocation_id) {
        return -1;
    }

    coordinator->runtime_dispatch_request_handoff_driver_state = {};
    coordinator->runtime_dispatch_request_handoff_driver_state.version =
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_HANDOFF_DRIVER_STATE_VERSION;
    coordinator->runtime_dispatch_request_handoff_driver_state.invocation_id =
        request->invocation_id;
    coordinator->runtime_dispatch_request_handoff_driver_state.request_owner = coordinator;
    coordinator->runtime_dispatch_request_handoff_driver_state.runtime_dispatch_scaffold_status =
        &coordinator->runtime_dispatch_scaffold_status;
    coordinator->runtime_dispatch_request_handoff_driver_state.runtime_path =
        &coordinator->descriptor_allocation.runtime_path;
    coordinator->runtime_dispatch_request_handoff_driver_state.status =
        PTO_CUDA_RUNTIME_FUSION_STATUS_UNSUPPORTED;
    coordinator->runtime_dispatch_request_handoff_driver_state.failure_fields =
        PTO_CUDA_RUNTIME_FUSION_FAILURE_UNSUPPORTED_BOUNDARY;

    coordinator->runtime_dispatch_request_handoff_scaffold_status = {};
    coordinator->runtime_dispatch_request_handoff_scaffold_status.version =
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_REQUEST_HANDOFF_SCAFFOLD_STATUS_VERSION;
    coordinator->runtime_dispatch_request_handoff_scaffold_status.invocation_id =
        request->invocation_id;
    coordinator->runtime_dispatch_request_handoff_scaffold_status.request_owner = coordinator;
    coordinator->runtime_dispatch_request_handoff_scaffold_status.runtime_dispatch_scaffold_status =
        &coordinator->runtime_dispatch_scaffold_status;
    coordinator->runtime_dispatch_request_handoff_scaffold_status.driver_state =
        &coordinator->runtime_dispatch_request_handoff_driver_state;
    coordinator->runtime_dispatch_request_handoff_scaffold_status.handoff_eligible = 1U;
    coordinator->runtime_dispatch_request_handoff_scaffold_status.status =
        PTO_CUDA_RUNTIME_FUSION_STATUS_UNSUPPORTED;
    coordinator->runtime_dispatch_request_handoff_scaffold_status.failure_fields =
        PTO_CUDA_RUNTIME_FUSION_FAILURE_UNSUPPORTED_BOUNDARY;
    return 0;
}

inline int pto_cuda_runtime_fusion_failure_is_runtime_dispatch_scaffold_failed(uint32_t failures) {
    return (failures & PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_RUNTIME_DISPATCH_SCAFFOLD) != 0U;
}

inline int pto_cuda_runtime_fusion_failure_is_runtime_dispatch_handoff_failed(uint32_t failures) {
    return (failures & PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_RUNTIME_DISPATCH_HANDOFF_DRIVER) != 0U;
}

inline int persistent_device_uccl_ep_runtime_fusion_entry(
    const PtoCudaRuntimeFusionRequest *request, PtoCudaRuntimeFusionResult *result
) {
    if (request == nullptr || result == nullptr ||
        request->version != PTO_CUDA_RUNTIME_FUSION_REQUEST_VERSION) {
        return -1;
    }

    PtoCudaRuntimeFusionResult out = {};
    out.version = PTO_CUDA_RUNTIME_FUSION_RESULT_VERSION;
    out.status = PTO_CUDA_RUNTIME_FUSION_STATUS_UNSUPPORTED;
    out.actual_fused_cross_gpu_execution = 0;
    out.callable_id = request->callable_id;
    out.producer = "persistent_device_uccl_ep_runtime_fusion";
    out.entry_name = "persistent_device_uccl_ep_runtime_fusion_entry";
    out.reason = "persistent_device_uccl_ep_runtime_fusion is not implemented";

    if (request->comm_descriptor != nullptr) {
        out.rank = request->comm_descriptor->rank;
        out.device_id = request->comm_descriptor->device_id;
        out.world_size = request->comm_descriptor->world_size;
        out.capability_crc32 = request->comm_descriptor->capability_crc32;
    } else {
        out.failure_fields |= PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_RANK_DEVICE_METADATA;
    }

    if (request->chip_storage_task_args == nullptr ||
        request->chip_storage_task_args_size != sizeof(ChipStorageTaskArgs)) {
        out.failure_fields |= PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_CHIP_STORAGE_TASK_ARGS;
    }
    if (request->persistent_graph_descriptor == nullptr) {
        out.failure_fields |= PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_PERSISTENT_GRAPH_DESCRIPTOR;
    }
    if (request->uccl_ep_capability_metadata == nullptr) {
        out.failure_fields |= PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_UCCL_EP_CAPABILITY;
    }
    if (!pto_cuda_runtime_fusion_request_has_private_coordinator_shape(request)) {
        out.failure_fields |= PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_COORDINATOR;
    } else {
        const PtoCudaRuntimeFusionCoordinator *coordinator =
            static_cast<const PtoCudaRuntimeFusionCoordinator *>(request->coordinator);
        out.failure_fields |= pto_cuda_runtime_fusion_validate_private_coordinator(request, coordinator);
    }
    if (request->descriptor_allocator == nullptr) {
        out.failure_fields |= PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_DESCRIPTOR_ALLOCATOR;
    }
    if (request->uccl_ep_runtime == nullptr) {
        out.failure_fields |= PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_UCCL_EP_RUNTIME;
    } else {
        const PtoCudaUcclEpRuntimePath *runtime_path =
            static_cast<const PtoCudaUcclEpRuntimePath *>(request->uccl_ep_runtime);
        out.failure_fields |= pto_cuda_runtime_fusion_validate_uccl_ep_runtime_path(request, runtime_path);
    }
    if (request->validation_policy == nullptr) {
        out.failure_fields |= PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_VALIDATION_POLICY;
    }
    if (request->output_sink == nullptr) {
        out.failure_fields |= PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_OUTPUT_SINK;
    }

    if (pto_cuda_runtime_fusion_evidence_source_is_forbidden(request->pass_evidence_source)) {
        out.status = PTO_CUDA_RUNTIME_FUSION_STATUS_FAILED;
        out.failure_fields |= PTO_CUDA_RUNTIME_FUSION_FAILURE_FABRICATED_OR_UNTRUSTED_PASS_EVIDENCE;
        out.reason = "forbidden source attempted to provide runtime fusion pass evidence";
    } else if (pto_cuda_runtime_fusion_failure_is_runtime_dispatch_scaffold_failed(out.failure_fields)) {
        out.status = PTO_CUDA_RUNTIME_FUSION_STATUS_FAILED;
        out.reason = "private UCCL-EP runtime dispatch scaffold/status gate validation failed";
    } else if (pto_cuda_runtime_fusion_failure_is_runtime_dispatch_handoff_failed(out.failure_fields)) {
        out.status = PTO_CUDA_RUNTIME_FUSION_STATUS_FAILED;
        out.reason = "private UCCL-EP runtime dispatch request/driver handoff validation failed";
    } else if (pto_cuda_runtime_fusion_failure_is_runtime_path_failed(out.failure_fields)) {
        out.status = PTO_CUDA_RUNTIME_FUSION_STATUS_FAILED;
        out.reason = "private UCCL-EP runtime path validation failed";
    } else {
        out.failure_fields |= PTO_CUDA_RUNTIME_FUSION_FAILURE_UNSUPPORTED_BOUNDARY;
    }

    if (pto_cuda_runtime_fusion_request_has_private_coordinator_shape(request)) {
        const PtoCudaRuntimeFusionCoordinator *coordinator =
            static_cast<const PtoCudaRuntimeFusionCoordinator *>(request->coordinator);
        if (coordinator->version == PTO_CUDA_RUNTIME_FUSION_COORDINATOR_VERSION &&
            coordinator->output_sink == request->output_sink && coordinator->output_sink != nullptr) {
            *coordinator->output_sink = out;
        }
    }
    *result = out;
    return 0;
}

#endif  // SRC_CUDA_PLATFORM_INCLUDE_HOST_PTO_CUDA_RUNTIME_FUSION_ABI_H_
