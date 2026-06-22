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

#include <stddef.h>
#include <stdint.h>

static const uint32_t PTO_CUDA_RUNTIME_FUSION_REQUEST_VERSION = 1;
static const uint32_t PTO_CUDA_RUNTIME_FUSION_RESULT_VERSION = 1;

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
};

struct PtoCudaRuntimeFusionRequest {
    uint32_t version;
    int32_t callable_id;
    const void *chip_storage_task_args;
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
        default:
            return "unknown_failure";
    }
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

    if (request->chip_storage_task_args == nullptr || request->chip_storage_task_args_size == 0U) {
        out.failure_fields |= PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_CHIP_STORAGE_TASK_ARGS;
    }
    if (request->persistent_graph_descriptor == nullptr) {
        out.failure_fields |= PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_PERSISTENT_GRAPH_DESCRIPTOR;
    }
    if (request->uccl_ep_capability_metadata == nullptr) {
        out.failure_fields |= PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_UCCL_EP_CAPABILITY;
    }
    if (request->coordinator == nullptr) {
        out.failure_fields |= PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_COORDINATOR;
    }
    if (request->descriptor_allocator == nullptr) {
        out.failure_fields |= PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_DESCRIPTOR_ALLOCATOR;
    }
    if (request->uccl_ep_runtime == nullptr) {
        out.failure_fields |= PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_UCCL_EP_RUNTIME;
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
    } else {
        out.failure_fields |= PTO_CUDA_RUNTIME_FUSION_FAILURE_UNSUPPORTED_BOUNDARY;
    }

    *result = out;
    return 0;
}

#endif  // SRC_CUDA_PLATFORM_INCLUDE_HOST_PTO_CUDA_RUNTIME_FUSION_ABI_H_
