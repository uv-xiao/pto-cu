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
static const uint32_t PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_SCAFFOLD_STATUS_VERSION = 1;
static const uint32_t PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_SCAFFOLD_STATUS_VERSION = 1;
static const uint32_t PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_REQUEST_SCAFFOLD_STATUS_VERSION = 1;
static const uint32_t PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_DISPATCH_REQUEST_SCAFFOLD_STATUS_VERSION = 1;
static const uint32_t PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_REQUEST_SCAFFOLD_STATUS_VERSION = 1;
static const uint32_t PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD_STATUS_VERSION = 1;
static const uint32_t
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_SCAFFOLD_STATUS_VERSION = 1;
static const uint32_t
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_SCAFFOLD_STATUS_VERSION = 1;
static const uint32_t
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_SCAFFOLD_STATUS_VERSION = 1;
static const uint32_t
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_SCAFFOLD_STATUS_VERSION = 1;

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
    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH = 1U << 19U,
    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_INVOCATION_MISMATCH = 1U << 20U,
    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RUNTIME_PATH_MISMATCH = 1U << 21U,
    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_DESCRIPTOR_TOKEN_MISMATCH = 1U << 22U,
    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RANK_DEVICE_MISMATCH = 1U << 23U,
    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_STATUS_SINK_MISMATCH = 1U << 24U,
    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_PUBLIC_API_SOURCED_STATE = 1U << 25U,
    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_FABRICATED_PASS_EVIDENCE = 1U << 26U,
    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_SCAFFOLD = 1U << 27U,
    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_REQUEST_SCAFFOLD = 1U << 28U,
    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_DISPATCH_REQUEST_SCAFFOLD = 1U << 29U,
    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_REQUEST_SCAFFOLD = 1U << 30U,
    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD = 1U << 31U,
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

enum PtoCudaUcclEpRuntimeDispatchDriverStatus : uint32_t {
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_STATUS_MISSING = 1,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_STATUS_STALE = 2,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_STATUS_NOT_BOUND_TO_HANDOFF = 3,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_STATUS_NO_DISPATCH_BACKEND = 4,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_STATUS_NO_COMBINE_BACKEND = 5,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_STATUS_UNSUPPORTED_BOUNDARY = 6,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_STATUS_OWNER_MISMATCH = 7,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_STATUS_INVOCATION_MISMATCH = 8,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_STATUS_RUNTIME_PATH_MISMATCH = 9,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_STATUS_DESCRIPTOR_TOKEN_MISMATCH = 10,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_STATUS_RANK_DEVICE_MISMATCH = 11,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_STATUS_STATUS_SINK_MISMATCH = 12,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_STATUS_PUBLIC_API_SOURCED_STATE = 13,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_STATUS_FABRICATED_PASS_EVIDENCE = 14,
};

enum PtoCudaUcclEpRuntimeDispatchDriverBackendStatus : uint32_t {
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_STATUS_REQUEST_UNBOUND = 1,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_STATUS_DISPATCH_BACKEND_PLACEHOLDER = 2,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_STATUS_COMBINE_BACKEND_PLACEHOLDER = 3,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_STATUS_STATUS_SINK_UNBOUND = 4,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_STATUS_UNSUPPORTED_BOUNDARY = 5,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_STATUS_OWNER_MISMATCH = 6,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_STATUS_INVOCATION_MISMATCH = 7,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_STATUS_RUNTIME_PATH_MISMATCH = 8,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_STATUS_DESCRIPTOR_TOKEN_MISMATCH = 9,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_STATUS_RANK_DEVICE_MISMATCH = 10,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_STATUS_STATUS_SINK_MISMATCH = 11,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_STATUS_PUBLIC_API_SOURCED_STATE = 12,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_STATUS_FABRICATED_PASS_EVIDENCE = 13,
};

enum PtoCudaUcclEpRuntimeDispatchDriverBackendRequestStatus : uint32_t {
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_REQUEST_STATUS_PENDING = 1,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_REQUEST_STATUS_DISPATCH_REQUEST_PLACEHOLDER = 2,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_REQUEST_STATUS_COMBINE_REQUEST_PLACEHOLDER = 3,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_REQUEST_STATUS_STATUS_SINK_UNBOUND = 4,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_REQUEST_STATUS_UNSUPPORTED_BOUNDARY = 5,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_REQUEST_STATUS_OWNER_MISMATCH = 6,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_REQUEST_STATUS_INVOCATION_MISMATCH = 7,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_REQUEST_STATUS_RUNTIME_PATH_MISMATCH = 8,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_REQUEST_STATUS_DESCRIPTOR_TOKEN_MISMATCH = 9,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_REQUEST_STATUS_RANK_DEVICE_MISMATCH = 10,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_REQUEST_STATUS_STATUS_SINK_MISMATCH = 11,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_REQUEST_STATUS_PUBLIC_API_SOURCED_STATE = 12,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_REQUEST_STATUS_PROVENANCE_SOURCED_STATE = 13,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_REQUEST_STATUS_FABRICATED_PASS_EVIDENCE = 14,
};

enum PtoCudaUcclEpRuntimeDispatchDriverBackendDispatchRequestStatus : uint32_t {
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_DISPATCH_REQUEST_STATUS_PENDING = 1,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_DISPATCH_REQUEST_STATUS_PAYLOAD_DESCRIPTOR_PLACEHOLDER = 2,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_DISPATCH_REQUEST_STATUS_OUTPUT_STATUS_SINK_UNBOUND = 3,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_DISPATCH_REQUEST_STATUS_UNSUPPORTED_BOUNDARY = 4,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_DISPATCH_REQUEST_STATUS_PAYLOAD_TRANSFER_UNIMPLEMENTED = 5,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_DISPATCH_REQUEST_STATUS_OWNER_MISMATCH = 6,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_DISPATCH_REQUEST_STATUS_INVOCATION_MISMATCH = 7,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_DISPATCH_REQUEST_STATUS_REQUEST_SCAFFOLD_MISMATCH = 8,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_DISPATCH_REQUEST_STATUS_DESCRIPTOR_TOKEN_MISMATCH = 9,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_DISPATCH_REQUEST_STATUS_RANK_DEVICE_MISMATCH = 10,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_DISPATCH_REQUEST_STATUS_STATUS_SINK_MISMATCH = 11,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_DISPATCH_REQUEST_STATUS_PUBLIC_API_SOURCED_STATE = 12,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_DISPATCH_REQUEST_STATUS_PROVENANCE_SOURCED_STATE = 13,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_DISPATCH_REQUEST_STATUS_FABRICATED_PASS_EVIDENCE = 14,
};

enum PtoCudaUcclEpRuntimeDispatchDriverBackendCombineRequestStatus : uint32_t {
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_REQUEST_STATUS_PENDING = 1,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_REQUEST_STATUS_PAYLOAD_DESCRIPTOR_PLACEHOLDER = 2,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_REQUEST_STATUS_OUTPUT_STATUS_SINK_UNBOUND = 3,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_REQUEST_STATUS_UNSUPPORTED_BOUNDARY = 4,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_REQUEST_STATUS_PAYLOAD_TRANSFER_UNIMPLEMENTED = 5,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_REQUEST_STATUS_OWNER_MISMATCH = 6,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_REQUEST_STATUS_INVOCATION_MISMATCH = 7,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_REQUEST_STATUS_REQUEST_SCAFFOLD_MISMATCH = 8,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_REQUEST_STATUS_DESCRIPTOR_TOKEN_MISMATCH = 9,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_REQUEST_STATUS_RANK_DEVICE_MISMATCH = 10,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_REQUEST_STATUS_STATUS_SINK_MISMATCH = 11,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_REQUEST_STATUS_PUBLIC_API_SOURCED_STATE = 12,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_REQUEST_STATUS_PROVENANCE_SOURCED_STATE = 13,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_REQUEST_STATUS_FABRICATED_PASS_EVIDENCE = 14,
};

enum PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadStatus : uint32_t {
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_STATUS_PENDING = 1,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_STATUS_DESCRIPTOR_PLACEHOLDER = 2,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_STATUS_OUTPUT_STATUS_SINK_UNBOUND = 3,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_STATUS_UNSUPPORTED_BOUNDARY = 4,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_STATUS_TRANSFER_UNIMPLEMENTED = 5,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_STATUS_OWNER_MISMATCH = 6,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_STATUS_INVOCATION_MISMATCH = 7,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_STATUS_REQUEST_SCAFFOLD_MISMATCH = 8,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_STATUS_DESCRIPTOR_TOKEN_MISMATCH = 9,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_STATUS_RANK_DEVICE_MISMATCH = 10,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_STATUS_STATUS_SINK_MISMATCH = 11,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_STATUS_PUBLIC_API_SOURCED_STATE = 12,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_STATUS_PROVENANCE_SOURCED_STATE = 13,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_STATUS_FABRICATED_PASS_EVIDENCE = 14,
};

enum PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadTransferStatus : uint32_t {
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_PENDING = 1,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_TRANSFER_UNIMPLEMENTED = 2,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_OUTPUT_STATUS_SINK_UNBOUND = 3,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_UNSUPPORTED_BOUNDARY = 4,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_OWNER_MISMATCH = 5,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_INVOCATION_MISMATCH = 6,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_REQUEST_SCAFFOLD_MISMATCH = 7,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_DISPATCH_REQUEST_SCAFFOLD_MISMATCH = 8,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_COMBINE_REQUEST_SCAFFOLD_MISMATCH = 9,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_PAYLOAD_SCAFFOLD_MISMATCH = 10,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_DESCRIPTOR_TOKEN_MISMATCH = 11,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_RANK_DEVICE_MISMATCH = 12,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_STATUS_SINK_MISMATCH = 13,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_PUBLIC_API_SOURCED_STATE = 14,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_PROVENANCE_SOURCED_STATE = 15,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_FABRICATED_PASS_EVIDENCE = 16,
};

enum PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadTransferCompletionStatus : uint32_t {
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_STATUS_PENDING = 1,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_STATUS_COMPLETION_UNIMPLEMENTED = 2,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_STATUS_STATUS_SINK_UNBOUND = 3,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_STATUS_UNSUPPORTED_BOUNDARY = 4,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_STATUS_OWNER_MISMATCH = 5,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_STATUS_INVOCATION_MISMATCH = 6,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_STATUS_TRANSFER_SCAFFOLD_MISMATCH = 7,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_STATUS_DESCRIPTOR_TOKEN_MISMATCH = 8,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_STATUS_RANK_DEVICE_MISMATCH = 9,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_STATUS_STATUS_SINK_MISMATCH = 10,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_STATUS_PUBLIC_API_SOURCED_STATE = 11,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_STATUS_PROVENANCE_SOURCED_STATE = 12,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_STATUS_FABRICATED_PASS_EVIDENCE = 13,
};

enum PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadTransferCompletionHandoffStatus : uint32_t {
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_PENDING = 1,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_HANDOFF_UNIMPLEMENTED = 2,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_STATUS_SINK_UNBOUND = 3,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_UNSUPPORTED_BOUNDARY = 4,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_OWNER_MISMATCH = 5,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_INVOCATION_MISMATCH = 6,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_COMPLETION_SCAFFOLD_MISMATCH = 7,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_TRANSFER_SCAFFOLD_MISMATCH = 8,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_DESCRIPTOR_TOKEN_MISMATCH = 9,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_RANK_DEVICE_MISMATCH = 10,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_STATUS_SINK_MISMATCH = 11,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_PUBLIC_API_SOURCED_STATE = 12,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_PROVENANCE_SOURCED_STATE = 13,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_FABRICATED_PASS_EVIDENCE = 14,
};

enum PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadTransferCompletionHandoffResultStatus
    : uint32_t {
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_PENDING = 1,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_RESULT_UNIMPLEMENTED = 2,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_STATUS_SINK_UNBOUND = 3,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_RESULT_SINK_UNBOUND = 4,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_UNSUPPORTED_BOUNDARY = 5,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_OWNER_MISMATCH = 6,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_INVOCATION_MISMATCH = 7,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_BACKEND_REQUEST_SCAFFOLD_MISMATCH = 8,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_DISPATCH_REQUEST_SCAFFOLD_MISMATCH = 9,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_COMBINE_REQUEST_SCAFFOLD_MISMATCH = 10,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_PAYLOAD_SCAFFOLD_MISMATCH = 11,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_TRANSFER_SCAFFOLD_MISMATCH = 12,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_COMPLETION_SCAFFOLD_MISMATCH = 13,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_HANDOFF_SCAFFOLD_MISMATCH = 14,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_DESCRIPTOR_TOKEN_MISMATCH = 15,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_RANK_DEVICE_MISMATCH = 16,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_STATUS_SINK_MISMATCH = 17,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_RESULT_SINK_MISMATCH = 18,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_PUBLIC_API_SOURCED_STATE = 19,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_PROVENANCE_SOURCED_STATE = 20,
    PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_FABRICATED_PASS_EVIDENCE = 21,
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

struct PtoCudaUcclEpRuntimeDispatchDriverScaffoldStatus {
    uint32_t version;
    uint64_t invocation_id;
    const void *driver_owner;
    const PtoCudaUcclEpRuntimeDispatchRequestHandoffScaffoldStatus *handoff_status;
    const PtoCudaUcclEpRuntimeDispatchHandoffDriverState *driver_state;
    const PtoCudaUcclEpRuntimePath *runtime_path;
    PtoCudaRuntimeFusionResult *output_sink;
    const void *dispatch_backend;
    const void *combine_backend;
    uint32_t status;
    uint32_t failure_fields;
};

struct PtoCudaUcclEpRuntimeDispatchDriverBackendScaffoldStatus {
    uint32_t version;
    uint64_t invocation_id;
    const void *backend_owner;
    const PtoCudaUcclEpRuntimeDispatchDriverScaffoldStatus *driver_status;
    const PtoCudaUcclEpRuntimePath *runtime_path;
    PtoCudaRuntimeFusionResult *status_sink;
    const void *dispatch_backend;
    const void *combine_backend;
    uint64_t shared_token;
    uint32_t rank;
    uint32_t device_id;
    uint32_t world_size;
    uint32_t status;
    uint32_t failure_fields;
};

struct PtoCudaUcclEpRuntimeDispatchDriverBackendRequestScaffoldStatus {
    uint32_t version;
    uint64_t invocation_id;
    const void *request_owner;
    const PtoCudaUcclEpRuntimeDispatchDriverBackendScaffoldStatus *backend_status;
    const PtoCudaUcclEpRuntimePath *runtime_path;
    PtoCudaRuntimeFusionResult *status_sink;
    const void *dispatch_request;
    const void *combine_request;
    uint64_t shared_token;
    uint32_t rank;
    uint32_t device_id;
    uint32_t world_size;
    uint32_t status;
    uint32_t failure_fields;
};

struct PtoCudaUcclEpRuntimeDispatchDriverBackendDispatchRequestScaffoldStatus {
    uint32_t version;
    uint64_t invocation_id;
    const void *request_owner;
    const PtoCudaUcclEpRuntimeDispatchDriverBackendRequestScaffoldStatus *backend_request_status;
    const PtoCudaUcclEpRuntimePath *runtime_path;
    PtoCudaRuntimeFusionResult *status_sink;
    const void *dispatch_payload_descriptor;
    uint64_t shared_token;
    uint32_t rank;
    uint32_t device_id;
    uint32_t world_size;
    uint32_t status;
    uint32_t failure_fields;
};

struct PtoCudaUcclEpRuntimeDispatchDriverBackendCombineRequestScaffoldStatus {
    uint32_t version;
    uint64_t invocation_id;
    const void *request_owner;
    const PtoCudaUcclEpRuntimeDispatchDriverBackendRequestScaffoldStatus *backend_request_status;
    const PtoCudaUcclEpRuntimeDispatchDriverBackendDispatchRequestScaffoldStatus *dispatch_request_status;
    const PtoCudaUcclEpRuntimePath *runtime_path;
    PtoCudaRuntimeFusionResult *status_sink;
    const void *combine_payload_descriptor;
    uint64_t shared_token;
    uint32_t rank;
    uint32_t device_id;
    uint32_t world_size;
    uint32_t status;
    uint32_t failure_fields;
};

struct PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadScaffoldStatus {
    uint32_t version;
    uint64_t invocation_id;
    const void *payload_owner;
    const PtoCudaUcclEpRuntimeDispatchDriverBackendCombineRequestScaffoldStatus *combine_request_status;
    const PtoCudaUcclEpRuntimePath *runtime_path;
    PtoCudaRuntimeFusionResult *status_sink;
    const void *combine_payload_descriptor;
    uint64_t shared_token;
    uint32_t rank;
    uint32_t device_id;
    uint32_t world_size;
    uint32_t status;
    uint32_t failure_fields;
};

struct PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadTransferScaffoldStatus {
    uint32_t version;
    uint64_t invocation_id;
    const void *transfer_owner;
    const PtoCudaUcclEpRuntimeDispatchDriverBackendRequestScaffoldStatus *backend_request_status;
    const PtoCudaUcclEpRuntimeDispatchDriverBackendDispatchRequestScaffoldStatus *dispatch_request_status;
    const PtoCudaUcclEpRuntimeDispatchDriverBackendCombineRequestScaffoldStatus *combine_request_status;
    const PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadScaffoldStatus *combine_payload_status;
    const PtoCudaUcclEpRuntimePath *runtime_path;
    PtoCudaRuntimeFusionResult *status_sink;
    const void *combine_payload_descriptor;
    uint64_t shared_token;
    uint32_t rank;
    uint32_t device_id;
    uint32_t world_size;
    uint32_t status;
    uint32_t failure_fields;
};

struct PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadTransferCompletionScaffoldStatus {
    uint32_t version;
    uint64_t invocation_id;
    const void *completion_owner;
    const PtoCudaUcclEpRuntimeDispatchDriverBackendRequestScaffoldStatus *backend_request_status;
    const PtoCudaUcclEpRuntimeDispatchDriverBackendDispatchRequestScaffoldStatus *dispatch_request_status;
    const PtoCudaUcclEpRuntimeDispatchDriverBackendCombineRequestScaffoldStatus *combine_request_status;
    const PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadScaffoldStatus *combine_payload_status;
    const PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadTransferScaffoldStatus *transfer_status;
    const PtoCudaUcclEpRuntimePath *runtime_path;
    PtoCudaRuntimeFusionResult *status_sink;
    PtoCudaRuntimeFusionResult *completion_sink;
    const void *combine_payload_descriptor;
    uint64_t shared_token;
    uint32_t rank;
    uint32_t device_id;
    uint32_t world_size;
    uint32_t status;
    uint32_t failure_fields;
};

struct PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadTransferCompletionHandoffScaffoldStatus {
    uint32_t version;
    uint64_t invocation_id;
    const void *handoff_owner;
    const PtoCudaUcclEpRuntimeDispatchDriverBackendRequestScaffoldStatus *backend_request_status;
    const PtoCudaUcclEpRuntimeDispatchDriverBackendDispatchRequestScaffoldStatus *dispatch_request_status;
    const PtoCudaUcclEpRuntimeDispatchDriverBackendCombineRequestScaffoldStatus *combine_request_status;
    const PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadScaffoldStatus *combine_payload_status;
    const PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadTransferScaffoldStatus *transfer_status;
    const PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadTransferCompletionScaffoldStatus *completion_status;
    const PtoCudaUcclEpRuntimePath *runtime_path;
    PtoCudaRuntimeFusionResult *status_sink;
    PtoCudaRuntimeFusionResult *handoff_sink;
    const void *combine_payload_descriptor;
    uint64_t shared_token;
    uint32_t rank;
    uint32_t device_id;
    uint32_t world_size;
    uint32_t status;
    uint32_t failure_fields;
};

struct PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadTransferCompletionHandoffResultScaffoldStatus {
    uint32_t version;
    uint64_t invocation_id;
    const void *result_owner;
    const PtoCudaUcclEpRuntimeDispatchDriverBackendRequestScaffoldStatus *backend_request_status;
    const PtoCudaUcclEpRuntimeDispatchDriverBackendDispatchRequestScaffoldStatus *dispatch_request_status;
    const PtoCudaUcclEpRuntimeDispatchDriverBackendCombineRequestScaffoldStatus *combine_request_status;
    const PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadScaffoldStatus *combine_payload_status;
    const PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadTransferScaffoldStatus *transfer_status;
    const PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadTransferCompletionScaffoldStatus *completion_status;
    const PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadTransferCompletionHandoffScaffoldStatus *handoff_status;
    const PtoCudaUcclEpRuntimePath *runtime_path;
    PtoCudaRuntimeFusionResult *status_sink;
    PtoCudaRuntimeFusionResult *result_sink;
    const void *combine_payload_descriptor;
    uint64_t shared_token;
    uint32_t rank;
    uint32_t device_id;
    uint32_t world_size;
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
    PtoCudaUcclEpRuntimeDispatchDriverScaffoldStatus runtime_dispatch_driver_scaffold_status;
    PtoCudaUcclEpRuntimeDispatchDriverBackendScaffoldStatus
        runtime_dispatch_driver_backend_scaffold_status;
    PtoCudaUcclEpRuntimeDispatchDriverBackendRequestScaffoldStatus
        runtime_dispatch_driver_backend_request_scaffold_status;
    PtoCudaUcclEpRuntimeDispatchDriverBackendDispatchRequestScaffoldStatus
        runtime_dispatch_driver_backend_dispatch_request_scaffold_status;
    PtoCudaUcclEpRuntimeDispatchDriverBackendCombineRequestScaffoldStatus
        runtime_dispatch_driver_backend_combine_request_scaffold_status;
    PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadScaffoldStatus
        runtime_dispatch_driver_backend_combine_payload_scaffold_status;
    PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadTransferScaffoldStatus
        runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status;
    PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadTransferCompletionScaffoldStatus
        runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status;
    PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadTransferCompletionHandoffScaffoldStatus
        runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status;
    PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadTransferCompletionHandoffResultScaffoldStatus
        runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status;
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
        case PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH:
            return "driver_owner_mismatch";
        case PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_INVOCATION_MISMATCH:
            return "driver_invocation_mismatch";
        case PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RUNTIME_PATH_MISMATCH:
            return "driver_runtime_path_mismatch";
        case PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_DESCRIPTOR_TOKEN_MISMATCH:
            return "driver_descriptor_token_mismatch";
        case PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RANK_DEVICE_MISMATCH:
            return "driver_rank_device_mismatch";
        case PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_STATUS_SINK_MISMATCH:
            return "driver_status_sink_mismatch";
        case PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_PUBLIC_API_SOURCED_STATE:
            return "driver_public_api_sourced_state";
        case PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_FABRICATED_PASS_EVIDENCE:
            return "driver_fabricated_pass_evidence";
        case PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_SCAFFOLD:
            return "driver_backend_scaffold_status";
        case PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_REQUEST_SCAFFOLD:
            return "driver_backend_request_scaffold_status";
        case PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_DISPATCH_REQUEST_SCAFFOLD:
            return "driver_backend_dispatch_request_scaffold_status";
        case PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_REQUEST_SCAFFOLD:
            return "driver_backend_combine_request_scaffold_status";
        case PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD:
            return "driver_backend_combine_payload_scaffold_status";
        default:
            return "unknown_failure";
    }
}

inline const char *pto_cuda_uccl_ep_runtime_dispatch_driver_status_name(uint32_t status) {
    switch (status) {
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_STATUS_MISSING:
            return "driver_missing";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_STATUS_STALE:
            return "driver_stale";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_STATUS_NOT_BOUND_TO_HANDOFF:
            return "driver_not_bound_to_handoff";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_STATUS_NO_DISPATCH_BACKEND:
            return "driver_no_dispatch_backend";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_STATUS_NO_COMBINE_BACKEND:
            return "driver_no_combine_backend";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_STATUS_UNSUPPORTED_BOUNDARY:
            return "driver_unsupported_boundary";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_STATUS_OWNER_MISMATCH:
            return "driver_owner_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_STATUS_INVOCATION_MISMATCH:
            return "driver_invocation_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_STATUS_RUNTIME_PATH_MISMATCH:
            return "driver_runtime_path_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_STATUS_DESCRIPTOR_TOKEN_MISMATCH:
            return "driver_descriptor_token_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_STATUS_RANK_DEVICE_MISMATCH:
            return "driver_rank_device_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_STATUS_STATUS_SINK_MISMATCH:
            return "driver_status_sink_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_STATUS_PUBLIC_API_SOURCED_STATE:
            return "driver_public_api_sourced_state";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_STATUS_FABRICATED_PASS_EVIDENCE:
            return "driver_fabricated_pass_evidence";
        default:
            return "unknown_driver_status";
    }
}

inline const char *pto_cuda_uccl_ep_runtime_dispatch_driver_backend_status_name(uint32_t status) {
    switch (status) {
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_STATUS_REQUEST_UNBOUND:
            return "driver_backend_request_unbound";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_STATUS_DISPATCH_BACKEND_PLACEHOLDER:
            return "driver_dispatch_backend_placeholder";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_STATUS_COMBINE_BACKEND_PLACEHOLDER:
            return "driver_combine_backend_placeholder";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_STATUS_STATUS_SINK_UNBOUND:
            return "driver_status_sink_unbound";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_STATUS_UNSUPPORTED_BOUNDARY:
            return "driver_backend_map_unsupported_boundary";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_STATUS_OWNER_MISMATCH:
            return "driver_backend_owner_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_STATUS_INVOCATION_MISMATCH:
            return "driver_backend_invocation_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_STATUS_RUNTIME_PATH_MISMATCH:
            return "driver_backend_runtime_path_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_STATUS_DESCRIPTOR_TOKEN_MISMATCH:
            return "driver_backend_descriptor_token_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_STATUS_RANK_DEVICE_MISMATCH:
            return "driver_backend_rank_device_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_STATUS_STATUS_SINK_MISMATCH:
            return "driver_backend_status_sink_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_STATUS_PUBLIC_API_SOURCED_STATE:
            return "driver_backend_public_api_sourced_state";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_STATUS_FABRICATED_PASS_EVIDENCE:
            return "driver_backend_fabricated_pass_evidence";
        default:
            return "unknown_driver_backend_status";
    }
}

inline const char *pto_cuda_uccl_ep_runtime_dispatch_driver_backend_request_status_name(
    uint32_t status
) {
    switch (status) {
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_REQUEST_STATUS_PENDING:
            return "driver_backend_request_pending";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_REQUEST_STATUS_DISPATCH_REQUEST_PLACEHOLDER:
            return "driver_backend_dispatch_request_placeholder";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_REQUEST_STATUS_COMBINE_REQUEST_PLACEHOLDER:
            return "driver_backend_combine_request_placeholder";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_REQUEST_STATUS_STATUS_SINK_UNBOUND:
            return "driver_backend_request_status_sink_unbound";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_REQUEST_STATUS_UNSUPPORTED_BOUNDARY:
            return "driver_backend_request_map_unsupported_boundary";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_REQUEST_STATUS_OWNER_MISMATCH:
            return "driver_backend_request_owner_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_REQUEST_STATUS_INVOCATION_MISMATCH:
            return "driver_backend_request_invocation_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_REQUEST_STATUS_RUNTIME_PATH_MISMATCH:
            return "driver_backend_request_runtime_path_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_REQUEST_STATUS_DESCRIPTOR_TOKEN_MISMATCH:
            return "driver_backend_request_descriptor_token_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_REQUEST_STATUS_RANK_DEVICE_MISMATCH:
            return "driver_backend_request_rank_device_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_REQUEST_STATUS_STATUS_SINK_MISMATCH:
            return "driver_backend_request_status_sink_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_REQUEST_STATUS_PUBLIC_API_SOURCED_STATE:
            return "driver_backend_request_public_api_sourced_state";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_REQUEST_STATUS_PROVENANCE_SOURCED_STATE:
            return "driver_backend_request_provenance_sourced_state";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_REQUEST_STATUS_FABRICATED_PASS_EVIDENCE:
            return "driver_backend_request_fabricated_pass_evidence";
        default:
            return "unknown_driver_backend_request_status";
    }
}

inline const char *pto_cuda_uccl_ep_runtime_dispatch_driver_backend_dispatch_request_status_name(
    uint32_t status
) {
    switch (status) {
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_DISPATCH_REQUEST_STATUS_PENDING:
            return "driver_backend_dispatch_request_pending";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_DISPATCH_REQUEST_STATUS_PAYLOAD_DESCRIPTOR_PLACEHOLDER:
            return "driver_backend_dispatch_payload_descriptor_placeholder";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_DISPATCH_REQUEST_STATUS_OUTPUT_STATUS_SINK_UNBOUND:
            return "driver_backend_dispatch_output_status_sink_unbound";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_DISPATCH_REQUEST_STATUS_UNSUPPORTED_BOUNDARY:
            return "driver_backend_dispatch_request_map_unsupported_boundary";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_DISPATCH_REQUEST_STATUS_PAYLOAD_TRANSFER_UNIMPLEMENTED:
            return "driver_backend_dispatch_payload_transfer_unimplemented";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_DISPATCH_REQUEST_STATUS_OWNER_MISMATCH:
            return "driver_backend_dispatch_request_owner_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_DISPATCH_REQUEST_STATUS_INVOCATION_MISMATCH:
            return "driver_backend_dispatch_request_invocation_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_DISPATCH_REQUEST_STATUS_REQUEST_SCAFFOLD_MISMATCH:
            return "driver_backend_dispatch_request_scaffold_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_DISPATCH_REQUEST_STATUS_DESCRIPTOR_TOKEN_MISMATCH:
            return "driver_backend_dispatch_request_descriptor_token_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_DISPATCH_REQUEST_STATUS_RANK_DEVICE_MISMATCH:
            return "driver_backend_dispatch_request_rank_device_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_DISPATCH_REQUEST_STATUS_STATUS_SINK_MISMATCH:
            return "driver_backend_dispatch_request_status_sink_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_DISPATCH_REQUEST_STATUS_PUBLIC_API_SOURCED_STATE:
            return "driver_backend_dispatch_request_public_api_sourced_state";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_DISPATCH_REQUEST_STATUS_PROVENANCE_SOURCED_STATE:
            return "driver_backend_dispatch_request_provenance_sourced_state";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_DISPATCH_REQUEST_STATUS_FABRICATED_PASS_EVIDENCE:
            return "driver_backend_dispatch_request_fabricated_pass_evidence";
        default:
            return "unknown_driver_backend_dispatch_request_status";
    }
}

inline const char *pto_cuda_uccl_ep_runtime_dispatch_driver_backend_combine_request_status_name(
    uint32_t status
) {
    switch (status) {
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_REQUEST_STATUS_PENDING:
            return "driver_backend_combine_request_pending";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_REQUEST_STATUS_PAYLOAD_DESCRIPTOR_PLACEHOLDER:
            return "driver_backend_combine_payload_descriptor_placeholder";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_REQUEST_STATUS_OUTPUT_STATUS_SINK_UNBOUND:
            return "driver_backend_combine_output_status_sink_unbound";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_REQUEST_STATUS_UNSUPPORTED_BOUNDARY:
            return "driver_backend_combine_request_map_unsupported_boundary";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_REQUEST_STATUS_PAYLOAD_TRANSFER_UNIMPLEMENTED:
            return "driver_backend_combine_payload_transfer_unimplemented";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_REQUEST_STATUS_OWNER_MISMATCH:
            return "driver_backend_combine_request_owner_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_REQUEST_STATUS_INVOCATION_MISMATCH:
            return "driver_backend_combine_request_invocation_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_REQUEST_STATUS_REQUEST_SCAFFOLD_MISMATCH:
            return "driver_backend_combine_request_scaffold_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_REQUEST_STATUS_DESCRIPTOR_TOKEN_MISMATCH:
            return "driver_backend_combine_request_descriptor_token_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_REQUEST_STATUS_RANK_DEVICE_MISMATCH:
            return "driver_backend_combine_request_rank_device_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_REQUEST_STATUS_STATUS_SINK_MISMATCH:
            return "driver_backend_combine_request_status_sink_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_REQUEST_STATUS_PUBLIC_API_SOURCED_STATE:
            return "driver_backend_combine_request_public_api_sourced_state";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_REQUEST_STATUS_PROVENANCE_SOURCED_STATE:
            return "driver_backend_combine_request_provenance_sourced_state";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_REQUEST_STATUS_FABRICATED_PASS_EVIDENCE:
            return "driver_backend_combine_request_fabricated_pass_evidence";
        default:
            return "unknown_driver_backend_combine_request_status";
    }
}

inline const char *pto_cuda_uccl_ep_runtime_dispatch_driver_backend_combine_payload_status_name(
    uint32_t status
) {
    switch (status) {
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_STATUS_PENDING:
            return "driver_backend_combine_payload_pending";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_STATUS_DESCRIPTOR_PLACEHOLDER:
            return "driver_backend_combine_payload_descriptor_placeholder";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_STATUS_OUTPUT_STATUS_SINK_UNBOUND:
            return "driver_backend_combine_payload_output_status_sink_unbound";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_STATUS_UNSUPPORTED_BOUNDARY:
            return "driver_backend_combine_payload_map_unsupported_boundary";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_STATUS_TRANSFER_UNIMPLEMENTED:
            return "driver_backend_combine_payload_transfer_unimplemented";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_STATUS_OWNER_MISMATCH:
            return "driver_backend_combine_payload_owner_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_STATUS_INVOCATION_MISMATCH:
            return "driver_backend_combine_payload_invocation_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_STATUS_REQUEST_SCAFFOLD_MISMATCH:
            return "driver_backend_combine_payload_request_scaffold_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_STATUS_DESCRIPTOR_TOKEN_MISMATCH:
            return "driver_backend_combine_payload_descriptor_token_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_STATUS_RANK_DEVICE_MISMATCH:
            return "driver_backend_combine_payload_rank_device_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_STATUS_STATUS_SINK_MISMATCH:
            return "driver_backend_combine_payload_status_sink_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_STATUS_PUBLIC_API_SOURCED_STATE:
            return "driver_backend_combine_payload_public_api_sourced_state";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_STATUS_PROVENANCE_SOURCED_STATE:
            return "driver_backend_combine_payload_provenance_sourced_state";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_STATUS_FABRICATED_PASS_EVIDENCE:
            return "driver_backend_combine_payload_fabricated_pass_evidence";
        default:
            return "unknown_driver_backend_combine_payload_status";
    }
}

inline const char *
pto_cuda_uccl_ep_runtime_dispatch_driver_backend_combine_payload_transfer_status_name(
    uint32_t status
) {
    switch (status) {
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_PENDING:
            return "driver_backend_combine_payload_transfer_pending";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_TRANSFER_UNIMPLEMENTED:
            return "driver_backend_combine_payload_transfer_unimplemented";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_OUTPUT_STATUS_SINK_UNBOUND:
            return "driver_backend_combine_payload_transfer_output_status_sink_unbound";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_UNSUPPORTED_BOUNDARY:
            return "driver_backend_combine_payload_transfer_map_unsupported_boundary";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_OWNER_MISMATCH:
            return "driver_backend_combine_payload_transfer_owner_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_INVOCATION_MISMATCH:
            return "driver_backend_combine_payload_transfer_invocation_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_REQUEST_SCAFFOLD_MISMATCH:
            return "driver_backend_combine_payload_transfer_request_scaffold_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_DISPATCH_REQUEST_SCAFFOLD_MISMATCH:
            return "driver_backend_combine_payload_transfer_dispatch_request_scaffold_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_COMBINE_REQUEST_SCAFFOLD_MISMATCH:
            return "driver_backend_combine_payload_transfer_combine_request_scaffold_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_PAYLOAD_SCAFFOLD_MISMATCH:
            return "driver_backend_combine_payload_transfer_payload_scaffold_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_DESCRIPTOR_TOKEN_MISMATCH:
            return "driver_backend_combine_payload_transfer_descriptor_token_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_RANK_DEVICE_MISMATCH:
            return "driver_backend_combine_payload_transfer_rank_device_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_STATUS_SINK_MISMATCH:
            return "driver_backend_combine_payload_transfer_status_sink_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_PUBLIC_API_SOURCED_STATE:
            return "driver_backend_combine_payload_transfer_public_api_sourced_state";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_PROVENANCE_SOURCED_STATE:
            return "driver_backend_combine_payload_transfer_provenance_sourced_state";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_FABRICATED_PASS_EVIDENCE:
            return "driver_backend_combine_payload_transfer_fabricated_pass_evidence";
        default:
            return "unknown_driver_backend_combine_payload_transfer_status";
    }
}

inline const char *
pto_cuda_uccl_ep_runtime_dispatch_driver_backend_combine_payload_transfer_completion_status_name(
    uint32_t status
) {
    switch (status) {
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_STATUS_PENDING:
            return "driver_backend_combine_payload_transfer_completion_pending";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_STATUS_COMPLETION_UNIMPLEMENTED:
            return "driver_backend_combine_payload_transfer_completion_unimplemented";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_STATUS_STATUS_SINK_UNBOUND:
            return "driver_backend_combine_payload_transfer_completion_status_sink_unbound";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_STATUS_UNSUPPORTED_BOUNDARY:
            return "driver_backend_combine_payload_transfer_completion_map_unsupported_boundary";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_STATUS_OWNER_MISMATCH:
            return "driver_backend_combine_payload_transfer_completion_owner_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_STATUS_INVOCATION_MISMATCH:
            return "driver_backend_combine_payload_transfer_completion_invocation_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_STATUS_TRANSFER_SCAFFOLD_MISMATCH:
            return "driver_backend_combine_payload_transfer_completion_transfer_scaffold_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_STATUS_DESCRIPTOR_TOKEN_MISMATCH:
            return "driver_backend_combine_payload_transfer_completion_descriptor_token_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_STATUS_RANK_DEVICE_MISMATCH:
            return "driver_backend_combine_payload_transfer_completion_rank_device_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_STATUS_STATUS_SINK_MISMATCH:
            return "driver_backend_combine_payload_transfer_completion_status_sink_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_STATUS_PUBLIC_API_SOURCED_STATE:
            return "driver_backend_combine_payload_transfer_completion_public_api_sourced_state";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_STATUS_PROVENANCE_SOURCED_STATE:
            return "driver_backend_combine_payload_transfer_completion_provenance_sourced_state";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_STATUS_FABRICATED_PASS_EVIDENCE:
            return "driver_backend_combine_payload_transfer_completion_fabricated_pass_evidence";
        default:
            return "unknown_driver_backend_combine_payload_transfer_completion_status";
    }
}

inline const char *
pto_cuda_uccl_ep_runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_status_name(
    uint32_t status
) {
    switch (status) {
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_PENDING:
            return "driver_backend_combine_payload_transfer_completion_handoff_pending";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_HANDOFF_UNIMPLEMENTED:
            return "driver_backend_combine_payload_transfer_completion_handoff_unimplemented";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_STATUS_SINK_UNBOUND:
            return "driver_backend_combine_payload_transfer_completion_handoff_status_sink_unbound";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_UNSUPPORTED_BOUNDARY:
            return "driver_backend_combine_payload_transfer_completion_handoff_map_unsupported_boundary";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_OWNER_MISMATCH:
            return "driver_backend_combine_payload_transfer_completion_handoff_owner_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_INVOCATION_MISMATCH:
            return "driver_backend_combine_payload_transfer_completion_handoff_invocation_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_COMPLETION_SCAFFOLD_MISMATCH:
            return "driver_backend_combine_payload_transfer_completion_handoff_completion_scaffold_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_TRANSFER_SCAFFOLD_MISMATCH:
            return "driver_backend_combine_payload_transfer_completion_handoff_transfer_scaffold_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_DESCRIPTOR_TOKEN_MISMATCH:
            return "driver_backend_combine_payload_transfer_completion_handoff_descriptor_token_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_RANK_DEVICE_MISMATCH:
            return "driver_backend_combine_payload_transfer_completion_handoff_rank_device_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_STATUS_SINK_MISMATCH:
            return "driver_backend_combine_payload_transfer_completion_handoff_status_sink_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_PUBLIC_API_SOURCED_STATE:
            return "driver_backend_combine_payload_transfer_completion_handoff_public_api_sourced_state";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_PROVENANCE_SOURCED_STATE:
            return "driver_backend_combine_payload_transfer_completion_handoff_provenance_sourced_state";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_FABRICATED_PASS_EVIDENCE:
            return "driver_backend_combine_payload_transfer_completion_handoff_fabricated_pass_evidence";
        default:
            return "unknown_driver_backend_combine_payload_transfer_completion_handoff_status";
    }
}

inline const char *
pto_cuda_uccl_ep_runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_status_name(
    uint32_t status
) {
    switch (status) {
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_PENDING:
            return "driver_backend_combine_payload_transfer_completion_handoff_result_pending";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_RESULT_UNIMPLEMENTED:
            return "driver_backend_combine_payload_transfer_completion_handoff_result_unimplemented";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_STATUS_SINK_UNBOUND:
            return "driver_backend_combine_payload_transfer_completion_handoff_result_status_sink_unbound";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_RESULT_SINK_UNBOUND:
            return "driver_backend_combine_payload_transfer_completion_handoff_result_sink_unbound";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_UNSUPPORTED_BOUNDARY:
            return "driver_backend_combine_payload_transfer_completion_handoff_result_map_unsupported_boundary";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_OWNER_MISMATCH:
            return "driver_backend_combine_payload_transfer_completion_handoff_result_owner_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_INVOCATION_MISMATCH:
            return "driver_backend_combine_payload_transfer_completion_handoff_result_invocation_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_BACKEND_REQUEST_SCAFFOLD_MISMATCH:
            return "driver_backend_combine_payload_transfer_completion_handoff_result_backend_request_scaffold_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_DISPATCH_REQUEST_SCAFFOLD_MISMATCH:
            return "driver_backend_combine_payload_transfer_completion_handoff_result_dispatch_request_scaffold_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_COMBINE_REQUEST_SCAFFOLD_MISMATCH:
            return "driver_backend_combine_payload_transfer_completion_handoff_result_combine_request_scaffold_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_PAYLOAD_SCAFFOLD_MISMATCH:
            return "driver_backend_combine_payload_transfer_completion_handoff_result_payload_scaffold_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_TRANSFER_SCAFFOLD_MISMATCH:
            return "driver_backend_combine_payload_transfer_completion_handoff_result_transfer_scaffold_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_COMPLETION_SCAFFOLD_MISMATCH:
            return "driver_backend_combine_payload_transfer_completion_handoff_result_completion_scaffold_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_HANDOFF_SCAFFOLD_MISMATCH:
            return "driver_backend_combine_payload_transfer_completion_handoff_result_handoff_scaffold_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_DESCRIPTOR_TOKEN_MISMATCH:
            return "driver_backend_combine_payload_transfer_completion_handoff_result_descriptor_token_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_RANK_DEVICE_MISMATCH:
            return "driver_backend_combine_payload_transfer_completion_handoff_result_rank_device_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_STATUS_SINK_MISMATCH:
            return "driver_backend_combine_payload_transfer_completion_handoff_result_status_sink_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_RESULT_SINK_MISMATCH:
            return "driver_backend_combine_payload_transfer_completion_handoff_result_sink_mismatch";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_PUBLIC_API_SOURCED_STATE:
            return "driver_backend_combine_payload_transfer_completion_handoff_result_public_api_sourced_state";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_PROVENANCE_SOURCED_STATE:
            return "driver_backend_combine_payload_transfer_completion_handoff_result_provenance_sourced_state";
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_FABRICATED_PASS_EVIDENCE:
            return "driver_backend_combine_payload_transfer_completion_handoff_result_fabricated_pass_evidence";
        default:
            return "unknown_driver_backend_combine_payload_transfer_completion_handoff_result_status";
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
    coordinator->runtime_dispatch_driver_scaffold_status.version =
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_SCAFFOLD_STATUS_VERSION;
    coordinator->runtime_dispatch_driver_scaffold_status.invocation_id = request->invocation_id;
    coordinator->runtime_dispatch_driver_scaffold_status.driver_owner = coordinator;
    coordinator->runtime_dispatch_driver_scaffold_status.handoff_status =
        &coordinator->runtime_dispatch_request_handoff_scaffold_status;
    coordinator->runtime_dispatch_driver_scaffold_status.driver_state =
        &coordinator->runtime_dispatch_request_handoff_driver_state;
    coordinator->runtime_dispatch_driver_scaffold_status.runtime_path =
        &coordinator->descriptor_allocation.runtime_path;
    coordinator->runtime_dispatch_driver_scaffold_status.output_sink = output_sink;
    coordinator->runtime_dispatch_driver_scaffold_status.status =
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_STATUS_UNSUPPORTED_BOUNDARY;
    coordinator->runtime_dispatch_driver_scaffold_status.failure_fields =
        PTO_CUDA_RUNTIME_FUSION_FAILURE_UNSUPPORTED_BOUNDARY;
    coordinator->runtime_dispatch_driver_backend_scaffold_status.version =
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_SCAFFOLD_STATUS_VERSION;
    coordinator->runtime_dispatch_driver_backend_scaffold_status.invocation_id =
        request->invocation_id;
    coordinator->runtime_dispatch_driver_backend_scaffold_status.backend_owner = coordinator;
    coordinator->runtime_dispatch_driver_backend_scaffold_status.driver_status =
        &coordinator->runtime_dispatch_driver_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_scaffold_status.runtime_path =
        &coordinator->descriptor_allocation.runtime_path;
    coordinator->runtime_dispatch_driver_backend_scaffold_status.status_sink = output_sink;
    coordinator->runtime_dispatch_driver_backend_scaffold_status.shared_token =
        coordinator->descriptor_allocation.dispatch_descriptor.shared_token;
    coordinator->runtime_dispatch_driver_backend_scaffold_status.rank =
        coordinator->descriptor_allocation.dispatch_descriptor.rank;
    coordinator->runtime_dispatch_driver_backend_scaffold_status.device_id =
        coordinator->descriptor_allocation.dispatch_descriptor.device_id;
    coordinator->runtime_dispatch_driver_backend_scaffold_status.world_size =
        coordinator->descriptor_allocation.dispatch_descriptor.world_size;
    coordinator->runtime_dispatch_driver_backend_scaffold_status.status =
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_STATUS_UNSUPPORTED_BOUNDARY;
    coordinator->runtime_dispatch_driver_backend_scaffold_status.failure_fields =
        PTO_CUDA_RUNTIME_FUSION_FAILURE_UNSUPPORTED_BOUNDARY;
    coordinator->runtime_dispatch_driver_backend_request_scaffold_status.version =
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_REQUEST_SCAFFOLD_STATUS_VERSION;
    coordinator->runtime_dispatch_driver_backend_request_scaffold_status.invocation_id =
        request->invocation_id;
    coordinator->runtime_dispatch_driver_backend_request_scaffold_status.request_owner =
        coordinator;
    coordinator->runtime_dispatch_driver_backend_request_scaffold_status.backend_status =
        &coordinator->runtime_dispatch_driver_backend_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_request_scaffold_status.runtime_path =
        &coordinator->descriptor_allocation.runtime_path;
    coordinator->runtime_dispatch_driver_backend_request_scaffold_status.status_sink =
        output_sink;
    coordinator->runtime_dispatch_driver_backend_request_scaffold_status.shared_token =
        coordinator->descriptor_allocation.dispatch_descriptor.shared_token;
    coordinator->runtime_dispatch_driver_backend_request_scaffold_status.rank =
        coordinator->descriptor_allocation.dispatch_descriptor.rank;
    coordinator->runtime_dispatch_driver_backend_request_scaffold_status.device_id =
        coordinator->descriptor_allocation.dispatch_descriptor.device_id;
    coordinator->runtime_dispatch_driver_backend_request_scaffold_status.world_size =
        coordinator->descriptor_allocation.dispatch_descriptor.world_size;
    coordinator->runtime_dispatch_driver_backend_request_scaffold_status.status =
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_REQUEST_STATUS_UNSUPPORTED_BOUNDARY;
    coordinator->runtime_dispatch_driver_backend_request_scaffold_status.failure_fields =
        PTO_CUDA_RUNTIME_FUSION_FAILURE_UNSUPPORTED_BOUNDARY;
    coordinator->runtime_dispatch_driver_backend_dispatch_request_scaffold_status.version =
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_DISPATCH_REQUEST_SCAFFOLD_STATUS_VERSION;
    coordinator->runtime_dispatch_driver_backend_dispatch_request_scaffold_status.invocation_id =
        request->invocation_id;
    coordinator->runtime_dispatch_driver_backend_dispatch_request_scaffold_status.request_owner =
        coordinator;
    coordinator->runtime_dispatch_driver_backend_dispatch_request_scaffold_status.backend_request_status =
        &coordinator->runtime_dispatch_driver_backend_request_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_dispatch_request_scaffold_status.runtime_path =
        &coordinator->descriptor_allocation.runtime_path;
    coordinator->runtime_dispatch_driver_backend_dispatch_request_scaffold_status.status_sink =
        output_sink;
    coordinator->runtime_dispatch_driver_backend_dispatch_request_scaffold_status.shared_token =
        coordinator->descriptor_allocation.dispatch_descriptor.shared_token;
    coordinator->runtime_dispatch_driver_backend_dispatch_request_scaffold_status.rank =
        coordinator->descriptor_allocation.dispatch_descriptor.rank;
    coordinator->runtime_dispatch_driver_backend_dispatch_request_scaffold_status.device_id =
        coordinator->descriptor_allocation.dispatch_descriptor.device_id;
    coordinator->runtime_dispatch_driver_backend_dispatch_request_scaffold_status.world_size =
        coordinator->descriptor_allocation.dispatch_descriptor.world_size;
    coordinator->runtime_dispatch_driver_backend_dispatch_request_scaffold_status.status =
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_DISPATCH_REQUEST_STATUS_UNSUPPORTED_BOUNDARY;
    coordinator->runtime_dispatch_driver_backend_dispatch_request_scaffold_status.failure_fields =
        PTO_CUDA_RUNTIME_FUSION_FAILURE_UNSUPPORTED_BOUNDARY;
    coordinator->runtime_dispatch_driver_backend_combine_request_scaffold_status.version =
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_REQUEST_SCAFFOLD_STATUS_VERSION;
    coordinator->runtime_dispatch_driver_backend_combine_request_scaffold_status.invocation_id =
        request->invocation_id;
    coordinator->runtime_dispatch_driver_backend_combine_request_scaffold_status.request_owner =
        coordinator;
    coordinator->runtime_dispatch_driver_backend_combine_request_scaffold_status.backend_request_status =
        &coordinator->runtime_dispatch_driver_backend_request_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_combine_request_scaffold_status.dispatch_request_status =
        &coordinator->runtime_dispatch_driver_backend_dispatch_request_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_combine_request_scaffold_status.runtime_path =
        &coordinator->descriptor_allocation.runtime_path;
    coordinator->runtime_dispatch_driver_backend_combine_request_scaffold_status.status_sink =
        output_sink;
    coordinator->runtime_dispatch_driver_backend_combine_request_scaffold_status.shared_token =
        coordinator->descriptor_allocation.combine_descriptor.shared_token;
    coordinator->runtime_dispatch_driver_backend_combine_request_scaffold_status.rank =
        coordinator->descriptor_allocation.combine_descriptor.rank;
    coordinator->runtime_dispatch_driver_backend_combine_request_scaffold_status.device_id =
        coordinator->descriptor_allocation.combine_descriptor.device_id;
    coordinator->runtime_dispatch_driver_backend_combine_request_scaffold_status.world_size =
        coordinator->descriptor_allocation.combine_descriptor.world_size;
    coordinator->runtime_dispatch_driver_backend_combine_request_scaffold_status.status =
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_REQUEST_STATUS_UNSUPPORTED_BOUNDARY;
    coordinator->runtime_dispatch_driver_backend_combine_request_scaffold_status.failure_fields =
        PTO_CUDA_RUNTIME_FUSION_FAILURE_UNSUPPORTED_BOUNDARY;
    coordinator->runtime_dispatch_driver_backend_combine_payload_scaffold_status.version =
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD_STATUS_VERSION;
    coordinator->runtime_dispatch_driver_backend_combine_payload_scaffold_status.invocation_id =
        request->invocation_id;
    coordinator->runtime_dispatch_driver_backend_combine_payload_scaffold_status.payload_owner =
        coordinator;
    coordinator->runtime_dispatch_driver_backend_combine_payload_scaffold_status.combine_request_status =
        &coordinator->runtime_dispatch_driver_backend_combine_request_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_combine_payload_scaffold_status.runtime_path =
        &coordinator->descriptor_allocation.runtime_path;
    coordinator->runtime_dispatch_driver_backend_combine_payload_scaffold_status.status_sink =
        output_sink;
    coordinator->runtime_dispatch_driver_backend_combine_payload_scaffold_status.combine_payload_descriptor =
        coordinator->descriptor_allocation.runtime_path.combine_descriptor;
    coordinator->runtime_dispatch_driver_backend_combine_payload_scaffold_status.shared_token =
        coordinator->descriptor_allocation.combine_descriptor.shared_token;
    coordinator->runtime_dispatch_driver_backend_combine_payload_scaffold_status.rank =
        coordinator->descriptor_allocation.combine_descriptor.rank;
    coordinator->runtime_dispatch_driver_backend_combine_payload_scaffold_status.device_id =
        coordinator->descriptor_allocation.combine_descriptor.device_id;
    coordinator->runtime_dispatch_driver_backend_combine_payload_scaffold_status.world_size =
        coordinator->descriptor_allocation.combine_descriptor.world_size;
    coordinator->runtime_dispatch_driver_backend_combine_payload_scaffold_status.status =
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_STATUS_UNSUPPORTED_BOUNDARY;
    coordinator->runtime_dispatch_driver_backend_combine_payload_scaffold_status.failure_fields =
        PTO_CUDA_RUNTIME_FUSION_FAILURE_UNSUPPORTED_BOUNDARY;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.version =
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_SCAFFOLD_STATUS_VERSION;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.invocation_id =
        request->invocation_id;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.transfer_owner =
        coordinator;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.backend_request_status =
        &coordinator->runtime_dispatch_driver_backend_request_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.dispatch_request_status =
        &coordinator->runtime_dispatch_driver_backend_dispatch_request_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.combine_request_status =
        &coordinator->runtime_dispatch_driver_backend_combine_request_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.combine_payload_status =
        &coordinator->runtime_dispatch_driver_backend_combine_payload_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.runtime_path =
        &coordinator->descriptor_allocation.runtime_path;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.status_sink =
        output_sink;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.combine_payload_descriptor =
        coordinator->descriptor_allocation.runtime_path.combine_descriptor;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.shared_token =
        coordinator->descriptor_allocation.combine_descriptor.shared_token;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.rank =
        coordinator->descriptor_allocation.combine_descriptor.rank;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.device_id =
        coordinator->descriptor_allocation.combine_descriptor.device_id;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.world_size =
        coordinator->descriptor_allocation.combine_descriptor.world_size;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.status =
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_UNSUPPORTED_BOUNDARY;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.failure_fields =
        PTO_CUDA_RUNTIME_FUSION_FAILURE_UNSUPPORTED_BOUNDARY;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.version =
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_SCAFFOLD_STATUS_VERSION;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.invocation_id =
        request->invocation_id;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.completion_owner =
        coordinator;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.backend_request_status =
        &coordinator->runtime_dispatch_driver_backend_request_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.dispatch_request_status =
        &coordinator->runtime_dispatch_driver_backend_dispatch_request_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.combine_request_status =
        &coordinator->runtime_dispatch_driver_backend_combine_request_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.combine_payload_status =
        &coordinator->runtime_dispatch_driver_backend_combine_payload_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.transfer_status =
        &coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.runtime_path =
        &coordinator->descriptor_allocation.runtime_path;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.status_sink =
        output_sink;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.completion_sink =
        output_sink;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.combine_payload_descriptor =
        coordinator->descriptor_allocation.runtime_path.combine_descriptor;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.shared_token =
        coordinator->descriptor_allocation.combine_descriptor.shared_token;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.rank =
        coordinator->descriptor_allocation.combine_descriptor.rank;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.device_id =
        coordinator->descriptor_allocation.combine_descriptor.device_id;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.world_size =
        coordinator->descriptor_allocation.combine_descriptor.world_size;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.status =
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_STATUS_UNSUPPORTED_BOUNDARY;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.failure_fields =
        PTO_CUDA_RUNTIME_FUSION_FAILURE_UNSUPPORTED_BOUNDARY;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.version =
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_SCAFFOLD_STATUS_VERSION;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.invocation_id =
        request->invocation_id;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.handoff_owner =
        coordinator;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.backend_request_status =
        &coordinator->runtime_dispatch_driver_backend_request_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.dispatch_request_status =
        &coordinator->runtime_dispatch_driver_backend_dispatch_request_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.combine_request_status =
        &coordinator->runtime_dispatch_driver_backend_combine_request_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.combine_payload_status =
        &coordinator->runtime_dispatch_driver_backend_combine_payload_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.transfer_status =
        &coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.completion_status =
        &coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.runtime_path =
        &coordinator->descriptor_allocation.runtime_path;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.status_sink =
        output_sink;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.handoff_sink =
        output_sink;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.combine_payload_descriptor =
        coordinator->descriptor_allocation.runtime_path.combine_descriptor;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.shared_token =
        coordinator->descriptor_allocation.combine_descriptor.shared_token;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.rank =
        coordinator->descriptor_allocation.combine_descriptor.rank;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.device_id =
        coordinator->descriptor_allocation.combine_descriptor.device_id;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.world_size =
        coordinator->descriptor_allocation.combine_descriptor.world_size;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.status =
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_UNSUPPORTED_BOUNDARY;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.failure_fields =
        PTO_CUDA_RUNTIME_FUSION_FAILURE_UNSUPPORTED_BOUNDARY;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.version =
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_SCAFFOLD_STATUS_VERSION;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.invocation_id =
        request->invocation_id;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.result_owner =
        coordinator;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.backend_request_status =
        &coordinator->runtime_dispatch_driver_backend_request_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.dispatch_request_status =
        &coordinator->runtime_dispatch_driver_backend_dispatch_request_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.combine_request_status =
        &coordinator->runtime_dispatch_driver_backend_combine_request_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.combine_payload_status =
        &coordinator->runtime_dispatch_driver_backend_combine_payload_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.transfer_status =
        &coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.completion_status =
        &coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.handoff_status =
        &coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.runtime_path =
        &coordinator->descriptor_allocation.runtime_path;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.status_sink =
        output_sink;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.result_sink =
        output_sink;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.combine_payload_descriptor =
        coordinator->descriptor_allocation.runtime_path.combine_descriptor;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.shared_token =
        coordinator->descriptor_allocation.combine_descriptor.shared_token;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.rank =
        coordinator->descriptor_allocation.combine_descriptor.rank;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.device_id =
        coordinator->descriptor_allocation.combine_descriptor.device_id;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.world_size =
        coordinator->descriptor_allocation.combine_descriptor.world_size;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.status =
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_UNSUPPORTED_BOUNDARY;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.failure_fields =
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

inline uint32_t pto_cuda_runtime_fusion_validate_runtime_dispatch_driver_backend_scaffold_status(
    const PtoCudaRuntimeFusionRequest *request, const PtoCudaRuntimeFusionCoordinator *coordinator
) {
    const PtoCudaUcclEpRuntimeDispatchDriverBackendScaffoldStatus *backend_status =
        &coordinator->runtime_dispatch_driver_backend_scaffold_status;
    const PtoCudaUcclEpRuntimePath *runtime_path = &coordinator->descriptor_allocation.runtime_path;
    const PtoCudaUcclEpRuntimeDescriptorView *dispatch_descriptor = runtime_path->dispatch_descriptor;
    uint32_t failures = 0;

    if (backend_status->version !=
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_SCAFFOLD_STATUS_VERSION) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH;
    }
    if (backend_status->invocation_id != request->invocation_id) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_INVOCATION_MISMATCH;
    }
    if (backend_status->backend_owner != coordinator ||
        backend_status->driver_status != &coordinator->runtime_dispatch_driver_scaffold_status) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH;
    }
    if (backend_status->runtime_path != runtime_path) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RUNTIME_PATH_MISMATCH;
    }
    if (backend_status->status_sink == nullptr || backend_status->status_sink != request->output_sink) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_STATUS_SINK_MISMATCH;
    }
    if (dispatch_descriptor == nullptr || backend_status->shared_token == 0U ||
        backend_status->shared_token != dispatch_descriptor->shared_token) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_DESCRIPTOR_TOKEN_MISMATCH;
    }
    if (request->comm_descriptor != nullptr &&
        (backend_status->rank != request->comm_descriptor->rank ||
         backend_status->device_id != request->comm_descriptor->device_id ||
         backend_status->world_size != request->comm_descriptor->world_size)) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RANK_DEVICE_MISMATCH;
    }

    const uint32_t propagated_backend_failures =
        backend_status->failure_fields &
        (PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_INVOCATION_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RUNTIME_PATH_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_DESCRIPTOR_TOKEN_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RANK_DEVICE_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_STATUS_SINK_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_PUBLIC_API_SOURCED_STATE |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_FABRICATED_PASS_EVIDENCE |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_SCAFFOLD);
    if (propagated_backend_failures != 0U) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_SCAFFOLD |
                    propagated_backend_failures;
    }

    switch (backend_status->status) {
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_STATUS_OWNER_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_STATUS_INVOCATION_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_INVOCATION_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_STATUS_RUNTIME_PATH_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RUNTIME_PATH_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_STATUS_DESCRIPTOR_TOKEN_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_DESCRIPTOR_TOKEN_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_STATUS_RANK_DEVICE_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RANK_DEVICE_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_STATUS_STATUS_SINK_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_STATUS_SINK_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_STATUS_PUBLIC_API_SOURCED_STATE:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_PUBLIC_API_SOURCED_STATE;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_STATUS_FABRICATED_PASS_EVIDENCE:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_FABRICATED_PASS_EVIDENCE |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_FABRICATED_OR_UNTRUSTED_PASS_EVIDENCE;
            break;
        default:
            break;
    }
    return failures;
}

inline uint32_t
pto_cuda_runtime_fusion_validate_runtime_dispatch_driver_backend_request_scaffold_status(
    const PtoCudaRuntimeFusionRequest *request, const PtoCudaRuntimeFusionCoordinator *coordinator
) {
    const PtoCudaUcclEpRuntimeDispatchDriverBackendRequestScaffoldStatus *request_status =
        &coordinator->runtime_dispatch_driver_backend_request_scaffold_status;
    const PtoCudaUcclEpRuntimePath *runtime_path = &coordinator->descriptor_allocation.runtime_path;
    const PtoCudaUcclEpRuntimeDescriptorView *dispatch_descriptor = runtime_path->dispatch_descriptor;
    uint32_t failures = 0;

    if (request_status->version !=
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_REQUEST_SCAFFOLD_STATUS_VERSION) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_REQUEST_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH;
    }
    if (request_status->invocation_id != request->invocation_id) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_REQUEST_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_INVOCATION_MISMATCH;
    }
    if (request_status->request_owner != coordinator ||
        request_status->backend_status !=
            &coordinator->runtime_dispatch_driver_backend_scaffold_status) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_REQUEST_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH;
    }
    if (request_status->runtime_path != runtime_path) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_REQUEST_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RUNTIME_PATH_MISMATCH;
    }
    if (request_status->status_sink == nullptr ||
        request_status->status_sink != request->output_sink) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_REQUEST_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_STATUS_SINK_MISMATCH;
    }
    if (dispatch_descriptor == nullptr || request_status->shared_token == 0U ||
        request_status->shared_token != dispatch_descriptor->shared_token) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_REQUEST_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_DESCRIPTOR_TOKEN_MISMATCH;
    }
    if (request->comm_descriptor != nullptr &&
        (request_status->rank != request->comm_descriptor->rank ||
         request_status->device_id != request->comm_descriptor->device_id ||
         request_status->world_size != request->comm_descriptor->world_size)) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_REQUEST_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RANK_DEVICE_MISMATCH;
    }

    const uint32_t propagated_backend_request_failures =
        request_status->failure_fields &
        (PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_INVOCATION_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RUNTIME_PATH_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_DESCRIPTOR_TOKEN_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RANK_DEVICE_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_STATUS_SINK_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_PUBLIC_API_SOURCED_STATE |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_FABRICATED_PASS_EVIDENCE |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_REQUEST_SCAFFOLD);
    if (propagated_backend_request_failures != 0U) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_REQUEST_SCAFFOLD |
                    propagated_backend_request_failures;
    }

    switch (request_status->status) {
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_REQUEST_STATUS_OWNER_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_REQUEST_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_REQUEST_STATUS_INVOCATION_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_REQUEST_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_INVOCATION_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_REQUEST_STATUS_RUNTIME_PATH_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_REQUEST_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RUNTIME_PATH_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_REQUEST_STATUS_DESCRIPTOR_TOKEN_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_REQUEST_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_DESCRIPTOR_TOKEN_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_REQUEST_STATUS_RANK_DEVICE_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_REQUEST_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RANK_DEVICE_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_REQUEST_STATUS_STATUS_SINK_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_REQUEST_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_STATUS_SINK_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_REQUEST_STATUS_PUBLIC_API_SOURCED_STATE:
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_REQUEST_STATUS_PROVENANCE_SOURCED_STATE:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_REQUEST_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_PUBLIC_API_SOURCED_STATE;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_REQUEST_STATUS_FABRICATED_PASS_EVIDENCE:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_REQUEST_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_FABRICATED_PASS_EVIDENCE |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_FABRICATED_OR_UNTRUSTED_PASS_EVIDENCE;
            break;
        default:
            break;
    }
    return failures;
}

inline uint32_t
pto_cuda_runtime_fusion_validate_runtime_dispatch_driver_backend_dispatch_request_scaffold_status(
    const PtoCudaRuntimeFusionRequest *request, const PtoCudaRuntimeFusionCoordinator *coordinator
) {
    const PtoCudaUcclEpRuntimeDispatchDriverBackendDispatchRequestScaffoldStatus
        *dispatch_request_status =
            &coordinator->runtime_dispatch_driver_backend_dispatch_request_scaffold_status;
    const PtoCudaUcclEpRuntimePath *runtime_path = &coordinator->descriptor_allocation.runtime_path;
    const PtoCudaUcclEpRuntimeDescriptorView *dispatch_descriptor = runtime_path->dispatch_descriptor;
    uint32_t failures = 0;

    if (dispatch_request_status->version !=
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_DISPATCH_REQUEST_SCAFFOLD_STATUS_VERSION) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_DISPATCH_REQUEST_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH;
    }
    if (dispatch_request_status->invocation_id != request->invocation_id) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_DISPATCH_REQUEST_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_INVOCATION_MISMATCH;
    }
    if (dispatch_request_status->request_owner != coordinator ||
        dispatch_request_status->backend_request_status !=
            &coordinator->runtime_dispatch_driver_backend_request_scaffold_status) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_DISPATCH_REQUEST_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH;
    }
    if (dispatch_request_status->runtime_path != runtime_path) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_DISPATCH_REQUEST_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RUNTIME_PATH_MISMATCH;
    }
    if (dispatch_request_status->status_sink == nullptr ||
        dispatch_request_status->status_sink != request->output_sink) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_DISPATCH_REQUEST_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_STATUS_SINK_MISMATCH;
    }
    if (dispatch_descriptor == nullptr || dispatch_request_status->shared_token == 0U ||
        dispatch_request_status->shared_token != dispatch_descriptor->shared_token) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_DISPATCH_REQUEST_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_DESCRIPTOR_TOKEN_MISMATCH;
    }
    if (request->comm_descriptor != nullptr &&
        (dispatch_request_status->rank != request->comm_descriptor->rank ||
         dispatch_request_status->device_id != request->comm_descriptor->device_id ||
         dispatch_request_status->world_size != request->comm_descriptor->world_size)) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_DISPATCH_REQUEST_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RANK_DEVICE_MISMATCH;
    }

    const uint32_t propagated_dispatch_request_failures =
        dispatch_request_status->failure_fields &
        (PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_INVOCATION_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RUNTIME_PATH_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_DESCRIPTOR_TOKEN_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RANK_DEVICE_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_STATUS_SINK_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_PUBLIC_API_SOURCED_STATE |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_FABRICATED_PASS_EVIDENCE |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_REQUEST_SCAFFOLD |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_DISPATCH_REQUEST_SCAFFOLD);
    if (propagated_dispatch_request_failures != 0U) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_DISPATCH_REQUEST_SCAFFOLD |
                    propagated_dispatch_request_failures;
    }

    switch (dispatch_request_status->status) {
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_DISPATCH_REQUEST_STATUS_OWNER_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_DISPATCH_REQUEST_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_DISPATCH_REQUEST_STATUS_INVOCATION_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_DISPATCH_REQUEST_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_INVOCATION_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_DISPATCH_REQUEST_STATUS_REQUEST_SCAFFOLD_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_DISPATCH_REQUEST_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_REQUEST_SCAFFOLD;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_DISPATCH_REQUEST_STATUS_DESCRIPTOR_TOKEN_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_DISPATCH_REQUEST_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_DESCRIPTOR_TOKEN_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_DISPATCH_REQUEST_STATUS_RANK_DEVICE_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_DISPATCH_REQUEST_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RANK_DEVICE_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_DISPATCH_REQUEST_STATUS_STATUS_SINK_MISMATCH:
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_DISPATCH_REQUEST_STATUS_OUTPUT_STATUS_SINK_UNBOUND:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_DISPATCH_REQUEST_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_STATUS_SINK_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_DISPATCH_REQUEST_STATUS_PUBLIC_API_SOURCED_STATE:
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_DISPATCH_REQUEST_STATUS_PROVENANCE_SOURCED_STATE:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_DISPATCH_REQUEST_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_PUBLIC_API_SOURCED_STATE;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_DISPATCH_REQUEST_STATUS_FABRICATED_PASS_EVIDENCE:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_DISPATCH_REQUEST_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_FABRICATED_PASS_EVIDENCE |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_FABRICATED_OR_UNTRUSTED_PASS_EVIDENCE;
            break;
        default:
            break;
    }
    return failures;
}

inline uint32_t
pto_cuda_runtime_fusion_validate_runtime_dispatch_driver_backend_combine_request_scaffold_status(
    const PtoCudaRuntimeFusionRequest *request, const PtoCudaRuntimeFusionCoordinator *coordinator
) {
    const PtoCudaUcclEpRuntimeDispatchDriverBackendCombineRequestScaffoldStatus
        *combine_request_status =
            &coordinator->runtime_dispatch_driver_backend_combine_request_scaffold_status;
    const PtoCudaUcclEpRuntimePath *runtime_path = &coordinator->descriptor_allocation.runtime_path;
    const PtoCudaUcclEpRuntimeDescriptorView *combine_descriptor = runtime_path->combine_descriptor;
    uint32_t failures = 0;

    if (combine_request_status->version !=
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_REQUEST_SCAFFOLD_STATUS_VERSION) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_REQUEST_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH;
    }
    if (combine_request_status->invocation_id != request->invocation_id) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_REQUEST_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_INVOCATION_MISMATCH;
    }
    if (combine_request_status->request_owner != coordinator ||
        combine_request_status->backend_request_status !=
            &coordinator->runtime_dispatch_driver_backend_request_scaffold_status ||
        combine_request_status->dispatch_request_status !=
            &coordinator->runtime_dispatch_driver_backend_dispatch_request_scaffold_status) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_REQUEST_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH;
    }
    if (combine_request_status->runtime_path != runtime_path) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_REQUEST_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RUNTIME_PATH_MISMATCH;
    }
    if (combine_request_status->status_sink == nullptr ||
        combine_request_status->status_sink != request->output_sink) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_REQUEST_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_STATUS_SINK_MISMATCH;
    }
    if (combine_descriptor == nullptr || combine_request_status->shared_token == 0U ||
        combine_request_status->shared_token != combine_descriptor->shared_token) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_REQUEST_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_DESCRIPTOR_TOKEN_MISMATCH;
    }
    if (request->comm_descriptor != nullptr &&
        (combine_request_status->rank != request->comm_descriptor->rank ||
         combine_request_status->device_id != request->comm_descriptor->device_id ||
         combine_request_status->world_size != request->comm_descriptor->world_size)) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_REQUEST_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RANK_DEVICE_MISMATCH;
    }

    const uint32_t propagated_combine_request_failures =
        combine_request_status->failure_fields &
        (PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_INVOCATION_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RUNTIME_PATH_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_DESCRIPTOR_TOKEN_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RANK_DEVICE_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_STATUS_SINK_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_PUBLIC_API_SOURCED_STATE |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_FABRICATED_PASS_EVIDENCE |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_REQUEST_SCAFFOLD |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_DISPATCH_REQUEST_SCAFFOLD |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_REQUEST_SCAFFOLD);
    if (propagated_combine_request_failures != 0U) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_REQUEST_SCAFFOLD |
                    propagated_combine_request_failures;
    }

    switch (combine_request_status->status) {
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_REQUEST_STATUS_OWNER_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_REQUEST_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_REQUEST_STATUS_INVOCATION_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_REQUEST_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_INVOCATION_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_REQUEST_STATUS_REQUEST_SCAFFOLD_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_REQUEST_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_DISPATCH_REQUEST_SCAFFOLD;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_REQUEST_STATUS_DESCRIPTOR_TOKEN_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_REQUEST_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_DESCRIPTOR_TOKEN_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_REQUEST_STATUS_RANK_DEVICE_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_REQUEST_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RANK_DEVICE_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_REQUEST_STATUS_STATUS_SINK_MISMATCH:
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_REQUEST_STATUS_OUTPUT_STATUS_SINK_UNBOUND:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_REQUEST_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_STATUS_SINK_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_REQUEST_STATUS_PUBLIC_API_SOURCED_STATE:
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_REQUEST_STATUS_PROVENANCE_SOURCED_STATE:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_REQUEST_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_PUBLIC_API_SOURCED_STATE;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_REQUEST_STATUS_FABRICATED_PASS_EVIDENCE:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_REQUEST_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_FABRICATED_PASS_EVIDENCE |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_FABRICATED_OR_UNTRUSTED_PASS_EVIDENCE;
            break;
        default:
            break;
    }
    return failures;
}

inline uint32_t
pto_cuda_runtime_fusion_validate_runtime_dispatch_driver_backend_combine_payload_scaffold_status(
    const PtoCudaRuntimeFusionRequest *request, const PtoCudaRuntimeFusionCoordinator *coordinator
) {
    const PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadScaffoldStatus
        *combine_payload_status =
            &coordinator->runtime_dispatch_driver_backend_combine_payload_scaffold_status;
    const PtoCudaUcclEpRuntimePath *runtime_path = &coordinator->descriptor_allocation.runtime_path;
    const PtoCudaUcclEpRuntimeDescriptorView *combine_descriptor = runtime_path->combine_descriptor;
    uint32_t failures = 0;

    if (combine_payload_status->version !=
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD_STATUS_VERSION) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH;
    }
    if (combine_payload_status->invocation_id != request->invocation_id) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_INVOCATION_MISMATCH;
    }
    if (combine_payload_status->payload_owner != coordinator ||
        combine_payload_status->combine_request_status !=
            &coordinator->runtime_dispatch_driver_backend_combine_request_scaffold_status) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH;
    }
    if (combine_payload_status->runtime_path != runtime_path) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RUNTIME_PATH_MISMATCH;
    }
    if (combine_payload_status->status_sink == nullptr ||
        combine_payload_status->status_sink != request->output_sink) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_STATUS_SINK_MISMATCH;
    }
    if (combine_descriptor == nullptr || combine_payload_status->combine_payload_descriptor != combine_descriptor ||
        combine_payload_status->shared_token == 0U ||
        combine_payload_status->shared_token != combine_descriptor->shared_token) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_DESCRIPTOR_TOKEN_MISMATCH;
    }
    if (request->comm_descriptor != nullptr &&
        (combine_payload_status->rank != request->comm_descriptor->rank ||
         combine_payload_status->device_id != request->comm_descriptor->device_id ||
         combine_payload_status->world_size != request->comm_descriptor->world_size)) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RANK_DEVICE_MISMATCH;
    }

    const uint32_t propagated_combine_payload_failures =
        combine_payload_status->failure_fields &
        (PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_INVOCATION_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RUNTIME_PATH_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_DESCRIPTOR_TOKEN_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RANK_DEVICE_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_STATUS_SINK_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_PUBLIC_API_SOURCED_STATE |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_FABRICATED_PASS_EVIDENCE |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_REQUEST_SCAFFOLD |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD);
    if (propagated_combine_payload_failures != 0U) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    propagated_combine_payload_failures;
    }

    switch (combine_payload_status->status) {
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_STATUS_OWNER_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_STATUS_INVOCATION_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_INVOCATION_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_STATUS_REQUEST_SCAFFOLD_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_REQUEST_SCAFFOLD;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_STATUS_DESCRIPTOR_TOKEN_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_DESCRIPTOR_TOKEN_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_STATUS_RANK_DEVICE_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RANK_DEVICE_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_STATUS_STATUS_SINK_MISMATCH:
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_STATUS_OUTPUT_STATUS_SINK_UNBOUND:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_STATUS_SINK_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_STATUS_PUBLIC_API_SOURCED_STATE:
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_STATUS_PROVENANCE_SOURCED_STATE:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_PUBLIC_API_SOURCED_STATE;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_STATUS_FABRICATED_PASS_EVIDENCE:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_FABRICATED_PASS_EVIDENCE |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_FABRICATED_OR_UNTRUSTED_PASS_EVIDENCE;
            break;
        default:
            break;
    }
    return failures;
}

inline uint32_t
pto_cuda_runtime_fusion_validate_runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status(
    const PtoCudaRuntimeFusionRequest *request, const PtoCudaRuntimeFusionCoordinator *coordinator
) {
    const PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadTransferScaffoldStatus
        *transfer_status =
            &coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status;
    const PtoCudaUcclEpRuntimePath *runtime_path = &coordinator->descriptor_allocation.runtime_path;
    const PtoCudaUcclEpRuntimeDescriptorView *combine_descriptor = runtime_path->combine_descriptor;
    const PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadScaffoldStatus
        *combine_payload_status =
            &coordinator->runtime_dispatch_driver_backend_combine_payload_scaffold_status;
    uint32_t failures = 0;

    if (transfer_status->version !=
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_SCAFFOLD_STATUS_VERSION) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH;
    }
    if (transfer_status->invocation_id != request->invocation_id) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_INVOCATION_MISMATCH;
    }
    if (transfer_status->transfer_owner != coordinator) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH;
    }
    if (transfer_status->backend_request_status !=
        &coordinator->runtime_dispatch_driver_backend_request_scaffold_status) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_REQUEST_SCAFFOLD;
    }
    if (transfer_status->dispatch_request_status !=
        &coordinator->runtime_dispatch_driver_backend_dispatch_request_scaffold_status) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_DISPATCH_REQUEST_SCAFFOLD;
    }
    if (transfer_status->combine_request_status !=
        &coordinator->runtime_dispatch_driver_backend_combine_request_scaffold_status) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_REQUEST_SCAFFOLD;
    }
    if (transfer_status->combine_payload_status != combine_payload_status) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH;
    }
    if (transfer_status->runtime_path != runtime_path) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RUNTIME_PATH_MISMATCH;
    }
    if (transfer_status->status_sink == nullptr ||
        transfer_status->status_sink != request->output_sink) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_STATUS_SINK_MISMATCH;
    }
    if (combine_descriptor == nullptr ||
        transfer_status->combine_payload_descriptor != combine_descriptor ||
        transfer_status->combine_payload_descriptor !=
            combine_payload_status->combine_payload_descriptor ||
        transfer_status->shared_token == 0U ||
        transfer_status->shared_token != combine_descriptor->shared_token ||
        transfer_status->shared_token != combine_payload_status->shared_token) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_DESCRIPTOR_TOKEN_MISMATCH;
    }
    if (request->comm_descriptor != nullptr &&
        (transfer_status->rank != request->comm_descriptor->rank ||
         transfer_status->device_id != request->comm_descriptor->device_id ||
         transfer_status->world_size != request->comm_descriptor->world_size ||
         transfer_status->rank != combine_payload_status->rank ||
         transfer_status->device_id != combine_payload_status->device_id ||
         transfer_status->world_size != combine_payload_status->world_size)) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RANK_DEVICE_MISMATCH;
    }

    const uint32_t propagated_transfer_failures =
        transfer_status->failure_fields &
        (PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_INVOCATION_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RUNTIME_PATH_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_DESCRIPTOR_TOKEN_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RANK_DEVICE_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_STATUS_SINK_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_PUBLIC_API_SOURCED_STATE |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_FABRICATED_PASS_EVIDENCE |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_REQUEST_SCAFFOLD |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_DISPATCH_REQUEST_SCAFFOLD |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_REQUEST_SCAFFOLD |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD);
    if (propagated_transfer_failures != 0U) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    propagated_transfer_failures;
    }

    switch (transfer_status->status) {
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_OWNER_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_INVOCATION_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_INVOCATION_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_REQUEST_SCAFFOLD_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_REQUEST_SCAFFOLD;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_DISPATCH_REQUEST_SCAFFOLD_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_DISPATCH_REQUEST_SCAFFOLD;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_COMBINE_REQUEST_SCAFFOLD_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_REQUEST_SCAFFOLD;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_PAYLOAD_SCAFFOLD_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_DESCRIPTOR_TOKEN_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_DESCRIPTOR_TOKEN_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_RANK_DEVICE_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RANK_DEVICE_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_STATUS_SINK_MISMATCH:
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_OUTPUT_STATUS_SINK_UNBOUND:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_STATUS_SINK_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_PUBLIC_API_SOURCED_STATE:
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_PROVENANCE_SOURCED_STATE:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_PUBLIC_API_SOURCED_STATE;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_FABRICATED_PASS_EVIDENCE:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_FABRICATED_PASS_EVIDENCE |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_FABRICATED_OR_UNTRUSTED_PASS_EVIDENCE;
            break;
        default:
            break;
    }
    return failures;
}

inline uint32_t
pto_cuda_runtime_fusion_validate_runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status(
    const PtoCudaRuntimeFusionRequest *request, const PtoCudaRuntimeFusionCoordinator *coordinator
) {
    const PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadTransferCompletionScaffoldStatus
        *completion_status =
            &coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status;
    const PtoCudaUcclEpRuntimePath *runtime_path = &coordinator->descriptor_allocation.runtime_path;
    const PtoCudaUcclEpRuntimeDescriptorView *combine_descriptor = runtime_path->combine_descriptor;
    const PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadScaffoldStatus
        *combine_payload_status =
            &coordinator->runtime_dispatch_driver_backend_combine_payload_scaffold_status;
    const PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadTransferScaffoldStatus
        *transfer_status =
            &coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status;
    uint32_t failures = 0;

    if (completion_status->version !=
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_SCAFFOLD_STATUS_VERSION) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH;
    }
    if (completion_status->invocation_id != request->invocation_id) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_INVOCATION_MISMATCH;
    }
    if (completion_status->completion_owner != coordinator) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH;
    }
    if (completion_status->backend_request_status !=
        &coordinator->runtime_dispatch_driver_backend_request_scaffold_status) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_REQUEST_SCAFFOLD;
    }
    if (completion_status->dispatch_request_status !=
        &coordinator->runtime_dispatch_driver_backend_dispatch_request_scaffold_status) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_DISPATCH_REQUEST_SCAFFOLD;
    }
    if (completion_status->combine_request_status !=
        &coordinator->runtime_dispatch_driver_backend_combine_request_scaffold_status) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_REQUEST_SCAFFOLD;
    }
    if (completion_status->combine_payload_status != combine_payload_status ||
        completion_status->transfer_status != transfer_status) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD;
    }
    if (completion_status->runtime_path != runtime_path) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RUNTIME_PATH_MISMATCH;
    }
    if (completion_status->status_sink == nullptr ||
        completion_status->status_sink != request->output_sink ||
        completion_status->completion_sink == nullptr ||
        completion_status->completion_sink != request->output_sink) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_STATUS_SINK_MISMATCH;
    }
    if (combine_descriptor == nullptr ||
        completion_status->combine_payload_descriptor != combine_descriptor ||
        completion_status->combine_payload_descriptor !=
            combine_payload_status->combine_payload_descriptor ||
        completion_status->combine_payload_descriptor != transfer_status->combine_payload_descriptor ||
        completion_status->shared_token == 0U ||
        completion_status->shared_token != combine_descriptor->shared_token ||
        completion_status->shared_token != combine_payload_status->shared_token ||
        completion_status->shared_token != transfer_status->shared_token) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_DESCRIPTOR_TOKEN_MISMATCH;
    }
    if (request->comm_descriptor != nullptr &&
        (completion_status->rank != request->comm_descriptor->rank ||
         completion_status->device_id != request->comm_descriptor->device_id ||
         completion_status->world_size != request->comm_descriptor->world_size ||
         completion_status->rank != combine_payload_status->rank ||
         completion_status->device_id != combine_payload_status->device_id ||
         completion_status->world_size != combine_payload_status->world_size ||
         completion_status->rank != transfer_status->rank ||
         completion_status->device_id != transfer_status->device_id ||
         completion_status->world_size != transfer_status->world_size)) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RANK_DEVICE_MISMATCH;
    }

    const uint32_t propagated_completion_failures =
        completion_status->failure_fields &
        (PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_INVOCATION_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RUNTIME_PATH_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_DESCRIPTOR_TOKEN_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RANK_DEVICE_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_STATUS_SINK_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_PUBLIC_API_SOURCED_STATE |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_FABRICATED_PASS_EVIDENCE |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_REQUEST_SCAFFOLD |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_DISPATCH_REQUEST_SCAFFOLD |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_REQUEST_SCAFFOLD |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD);
    if (propagated_completion_failures != 0U) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    propagated_completion_failures;
    }

    switch (completion_status->status) {
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_STATUS_OWNER_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_STATUS_INVOCATION_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_INVOCATION_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_STATUS_TRANSFER_SCAFFOLD_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_STATUS_DESCRIPTOR_TOKEN_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_DESCRIPTOR_TOKEN_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_STATUS_RANK_DEVICE_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RANK_DEVICE_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_STATUS_STATUS_SINK_MISMATCH:
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_STATUS_STATUS_SINK_UNBOUND:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_STATUS_SINK_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_STATUS_PUBLIC_API_SOURCED_STATE:
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_STATUS_PROVENANCE_SOURCED_STATE:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_PUBLIC_API_SOURCED_STATE;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_STATUS_FABRICATED_PASS_EVIDENCE:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_FABRICATED_PASS_EVIDENCE |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_FABRICATED_OR_UNTRUSTED_PASS_EVIDENCE;
            break;
        default:
            break;
    }
    return failures;
}

inline uint32_t
pto_cuda_runtime_fusion_validate_runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status(
    const PtoCudaRuntimeFusionRequest *request, const PtoCudaRuntimeFusionCoordinator *coordinator
) {
    const PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadTransferCompletionHandoffScaffoldStatus
        *handoff_status =
            &coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status;
    const PtoCudaUcclEpRuntimePath *runtime_path = &coordinator->descriptor_allocation.runtime_path;
    const PtoCudaUcclEpRuntimeDescriptorView *combine_descriptor = runtime_path->combine_descriptor;
    const PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadScaffoldStatus
        *combine_payload_status =
            &coordinator->runtime_dispatch_driver_backend_combine_payload_scaffold_status;
    const PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadTransferScaffoldStatus
        *transfer_status =
            &coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status;
    const PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadTransferCompletionScaffoldStatus
        *completion_status =
            &coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status;
    uint32_t failures = 0;

    if (handoff_status->version !=
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_SCAFFOLD_STATUS_VERSION) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH;
    }
    if (handoff_status->invocation_id != request->invocation_id) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_INVOCATION_MISMATCH;
    }
    if (handoff_status->handoff_owner != coordinator) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH;
    }
    if (handoff_status->backend_request_status !=
        &coordinator->runtime_dispatch_driver_backend_request_scaffold_status) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_REQUEST_SCAFFOLD;
    }
    if (handoff_status->dispatch_request_status !=
        &coordinator->runtime_dispatch_driver_backend_dispatch_request_scaffold_status) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_DISPATCH_REQUEST_SCAFFOLD;
    }
    if (handoff_status->combine_request_status !=
        &coordinator->runtime_dispatch_driver_backend_combine_request_scaffold_status) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_REQUEST_SCAFFOLD;
    }
    if (handoff_status->combine_payload_status != combine_payload_status ||
        handoff_status->transfer_status != transfer_status ||
        handoff_status->completion_status != completion_status) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD;
    }
    if (handoff_status->runtime_path != runtime_path) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RUNTIME_PATH_MISMATCH;
    }
    if (handoff_status->status_sink == nullptr ||
        handoff_status->status_sink != request->output_sink ||
        handoff_status->handoff_sink == nullptr ||
        handoff_status->handoff_sink != request->output_sink) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_STATUS_SINK_MISMATCH;
    }
    if (combine_descriptor == nullptr ||
        handoff_status->combine_payload_descriptor != combine_descriptor ||
        handoff_status->combine_payload_descriptor !=
            combine_payload_status->combine_payload_descriptor ||
        handoff_status->combine_payload_descriptor != transfer_status->combine_payload_descriptor ||
        handoff_status->combine_payload_descriptor != completion_status->combine_payload_descriptor ||
        handoff_status->shared_token == 0U ||
        handoff_status->shared_token != combine_descriptor->shared_token ||
        handoff_status->shared_token != combine_payload_status->shared_token ||
        handoff_status->shared_token != transfer_status->shared_token ||
        handoff_status->shared_token != completion_status->shared_token) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_DESCRIPTOR_TOKEN_MISMATCH;
    }
    if (request->comm_descriptor != nullptr &&
        (handoff_status->rank != request->comm_descriptor->rank ||
         handoff_status->device_id != request->comm_descriptor->device_id ||
         handoff_status->world_size != request->comm_descriptor->world_size ||
         handoff_status->rank != combine_payload_status->rank ||
         handoff_status->device_id != combine_payload_status->device_id ||
         handoff_status->world_size != combine_payload_status->world_size ||
         handoff_status->rank != transfer_status->rank ||
         handoff_status->device_id != transfer_status->device_id ||
         handoff_status->world_size != transfer_status->world_size ||
         handoff_status->rank != completion_status->rank ||
         handoff_status->device_id != completion_status->device_id ||
         handoff_status->world_size != completion_status->world_size)) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RANK_DEVICE_MISMATCH;
    }

    const uint32_t propagated_handoff_failures =
        handoff_status->failure_fields &
        (PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_INVOCATION_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RUNTIME_PATH_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_DESCRIPTOR_TOKEN_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RANK_DEVICE_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_STATUS_SINK_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_PUBLIC_API_SOURCED_STATE |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_FABRICATED_PASS_EVIDENCE |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_REQUEST_SCAFFOLD |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_DISPATCH_REQUEST_SCAFFOLD |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_REQUEST_SCAFFOLD |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD);
    if (propagated_handoff_failures != 0U) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    propagated_handoff_failures;
    }

    switch (handoff_status->status) {
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_OWNER_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_INVOCATION_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_INVOCATION_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_COMPLETION_SCAFFOLD_MISMATCH:
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_TRANSFER_SCAFFOLD_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_DESCRIPTOR_TOKEN_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_DESCRIPTOR_TOKEN_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_RANK_DEVICE_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RANK_DEVICE_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_STATUS_SINK_MISMATCH:
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_STATUS_SINK_UNBOUND:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_STATUS_SINK_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_PUBLIC_API_SOURCED_STATE:
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_PROVENANCE_SOURCED_STATE:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_PUBLIC_API_SOURCED_STATE;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_FABRICATED_PASS_EVIDENCE:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_FABRICATED_PASS_EVIDENCE |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_FABRICATED_OR_UNTRUSTED_PASS_EVIDENCE;
            break;
        default:
            break;
    }
    return failures;
}

inline uint32_t
pto_cuda_runtime_fusion_validate_runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status(
    const PtoCudaRuntimeFusionRequest *request, const PtoCudaRuntimeFusionCoordinator *coordinator
) {
    const PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadTransferCompletionHandoffResultScaffoldStatus
        *result_status =
            &coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status;
    const PtoCudaUcclEpRuntimePath *runtime_path = &coordinator->descriptor_allocation.runtime_path;
    const PtoCudaUcclEpRuntimeDescriptorView *combine_descriptor = runtime_path->combine_descriptor;
    const PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadScaffoldStatus
        *combine_payload_status =
            &coordinator->runtime_dispatch_driver_backend_combine_payload_scaffold_status;
    const PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadTransferScaffoldStatus
        *transfer_status =
            &coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status;
    const PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadTransferCompletionScaffoldStatus
        *completion_status =
            &coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status;
    const PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadTransferCompletionHandoffScaffoldStatus
        *handoff_status =
            &coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status;
    uint32_t failures = 0;

    if (result_status->version !=
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_SCAFFOLD_STATUS_VERSION) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH;
    }
    if (result_status->invocation_id != request->invocation_id) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_INVOCATION_MISMATCH;
    }
    if (result_status->result_owner != coordinator) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH;
    }
    if (result_status->backend_request_status !=
        &coordinator->runtime_dispatch_driver_backend_request_scaffold_status) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_REQUEST_SCAFFOLD;
    }
    if (result_status->dispatch_request_status !=
        &coordinator->runtime_dispatch_driver_backend_dispatch_request_scaffold_status) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_DISPATCH_REQUEST_SCAFFOLD;
    }
    if (result_status->combine_request_status !=
        &coordinator->runtime_dispatch_driver_backend_combine_request_scaffold_status) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_REQUEST_SCAFFOLD;
    }
    if (result_status->combine_payload_status != combine_payload_status ||
        result_status->transfer_status != transfer_status ||
        result_status->completion_status != completion_status ||
        result_status->handoff_status != handoff_status) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD;
    }
    if (result_status->runtime_path != runtime_path) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RUNTIME_PATH_MISMATCH;
    }
    if (result_status->status_sink == nullptr ||
        result_status->status_sink != request->output_sink) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_STATUS_SINK_MISMATCH;
    }
    if (result_status->result_sink == nullptr ||
        result_status->result_sink != request->output_sink) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_STATUS_SINK_MISMATCH;
    }
    if (combine_descriptor == nullptr ||
        result_status->combine_payload_descriptor != combine_descriptor ||
        result_status->combine_payload_descriptor !=
            combine_payload_status->combine_payload_descriptor ||
        result_status->combine_payload_descriptor != transfer_status->combine_payload_descriptor ||
        result_status->combine_payload_descriptor != completion_status->combine_payload_descriptor ||
        result_status->combine_payload_descriptor != handoff_status->combine_payload_descriptor ||
        result_status->shared_token == 0U ||
        result_status->shared_token != combine_descriptor->shared_token ||
        result_status->shared_token != combine_payload_status->shared_token ||
        result_status->shared_token != transfer_status->shared_token ||
        result_status->shared_token != completion_status->shared_token ||
        result_status->shared_token != handoff_status->shared_token) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_DESCRIPTOR_TOKEN_MISMATCH;
    }
    if (request->comm_descriptor != nullptr &&
        (result_status->rank != request->comm_descriptor->rank ||
         result_status->device_id != request->comm_descriptor->device_id ||
         result_status->world_size != request->comm_descriptor->world_size ||
         result_status->rank != combine_payload_status->rank ||
         result_status->device_id != combine_payload_status->device_id ||
         result_status->world_size != combine_payload_status->world_size ||
         result_status->rank != transfer_status->rank ||
         result_status->device_id != transfer_status->device_id ||
         result_status->world_size != transfer_status->world_size ||
         result_status->rank != completion_status->rank ||
         result_status->device_id != completion_status->device_id ||
         result_status->world_size != completion_status->world_size ||
         result_status->rank != handoff_status->rank ||
         result_status->device_id != handoff_status->device_id ||
         result_status->world_size != handoff_status->world_size)) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RANK_DEVICE_MISMATCH;
    }

    const uint32_t propagated_result_failures =
        result_status->failure_fields &
        (PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_INVOCATION_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RUNTIME_PATH_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_DESCRIPTOR_TOKEN_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RANK_DEVICE_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_STATUS_SINK_MISMATCH |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_PUBLIC_API_SOURCED_STATE |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_FABRICATED_PASS_EVIDENCE |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_REQUEST_SCAFFOLD |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_DISPATCH_REQUEST_SCAFFOLD |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_REQUEST_SCAFFOLD |
         PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD);
    if (propagated_result_failures != 0U) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                    propagated_result_failures;
    }

    switch (result_status->status) {
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_OWNER_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_INVOCATION_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_INVOCATION_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_BACKEND_REQUEST_SCAFFOLD_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_REQUEST_SCAFFOLD;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_DISPATCH_REQUEST_SCAFFOLD_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_DISPATCH_REQUEST_SCAFFOLD;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_COMBINE_REQUEST_SCAFFOLD_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_REQUEST_SCAFFOLD;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_PAYLOAD_SCAFFOLD_MISMATCH:
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_TRANSFER_SCAFFOLD_MISMATCH:
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_COMPLETION_SCAFFOLD_MISMATCH:
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_HANDOFF_SCAFFOLD_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_DESCRIPTOR_TOKEN_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_DESCRIPTOR_TOKEN_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_RANK_DEVICE_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RANK_DEVICE_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_STATUS_SINK_MISMATCH:
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_STATUS_SINK_UNBOUND:
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_RESULT_SINK_MISMATCH:
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_RESULT_SINK_UNBOUND:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_STATUS_SINK_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_PUBLIC_API_SOURCED_STATE:
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_PROVENANCE_SOURCED_STATE:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_PUBLIC_API_SOURCED_STATE;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_FABRICATED_PASS_EVIDENCE:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_FABRICATED_PASS_EVIDENCE |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_FABRICATED_OR_UNTRUSTED_PASS_EVIDENCE;
            break;
        default:
            break;
    }
    return failures;
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
    const PtoCudaUcclEpRuntimeDispatchDriverScaffoldStatus *driver_status =
        &coordinator->runtime_dispatch_driver_scaffold_status;
    if (driver_status->version !=
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_SCAFFOLD_STATUS_VERSION) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH;
    }
    if (driver_status->invocation_id != request->invocation_id) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_INVOCATION_MISMATCH;
    }
    if (driver_status->driver_owner != coordinator ||
        driver_status->handoff_status !=
            &coordinator->runtime_dispatch_request_handoff_scaffold_status ||
        driver_status->driver_state !=
            &coordinator->runtime_dispatch_request_handoff_driver_state) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH;
    }
    if (driver_status->runtime_path != &coordinator->descriptor_allocation.runtime_path) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RUNTIME_PATH_MISMATCH;
    }
    if (driver_status->output_sink == nullptr || driver_status->output_sink != request->output_sink) {
        failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_STATUS_SINK_MISMATCH;
    }
    failures |= driver_status->failure_fields &
                (PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH |
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_INVOCATION_MISMATCH |
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RUNTIME_PATH_MISMATCH |
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_DESCRIPTOR_TOKEN_MISMATCH |
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RANK_DEVICE_MISMATCH |
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_STATUS_SINK_MISMATCH |
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_PUBLIC_API_SOURCED_STATE |
                 PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_FABRICATED_PASS_EVIDENCE);
    switch (driver_status->status) {
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_STATUS_OWNER_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_STATUS_INVOCATION_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_INVOCATION_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_STATUS_RUNTIME_PATH_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RUNTIME_PATH_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_STATUS_DESCRIPTOR_TOKEN_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_DESCRIPTOR_TOKEN_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_STATUS_RANK_DEVICE_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RANK_DEVICE_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_STATUS_STATUS_SINK_MISMATCH:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_STATUS_SINK_MISMATCH;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_STATUS_PUBLIC_API_SOURCED_STATE:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_PUBLIC_API_SOURCED_STATE;
            break;
        case PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_STATUS_FABRICATED_PASS_EVIDENCE:
            failures |= PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_FABRICATED_PASS_EVIDENCE |
                        PTO_CUDA_RUNTIME_FUSION_FAILURE_FABRICATED_OR_UNTRUSTED_PASS_EVIDENCE;
            break;
        default:
            break;
    }
    failures |= pto_cuda_runtime_fusion_validate_runtime_dispatch_driver_backend_scaffold_status(
        request, coordinator
    );
    failures |=
        pto_cuda_runtime_fusion_validate_runtime_dispatch_driver_backend_request_scaffold_status(
            request, coordinator
        );
    failures |=
        pto_cuda_runtime_fusion_validate_runtime_dispatch_driver_backend_dispatch_request_scaffold_status(
            request, coordinator
        );
    failures |=
        pto_cuda_runtime_fusion_validate_runtime_dispatch_driver_backend_combine_request_scaffold_status(
            request, coordinator
        );
    failures |=
        pto_cuda_runtime_fusion_validate_runtime_dispatch_driver_backend_combine_payload_scaffold_status(
            request, coordinator
        );
    failures |=
        pto_cuda_runtime_fusion_validate_runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status(
            request, coordinator
        );
    failures |=
        pto_cuda_runtime_fusion_validate_runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status(
            request, coordinator
        );
    failures |=
        pto_cuda_runtime_fusion_validate_runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status(
            request, coordinator
        );
    failures |=
        pto_cuda_runtime_fusion_validate_runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status(
            request, coordinator
        );
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

inline int pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_scaffold_status(
    const PtoCudaRuntimeFusionRequest *request, PtoCudaRuntimeFusionCoordinator *coordinator
) {
    if (request == nullptr || coordinator == nullptr ||
        coordinator->version != PTO_CUDA_RUNTIME_FUSION_COORDINATOR_VERSION ||
        coordinator->invocation_id != request->invocation_id) {
        return -1;
    }

    coordinator->runtime_dispatch_driver_scaffold_status = {};
    coordinator->runtime_dispatch_driver_scaffold_status.version =
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_SCAFFOLD_STATUS_VERSION;
    coordinator->runtime_dispatch_driver_scaffold_status.invocation_id =
        request->invocation_id;
    coordinator->runtime_dispatch_driver_scaffold_status.driver_owner = coordinator;
    coordinator->runtime_dispatch_driver_scaffold_status.handoff_status =
        &coordinator->runtime_dispatch_request_handoff_scaffold_status;
    coordinator->runtime_dispatch_driver_scaffold_status.driver_state =
        &coordinator->runtime_dispatch_request_handoff_driver_state;
    coordinator->runtime_dispatch_driver_scaffold_status.runtime_path =
        &coordinator->descriptor_allocation.runtime_path;
    coordinator->runtime_dispatch_driver_scaffold_status.output_sink =
        coordinator->output_sink;
    coordinator->runtime_dispatch_driver_scaffold_status.status =
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_STATUS_UNSUPPORTED_BOUNDARY;
    coordinator->runtime_dispatch_driver_scaffold_status.failure_fields =
        PTO_CUDA_RUNTIME_FUSION_FAILURE_UNSUPPORTED_BOUNDARY;
    return 0;
}

inline int pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_scaffold_status(
    const PtoCudaRuntimeFusionRequest *request, PtoCudaRuntimeFusionCoordinator *coordinator
) {
    if (request == nullptr || coordinator == nullptr ||
        coordinator->version != PTO_CUDA_RUNTIME_FUSION_COORDINATOR_VERSION ||
        coordinator->invocation_id != request->invocation_id) {
        return -1;
    }

    coordinator->runtime_dispatch_driver_backend_scaffold_status = {};
    coordinator->runtime_dispatch_driver_backend_scaffold_status.version =
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_SCAFFOLD_STATUS_VERSION;
    coordinator->runtime_dispatch_driver_backend_scaffold_status.invocation_id =
        request->invocation_id;
    coordinator->runtime_dispatch_driver_backend_scaffold_status.backend_owner = coordinator;
    coordinator->runtime_dispatch_driver_backend_scaffold_status.driver_status =
        &coordinator->runtime_dispatch_driver_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_scaffold_status.runtime_path =
        &coordinator->descriptor_allocation.runtime_path;
    coordinator->runtime_dispatch_driver_backend_scaffold_status.status_sink =
        coordinator->output_sink;
    coordinator->runtime_dispatch_driver_backend_scaffold_status.shared_token =
        coordinator->descriptor_allocation.dispatch_descriptor.shared_token;
    coordinator->runtime_dispatch_driver_backend_scaffold_status.rank =
        coordinator->descriptor_allocation.dispatch_descriptor.rank;
    coordinator->runtime_dispatch_driver_backend_scaffold_status.device_id =
        coordinator->descriptor_allocation.dispatch_descriptor.device_id;
    coordinator->runtime_dispatch_driver_backend_scaffold_status.world_size =
        coordinator->descriptor_allocation.dispatch_descriptor.world_size;
    coordinator->runtime_dispatch_driver_backend_scaffold_status.status =
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_STATUS_UNSUPPORTED_BOUNDARY;
    coordinator->runtime_dispatch_driver_backend_scaffold_status.failure_fields =
        PTO_CUDA_RUNTIME_FUSION_FAILURE_UNSUPPORTED_BOUNDARY;
    return 0;
}

inline int pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_request_scaffold_status(
    const PtoCudaRuntimeFusionRequest *request, PtoCudaRuntimeFusionCoordinator *coordinator
) {
    if (request == nullptr || coordinator == nullptr ||
        coordinator->version != PTO_CUDA_RUNTIME_FUSION_COORDINATOR_VERSION ||
        coordinator->invocation_id != request->invocation_id) {
        return -1;
    }

    coordinator->runtime_dispatch_driver_backend_request_scaffold_status = {};
    coordinator->runtime_dispatch_driver_backend_request_scaffold_status.version =
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_REQUEST_SCAFFOLD_STATUS_VERSION;
    coordinator->runtime_dispatch_driver_backend_request_scaffold_status.invocation_id =
        request->invocation_id;
    coordinator->runtime_dispatch_driver_backend_request_scaffold_status.request_owner =
        coordinator;
    coordinator->runtime_dispatch_driver_backend_request_scaffold_status.backend_status =
        &coordinator->runtime_dispatch_driver_backend_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_request_scaffold_status.runtime_path =
        &coordinator->descriptor_allocation.runtime_path;
    coordinator->runtime_dispatch_driver_backend_request_scaffold_status.status_sink =
        coordinator->output_sink;
    coordinator->runtime_dispatch_driver_backend_request_scaffold_status.shared_token =
        coordinator->descriptor_allocation.dispatch_descriptor.shared_token;
    coordinator->runtime_dispatch_driver_backend_request_scaffold_status.rank =
        coordinator->descriptor_allocation.dispatch_descriptor.rank;
    coordinator->runtime_dispatch_driver_backend_request_scaffold_status.device_id =
        coordinator->descriptor_allocation.dispatch_descriptor.device_id;
    coordinator->runtime_dispatch_driver_backend_request_scaffold_status.world_size =
        coordinator->descriptor_allocation.dispatch_descriptor.world_size;
    coordinator->runtime_dispatch_driver_backend_request_scaffold_status.status =
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_REQUEST_STATUS_UNSUPPORTED_BOUNDARY;
    coordinator->runtime_dispatch_driver_backend_request_scaffold_status.failure_fields =
        PTO_CUDA_RUNTIME_FUSION_FAILURE_UNSUPPORTED_BOUNDARY;
    return 0;
}

inline int
pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_dispatch_request_scaffold_status(
    const PtoCudaRuntimeFusionRequest *request, PtoCudaRuntimeFusionCoordinator *coordinator
) {
    if (request == nullptr || coordinator == nullptr ||
        coordinator->version != PTO_CUDA_RUNTIME_FUSION_COORDINATOR_VERSION ||
        coordinator->invocation_id != request->invocation_id) {
        return -1;
    }

    coordinator->runtime_dispatch_driver_backend_dispatch_request_scaffold_status = {};
    coordinator->runtime_dispatch_driver_backend_dispatch_request_scaffold_status.version =
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_DISPATCH_REQUEST_SCAFFOLD_STATUS_VERSION;
    coordinator->runtime_dispatch_driver_backend_dispatch_request_scaffold_status.invocation_id =
        request->invocation_id;
    coordinator->runtime_dispatch_driver_backend_dispatch_request_scaffold_status.request_owner =
        coordinator;
    coordinator->runtime_dispatch_driver_backend_dispatch_request_scaffold_status.backend_request_status =
        &coordinator->runtime_dispatch_driver_backend_request_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_dispatch_request_scaffold_status.runtime_path =
        &coordinator->descriptor_allocation.runtime_path;
    coordinator->runtime_dispatch_driver_backend_dispatch_request_scaffold_status.status_sink =
        coordinator->output_sink;
    coordinator->runtime_dispatch_driver_backend_dispatch_request_scaffold_status.shared_token =
        coordinator->descriptor_allocation.dispatch_descriptor.shared_token;
    coordinator->runtime_dispatch_driver_backend_dispatch_request_scaffold_status.rank =
        coordinator->descriptor_allocation.dispatch_descriptor.rank;
    coordinator->runtime_dispatch_driver_backend_dispatch_request_scaffold_status.device_id =
        coordinator->descriptor_allocation.dispatch_descriptor.device_id;
    coordinator->runtime_dispatch_driver_backend_dispatch_request_scaffold_status.world_size =
        coordinator->descriptor_allocation.dispatch_descriptor.world_size;
    coordinator->runtime_dispatch_driver_backend_dispatch_request_scaffold_status.status =
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_DISPATCH_REQUEST_STATUS_UNSUPPORTED_BOUNDARY;
    coordinator->runtime_dispatch_driver_backend_dispatch_request_scaffold_status.failure_fields =
        PTO_CUDA_RUNTIME_FUSION_FAILURE_UNSUPPORTED_BOUNDARY;
    return 0;
}

inline int
pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_combine_request_scaffold_status(
    const PtoCudaRuntimeFusionRequest *request, PtoCudaRuntimeFusionCoordinator *coordinator
) {
    if (request == nullptr || coordinator == nullptr ||
        coordinator->version != PTO_CUDA_RUNTIME_FUSION_COORDINATOR_VERSION ||
        coordinator->invocation_id != request->invocation_id) {
        return -1;
    }

    coordinator->runtime_dispatch_driver_backend_combine_request_scaffold_status = {};
    coordinator->runtime_dispatch_driver_backend_combine_request_scaffold_status.version =
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_REQUEST_SCAFFOLD_STATUS_VERSION;
    coordinator->runtime_dispatch_driver_backend_combine_request_scaffold_status.invocation_id =
        request->invocation_id;
    coordinator->runtime_dispatch_driver_backend_combine_request_scaffold_status.request_owner =
        coordinator;
    coordinator->runtime_dispatch_driver_backend_combine_request_scaffold_status.backend_request_status =
        &coordinator->runtime_dispatch_driver_backend_request_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_combine_request_scaffold_status.dispatch_request_status =
        &coordinator->runtime_dispatch_driver_backend_dispatch_request_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_combine_request_scaffold_status.runtime_path =
        &coordinator->descriptor_allocation.runtime_path;
    coordinator->runtime_dispatch_driver_backend_combine_request_scaffold_status.status_sink =
        coordinator->output_sink;
    coordinator->runtime_dispatch_driver_backend_combine_request_scaffold_status.shared_token =
        coordinator->descriptor_allocation.combine_descriptor.shared_token;
    coordinator->runtime_dispatch_driver_backend_combine_request_scaffold_status.rank =
        coordinator->descriptor_allocation.combine_descriptor.rank;
    coordinator->runtime_dispatch_driver_backend_combine_request_scaffold_status.device_id =
        coordinator->descriptor_allocation.combine_descriptor.device_id;
    coordinator->runtime_dispatch_driver_backend_combine_request_scaffold_status.world_size =
        coordinator->descriptor_allocation.combine_descriptor.world_size;
    coordinator->runtime_dispatch_driver_backend_combine_request_scaffold_status.status =
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_REQUEST_STATUS_UNSUPPORTED_BOUNDARY;
    coordinator->runtime_dispatch_driver_backend_combine_request_scaffold_status.failure_fields =
        PTO_CUDA_RUNTIME_FUSION_FAILURE_UNSUPPORTED_BOUNDARY;
    return 0;
}

inline int
pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_combine_payload_scaffold_status(
    const PtoCudaRuntimeFusionRequest *request, PtoCudaRuntimeFusionCoordinator *coordinator
) {
    if (request == nullptr || coordinator == nullptr ||
        coordinator->version != PTO_CUDA_RUNTIME_FUSION_COORDINATOR_VERSION ||
        coordinator->invocation_id != request->invocation_id) {
        return -1;
    }

    coordinator->runtime_dispatch_driver_backend_combine_payload_scaffold_status = {};
    coordinator->runtime_dispatch_driver_backend_combine_payload_scaffold_status.version =
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD_STATUS_VERSION;
    coordinator->runtime_dispatch_driver_backend_combine_payload_scaffold_status.invocation_id =
        request->invocation_id;
    coordinator->runtime_dispatch_driver_backend_combine_payload_scaffold_status.payload_owner =
        coordinator;
    coordinator->runtime_dispatch_driver_backend_combine_payload_scaffold_status.combine_request_status =
        &coordinator->runtime_dispatch_driver_backend_combine_request_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_combine_payload_scaffold_status.runtime_path =
        &coordinator->descriptor_allocation.runtime_path;
    coordinator->runtime_dispatch_driver_backend_combine_payload_scaffold_status.status_sink =
        coordinator->output_sink;
    coordinator->runtime_dispatch_driver_backend_combine_payload_scaffold_status.combine_payload_descriptor =
        coordinator->descriptor_allocation.runtime_path.combine_descriptor;
    coordinator->runtime_dispatch_driver_backend_combine_payload_scaffold_status.shared_token =
        coordinator->descriptor_allocation.combine_descriptor.shared_token;
    coordinator->runtime_dispatch_driver_backend_combine_payload_scaffold_status.rank =
        coordinator->descriptor_allocation.combine_descriptor.rank;
    coordinator->runtime_dispatch_driver_backend_combine_payload_scaffold_status.device_id =
        coordinator->descriptor_allocation.combine_descriptor.device_id;
    coordinator->runtime_dispatch_driver_backend_combine_payload_scaffold_status.world_size =
        coordinator->descriptor_allocation.combine_descriptor.world_size;
    coordinator->runtime_dispatch_driver_backend_combine_payload_scaffold_status.status =
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_STATUS_UNSUPPORTED_BOUNDARY;
    coordinator->runtime_dispatch_driver_backend_combine_payload_scaffold_status.failure_fields =
        PTO_CUDA_RUNTIME_FUSION_FAILURE_UNSUPPORTED_BOUNDARY;
    return 0;
}

inline int
pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status(
    const PtoCudaRuntimeFusionRequest *request, PtoCudaRuntimeFusionCoordinator *coordinator
) {
    if (request == nullptr || coordinator == nullptr ||
        coordinator->version != PTO_CUDA_RUNTIME_FUSION_COORDINATOR_VERSION ||
        coordinator->invocation_id != request->invocation_id) {
        return -1;
    }

    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status = {};
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.version =
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_SCAFFOLD_STATUS_VERSION;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.invocation_id =
        request->invocation_id;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.transfer_owner =
        coordinator;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.backend_request_status =
        &coordinator->runtime_dispatch_driver_backend_request_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.dispatch_request_status =
        &coordinator->runtime_dispatch_driver_backend_dispatch_request_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.combine_request_status =
        &coordinator->runtime_dispatch_driver_backend_combine_request_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.combine_payload_status =
        &coordinator->runtime_dispatch_driver_backend_combine_payload_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.runtime_path =
        &coordinator->descriptor_allocation.runtime_path;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.status_sink =
        coordinator->output_sink;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.combine_payload_descriptor =
        coordinator->descriptor_allocation.runtime_path.combine_descriptor;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.shared_token =
        coordinator->descriptor_allocation.combine_descriptor.shared_token;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.rank =
        coordinator->descriptor_allocation.combine_descriptor.rank;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.device_id =
        coordinator->descriptor_allocation.combine_descriptor.device_id;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.world_size =
        coordinator->descriptor_allocation.combine_descriptor.world_size;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.status =
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_STATUS_UNSUPPORTED_BOUNDARY;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status.failure_fields =
        PTO_CUDA_RUNTIME_FUSION_FAILURE_UNSUPPORTED_BOUNDARY;
    return 0;
}

inline int
pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status(
    const PtoCudaRuntimeFusionRequest *request, PtoCudaRuntimeFusionCoordinator *coordinator
) {
    if (request == nullptr || coordinator == nullptr ||
        coordinator->version != PTO_CUDA_RUNTIME_FUSION_COORDINATOR_VERSION ||
        coordinator->invocation_id != request->invocation_id) {
        return -1;
    }

    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status = {};
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.version =
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_SCAFFOLD_STATUS_VERSION;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.invocation_id =
        request->invocation_id;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.completion_owner =
        coordinator;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.backend_request_status =
        &coordinator->runtime_dispatch_driver_backend_request_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.dispatch_request_status =
        &coordinator->runtime_dispatch_driver_backend_dispatch_request_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.combine_request_status =
        &coordinator->runtime_dispatch_driver_backend_combine_request_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.combine_payload_status =
        &coordinator->runtime_dispatch_driver_backend_combine_payload_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.transfer_status =
        &coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.runtime_path =
        &coordinator->descriptor_allocation.runtime_path;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.status_sink =
        coordinator->output_sink;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.completion_sink =
        coordinator->output_sink;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.combine_payload_descriptor =
        coordinator->descriptor_allocation.runtime_path.combine_descriptor;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.shared_token =
        coordinator->descriptor_allocation.combine_descriptor.shared_token;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.rank =
        coordinator->descriptor_allocation.combine_descriptor.rank;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.device_id =
        coordinator->descriptor_allocation.combine_descriptor.device_id;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.world_size =
        coordinator->descriptor_allocation.combine_descriptor.world_size;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.status =
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_STATUS_UNSUPPORTED_BOUNDARY;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status.failure_fields =
        PTO_CUDA_RUNTIME_FUSION_FAILURE_UNSUPPORTED_BOUNDARY;
    return 0;
}

inline int
pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status(
    const PtoCudaRuntimeFusionRequest *request, PtoCudaRuntimeFusionCoordinator *coordinator
) {
    if (request == nullptr || coordinator == nullptr ||
        coordinator->version != PTO_CUDA_RUNTIME_FUSION_COORDINATOR_VERSION ||
        coordinator->invocation_id != request->invocation_id) {
        return -1;
    }

    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status = {};
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.version =
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_SCAFFOLD_STATUS_VERSION;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.invocation_id =
        request->invocation_id;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.handoff_owner =
        coordinator;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.backend_request_status =
        &coordinator->runtime_dispatch_driver_backend_request_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.dispatch_request_status =
        &coordinator->runtime_dispatch_driver_backend_dispatch_request_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.combine_request_status =
        &coordinator->runtime_dispatch_driver_backend_combine_request_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.combine_payload_status =
        &coordinator->runtime_dispatch_driver_backend_combine_payload_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.transfer_status =
        &coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.completion_status =
        &coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.runtime_path =
        &coordinator->descriptor_allocation.runtime_path;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.status_sink =
        coordinator->output_sink;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.handoff_sink =
        coordinator->output_sink;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.combine_payload_descriptor =
        coordinator->descriptor_allocation.runtime_path.combine_descriptor;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.shared_token =
        coordinator->descriptor_allocation.combine_descriptor.shared_token;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.rank =
        coordinator->descriptor_allocation.combine_descriptor.rank;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.device_id =
        coordinator->descriptor_allocation.combine_descriptor.device_id;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.world_size =
        coordinator->descriptor_allocation.combine_descriptor.world_size;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.status =
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_STATUS_UNSUPPORTED_BOUNDARY;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status.failure_fields =
        PTO_CUDA_RUNTIME_FUSION_FAILURE_UNSUPPORTED_BOUNDARY;
    return 0;
}

inline int
pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status(
    const PtoCudaRuntimeFusionRequest *request, PtoCudaRuntimeFusionCoordinator *coordinator
) {
    if (request == nullptr || coordinator == nullptr ||
        coordinator->version != PTO_CUDA_RUNTIME_FUSION_COORDINATOR_VERSION ||
        coordinator->invocation_id != request->invocation_id) {
        return -1;
    }

    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status =
        {};
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.version =
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_SCAFFOLD_STATUS_VERSION;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.invocation_id =
        request->invocation_id;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.result_owner =
        coordinator;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.backend_request_status =
        &coordinator->runtime_dispatch_driver_backend_request_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.dispatch_request_status =
        &coordinator->runtime_dispatch_driver_backend_dispatch_request_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.combine_request_status =
        &coordinator->runtime_dispatch_driver_backend_combine_request_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.combine_payload_status =
        &coordinator->runtime_dispatch_driver_backend_combine_payload_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.transfer_status =
        &coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.completion_status =
        &coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.handoff_status =
        &coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.runtime_path =
        &coordinator->descriptor_allocation.runtime_path;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.status_sink =
        coordinator->output_sink;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.result_sink =
        coordinator->output_sink;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.combine_payload_descriptor =
        coordinator->descriptor_allocation.runtime_path.combine_descriptor;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.shared_token =
        coordinator->descriptor_allocation.combine_descriptor.shared_token;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.rank =
        coordinator->descriptor_allocation.combine_descriptor.rank;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.device_id =
        coordinator->descriptor_allocation.combine_descriptor.device_id;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.world_size =
        coordinator->descriptor_allocation.combine_descriptor.world_size;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.status =
        PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_STATUS_UNSUPPORTED_BOUNDARY;
    coordinator->runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status.failure_fields =
        PTO_CUDA_RUNTIME_FUSION_FAILURE_UNSUPPORTED_BOUNDARY;
    return 0;
}

inline int pto_cuda_runtime_fusion_failure_is_runtime_dispatch_scaffold_failed(uint32_t failures) {
    return (failures & PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_RUNTIME_DISPATCH_SCAFFOLD) != 0U;
}

inline int pto_cuda_runtime_fusion_failure_is_runtime_dispatch_handoff_failed(uint32_t failures) {
    return (failures & PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_RUNTIME_DISPATCH_HANDOFF_DRIVER) != 0U;
}

inline int pto_cuda_runtime_fusion_failure_is_runtime_dispatch_driver_backend_failed(
    uint32_t failures
) {
    return (failures & PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_SCAFFOLD) != 0U;
}

inline int pto_cuda_runtime_fusion_failure_is_runtime_dispatch_driver_backend_request_failed(
    uint32_t failures
) {
    return (failures & PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_REQUEST_SCAFFOLD) != 0U;
}

inline int
pto_cuda_runtime_fusion_failure_is_runtime_dispatch_driver_backend_dispatch_request_failed(
    uint32_t failures
) {
    return (failures &
            PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_DISPATCH_REQUEST_SCAFFOLD) != 0U;
}

inline int
pto_cuda_runtime_fusion_failure_is_runtime_dispatch_driver_backend_combine_request_failed(
    uint32_t failures
) {
    return (failures &
            PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_REQUEST_SCAFFOLD) != 0U;
}

inline int
pto_cuda_runtime_fusion_failure_is_runtime_dispatch_driver_backend_combine_payload_failed(
    uint32_t failures
) {
    return (failures &
            PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD) != 0U;
}

inline int pto_cuda_runtime_fusion_failure_is_runtime_dispatch_driver_failed(uint32_t failures) {
    const uint32_t driver_failed_mask =
        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_OWNER_MISMATCH |
        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_INVOCATION_MISMATCH |
        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RUNTIME_PATH_MISMATCH |
        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_DESCRIPTOR_TOKEN_MISMATCH |
        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_RANK_DEVICE_MISMATCH |
        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_STATUS_SINK_MISMATCH |
        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_PUBLIC_API_SOURCED_STATE |
        PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_FABRICATED_PASS_EVIDENCE;
    return (failures & driver_failed_mask) != 0U;
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
    } else if (pto_cuda_runtime_fusion_failure_is_runtime_dispatch_driver_backend_dispatch_request_failed(
                   out.failure_fields
               )) {
        out.status = PTO_CUDA_RUNTIME_FUSION_STATUS_FAILED;
        out.reason =
            "private UCCL-EP runtime dispatch driver backend dispatch request scaffold/status validation failed";
    } else if (pto_cuda_runtime_fusion_failure_is_runtime_dispatch_driver_backend_combine_request_failed(
                   out.failure_fields
               )) {
        out.status = PTO_CUDA_RUNTIME_FUSION_STATUS_FAILED;
        out.reason =
            "private UCCL-EP runtime dispatch driver backend combine request scaffold/status validation failed";
    } else if (pto_cuda_runtime_fusion_failure_is_runtime_dispatch_driver_backend_combine_payload_failed(
                   out.failure_fields
               )) {
        out.status = PTO_CUDA_RUNTIME_FUSION_STATUS_FAILED;
        out.reason =
            "private UCCL-EP runtime dispatch driver backend combine payload scaffold/status validation failed";
    } else if (pto_cuda_runtime_fusion_failure_is_runtime_dispatch_driver_backend_request_failed(
                   out.failure_fields
               )) {
        out.status = PTO_CUDA_RUNTIME_FUSION_STATUS_FAILED;
        out.reason =
            "private UCCL-EP runtime dispatch driver backend request scaffold/status validation failed";
    } else if (pto_cuda_runtime_fusion_failure_is_runtime_dispatch_driver_backend_failed(out.failure_fields)) {
        out.status = PTO_CUDA_RUNTIME_FUSION_STATUS_FAILED;
        out.reason = "private UCCL-EP runtime dispatch driver backend scaffold/status validation failed";
    } else if (pto_cuda_runtime_fusion_failure_is_runtime_dispatch_driver_failed(out.failure_fields)) {
        out.status = PTO_CUDA_RUNTIME_FUSION_STATUS_FAILED;
        out.reason = "private UCCL-EP runtime dispatch driver scaffold/status validation failed";
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
