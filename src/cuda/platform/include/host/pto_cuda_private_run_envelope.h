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

#ifndef SRC_CUDA_PLATFORM_INCLUDE_HOST_PTO_CUDA_PRIVATE_RUN_ENVELOPE_H_
#define SRC_CUDA_PLATFORM_INCLUDE_HOST_PTO_CUDA_PRIVATE_RUN_ENVELOPE_H_

#include "host/pto_cuda_persistent_device_abi.h"
#include "task_interface/task_args.h"

#include <stddef.h>
#include <stdint.h>

static const uint32_t PTO_CUDA_PRIVATE_RUN_ENVELOPE_VERSION = 2;

enum PtoCudaPrivateRunEnvelopeStatus : int32_t {
    PTO_CUDA_PRIVATE_RUN_ENVELOPE_OK = 0,
    PTO_CUDA_PRIVATE_RUN_ENVELOPE_NULL_POINTER = -1,
    PTO_CUDA_PRIVATE_RUN_ENVELOPE_STALE = -2,
    PTO_CUDA_PRIVATE_RUN_ENVELOPE_CALLABLE_MISMATCH = -3,
    PTO_CUDA_PRIVATE_RUN_ENVELOPE_CROSS_INVOCATION = -4,
    PTO_CUDA_PRIVATE_RUN_ENVELOPE_CALLABLE_TYPE_MISMATCH = -5,
    PTO_CUDA_PRIVATE_RUN_ENVELOPE_WRONG_RUNTIME_ARGS_SIZE = -6,
    PTO_CUDA_PRIVATE_RUN_ENVELOPE_WRONG_CHIP_STORAGE_SIZE = -7,
};

struct PtoCudaPrivateRunArgsEnvelope {
    uint32_t version;
    int32_t callable_id;
    uint64_t invocation_id;
    const void *runtime_task_args;
    size_t runtime_task_args_size;
    const ChipStorageTaskArgs *chip_storage_task_args;
    size_t chip_storage_task_args_size;
};

inline int pto_cuda_private_run_envelope_init(
    PtoCudaPrivateRunArgsEnvelope *envelope, int32_t callable_id, uint64_t invocation_id,
    const PtoCudaPersistentDagArgs *runtime_task_args, size_t runtime_task_args_size,
    const ChipStorageTaskArgs *chip_storage_task_args, size_t chip_storage_task_args_size
) {
    if (envelope == nullptr || runtime_task_args == nullptr || chip_storage_task_args == nullptr) {
        return PTO_CUDA_PRIVATE_RUN_ENVELOPE_NULL_POINTER;
    }
    if (runtime_task_args_size != sizeof(PtoCudaPersistentDagArgs)) {
        return PTO_CUDA_PRIVATE_RUN_ENVELOPE_WRONG_RUNTIME_ARGS_SIZE;
    }
    if (chip_storage_task_args_size != sizeof(ChipStorageTaskArgs)) {
        return PTO_CUDA_PRIVATE_RUN_ENVELOPE_WRONG_CHIP_STORAGE_SIZE;
    }

    envelope->version = PTO_CUDA_PRIVATE_RUN_ENVELOPE_VERSION;
    envelope->callable_id = callable_id;
    envelope->invocation_id = invocation_id;
    envelope->runtime_task_args = runtime_task_args;
    envelope->runtime_task_args_size = runtime_task_args_size;
    envelope->chip_storage_task_args = chip_storage_task_args;
    envelope->chip_storage_task_args_size = chip_storage_task_args_size;
    return PTO_CUDA_PRIVATE_RUN_ENVELOPE_OK;
}

inline int pto_cuda_private_run_envelope_validate(
    const PtoCudaPrivateRunArgsEnvelope *envelope, int32_t expected_callable_id, uint64_t expected_invocation_id,
    uint32_t prepared_callable_op, size_t expected_runtime_task_args_size
) {
    if (envelope == nullptr || envelope->runtime_task_args == nullptr || envelope->chip_storage_task_args == nullptr) {
        return PTO_CUDA_PRIVATE_RUN_ENVELOPE_NULL_POINTER;
    }
    if (envelope->version != PTO_CUDA_PRIVATE_RUN_ENVELOPE_VERSION) {
        return PTO_CUDA_PRIVATE_RUN_ENVELOPE_STALE;
    }
    if (envelope->callable_id != expected_callable_id) {
        return PTO_CUDA_PRIVATE_RUN_ENVELOPE_CALLABLE_MISMATCH;
    }
    if (envelope->invocation_id != expected_invocation_id) {
        return PTO_CUDA_PRIVATE_RUN_ENVELOPE_CROSS_INVOCATION;
    }
    if (prepared_callable_op != PTO_CUDA_PERSISTENT_OP_DAG_F32_RING) {
        return PTO_CUDA_PRIVATE_RUN_ENVELOPE_CALLABLE_TYPE_MISMATCH;
    }
    if (envelope->runtime_task_args_size != expected_runtime_task_args_size) {
        return PTO_CUDA_PRIVATE_RUN_ENVELOPE_WRONG_RUNTIME_ARGS_SIZE;
    }
    if (envelope->chip_storage_task_args_size != sizeof(ChipStorageTaskArgs)) {
        return PTO_CUDA_PRIVATE_RUN_ENVELOPE_WRONG_CHIP_STORAGE_SIZE;
    }
    return PTO_CUDA_PRIVATE_RUN_ENVELOPE_OK;
}

#endif  // SRC_CUDA_PLATFORM_INCLUDE_HOST_PTO_CUDA_PRIVATE_RUN_ENVELOPE_H_
