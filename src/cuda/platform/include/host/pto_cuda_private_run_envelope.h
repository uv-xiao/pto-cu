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

#include "task_interface/task_args.h"

#include <stddef.h>
#include <stdint.h>

static const uint32_t PTO_CUDA_PRIVATE_RUN_ENVELOPE_VERSION = 1;

struct PtoCudaPrivateRunArgsEnvelope {
    uint32_t version;
    const void *runtime_task_args;
    size_t runtime_task_args_size;
    const ChipStorageTaskArgs *chip_storage_task_args;
    size_t chip_storage_task_args_size;
};

#endif  // SRC_CUDA_PLATFORM_INCLUDE_HOST_PTO_CUDA_PRIVATE_RUN_ENVELOPE_H_
