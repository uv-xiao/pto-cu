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

#ifndef SRC_CUDA_PLATFORM_INCLUDE_HOST_PTO_CUDA_COMM_DESCRIPTOR_ABI_H_
#define SRC_CUDA_PLATFORM_INCLUDE_HOST_PTO_CUDA_COMM_DESCRIPTOR_ABI_H_

#include <stddef.h>
#include <stdint.h>

enum PtoCudaCommBackend : uint32_t {
    PTO_CUDA_COMM_BACKEND_MOCK = 0,
    PTO_CUDA_COMM_BACKEND_NCCL = 1,
};

static const size_t PTO_CUDA_COMM_DESCRIPTOR_BYTES = 20;

struct PtoCudaCommDeviceDescriptor {
    uint32_t backend_code;
    uint32_t rank;
    uint32_t device_id;
    uint32_t world_size;
    uint32_t capability_crc32;
};

static_assert(sizeof(PtoCudaCommDeviceDescriptor) == 20, "PtoCudaCommDeviceDescriptor must match Python bytes");

inline uint32_t pto_cuda_comm_load_le_u32(const uint8_t *bytes) {
    return static_cast<uint32_t>(bytes[0]) | (static_cast<uint32_t>(bytes[1]) << 8) |
           (static_cast<uint32_t>(bytes[2]) << 16) | (static_cast<uint32_t>(bytes[3]) << 24);
}

inline int pto_cuda_comm_descriptor_backend_is_supported(uint32_t backend_code) {
    return backend_code == PTO_CUDA_COMM_BACKEND_MOCK || backend_code == PTO_CUDA_COMM_BACKEND_NCCL;
}

inline int pto_cuda_comm_descriptor_from_bytes(
    const void *descriptor_bytes, size_t descriptor_size, PtoCudaCommDeviceDescriptor *out
) {
    if (descriptor_bytes == nullptr || out == nullptr || descriptor_size != PTO_CUDA_COMM_DESCRIPTOR_BYTES) {
        return -1;
    }

    const uint8_t *bytes = static_cast<const uint8_t *>(descriptor_bytes);
    PtoCudaCommDeviceDescriptor descriptor = {
        pto_cuda_comm_load_le_u32(bytes),      pto_cuda_comm_load_le_u32(bytes + 4),
        pto_cuda_comm_load_le_u32(bytes + 8),  pto_cuda_comm_load_le_u32(bytes + 12),
        pto_cuda_comm_load_le_u32(bytes + 16),
    };
    if (!pto_cuda_comm_descriptor_backend_is_supported(descriptor.backend_code)) {
        return -1;
    }
    if (descriptor.world_size == 0 || descriptor.rank >= descriptor.world_size) {
        return -1;
    }

    *out = descriptor;
    return 0;
}

#endif  // SRC_CUDA_PLATFORM_INCLUDE_HOST_PTO_CUDA_COMM_DESCRIPTOR_ABI_H_
