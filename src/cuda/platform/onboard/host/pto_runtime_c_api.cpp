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

#include "pto_runtime_c_api.h"

#include "host/pto_cuda_comm_descriptor_abi.h"
#include "host/pto_cuda_host_schedule_abi.h"
#include "host/pto_cuda_persistent_device_abi.h"
#include "platform_comm/comm.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

struct CommHandle_ {};

namespace {

struct PtoCudaRuntime {
    uint32_t reserved = 0;
};

struct PtoCudaCallableHeader {
    uint32_t version;
    uint32_t op;
    const void *image;
    size_t image_size;
    const char *entry_name;
    uint32_t grid_dim;
    uint32_t block_dim;
    size_t shared_mem_bytes;
};

struct PreparedCallable {
    CUmodule module = nullptr;
    CUfunction function = nullptr;
    uint32_t op = 0;
    uint32_t grid_dim = 0;
    uint32_t block_dim = 0;
    uint32_t stream_id = 0;
    size_t shared_mem_bytes = 0;
};

using ncclResult_t = int;
using ncclComm_t = void *;

struct CudaNcclUniqueId {
    char internal[128];
};

struct CudaNcclApi {
    using NcclGetUniqueIdFn = ncclResult_t (*)(CudaNcclUniqueId *);
    using NcclGetErrorStringFn = const char *(*)(ncclResult_t);
    using NcclCommInitRankFn = ncclResult_t (*)(ncclComm_t *, int, CudaNcclUniqueId, int);
    using NcclAllReduceFn = ncclResult_t (*)(const void *, void *, size_t, int, int, ncclComm_t, cudaStream_t);
    using NcclReduceScatterFn = ncclResult_t (*)(const void *, void *, size_t, int, int, ncclComm_t, cudaStream_t);
    using NcclAllGatherFn = ncclResult_t (*)(const void *, void *, size_t, int, ncclComm_t, cudaStream_t);
    using NcclSendFn = ncclResult_t (*)(const void *, size_t, int, int, ncclComm_t, cudaStream_t);
    using NcclRecvFn = ncclResult_t (*)(void *, size_t, int, int, ncclComm_t, cudaStream_t);
    using NcclGroupStartFn = ncclResult_t (*)();
    using NcclGroupEndFn = ncclResult_t (*)();
    using NcclCommDestroyFn = ncclResult_t (*)(ncclComm_t);

    void *library = nullptr;
    NcclGetUniqueIdFn ncclGetUniqueId = nullptr;
    NcclGetErrorStringFn ncclGetErrorString = nullptr;
    NcclCommInitRankFn ncclCommInitRank = nullptr;
    NcclAllReduceFn ncclAllReduce = nullptr;
    NcclReduceScatterFn ncclReduceScatter = nullptr;
    NcclAllGatherFn ncclAllGather = nullptr;
    NcclSendFn ncclSend = nullptr;
    NcclRecvFn ncclRecv = nullptr;
    NcclGroupStartFn ncclGroupStart = nullptr;
    NcclGroupEndFn ncclGroupEnd = nullptr;
    NcclCommDestroyFn ncclCommDestroy = nullptr;
};

struct CudaCommStream {
    cudaStream_t stream = nullptr;
    PtoCudaCommDeviceDescriptor descriptor = {};
    bool has_descriptor = false;
};

struct CudaCommHandle : CommHandle_ {
    PtoCudaCommDeviceDescriptor descriptor = {};
    cudaStream_t stream = nullptr;
    std::string rootinfo_path;
    const CudaNcclApi *nccl_api = nullptr;
    ncclComm_t nccl_comm = nullptr;
};

constexpr uint32_t kDefaultStreamPoolSize = 4;
constexpr uint32_t kMaxStreamPoolSize = 64;
constexpr int kNcclRootinfoWaitMs = 30000;
constexpr int kNcclRootinfoPollMs = 10;
constexpr int PTO_CUDA_NCCL_FLOAT32 = 7;
constexpr int PTO_CUDA_NCCL_SUM = 0;

thread_local std::string g_comm_last_error;
std::string g_nccl_load_error;

void clear_comm_error() { g_comm_last_error.clear(); }

void set_comm_error(const std::string &msg) { g_comm_last_error = msg; }

std::string cuda_error_message(const char *op, cudaError_t rc) {
    std::string msg = op;
    msg += " failed: ";
    msg += cudaGetErrorString(rc);
    return msg;
}

std::string nccl_error_message(const CudaNcclApi *api, const char *op, ncclResult_t rc) {
    std::string msg = op;
    msg += " failed with code ";
    msg += std::to_string(static_cast<int>(rc));
    if (api != nullptr && api->ncclGetErrorString != nullptr) {
        const char *detail = api->ncclGetErrorString(rc);
        if (detail != nullptr && detail[0] != '\0') {
            msg += ": ";
            msg += detail;
        }
    }
    return msg;
}

uint32_t configured_stream_pool_size() {
    const char *value = std::getenv("PTO_CUDA_STREAM_POOL_SIZE");
    if (value == nullptr || value[0] == '\0') {
        return kDefaultStreamPoolSize;
    }
    char *end = nullptr;
    unsigned long parsed = std::strtoul(value, &end, 10);
    if (end == value || *end != '\0' || parsed == 0 || parsed > kMaxStreamPoolSize) {
        return kDefaultStreamPoolSize;
    }
    return static_cast<uint32_t>(parsed);
}

template <typename T>
T load_nccl_symbol(void *library, const char *name) {
    return reinterpret_cast<T>(dlsym(library, name));
}

const CudaNcclApi *load_nccl_api() {
    static CudaNcclApi api = []() {
        CudaNcclApi loaded;
        const char *override_path = std::getenv("PTO_CUDA_NCCL_LIBRARY");
        if (override_path != nullptr && override_path[0] != '\0') {
            loaded.library = dlopen(override_path, RTLD_NOW | RTLD_LOCAL);
            if (loaded.library == nullptr) {
                const char *error = dlerror();
                g_nccl_load_error =
                    std::string("PTO_CUDA_NCCL_LIBRARY=") + override_path +
                    (error != nullptr ? std::string(": ") + error : "");
            }
        }
        if (loaded.library == nullptr) {
            loaded.library = dlopen("libnccl.so.2", RTLD_NOW | RTLD_LOCAL);
        }
        if (loaded.library == nullptr) {
            loaded.library = dlopen("libnccl.so", RTLD_NOW | RTLD_LOCAL);
        }
        if (loaded.library == nullptr) {
            if (g_nccl_load_error.empty()) {
                const char *error = dlerror();
                g_nccl_load_error =
                    std::string("libnccl.so.2/libnccl.so") +
                    (error != nullptr ? std::string(": ") + error : " not found");
            }
            return loaded;
        }

        loaded.ncclGetUniqueId = load_nccl_symbol<CudaNcclApi::NcclGetUniqueIdFn>(loaded.library, "ncclGetUniqueId");
        loaded.ncclGetErrorString =
            load_nccl_symbol<CudaNcclApi::NcclGetErrorStringFn>(loaded.library, "ncclGetErrorString");
        loaded.ncclCommInitRank = load_nccl_symbol<CudaNcclApi::NcclCommInitRankFn>(loaded.library, "ncclCommInitRank");
        loaded.ncclAllReduce = load_nccl_symbol<CudaNcclApi::NcclAllReduceFn>(loaded.library, "ncclAllReduce");
        loaded.ncclReduceScatter =
            load_nccl_symbol<CudaNcclApi::NcclReduceScatterFn>(loaded.library, "ncclReduceScatter");
        loaded.ncclAllGather = load_nccl_symbol<CudaNcclApi::NcclAllGatherFn>(loaded.library, "ncclAllGather");
        loaded.ncclSend = load_nccl_symbol<CudaNcclApi::NcclSendFn>(loaded.library, "ncclSend");
        loaded.ncclRecv = load_nccl_symbol<CudaNcclApi::NcclRecvFn>(loaded.library, "ncclRecv");
        loaded.ncclGroupStart = load_nccl_symbol<CudaNcclApi::NcclGroupStartFn>(loaded.library, "ncclGroupStart");
        loaded.ncclGroupEnd = load_nccl_symbol<CudaNcclApi::NcclGroupEndFn>(loaded.library, "ncclGroupEnd");
        loaded.ncclCommDestroy = load_nccl_symbol<CudaNcclApi::NcclCommDestroyFn>(loaded.library, "ncclCommDestroy");
        if (loaded.ncclGetUniqueId == nullptr || loaded.ncclGetErrorString == nullptr ||
            loaded.ncclCommInitRank == nullptr ||
            loaded.ncclAllReduce == nullptr || loaded.ncclReduceScatter == nullptr || loaded.ncclAllGather == nullptr ||
            loaded.ncclSend == nullptr || loaded.ncclRecv == nullptr || loaded.ncclGroupStart == nullptr ||
            loaded.ncclGroupEnd == nullptr || loaded.ncclCommDestroy == nullptr) {
            g_nccl_load_error = "NCCL library is missing required symbols";
            dlclose(loaded.library);
            loaded = {};
        }
        return loaded;
    }();

    if (api.library == nullptr) {
        return nullptr;
    }
    return &api;
}

bool write_nccl_unique_id(const char *rootinfo_path, const CudaNcclUniqueId &unique_id) {
    if (rootinfo_path == nullptr || rootinfo_path[0] == '\0') {
        return false;
    }
    std::ofstream output(rootinfo_path, std::ios::binary | std::ios::trunc);
    if (!output) {
        return false;
    }
    output.write(unique_id.internal, sizeof(unique_id.internal));
    return output.good();
}

bool read_nccl_unique_id(const char *rootinfo_path, CudaNcclUniqueId *unique_id) {
    if (rootinfo_path == nullptr || rootinfo_path[0] == '\0' || unique_id == nullptr) {
        return false;
    }
    for (int waited_ms = 0; waited_ms <= kNcclRootinfoWaitMs; waited_ms += kNcclRootinfoPollMs) {
        std::ifstream input(rootinfo_path, std::ios::binary);
        if (input) {
            input.read(unique_id->internal, sizeof(unique_id->internal));
            if (input.gcount() == static_cast<std::streamsize>(sizeof(unique_id->internal))) {
                return true;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(kNcclRootinfoPollMs));
    }
    return false;
}

class CudaDeviceRunner {
public:
    ~CudaDeviceRunner() { finalize(); }

    int init(int device_id) {
        device_id_ = device_id;
        CUresult cu_rc = cuInit(0);
        if (cu_rc != CUDA_SUCCESS) {
            return -1;
        }
        cudaError_t rc = cudaSetDevice(device_id_);
        if (rc != cudaSuccess) {
            return -1;
        }
        streams_.resize(configured_stream_pool_size(), nullptr);
        for (auto &stream : streams_) {
            rc = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
            if (rc != cudaSuccess) {
                finalize();
                return -1;
            }
        }
        return 0;
    }

    int finalize() {
        for (auto &entry : prepared_) {
            if (entry.second.module != nullptr) {
                cuModuleUnload(entry.second.module);
                entry.second.module = nullptr;
            }
        }
        prepared_.clear();
        for (auto &stream : streams_) {
            if (stream != nullptr) {
                cudaStreamDestroy(stream);
                stream = nullptr;
            }
        }
        streams_.clear();
        return cudaDeviceSynchronize() == cudaSuccess ? 0 : -1;
    }

    void *malloc(size_t size) {
        if (size == 0) {
            return nullptr;
        }
        void *ptr = nullptr;
        if (cudaSetDevice(device_id_) != cudaSuccess) {
            return nullptr;
        }
        if (cudaMalloc(&ptr, size) != cudaSuccess) {
            return nullptr;
        }
        return ptr;
    }

    void free(void *ptr) {
        if (ptr == nullptr) {
            return;
        }
        cudaSetDevice(device_id_);
        cudaFree(ptr);
    }

    int copy_to_device(void *dev_ptr, const void *host_ptr, size_t size) {
        if (dev_ptr == nullptr || host_ptr == nullptr) {
            return -1;
        }
        if (cudaSetDevice(device_id_) != cudaSuccess) {
            return -1;
        }
        cudaStream_t stream = default_stream();
        if (stream == nullptr) {
            return -1;
        }
        cudaError_t rc = cudaMemcpyAsync(dev_ptr, host_ptr, size, cudaMemcpyHostToDevice, stream);
        if (rc != cudaSuccess) {
            return -1;
        }
        return cudaStreamSynchronize(stream) == cudaSuccess ? 0 : -1;
    }

    int copy_from_device(void *host_ptr, const void *dev_ptr, size_t size) {
        if (host_ptr == nullptr || dev_ptr == nullptr) {
            return -1;
        }
        if (cudaSetDevice(device_id_) != cudaSuccess) {
            return -1;
        }
        cudaStream_t stream = default_stream();
        if (stream == nullptr) {
            return -1;
        }
        cudaError_t rc = cudaMemcpyAsync(host_ptr, dev_ptr, size, cudaMemcpyDeviceToHost, stream);
        if (rc != cudaSuccess) {
            return -1;
        }
        return cudaStreamSynchronize(stream) == cudaSuccess ? 0 : -1;
    }

    int prepare(int32_t callable_id, const PtoCudaHostCallable *callable) {
        auto *header = static_cast<const PtoCudaCallableHeader *>(static_cast<const void *>(callable));
        if (header == nullptr || header->image == nullptr || header->image_size == 0 || header->entry_name == nullptr) {
            return -1;
        }
        if (header->version != 1 && header->version != 2) {
            return -1;
        }
        if (header->op != PTO_CUDA_HOST_OP_VECTOR_ADD_F32 && header->op != PTO_CUDA_HOST_OP_VECTOR_SCALE_F32 &&
            header->op != PTO_CUDA_HOST_OP_VECTOR_AXPY_F32 && header->op != PTO_CUDA_HOST_OP_VECTOR_UNARY_F32 &&
            header->op != PTO_CUDA_HOST_OP_VECTOR_AFFINE_F32 && header->op != PTO_CUDA_HOST_OP_VECTOR_TRIAD_F32 &&
            header->op != PTO_CUDA_HOST_OP_VECTOR_QUAD_F32 && header->op != PTO_CUDA_HOST_OP_VECTOR_GENERIC_ARGS_F32 &&
            header->op != PTO_CUDA_HOST_OP_VECTOR_GENERIC_ARGS4_F32 &&
            header->op != PTO_CUDA_PERSISTENT_OP_VECTOR_ADD_F32_TASKS &&
            header->op != PTO_CUDA_PERSISTENT_OP_VECTOR_ADD_F32_GRID &&
            header->op != PTO_CUDA_PERSISTENT_OP_VECTOR_ADD_F32_QUEUE &&
            header->op != PTO_CUDA_PERSISTENT_OP_DAG_F32_RING) {
            return -1;
        }
        uint32_t stream_id = 0;
        if (header->version >= 2) {
            stream_id = callable->stream_id;
        }
        if (stream_id >= streams_.size()) {
            return -1;
        }
        if (cudaSetDevice(device_id_) != cudaSuccess) {
            return -1;
        }

        unregister(callable_id);

        std::vector<char> image(
            static_cast<const char *>(header->image), static_cast<const char *>(header->image) + header->image_size
        );
        if (image.empty() || image.back() != '\0') {
            image.push_back('\0');
        }

        PreparedCallable prepared;
        CUresult cu_rc = cuModuleLoadData(&prepared.module, image.data());
        if (cu_rc != CUDA_SUCCESS) {
            return -1;
        }
        cu_rc = cuModuleGetFunction(&prepared.function, prepared.module, header->entry_name);
        if (cu_rc != CUDA_SUCCESS) {
            cuModuleUnload(prepared.module);
            return -1;
        }

        prepared.op = header->op;
        prepared.grid_dim = header->grid_dim;
        prepared.block_dim = header->block_dim;
        prepared.stream_id = stream_id;
        prepared.shared_mem_bytes = header->shared_mem_bytes;
        prepared_[callable_id] = prepared;
        return 0;
    }

    int unregister(int32_t callable_id) {
        auto it = prepared_.find(callable_id);
        if (it == prepared_.end()) {
            return 0;
        }
        if (it->second.module != nullptr) {
            cuModuleUnload(it->second.module);
        }
        prepared_.erase(it);
        return 0;
    }

    int set_comm_descriptor(const void *descriptor_bytes, size_t descriptor_size) {
        PtoCudaCommDeviceDescriptor parsed = {};
        int rc = pto_cuda_comm_descriptor_from_bytes(descriptor_bytes, descriptor_size, &parsed);
        if (rc != 0) {
            return rc;
        }
        if (parsed.device_id != static_cast<uint32_t>(device_id_)) {
            return -1;
        }
        comm_descriptor_ = parsed;
        has_comm_descriptor_ = true;
        return 0;
    }

    int run(int32_t callable_id, const void *args, PtoRunTiming *out_timing) {
        if (out_timing != nullptr) {
            std::memset(out_timing, 0, sizeof(*out_timing));
        }
        if (args == nullptr) {
            return -1;
        }
        auto it = prepared_.find(callable_id);
        if (it == prepared_.end()) {
            return -1;
        }
        PreparedCallable &prepared = it->second;
        if (prepared.op != PTO_CUDA_HOST_OP_VECTOR_ADD_F32 && prepared.op != PTO_CUDA_HOST_OP_VECTOR_SCALE_F32 &&
            prepared.op != PTO_CUDA_HOST_OP_VECTOR_AXPY_F32 && prepared.op != PTO_CUDA_HOST_OP_VECTOR_UNARY_F32 &&
            prepared.op != PTO_CUDA_HOST_OP_VECTOR_AFFINE_F32 && prepared.op != PTO_CUDA_HOST_OP_VECTOR_TRIAD_F32 &&
            prepared.op != PTO_CUDA_HOST_OP_VECTOR_QUAD_F32 &&
            prepared.op != PTO_CUDA_HOST_OP_VECTOR_GENERIC_ARGS_F32 &&
            prepared.op != PTO_CUDA_HOST_OP_VECTOR_GENERIC_ARGS4_F32 &&
            prepared.op != PTO_CUDA_PERSISTENT_OP_VECTOR_ADD_F32_TASKS &&
            prepared.op != PTO_CUDA_PERSISTENT_OP_VECTOR_ADD_F32_GRID &&
            prepared.op != PTO_CUDA_PERSISTENT_OP_VECTOR_ADD_F32_QUEUE &&
            prepared.op != PTO_CUDA_PERSISTENT_OP_DAG_F32_RING) {
            return -1;
        }
        if (cudaSetDevice(device_id_) != cudaSuccess) {
            return -1;
        }
        cudaStream_t stream = stream_for(prepared.stream_id);
        if (stream == nullptr) {
            return -1;
        }

        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        if (cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) != cudaSuccess) {
            if (start != nullptr) cudaEventDestroy(start);
            if (stop != nullptr) cudaEventDestroy(stop);
            return -1;
        }

        auto host_start = std::chrono::steady_clock::now();
        cudaEventRecord(start, stream);
        const float *a = nullptr;
        const float *b = nullptr;
        const float *c = nullptr;
        const float *d = nullptr;
        const float *e = nullptr;
        const float *f = nullptr;
        float *out = nullptr;
        float alpha = 0.0f;
        float beta = 0.0f;
        float gamma = 0.0f;
        float delta = 0.0f;
        uint64_t n = 0;
        const PtoCudaPersistentVectorAddTask *tasks = nullptr;
        uint64_t task_count = 0;
        uint32_t worker_blocks_per_task = 1;
        const PtoCudaPersistentVectorAddQueueState *queue_state = nullptr;
        const PtoCudaPersistentDagState *dag_state = nullptr;
        void *kernel_args[12] = {};
        if (prepared.op == PTO_CUDA_HOST_OP_VECTOR_ADD_F32) {
            auto *typed_args = static_cast<const PtoCudaVectorAddArgs *>(args);
            if (typed_args->a == nullptr || typed_args->b == nullptr || typed_args->out == nullptr ||
                typed_args->n == 0) {
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                return -1;
            }
            a = typed_args->a;
            b = typed_args->b;
            out = typed_args->out;
            n = typed_args->n;
            kernel_args[0] = &a;
            kernel_args[1] = &b;
            kernel_args[2] = &out;
            kernel_args[3] = &n;
        } else if (prepared.op == PTO_CUDA_HOST_OP_VECTOR_SCALE_F32) {
            auto *typed_args = static_cast<const PtoCudaVectorScaleArgs *>(args);
            if (typed_args->a == nullptr || typed_args->out == nullptr || typed_args->n == 0) {
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                return -1;
            }
            a = typed_args->a;
            out = typed_args->out;
            alpha = typed_args->alpha;
            n = typed_args->n;
            kernel_args[0] = &a;
            kernel_args[1] = &out;
            kernel_args[2] = &alpha;
            kernel_args[3] = &n;
        } else if (prepared.op == PTO_CUDA_HOST_OP_VECTOR_UNARY_F32) {
            auto *typed_args = static_cast<const PtoCudaVectorUnaryArgs *>(args);
            if (typed_args->a == nullptr || typed_args->out == nullptr || typed_args->n == 0) {
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                return -1;
            }
            a = typed_args->a;
            out = typed_args->out;
            n = typed_args->n;
            kernel_args[0] = &a;
            kernel_args[1] = &out;
            kernel_args[2] = &n;
        } else if (prepared.op == PTO_CUDA_HOST_OP_VECTOR_AXPY_F32) {
            auto *typed_args = static_cast<const PtoCudaVectorAxpyArgs *>(args);
            if (typed_args->a == nullptr || typed_args->b == nullptr || typed_args->out == nullptr ||
                typed_args->n == 0) {
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                return -1;
            }
            a = typed_args->a;
            b = typed_args->b;
            out = typed_args->out;
            alpha = typed_args->alpha;
            n = typed_args->n;
            kernel_args[0] = &a;
            kernel_args[1] = &b;
            kernel_args[2] = &out;
            kernel_args[3] = &alpha;
            kernel_args[4] = &n;
        } else if (prepared.op == PTO_CUDA_HOST_OP_VECTOR_AFFINE_F32) {
            auto *typed_args = static_cast<const PtoCudaVectorAffineArgs *>(args);
            if (typed_args->a == nullptr || typed_args->b == nullptr || typed_args->out == nullptr ||
                typed_args->n == 0) {
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                return -1;
            }
            a = typed_args->a;
            b = typed_args->b;
            out = typed_args->out;
            alpha = typed_args->alpha;
            beta = typed_args->beta;
            n = typed_args->n;
            kernel_args[0] = &a;
            kernel_args[1] = &b;
            kernel_args[2] = &out;
            kernel_args[3] = &alpha;
            kernel_args[4] = &beta;
            kernel_args[5] = &n;
        } else if (prepared.op == PTO_CUDA_HOST_OP_VECTOR_TRIAD_F32) {
            auto *typed_args = static_cast<const PtoCudaVectorTernaryArgs *>(args);
            if (typed_args->a == nullptr || typed_args->b == nullptr || typed_args->c == nullptr ||
                typed_args->out == nullptr || typed_args->n == 0) {
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                return -1;
            }
            a = typed_args->a;
            b = typed_args->b;
            c = typed_args->c;
            out = typed_args->out;
            n = typed_args->n;
            kernel_args[0] = &a;
            kernel_args[1] = &b;
            kernel_args[2] = &c;
            kernel_args[3] = &out;
            kernel_args[4] = &n;
        } else if (prepared.op == PTO_CUDA_HOST_OP_VECTOR_QUAD_F32) {
            auto *typed_args = static_cast<const PtoCudaVectorQuaternaryArgs *>(args);
            if (typed_args->a == nullptr || typed_args->b == nullptr || typed_args->c == nullptr ||
                typed_args->d == nullptr || typed_args->out == nullptr || typed_args->n == 0) {
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                return -1;
            }
            a = typed_args->a;
            b = typed_args->b;
            c = typed_args->c;
            d = typed_args->d;
            out = typed_args->out;
            n = typed_args->n;
            kernel_args[0] = &a;
            kernel_args[1] = &b;
            kernel_args[2] = &c;
            kernel_args[3] = &d;
            kernel_args[4] = &out;
            kernel_args[5] = &n;
        } else if (prepared.op == PTO_CUDA_HOST_OP_VECTOR_GENERIC_ARGS_F32) {
            auto *typed_args = static_cast<const PtoCudaVectorGenericArgs *>(args);
            if (typed_args->a == nullptr || typed_args->b == nullptr || typed_args->out == nullptr ||
                typed_args->tensor_arg_count < 2 || typed_args->scalar_arg_count < 2 ||
                typed_args->tensor_args[0] == nullptr || typed_args->tensor_args[1] == nullptr || typed_args->n == 0) {
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                return -1;
            }
            a = typed_args->a;
            b = typed_args->b;
            out = typed_args->out;
            c = typed_args->tensor_args[0];
            d = typed_args->tensor_args[1];
            alpha = typed_args->scalar_args[0];
            beta = typed_args->scalar_args[1];
            n = typed_args->n;
            kernel_args[0] = &a;
            kernel_args[1] = &b;
            kernel_args[2] = &out;
            kernel_args[3] = &c;
            kernel_args[4] = &d;
            kernel_args[5] = &alpha;
            kernel_args[6] = &beta;
            kernel_args[7] = &n;
        } else if (prepared.op == PTO_CUDA_HOST_OP_VECTOR_GENERIC_ARGS4_F32) {
            auto *typed_args = static_cast<const PtoCudaVectorGenericArgs *>(args);
            if (typed_args->a == nullptr || typed_args->b == nullptr || typed_args->out == nullptr ||
                typed_args->tensor_arg_count != 4 || typed_args->scalar_arg_count != 4 || typed_args->n == 0) {
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                return -1;
            }
            for (uint32_t idx = 0; idx < 4; ++idx) {
                if (typed_args->tensor_args[idx] == nullptr) {
                    cudaEventDestroy(start);
                    cudaEventDestroy(stop);
                    return -1;
                }
            }
            a = typed_args->a;
            b = typed_args->b;
            out = typed_args->out;
            c = typed_args->tensor_args[0];
            d = typed_args->tensor_args[1];
            e = typed_args->tensor_args[2];
            f = typed_args->tensor_args[3];
            alpha = typed_args->scalar_args[0];
            beta = typed_args->scalar_args[1];
            gamma = typed_args->scalar_args[2];
            delta = typed_args->scalar_args[3];
            n = typed_args->n;
            kernel_args[0] = &a;
            kernel_args[1] = &b;
            kernel_args[2] = &out;
            kernel_args[3] = &c;
            kernel_args[4] = &d;
            kernel_args[5] = &e;
            kernel_args[6] = &f;
            kernel_args[7] = &alpha;
            kernel_args[8] = &beta;
            kernel_args[9] = &gamma;
            kernel_args[10] = &delta;
            kernel_args[11] = &n;
        } else if (prepared.op == PTO_CUDA_PERSISTENT_OP_VECTOR_ADD_F32_TASKS) {
            auto *typed_args = static_cast<const PtoCudaPersistentVectorAddArgs *>(args);
            if (typed_args->tasks == nullptr || typed_args->task_count == 0) {
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                return -1;
            }
            tasks = typed_args->tasks;
            task_count = typed_args->task_count;
            kernel_args[0] = &tasks;
            kernel_args[1] = &task_count;
        } else if (prepared.op == PTO_CUDA_PERSISTENT_OP_VECTOR_ADD_F32_GRID) {
            auto *typed_args = static_cast<const PtoCudaPersistentVectorAddGridArgs *>(args);
            if (typed_args->tasks == nullptr || typed_args->task_count == 0 ||
                typed_args->worker_blocks_per_task == 0) {
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                return -1;
            }
            tasks = typed_args->tasks;
            task_count = typed_args->task_count;
            worker_blocks_per_task = typed_args->worker_blocks_per_task;
            kernel_args[0] = &tasks;
            kernel_args[1] = &task_count;
            kernel_args[2] = &worker_blocks_per_task;
        } else if (prepared.op == PTO_CUDA_PERSISTENT_OP_VECTOR_ADD_F32_QUEUE) {
            auto *typed_args = static_cast<const PtoCudaPersistentVectorAddQueueArgs *>(args);
            if (typed_args->state == nullptr) {
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                return -1;
            }
            queue_state = typed_args->state;
            kernel_args[0] = &queue_state;
        } else if (prepared.op == PTO_CUDA_PERSISTENT_OP_DAG_F32_RING) {
            auto *typed_args = static_cast<const PtoCudaPersistentDagArgs *>(args);
            if (typed_args->state == nullptr) {
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                return -1;
            }
            dag_state = typed_args->state;
            kernel_args[0] = &dag_state;
        }
        CUresult cu_rc = cuLaunchKernel(
            prepared.function, prepared.grid_dim, 1, 1, prepared.block_dim, 1, 1, prepared.shared_mem_bytes,
            reinterpret_cast<CUstream>(stream), kernel_args, nullptr
        );
        cudaEventRecord(stop, stream);
        cudaError_t sync_rc = cudaStreamSynchronize(stream);
        auto host_stop = std::chrono::steady_clock::now();

        if (out_timing != nullptr) {
            out_timing->host_wall_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(host_stop - host_start).count()
            );
            float elapsed_ms = 0.0F;
            if (cudaEventElapsedTime(&elapsed_ms, start, stop) == cudaSuccess) {
                out_timing->device_wall_ns = static_cast<uint64_t>(elapsed_ms * 1000000.0F);
            }
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        if (cu_rc != CUDA_SUCCESS || sync_rc != cudaSuccess) {
            return -1;
        }
        return 0;
    }

    const PtoCudaCommDeviceDescriptor *comm_descriptor_or_null() const {
        return has_comm_descriptor_ ? &comm_descriptor_ : nullptr;
    }

    CudaCommStream *create_comm_stream() {
        const PtoCudaCommDeviceDescriptor *descriptor = comm_descriptor_or_null();
        if (descriptor == nullptr) {
            return nullptr;
        }
        if (cudaSetDevice(device_id_) != cudaSuccess) {
            return nullptr;
        }

        auto *comm_stream = new CudaCommStream();
        comm_stream->descriptor = *descriptor;
        comm_stream->has_descriptor = true;
        cudaError_t rc = cudaStreamCreateWithFlags(&comm_stream->stream, cudaStreamNonBlocking);
        if (rc != cudaSuccess) {
            delete comm_stream;
            return nullptr;
        }
        return comm_stream;
    }

    int destroy_comm_stream(CudaCommStream *comm_stream) {
        if (comm_stream == nullptr) {
            return 0;
        }
        int rc = 0;
        if (comm_stream->stream != nullptr) {
            rc = cudaStreamDestroy(comm_stream->stream) == cudaSuccess ? 0 : -1;
        }
        delete comm_stream;
        return rc;
    }

private:
    cudaStream_t stream_for(uint32_t stream_id) {
        if (stream_id >= streams_.size()) {
            return nullptr;
        }
        return streams_[stream_id];
    }

    cudaStream_t default_stream() { return stream_for(0); }

    int device_id_ = 0;
    std::vector<cudaStream_t> streams_;
    std::unordered_map<int32_t, PreparedCallable> prepared_;
    PtoCudaCommDeviceDescriptor comm_descriptor_ = {};
    bool has_comm_descriptor_ = false;
};

CudaDeviceRunner *runner(DeviceContextHandle ctx) { return static_cast<CudaDeviceRunner *>(ctx); }

CommHandle init_nccl_comm_from_descriptor(
    const PtoCudaCommDeviceDescriptor &descriptor, cudaStream_t stream, const char *rootinfo_path
);

CommHandle init_comm_from_descriptor(int rank, int nranks, CudaCommStream *comm_stream, const char *rootinfo_path) {
    if (comm_stream == nullptr || !comm_stream->has_descriptor || comm_stream->stream == nullptr) {
        set_comm_error("comm stream has no CUDA communication descriptor");
        return nullptr;
    }
    const PtoCudaCommDeviceDescriptor &descriptor = comm_stream->descriptor;
    if (rank < 0 || nranks <= 0 || descriptor.rank != static_cast<uint32_t>(rank) ||
        descriptor.world_size != static_cast<uint32_t>(nranks)) {
        set_comm_error(
            "descriptor rank/world_size mismatch: descriptor rank=" + std::to_string(descriptor.rank) +
            " world_size=" + std::to_string(descriptor.world_size) + " request rank=" + std::to_string(rank) +
            " nranks=" + std::to_string(nranks)
        );
        return nullptr;
    }
    if (descriptor.backend_code == PTO_CUDA_COMM_BACKEND_NCCL) {
        return init_nccl_comm_from_descriptor(descriptor, comm_stream->stream, rootinfo_path);
    }
    if (descriptor.backend_code != PTO_CUDA_COMM_BACKEND_MOCK) {
        set_comm_error("unsupported CUDA communication backend code " + std::to_string(descriptor.backend_code));
        return nullptr;
    }

    auto *handle = new CudaCommHandle();
    handle->descriptor = descriptor;
    handle->stream = comm_stream->stream;
    if (rootinfo_path != nullptr) {
        handle->rootinfo_path = rootinfo_path;
    }
    return static_cast<CommHandle>(handle);
}

CommHandle init_nccl_comm_from_descriptor(
    const PtoCudaCommDeviceDescriptor &descriptor, cudaStream_t stream, const char *rootinfo_path
) {
    const CudaNcclApi *api = load_nccl_api();
    if (api == nullptr) {
        set_comm_error(
            "NCCL API unavailable: could not load libnccl with required symbols" +
            (g_nccl_load_error.empty() ? std::string() : std::string(" (") + g_nccl_load_error + ")")
        );
        return nullptr;
    }
    if (rootinfo_path == nullptr || rootinfo_path[0] == '\0') {
        set_comm_error("NCCL rootinfo path is empty");
        return nullptr;
    }
    cudaError_t set_device_rc = cudaSetDevice(static_cast<int>(descriptor.device_id));
    if (set_device_rc != cudaSuccess) {
        set_comm_error(cuda_error_message("cudaSetDevice", set_device_rc));
        return nullptr;
    }

    CudaNcclUniqueId unique_id = {};
    if (descriptor.rank == 0) {
        ncclResult_t unique_id_rc = api->ncclGetUniqueId(&unique_id);
        if (unique_id_rc != 0) {
            set_comm_error(nccl_error_message(api, "ncclGetUniqueId", unique_id_rc));
            return nullptr;
        }
        if (!write_nccl_unique_id(rootinfo_path, unique_id)) {
            set_comm_error(std::string("failed to write NCCL unique id to ") + rootinfo_path);
            return nullptr;
        }
    } else if (!read_nccl_unique_id(rootinfo_path, &unique_id)) {
        set_comm_error(std::string("failed to read NCCL unique id from ") + rootinfo_path);
        return nullptr;
    }

    ncclComm_t nccl_comm = nullptr;
    ncclResult_t init_rc = api->ncclCommInitRank(
        &nccl_comm, static_cast<int>(descriptor.world_size), unique_id, static_cast<int>(descriptor.rank)
    );
    if (init_rc != 0 || nccl_comm == nullptr) {
        set_comm_error(
            init_rc != 0 ? nccl_error_message(api, "ncclCommInitRank", init_rc)
                         : "ncclCommInitRank returned a null communicator"
        );
        return nullptr;
    }

    auto *handle = new CudaCommHandle();
    handle->descriptor = descriptor;
    handle->stream = stream;
    handle->rootinfo_path = rootinfo_path;
    handle->nccl_api = api;
    handle->nccl_comm = nccl_comm;
    return static_cast<CommHandle>(handle);
}

}  // namespace

extern "C" {

DeviceContextHandle create_device_context(void) {
    try {
        return static_cast<DeviceContextHandle>(new CudaDeviceRunner());
    } catch (...) {
        return nullptr;
    }
}

void destroy_device_context(DeviceContextHandle ctx) { delete runner(ctx); }

size_t get_runtime_size(void) { return sizeof(PtoCudaRuntime); }

void *device_malloc_ctx(DeviceContextHandle ctx, size_t size) {
    if (ctx == nullptr) return nullptr;
    try {
        return runner(ctx)->malloc(size);
    } catch (...) {
        return nullptr;
    }
}

void device_free_ctx(DeviceContextHandle ctx, void *dev_ptr) {
    if (ctx == nullptr) return;
    try {
        runner(ctx)->free(dev_ptr);
    } catch (...) {}
}

int copy_to_device_ctx(DeviceContextHandle ctx, void *dev_ptr, const void *host_ptr, size_t size) {
    if (ctx == nullptr) return -1;
    try {
        return runner(ctx)->copy_to_device(dev_ptr, host_ptr, size);
    } catch (...) {
        return -1;
    }
}

int copy_from_device_ctx(DeviceContextHandle ctx, void *host_ptr, const void *dev_ptr, size_t size) {
    if (ctx == nullptr) return -1;
    try {
        return runner(ctx)->copy_from_device(host_ptr, dev_ptr, size);
    } catch (...) {
        return -1;
    }
}

int simpler_init(
    DeviceContextHandle ctx, int device_id, const uint8_t *aicpu_binary, size_t aicpu_size,
    const uint8_t *aicore_binary, size_t aicore_size
) {
    (void)aicpu_binary;
    (void)aicpu_size;
    (void)aicore_binary;
    (void)aicore_size;
    if (ctx == nullptr) return -1;
    try {
        return runner(ctx)->init(device_id);
    } catch (...) {
        return -1;
    }
}

int simpler_init_roles(DeviceContextHandle ctx, int device_id, const PtoRuntimeBinaryMap *binaries) {
    if (ctx == nullptr || binaries == nullptr) return -1;

    bool has_device = false;
    for (size_t i = 0; i < binaries->count; ++i) {
        const PtoRuntimeBinaryRole &entry = binaries->entries[i];
        if (entry.role == nullptr) return -1;
        if (std::string(entry.role) == "host") return -1;
        if (std::string(entry.role) == "device") {
            has_device = entry.binary != nullptr;
        }
    }
    if (!has_device) return -1;

    try {
        return runner(ctx)->init(device_id);
    } catch (...) {
        return -1;
    }
}

int finalize_device(DeviceContextHandle ctx) {
    if (ctx == nullptr) return -1;
    try {
        return runner(ctx)->finalize();
    } catch (...) {
        return -1;
    }
}

int prepare_callable(DeviceContextHandle ctx, int32_t callable_id, const void *callable) {
    if (ctx == nullptr || callable == nullptr) return -1;
    try {
        return runner(ctx)->prepare(callable_id, static_cast<const PtoCudaHostCallable *>(callable));
    } catch (...) {
        return -1;
    }
}

int run_prepared(
    DeviceContextHandle ctx, RuntimeHandle runtime, int32_t callable_id, const void *args, int block_dim,
    int aicpu_thread_num, int enable_l2_swimlane, int enable_dump_tensor, int enable_pmu, int enable_dep_gen,
    const char *output_prefix, PtoRunTiming *out_timing
) {
    (void)runtime;
    (void)block_dim;
    (void)aicpu_thread_num;
    (void)enable_l2_swimlane;
    (void)enable_dump_tensor;
    (void)enable_pmu;
    (void)enable_dep_gen;
    (void)output_prefix;
    if (ctx == nullptr) return -1;
    try {
        return runner(ctx)->run(callable_id, args, out_timing);
    } catch (...) {
        return -1;
    }
}

int unregister_callable(DeviceContextHandle ctx, int32_t callable_id) {
    if (ctx == nullptr) return -1;
    try {
        return runner(ctx)->unregister(callable_id);
    } catch (...) {
        return -1;
    }
}

int configure_cuda_comm_descriptor(DeviceContextHandle ctx, const void *descriptor_bytes, size_t descriptor_size) {
    if (ctx == nullptr) return -1;
    try {
        return runner(ctx)->set_comm_descriptor(descriptor_bytes, descriptor_size);
    } catch (...) {
        return -1;
    }
}

size_t get_aicpu_dlopen_count(DeviceContextHandle ctx) {
    (void)ctx;
    return 0;
}

size_t get_host_dlopen_count(DeviceContextHandle ctx) {
    (void)ctx;
    return 0;
}

int ensure_acl_ready_ctx(DeviceContextHandle ctx, int device_id) {
    (void)ctx;
    (void)device_id;
    return 0;
}

void *create_comm_stream_ctx(DeviceContextHandle ctx) {
    if (ctx == nullptr) return nullptr;
    try {
        return runner(ctx)->create_comm_stream();
    } catch (...) {
        return nullptr;
    }
}

int destroy_comm_stream_ctx(DeviceContextHandle ctx, void *stream) {
    if (ctx == nullptr) return -1;
    try {
        return runner(ctx)->destroy_comm_stream(static_cast<CudaCommStream *>(stream));
    } catch (...) {
        return -1;
    }
}

CommHandle comm_init(int rank, int nranks, void *stream, const char *rootinfo_path) {
    try {
        clear_comm_error();
        return init_comm_from_descriptor(rank, nranks, static_cast<CudaCommStream *>(stream), rootinfo_path);
    } catch (const std::exception &e) {
        set_comm_error(std::string("comm_init exception: ") + e.what());
        return nullptr;
    } catch (...) {
        set_comm_error("comm_init exception: unknown");
        return nullptr;
    }
}

const char *comm_last_error(void) { return g_comm_last_error.c_str(); }

int comm_alloc_windows(CommHandle h, size_t win_size, uint64_t *device_ctx_out) {
    (void)h;
    (void)win_size;
    (void)device_ctx_out;
    return -1;
}

int comm_get_local_window_base(CommHandle h, uint64_t *base_out) {
    (void)h;
    (void)base_out;
    return -1;
}

int comm_get_window_size(CommHandle h, size_t *size_out) {
    (void)h;
    (void)size_out;
    return -1;
}

int comm_derive_context(
    CommHandle h, const uint32_t *rank_ids, size_t rank_count, uint32_t domain_rank, size_t window_offset,
    size_t window_size, uint64_t *device_ctx_out
) {
    (void)h;
    (void)rank_ids;
    (void)rank_count;
    (void)domain_rank;
    (void)window_offset;
    (void)window_size;
    (void)device_ctx_out;
    return -1;
}

int comm_alloc_domain_windows(
    CommHandle h, uint64_t allocation_id, const uint32_t *rank_ids, size_t rank_count, uint32_t domain_rank,
    size_t window_size, uint64_t *device_ctx_out, uint64_t *local_window_base_out
) {
    (void)h;
    (void)allocation_id;
    (void)rank_ids;
    (void)rank_count;
    (void)domain_rank;
    (void)window_size;
    (void)device_ctx_out;
    (void)local_window_base_out;
    return -1;
}

int comm_release_domain_windows(CommHandle h, uint64_t allocation_id, size_t rank_count, uint32_t domain_rank) {
    (void)h;
    (void)allocation_id;
    (void)rank_count;
    (void)domain_rank;
    return -1;
}

int comm_barrier(CommHandle h) {
    if (h == nullptr) return -1;
    auto *handle = static_cast<CudaCommHandle *>(h);
    return handle->descriptor.backend_code == PTO_CUDA_COMM_BACKEND_MOCK ? 0 : -1;
}

static CudaCommHandle *cuda_nccl_handle_or_null(CommHandle h, const float *send, float *recv, size_t count) {
    if (h == nullptr || send == nullptr || recv == nullptr || count == 0) return nullptr;
    auto *handle = static_cast<CudaCommHandle *>(h);
    if (handle->descriptor.backend_code != PTO_CUDA_COMM_BACKEND_NCCL || handle->nccl_api == nullptr ||
        handle->nccl_comm == nullptr || handle->stream == nullptr) {
        return nullptr;
    }
    if (cudaSetDevice(static_cast<int>(handle->descriptor.device_id)) != cudaSuccess) {
        return nullptr;
    }
    return handle;
}

static int finish_nccl_stream(cudaStream_t stream, ncclResult_t rc) {
    if (rc != 0) {
        return -1;
    }
    return cudaStreamSynchronize(stream) == cudaSuccess ? 0 : -1;
}

int comm_all_reduce_f32(CommHandle h, const float *send, float *recv, size_t count) {
    CudaCommHandle *handle = cuda_nccl_handle_or_null(h, send, recv, count);
    if (handle == nullptr) return -1;
    return finish_nccl_stream(
        handle->stream,
        handle->nccl_api->ncclAllReduce(
            send, recv, count, PTO_CUDA_NCCL_FLOAT32, PTO_CUDA_NCCL_SUM, handle->nccl_comm, handle->stream
        )
    );
}

int comm_reduce_scatter_f32(CommHandle h, const float *send, float *recv, size_t recv_count) {
    CudaCommHandle *handle = cuda_nccl_handle_or_null(h, send, recv, recv_count);
    if (handle == nullptr) return -1;
    return finish_nccl_stream(
        handle->stream,
        handle->nccl_api->ncclReduceScatter(
            send, recv, recv_count, PTO_CUDA_NCCL_FLOAT32, PTO_CUDA_NCCL_SUM, handle->nccl_comm, handle->stream
        )
    );
}

int comm_all_gather_f32(CommHandle h, const float *send, float *recv, size_t send_count) {
    CudaCommHandle *handle = cuda_nccl_handle_or_null(h, send, recv, send_count);
    if (handle == nullptr) return -1;
    return finish_nccl_stream(
        handle->stream, handle->nccl_api->ncclAllGather(
                            send, recv, send_count, PTO_CUDA_NCCL_FLOAT32, handle->nccl_comm, handle->stream
                        )
    );
}

int comm_send_recv_f32(CommHandle h, const float *send, float *recv, size_t count, int dst_rank, int src_rank) {
    CudaCommHandle *handle = cuda_nccl_handle_or_null(h, send, recv, count);
    if (handle == nullptr) return -1;
    int world_size = static_cast<int>(handle->descriptor.world_size);
    if (dst_rank < 0 || src_rank < 0 || dst_rank >= world_size || src_rank >= world_size) return -1;

    ncclResult_t rc = handle->nccl_api->ncclGroupStart();
    if (rc == 0) {
        rc =
            handle->nccl_api->ncclSend(send, count, PTO_CUDA_NCCL_FLOAT32, dst_rank, handle->nccl_comm, handle->stream);
    }
    ncclResult_t recv_rc = rc == 0 ? handle->nccl_api->ncclRecv(
                                         recv, count, PTO_CUDA_NCCL_FLOAT32, src_rank, handle->nccl_comm, handle->stream
                                     ) :
                                     rc;
    ncclResult_t end_rc = handle->nccl_api->ncclGroupEnd();
    if (rc != 0 || recv_rc != 0 || end_rc != 0) {
        return -1;
    }
    return cudaStreamSynchronize(handle->stream) == cudaSuccess ? 0 : -1;
}

int comm_destroy(CommHandle h) {
    auto *handle = static_cast<CudaCommHandle *>(h);
    int rc = 0;
    if (handle != nullptr && handle->nccl_comm != nullptr && handle->nccl_api != nullptr) {
        rc = handle->nccl_api->ncclCommDestroy(handle->nccl_comm) == 0 ? 0 : -1;
        handle->nccl_comm = nullptr;
    }
    delete handle;
    return rc;
}

}  // extern "C"
