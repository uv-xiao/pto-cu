"""Persistent DAG task ABI reflection for materialized weight descriptors."""

from __future__ import annotations

import ctypes
from typing import Any

from .common import ROOT


def dag_task_abi() -> dict[str, Any]:
    import sys

    sys.path.insert(0, str(ROOT))
    from simpler_setup.cuda_callable_compiler import CudaPersistentDagTask

    fields = {
        "func_id": CudaPersistentDagTask.func_id.offset,
        "tensor_args": CudaPersistentDagTask.tensor_args.offset,
        "tensor_arg_dtypes": CudaPersistentDagTask.tensor_arg_dtypes.offset,
        "scalar_args": CudaPersistentDagTask.scalar_args.offset,
        "tensor_arg_count": CudaPersistentDagTask.tensor_arg_count.offset,
        "scalar_arg_count": CudaPersistentDagTask.scalar_arg_count.offset,
    }
    return {
        "task_struct": "CudaPersistentDagTask",
        "c_header_struct": "PtoCudaPersistentDagTask",
        "python_source": "simpler_setup/cuda_callable_compiler.py",
        "c_header_source": (
            "src/cuda/platform/include/host/"
            "pto_cuda_persistent_device_abi.h"
        ),
        "sizeof_bytes": ctypes.sizeof(CudaPersistentDagTask),
        "field_offsets": fields,
        "tensor_arg_capacity": 4,
    }
