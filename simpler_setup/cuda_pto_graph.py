# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CUDA persistent-device lowering for normal PTO submit graphs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

from .cuda_normal_graph import (
    CudaNormalGraphNode,
    LoweredCudaNormalGraph,
    TaskFactory,
    lower_normal_graph_from_dependents,
)


_TENSOR_ARG_ROLE_BY_NAME = {
    "INPUT": "input",
    "OUTPUT": "output",
    "INOUT": "inout",
    "OUTPUT_EXISTING": "output_existing",
    "NO_DEP": "no_dep",
}


@dataclass(frozen=True)
class CudaPtoTaskArg:
    """Tensor argument metadata used for PTO-style dependency inference."""

    name: str
    role: str = "input"
    ptr: int = 0
    child_memory: bool = False


@dataclass(frozen=True)
class CudaPtoTaskSubmit:
    """One PTO-style callable submission before CUDA scheduler materialization."""

    key: str
    callable_id: int
    args: tuple[CudaPtoTaskArg, ...]
    worker_id: int = -1
    attrs: Optional[Mapping[str, Any]] = None


class CudaPtoGraphRecorder:
    """Recording orchestrator for CUDA persistent-device graph construction."""

    def __init__(self, tensor_names: Sequence[Sequence[str] | None] | None = None) -> None:
        self._tensor_names = tuple(tensor_names or ())
        self.submits: list[CudaPtoTaskSubmit] = []

    def submit_next_level(self, callable_id: int, args: Any, config: Any = None, *, worker: int = -1):
        submit_id = len(self.submits)
        self.submits.append(
            cuda_pto_submit_from_task_args(
                f"task{submit_id}",
                int(callable_id),
                args,
                worker_id=int(worker),
                tensor_names=self._tensor_names_for(submit_id),
                attrs={"config": config},
            )
        )
        return submit_id

    def submit_next_level_group(
        self,
        callable_id: int,
        args_list: Sequence[Any],
        config: Any = None,
        *,
        workers: Sequence[int] | None = None,
    ):
        results = []
        for index, args in enumerate(args_list):
            worker = -1 if not workers else int(workers[index])
            results.append(self.submit_next_level(callable_id, args, config, worker=worker))
        return results

    def submit_sub(self, callable_id: int, args: Any = None):
        raise NotImplementedError("CUDA persistent-device graph recording only supports next-level submits")

    def submit_sub_group(self, callable_id: int, args_list: Sequence[Any]):
        raise NotImplementedError("CUDA persistent-device graph recording only supports next-level submits")

    def _tensor_names_for(self, submit_id: int) -> Sequence[str] | None:
        if submit_id >= len(self._tensor_names):
            return None
        return self._tensor_names[submit_id]


def record_cuda_pto_submits(
    orch_fn: Any,
    args: Any = None,
    config: Any = None,
    *,
    tensor_names: Sequence[Sequence[str] | None] | None = None,
) -> tuple[CudaPtoTaskSubmit, ...]:
    """Run a Python orchestration function and record CUDA PTO submits."""

    recorder = CudaPtoGraphRecorder(tensor_names=tensor_names)
    orch_fn(recorder, args, config)
    return tuple(recorder.submits)


def cuda_pto_submit_from_task_args(
    key: str,
    callable_id: int,
    task_args: Any,
    *,
    worker_id: int = -1,
    tensor_names: Sequence[str] | None = None,
    attrs: Optional[Mapping[str, Any]] = None,
) -> CudaPtoTaskSubmit:
    """Convert a PTO TaskArgs object into a CUDA persistent submit record."""

    tensor_count = int(task_args.tensor_count())
    if tensor_names is not None and len(tensor_names) != tensor_count:
        raise ValueError("CUDA PTO tensor_names length must match TaskArgs tensor_count")

    args = []
    for index in range(tensor_count):
        tensor = task_args.tensor(index)
        tag = task_args.tag(index)
        name = tensor_names[index] if tensor_names is not None else _default_tensor_name(tensor, index)
        args.append(
            CudaPtoTaskArg(
                str(name),
                role=_role_from_tensor_arg_type(tag),
                ptr=int(getattr(tensor, "data", 0)),
                child_memory=bool(getattr(tensor, "child_memory", False)),
            )
        )
    return CudaPtoTaskSubmit(
        key=str(key),
        callable_id=int(callable_id),
        args=tuple(args),
        worker_id=int(worker_id),
        attrs=attrs,
    )


def lower_cuda_pto_task_graph(
    submits: Sequence[CudaPtoTaskSubmit],
    make_task: TaskFactory,
) -> LoweredCudaNormalGraph:
    """Lower PTO-style submits into CUDA persistent DAG scheduler arrays."""

    nodes, dependents = cuda_normal_graph_from_pto_submits(submits)
    return lower_normal_graph_from_dependents(nodes, dependents, make_task)


def cuda_normal_graph_from_pto_submits(
    submits: Sequence[CudaPtoTaskSubmit],
) -> tuple[list[CudaNormalGraphNode], list[list[int]]]:
    """Build normal graph nodes and dependents from tagged PTO submissions."""

    _validate_submit_keys(submits)
    dependents = _dependents_from_tagged_submits(submits)
    nodes = [
        CudaNormalGraphNode(
            key=submit.key,
            func_id=int(submit.callable_id),
            attrs={"submit": submit, "task_args": submit.args},
        )
        for submit in submits
    ]
    return nodes, dependents


def _dependents_from_tagged_submits(
    submits: Sequence[CudaPtoTaskSubmit],
) -> list[list[int]]:
    producers: dict[tuple[Any, int], int] = {}
    dependents: list[list[int]] = [[] for _ in submits]

    for task_id, submit in enumerate(submits):
        for arg in submit.args:
            role = _normalize_role(arg.role)
            key = _tensor_key(arg, submit)
            if key is None or role in {"no_dep"}:
                continue
            if role in {"input", "inout"}:
                producer = producers.get(key)
                if producer is not None and producer != task_id:
                    _append_unique(dependents[producer], task_id)
            if role in {"output", "output_existing", "inout"}:
                producers[key] = task_id
    return dependents


def _tensor_key(arg: CudaPtoTaskArg, submit: CudaPtoTaskSubmit) -> tuple[Any, int] | None:
    if arg.ptr == 0 and not arg.name:
        return None
    key = int(arg.ptr) if arg.ptr else str(arg.name)
    worker_id = int(submit.worker_id) if arg.child_memory else -1
    return key, worker_id


def _default_tensor_name(tensor: Any, index: int) -> str:
    if int(getattr(tensor, "data", 0)) == 0:
        return ""
    return f"tensor_{index}"


def _role_from_tensor_arg_type(tag: Any) -> str:
    name = getattr(tag, "name", str(tag).split(".")[-1])
    try:
        return _TENSOR_ARG_ROLE_BY_NAME[str(name)]
    except KeyError as exc:
        raise ValueError(f"unsupported PTO TensorArgType for CUDA graph lowering: {tag}") from exc


def _normalize_role(role: str) -> str:
    normalized = str(role).lower()
    aliases = {
        "in": "input",
        "out": "output",
        "none": "no_dep",
        "ignore": "no_dep",
    }
    normalized = aliases.get(normalized, normalized)
    allowed = {"input", "output", "output_existing", "inout", "no_dep"}
    if normalized not in allowed:
        raise ValueError(f"unsupported CUDA PTO task argument role: {role}")
    return normalized


def _append_unique(values: list[int], value: int) -> None:
    if value not in values:
        values.append(value)


def _validate_submit_keys(submits: Sequence[CudaPtoTaskSubmit]) -> None:
    seen: set[str] = set()
    for submit in submits:
        if not submit.key:
            raise ValueError("CUDA PTO task submit key must be non-empty")
        if submit.key in seen:
            raise ValueError(f"duplicate CUDA PTO task submit key: {submit.key}")
        seen.add(submit.key)
