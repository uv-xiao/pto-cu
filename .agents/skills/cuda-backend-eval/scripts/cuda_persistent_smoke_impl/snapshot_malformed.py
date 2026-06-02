"""Malformed live C++ snapshot cases for CUDA PTO normal graph lowering."""

from __future__ import annotations

from copy import copy
from typing import Any

from simpler.task_interface import (
    CallConfig,
    ContinuousTensor,
    DataType,
    TaskArgs,
    TensorArgType,
    _Worker,
)
from simpler_setup.cuda_pto_graph import (
    cuda_pto_submit_from_orchestrator_snapshot,
    lower_cuda_pto_task_graph,
)


def run_cpp_snapshot_malformed_cases(*, n: int = 128) -> dict[str, Any]:
    snapshot = _make_live_snapshot(n=n)
    cases = [
        _expect_error(
            "multi_task_args_per_submit",
            "CUDA PTO snapshot conversion supports one TaskArgs per submit",
            lambda: cuda_pto_submit_from_orchestrator_snapshot(_with_extra_task_args(snapshot)),
        ),
        _expect_error(
            "tensor_name_arity_mismatch",
            "CUDA PTO tensor_names length must match TaskArgs tensor_count",
            lambda: cuda_pto_submit_from_orchestrator_snapshot(snapshot, tensor_names=(("a",),)),
        ),
        _expect_error(
            "duplicate_snapshot_slot_key",
            "duplicate CUDA PTO task submit key: slot0",
            lambda: lower_cuda_pto_task_graph(
                cuda_pto_submit_from_orchestrator_snapshot(_with_duplicate_slot(snapshot)),
                lambda node, begin, count, fanin: (node.key, begin, count, fanin),
            ),
        ),
    ]
    return {
        "status": "pass" if all(case["status"] == "pass" for case in cases) else "fail",
        "graph_source": "cpp_orchestrator_snapshot",
        "case_count": len(cases),
        "cases": cases,
    }


def _make_live_snapshot(*, n: int) -> tuple[Any, ...]:
    def tensor(ptr: int):
        return ContinuousTensor.make(ptr, (n,), DataType.FLOAT32, False)

    task0 = TaskArgs()
    task0.add_tensor(tensor(10), TensorArgType.INPUT)
    task0.add_tensor(tensor(20), TensorArgType.OUTPUT)
    task1 = TaskArgs()
    task1.add_tensor(tensor(20), TensorArgType.INPUT)
    task1.add_tensor(tensor(30), TensorArgType.OUTPUT)

    worker = _Worker(3, 0)
    worker.init()
    orch = worker.get_orchestrator()
    try:
        orch._scope_begin()
        orch.submit_next_level(1, task0, CallConfig())
        orch.submit_next_level(1, task1, CallConfig())
        snapshot = orch._debug_next_level_submits()
    finally:
        worker.close()
    return tuple(snapshot)


def _expect_error(case: str, expected: str, action) -> dict[str, str]:
    try:
        action()
    except ValueError as exc:
        actual = str(exc)
        if actual == expected:
            return {"case": case, "status": "pass", "expected_error": expected}
        return {
            "case": case,
            "status": "fail",
            "expected_error": expected,
            "actual_error": actual,
        }
    return {"case": case, "status": "fail", "expected_error": expected}


def _with_extra_task_args(snapshot: tuple[Any, ...]) -> tuple[Any, ...]:
    mutated = [copy(dict(entry)) for entry in snapshot]
    mutated[0]["args_list"] = [*mutated[0]["args_list"], mutated[0]["args_list"][0]]
    return tuple(mutated)


def _with_duplicate_slot(snapshot: tuple[Any, ...]) -> tuple[Any, ...]:
    mutated = [copy(dict(entry)) for entry in snapshot]
    mutated[1]["task_slot"] = mutated[0]["task_slot"]
    return tuple(mutated)
