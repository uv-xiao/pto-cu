# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Normal graph lowering helpers for CUDA persistent-device descriptors."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class CudaNormalGraphNode:
    """A normalized task node before CUDA persistent DAG materialization."""

    key: str
    depends_on: tuple[str, ...] = ()
    func_id: int = 0
    a: int = 0
    b: int = 0
    out: int = 0
    n: int = 0
    attrs: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class LoweredCudaNormalGraph:
    """CUDA persistent DAG arrays derived from normal graph edges."""

    fanin: list[int]
    dependents: list[int]
    tasks: list[Any]


TaskFactory = Callable[[CudaNormalGraphNode, int, int, int], Any]


def lower_normal_graph(
    nodes: Sequence[CudaNormalGraphNode],
    make_task: TaskFactory,
) -> LoweredCudaNormalGraph:
    """Lower normal graph nodes into fan-in/dependent arrays plus task records."""

    positions = _node_positions(nodes)
    dependents_by_node = [[] for _ in nodes]
    for dst_idx, node in enumerate(nodes):
        for dep_key in node.depends_on:
            src_idx = positions.get(dep_key)
            if src_idx is None:
                raise ValueError(f"unknown dependency {dep_key!r} for node {node.key!r}")
            dependents_by_node[src_idx].append(dst_idx)
    return lower_normal_graph_from_dependents(nodes, dependents_by_node, make_task)


def lower_normal_graph_from_dependents(
    nodes: Sequence[CudaNormalGraphNode],
    dependents_by_node: Sequence[Sequence[int]],
    make_task: TaskFactory,
) -> LoweredCudaNormalGraph:
    """Lower nodes plus explicit dependent lists into CUDA DAG arrays."""

    if len(nodes) != len(dependents_by_node):
        raise ValueError("normal graph node and dependent-list counts must match")

    fanin = [0 for _ in nodes]
    flat_dependents: list[int] = []
    for src_idx, node_dependents in enumerate(dependents_by_node):
        normalized_dependents = []
        for dependent in node_dependents:
            dst_idx = int(dependent)
            if dst_idx < 0 or dst_idx >= len(nodes):
                raise ValueError(f"dependent task id {dst_idx} for node {src_idx} is outside the graph")
            fanin[dst_idx] += 1
            normalized_dependents.append(dst_idx)
        flat_dependents.extend(normalized_dependents)

    dependent_begin = 0
    tasks = []
    for idx, node in enumerate(nodes):
        dependent_count = len(dependents_by_node[idx])
        tasks.append(make_task(node, dependent_begin, dependent_count, fanin[idx]))
        dependent_begin += dependent_count
    return LoweredCudaNormalGraph(fanin=fanin, dependents=flat_dependents, tasks=tasks)


def _node_positions(nodes: Iterable[CudaNormalGraphNode]) -> dict[str, int]:
    positions: dict[str, int] = {}
    for idx, node in enumerate(nodes):
        if not node.key:
            raise ValueError("graph node key must be non-empty")
        if node.key in positions:
            raise ValueError(f"duplicate graph node key {node.key!r}")
        positions[node.key] = idx
    return positions
