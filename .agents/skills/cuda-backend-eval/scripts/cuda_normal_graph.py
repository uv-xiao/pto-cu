#!/usr/bin/env python3
"""Normal PTO graph lowering helpers for CUDA persistent-device smoke paths."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CudaNormalGraphNode:
    """A normalized task node before CUDA persistent DAG ABI materialization."""

    key: str
    func_id: int
    a: int
    b: int
    out: int
    n: int
    depends_on: tuple[str, ...] = ()
    attrs: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class LoweredCudaNormalGraph:
    """CUDA persistent DAG descriptor arrays derived from normal graph edges."""

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
    fanin = [0 for _ in nodes]
    for dst_idx, node in enumerate(nodes):
        fanin[dst_idx] = len(node.depends_on)
        for dep_key in node.depends_on:
            src_idx = positions.get(dep_key)
            if src_idx is None:
                raise ValueError(f"unknown dependency {dep_key!r} for node {node.key!r}")
            dependents_by_node[src_idx].append(dst_idx)

    flat_dependents: list[int] = []
    tasks = []
    for node, node_dependents, initial_fanin in zip(nodes, dependents_by_node, fanin, strict=True):
        dependent_begin = len(flat_dependents)
        flat_dependents.extend(node_dependents)
        tasks.append(make_task(node, dependent_begin, len(node_dependents), initial_fanin))
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
