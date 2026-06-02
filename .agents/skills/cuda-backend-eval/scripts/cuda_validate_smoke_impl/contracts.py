"""Public validation contracts for CUDA smoke artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

REPORT_FILES = (
    "cuda-smoke-report.md",
    "cuda-smoke-report.svg",
)


@dataclass(frozen=True)
class ResourcePolicyExpectation:
    scheduler_blocks: int | None = None
    worker_blocks: int | None = None
    worker_blocks_per_task: int | None = None
    stream_id: int | None = None
    block_dim: int | None = None
    grid_dim: int | None = None

    def fields(self) -> tuple[tuple[str, int | None], ...]:
        return (
            ("scheduler_blocks", self.scheduler_blocks),
            ("worker_blocks", self.worker_blocks),
            ("worker_blocks_per_task", self.worker_blocks_per_task),
            ("stream_id", self.stream_id),
            ("block_dim", self.block_dim),
            ("grid_dim", self.grid_dim),
        )

    def is_empty(self) -> bool:
        return all(expected is None for _, expected in self.fields())


@dataclass(frozen=True)
class SmokeValidationExpectation:
    artifact_dir: Path | None = None
    required_artifacts: Sequence[str] = ()
    runtime: str | None = None
    mode: str | None = None
    dag_shape: str | None = None
    repeat_runs: int | None = None
    completed_count: int | None = None
    dispatch: str | None = None
    tensor_tile: str | None = None
    scalar_args: str | None = None
    tensor_args: str | None = None
    graph_fanin: str | None = None
    graph_dependents: str | None = None
    graph_lowering: str | None = None
    graph_task_arg_key: str | None = None
    graph_task_args: str | None = None
    graph_node_attrs: str | None = None
    graph_node_ops: str | None = None
    scratch_reuse: str | None = None
    scheduler_init_count: int | None = None
    scheduler_loop_count: int | None = None
    scheduler_processed_count: int | None = None
    scheduler_processed_block_count: int | None = None
    resource_policy: ResourcePolicyExpectation | None = None
    require_report_files: bool = False
    require_report_scalar_args: bool = False
    require_report_tensor_args: bool = False
    require_report_graph_topology: bool = False
    require_report_graph_task_args: bool = False
    require_report_graph_node_attrs: bool = False
    require_report_graph_node_ops: bool = False
