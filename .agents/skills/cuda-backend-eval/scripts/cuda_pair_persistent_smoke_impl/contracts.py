"""Shared contracts for paired CUDA persistent smoke capture."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


def is_tensor_tile_shape(dag_shape: str) -> bool:
    return dag_shape in {"graph_tensor_core_tile", "graph_tensor_tile", "tensor_core_tile", "tensor_tile"}


@dataclass(frozen=True)
class PairedPersistentSmokeConfig:
    remote: str = "bizhaoh200"
    remote_workdir: str = "/data/shibizhao/pto-cu"
    branch: str = "design/nvidia-backend"
    output_root: Path = Path("tmp/cuda-backend")
    local_device: int = 0
    remote_device: int = 0
    n: int = 1024
    mode: str = "dag"
    dag_shape: str = "fork_join"
    task_count: int = 3
    queue_capacity: int = 2
    worker_blocks_per_task: int = 1
    worker_blocks: int | None = None
    scheduler_blocks: int = 1
    stream_id: int = 0
    block_dim: int = 256
    repeat_runs: int = 1
    tensor_rows: int = 16
    tensor_cols: int = 16
    tensor_inner: int = 16
    local_arch: str = "compute_80"
    remote_arch: str = "compute_90"
    local_python: str = sys.executable
    remote_python: str = ".venv/bin/python"
    ssh_connect_timeout: int = 8
    remote_git_low_speed_limit: int = 1
    remote_git_low_speed_time: int = 30
    remote_git_fetch_timeout: int = 60
    refresh_remote: bool = True
    sync_remote_tree: bool = False
    validate_smoke: bool = True
