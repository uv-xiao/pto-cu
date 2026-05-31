#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under
# the terms and conditions of CANN Open Software License Agreement Version 2.0.
# Please refer to the License for details. You may not use this file except in
# compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text.
# -----------------------------------------------------------------------------
"""Run paired A100/H200 CUDA Runtime and Driver direct-launch sweeps."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import cuda_benchmark  # noqa: E402

Runner = Callable[..., subprocess.CompletedProcess]

DIRECT_LAUNCH_BASELINES: tuple[str, ...] = (
    "direct_runtime",
    "direct_driver",
    "direct_runtime_sgemm",
    "direct_driver_sgemm",
)
TENSOR_BASELINES = frozenset({"direct_runtime_sgemm", "direct_driver_sgemm"})
SOURCE_PAPERS = (
    {
        "id": "arXiv:2605.03190",
        "label": "VDCores",
        "path": "tmp/sources/arxiv-2605.03190-vdcores.txt",
    },
    {
        "id": "arXiv:2512.22219v1",
        "label": "MPK persistent kernel",
        "path": "tmp/sources/arxiv-2512.22219v1-mirage-persistent-kernel.txt",
    },
)
PAPER_SETUP = (
    "Direct CUDA launch sweep for selected vector and tensor launch shapes; "
    "captures CUDA Runtime and CUDA Driver launch paths without CUDA Graph replay."
)


@dataclass(frozen=True)
class DirectLaunchSweepConfig:
    remote: str = "bizhaoh200"
    remote_workdir: str = "/data/shibizhao/pto-cu"
    branch: str = "goal/nvidia-paper-ready"
    output_root: Path = Path("tmp/cuda-backend")
    local_device: int = 0
    remote_device: int = 0
    local_machine: str = "hina"
    remote_machine: str = "dasys-h200x8"
    sizes: tuple[int, ...] = (1024, 4096, 65536)
    repeats: int = 10
    baselines: tuple[str, ...] = DIRECT_LAUNCH_BASELINES
    tensor_rows: int = 16
    tensor_cols: int = 16
    tensor_inner: int = 16
    local_arch: str = "compute_80"
    remote_arch: str = "compute_90"
    local_python: str = sys.executable
    remote_python: str = ".venv/bin/python"
    remote_cuda_home: str = "/usr/local/cuda-12.8"
    ssh_connect_timeout: int = 8
    remote_git_low_speed_limit: int = 1
    remote_git_low_speed_time: int = 30
    remote_git_fetch_timeout: int = 60
    refresh_remote: bool = True
    sync_remote_tree: bool = False


def _git_commit(runner: Runner = subprocess.run) -> str:
    result = runner(
        ["git", "rev-parse", "--short", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _csv(values: Sequence[int]) -> str:
    return ",".join(str(value) for value in values)


def _is_tensor_baseline(baseline: str) -> bool:
    return baseline in TENSOR_BASELINES


def _sample_args(
    *,
    device: int,
    arch: str,
    baseline: str,
    n: int,
    config: DirectLaunchSweepConfig,
) -> list[str]:
    args = [
        "--device",
        str(device),
        "--single-baseline",
        baseline,
        "--sizes",
        str(n),
        "--arch",
        arch,
    ]
    if _is_tensor_baseline(baseline):
        args.extend(
            [
                "--tensor-rows",
                str(config.tensor_rows),
                "--tensor-cols",
                str(config.tensor_cols),
                "--tensor-inner",
                str(config.tensor_inner),
            ]
        )
    return args


def build_remote_sync_command(config: DirectLaunchSweepConfig) -> list[str]:
    return [
        "rsync",
        "-a",
        "--delete",
        "--exclude=.venv",
        "--exclude=build",
        "--exclude=tmp",
        "--exclude=__pycache__",
        "--exclude=.pytest_cache",
        f"{Path.cwd()}/",
        f"{config.remote}:{config.remote_workdir}/",
    ]


def build_local_sample_command(
    config: DirectLaunchSweepConfig,
    *,
    baseline: str,
    n: int,
) -> list[str]:
    return [
        "env",
        f"PYTHONPATH={Path.cwd()}:{Path.cwd() / 'python'}",
        config.local_python,
        ".agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py",
        *_sample_args(
            device=config.local_device,
            arch=config.local_arch,
            baseline=baseline,
            n=n,
            config=config,
        ),
    ]


def _remote_prefix(config: DirectLaunchSweepConfig) -> list[str]:
    commands = [f"cd {shlex.quote(config.remote_workdir)}"]
    if config.refresh_remote and not config.sync_remote_tree:
        fetch_command = (
            f"timeout {config.remote_git_fetch_timeout} "
            "git "
            f"-c http.lowSpeedLimit={config.remote_git_low_speed_limit} "
            f"-c http.lowSpeedTime={config.remote_git_low_speed_time} "
            f"fetch origin {shlex.quote(config.branch)} >/dev/null"
        )
        commands.extend(
            [
                fetch_command,
                f"git checkout -B {shlex.quote(config.branch)} FETCH_HEAD >/dev/null",
            ]
        )
    return commands


def build_remote_sample_command(
    config: DirectLaunchSweepConfig,
    *,
    baseline: str,
    n: int,
    commit: str,
) -> list[str]:
    benchmark = [
        config.remote_python,
        ".agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py",
        *_sample_args(
            device=config.remote_device,
            arch=config.remote_arch,
            baseline=baseline,
            n=n,
            config=config,
        ),
    ]
    remote_cuda_home = shlex.quote(config.remote_cuda_home)
    remote_env_parts = [
        f"CUDA_HOME={remote_cuda_home}",
        f"PATH={remote_cuda_home}/bin:$PATH",
        "PYTHONPATH=$PWD:$PWD/python",
    ]
    if config.sync_remote_tree:
        remote_env_parts.append(f"PTO_SOURCE_COMMIT={shlex.quote(commit)}")
    commands = _remote_prefix(config)
    commands.append(
        f"{' '.join(remote_env_parts)} "
        f"{' '.join(shlex.quote(part) for part in benchmark)}"
    )
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={config.ssh_connect_timeout}",
        config.remote,
        " && ".join(commands),
    ]


def _display_command(command: Sequence[str]) -> str:
    return shlex.join(command).replace(str(Path.cwd()), "$PWD")


def build_command_examples(
    config: DirectLaunchSweepConfig,
    commit: str,
) -> dict[str, str]:
    baseline = config.baselines[0]
    n = config.sizes[0]
    examples = {
        "local_sample": _display_command(
            build_local_sample_command(config, baseline=baseline, n=n),
        ),
        "remote_sample": _display_command(
            build_remote_sample_command(
                config,
                baseline=baseline,
                n=n,
                commit=commit,
            ),
        ),
    }
    if config.sync_remote_tree:
        examples["sync_remote_tree"] = _display_command(
            build_remote_sync_command(config)
        )
    return examples


def _sample_from_stdout(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise ValueError("benchmark command did not print a JSON sample")


def _tensor_tile(config: DirectLaunchSweepConfig) -> dict[str, int]:
    return {
        "rows": config.tensor_rows,
        "cols": config.tensor_cols,
        "inner": config.tensor_inner,
    }


def _dry_run_sample(
    *,
    artifact: str,
    machine: str,
    baseline: str,
    n: int,
    repeat: int,
    config: DirectLaunchSweepConfig,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "artifact": artifact,
        "machine": machine,
        "baseline": baseline,
        "n": n,
        "task_count": 1,
        "block_dim": 128,
        "worker_blocks_per_task": 1,
        "repeat": repeat,
        "host_wall_ns": 0,
        "device_wall_ns": 0,
        "status": "dry-run",
    }
    if _is_tensor_baseline(baseline):
        row["tensor_tile"] = _tensor_tile(config)
    return row


def _run_sample(
    command: list[str],
    *,
    artifact: str,
    machine: str,
    baseline: str,
    n: int,
    repeat: int,
    config: DirectLaunchSweepConfig,
    runner: Runner,
    dry_run: bool,
) -> dict[str, Any]:
    print(" ".join(shlex.quote(part) for part in command), flush=True)
    if dry_run:
        return _dry_run_sample(
            artifact=artifact,
            machine=machine,
            baseline=baseline,
            n=n,
            repeat=repeat,
            config=config,
        )
    result = runner(command, check=True, capture_output=True, text=True)
    sample = _sample_from_stdout(result.stdout)
    sample["artifact"] = artifact
    sample["machine"] = machine
    sample.setdefault("baseline", baseline)
    sample.setdefault("n", n)
    sample["repeat"] = repeat
    if _is_tensor_baseline(baseline):
        sample.setdefault("tensor_tile", _tensor_tile(config))
    return sample


def _validate_baselines(baselines: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(part for part in baselines if part)
    if not normalized:
        raise ValueError("at least one direct-launch baseline is required")
    unknown = [
        baseline for baseline in normalized if baseline not in DIRECT_LAUNCH_BASELINES
    ]
    if unknown:
        allowed = ", ".join(DIRECT_LAUNCH_BASELINES)
        raise ValueError(f"unknown direct-launch baseline(s): {unknown}; allowed: {allowed}")
    return normalized


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    values = tuple(int(part) for part in value.split(",") if part)
    if not values:
        raise argparse.ArgumentTypeError("at least one integer is required")
    if any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("integer values must be positive")
    return values


def _parse_baselines(value: str) -> tuple[str, ...]:
    try:
        return _validate_baselines(tuple(part.strip() for part in value.split(",")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def output_dir_for(config: DirectLaunchSweepConfig, commit: str) -> Path:
    return config.output_root / f"direct-launch-sweep-{commit}"


def build_validate_command(
    config: DirectLaunchSweepConfig,
    commit: str,
) -> list[str]:
    baseline_args = [
        part for baseline in config.baselines for part in ("--require-baseline", baseline)
    ]
    tensor_tile_args = [
        part
        for baseline in config.baselines
        if _is_tensor_baseline(baseline)
        for part in (
            "--require-tensor-tile",
            f"{baseline}={config.tensor_rows}x{config.tensor_cols}x{config.tensor_inner}",
        )
    ]
    return [
        "env",
        f"PYTHONPATH={Path.cwd()}:{Path.cwd() / 'python'}",
        config.local_python,
        ".agents/skills/cuda-backend-eval/scripts/cuda_validate_capture.py",
        str(output_dir_for(config, commit) / "cuda-benchmark.json"),
        "--require-machine",
        config.local_machine,
        "--require-machine",
        config.remote_machine,
        "--require-size",
        _csv(config.sizes),
        "--expected-repeats",
        str(config.repeats),
        "--expected-result-count",
        str(2 * len(config.baselines) * len(config.sizes) * config.repeats),
        *baseline_args,
        *tensor_tile_args,
        "--require-report-files",
        "--require-command-examples",
        "--require-source-papers",
    ]


def run_direct_launch_sweep(
    config: DirectLaunchSweepConfig,
    *,
    runner: Runner = subprocess.run,
    dry_run: bool = False,
) -> dict[str, Any]:
    config = DirectLaunchSweepConfig(
        **{**config.__dict__, "baselines": _validate_baselines(config.baselines)}
    )
    commit = _git_commit(runner)
    if config.sync_remote_tree:
        sync_command = build_remote_sync_command(config)
        print(" ".join(shlex.quote(part) for part in sync_command), flush=True)
        if not dry_run:
            runner(sync_command, check=True)

    results: list[dict[str, Any]] = []
    for baseline in config.baselines:
        for n in config.sizes:
            for repeat in range(config.repeats):
                results.append(
                    _run_sample(
                        build_local_sample_command(config, baseline=baseline, n=n),
                        artifact="a100",
                        machine=config.local_machine,
                        baseline=baseline,
                        n=n,
                        repeat=repeat,
                        config=config,
                        runner=runner,
                        dry_run=dry_run,
                    )
                )
                results.append(
                    _run_sample(
                        build_remote_sample_command(
                            config,
                            baseline=baseline,
                            n=n,
                            commit=commit,
                        ),
                        artifact="h200",
                        machine=config.remote_machine,
                        baseline=baseline,
                        n=n,
                        repeat=repeat,
                        config=config,
                        runner=runner,
                        dry_run=dry_run,
                    )
                )

    payload = {
        "metadata": {
            "label": f"direct-launch-sweep-{commit}",
            "git_commit": commit,
            "machine": "paired A100/H200",
            "paper_setup": PAPER_SETUP,
            "sizes": list(config.sizes),
            "repeats": config.repeats,
            "baselines": list(config.baselines),
            "tensor_tile": _tensor_tile(config),
            "source_papers": list(SOURCE_PAPERS),
            "command_examples": build_command_examples(config, commit),
        },
        "results": results,
    }
    output_dir = output_dir_for(config, commit)
    cuda_benchmark.write_report(payload, output_dir)
    print(output_dir / "cuda-benchmark.json")
    print(output_dir / "cuda-benchmark.md")
    print(output_dir / "cuda-benchmark.svg")
    print(output_dir / "cuda-benchmark-ratios.svg")
    print(output_dir / "cuda-benchmark-dag-deltas.svg")
    print(output_dir / "cuda-benchmark-throughput.svg")
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", default="bizhaoh200")
    parser.add_argument("--remote-workdir", default="/data/shibizhao/pto-cu")
    parser.add_argument("--branch", default="goal/nvidia-paper-ready")
    parser.add_argument("--output-root", type=Path, default=Path("tmp/cuda-backend"))
    parser.add_argument("--local-device", type=int, default=0)
    parser.add_argument("--remote-device", type=int, default=0)
    parser.add_argument("--local-machine", default="hina")
    parser.add_argument("--remote-machine", default="dasys-h200x8")
    parser.add_argument("--sizes", type=_parse_int_tuple, default=(1024, 4096, 65536))
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--baselines", type=_parse_baselines, default=DIRECT_LAUNCH_BASELINES)
    parser.add_argument("--tensor-rows", type=int, default=16)
    parser.add_argument("--tensor-cols", type=int, default=16)
    parser.add_argument("--tensor-inner", type=int, default=16)
    parser.add_argument("--local-arch", default="compute_80")
    parser.add_argument("--remote-arch", default="compute_90")
    parser.add_argument("--local-python", default=sys.executable)
    parser.add_argument("--remote-python", default=".venv/bin/python")
    parser.add_argument("--remote-cuda-home", default="/usr/local/cuda-12.8")
    parser.add_argument("--ssh-connect-timeout", type=int, default=8)
    parser.add_argument("--remote-git-low-speed-limit", type=int, default=1)
    parser.add_argument("--remote-git-low-speed-time", type=int, default=30)
    parser.add_argument("--remote-git-fetch-timeout", type=int, default=60)
    parser.add_argument("--skip-remote-refresh", action="store_true")
    parser.add_argument("--sync-remote-tree", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = DirectLaunchSweepConfig(
        remote=args.remote,
        remote_workdir=args.remote_workdir,
        branch=args.branch,
        output_root=args.output_root,
        local_device=args.local_device,
        remote_device=args.remote_device,
        local_machine=args.local_machine,
        remote_machine=args.remote_machine,
        sizes=args.sizes,
        repeats=args.repeats,
        baselines=args.baselines,
        tensor_rows=args.tensor_rows,
        tensor_cols=args.tensor_cols,
        tensor_inner=args.tensor_inner,
        local_arch=args.local_arch,
        remote_arch=args.remote_arch,
        local_python=args.local_python,
        remote_python=args.remote_python,
        remote_cuda_home=args.remote_cuda_home,
        ssh_connect_timeout=args.ssh_connect_timeout,
        remote_git_low_speed_limit=args.remote_git_low_speed_limit,
        remote_git_low_speed_time=args.remote_git_low_speed_time,
        remote_git_fetch_timeout=args.remote_git_fetch_timeout,
        refresh_remote=not args.skip_remote_refresh and not args.sync_remote_tree,
        sync_remote_tree=args.sync_remote_tree,
    )
    run_direct_launch_sweep(config, dry_run=args.dry_run)
    if not args.dry_run:
        validate_command = build_validate_command(config, _git_commit())
        print(" ".join(shlex.quote(part) for part in validate_command), flush=True)
        subprocess.run(validate_command, check=True)


if __name__ == "__main__":
    main()
