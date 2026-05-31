#!/usr/bin/env python3
"""Run paired local A100 and remote H200 paper-baseline readiness probes."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PairedPaperBaselineProbeConfig:
    remote: str = "bizhaoh200"
    remote_workdir: str = "/data/shibizhao/pto-cu"
    branch: str = "goal/nvidia-paper-ready"
    output_root: Path = Path("tmp/cuda-backend/paper-baselines/probes")
    local_python: str = sys.executable
    remote_python: str = ".venv/bin/python"
    remote_cuda_home: str = "/usr/local/cuda-12.8"
    ssh_connect_timeout: int = 8
    remote_git_low_speed_limit: int = 1
    remote_git_low_speed_time: int = 30
    remote_git_fetch_timeout: int = 60
    refresh_remote: bool = True
    sync_remote_tree: bool = False


def _git_commit(runner=subprocess.run) -> str:
    result = runner(
        ["git", "rev-parse", "--short", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _output_dir(config: PairedPaperBaselineProbeConfig, commit: str) -> Path:
    return config.output_root / f"paired-a100-h200-{commit}"


def _artifact_root(config: PairedPaperBaselineProbeConfig, commit: str) -> str:
    return f"{_output_dir(config, commit)}/"


def build_remote_sync_command(config: PairedPaperBaselineProbeConfig) -> list[str]:
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


def build_remote_tmp_mkdir_command(config: PairedPaperBaselineProbeConfig) -> list[str]:
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={config.ssh_connect_timeout}",
        config.remote,
        f"mkdir -p {shlex.quote(config.remote_workdir)}/tmp",
    ]


def build_remote_baseline_source_sync_command(
    config: PairedPaperBaselineProbeConfig,
) -> list[str]:
    return [
        "rsync",
        "-a",
        "--delete",
        "--exclude=build/",
        "--exclude=*.egg-info/",
        "--exclude=__pycache__/",
        "--exclude=*.pyc",
        "--exclude=*.cpython-*.so",
        "tmp/baselines/",
        f"{config.remote}:{config.remote_workdir}/tmp/baselines/",
    ]


def build_local_probe_command(
    config: PairedPaperBaselineProbeConfig,
    commit: str,
) -> list[str]:
    output = _output_dir(config, commit) / "a100-probe.json"
    return [
        "env",
        f"PYTHONPATH={Path.cwd()}:{Path.cwd() / 'python'}",
        config.local_python,
        ".agents/skills/cuda-backend-eval/scripts/paper_baseline_probe.py",
        "--output",
        str(output),
        "--artifact-root",
        _artifact_root(config, commit),
    ]


def _remote_shell_command(
    config: PairedPaperBaselineProbeConfig,
    commit: str,
) -> str:
    output = _output_dir(config, commit) / "h200-probe.json"
    probe = [
        config.remote_python,
        ".agents/skills/cuda-backend-eval/scripts/paper_baseline_probe.py",
        "--output",
        str(output),
        "--artifact-root",
        _artifact_root(config, commit),
    ]
    cuda_home = shlex.quote(config.remote_cuda_home)
    remote_env = (
        f"CUDA_HOME={cuda_home} PATH={cuda_home}/bin:$PATH "
        "PYTHONPATH=$PWD:$PWD/python"
    )
    fetch_command = (
        f"timeout {config.remote_git_fetch_timeout} "
        "git "
        f"-c http.lowSpeedLimit={config.remote_git_low_speed_limit} "
        f"-c http.lowSpeedTime={config.remote_git_low_speed_time} "
        f"fetch origin {shlex.quote(config.branch)} >/dev/null"
    )
    commands = [f"cd {shlex.quote(config.remote_workdir)}"]
    if config.refresh_remote and not config.sync_remote_tree:
        commands.extend(
            [
                fetch_command,
                f"git checkout -B {shlex.quote(config.branch)} FETCH_HEAD >/dev/null",
            ]
        )
    commands.append(f"{remote_env} {' '.join(shlex.quote(part) for part in probe)}")
    return " && ".join(commands)


def build_remote_probe_command(
    config: PairedPaperBaselineProbeConfig,
    commit: str,
) -> list[str]:
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={config.ssh_connect_timeout}",
        config.remote,
        _remote_shell_command(config, commit),
    ]


def build_scp_command(
    config: PairedPaperBaselineProbeConfig,
    commit: str,
) -> list[str]:
    remote_output = _output_dir(config, commit) / "h200-probe.json"
    local_output = _output_dir(config, commit) / "h200-probe.json"
    return [
        "scp",
        f"{config.remote}:{config.remote_workdir}/{remote_output}",
        str(local_output),
    ]


def _display_command(command: Sequence[str]) -> str:
    return shlex.join(command).replace(str(Path.cwd()), "$PWD")


def build_command_examples(
    config: PairedPaperBaselineProbeConfig,
    commit: str,
) -> dict[str, str]:
    examples = {
        "local_sample": _display_command(build_local_probe_command(config, commit)),
        "remote_sample": _display_command(build_remote_probe_command(config, commit)),
        "copy_remote": _display_command(build_scp_command(config, commit)),
        "sync_baseline_sources": _display_command(
            build_remote_baseline_source_sync_command(config)
        ),
    }
    if config.sync_remote_tree:
        examples["sync_remote_tree"] = _display_command(build_remote_sync_command(config))
    return examples


def write_summary(
    config: PairedPaperBaselineProbeConfig,
    commit: str,
    *,
    sync_path: bool,
) -> None:
    output_dir = _output_dir(config, commit)
    summary_path = output_dir / "paired-probe-summary.json"
    summary = {
        "commit": commit,
        "output_dir": str(output_dir),
        "remote": config.remote,
        "remote_workdir": config.remote_workdir,
        "remote_cuda_home": config.remote_cuda_home,
        "sync_remote_tree": sync_path,
        "artifacts": {
            "a100": str(output_dir / "a100-probe.json"),
            "h200": str(output_dir / "h200-probe.json"),
        },
        "commands": build_command_examples(config, commit),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def run_command(command: Sequence[str]) -> None:
    subprocess.run(list(command), check=True)


def run_paired_probe(
    config: PairedPaperBaselineProbeConfig,
    *,
    dry_run: bool = False,
) -> Path:
    commit = _git_commit()
    output_dir = _output_dir(config, commit)
    output_dir.mkdir(parents=True, exist_ok=True)
    commands = [build_local_probe_command(config, commit)]
    if config.sync_remote_tree:
        commands.append(build_remote_sync_command(config))
    commands.extend(
        [
            build_remote_tmp_mkdir_command(config),
            build_remote_baseline_source_sync_command(config),
        ]
    )
    commands.extend(
        [
            build_remote_probe_command(config, commit),
            build_scp_command(config, commit),
        ]
    )
    if dry_run:
        for command in commands:
            print(_display_command(command))
        return output_dir
    for command in commands:
        run_command(command)
    write_summary(config, commit, sync_path=config.sync_remote_tree)
    print(f"wrote paired paper-baseline probes under {output_dir}")
    return output_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", default="bizhaoh200")
    parser.add_argument("--remote-workdir", default="/data/shibizhao/pto-cu")
    parser.add_argument("--branch", default="goal/nvidia-paper-ready")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("tmp/cuda-backend/paper-baselines/probes"),
    )
    parser.add_argument("--local-python", default=sys.executable)
    parser.add_argument("--remote-python", default=".venv/bin/python")
    parser.add_argument("--remote-cuda-home", default="/usr/local/cuda-12.8")
    parser.add_argument("--skip-remote-refresh", action="store_true")
    parser.add_argument("--sync-remote-tree", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = PairedPaperBaselineProbeConfig(
        remote=args.remote,
        remote_workdir=args.remote_workdir,
        branch=args.branch,
        output_root=args.output_root,
        local_python=args.local_python,
        remote_python=args.remote_python,
        remote_cuda_home=args.remote_cuda_home,
        refresh_remote=not args.skip_remote_refresh and not args.sync_remote_tree,
        sync_remote_tree=args.sync_remote_tree,
    )
    run_paired_probe(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
