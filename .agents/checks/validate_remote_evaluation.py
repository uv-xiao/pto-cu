#!/usr/bin/env python3
"""Validate CUDA remote-evaluation Git refresh and tree-sync contracts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = ROOT / ".agents" / "skills" / "cuda-backend-eval" / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))


def fail(message: str) -> None:
    raise SystemExit(f"remote evaluation validation failed: {message}")


def load_module(name: str) -> Any:
    path = SCRIPT_ROOT / f"{name}.py"
    if not path.is_file():
        fail(f"missing script: {path.relative_to(ROOT)}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        fail(f"could not load script: {path.relative_to(ROOT)}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def require_text(path: Path, needles: list[str]) -> None:
    if not path.is_file():
        fail(f"missing file: {path.relative_to(ROOT)}")
    text = path.read_text(encoding="utf-8", errors="replace")
    for needle in needles:
        if needle not in text:
            fail(f"{path.relative_to(ROOT)} missing required text: {needle}")


def check_sync_command(command: list[str], owner: str) -> None:
    if command[:3] != ["rsync", "-a", "--delete"]:
        fail(f"{owner} tree-sync command must start with rsync -a --delete")
    for excluded in (
        "--exclude=.venv",
        "--exclude=build",
        "--exclude=tmp",
        "--exclude=__pycache__",
        "--exclude=.pytest_cache",
    ):
        if excluded not in command:
            fail(f"{owner} tree-sync command missing {excluded}")
    if not command[-2].endswith("/"):
        fail(f"{owner} tree-sync source must be a directory path")
    if command[-1] != "h200-box:/remote/pto-cu/":
        fail(f"{owner} tree-sync destination is unstable: {command[-1]}")


def check_remote_shell(command: list[str], *, owner: str, expect_git: bool) -> None:
    if command[:5] != ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]:
        fail(f"{owner} remote command must use non-interactive ssh")
    shell = command[-1]
    for needle in (
        "cd /remote/pto-cu",
        "CUDA_HOME=",
        "PATH=",
        "PYTHONPATH=$PWD:$PWD/python",
    ):
        if needle not in shell:
            fail(f"{owner} remote command missing {needle}")
    has_git = "git fetch origin" in shell or "git checkout" in shell
    if expect_git and not has_git:
        fail(f"{owner} default remote command must refresh Git")
    if not expect_git and has_git:
        fail(f"{owner} tree-sync remote command must not refresh Git")


def check_pair_script(
    *,
    module_name: str,
    config_name: str,
    remote_builder_name: str,
    remote_builder_arg: str,
) -> None:
    module = load_module(module_name)
    config_class = getattr(module, config_name)
    remote_builder = getattr(module, remote_builder_name)
    config_kwargs = {
        "remote": "h200-box",
        "remote_workdir": "/remote/pto-cu",
        "branch": "goal/nvidia-paper-ready",
        "local_python": ".venv/bin/python",
        "remote_python": ".venv/bin/python",
    }
    config = config_class(**config_kwargs)
    sync_config = config_class(
        **config_kwargs,
        refresh_remote=False,
        sync_remote_tree=True,
    )

    check_sync_command(
        module.build_remote_sync_command(sync_config),
        f"{module_name}.build_remote_sync_command",
    )
    check_remote_shell(
        remote_builder(config, remote_builder_arg),
        owner=f"{module_name}.{remote_builder_name}",
        expect_git=True,
    )
    check_remote_shell(
        remote_builder(sync_config, remote_builder_arg),
        owner=f"{module_name}.{remote_builder_name} tree-sync",
        expect_git=False,
    )


def check_lifecycle_examples() -> None:
    module = load_module("cuda_persistent_lifecycle_matrix")
    config = module.LifecycleMatrixConfig(
        remote="h200-box",
        remote_workdir="/remote/pto-cu",
        branch="goal/nvidia-paper-ready",
        local_python=".venv/bin/python",
        remote_python=".venv/bin/python",
        sync_remote_tree=True,
    )
    examples = module.build_command_examples(config, "abc123")
    for key in ("local_sample", "remote_sample", "sync_remote_tree"):
        if key not in examples:
            fail(f"lifecycle matrix command examples missing {key}")
    if "--sync-remote-tree" not in examples["local_sample"]:
        fail("lifecycle matrix local sample must record --sync-remote-tree")
    if "ssh " not in examples["remote_sample"] or " h200-box " not in examples["remote_sample"]:
        fail("lifecycle matrix remote sample must use ssh")
    if "rsync -a --delete" not in examples["sync_remote_tree"]:
        fail("lifecycle matrix sync example must use rsync")


def validate_remote_evaluation(root: Path = ROOT) -> None:
    check_pair_script(
        module_name="cuda_pair_smoke",
        config_name="PairedSmokeConfig",
        remote_builder_name="build_remote_smoke_command",
        remote_builder_arg="abc123",
    )
    check_pair_script(
        module_name="cuda_pair_persistent_smoke",
        config_name="PairedPersistentSmokeConfig",
        remote_builder_name="build_remote_smoke_command",
        remote_builder_arg="abc123",
    )
    check_pair_script(
        module_name="cuda_pair_benchmark",
        config_name="PairedBenchmarkConfig",
        remote_builder_name="build_remote_benchmark_command",
        remote_builder_arg="abc123",
    )
    check_pair_script(
        module_name="cuda_pair_stream_benchmark",
        config_name="PairedStreamBenchmarkConfig",
        remote_builder_name="build_remote_benchmark_command",
        remote_builder_arg="abc123",
    )
    check_lifecycle_examples()
    require_text(
        root / ".agents" / "rules" / "remote-evaluation.md",
        [
            "First try remote Git refresh",
            "tree sync",
            "rsync -a --delete",
            "CUDA_HOME",
            "Never claim paired A100/H200 validation from a local-only run.",
        ],
    )
    require_text(
        root / "docs" / "in_progress" / "nvidia_backend_paper_ready" / "work_preparation.md",
        [
            "Git path",
            "Tree-sync fallback",
            "local commit",
            "remote commit",
        ],
    )


def main() -> None:
    validate_remote_evaluation()
    print("remote evaluation validation passed")


if __name__ == "__main__":
    main()
