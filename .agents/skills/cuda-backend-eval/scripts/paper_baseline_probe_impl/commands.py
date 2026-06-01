from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .paths import ROOT


def run_command(command: list[str], *, cwd: Path | None = None) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
            check=False,
        )
        return {
            "command": " ".join(command),
            "returncode": result.returncode,
            "output": result.stdout[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": " ".join(command),
            "returncode": 124,
            "output": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
        }


def git_commit(path: Path) -> str:
    result = run_command(["git", "rev-parse", "HEAD"], cwd=path)
    if result["returncode"] != 0:
        return "unknown"
    return str(result["output"]).strip()


def nvidia_smi() -> list[str]:
    result = run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader",
        ]
    )
    if result["returncode"] != 0:
        return []
    return [line.strip() for line in result["output"].splitlines() if line.strip()]


def nvcc_version() -> str:
    result = run_command(["nvcc", "--version"])
    if result["returncode"] != 0:
        return "unavailable"
    for line in result["output"].splitlines():
        if "release" in line:
            return line.strip()
    return result["output"].splitlines()[-1].strip()
